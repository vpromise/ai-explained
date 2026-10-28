"""AI, Explained · Episode 33 · Decoding & Sampling

Running example: a language model finishing "Once upon a…". A TOY model (stated on screen and
in the narration): eight candidate next tokens after "Once upon a" with made-up scores, and a
small lookup table for later steps (it looks at the last two words only). Everything shown is
computed below from those scores:

- the distribution: softmax of the scores, time 60 %, midnight 13 %, dream 9 %, hill 7 %,
  star 5 %, Tuesday 3 %, spoon 2 %, banana 1 %;
- greedy: always the top token; the toy table sends it round "there was a time" for ever;
- sampling: one weighted roll per token with numpy's default_rng(seed), seeds 1-4 (seed 4
  lands in the tail: "Once upon a Tuesday.");
- temperature: softmax(z / T), T = 0.5 → time 91 %, T = 2 → 33 %, T → 0 → greedy;
- top-k (k = 3): keep 82 % of the mass, renormalise → time 73 %;
- top-p (p = 0.9): running total 60, 73, 82, 88, 93 % → five tokens; a sure context keeps
  one token, an unsure one keeps seven;
- beam search (width 2, three steps): greedy's "time there was" has total probability 17 %,
  the beam finds "time long ago" at 19 %.

Every number the narration quotes is asserted below, so the film cannot drift from its words.
Sources: top-k sampling for story generation, Fan et al., 2018 (Hierarchical Neural Story
Generation, ACL); nucleus (top-p) sampling and the repetition of greedy / beam search text,
Holtzman et al., 2019 (The Curious Case of Neural Text Degeneration, arXiv 1904.09751, ICLR 2020).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- colours

BARS, BARS_FILL = VIOLET, VIOLET_FILL  # the model's probabilities
PICK, PICK_FILL = AMBER, AMBER_FILL  # greedy's pick, the temperature knob, running totals
BEAM, BEAM_FILL = TEAL, TEAL_FILL  # kept beams
ODD = BAD  # loops and the odd tail

# ---------------------------------------------------------------------------- the toy model


def softmax(z, t=1.0):
    z = np.asarray(z, dtype=float) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def scores(**probs):
    """A toy context given as probabilities, stored as scores (their logs): softmax gives them back."""
    return {k: float(np.log(v)) for k, v in probs.items()}


PROMPT = ("Once", "upon", "a")
TOKENS = ("time", "midnight", "dream", "hill", "star", "Tuesday", "spoon", "banana")
LOGITS = np.array([3.0, 1.5, 1.1, 0.8, 0.5, 0.1, -0.4, -0.8])  # made-up scores after "Once upon a"
P = softmax(LOGITS)
END = "."

# the toy's next-token table: by the last two words, else by the last word; a noun ends the story
PAIRS = {
    ("Once", "upon"): scores(a=0.93, the=0.03, my=0.02, every=0.01, our=0.01),  # the model is sure
    ("upon", "a"): dict(zip(TOKENS, LOGITS)),
    ("there", "was"): scores(a=0.6, an=0.4),
    ("was", "a"): scores(time=0.3, king=0.22, girl=0.2, fox=0.16, dragon=0.12),
    ("lived", "a"): scores(king=0.17, fox=0.16, girl=0.15, dragon=0.14, baker=0.12, witch=0.11, giant=0.08, miller=0.07),  # unsure
}
LAST = {
    "time": scores(there=0.40, long=0.34, far=0.26),
    "long": scores(ago=0.95, before=0.05),
    "far": scores(away=1.0),
    "ago": scores(there=1.0),
    "before": scores(there=1.0),
    "away": scores(there=1.0),
    "there": scores(was=0.7, lived=0.3),
    "lived": scores(a=0.6, an=0.4),
    "an": scores(old=0.7, owl=0.3),
    "old": scores(king=0.5, fox=0.3, witch=0.2),
    "midnight": scores(dreary=0.6, clear=0.4),
    "dream": scores(there=0.6, **{END: 0.4}),
    "hill": scores(there=0.8, **{END: 0.2}),
    "star": scores(there=0.4, **{END: 0.6}),
}
ENDS = set("king girl fox dragon baker witch giant miller owl dreary clear Tuesday spoon banana".split())


def next_token(words):
    """The toy model: candidate tokens and their probabilities (softmax of the scores), sorted."""
    key = tuple(words[-2:])
    if key in PAIRS:
        table = PAIRS[key]
    elif words[-1] in ENDS:
        return [END], np.array([1.0])
    else:
        table = LAST[words[-1]]
    names = list(table)
    probs = softmax([table[n] for n in names])
    order = np.argsort(-probs, kind="stable")
    return [names[i] for i in order], probs[order]


def greedy(steps):
    words, trace = list(PROMPT), []
    for _ in range(steps):
        names, probs = next_token(words)
        trace.append((tuple(words[-2:]), names, probs))
        words.append(names[0])
        if words[-1] == END:
            break
    return words, trace


def sample(seed, limit=12):
    """Weighted rolls with a seeded generator: a uniform number u in [0, 1) picks the token whose
    slice of the strip it lands in."""
    rng = np.random.default_rng(seed)
    words, rolls = list(PROMPT), []
    for _ in range(limit):
        names, probs = next_token(words)
        u = float(rng.random())
        rolls.append(u)
        words.append(names[min(int(np.searchsorted(np.cumsum(probs), u, side="right")), len(names) - 1)])
        if words[-1] == END:
            break
    return words, rolls


def story(words):
    return " ".join(words).replace(" .", ".")


def beam_search(width, steps):
    """Keeps the `width` best partial continuations by total probability; returns each step's
    candidates (sentence, total probability, parent) and the ones kept."""
    beams = [(tuple(PROMPT), 1.0)]
    history = []
    for _ in range(steps):
        candidates = []
        for words, total in beams:
            names, probs = next_token(list(words))
            candidates += [(words + (n,), total * float(q), words) for n, q in zip(names, probs)]
        candidates.sort(key=lambda c: -c[1])
        kept = candidates[:width]
        history.append((candidates, kept))
        beams = [(w, t) for w, t, _ in kept]
    return history


def pct(v):
    return int(round(100 * float(v)))


# the distribution
assert [pct(v) for v in P] == [60, 13, 9, 7, 5, 3, 2, 1] and sum(pct(v) for v in P) == 100  # "time gets sixty percent"
TAIL = float(P[5:].sum())
assert pct(TAIL) == 7 and round(1 / TAIL) == 15  # "about seven percent: one roll in fifteen"

# greedy loops: there, was, a, then time again, for ever
GREEDY_WORDS, GREEDY_TRACE = greedy(16)
assert GREEDY_WORDS[3:8] == ["time", "there", "was", "a", "time"]  # "there. was. a." "time is on top again"
assert GREEDY_WORDS[7:] == GREEDY_WORDS[3:15] and END not in GREEDY_WORDS  # round in circles
GREEDY_STEPS = GREEDY_TRACE[:5]

# sampling, seeds 1-4
SEEDS = (1, 2, 3, 4)
SAMPLES = {s: sample(s) for s in SEEDS}
assert all(SAMPLES[s][0][3] == "time" for s in (1, 2, 3))  # "every story starts with time"
assert len({story(SAMPLES[s][0]) for s in (1, 2, 3)}) == 3  # "three different stories"
assert story(SAMPLES[4][0]) == "Once upon a Tuesday."  # "seed four lands in the tail"
assert np.cumsum(P)[4] < SAMPLES[4][1][0]  # its roll is past star: in the tail

# temperature
COLD, HOT, FREEZE = 0.5, 2.0, 0.12
assert pct(softmax(LOGITS, COLD)[0]) == 91  # "time jumps to ninety-one percent"
assert pct(softmax(LOGITS, HOT)[0]) == 33  # "time falls to thirty-three percent"
assert np.all(softmax(LOGITS, HOT)[5:] > P[5:])  # "while the odd tail grows"
assert softmax(LOGITS, FREEZE)[0] > 0.999  # "time wins every roll"

# top-k
K = 3
KEPT_K = float(P[:K].sum())
P_K = np.where(np.arange(len(P)) < K, P, 0.0) / KEPT_K
assert pct(KEPT_K) == 82 and pct(P_K[0]) == 73  # "eighty-two percent", "seventy-three percent"
SURE_NAMES, SURE_P = next_token(["Once", "upon"])
UNSURE_NAMES, UNSURE_P = next_token(["lived", "a"])
assert SURE_NAMES[:K] == ["a", "the", "my"] and pct(SURE_P[0]) == 93  # "the model is sure", yet keeps three
assert UNSURE_NAMES[K] == "dragon" and pct(UNSURE_P[K]) == 14 and pct(UNSURE_P[K - 1]) == 15  # "dragon, almost as likely as girl"
assert UNSURE_NAMES[K - 1] == "girl"

# top-p
TOP_P = 0.9


def nucleus(probs, p=TOP_P):
    return int(np.searchsorted(np.cumsum(probs), p) + 1)


RUNNING = np.cumsum(P)
N_P = nucleus(P)
assert N_P == 5 and [pct(v) for v in RUNNING[:5]] == [60, 73, 82, 88, 93]  # "five tokens, from time down to star"
P_P = np.where(np.arange(len(P)) < N_P, P, 0.0) / RUNNING[N_P - 1]
assert nucleus(SURE_P) == 1 and nucleus(UNSURE_P) == 7  # "shrinks to one token", "grows to seven"

# beam search, width 2, three steps
BEAMS = beam_search(2, 3)
assert [c[0][-1] for c in BEAMS[0][1]] == ["time", "midnight"]  # step one keeps time and midnight
assert [c[0][-1] for c in BEAMS[1][1]] == ["there", "long"]
assert [c[0][3:] for c in BEAMS[2][1]] == [("time", "long", "ago"), ("time", "there", "was")]
GREEDY_TOTAL = float(np.prod([probs[0] for _, _, probs in GREEDY_TRACE[:3]]))
BEAM_TOTAL = BEAMS[2][1][0][1]
assert pct(GREEDY_TOTAL) == 17 and pct(BEAM_TOTAL) == 19  # "seventeen percent", "at nineteen"
assert tuple(GREEDY_WORDS[3:6]) == ("time", "there", "was")  # greedy's path

# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """An icon in another accent, like kit.recolored(), but saved as assets/ep33-….svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    stem = name if name.startswith("ep33-") else f"ep33-{name}"
    out = f"{stem}-{color.lstrip('#').lower()}"
    svg = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{out}.svg"
    if not path.exists() or path.read_text() != svg:
        path.write_text(svg)
    return out


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def mix(a, b, t):
    """A colour between two hex colours."""
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


def sit(words, s, baseline):
    """Shift a Text so its letters without descenders stand on `baseline`."""
    glyph_list = [g for g, c in zip(words, s.replace(" ", "")) if c not in "gjpqy,;()…"] or list(words)
    words.shift((baseline - min(g.get_bottom()[1] for g in glyph_list)) * m.UP)
    return words


def word_chip(word, color=SKY, fill=SKY_FILL, size=28, height=0.72, pad=0.5):
    width = max(text(word, size, weight=m.SEMIBOLD).width + pad, 0.8)
    return chip(word, color, fill, width=width, height=height, size=size)


def flow(words, colors, left, baseline, width, size=30, line_gap=0.62, space=0.2):
    """Words laid out like a paragraph: left to right, wrapping at `width`, on shared baselines."""
    out = m.VGroup()
    x = left
    for w, c in zip(words, colors):
        t = text(w, size, c, weight=m.SEMIBOLD)
        if x > left and x + t.width > left + width:
            x, baseline = left, baseline - line_gap
        sit(t, w, baseline)
        t.shift((x - t.get_left()[0]) * m.RIGHT)
        x += t.width + space
        out.add(t)
    return out


class Rows:
    """A next-token distribution as rows: token · bar · percent (optionally a running total)."""

    def __init__(self, names, x_label=-4.75, y_top=1.5, gap=0.54, scale=4.6, size=24, bar_h=0.3):
        self.names, self.x_label, self.y_top, self.gap, self.scale, self.size, self.bar_h = names, x_label, y_top, gap, scale, size, bar_h

    def y(self, k):
        return self.y_top - self.gap * k

    def bar_end(self, k, p):
        return self.x_label + 0.25 + max(self.scale * p, 0.03)

    def row(self, k, p, bar=BARS, label=INK, number=MUTED, opacity=0.9):
        y = self.y(k)
        base = y - 0.1
        name = self.names[k]
        word = sit(code(name, self.size, label), name, base)
        word.shift((self.x_label - word.get_right()[0]) * m.RIGHT)
        width = max(self.scale * p, 0.03)
        rect = m.Rectangle(width=width, height=self.bar_h, stroke_width=0).set_fill(bar, opacity=opacity)
        rect.move_to([self.x_label + 0.25 + width / 2, y, 0])
        s = f"{100 * p:.0f}%"
        value = sit(code(s, self.size - 2, number), s, base)
        value.shift((rect.get_right()[0] + 0.15 - value.get_left()[0]) * m.RIGHT)
        return m.VGroup(word, rect, value)

    def build(self, probs, styles=None):
        styles = styles or {}
        return m.VGroup(*[self.row(k, float(p), **styles.get(k, {})) for k, p in enumerate(probs)])


def dim_style(t):
    """A row fading out of the running (t = 0 normal, 1 dropped)."""
    return dict(bar=mix(BARS, FAINT, t), label=mix(INK, FAINT, t), number=mix(MUTED, FAINT, t))


def node_box(symbol, label, sub, color, fill, x, y=0.55, width=2.1, height=2.3):
    box = card(width, height, color, fill).move_to([x, y, 0])
    symbol.move_to([x, y + 0.42, 0])
    words = m.VGroup(text(label, 24, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1)
    words.move_to([x, y - 0.6, 0])
    return m.VGroup(box, symbol, words)


def beam_chip(word, total, color=BEAM, fill=BEAM_FILL, width=2.5, height=0.66):
    box = card(width, height, color, fill, radius=0.18, stroke=2.5)
    base = box.get_center()[1] - 0.1
    w = sit(text(word, 24, INK, weight=m.SEMIBOLD), word, base)
    w.shift((box.get_left()[0] + 0.25 - w.get_left()[0]) * m.RIGHT)
    s = f"{100 * total:.0f}%"
    v = sit(code(s, 20, MUTED), s, base)
    v.shift((box.get_right()[0] - 0.22 - v.get_right()[0]) * m.RIGHT)
    return m.VGroup(box, w, v)


def restyle(c, color, fill, ink=INK, number=MUTED):
    """The same beam chip in other colours (a target for Transform)."""
    out = c.copy()
    out[0].set_stroke(color).set_fill(fill, opacity=1)
    out[1].set_color(ink)
    out[2].set_color(number)
    return out


def link(a, b, color=EDGE, width=3):
    return m.Line(a.get_right() + 0.06 * m.RIGHT, b.get_left() + 0.06 * m.LEFT).set_stroke(color, width=width)


def mini_bars(color=BARS):
    """A tiny distribution with its tail trimmed: the top-k / top-p node's picture."""
    heights = (0.75, 0.45, 0.3, 0.16, 0.1)
    bars = m.VGroup()
    for k, h in enumerate(heights):
        c = color if k < 3 else FAINT
        bars.add(m.Rectangle(width=0.16, height=h, stroke_width=0).set_fill(c, opacity=0.95).move_to([0.24 * k, h / 2, 0]))
    cut = m.DashedLine([0.6, -0.05, 0], [0.6, 0.85, 0], dash_length=0.07).set_stroke(INK, width=2.5)
    return m.VGroup(bars, cut)


# ---------------------------------------------------------------------------- the episode


class Ep33DecodingSampling(Episode):
    number = 33

    def lesson(self):
        self.hook()
        self.odds()
        self.greedy()
        self.sampling()
        self.temperature()
        self.top_k()
        self.top_p()
        self.beam()
        self.practice()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A language model finishing 'Once upon a…': it gives a probability to every token; choosing one is decoding.")
        halo = glow(BARS, radius=2.3).move_to([0, 1.55, 0])
        model = icon("llm", 1.35).move_to([0, 1.55, 0])
        chips = [word_chip(w) for w in PROMPT]
        blank = chip("?", BARS, BARS_FILL, width=1.0)
        row = m.VGroup(*chips, blank).arrange(m.RIGHT, buff=0.18).move_to([0, 0.0, 0])
        on_baseline_chips = row  # chips share a baseline already (kit.chip)

        spots = ((0.0, -1.55), (-3.3, -1.2), (3.2, -1.25), (-2.2, -2.35), (2.3, -2.4), (-5.0, -2.0), (5.0, -2.05), (0.1, -2.95))
        cloud = m.VGroup()
        for name, p, (x, y) in zip(TOKENS, P, spots):
            size = int(round(22 + 58 * np.sqrt(p)))
            cloud.add(text(name, size, mix(FAINT, BARS, 0.45 + 0.55 * np.sqrt(p / P[0])), weight=m.BOLD).move_to([x, y, 0]))
        idea = text("decoding: from probabilities to one token", 30, MUTED).move_to([0, -1.45, 0])
        methods = m.VGroup(*[pill(s, TEAL, TEAL_FILL, 24) for s in ("greedy", "temperature", "top-k", "top-p", "beam search")])
        methods.arrange(m.RIGHT, buff=0.35).move_to([0, -2.55, 0])

        self.say(
            "So... [a language model is writing a story.] [It has Once upon a, and it needs the next word.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(model, shift=0.3 * m.UP), run_time=1.2),
            m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in on_baseline_chips], lag_ratio=0.25),
        )
        self.say(
            "[It doesn't come up with one next word.] [It gives a probability to every token it could say next.]",
            m.Circumscribe(blank, color=BARS, buff=0.1),
            m.LaggedStart(*[m.FadeIn(w, scale=0.7) for w in cloud], lag_ratio=0.15),
        )
        self.say(
            "[Picking one of them is called decoding.] [Today: five ways to do it.]",
            m.AnimationGroup(m.FadeOut(cloud, shift=0.2 * m.DOWN), m.FadeIn(idea, shift=0.2 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.UP) for p in methods], lag_ratio=0.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the odds

    def odds(self):
        self.next_section(
            "The odds",
            notes="A TOY model (made-up scores) for the next token after 'Once upon a': "
            + ", ".join(f"{n} {100 * p:.1f}%" for n, p in zip(TOKENS, P))
            + f". The three least likely share {100 * TAIL:.1f}%.",
        )
        rows = Rows(TOKENS)
        self.rows = rows
        bars = rows.build(P)
        self.bars = bars
        context = text("next token after “Once upon a”", 22, MUTED).move_to([-3.35, 2.32, 0])
        self.context = context

        prompt_card = card(5.3, 1.25, SKY, SKY_FILL).move_to([3.95, 0.95, 0])
        prompt = text("Once upon a …", 34, INK, weight=m.SEMIBOLD).move_to(prompt_card)
        toy = pill("a toy model: 8 possible tokens", BARS, BARS_FILL, 24).move_to([3.95, -0.45, 0])
        made_up = text("numbers made up for this example", 22, MUTED).move_to([3.95, -1.25, 0])
        tail_box = m.SurroundingRectangle(m.VGroup(*bars[5:]), buff=0.12, corner_radius=0.12).set_stroke(ODD, width=2.5)
        tail_label = text("a tail of odd ones", 24, ODD, weight=m.SEMIBOLD).next_to(tail_box, m.RIGHT, buff=0.3)
        self.right = m.VGroup(prompt_card, prompt, toy, made_up)

        self.say(
            "[Here's a toy model's guess, with numbers made up for this example.]",
            m.AnimationGroup(
                self.new_header("THE ODDS", "A toy model's next-token guess"),
                m.FadeIn(context),
                m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in bars], lag_ratio=0.12),
                m.FadeIn(prompt_card), m.FadeIn(prompt), m.FadeIn(toy, shift=0.15 * m.UP), m.FadeIn(made_up),
            ),
        )
        self.say(
            "[Time gets sixty percent.] [Then midnight, dream and hill], [and a tail of odd ones, like spoon and banana.]",
            m.Indicate(bars[0], color=BARS, scale_factor=1.06),
            m.LaggedStart(*[m.Indicate(bars[k][0], color=INK, scale_factor=1.15) for k in (1, 2, 3)], lag_ratio=0.3),
            m.AnimationGroup(m.Create(tail_box), m.FadeIn(tail_label, shift=0.15 * m.LEFT)),
        )
        self.wait(0.3)
        self.play(m.FadeOut(m.VGroup(self.right, tail_box, tail_label)), run_time=0.6)

    # ------------------------------------------------------------------ greedy

    def greedy(self):
        self.next_section(
            "Greedy",
            notes="Always the top token. The toy table goes time → there → was → a → time …, a loop "
            "(text degeneration: Holtzman et al., 2019).",
        )
        rows0, bars0, context0 = self.rows, self.bars, self.context
        story_card = card(5.5, 3.9, EDGE, CARD).move_to([3.95, 0.05, 0])
        story_title = text("the story so far", 22, MUTED).move_to(story_card.get_top() + 0.45 * m.DOWN).align_to(story_card, m.LEFT).shift(0.4 * m.RIGHT)
        words = list(GREEDY_WORDS[:15]) + [GREEDY_WORDS[15] + "…"]
        colors = [INK] * 3 + [PICK] * 4 + [ODD] * (len(words) - 7)
        flowed = flow(words, colors, story_card.get_left()[0] + 0.4, story_card.get_top()[1] - 1.15, 4.75, size=30)
        prompt_words, picks, loop = flowed[:3], flowed[3:7], flowed[7:]

        def step_rows(k, top_color=PICK):
            key, names, probs = GREEDY_STEPS[k]
            r = Rows(names)
            built = r.build(probs, {0: dict(bar=top_color, label=top_color, number=top_color)})
            label = text(f"next token after “…{' '.join(key)}”", 22, MUTED).move_to(context0).align_to(context0, m.LEFT)
            return built, label

        first = rows0.build(P, {0: dict(bar=PICK, label=PICK, number=PICK)})
        steps = [step_rows(k) for k in range(1, 4)]
        last, last_label = step_rows(4, ODD)
        stuck = pill("stuck in a loop", ODD, BAD_FILL, 22).next_to(story_card.get_bottom(), m.UP, buff=0.3).align_to(story_card, m.RIGHT).shift(0.35 * m.LEFT)
        good = pill("fast · same answer every time", GOOD, GOOD_FILL, 22)
        bad = pill("can loop · often sounds flat", ODD, BAD_FILL, 22)
        verdicts = m.VGroup(good, bad).arrange(m.RIGHT, buff=0.35).move_to([-0.6, -3.2, 0])
        cite = text("Holtzman et al., 2019", 20, MUTED).next_to(story_card, m.DOWN, buff=0.2).align_to(story_card, m.RIGHT)

        self.say(
            "[The simplest rule is greedy decoding]: [always take the top token.]",
            m.AnimationGroup(
                self.new_header("GREEDY", "Always take the top token"),
                m.FadeIn(story_card), m.FadeIn(story_title), m.FadeIn(prompt_words),
            ),
            m.AnimationGroup(m.FadeTransform(bars0, first), m.FadeIn(picks[0], target_position=bars0[0][0])),
        )
        self.say(
            "[Then it guesses again]: [there.] [And again]: [was.] [And again]: [a.]",
            m.AnimationGroup(m.FadeOut(first), m.FadeOut(context0), m.FadeIn(steps[0][0]), m.FadeIn(steps[0][1]), run_time=0.6),
            m.FadeIn(picks[1], target_position=steps[0][0][0][0], run_time=0.5),
            m.AnimationGroup(m.FadeOut(steps[0][0]), m.FadeOut(steps[0][1]), m.FadeIn(steps[1][0]), m.FadeIn(steps[1][1]), run_time=0.6),
            m.FadeIn(picks[2], target_position=steps[1][0][0][0], run_time=0.5),
            m.AnimationGroup(m.FadeOut(steps[1][0]), m.FadeOut(steps[1][1]), m.FadeIn(steps[2][0]), m.FadeIn(steps[2][1]), run_time=0.6),
            m.FadeIn(picks[3], target_position=steps[2][0][0][0], run_time=0.5),
        )
        self.say(
            "[But now, time is on top again], [and the story goes round in circles.]",
            m.Succession(
                m.AnimationGroup(m.FadeOut(steps[2][0]), m.FadeOut(steps[2][1]), m.FadeIn(last), m.FadeIn(last_label), run_time=0.6),
                m.FadeIn(loop[0], target_position=last[0][0], run_time=0.6),
            ),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(w, shift=0.1 * m.RIGHT) for w in loop[1:]], lag_ratio=0.3, run_time=2.4),
                m.FadeIn(stuck, shift=0.15 * m.UP),
            ),
        )
        self.say(
            "[Greedy is fast and repeatable], [but it can get stuck in loops like this], [and it often sounds flat.]",
            m.FadeIn(good, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(bad, shift=0.2 * m.UP), m.FadeIn(cite)),
            m.Circumscribe(loop, color=ODD, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ sampling

    def sampling(self):
        self.next_section(
            "Sampling",
            notes="One weighted roll per token: u = default_rng(seed).random() picks the slice of the strip it lands in. "
            + " | ".join(f"seed {s}: first roll {SAMPLES[s][1][0]:.3f} → {story(SAMPLES[s][0])}" for s in SEEDS)
            + f". Tail (Tuesday, spoon, banana) = {100 * TAIL:.1f}%.",
        )
        rows = Rows(TOKENS)
        bars = rows.build(P)
        context = text("next token after “Once upon a”", 22, MUTED).move_to([-3.35, 2.32, 0])
        dice = icon("ep33-dice", 1.2).move_to([4.75, -1.0, 0])

        left, right, y, h = -6.0, 6.0, 0.95, 0.62
        span = right - left
        edges = np.concatenate([[0.0], np.cumsum(P)])
        segments = m.VGroup()
        for k in range(len(P)):
            seg = m.Rectangle(width=span * P[k], height=h).set_fill(BARS, opacity=0.9 - 0.06 * (k % 2)).set_stroke(BG, width=3)
            segments.add(seg.move_to([left + span * (edges[k] + edges[k + 1]) / 2, y, 0]))
        names = m.VGroup()
        for k in range(5):
            s = f"{TOKENS[k]} {pct(P[k])}%" if k == 0 else TOKENS[k]
            names.add(sit(text(s, 22, INK, weight=m.SEMIBOLD), s, y + h / 2 + 0.2).set_x(segments[k].get_x()))
        tail = m.VGroup(*segments[5:])
        tail_brace = m.Brace(tail, m.UP, buff=0.08, color=ODD)
        tail_label = text("the tail", 22, ODD, weight=m.SEMIBOLD).next_to(tail_brace, m.UP, buff=0.1).align_to([right, 0, 0], m.RIGHT)
        one_in = pill(f"the tail: {pct(TAIL)}% of rolls, about one in {round(1 / TAIL)}", ODD, BAD_FILL, 22).move_to([3.6, -2.75, 0])
        time_brace = m.Brace(segments[0], m.DOWN, buff=0.08, color=BARS)

        markers, seed_tags, stories = m.VGroup(), m.VGroup(), m.VGroup()
        for i, s in enumerate(SEEDS):
            u = SAMPLES[s][1][0]
            x = left + span * u
            tri = m.Triangle(fill_color=INK, fill_opacity=1, stroke_width=0).scale(0.13).move_to([x, y - h / 2 - 0.2, 0])
            tag = code(str(s), 22, INK).next_to(tri, m.DOWN, buff=0.1)
            markers.add(m.VGroup(tri, tag))
            words = story(SAMPLES[s][0])
            rest = words[len("Once upon a ") :]
            line_y = -0.85 - 0.62 * i
            label = code(f"seed {s}", 22, MUTED).move_to([-5.35, line_y, 0])
            tale = text(words, 28, INK, t2c={rest: ODD if s == 4 else BARS})
            sit(tale, words, line_y - 0.1)
            tale.shift((-4.3 - tale.get_left()[0]) * m.RIGHT)
            sit(label, f"seed {s}", line_y - 0.1)
            seed_tags.add(label)
            stories.add(tale)

        self.say(
            "[Instead, we can sample]: [roll a die weighted by the bars.]",
            m.AnimationGroup(
                self.new_header("SAMPLING", "Roll a weighted die"),
                m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in bars], lag_ratio=0.08, run_time=1.0),
                m.FadeIn(context), m.FadeIn(dice, scale=0.8),
            ),
            m.AnimationGroup(
                *[m.FadeTransform(b, m.VGroup(seg, names[k]) if k < 5 else seg) for k, (b, seg) in enumerate(zip(bars, segments))],
                m.FadeOut(context),
                run_time=1.4,
            ),
        )
        self.say(
            "[Time covers sixty percent of this strip], [so it wins most rolls, but not all.]",
            m.AnimationGroup(m.GrowFromCenter(time_brace), m.Indicate(names[0], color=BARS, scale_factor=1.1)),
            m.LaggedStart(*[m.Indicate(segments[k], color=INK, scale_factor=1.0) for k in (1, 2, 3, 4)], lag_ratio=0.25),
        )
        self.say(
            "[With random seeds one, two and three], [every story starts with time], [then wanders off its own way.]",
            m.AnimationGroup(m.FadeOut(time_brace), m.LaggedStart(*[m.FadeIn(mk, shift=0.25 * m.UP) for mk in markers[:3]], lag_ratio=0.3)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.RIGHT) for t in seed_tags[:3]], lag_ratio=0.2),
            m.LaggedStart(*[m.AddTextLetterByLetter(t, time_per_char=0.03) for t in stories[:3]], lag_ratio=0.35),
        )
        self.say(
            "[Seed four lands in the tail]: [Once upon a Tuesday.]",
            m.AnimationGroup(m.FadeIn(markers[3], shift=0.25 * m.UP), m.Indicate(segments[5], color=ODD, scale_factor=1.0), m.FadeIn(seed_tags[3])),
            m.AddTextLetterByLetter(stories[3], time_per_char=0.04),
        )
        self.say(
            "[The three least likely tokens share about seven percent]: [one roll in fifteen.] [Over a long text, those rolls add up.]",
            m.AnimationGroup(m.GrowFromCenter(tail_brace), m.FadeIn(tail_label, shift=0.15 * m.DOWN)),
            m.FadeIn(one_in, shift=0.15 * m.UP),
            m.Circumscribe(stories[3], color=ODD, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ temperature

    def temperature(self):
        self.next_section(
            "Temperature",
            notes="p_i = exp(z_i / T) / sum_j exp(z_j / T) (Episode 16). "
            + "; ".join(f"T = {t:g}: time {100 * softmax(LOGITS, t)[0]:.1f}%" for t in (FREEZE, COLD, 1.0, HOT)),
        )
        rows = Rows(TOKENS)
        T = m.ValueTracker(1.0)
        live = m.always_redraw(lambda: rows.build(softmax(LOGITS, T.get_value())))
        context = text("next token after “Once upon a”", 22, MUTED).move_to([-3.35, 2.32, 0])

        f = m.MathTex("p_i", "=", r"\frac{", "e^{", "z_i", "/", "T", "}", "}{", r"\sum_j", "e^{", "z_j", "/", "T", "}", "}", font_size=58, color=INK)
        f[0].set_color(BARS)
        for k in (4, 11):
            f[k].set_color(SKY)
        for k in (6, 13):
            f[k].set_color(PICK)
        f.move_to([3.85, 1.55, 0])
        top = m.VGroup(f[2], f[4], f[5], f[6])  # the numerator (its e is drawn by the \frac{ part)
        bottom = m.VGroup(f[9], f[10], f[11], f[12], f[13])
        back = text("softmax with a temperature · Episode 16", 20, MUTED).move_to([3.85, 0.45, 0])

        def term(symbol, color, words, y):
            s = m.MathTex(symbol, font_size=40, color=color)
            d = text(words, 22, MUTED)
            s.move_to([1.85, y, 0])
            d.next_to([2.45, y, 0], m.RIGHT, buff=0)
            return m.VGroup(s, d)

        legend = m.VGroup(
            term("z_i", SKY, "the model's score for token i", -0.1),
            term("T", PICK, "the temperature", -0.68),
            term(r"\sum_j", INK, "a sum over all tokens: total 1", -1.36),
        )

        lo, hi, x0, x1, track_y = 0.1, 3.0, 3.25, 6.3, -2.45

        def at(t):
            return x0 + (np.log(t) - np.log(lo)) / (np.log(hi) - np.log(lo)) * (x1 - x0)

        track = m.Line([x0, track_y, 0], [x1, track_y, 0]).set_stroke(EDGE, width=8)
        marks = (COLD, 1.0, HOT)
        ticks = m.VGroup(*[m.Line([at(t), track_y - 0.16, 0], [at(t), track_y + 0.16, 0]).set_stroke(MUTED, width=2.5) for t in marks])
        tick_labels = m.VGroup(*[text(f"{t:g}", 22, MUTED).move_to([at(t), track_y - 0.48, 0]) for t in marks])
        knob = m.always_redraw(lambda: m.Circle(radius=0.17).set_stroke(INK, width=3).set_fill(PICK, opacity=1).move_to([at(T.get_value()), track_y, 0]))
        readout = m.always_redraw(lambda: code(f"T = {T.get_value():.2f}", 26, PICK, weight=m.BOLD).move_to([x0 - 1.15, track_y, 0]))
        slider = m.VGroup(track, ticks, tick_labels)
        greedy_note = pill("T → 0: greedy", PICK, PICK_FILL, 22).move_to([-4.7, -3.2, 0])
        focused = pill("low T: focused", PICK, PICK_FILL, 22)
        varied = pill("high T: varied, riskier", ODD, BAD_FILL, 22)
        moods = m.VGroup(focused, varied).arrange(m.RIGHT, buff=0.3).move_to([-2.85, -3.2, 0])
        tail_zone = m.Rectangle(width=5.8, height=1.5).move_to([-3.4, rows.y(6), 0]).set_stroke(width=0).set_fill(opacity=0)

        def flash(t):
            p = softmax(LOGITS, t)[0]
            return m.Flash([rows.bar_end(0, p) + 0.5, rows.y(0), 0], color=PICK, flash_radius=0.5, line_length=0.18)

        self.say(
            "[Temperature, from episode sixteen, reshapes the bars before we roll.]",
            m.AnimationGroup(
                self.new_header("TEMPERATURE", "Sharpen or flatten first"),
                m.FadeIn(live), m.FadeIn(context), m.Write(f, run_time=1.4), m.FadeIn(back),
            ),
        )
        self.say(
            "[Each score is divided by the temperature], [and raised as a power of e.]",
            m.AnimationGroup(m.FadeIn(legend[0], shift=0.15 * m.UP), m.FadeIn(legend[1], shift=0.15 * m.UP), m.Indicate(m.VGroup(f[4], f[5], f[6]), color=PICK, scale_factor=1.2)),
            m.Circumscribe(top, color=INK, buff=0.08),
        )
        self.say(
            "[Dividing by the sum over all tokens makes the bars add up to one.]",
            m.AnimationGroup(m.FadeIn(legend[2], shift=0.15 * m.UP), m.Circumscribe(bottom, color=INK, buff=0.08)),
        )
        self.say(
            "[At a temperature of one, nothing changes.] [Turn it down to zero point five], [and time jumps to ninety-one percent.]",
            m.AnimationGroup(m.FadeIn(slider), m.FadeIn(knob), m.FadeIn(readout)),
            T.animate(run_time=1.6).set_value(COLD),
            flash(COLD),
        )
        self.say(
            "[Turn it up to two], [and time falls to thirty-three percent], [while the odd tail grows.]",
            T.animate(run_time=1.8).set_value(HOT),
            flash(HOT),
            m.Circumscribe(tail_zone, color=ODD, buff=0.05),
        )
        self.say(
            "[As the temperature heads toward zero], [time wins every roll: sampling becomes greedy.]",
            T.animate(run_time=2.0).set_value(FREEZE),
            m.FadeIn(greedy_note, shift=0.2 * m.UP),
        )
        self.say(
            "[So: a low temperature for focused answers], [a higher one for variety, at the risk of nonsense.]",
            m.AnimationGroup(T.animate(run_time=1.4).set_value(COLD), m.FadeOut(greedy_note), m.FadeIn(focused, shift=0.2 * m.UP)),
            m.AnimationGroup(T.animate(run_time=1.6).set_value(HOT), m.FadeIn(varied, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.stop(live, knob, readout)
        self.clear()

    # ------------------------------------------------------------------ top-k

    def panels(self):
        """The two contexts side by side: the model sure ('Once upon …') and unsure ('… there lived a …')."""
        out = []
        for x, title, names, probs, tag, tag_color in (
            (-3.4, "Once upon …", SURE_NAMES, SURE_P, "model is sure", GOOD),
            (3.4, "… there lived a …", UNSURE_NAMES, UNSURE_P, "model is unsure", PICK),
        ):
            box = card(6.3, 5.25, EDGE, CARD, stroke=2).move_to([x, -0.55, 0])
            head = text(title, 28, INK, weight=m.BOLD)
            head.move_to(box.get_top() + 0.5 * m.DOWN).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
            badge = text(tag, 22, tag_color, weight=m.SEMIBOLD).align_to(head, m.DOWN).align_to(box, m.RIGHT).shift(0.35 * m.LEFT)
            rows = Rows(names, x_label=box.get_left()[0] + 1.6, y_top=box.get_top()[1] - 1.15, gap=0.45, scale=3.35, size=22, bar_h=0.26)
            out.append(dict(box=box, head=head, badge=badge, rows=rows, probs=probs, names=names, built=rows.build(probs).set_z_index(2)))
        return out

    def top_k(self):
        self.next_section(
            "Top-k",
            notes=f"k = {K}: keep {100 * KEPT_K:.1f}% of the mass, renormalise → time {100 * P_K[0]:.1f}%. "
            f"Sure context keeps a, the, my (a = {100 * SURE_P[0]:.0f}%); unsure context cuts dragon ({100 * UNSURE_P[3]:.0f}%). "
            "Top-k sampling for stories: Fan et al., 2018.",
        )
        rows = Rows(TOKENS)
        s = m.ValueTracker(0.0)

        def probs_now():
            v = s.get_value()
            cut = np.where(np.arange(len(P)) < K, P, 0.0)
            if v <= 1:
                return (1 - v) * P + v * cut
            return (2 - v) * cut + (v - 1) * P_K

        live = m.always_redraw(lambda: rows.build(probs_now(), {k: dim_style(min(s.get_value(), 1)) for k in range(K, len(P))}))
        context = text("next token after “Once upon a”", 22, MUTED).move_to([-3.35, 2.32, 0])

        k_chip = chip("k = 3", PICK, PICK_FILL, width=1.9, height=0.8, size=32).move_to([2.4, 1.6, 0])
        rule_lines = m.VGroup(
            text("keep the 3 most likely tokens", 26, INK),
            text("set the rest to zero", 26, INK),
            text(f"renormalise: divide by {KEPT_K:.2f}", 26, INK, t2c={f"{KEPT_K:.2f}": PICK}),
        ).arrange(m.DOWN, buff=0.32, aligned_edge=m.LEFT).move_to([3.9, 0.15, 0])
        nums = m.VGroup(*[code(str(i + 1), 24, PICK, weight=m.BOLD).next_to(r, m.LEFT, buff=0.3) for i, r in enumerate(rule_lines)])
        cite = text("Fan et al., 2018", 20, MUTED).move_to([3.9, -1.55, 0])
        kept_box = m.SurroundingRectangle(m.VGroup(*rows.build(P)[:K]), buff=0.12, corner_radius=0.12).set_stroke(PICK, width=2.5)
        kept_label = code(f"{pct(KEPT_K)}%", 24, PICK, weight=m.BOLD).next_to(kept_box, m.RIGHT, buff=0.25)

        self.say(
            "[The bluntest trim is called top-k.] [It keeps only the most likely few]: [in our case, the top three.]",
            m.AnimationGroup(self.new_header("TOP-K", "Keep the k best"), m.FadeIn(live), m.FadeIn(context), m.FadeIn(cite)),
            m.AnimationGroup(m.FadeIn(k_chip, scale=0.8), m.FadeIn(nums[0]), m.FadeIn(rule_lines[0], shift=0.15 * m.UP)),
            m.AnimationGroup(s.animate(run_time=1.3).set_value(1.0), m.FadeIn(nums[1]), m.FadeIn(rule_lines[1], shift=0.15 * m.UP)),
        )
        self.say(
            "[Then renormalise]: [divide by what's left, eighty-two percent], [so time becomes seventy-three percent.]",
            m.AnimationGroup(m.FadeIn(nums[2]), m.FadeIn(rule_lines[2], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(kept_box), m.FadeIn(kept_label)),
            m.AnimationGroup(m.FadeOut(kept_box), m.FadeOut(kept_label), s.animate(run_time=1.4).set_value(2.0)),
        )
        self.wait(0.2)
        self.stop(live)

        sure, unsure = self.panels()

        def cut_rows(panel):
            styles = {k: dict(bar=ODD, label=ODD, number=ODD) for k in range(K)} if panel is sure else {}
            styles.update({k: dim_style(1.0) for k in range(K, len(panel["probs"]))})
            if panel is sure:
                styles[0] = {}
            return panel["rows"].build(panel["probs"], styles).set_z_index(2)

        sure_cut, unsure_cut = cut_rows(sure), cut_rows(unsure)
        line_k = {}
        for panel in (sure, unsure):
            r = panel["rows"]
            y = (r.y(K - 1) + r.y(K)) / 2
            line_k[id(panel)] = m.DashedLine([panel["box"].get_left()[0] + 0.3, y, 0], [panel["box"].get_right()[0] - 0.3, y, 0], dash_length=0.1).set_stroke(PICK, width=2.5)
        verdict_sure = text("still keeps 3: “the”, “my”", 24, ODD, weight=m.SEMIBOLD).move_to(sure["box"].get_bottom() + 0.42 * m.UP)
        verdict_unsure = text("cuts “dragon” (14%)", 24, ODD, weight=m.SEMIBOLD).move_to(unsure["box"].get_bottom() + 0.42 * m.UP)
        dragon = unsure_cut[K]

        self.say(
            "[The catch: that number is fixed.] [After Once upon, the model is sure], [yet top-k still keeps three tokens.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (live, context, k_chip, rule_lines, nums, cite)],
                *[m.FadeIn(p[key]) for p in (sure, unsure) for key in ("box", "head")],
                run_time=1.0,
            ),
            m.AnimationGroup(m.FadeIn(sure["badge"]), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in sure["built"]], lag_ratio=0.1)),
            m.AnimationGroup(*[m.ReplacementTransform(a, b) for a, b in zip(sure["built"], sure_cut)], m.Create(line_k[id(sure)]), m.FadeIn(verdict_sure, shift=0.15 * m.UP)),
        )
        self.say(
            "[After there lived a, it's unsure], [and the cut throws away dragon, almost as likely as girl.]",
            m.AnimationGroup(m.FadeIn(unsure["badge"]), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in unsure["built"]], lag_ratio=0.1)),
            m.Succession(
                m.AnimationGroup(m.ReplacementTransform(unsure["built"], unsure_cut), m.Create(line_k[id(unsure)]), run_time=1.0),
                m.AnimationGroup(m.Indicate(dragon[0], color=ODD, scale_factor=1.2), m.FadeIn(verdict_unsure, shift=0.15 * m.UP)),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ top-p

    def top_p(self):
        self.next_section(
            "Top-p",
            notes=f"Nucleus sampling (Holtzman et al., 2019): the smallest set whose total reaches p = {TOP_P}. "
            f"Running total {', '.join(f'{100 * v:.1f}%' for v in RUNNING[:N_P])} → {N_P} tokens; renormalised time {100 * P_P[0]:.1f}%. "
            f"Sure context keeps {nucleus(SURE_P)}, unsure keeps {nucleus(UNSURE_P)}.",
        )
        rows = Rows(TOKENS)
        s = m.ValueTracker(0.0)

        def probs_now():
            v = s.get_value()
            cut = np.where(np.arange(len(P)) < N_P, P, 0.0)
            if v <= 1:
                return (1 - v) * P + v * cut
            return (2 - v) * cut + (v - 1) * P_P

        live = m.always_redraw(lambda: rows.build(probs_now(), {k: dim_style(min(s.get_value(), 1)) for k in range(N_P, len(P))}))
        context = text("next token after “Once upon a”", 22, MUTED).move_to([-3.35, 2.32, 0])
        col_x = 0.55
        heading = text("running total", 22, PICK, weight=m.SEMIBOLD).move_to([col_x, 2.32, 0])
        totals = m.VGroup()
        for k in range(N_P):
            st = f"{pct(RUNNING[k])}%"
            totals.add(sit(code(st, 24, PICK), st, rows.y(k) - 0.1).set_x(col_x))
        threshold_y = (rows.y(N_P - 1) + rows.y(N_P)) / 2
        threshold = m.DashedLine([-0.25, threshold_y, 0], [1.35, threshold_y, 0], dash_length=0.1).set_stroke(PICK, width=2.5)
        p_tag = text("reached 0.9: stop", 22, PICK, weight=m.SEMIBOLD).next_to([1.5, threshold_y, 0], m.RIGHT, buff=0.1)
        nucleus_box = m.SurroundingRectangle(m.VGroup(*rows.build(P_P)[:N_P]), buff=0.1, corner_radius=0.12).set_stroke(GOOD, width=2.5)
        nucleus_label = pill(f"the nucleus: {N_P} tokens", GOOD, GOOD_FILL, 22).move_to([4.6, -1.8, 0])

        p_chip = chip("p = 0.9", PICK, PICK_FILL, width=2.1, height=0.8, size=32).move_to([4.6, 1.6, 0])
        rule = m.VGroup(
            text("keep the smallest set of tokens", 24, INK),
            text("whose total reaches p", 24, INK),
        ).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([4.6, 0.55, 0])
        cite = text("Holtzman et al., 2019", 20, MUTED).move_to([4.6, -0.35, 0])

        self.say(
            "[A smarter trim is called top-p, or nucleus sampling.] [Add up the sorted probabilities, starting from the top.]",
            m.AnimationGroup(
                self.new_header("TOP-P", "Nucleus sampling"),
                m.FadeIn(live), m.FadeIn(context), m.FadeIn(p_chip, scale=0.8), m.FadeIn(rule), m.FadeIn(cite),
            ),
            m.AnimationGroup(m.FadeIn(heading), m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.DOWN) for t in totals[:N_P]], lag_ratio=0.35, run_time=2.2)),
        )
        self.say(
            "[Stop as soon as the total reaches a set level, here zero point nine]: [that's five tokens, from time down to star.]",
            m.AnimationGroup(m.Create(threshold), m.FadeIn(p_tag, shift=0.15 * m.LEFT), m.Indicate(totals[N_P - 1], color=PICK, scale_factor=1.25)),
            m.AnimationGroup(m.Create(nucleus_box), m.FadeIn(nucleus_label), s.animate(run_time=1.2).set_value(1.0)),
        )
        self.say(
            "[Renormalise those five, and sample as before.]",
            s.animate(run_time=1.4).set_value(2.0),
        )
        self.wait(0.2)
        self.stop(live)

        sure, unsure = self.panels()
        boxes, verdicts = [], []
        for panel, words in ((sure, "keeps 1 token"), (unsure, "keeps 7 tokens")):
            n = nucleus(panel["probs"])
            boxes.append(m.SurroundingRectangle(m.VGroup(*panel["built"][:n]), buff=0.07, corner_radius=0.1).set_stroke(GOOD, width=2.5).set_z_index(3))
            verdicts.append(text(words, 24, GOOD, weight=m.SEMIBOLD).move_to(panel["box"].get_bottom() + 0.42 * m.UP))
            assert words.startswith(f"keeps {n} ")
        dims = [
            panel["rows"].build(panel["probs"], {k: dim_style(1.0) for k in range(nucleus(panel["probs"]), len(panel["probs"]))}).set_z_index(2)
            for panel in (sure, unsure)
        ]

        self.say(
            "[When the model is sure, the nucleus shrinks to one token.] [When it's unsure, it grows to seven.]",
            m.Succession(
                m.AnimationGroup(
                    *[m.FadeOut(mob) for mob in (live, context, heading, totals, threshold, p_tag, nucleus_box, nucleus_label, p_chip, rule, cite)],
                    *[m.FadeIn(p[key]) for p in (sure, unsure) for key in ("box", "head", "badge")],
                    m.FadeIn(sure["built"]),
                    run_time=0.9,
                ),
                m.AnimationGroup(m.ReplacementTransform(sure["built"], dims[0]), m.Create(boxes[0]), m.FadeIn(verdicts[0], shift=0.15 * m.UP), run_time=1.0),
            ),
            m.Succession(
                m.FadeIn(unsure["built"], run_time=0.6),
                m.AnimationGroup(m.ReplacementTransform(unsure["built"], dims[1]), m.Create(boxes[1]), m.FadeIn(verdicts[1], shift=0.15 * m.UP), run_time=1.0),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ beam search

    def beam(self):
        (c1, k1), (c2, k2), (c3, k3) = BEAMS
        self.next_section(
            "Beam search",
            notes="Width 2, three steps, total probability = product of the step probabilities. "
            + " | ".join(f"step {i + 1}: " + ", ".join(f"{' '.join(w[3:])} {100 * t:.1f}%" for w, t, _ in c) for i, (c, _) in enumerate(BEAMS))
            + f". Greedy's path {100 * GREEDY_TOTAL:.1f}%, the beam's best {100 * BEAM_TOTAL:.1f}%. "
            "Common in machine translation; less so for open-ended chat and stories (Holtzman et al., 2019).",
        )
        dy = -0.25
        root = word_chip("Once upon a", SKY, SKY_FILL, size=26, height=0.66).move_to([-4.45, 0.4 + dy, 0])
        xs = (-1.5, 1.55, 4.6)
        layout = [
            [(c, y) for c, y in zip(c1[:3], (1.3, 0.4, -0.5))],
            [(c, y) for c, y in zip(sorted(c2, key=lambda c: (c[2] != k1[0][0], -c[1]))[:5], (1.9, 1.0, 0.1, -0.8, -1.7))],
            [(c, y) for c, y in zip(sorted(c3, key=lambda c: (c[2] != k2[0][0], -c[1]))[:4], (2.15, 1.45, 0.75, 0.05))],
        ]
        chips, links, places = [], [], {tuple(PROMPT): root}
        for col, (x, items) in enumerate(zip(xs, layout)):
            col_chips, col_links = {}, []
            for (words, total, parent), y in items:
                c = beam_chip(words[-1], total, EDGE, CARD).move_to([x, y + dy, 0])
                places[words] = c
                col_chips[words] = c
                col_links.append(link(places[parent], c))
            chips.append(col_chips)
            links.append(m.VGroup(*col_links))
        step_labels = m.VGroup(*[text(f"step {i + 1}", 22, MUTED, weight=m.SEMIBOLD).move_to([x, 2.6, 0]) for i, x in enumerate(xs)])
        width_chip = pill("beam width 2", BEAM, BEAM_FILL, 22).move_to([-4.45, -0.75 + dy, 0])

        def judged(col, kept):
            keep = {w for w, _, _ in kept}
            anims = []
            for words, c in chips[col].items():
                if words in keep:
                    anims.append(m.Transform(c, restyle(c, BEAM, BEAM_FILL)))
                else:
                    anims.append(m.Transform(c, restyle(c, FAINT, CARD, FAINT, FAINT)))
            return m.AnimationGroup(*anims)

        def path(words, color, fill):
            anims = []
            for n in range(4, len(words) + 1):
                c = places[words[:n]]
                anims.append(m.Transform(c, restyle(c, color, fill)))
            return m.AnimationGroup(*anims)

        greedy_path = tuple(GREEDY_WORDS[:6])
        beam_path = k3[0][0]
        after_time, after_there, after_long = next_token(["a", "time"]), next_token(["time", "there"]), next_token(["time", "long"])
        assert after_time[0][:2] == ["there", "long"] and after_there[0][0] == "was" and after_long[0][0] == "ago"
        g_calc = text(
            f"greedy: {pct(P[0])}% × {pct(after_time[1][0])}% × {pct(after_there[1][0])}% = {pct(GREEDY_TOTAL)}%", 24, PICK, weight=m.SEMIBOLD
        ).move_to([-3.3, -2.6, 0])
        b_calc = text(
            f"beam: {pct(P[0])}% × {pct(after_time[1][1])}% × {pct(after_long[1][0])}% = {pct(BEAM_TOTAL)}%", 24, GOOD, weight=m.SEMIBOLD
        ).move_to([3.3, -2.6, 0])
        uses = m.VGroup(
            pill("translation: one right answer", GOOD, GOOD_FILL, 22),
            pill("chat and stories: bland, repetitive", ODD, BAD_FILL, 22),
        ).arrange(m.RIGHT, buff=0.4).move_to([0, -3.35, 0])

        self.say(
            "[Beam search doesn't roll dice at all.] [It keeps several partial sentences, called beams], [here two.]",
            m.AnimationGroup(self.new_header("BEAM SEARCH", "Keep several partial sentences"), m.FadeIn(root, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(step_labels[0]), m.Create(links[0]), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in chips[0].values()], lag_ratio=0.2)),
            m.AnimationGroup(judged(0, k1), m.FadeIn(width_chip, shift=0.15 * m.UP)),
        )
        self.say(
            "[At each step, it extends every beam], [scores each by its total probability], [and keeps the best two again.]",
            m.AnimationGroup(m.FadeIn(step_labels[1]), m.Create(links[1]), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in chips[1].values()], lag_ratio=0.15)),
            m.LaggedStart(*[m.Indicate(c[2], color=INK, scale_factor=1.25) for c in chips[1].values()], lag_ratio=0.15),
            judged(1, k2),
        )
        self.say(
            "[Step three.] [Greedy's path, time there was, scores seventeen percent.] [Beam search finds time long ago, at nineteen.]",
            m.AnimationGroup(m.FadeIn(step_labels[2]), m.Create(links[2]), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in chips[2].values()], lag_ratio=0.15)),
            m.AnimationGroup(path(greedy_path, PICK, PICK_FILL), m.FadeIn(g_calc, shift=0.15 * m.UP)),
            m.AnimationGroup(path(beam_path, GOOD, GOOD_FILL), m.FadeIn(b_calc, shift=0.15 * m.UP)),
        )
        self.say(
            "[That's great for translation, where one answer is right.] [For chat and stories, it tends to sound bland and repetitive], [so sampling is more common there.]",
            m.FadeIn(uses[0], shift=0.2 * m.UP),
            m.FadeIn(uses[1], shift=0.2 * m.UP),
            m.Circumscribe(uses[1], color=ODD, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ in practice

    def practice(self):
        self.next_section("In practice", notes="A common recipe: scores → divide by T → softmax → top-k / top-p → one weighted roll. The settings card is an example, not a recommendation.")
        xs = (-5.55, -2.78, 0.0, 2.78, 5.55)
        nodes = [
            node_box(tex("z", 64, SKY), "scores", "one per token", SKY, SKY_FILL, xs[0]),
            node_box(tex(r"\div T", 56, PICK), "temperature", "sharpen, flatten", PICK, PICK_FILL, xs[1]),
            node_box(tex(r"\frac{e^{z}}{\sum e^{z}}", 40, BARS), "softmax", "add up to one", BARS, BARS_FILL, xs[2]),
            node_box(mini_bars(), "top-k · top-p", "trim the tail", TEAL, TEAL_FILL, xs[3]),
            node_box(icon(variant("ep33-dice", GOOD, GOOD_FILL), 0.85), "sample", "roll the die", GOOD, GOOD_FILL, xs[4]),
        ]
        arrows = [arrow(a[0].get_right(), b[0].get_left(), MUTED) for a, b in zip(nodes, nodes[1:])]
        settings = card(7.4, 1.25, EDGE, CARD).move_to([0, -2.35, 0])
        settings_text = code("temperature = 0.7     top_p = 0.9", 26, INK, t2c={"0.7": PICK, "0.9": TEAL}).move_to(settings)
        example = text("settings, for example", 20, MUTED).next_to(settings, m.UP, buff=0.15).align_to(settings, m.LEFT)

        self.say(
            "[A typical recipe chains it all together]: [scores], [divided by temperature], [softmax], [trimmed by top-k or top-p], [and one weighted roll.]",
            self.new_header("IN PRACTICE", "One recipe, many knobs"),
            m.FadeIn(nodes[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(nodes[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(nodes[2], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[2]), m.FadeIn(nodes[3], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[3]), m.FadeIn(nodes[4], shift=0.2 * m.UP)),
        )
        self.say(
            "[Many chat tools let you set temperature and top-p yourself.] [Now you know what those knobs do.]",
            m.AnimationGroup(m.FadeIn(settings), m.FadeIn(settings_text), m.FadeIn(example)),
            m.AnimationGroup(m.Circumscribe(nodes[1], color=PICK, buff=0.08), m.Circumscribe(nodes[3], color=TEAL, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Score, filter, sample.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Score", "Filter", "Sample"), (SKY, TEAL, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("The model gives the odds. Decoding makes the choice.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [score], [filter], [sample.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[The model gives the odds. Decoding makes the choice.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)

