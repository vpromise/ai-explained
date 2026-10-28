"""AI, Explained · Episode 27 · Pretraining a Language Model

Running example: one sentence, "Water boils at 100 degrees Celsius.", seven tokens. The model
guesses each next token from the tokens before it; cross-entropy per token (minus the log of
the probability it gave the true token) scores each guess; the sentence's loss is the average;
one gradient step makes every true token a little more likely. Then the same game a trillion
times: the data pipeline (crawl, filter, deduplicate, mix), the scale (tokens, GPUs, weeks to
months), an illustrative training run with checkpoints and a crash, and what comes out: a base
model, a text predictor that is not yet an assistant (Episode 29).

Illustrative (for teaching): the model's probabilities for the sentence, the guesses after
"Water boils at", the 50,000-token vocabulary, the size of the one-step nudge, the loss curve
(a power law plus seeded noise) and the base model's continuation. Reported facts (Llama 3,
Dubey et al., 2024, arXiv 2407.21783, checked 2026-10-21): about 15T training tokens; the
405B model on up to 16K H100 GPUs; final data mix ~50% general knowledge, 25% math and
reasoning, 17% code, 8% multilingual; 466 job interruptions in a 54-day snapshot of
pre-training. Every number the narration quotes is asserted below.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

SENTENCE = ("Water", "boils", "at", "100", "degrees", "Celsius", ".")
# p(true next token | every token before it) for tokens 2..7, from a trained model (illustrative)
P_TRUE = np.array([0.05, 0.55, 0.62, 0.90, 0.70, 0.80])
LOSSES = -np.log(P_TRUE)  # cross-entropy per token, natural log
SENT_LOSS = float(LOSSES.mean())
T = len(P_TRUE)
assert T == len(SENTENCE) - 1 == 6  # "all six"
assert round(100 * P_TRUE[0]) == 5 and round(float(LOSSES[0])) == 3  # boils: "five percent, a loss of about three"
assert round(float(LOSSES[2]), 2) == 0.48  # "minus the log of zero point six two is about zero point four eight"
assert round(float(LOSSES[3]), 1) == 0.1  # degrees: "about zero point one"
assert round(SENT_LOSS, 1) == 0.8  # "about zero point eight"
assert round(float(-np.log(0.99)), 2) == 0.01  # "if it had been sure ... almost zero"

# the model's guesses after "Water boils at" (illustrative); the rest of the vocabulary shares the remainder
VOCAB = 50_000
GUESSES = (("100", 0.62), ("a", 0.14), ("sea", 0.05), ("lower", 0.04))
REST = 1 - sum(p for _, p in GUESSES)
assert GUESSES[0][1] == P_TRUE[2] and round(100 * GUESSES[0][1]) == 62  # "sixty-two percent for one hundred"
assert abs(REST - 0.15) < 1e-9


def guess_probs(p):
    """All five bars when the true token gets p: the others keep their proportions, the total stays 1."""
    others = [q for _, q in GUESSES[1:]] + [REST]
    return [p] + [q * (1 - p) / (1 - GUESSES[0][1]) for q in others]


# pure guessing: every token gets 1/VOCAB, so each costs log(VOCAB)
START_LOSS = float(np.log(VOCAB))
assert round(START_LOSS, 1) == 10.8  # "near ten point eight"

# one gradient step, pictured as a nudge of +STEP to each true token's logit (illustrative)
STEP = 0.35
P_AFTER = P_TRUE * np.exp(STEP) / (P_TRUE * np.exp(STEP) + 1 - P_TRUE)
SENT_AFTER = float(-np.log(P_AFTER).mean())
assert np.all(P_AFTER > P_TRUE) and SENT_AFTER < SENT_LOSS  # "a little more likely"

# scale: Llama 3 (Dubey et al., 2024)
LLAMA3_TOKENS = 15e12  # "about 15T multilingual tokens"
YEARS = LLAMA3_TOKENS / (365.25 * 24 * 3600)  # read at one token per second
assert 450_000 < YEARS < 500_000  # "nearly half a million years"
YEARS_SHOWN = int(round(YEARS, -3))
assert YEARS_SHOWN == 475_000
INTERRUPTIONS, DAYS = 466, 54  # "four hundred and sixty-six interruptions in fifty-four days"

BLUE_FILL, LIME_FILL = "#142240", "#1C2810"
MIX = (
    ("general knowledge", 0.50, BLUE, BLUE_FILL),
    ("math & reasoning", 0.25, PINK, PINK_FILL),
    ("code", 0.17, LIME, LIME_FILL),
    ("multilingual", 0.08, SKY, SKY_FILL),
)
assert abs(sum(f for _, f, _, _ in MIX) - 1) < 1e-9 and MIX[0][1] == 0.5 and MIX[1][1] == 0.25  # "half", "a quarter"

# the pipeline's toy documents: 16 crawled, 4 junk filtered out, 3 copies removed, 9 mixed
_doc_rng = np.random.default_rng(2027)
DOC_LINES = [_doc_rng.uniform(0.11, 0.27, 3) for _ in range(16)]
TWINS = ((0, 4), (7, 9), (12, 15))  # (original, copy)
for _original, _copy in TWINS:
    DOC_LINES[_copy] = DOC_LINES[_original]
JUNK = (2, 6, 11, 14)
KEPT1 = [i for i in range(16) if i not in JUNK]
KEPT2 = [i for i in KEPT1 if i not in {c for _, c in TWINS}]
KIND = {0: 0, 3: 0, 8: 0, 13: 0, 1: 1, 10: 1, 5: 2, 12: 2, 7: 3}  # index → MIX category
assert sorted(KIND) == KEPT2

# an illustrative training run: loss against tokens seen (trillions), a power law plus seeded noise
RUN_D = np.linspace(0, 15, 751)
E_FLOOR, D0, ALPHA = 1.7, 0.05, 0.5
_rng = np.random.default_rng(27)
_noise = np.convolve(_rng.normal(0, 1, RUN_D.size + 8), np.ones(9) / 9, mode="valid")
RUN_L = E_FLOOR + (START_LOSS - E_FLOOR) * (1 + RUN_D / D0) ** -ALPHA + 0.25 * _noise * np.minimum(1, RUN_D / 0.5)
assert np.isclose(RUN_L[0], START_LOSS) and RUN_L[-1] < 2.6 and RUN_L.max() <= START_LOSS + 1e-9
CHECKPOINTS = np.arange(1.5, 15.01, 1.5)
CRASH = 10.3
RESUME = float(CHECKPOINTS[CHECKPOINTS < CRASH].max())  # 9.0


def run_loss(d):
    return float(np.interp(d, RUN_D, RUN_L))


# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """kit.recolored(), but written under this episode's prefix: assets/ep27-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    base = re.sub(r"^ep\d\d-", "", name)
    stem = f"ep27-{base}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{stem}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return stem


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def token_chip(word, color=SKY, fill=SKY_FILL, size=28, height=0.72, pad=0.5):
    """A token in a chip as wide as its word (the period token gets a drawn dot on the baseline)."""
    if word == ".":
        box = card(0.8, height, color, fill, radius=0.18, stroke=2.5)
        baseline = box.get_center()[1] - 0.449 * size / 48 / 2
        dot = m.Dot(radius=0.065, color=INK).move_to([box.get_x(), baseline + 0.065, 0])
        return m.VGroup(box, dot)
    width = max(text(word, size, weight=m.SEMIBOLD).width + pad, 0.8)
    return chip(word, color, fill, width=width, height=height, size=size)


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


def guess_rows(probs, x_label, y_top, true_color=VIOLET, gap=0.6, scale=4.6):
    """The model's guesses: token · bar · percent, one row each; the last row is everything else."""
    names = [g for g, _ in GUESSES] + [f"{VOCAB - len(GUESSES):,} others"]
    rows = m.VGroup()
    for k, (name, p) in enumerate(zip(names, probs)):
        y = y_top - gap * k
        last = k == len(GUESSES)
        color = true_color if k == 0 else (FAINT if last else VIOLET)
        bar = m.Rectangle(width=max(scale * p, 0.03), height=0.34, stroke_width=0).set_fill(color, opacity=0.9)
        bar.move_to([x_label + 0.22, y, 0], aligned_edge=m.LEFT)
        pct = code(f"{100 * p:.0f}%", 22, MUTED).next_to(bar, m.RIGHT, buff=0.15)
        if last:
            label = text(name, 22, MUTED).move_to([x_label, y, 0], aligned_edge=m.RIGHT)
        else:
            label = code(name, 24, true_color if k == 0 else INK).move_to([x_label, y, 0], aligned_edge=m.RIGHT)
            label.align_to(pct, m.DOWN)
        rows.add(m.VGroup(label, bar, pct))
    return rows


def mini_doc(lines, color=MUTED, fill=CARD):
    """A tiny document drawn in code: its line pattern tells copies apart from originals."""
    box = m.RoundedRectangle(corner_radius=0.08, width=0.56, height=0.72, stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1)
    marks = m.VGroup(*[m.Line(m.ORIGIN, 1.2 * w * m.RIGHT).set_stroke(color, width=4) for w in lines])
    marks.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    return m.VGroup(box, marks)


def row_xs(n, step=0.8):
    return [(i - (n - 1) / 2) * step for i in range(n)]


def xmark(at, color=BAD, size=0.17, width=7):
    return m.VGroup(
        m.Line(at + size * (m.UL), at + size * (m.DR)).set_stroke(color, width=width),
        m.Line(at + size * (m.UR), at + size * (m.DL)).set_stroke(color, width=width),
    )


# ---------------------------------------------------------------------------- the episode


class Ep27Pretraining(Episode):
    number = 27

    def lesson(self):
        self.hook()
        self.objective()
        self.cross_entropy()
        self.every_token()
        self.data()
        self.scale()
        self.training_run()
        self.result()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Before a chatbot exists: months of one game, guess the next token. One sentence, a trillion times.")
        halo = glow(VIOLET, radius=2.6).move_to([0, 1.4, 0])
        model = icon("llm", 1.5).move_to([0, 1.45, 0])
        hundred = token_chip("100", GOOD, GOOD_FILL)
        blank = chip("?", VIOLET, VIOLET_FILL, width=hundred[0].width)
        words = m.VGroup(*[token_chip(w) for w in SENTENCE[:3]], blank).arrange(m.RIGHT, buff=0.18).move_to([0, -0.4, 0])
        hundred.move_to(blank)
        power = m.ValueTracker(0)
        count = m.always_redraw(lambda: code(f"× {int(round(10 ** power.get_value())):,}", 44, AMBER).move_to([0, -1.75, 0]))
        tagline = text("one sentence, a trillion times", 28, MUTED).move_to([0, -2.6, 0])

        self.say(
            "So... [before a chatbot can answer a single question], [it spends months playing one simple game]: [guess the next token.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(model, shift=0.3 * m.UP), run_time=1.2),
            m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in words[:3]], lag_ratio=0.25),
            m.Succession(m.FadeIn(blank, scale=0.8), m.Indicate(blank, color=VIOLET, scale_factor=1.15)),
        )
        self.say(
            "[It checks the real next token in the text], [and adjusts itself, just a little.]",
            m.FadeTransform(blank, hundred),
            m.AnimationGroup(m.Indicate(model, color=VIOLET, scale_factor=1.08), m.Flash(model.get_center(), color=VIOLET, flash_radius=1.15, line_length=0.3)),
        )
        self.say(
            "[Today, we'll follow one sentence through that game], [and then repeat it a trillion times.]",
            m.Circumscribe(m.VGroup(*words[:3], hundred), color=SKY, buff=0.12),
            m.Succession(
                m.FadeIn(count, run_time=0.4),
                power.animate(run_time=2.6, rate_func=m.linear).set_value(12),
                m.FadeIn(tagline, shift=0.2 * m.UP, run_time=0.6),
            ),
        )
        self.stop(count)
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the objective

    def objective(self):
        self.next_section(
            "The objective",
            notes="Tokens (Episode 23); the context; a probability for every token in the vocabulary; the true next token is in the text "
            "(self-supervised, Episode 8). The guesses 62% / 14% / 5% / 4% and the 50,000-token vocabulary are illustrative.",
        )
        plain = text("Water boils at 100 degrees Celsius.", 40, INK, weight=m.SEMIBOLD).move_to([0, 1.7, 0])
        chips = m.VGroup(*[token_chip(w) for w in SENTENCE]).arrange(m.RIGHT, buff=0.16).move_to([0, 1.7, 0])
        covers = m.VGroup()
        for c in chips[3:]:
            cover = chip("?", EDGE, CARD, width=c[0].width, height=c[0].height).move_to(c)
            cover[1].set_color(MUTED)
            covers.add(cover)
        context = chips[:3]
        y_under = chips.get_bottom()[1] - 0.14
        under = m.Line([context.get_left()[0] + 0.05, y_under, 0], [context.get_right()[0] - 0.05, y_under, 0]).set_stroke(SKY, width=4)
        context_label = text("context", 22, SKY, weight=m.SEMIBOLD).next_to(under, m.DOWN, buff=0.12)
        model = icon("llm", 1.3).move_to([context.get_x(), -1.3, 0])
        model_label = text("the model", 22, VIOLET, weight=m.SEMIBOLD).next_to(model, m.DOWN, buff=0.15)
        down = arrow(context_label.get_bottom(), model.get_top(), MUTED, buff=0.12)
        rows = guess_rows(guess_probs(GUESSES[0][1]), x_label=0.9, y_top=0.05)
        rows_title = text("its guess for the next token", 22, MUTED).next_to(rows, m.UP, buff=0.3)
        out = arrow(model.get_right(), [rows.get_left()[0], model.get_y(), 0], MUTED, buff=0.2)
        true_chip = token_chip("100", GOOD, GOOD_FILL).move_to(covers[0])
        key = pill("the text is its own answer key · self-supervised, Episode 8", GOOD, GOOD_FILL, 22).move_to([0, -3.2, 0])

        self.say(
            "[Here's our sentence]: [water boils at one hundred degrees Celsius], [split into tokens, as in episode twenty-three.]",
            self.new_header("THE OBJECTIVE", "Predict the next token"),
            m.FadeIn(plain, shift=0.2 * m.UP),
            m.FadeTransform(plain, chips),
        )
        self.say(
            "[Hide the rest of the sentence after the first three tokens], [and ask the model what comes next.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.ReplacementTransform(c, cover) for c, cover in zip(chips[3:], covers)], lag_ratio=0.15),
                m.Create(under),
                m.FadeIn(context_label),
            ),
            m.AnimationGroup(m.GrowArrow(down), m.FadeIn(model, shift=0.2 * m.DOWN), m.FadeIn(model_label)),
        )
        self.say(
            "[It answers with a probability for every token it knows]: [for example, sixty-two percent for one hundred.]",
            m.AnimationGroup(
                m.GrowArrow(out),
                m.FadeIn(rows_title),
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2])) for r in rows], lag_ratio=0.2),
            ),
            m.Indicate(rows[0], color=VIOLET, scale_factor=1.08),
        )
        self.say(
            "[And the right answer is already in the text.] [No one labels anything]: [the text is its own answer key.]",
            m.AnimationGroup(
                m.FadeTransform(covers[0], true_chip),
                rows[0][1].animate.set_fill(GOOD),
                rows[0][0].animate.set_color(GOOD),
            ),
            m.Indicate(true_chip, color=GOOD, scale_factor=1.12),
            m.FadeIn(key, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.rows = rows
        self.leftovers = [*chips[:3], *covers[1:], true_chip, under, context_label, model, model_label, down, out, rows_title, key]

    # ------------------------------------------------------------------ cross-entropy per token

    def cross_entropy(self):
        self.next_section(
            "Cross-entropy per token",
            notes="loss_t = −log p(x_t | x_<t), natural log (Episode 16). −log 0.62 = 0.478. If p were 0.99 the loss is 0.01; at 0.05 it is 3.0. "
            "The bars move as one what-if: the other guesses keep their proportions.",
        )
        shift = np.array([-5.5, 0.85, 0])
        x_label, y_top = 0.9 + shift[0], 0.05 + shift[1]
        context_title = text("after: Water boils at", 24, SKY, weight=m.SEMIBOLD)
        context_title.move_to([self.rows.get_center()[0] + shift[0], y_top + 0.7, 0])
        p = m.ValueTracker(GUESSES[0][1])
        live = m.always_redraw(lambda: guess_rows(guess_probs(p.get_value()), x_label=x_label, y_top=y_top, true_color=GOOD))

        formula = m.MathTex(r"\ell_t", "=", r"-\log", r"p(", r"x_t", r"\mid", r"x_{<t}", r")", font_size=54, color=INK)
        for k, c in ((0, BAD), (3, VIOLET), (4, GOOD), (5, MUTED), (6, SKY), (7, VIOLET)):
            formula[k].set_color(c)
        formula.move_to([3.35, 1.75, 0])

        def entry(symbol, color, words):
            return m.VGroup(tex(symbol, 38, color), text(words, 24, MUTED)).arrange(m.RIGHT, buff=0.3)

        legend = m.VGroup(
            entry(r"\ell_t", BAD, "the loss for this token"),
            entry(r"p", VIOLET, "the model's probability"),
            entry(r"x_t", GOOD, "the true next token: 100"),
            entry(r"x_{<t}", SKY, "every token before it"),
        ).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
        column = max(e[0].get_right()[0] for e in legend) + 0.3
        for e in legend:  # symbols in one column, words in another
            e[1].shift((column - e[1].get_left()[0]) * m.RIGHT)
        legend.move_to([3.35, -0.2, 0])

        p_num = m.DecimalNumber(GUESSES[0][1], num_decimal_places=2, mob_class=Mono, font_size=36, color=GOOD)
        p_num.add_updater(lambda d: d.set_value(p.get_value()))
        l_num = m.DecimalNumber(-np.log(GUESSES[0][1]), num_decimal_places=2, mob_class=Mono, font_size=36, color=BAD)
        l_num.add_updater(lambda d: d.set_value(-np.log(p.get_value())))
        lhs = m.VGroup(code("−log", 36, INK), p_num).arrange(m.RIGHT, buff=0.22)
        rhs = m.VGroup(code("≈", 36, MUTED), l_num).arrange(m.RIGHT, buff=0.22)
        worked = m.VGroup(lhs, rhs).arrange(m.RIGHT, buff=0.25).move_to([3.35, -2.35, 0])

        self.say(
            "[So how wrong was that guess?] [We score it with cross-entropy, from episode sixteen.]",
            m.AnimationGroup(
                self.new_header("CROSS-ENTROPY", "The cost of one guess"),
                *[m.FadeOut(mob) for mob in self.leftovers],
                self.rows.animate.shift(shift),
                m.FadeIn(context_title, shift=0.2 * m.UP),
            ),
            m.Write(formula),
        )
        self.remove(self.rows)
        self.add(live)
        self.say(
            "[The loss is minus the log] [of the probability given to the true next token], [based on every token before it.]",
            m.AnimationGroup(m.Indicate(formula[2], color=BAD), m.FadeIn(legend[0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Indicate(m.VGroup(formula[3], formula[4], formula[7]), color=GOOD), m.FadeIn(legend[1:3], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Indicate(formula[6], color=SKY), m.FadeIn(legend[3], shift=0.15 * m.UP)),
        )
        self.say(
            "[Here, minus the log of zero point six two] [is about zero point four eight.]",
            m.FadeIn(lhs, shift=0.15 * m.UP),
            m.FadeIn(rhs, shift=0.15 * m.LEFT),
        )
        self.say(
            "[If it had been sure of one hundred, the loss would be almost zero.] [The less it expected the truth, the bigger the loss.]",
            p.animate(run_time=1.8).set_value(0.99),
            p.animate(run_time=2.4).set_value(0.05),
        )
        self.wait(0.3)
        self.stop(live, p_num, l_num)
        self.clear()

    # ------------------------------------------------------------------ every token, one pass

    def every_token(self):
        self.next_section(
            "Every token",
            notes=f"Six predictions (tokens 2-7). p = {', '.join(f'{q:.2f}' for q in P_TRUE)} (illustrative); losses "
            f"{', '.join(f'{v:.2f}' for v in LOSSES)}; mean {SENT_LOSS:.3f}. A causal Transformer scores all positions in one pass (Episode 24). "
            f"The step is pictured as +{STEP} on each true token's logit: mean loss {SENT_LOSS:.2f} → {SENT_AFTER:.2f}.",
        )
        chips = m.VGroup(*[token_chip(w) for w in SENTENCE]).arrange(m.RIGHT, buff=0.42).move_to([0.35, 1.5, 0])
        xs = [c.get_x() for c in chips]
        arcs = m.VGroup()
        for a, b in zip(chips, chips[1:]):
            arc = m.CurvedArrow(a.get_top() + np.array([0.1, 0.08, 0]), b.get_top() + np.array([-0.1, 0.08, 0]), angle=-1.3)
            arcs.add(arc.set_stroke(MUTED, width=3).set_color(MUTED))
        base_y, unit = -0.85, 1.9
        baseline = m.Line([xs[1] - 0.6, base_y, 0], [xs[-1] + 0.6, base_y, 0]).set_stroke(FAINT, width=2)

        def column(k, q):
            bar = m.Rectangle(width=0.56, height=max(unit * q, 0.02), stroke_width=0).set_fill(GOOD, opacity=0.9)
            bar.move_to([xs[k], base_y, 0], aligned_edge=m.DOWN)
            pct = code(f"{100 * q:.0f}%", 22, GOOD).move_to([xs[k], -1.25, 0])
            loss = code(f"{-np.log(q):.2f}", 26, BAD).move_to([xs[k], -1.8, 0])
            return m.VGroup(bar, pct, loss)

        cols = [column(k + 1, q) for k, q in enumerate(P_TRUE)]
        after = [column(k + 1, q) for k, q in enumerate(P_AFTER)]
        row_labels = m.VGroup(
            text("probability", 22, GOOD).move_to([xs[0], -1.25, 0]),
            text("loss", 24, BAD, weight=m.BOLD).move_to([xs[0], -1.8, 0]),
        )
        cursor = m.RoundedRectangle(corner_radius=0.2, width=chips[1][0].width + 0.2, height=chips[1][0].height + 0.2)
        cursor.set_stroke(GOOD, width=4).move_to(chips[1])
        y_ctx = chips.get_bottom()[1] - 0.12
        ctx_line = m.Line([chips[0].get_left()[0] + 0.05, y_ctx, 0], [chips[0].get_right()[0] - 0.05, y_ctx, 0]).set_stroke(SKY, width=4)
        passes = m.VGroup(
            pill("one pass · six predictions", VIOLET, VIOLET_FILL, 22),
            pill("each token sees only the past · Episode 24", SKY, SKY_FILL, 22),
        ).arrange(m.RIGHT, buff=0.4).move_to([0, -2.9, 0])

        avg = m.MathTex("L", "=", r"\frac{1}{T}", r"\sum_{t=1}^{T}", r"\ell_t", font_size=44, color=INK)
        avg[0].set_color(BAD)
        avg[4].set_color(BAD)
        t_note = text("T = 6 predictions", 22, MUTED)
        result = code(f"L ≈ {SENT_LOSS:.2f}", 32, BAD)
        bottom = m.VGroup(avg, t_note, result).arrange(m.RIGHT, buff=0.8).move_to([0.2, -2.9, 0])
        result_after = code(f"L ≈ {SENT_AFTER:.2f}", 32, BAD).move_to(result, aligned_edge=m.LEFT)

        self.say(
            "[Now slide along the whole sentence.] [Every token is a prediction of its own.]",
            m.AnimationGroup(
                self.new_header("EVERY TOKEN", "One sentence, six predictions"),
                m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.1),
                m.Create(baseline),
                m.FadeIn(row_labels),
            ),
            m.LaggedStart(*[m.Create(a) for a in arcs], lag_ratio=0.2),
        )
        self.say(
            "[Predicting boils, from water alone, is hard]: [five percent, a loss of about three.]",
            m.AnimationGroup(m.Create(cursor), m.Create(ctx_line)),
            m.AnimationGroup(m.GrowFromEdge(cols[0][0], m.DOWN), m.FadeIn(cols[0][1]), m.FadeIn(cols[0][2], scale=1.2)),
        )
        self.say(
            "[The rest come quickly], [and degrees, after one hundred, costs only about zero point one.]",
            m.AnimationGroup(
                m.FadeOut(ctx_line),
                m.FadeOut(cursor),
                m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(c[0], m.DOWN), m.FadeIn(c[1]), m.FadeIn(c[2])) for c in cols[1:]], lag_ratio=0.25),
            ),
            # cols[3][2] fades in on this same line: Indicate would capture its half-faded start and leave it dim
            m.AnimationGroup(m.Indicate(chips[4], color=GOOD), m.Circumscribe(cols[3][2], color=BAD, buff=0.08)),
        )
        self.say(
            "[A Transformer scores all six in a single pass], [because each position only looks backward.]",
            m.AnimationGroup(
                *[m.ShowPassingFlash(a.copy().set_stroke(VIOLET, width=6), time_width=0.8) for a in arcs],
                m.FadeIn(passes[0], shift=0.2 * m.UP),
                run_time=1.4,
            ),
            m.FadeIn(passes[1], shift=0.2 * m.UP),
        )
        self.say(
            "[The sentence's loss is the average]: [add up every token's loss, and divide by how many], [about zero point eight.]",
            m.AnimationGroup(m.FadeOut(passes), m.Write(avg)),
            m.AnimationGroup(m.LaggedStart(*[m.Indicate(c[2], color=BAD, scale_factor=1.25) for c in cols], lag_ratio=0.12), m.FadeIn(t_note)),
            m.FadeIn(result, scale=1.2),
        )
        self.say(
            "[Then backpropagation nudges every weight], [so each true token becomes a little more likely.]",
            m.LaggedStart(*[m.Indicate(chips[k], color=VIOLET, scale_factor=1.08) for k in range(6, -1, -1)], lag_ratio=0.12),
            m.AnimationGroup(*[m.Transform(c, a) for c, a in zip(cols, after)], m.Transform(result, result_after), run_time=1.4),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the data

    def data(self):
        self.next_section(
            "The data",
            notes="Crawl (billions of public web pages, plus books, code, papers), filter (spam, junk, personal information), deduplicate "
            "(copies get memorized, Episode 5), mix. The sixteen little documents are a cartoon. The mix is Llama 3's reported final "
            "pretraining mix: ~50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual (Dubey et al., 2024, section 3.1.2).",
        )
        steps = (
            (variant("ep08-world", TEAL, TEAL_FILL), "Crawl", "the public web"),
            ("ep27-funnel", "Filter", "spam, junk, privacy"),
            ("ep27-copies", "Deduplicate", "drop the copies"),
            ("ep27-mix", "Mix", "pick the proportions"),
        )
        nodes = m.VGroup(
            *[node(name, label, sub, TEAL, TEAL_FILL, width=2.75, height=1.85, label_size=26).move_to([x, 1.3, 0]) for (name, label, sub), x in zip(steps, (-5.1, -1.7, 1.7, 5.1))]
        )
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(nodes, nodes[1:])])
        doc_y = -0.6
        xs0, xs1, xs2 = row_xs(16), row_xs(len(KEPT1)), row_xs(len(KEPT2))
        docs = [mini_doc(DOC_LINES[i]).move_to([xs0[i], doc_y, 0]) for i in range(16)]
        at1 = {i: xs1[j] for j, i in enumerate(KEPT1)}
        at2 = {i: xs2[j] for j, i in enumerate(KEPT2)}
        one = mini_doc(DOC_LINES[0]).move_to([0, doc_y, 0])
        one_label = text("one sentence", 22, MUTED).next_to(one, m.DOWN, buff=0.18)
        twin_links = m.VGroup()
        for a, b in TWINS:
            start, end = np.array([at1[a], doc_y + 0.46, 0]), np.array([at1[b], doc_y + 0.46, 0])
            twin_links.add(m.DashedVMobject(m.ArcBetweenPoints(start, end, angle=-1.1).set_stroke(TEAL, width=3), num_dashes=14))
        mixed = {i: mini_doc(DOC_LINES[i], MIX[k][2], MIX[k][3]).move_to([at2[i], doc_y, 0]) for i, k in KIND.items()}

        left, width, bar_y = -5.5, 11.0, -2.3
        segs, labels = m.VGroup(), m.VGroup()
        x = left
        for name, frac, color, _ in MIX:
            w = width * frac
            segs.add(m.Rectangle(width=w, height=0.5, stroke_color=BG, stroke_width=3).set_fill(color, opacity=0.9).move_to([x + w / 2, bar_y, 0]))
            labels.add(m.VGroup(text(name, 22, color, weight=m.SEMIBOLD), code(f"{frac:.0%}", 22, MUTED)).arrange(m.DOWN, buff=0.08).next_to(segs[-1], m.DOWN, buff=0.15))
            x += w
        mix_title = m.VGroup(text("Llama 3's final data mix", 24, INK, weight=m.BOLD), text("Dubey et al., 2024", 20, MUTED)).arrange(m.RIGHT, buff=0.3, aligned_edge=m.DOWN)
        mix_title.next_to(segs, m.UP, buff=0.22).align_to(segs, m.LEFT)

        self.say(
            "[One sentence teaches very little.] [Pretraining needs trillions of tokens, gathered in four steps.]",
            m.AnimationGroup(self.new_header("THE DATA", "Crawl, filter, deduplicate, mix"), m.FadeIn(one, scale=0.8), m.FadeIn(one_label)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(n, shift=0.2 * m.UP), *([m.GrowArrow(links[i - 1])] if i else [])) for i, n in enumerate(nodes)], lag_ratio=0.3),
        )
        self.say(
            "[First, crawl]: [download billions of pages from the public web], [plus books, code and papers.]",
            m.Indicate(nodes[0], color=TEAL, scale_factor=1.06),
            m.AnimationGroup(m.FadeOut(one), m.FadeOut(one_label), m.LaggedStart(*[m.FadeIn(d, shift=0.3 * m.DOWN) for d in docs[:11]], lag_ratio=0.12)),
            m.LaggedStart(*[m.FadeIn(d, shift=0.3 * m.DOWN) for d in docs[11:]], lag_ratio=0.15),
        )
        self.say(
            "[Then filter]: [throw out spam, junk and personal information], [keeping text worth learning from.]",
            m.Indicate(nodes[1], color=TEAL, scale_factor=1.06),
            m.Succession(
                m.AnimationGroup(*[docs[i].animate.set_stroke(BAD) for i in JUNK], run_time=0.6),
                m.AnimationGroup(*[m.FadeOut(docs[i], shift=0.6 * m.DOWN) for i in JUNK], run_time=0.8),
            ),
            m.AnimationGroup(*[docs[i].animate.set_x(at1[i]) for i in KEPT1]),
        )
        self.say(
            "[Next, deduplicate]: [the web repeats itself], [and text seen too often gets memorized, as in episode five.]",
            m.Indicate(nodes[2], color=TEAL, scale_factor=1.06),
            m.AnimationGroup(*[m.Create(t) for t in twin_links], *[m.Indicate(docs[i], color=TEAL, scale_factor=1.15) for pair in TWINS for i in pair]),
            m.AnimationGroup(
                *[m.FadeOut(docs[c], shift=0.6 * m.DOWN) for _, c in TWINS],
                m.FadeOut(twin_links),
                *[docs[i].animate.set_x(at2[i]) for i in KEPT2],
            ),
        )
        self.say(
            "[Finally, mix]: [decide how much of each kind of text to train on.]",
            m.Indicate(nodes[3], color=TEAL, scale_factor=1.06),
            m.LaggedStart(*[m.Transform(docs[i], mixed[i]) for i in KEPT2], lag_ratio=0.12),
        )
        self.say(
            "[In Llama three, about half the final mix was general knowledge], [a quarter was math and reasoning], [and the rest was code and other languages.]",
            m.AnimationGroup(m.FadeIn(mix_title), m.GrowFromEdge(segs[0], m.LEFT), m.FadeIn(labels[0])),
            m.AnimationGroup(m.GrowFromEdge(segs[1], m.LEFT), m.FadeIn(labels[1])),
            m.AnimationGroup(m.GrowFromEdge(segs[2], m.LEFT), m.GrowFromEdge(segs[3], m.LEFT), m.FadeIn(labels[2:])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the scale

    def scale(self):
        self.next_section(
            "The scale",
            notes="Llama 3: about 15T tokens; 405B on up to 16K H100 GPUs; batches of 4M to 16M tokens per step (Dubey et al., 2024). "
            f"15e12 tokens at one per second = {YEARS:,.0f} years. 'Weeks to months' is deliberately loose.",
        )
        v = m.ValueTracker(0)
        lead = text("Llama 3 was trained on", 26, MUTED).move_to([-3.4, 1.85, 0])
        count = m.always_redraw(lambda: code(f"{int(round(v.get_value() * 1e12)):,}", 44, INK, weight=m.BOLD).move_to([-3.4, 1.15, 0]))
        unit = m.VGroup(text("tokens", 28, INK, weight=m.BOLD), text("Dubey et al., 2024", 20, MUTED)).arrange(m.RIGHT, buff=0.35, aligned_edge=m.DOWN)
        unit.move_to([-3.4, 0.45, 0])
        clock = icon(variant("clock", AMBER, AMBER_FILL), 0.95)
        years = m.VGroup(text("read aloud, one token per second", 22, MUTED), text(f"≈ {YEARS_SHOWN:,} years", 40, AMBER, weight=m.BOLD))
        years.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        years_row = m.VGroup(clock, years).arrange(m.RIGHT, buff=0.35).move_to([-3.4, -1.15, 0])

        gpus = m.VGroup(*[icon("ep27-gpu", 0.46) for _ in range(24)]).arrange_in_grid(rows=4, cols=6, buff=(0.22, 0.24)).move_to([3.45, 0.3, 0])
        gpu_title = text("thousands of GPUs", 28, AMBER, weight=m.BOLD).next_to(gpus, m.UP, buff=0.3)
        gpu_cite = text("Llama 3 405B: up to 16,000 GPUs · Dubey et al., 2024", 20, MUTED).next_to(gpus, m.DOWN, buff=0.3)
        weeks = pill("for weeks to months", AMBER, AMBER_FILL, 22).next_to(gpu_cite, m.DOWN, buff=0.3)
        step = pill("every step: millions of tokens, every weight updated together", VIOLET, VIOLET_FILL, 22).move_to([0, -3.2, 0])

        self.say(
            "[Llama three was trained on about fifteen trillion tokens.]",
            m.AnimationGroup(
                self.new_header("THE SCALE", "Trillions of tokens, thousands of GPUs"),
                m.FadeIn(lead),
                m.Succession(m.FadeIn(count, run_time=0.3), v.animate(run_time=2.3).set_value(15)),
                m.FadeIn(unit),
            ),
        )
        self.stop(count)
        self.say("[Read aloud at one token per second, that would take nearly half a million years.]", m.FadeIn(years_row, shift=0.2 * m.UP))
        self.say(
            "[So the work is split across thousands of GPUs], [running together for weeks to months.]",
            m.AnimationGroup(m.FadeIn(gpu_title), m.LaggedStart(*[m.FadeIn(g, scale=0.6) for g in gpus], lag_ratio=0.05), m.FadeIn(gpu_cite)),
            m.FadeIn(weeks, shift=0.2 * m.UP),
        )
        self.say(
            "[Each step, they read millions of tokens at once], [and update every weight together.]",
            m.AnimationGroup(m.FadeIn(step, shift=0.2 * m.UP), m.LaggedStart(*[m.Indicate(g, color=AMBER, scale_factor=1.15) for g in gpus], lag_ratio=0.04)),
            m.AnimationGroup(*[m.Indicate(g, color=VIOLET, scale_factor=1.12) for g in gpus]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the training run

    def training_run(self):
        self.next_section(
            "The training run",
            notes=f"Illustrative curve: L(D) = {E_FLOOR} + ({START_LOSS:.2f} − {E_FLOOR})·(1 + D/{D0})^−{ALPHA} plus seeded noise, D in trillions of tokens. "
            f"Start = ln {VOCAB:,} = {START_LOSS:.2f} (uniform guessing). Checkpoints every 1.5T here; a crash at {CRASH}T resumes from {RESUME}T. "
            f"Llama 3: {INTERRUPTIONS} job interruptions in a {DAYS}-day snapshot of pre-training (Dubey et al., 2024, section 3.3.4).",
        )
        ax = axes([0, 16.4, 5], [0, 12, 5], 10.6, 4.6).move_to([-0.15, -0.6, 0])

        def pt(d):
            return ax.c2p(d, run_loss(d))

        def segment(d0, d1, opacity=1.0):
            inside = RUN_D[(RUN_D > d0) & (RUN_D < d1)]
            return polyline([pt(d) for d in (d0, *inside, d1)], BAD, width=4.5, opacity=opacity)

        x_ticks = m.VGroup(*[text(str(k), 22, MUTED).next_to(ax.c2p(k, 0), m.DOWN, buff=0.2) for k in (0, 5, 10, 15)])
        y_ticks = m.VGroup(*[text(str(k), 22, MUTED).next_to(ax.c2p(0, k), m.LEFT, buff=0.2) for k in (5, 10)])
        x_label = text("tokens seen (trillions)", 22, MUTED).next_to(ax.c2p(7.5, 0), m.DOWN, buff=0.62)
        y_label = text("loss", 24, BAD, weight=m.SEMIBOLD).next_to(ax.y_axis.get_end(), m.LEFT, buff=0.2)
        tag = pill("illustrative curve", EDGE, CARD, 20).move_to([4.1, 1.55, 0])
        tag[1].set_color(MUTED)

        start = m.Dot(pt(0), radius=0.1, color=BAD).set_z_index(3)
        start_label = m.VGroup(code("−log(1/50,000) ≈ 10.8", 24, BAD), text("the cost of pure guessing", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        start_label.next_to(start, m.RIGHT, buff=0.35).shift(0.12 * m.DOWN)

        upto = m.ValueTracker(0.02)
        curve = m.always_redraw(lambda: segment(0, upto.get_value()))

        def flag(d, name="ep27-flag", height=0.42):
            f = icon(name, height)
            return f.move_to(pt(d) + np.array([-0.03, 0.02, 0]), aligned_edge=m.DL)

        before = m.VGroup(*[flag(d) for d in CHECKPOINTS if d < CRASH])
        later = m.VGroup(*[flag(d) for d in CHECKPOINTS if d > CRASH])
        legend = m.VGroup(icon("ep27-flag", 0.4), text("checkpoint: a snapshot of every weight", 22, INK)).arrange(m.RIGHT, buff=0.2).move_to([2.2, 0.45, 0])
        crash_at = pt(CRASH) + 1.15 * m.UP
        cross = xmark(crash_at)
        fail = text("hardware failure", 22, BAD, weight=m.BOLD).next_to(cross, m.RIGHT, buff=0.2)
        cite = text(f"Llama 3: {INTERRUPTIONS} interruptions in a {DAYS}-day stretch · Dubey et al., 2024", 20, MUTED).move_to([1.6, -0.15, 0])
        back = m.CurvedArrow(crash_at + np.array([-0.25, 0.0, 0]), before[-1].get_top() + np.array([0.0, 0.1, 0]), angle=0.9, tip_length=0.18)
        back.set_stroke(AMBER, width=3.5).set_color(AMBER)
        final = flag(CHECKPOINTS[-1], variant("ep27-flag", GOOD, GOOD_FILL))
        final_label = text("final checkpoint: the base model", 22, GOOD, weight=m.BOLD).next_to(final, m.UP, buff=0.25).align_to(final, m.RIGHT).shift(0.15 * m.RIGHT)

        self.say(
            "[Here's what a training run looks like.] [The curve is illustrative, but real ones share its shape.]",
            m.AnimationGroup(
                self.new_header("THE TRAINING RUN", "Watching the loss fall"),
                m.Create(ax),
                m.FadeIn(x_ticks),
                m.FadeIn(y_ticks),
                m.FadeIn(x_label),
                m.FadeIn(y_label),
            ),
            m.FadeIn(tag, shift=0.15 * m.DOWN),
        )
        self.say(
            "[It starts near ten point eight]: [minus the log of one in fifty thousand, the cost of pure guessing.]",
            m.FadeIn(start, scale=0.4),
            m.FadeIn(start_label, shift=0.15 * m.RIGHT),
        )
        self.add(curve)
        self.say(
            "[The loss drops fast as common words and grammar click into place], [then creeps down for weeks.]",
            upto.animate(rate_func=m.linear).set_value(1.5),
            upto.animate(run_time=3.0, rate_func=m.linear).set_value(CRASH),
        )
        self.say(
            "[Along the way, the run saves checkpoints]: [snapshots of every weight.]",
            m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.DOWN) for f in before], lag_ratio=0.2),
            m.FadeIn(legend, shift=0.15 * m.UP),
        )
        self.say(
            "[With thousands of GPUs, something breaks all the time]: "
            "[the Llama three team reported four hundred and sixty-six interruptions in fifty-four days.]",
            m.AnimationGroup(m.Create(cross), m.FadeIn(fail, shift=0.15 * m.DOWN), m.Flash(pt(CRASH), color=BAD, flash_radius=0.35)),
            m.FadeIn(cite, shift=0.15 * m.UP),
        )
        self.say(
            "[So training resumes from the last checkpoint], [not from scratch.]",
            m.AnimationGroup(
                m.Create(back),
                upto.animate(run_time=1.2).set_value(RESUME),
                m.Indicate(before[-1], color=AMBER, scale_factor=1.3),
            ),
            m.AnimationGroup(
                upto.animate(run_time=2.6, rate_func=m.linear).set_value(15),
                m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.DOWN) for f in later], lag_ratio=0.9),
            ),
        )
        self.stop(curve)
        self.say(
            "[And the final checkpoint is the finished base model.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (cross, fail, back, cite)],
                m.FadeTransform(later[-1], final),
                m.FadeIn(final_label, shift=0.15 * m.DOWN),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ what comes out

    def result(self):
        self.next_section(
            "The result",
            notes="A base model continues text; it does not follow instructions. The continuation shown is illustrative. "
            "Instruction tuning: Episode 29.",
        )
        model = icon("llm", 1.8).move_to([-4.55, 0.9, 0])
        name = text("base model", 30, VIOLET, weight=m.BOLD).next_to(model, m.DOWN, buff=0.3)
        sub = text("a text predictor", 22, MUTED).next_to(name, m.DOWN, buff=0.12)
        window = card(7.7, 4.1, EDGE, CARD).move_to([2.4, 0.15, 0])
        question = bubble(text("What is the boiling point of water?", 26, INK), SKY, SKY_FILL, tail="left")
        question.align_to(window, m.UL).shift(np.array([0.35, -0.3, 0]))
        continues = text("the base model continues:", 20, MUTED)
        lines = m.VGroup(
            *[text(s, 26, VIOLET) for s in ("What is the freezing point of water?", "What is the density of ice?", "What is the chemical formula for salt?")]
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        m.VGroup(continues, lines).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).next_to(question, m.DOWN, buff=0.25).align_to(question, m.LEFT)
        verdict = pill("it continues the text; it doesn't answer", BAD, BAD_FILL, 22).next_to(window, m.DOWN, buff=0.3)
        learned = pill("learned: facts, grammar, style", GOOD, GOOD_FILL, 22).next_to(sub, m.DOWN, buff=0.4)
        upcoming = pill("next: instruction tuning makes it an assistant · Episode 29", TEAL, TEAL_FILL, 22).move_to([0, -3.35, 0])

        self.say(
            "[What comes out is a superb text predictor, but not yet an assistant.]",
            m.AnimationGroup(
                self.new_header("THE RESULT", "A text predictor, not an assistant"),
                m.FadeIn(model, scale=0.8),
                m.FadeIn(name, shift=0.15 * m.UP),
                m.FadeIn(sub, shift=0.15 * m.UP),
            ),
        )
        self.say(
            "[Ask it a question], [and it may simply continue the text, with more questions, for example.]",
            m.AnimationGroup(m.FadeIn(window), m.FadeIn(question, shift=0.2 * m.UP)),
            m.Succession(
                m.FadeIn(continues, run_time=0.3),
                *[m.AddTextLetterByLetter(line, time_per_char=0.03) for line in lines],
                m.FadeIn(verdict, shift=0.2 * m.UP, run_time=0.5),
            ),
        )
        self.say(
            "[It has soaked up facts, grammar and style], [but not how to help. Turning it into an assistant is episode twenty-nine.]",
            m.FadeIn(learned, shift=0.15 * m.UP),
            m.FadeIn(upcoming, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Predict, score, repeat.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Predict", "Score", "Repeat"), (VIOLET, BAD, AMBER))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A chatbot begins as one simple guess, made trillions of times.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [predict], [score], [repeat.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A chatbot begins as one simple guess, made trillions of times.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
