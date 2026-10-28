"""AI, Explained · Episode 31 · Prompting & In-Context Learning

Running example: a sentiment classifier for product reviews built from a prompt, with zero weights
changed. The review to label all film long: "Great, it broke after two days." (negative, despite
"Great").

- Hook: the classic way (train on labelled reviews, every weight nudged) versus the new way (write a
  prompt, hand it to a frozen language model): in-context learning.
- Zero-shot: instruction + review; a large model often gets it right, but the reply's shape varies.
- Few-shot (Brown et al., 2020, the GPT-3 paper): three solved examples, one of them sarcastic; the
  model continues the pattern with one word from our label set.
- In context: same network, same weights; the examples change the activations and the next-word
  odds. Pretraining text is full of repeating patterns; the "lesson" lasts only as long as the prompt.
- Chain-of-thought (Wei et al., 2022; "let's think step by step": Kojima et al., 2022): a two-step
  word problem about our reviews, 23 − 20 = 3, 3 + 6 = 9 (a hurried answer: 29).
- Prompt structure: role, task, context, examples, output format (+ the input). Callback Ep 32 (RAG
  fills the context slot).
- The catch: wording and example order matter (Zhao et al., 2021; Lu et al., 2022), shown with a toy
  simulation; the context window is finite. Callbacks Ep 6 (measure it), Ep 29 (fine-tuning), Ep 43.

Colours, all film long: instruction/task TEAL (also the level accent of the headers), examples AMBER,
the review (input) SKY, the model and its reasoning VIOLET, context BLUE, output format PINK;
"Positive" GOOD and "Negative" BAD (labels), and ✓/✗ verdicts GOOD/BAD.

Illustrative, computed below with seeded numpy and asserted against the narration: the next-word odds
(softmax of made-up logits) and the order-sensitivity chart (a toy model with a recency bias).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

BLUE_FILL = "#121F3D"  # a dark tint of the palette's BLUE, as the other accents have
TASK = (TEAL, TEAL_FILL)
EXAMPLE = (AMBER, AMBER_FILL)
INPUT = (SKY, SKY_FILL)
MODEL = (VIOLET, VIOLET_FILL)
CONTEXT = (BLUE, BLUE_FILL)
FORMAT = (PINK, PINK_FILL)
LABEL_COLOR = {"Positive": GOOD, "Negative": BAD}

INSTRUCTION = "Is this review positive or negative?"
REVIEW = "Great, it broke after two days."
SHOTS = (
    ("Arrived fast, works perfectly.", "Positive"),
    ("Great, another broken cable.", "Negative"),  # the sarcastic one
    ("Love it, I use it every day.", "Positive"),
)

# ---------------------------------------------------------------------------- the numbers


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


# Next-word odds right after the prompt (illustrative logits, not measured on a real model).
CANDIDATES = ("Negative", "Positive", "The", "This", "other words")
P_ZERO = softmax(np.array([2.0, 0.2, 1.9, 1.4, 1.7]))  # instruction + review only
P_FEW = softmax(np.array([5.0, 1.8, 0.0, -0.4, 0.6]))  # with the three examples
assert round(100 * P_FEW[0]) == 94  # "ninety-four percent goes to Negative"
assert P_ZERO.max() < 0.35  # "spread out" without examples


# Order sensitivity, a toy model (illustrative): 200 test reviews whose evidence is the true label
# plus noise; the prompt adds a bias toward the labels of its examples, weighted by recency (the last
# example counts most, as Zhao et al., 2021 found) and by each example's own strength.
def order_accuracies(seed=31, n=200):
    rng = np.random.default_rng(seed)
    truth = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    evidence = truth + rng.normal(0, 0.8, n)
    labels = np.array([1, -1, 1, -1])  # examples A+, B− (sarcastic), C+, D−
    strength = np.array([1.0, 1.4, 0.7, 0.9])
    recency = np.array([0.1, 0.25, 0.5, 1.0])
    accs = []
    for order in itertools.permutations(range(4)):
        bias = sum(recency[p] * labels[k] * strength[k] for p, k in enumerate(order))
        accs.append(np.mean(np.sign(evidence + bias) == truth))
    return np.sort(np.array(accs))


ORDER_ACC = order_accuracies()
assert len(ORDER_ACC) == 24  # "all twenty-four orders" of four examples
LOW, HIGH = round(100 * ORDER_ACC[0]), round(100 * ORDER_ACC[-1])
assert (LOW, HIGH) == (58, 88)  # "from fifty-eight to eighty-eight percent"

# Chain-of-thought word problem
HAD, POSITIVE, NEW_NEGATIVE = 23, 20, 6
STEP1 = HAD - POSITIVE
STEP2 = STEP1 + NEW_NEGATIVE
HURRIED = HAD + NEW_NEGATIVE
assert (STEP1, STEP2, HURRIED) == (3, 9, 29)  # "three", "nine", "twenty-nine"

# The weights the hook's "classic way" nudges (illustrative)
_wr = np.random.default_rng(31)
W0 = _wr.normal(0, 0.6, (3, 3))
W1 = W0 + _wr.normal(0, 0.15, (3, 3))
W2 = W1 + _wr.normal(0, 0.15, (3, 3))

# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """kit.recolored(), written under this episode's prefix: assets/ep31-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    stem = f"ep31-{name}-{color.lstrip('#').lower()}"
    path = ASSETS / f"{stem}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return stem


def sit(words, baseline):
    """Put a Text's letters on a baseline (descenders and punctuation hang below it)."""
    source = words.text.replace(" ", "").replace("\n", "")
    sitting = [g for g, c in zip(words, source) if c.isalnum() and c not in "gjpqyQ"]
    if sitting:
        words.shift((baseline - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    return words


def line(s, color, left, baseline, size=22, t2c=None):
    """One left-aligned line of code on a baseline."""
    words = code(s, size, color, t2c=t2c or {})
    words.move_to([0, 0, 0]).align_to([left, 0, 0], m.LEFT)
    return sit(words, baseline).set_z_index(1)


def shot_line(review, label, left, baseline, size=22):
    return line(f"{review} → {label}", AMBER, left, baseline, size, t2c={"→": MUTED, label: LABEL_COLOR[label]})


def input_line(left, baseline, size=22):
    return line(f"{REVIEW} →", SKY, left, baseline, size, t2c={"→": MUTED})


def window(x, top, width, height, name="prompt.txt"):
    """A code window: a card with traffic lights, a file name and a rule; content goes below the rule."""
    box = card(width, height, EDGE, CARD, radius=0.22, stroke=2).move_to([x, top - height / 2, 0])
    lights = m.VGroup(*[m.Dot(radius=0.07, color=c) for c in (BAD, AMBER, GOOD)]).arrange(m.RIGHT, buff=0.13)
    lights.move_to([x - width / 2 + 0.3 + lights.width / 2, top - 0.29, 0])
    label = code(name, 20, MUTED).next_to(lights, m.RIGHT, buff=0.3)
    rule = m.Line([x - width / 2, top - 0.58, 0], [x + width / 2, top - 0.58, 0]).set_stroke(EDGE, width=2)
    return m.VGroup(box, lights, label, rule).set_z_index(-1)


def typed(words, rate=0.022):
    return m.AddTextLetterByLetter(words, time_per_char=rate)


def code_card(lines, color=EDGE, fill=CARD, size=20, pad=0.25, step=0.42, width=None):
    """A card of left-aligned code lines (a reply, a check)."""
    texts = [code(s, size, c) for s, c in lines]
    inner = max(t.width for t in texts)
    w = width or inner + 2 * pad + 0.1
    h = step * (len(texts) - 1) + 0.32 + 2 * pad
    box = card(w, h, color, fill, radius=0.18, stroke=2)
    top = box.get_top()[1] - pad - 0.26
    for k, t in enumerate(texts):
        t.move_to([0, 0, 0]).align_to(box.get_left() + (pad + 0.05) * m.RIGHT, m.LEFT)
        sit(t, top - k * step)
    return m.VGroup(box, *texts)


def panel(x, title, color, width=6.3, height=4.75, y=-1.05):
    box = card(width, height, color, CARD, stroke=2).move_to([x, y, 0]).set_z_index(-1)
    head = text(title, 24, color, weight=m.BOLD)
    head.move_to(box.get_corner(m.UL) + np.array([0.35 + head.width / 2, -0.42, 0]))
    return m.VGroup(box, head)


def level_pill(label, color, fill, size=22):
    """kit.pill(), with the letters centred on their cap height (descenders hang below)."""
    group = pill(label, color, fill, size)
    box, words = group
    source = words.text.replace(" ", "")
    sitting = [g for g, c in zip(words, source) if c.isalnum() and c not in "gjpqyQ"]
    if sitting:
        low = min(g.get_bottom()[1] for g in sitting)
        high = max(g.get_top()[1] for g in sitting)
        words.shift((box.get_y() - (low + high) / 2) * m.UP)
    return group


def label_pill(label, size=22):
    color = LABEL_COLOR[label]
    return level_pill(label, color, GOOD_FILL if color == GOOD else BAD_FILL, size)


def verdict(ok, size=22):
    return level_pill("✓ passes" if ok else "✗ fails", GOOD if ok else BAD, GOOD_FILL if ok else BAD_FILL, size)


def prompt_glyph():
    """A tiny prompt: instruction (teal), three examples (amber), the review (sky)."""
    box = card(1.45, 1.75, EDGE, CARD, radius=0.15, stroke=2)
    bars = m.VGroup()
    for k, (w, c) in enumerate(((1.0, TEAL), (0.92, AMBER), (0.8, AMBER), (0.95, AMBER), (0.85, SKY))):
        bar = m.RoundedRectangle(corner_radius=0.05, width=w, height=0.13, stroke_width=0).set_fill(c, opacity=0.95)
        bar.move_to(box.get_corner(m.UL) + np.array([0.22 + w / 2, -0.32 - 0.28 * k, 0]))
        bars.add(bar)
    return m.VGroup(box, bars)


def block(color, fill, width=1.5, height=0.36):
    return m.RoundedRectangle(corner_radius=0.08, width=width, height=height, stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1)


def network(xs=(-3.4, -2.3, -1.2, -0.1), counts=(4, 5, 5, 4), dy=0.62):
    layers = m.VGroup()
    for x, n in zip(xs, counts):
        layers.add(m.VGroup(*[m.Circle(radius=0.14).set_stroke(MUTED, width=2).set_fill(CARD, opacity=1).move_to([x, (i - (n - 1) / 2) * dy, 0]) for i in range(n)]))
    edges = m.VGroup()
    for a, b in zip(layers, layers[1:]):
        edges.add(m.VGroup(*[m.Line(p.get_center(), q.get_center(), buff=0.14).set_stroke(FAINT, width=1.5) for p in a for q in b]))
    layers.set_z_index(1)
    return edges, layers


def flow(edges, color):
    return m.LaggedStart(
        *[m.AnimationGroup(*[m.ShowPassingFlash(e.copy().set_stroke(color, width=4), time_width=0.6) for e in group]) for group in edges],
        lag_ratio=0.6,
        run_time=1.4,
    )


def activate(layers, color, seed):
    rng = np.random.default_rng(seed)
    return m.LaggedStart(
        *[m.AnimationGroup(*[n.animate.set_fill(color, opacity=float(rng.uniform(0.15, 0.95))) for n in layer]) for layer in layers],
        lag_ratio=0.5,
        run_time=1.4,
    )


def swap(old, new):
    """old fades out, new fades in. Unlike FadeTransform, it keeps `new` hidden until its bracket
    starts (FadeTransform in a later bracket shows its target from the start of the line)."""
    return m.AnimationGroup(m.FadeOut(old), m.FadeIn(new))


def pct(p):
    return "<1%" if p < 0.005 else f"{round(100 * p)}%"


def odds(probs, x_label=2.35, x_bar=2.5, y0=1.0, scale=3.3):
    """Next-word odds: word · bar · percent, one row per candidate."""
    rows = m.VGroup()
    for k, (name, p) in enumerate(zip(CANDIDATES, probs)):
        y = y0 - 0.5 * k
        color = LABEL_COLOR.get(name, MUTED)
        word = text(name, 22, INK if name in LABEL_COLOR else MUTED, weight=m.SEMIBOLD)
        word.move_to([0, 0, 0]).align_to([x_label, 0, 0], m.RIGHT)
        sit(word, y - 0.09)
        bar = m.Rectangle(width=max(scale * p, 0.04), height=0.28, stroke_width=0).set_fill(color, opacity=0.9)
        bar.move_to([x_bar, y, 0], aligned_edge=m.LEFT)
        share = text(pct(p), 20, MUTED).next_to(bar, m.RIGHT, buff=0.15)
        sit(share, y - 0.08)
        rows.add(m.VGroup(word, bar, share))
    return rows


def bottom(label, color, fill, size=22, y=-3.25, x=0.0):
    return pill(label, color, fill, size).move_to([x, y, 0])


def cite(s):
    return text(s, 20, MUTED)


# ---------------------------------------------------------------------------- the episode

WX, WW, TOP = -2.45, 8.4, 2.45  # the prompt window: centre x, width, top edge
LEFT = WX - WW / 2 + 0.42  # where its lines start
BASE0, STEP = TOP - 0.58 - 0.5, 0.48  # first baseline, line spacing
RX = 4.45  # the right column (model, reply)
CHAR22 = 0.1774  # Fira Code advance per character at font size 22 (measured)


def window_height(n_lines):
    return 0.58 + 0.5 + (n_lines - 1) * STEP + 0.45


class Ep31PromptingICL(Episode):
    number = 31

    def lesson(self):
        self.hook()
        self.zero_shot()
        self.few_shot()
        self.in_context()
        self.chain_of_thought()
        self.anatomy()
        self.brittle()
        self.finite()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A review classifier: train it (every weight nudged) or prompt it (no weight changes). In-context learning. Facts as of October 2026; the weights shown are illustrative.")
        review = text(f"“{REVIEW}”", 26, INK)
        review_box = card(review.width + 0.6, 0.85, SKY, SKY_FILL, radius=0.2)
        review_card = m.VGroup(review_box, review.move_to(review_box))
        choices = m.VGroup(label_pill("Positive"), label_pill("Negative")).arrange(m.RIGHT, buff=0.25)
        top_row = m.VGroup(review_card, choices).arrange(m.RIGHT, buff=1.0).move_to([0, 2.35, 0])
        top_arrow = arrow(review_card.get_right(), choices.get_left(), MUTED, buff=0.12)

        left = panel(-3.45, "The classic way · train it", MUTED)
        left[0].set_stroke(EDGE)
        right = panel(3.45, "The new way · prompt it", TEAL)
        right_title = text("In-context learning", 24, TEAL, weight=m.BOLD).move_to(right[1]).align_to(right[1], m.LEFT)

        data = icon(variant("database", MUTED, CARD), 1.05).move_to([-5.4, -0.75, 0])
        data_note = m.VGroup(text("thousands of", 20, MUTED), text("labelled reviews", 20, MUTED)).arrange(m.DOWN, buff=0.08).next_to(data, m.DOWN, buff=0.2)

        def grid(w):
            return m.VGroup(*[code(f"{v:+.2f}".replace("-", "−"), 20, INK) for v in w.flat]).arrange_in_grid(rows=3, buff=(0.32, 0.24))

        weights = grid(W0)
        frame = card(weights.width + 0.5, weights.height + 0.45, EDGE, CARD, radius=0.15, stroke=2)
        weight_box = m.VGroup(frame, weights.move_to(frame)).move_to([-2.25, -0.75, 0])
        weights_title = text("weights", 20, MUTED).next_to(frame, m.UP, buff=0.15)
        step1, step2 = grid(W1).move_to(weights), grid(W2).move_to(weights)
        to_weights = arrow([-4.75, -0.75, 0], frame.get_left(), MUTED, buff=0.1)
        trained = pill("training changes every weight", MUTED, CARD, 20).move_to([-3.45, -2.75, 0])

        glyph = prompt_glyph().move_to([1.3, -0.75, 0])
        model = icon("llm", 1.15).move_to([3.35, -0.75, 0])
        to_model = arrow(glyph.get_right(), model.get_left(), MUTED, buff=0.12)
        answer = label_pill("Negative").move_to([5.45, -0.75, 0])
        to_answer = arrow(model.get_right(), answer.get_left(), MUTED, buff=0.12)
        lock = icon(variant("lock", MUTED, CARD), 0.5)
        unchanged = text("weights changed: 0", 24, INK, weight=m.SEMIBOLD, t2c={"0": TEAL})
        frozen = m.VGroup(lock, unchanged).arrange(m.RIGHT, buff=0.25).move_to([3.45, -2.75, 0])

        self.say(
            "[Say you need a program that reads product reviews], [and labels each one positive or negative.]",
            m.FadeIn(review_card, shift=0.2 * m.UP),
            m.Succession(m.AnimationGroup(m.GrowArrow(top_arrow), m.FadeIn(choices, shift=0.2 * m.RIGHT)), m.Indicate(choices[1], color=BAD, scale_factor=1.12)),
        )
        self.say(
            "[The classic way is to train a model]: [collect thousands of labelled reviews], [then nudge its weights, step by step, until it gets them right.]",
            m.FadeIn(left),
            m.AnimationGroup(m.FadeIn(data, shift=0.2 * m.UP), m.FadeIn(data_note)),
            m.Succession(
                m.AnimationGroup(m.GrowArrow(to_weights), m.FadeIn(weight_box), m.FadeIn(weights_title)),
                m.AnimationGroup(*[m.Transform(a, b) for a, b in zip(weights, step1)], run_time=0.7),
                m.AnimationGroup(*[m.Transform(a, b) for a, b in zip(weights, step2)], run_time=0.7),
                m.FadeIn(trained, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[Today, there's another way]: [write a short prompt], [and hand it to a large language model.]",
            m.FadeIn(right),
            m.FadeIn(glyph, shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(to_model), m.FadeIn(model, scale=0.8)),
        )
        self.say(
            "[It labels the review], [and not a single weight changes.]",
            m.AnimationGroup(m.GrowArrow(to_answer), m.FadeIn(answer, shift=0.2 * m.RIGHT)),
            m.FadeIn(frozen, shift=0.2 * m.UP),
        )
        self.say(
            "[This is in-context learning]: [the model picks up the task from the prompt itself.]",
            m.AnimationGroup(swap(right[1], right_title), m.Circumscribe(right[0], color=TEAL, buff=0.06)),
            m.Succession(pulse(to_model, AMBER), pulse(to_answer, BAD)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ zero-shot

    def zero_shot(self):
        self.next_section("1 · Zero-shot", notes="Only an instruction and the review. A large model often answers correctly, but in a free form.")
        self.chrome = window(WX, TOP, WW, window_height(2))
        self.instruction = line(INSTRUCTION, TEAL, LEFT, BASE0)
        self.query = input_line(LEFT, BASE0 - STEP)
        self.model = icon("llm", 1.0).move_to([RX, 1.85, 0])
        self.feed = arrow([WX + WW / 2, 1.85, 0], self.model.get_left(), MUTED, buff=0.12)
        self.reply = code_card(
            [("The review is negative:", INK), ("it opens with “Great”,", INK), ("but it broke in days.", INK)], VIOLET, VIOLET_FILL, width=4.3
        ).move_to([RX, 0.2, 0])
        hedges = m.VGroup(
            code_card([("Mostly negative?", INK)], EDGE, CARD, width=4.3),
            code_card([("Bad.", INK)], EDGE, CARD, width=4.3),
        ).arrange(m.DOWN, buff=0.18)
        hedges.next_to(self.reply, m.DOWN, buff=0.18)
        self.hedges = hedges
        check = code_card([('assert label in ("Positive", "Negative")', INK)], EDGE, CARD)
        check.move_to([0, -1.85, 0]).align_to([WX - WW / 2, 0, 0], m.LEFT)
        self.check = check
        self.fails = verdict(False).next_to(check, m.RIGHT, buff=0.3)

        self.say(
            "[First, the simplest prompt]: zero-shot, with no examples at all.",
            self.new_header("1 · ZERO-SHOT", "Just describe the task"),
        )
        self.say(
            "[We write the instruction], [then the review], and leave the answer to the model.",
            m.Succession(m.FadeIn(self.chrome, scale=0.97), typed(self.instruction)),
            typed(self.query),
        )
        self.say(
            "[A large model often gets it right], [but its reply can take any shape]: [a sentence, a hedge, or another word.]",
            m.Succession(m.AnimationGroup(m.GrowArrow(self.feed), m.FadeIn(self.model, scale=0.8)), m.FadeIn(self.reply[0]), *[typed(t, 0.018) for t in self.reply[1:]]),
            m.Circumscribe(self.reply, color=VIOLET, buff=0.06),
            m.LaggedStart(*[m.FadeIn(h, shift=0.15 * m.UP) for h in hedges], lag_ratio=0.4),
        )
        self.say(
            "[That's awkward when your code expects exactly one of two labels.]",
            m.Succession(m.FadeIn(check, shift=0.2 * m.UP), m.FadeIn(self.fails, scale=1.2)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ few-shot

    def few_shot(self):
        self.next_section(
            "2 · Few-shot",
            notes="Three solved examples (one sarcastic) before the review. Zero-, one- and few-shot as defined in the GPT-3 paper: Brown et al., 2020, 'Language Models are Few-Shot Learners'.",
        )
        big = window(WX, TOP, WW, window_height(5))
        shots = m.VGroup(*[shot_line(r, lab, LEFT, BASE0 - (k + 1) * STEP) for k, (r, lab) in enumerate(SHOTS)])
        query_to = BASE0 - 4 * STEP
        one_word = code_card([("Negative", BAD)], VIOLET, VIOLET_FILL, size=28, width=4.3).move_to([RX, 0.55, 0])
        note = text("one word · one of our labels", 20, MUTED).next_to(one_word, m.DOWN, buff=0.2)
        passes = verdict(True).move_to(self.fails)
        source = cite("GPT-3 · Brown et al., 2020")
        kinds = m.VGroup(
            *[
                level_pill(f"{n}  ·  {k}", AMBER, AMBER_FILL, 20)
                for n, k in (("0 examples", "zero-shot"), ("1 example", "one-shot"), ("a few", "few-shot"))
            ]
        ).arrange(m.RIGHT, buff=0.25).move_to([0, -2.75, 0])
        source.next_to(kinds, m.DOWN, buff=0.2).align_to(kinds, m.RIGHT)

        self.say(
            "[Now, show instead of tell]: [few-shot prompting adds a few solved examples.]",
            m.AnimationGroup(self.new_header("2 · FEW-SHOT", "Show, don't tell"), m.FadeOut(self.hedges)),
            m.AnimationGroup(
                m.ReplacementTransform(self.chrome, big),
                self.query.animate.shift((query_to - (BASE0 - STEP)) * m.UP),
                m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.RIGHT) for s in shots], lag_ratio=0.3),
                run_time=1.6,
            ),
        )
        self.chrome = big
        self.say(
            "[Each one pairs a review with its label], [including a sarcastic one.]",
            m.LaggedStart(*[m.Indicate(glyphs(s, lab), color=LABEL_COLOR[lab], scale_factor=1.15) for s, (_, lab) in zip(shots, SHOTS)], lag_ratio=0.3),
            m.Circumscribe(shots[1], color=AMBER, buff=0.08),
        )
        self.say(
            "[The model simply continues the pattern]: [Negative.] [One word, and one of our labels.]",
            m.AnimationGroup(pulse(self.feed, AMBER), m.FadeOut(self.reply)),
            m.FadeIn(one_word, scale=1.1),
            m.AnimationGroup(m.FadeIn(note, shift=0.1 * m.UP), swap(self.fails, passes)),
        )
        self.say(
            "[A landmark paper in twenty twenty showed that large models can learn tasks this way], [from zero, one, or a few examples in the prompt.]",
            m.FadeIn(source, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(k, shift=0.15 * m.UP) for k in kinds], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ in context

    def in_context(self):
        self.next_section(
            "3 · In context",
            notes=(
                "Weights frozen; the examples change the activations and the next-word odds. Odds are illustrative: softmax of made-up logits, "
                f"zero-shot {', '.join(f'{c} {pct(p)}' for c, p in zip(CANDIDATES, P_ZERO))}; few-shot {', '.join(f'{c} {pct(p)}' for c, p in zip(CANDIDATES, P_FEW))}. "
                "Pattern cards: the GPT-3 paper's translation-style prompts (Brown et al., 2020)."
            ),
        )
        edges, layers = network()
        lock = icon(variant("lock", MUTED, CARD), 0.5)
        frozen = m.VGroup(lock, text("weights frozen", 22, MUTED, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2).move_to([-1.75, 2.1, 0])

        sx = -5.55
        teal, sky = block(*TASK), block(*INPUT)
        teal.move_to([sx, 0.225, 0])
        sky.move_to([sx, -0.225, 0])
        ambers = m.VGroup(*[block(*EXAMPLE).move_to([sx, 0.45 - 0.45 * k, 0]) for k in range(3)])
        stack_label = text("prompt", 20, MUTED).move_to([sx, -1.5, 0])
        into = arrow([sx + 0.85, 0, 0], [-3.6, 0, 0], MUTED, buff=0.05)
        out = arrow([0.12, 0, 0], [0.85, 0, 0], MUTED, buff=0.05)
        title = text("next-word odds", 22, MUTED, weight=m.SEMIBOLD).move_to([3.9, 1.65, 0])
        before = odds(P_ZERO)
        after = odds(P_FEW)
        reset = odds(P_ZERO)

        knows = bottom("the examples change the activations, not the weights", AMBER, AMBER_FILL)
        patterns = m.VGroup()
        fills = m.VGroup()
        for s, answer in (("cat → chat, dog →", "chien"), ("Mon, Tue, Wed,", "Thu"), ("2 → 4, 3 → 6, 5 →", "10")):
            box = card((len(s) + 1 + len(answer)) * CHAR22 + 0.6, 0.72, EDGE, CARD, radius=0.15, stroke=2)
            words = code(s, 22, INK, t2c={"→": MUTED})
            words.move_to(box).align_to(box.get_left() + 0.3 * m.RIGHT, m.LEFT)
            sit(words, box.get_y() - 0.1)
            mark = code("?", 22, AMBER).move_to(box)
            mark.align_to([words.get_left()[0] + (len(s) + 1) * CHAR22, 0, 0], m.LEFT)
            sit(mark, box.get_y() - 0.1)
            fill = code(answer, 22, AMBER).move_to(mark).align_to(mark, m.LEFT)
            sit(fill, box.get_y() - 0.1)
            patterns.add(m.VGroup(box, words, mark))
            fills.add(m.VGroup(mark, fill))
        patterns.arrange(m.RIGHT, buff=0.25).move_to([0, -2.75, 0])
        for (box, words, mark), (_, fill) in zip(patterns, fills):
            fill.move_to(mark).align_to(mark, m.LEFT)
            sit(fill, box.get_y() - 0.1)
        pretrain = text("seen in pretraining: patterns, everywhere", 20, MUTED).next_to(patterns, m.UP, buff=0.18)
        temporary = bottom("it lasts only as long as the prompt", TEAL, TEAL_FILL)

        def morph(src, dst):
            return m.AnimationGroup(*[m.Transform(a[1], b[1]) for a, b in zip(src, dst)], *[swap(a[2], b[2]) for a, b in zip(src, dst)])

        self.say(
            "[So where did the learning happen?] [Not in the weights]: [they stay frozen.]",
            m.AnimationGroup(self.new_header("3 · IN CONTEXT", "Learning without training"), m.Create(edges, run_time=1.2), m.FadeIn(layers)),
            m.FadeIn(frozen, shift=0.15 * m.DOWN),
            m.Circumscribe(edges, color=MUTED, buff=0.1),
        )
        self.say(
            "[Without examples, the prompt flows through the network], [and the next-word odds are spread out.]",
            m.Succession(m.AnimationGroup(m.FadeIn(teal, shift=0.2 * m.RIGHT), m.FadeIn(sky, shift=0.2 * m.RIGHT), m.FadeIn(stack_label), m.GrowArrow(into)), m.AnimationGroup(flow(edges, SKY), activate(layers, SKY, 1))),
            m.AnimationGroup(m.GrowArrow(out), m.FadeIn(title), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in before], lag_ratio=0.15)),
        )
        # after the morph, the shares on screen are after[k][2]; the bars stay before[k][1]
        self.say(
            "[Add the examples]: [same network, same weights], [but now, for example, ninety-four percent goes to Negative.]",
            m.AnimationGroup(teal.animate.move_to([sx, 0.9, 0]), sky.animate.move_to([sx, -0.9, 0]), m.LaggedStart(*[m.FadeIn(a, scale=0.8) for a in ambers], lag_ratio=0.25)),
            m.AnimationGroup(flow(edges, AMBER), activate(layers, AMBER, 2), m.Circumscribe(frozen, color=MUTED, buff=0.08)),
            morph(before, after),
        )
        self.say(
            "[The examples change the activations, not the weights.] [That's why it's called in-context learning.]",
            m.AnimationGroup(m.FadeIn(knows, shift=0.2 * m.UP), m.LaggedStart(*[m.Indicate(n, color=AMBER, scale_factor=1.3) for layer in layers for n in layer], lag_ratio=0.03)),
            m.Circumscribe(m.VGroup(teal, ambers, sky), color=AMBER, buff=0.12),
        )
        self.say(
            "Why does it work at all? [In pretraining, the model read countless documents full of repeating patterns], [and continuing a pattern is just predicting the next word.]",
            m.AnimationGroup(m.FadeOut(knows), m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in patterns], lag_ratio=0.25), m.FadeIn(pretrain)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeOut(mark), m.FadeIn(fill, shift=0.1 * m.LEFT)) for mark, fill in fills], lag_ratio=0.35),
        )
        self.say(
            "[And it lasts only as long as the prompt]: [start a new chat, and the lesson is gone.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(*[m.VGroup(p[0], p[1]) for p in patterns])), m.FadeOut(m.VGroup(*[f[1] for f in fills])), m.FadeOut(pretrain), m.FadeIn(temporary, shift=0.2 * m.UP)),
            m.AnimationGroup(
                m.FadeOut(ambers, scale=0.8),
                teal.animate.move_to([sx, 0.225, 0]),
                sky.animate.move_to([sx, -0.225, 0]),
                *[m.Transform(a[1], b[1]) for a, b in zip(before, reset)],
                *[swap(a[2], b[2]) for a, b in zip(after, reset)],
                *[n.animate.set_fill(CARD, opacity=1) for layer in layers for n in layer],
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ chain-of-thought

    def chain_of_thought(self):
        self.next_section(
            "4 · Chain-of-thought",
            notes=(
                f"Word problem: {HAD} reviews, {POSITIVE} positive, then {NEW_NEGATIVE} new negative ones: {HAD} − {POSITIVE} = {STEP1}, "
                f"{STEP1} + {NEW_NEGATIVE} = {STEP2}; the hurried answer {HURRIED} ({HAD} + {NEW_NEGATIVE}) is illustrative. "
                "Chain-of-thought prompting with worked examples: Wei et al., 2022 (gains on multi-step arithmetic, mostly in large models). "
                "'Let's think step by step': Kojima et al., 2022."
            ),
        )
        chrome = window(WX, TOP, WW, window_height(4))
        indent = LEFT + 3 * CHAR22  # under the text after "Q: "
        problem = m.VGroup(
            line(f"Q: We had {HAD} reviews; {POSITIVE} were positive.", SKY, LEFT, BASE0),
            line(f"Then {NEW_NEGATIVE} negative reviews came in.", SKY, indent, BASE0 - STEP),
            line("How many are negative now?", SKY, indent, BASE0 - 2 * STEP),
        )
        a_mark = line("A:", MUTED, LEFT, BASE0 - 3 * STEP)
        think = line("Let's think step by step.", TEAL, indent, BASE0 - 3 * STEP)
        model = icon("llm", 1.0).move_to([RX, 1.85, 0])
        feed = arrow([WX + WW / 2, 1.85, 0], model.get_left(), MUTED, buff=0.12)
        hurried = code_card([(f"A: {HURRIED}", INK)], VIOLET, VIOLET_FILL, size=28, width=4.3).move_to([RX, 0.55, 0])
        wrong = level_pill("✗ wrong", BAD, BAD_FILL, 22).next_to(hurried, m.DOWN, buff=0.25)

        box = card(4.2, 2.1, VIOLET, VIOLET_FILL, radius=0.18, stroke=2).move_to([RX, 0.2, 0])
        x_text = box.get_left()[0] + 1.0
        ys = [box.get_top()[1] - 0.62 - 0.6 * k for k in range(3)]
        steps = m.VGroup(
            sit(code(f"{HAD} − {POSITIVE} = {STEP1}", 24, INK).move_to([0, 0, 0]).align_to([x_text, 0, 0], m.LEFT), ys[0]),
            sit(code(f"{STEP1} + {NEW_NEGATIVE} = {STEP2}", 24, INK).move_to([0, 0, 0]).align_to([x_text, 0, 0], m.LEFT), ys[1]),
            sit(code(f"Answer: {STEP2}", 24, GOOD).move_to([0, 0, 0]).align_to([x_text, 0, 0], m.LEFT), ys[2]),
        ).set_z_index(1)
        links = m.VGroup(
            *[
                m.CurvedArrow([x_text - 0.25, ys[k] + 0.12, 0], [x_text - 0.25, ys[k + 1] + 0.12, 0], angle=m.PI * 0.8, tip_length=0.15).set_stroke(VIOLET, width=3).set_color(VIOLET)
                for k in range(2)
            ]
        ).set_z_index(1)
        right = level_pill("✓ right", GOOD, GOOD_FILL, 22).next_to(box, m.DOWN, buff=0.25)
        sources = m.VGroup(cite("chain-of-thought: Wei et al., 2022"), cite("“step by step”: Kojima et al., 2022")).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        sources.move_to([0, -1.25, 0]).align_to([WX - WW / 2, 0, 0], m.LEFT)
        gains = pill("helps most: large models, multi-step problems", GOOD, GOOD_FILL, 20)
        gains.move_to([0, -2.3, 0]).align_to([WX - WW / 2, 0, 0], m.LEFT)
        later = pill("Episode 43 · Reasoning models", PINK, PINK_FILL, 22).move_to([RX, -2.3, 0])

        self.say(
            "[Some problems take more than one step.]",
            m.AnimationGroup(self.new_header("4 · CHAIN-OF-THOUGHT", "Ask for the steps"), m.FadeIn(chrome, scale=0.97), m.LaggedStart(*[typed(p, 0.015) for p in problem], lag_ratio=0.9)),
        )
        self.say(
            "[Asked for the answer straight away], [a model might blurt out a wrong number, like twenty-nine.]",
            m.AnimationGroup(m.FadeIn(a_mark), m.GrowArrow(feed), m.FadeIn(model, scale=0.8)),
            m.Succession(m.FadeIn(hurried, scale=1.1), m.FadeIn(wrong, shift=0.15 * m.UP)),
        )
        self.say(
            "[Chain-of-thought prompting asks it to write out the steps first]: [with worked examples, or simply, let's think step by step.]",
            m.AnimationGroup(m.FadeOut(hurried), m.FadeOut(wrong), m.FadeIn(box)),
            typed(think, 0.03),
        )
        self.say(
            "[Twenty-three minus twenty leaves three negative reviews.] [Plus six makes nine.]",
            typed(steps[0], 0.04),
            m.Succession(typed(steps[1], 0.04), typed(steps[2], 0.04), m.FadeIn(right, shift=0.15 * m.UP)),
        )
        self.say(
            "[Each written step becomes context for the next], [so one hard jump becomes a chain of easy ones.]",
            m.LaggedStart(*[m.Create(link) for link in links], lag_ratio=0.6),
            m.Circumscribe(steps, color=VIOLET, buff=0.12),
        )
        self.say(
            "[In twenty twenty-two, researchers showed that this helps large models], [on multi-step problems like math word problems.]",
            m.FadeIn(sources, shift=0.1 * m.UP),
            m.FadeIn(gains, shift=0.15 * m.UP),
        )
        self.say(
            "[Reasoning models, in episode forty-three, take this idea much further.]",
            m.FadeIn(later, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ anatomy

    def anatomy(self):
        self.next_section(
            "5 · Anatomy",
            notes="Role, task, context, examples, output format, then the input. Episode 32 (RAG) puts retrieved documents in the context slot.",
        )
        x0, width = -3.6, 9.1
        left = x0 + 0.45
        top = TOP
        n_lines = 7
        row = 0.56
        height = 0.58 + 0.5 + (n_lines - 1) * row + 0.45
        chrome = window(x0 + width / 2, top, width, height, "classifier_prompt.txt")
        b = [top - 1.08 - row * k for k in range(n_lines)]
        role = line("You are a careful analyst of product reviews.", VIOLET, left, b[0])
        task = line("Label the sentiment of each review.", TEAL, left, b[1])
        context = line("The reviews are for headphones from our shop.", BLUE, left, b[2])
        shots = m.VGroup(shot_line(*SHOTS[0], left, b[3]), shot_line(*SHOTS[1], left, b[4]))
        fmt = line("Reply with one word: Positive or Negative.", PINK, left, b[5], t2c={"Positive": GOOD, "Negative": BAD})
        query = input_line(left, b[6])
        answer = line("Negative", BAD, query.get_right()[0] + 0.2, b[6])

        def tag(name, color, fill, y):
            t = pill(name, color, fill, 20)
            return t.move_to([0, y, 0]).align_to([x0 - 0.2, 0, 0], m.RIGHT)

        mid = lambda k: b[k] + 0.1  # noqa: E731  (a line's visual middle)
        tags = m.VGroup(
            tag("ROLE", *MODEL, mid(0)),
            tag("TASK", *TASK, mid(1)),
            tag("CONTEXT", *CONTEXT, mid(2)),
            tag("EXAMPLES", *EXAMPLE, (mid(3) + mid(4)) / 2),
            tag("FORMAT", *FORMAT, mid(5)),
        )
        input_tag = tag("INPUT", *INPUT, mid(6))
        rag = pill("Episode 32 · retrieval fills this slot", TEAL, TEAL_FILL, 20).move_to([0, -3.05, 0])
        check = code_card([('assert reply in ("Positive", "Negative")', INK)], EDGE, CARD)
        passes = verdict(True)
        m.VGroup(check, passes).arrange(m.RIGHT, buff=0.3).move_to([0, -3.05, 0])

        self.say(
            "[Let's write our classifier's full prompt.] [Good prompts usually have five parts.]",
            m.AnimationGroup(self.new_header("5 · STRUCTURE", "Anatomy of a prompt"), m.FadeIn(chrome, scale=0.97)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.2 * m.RIGHT) for t in tags], lag_ratio=0.2),
        )
        self.say(
            "[A role: who the model should be.] [The task: what it should do.]",
            m.AnimationGroup(m.Indicate(tags[0], color=VIOLET, scale_factor=1.1), typed(role)),
            m.AnimationGroup(m.Indicate(tags[1], color=TEAL, scale_factor=1.1), typed(task)),
        )
        self.say(
            "[Context: the background it needs.] [In episode thirty-two, retrieval puts whole documents right here.]",
            m.AnimationGroup(m.Indicate(tags[2], color=BLUE, scale_factor=1.1), typed(context)),
            m.AnimationGroup(m.FadeIn(rag, shift=0.2 * m.UP), m.Circumscribe(context, color=BLUE, buff=0.08)),
        )
        self.say(
            "[Examples, like the ones we wrote.] [And an output format: one word, positive or negative.]",
            m.AnimationGroup(m.Indicate(tags[3], color=AMBER, scale_factor=1.1), m.Succession(typed(shots[0], 0.018), typed(shots[1], 0.018))),
            m.AnimationGroup(m.Indicate(tags[4], color=PINK, scale_factor=1.1), typed(fmt)),
        )
        self.say(
            "[Then the review itself], [and the model answers: Negative.]",
            m.AnimationGroup(m.FadeIn(input_tag, shift=0.2 * m.RIGHT), typed(query)),
            typed(answer, 0.06),
        )
        self.say(
            "[Clear parts make the reply predictable], [and easy for your code to check.]",
            m.AnimationGroup(m.FadeOut(rag), m.Circumscribe(fmt, color=PINK, buff=0.08)),
            m.Succession(m.FadeIn(check, shift=0.2 * m.UP), m.FadeIn(passes, scale=1.2)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ brittle

    def brittle(self):
        orders = ", ".join(f"{round(100 * a)}" for a in ORDER_ACC)
        self.next_section(
            "6 · Brittle",
            notes=(
                "Accuracy can swing with prompt format, example choice and example order: Zhao et al., 2021 ('Calibrate Before Use'); "
                "Lu et al., 2022 ('Fantastically Ordered Prompts'). The chart is a toy simulation (illustrative): 200 test reviews, "
                f"a recency-weighted bias toward the examples' labels; the 24 orders score (%, sorted) {orders}."
            ),
        )
        x0, width, top = -4.25, 4.8, 2.35
        chrome = window(x0, top, width, 4.55)
        left = x0 - width / 2 + 0.3
        wording = line("Positive or negative?", TEAL, left, 1.12)
        reworded = line("Classify the sentiment:", TEAL, left, 1.12)
        names = (("A", "Positive"), ("B", "Negative"), ("C", "Positive"), ("D", "Negative"))
        slots = [0.62 - 0.55 * k for k in range(4)]
        rows = m.VGroup()
        for (n, lab), y in zip(names, slots):
            chip_box = card(width - 0.6, 0.46, AMBER, AMBER_FILL, radius=0.12, stroke=2).move_to([x0, y, 0])
            words = code(f"example {n} → {lab}", 20, INK, t2c={"→": MUTED, lab: LABEL_COLOR[lab]})
            words.move_to(chip_box).align_to(chip_box.get_left() + 0.22 * m.RIGHT, m.LEFT)
            sit(words, y - 0.08)
            rows.add(m.VGroup(chip_box, words).set_z_index(1))
        query = line("Great, it broke… →", SKY, left, slots[-1] - 0.62, 20, t2c={"→": MUTED})
        shuffle = (2, 0, 3, 1)  # new slot of each example

        ax = axes([0, 26, 5], [50, 100, 25], 7.0, 3.7)
        ax.shift(np.array([-0.5, -2.05, 0]) - ax.c2p(0, 50))
        ticks = m.VGroup(*[text(f"{v}%", 20, MUTED).next_to(ax.c2p(0, v), m.LEFT, buff=0.15) for v in (50, 75, 100)])
        y_title = text("accuracy", 20, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.15)
        x_title = text("all 24 orders, sorted", 20, MUTED).move_to([ax.c2p(12.5, 50)[0], -2.45, 0])
        bars = m.VGroup()
        for k, a in enumerate(ORDER_ACC):
            lo, hi = ax.c2p(k + 1, 50), ax.c2p(k + 1, 100 * a)
            bars.add(m.Rectangle(width=0.17, height=hi[1] - lo[1], stroke_width=0).set_fill(TEAL, opacity=0.85).move_to((lo + hi) / 2))
        low_label = text(f"{LOW}%", 24, BAD, weight=m.BOLD).move_to(ax.c2p(1, 84)).align_to(bars[0], m.LEFT)
        assert max(ORDER_ACC[:5]) < 0.8  # the label floats above the first bars, clear of them
        leader = m.Line(bars[0].get_top() + 0.08 * m.UP, [bars[0].get_x(), low_label.get_bottom()[1] - 0.08, 0]).set_stroke(BAD, width=2)
        high_label = text(f"{HIGH}%", 24, GOOD, weight=m.BOLD).next_to(bars[-1], m.UP, buff=0.12).align_to(bars[-1], m.RIGHT)
        sources = cite("Zhao et al., 2021 · Lu et al., 2022")
        sources.move_to([0, 2.35, 0]).align_to([6.5, 0, 0], m.RIGHT)
        advice = m.VGroup(
            pill("measure every prompt on a labelled test set", TEAL, TEAL_FILL, 20),
            pill("Episode 6 · Measuring a Model", SKY, SKY_FILL, 20),
        ).arrange(m.RIGHT, buff=0.3).move_to([0, -3.2, 0])

        self.say(
            "[Now the catch: prompts are brittle.]",
            m.AnimationGroup(
                self.new_header("THE CATCH", "Prompts are brittle"),
                m.FadeIn(chrome, scale=0.97), m.FadeIn(wording), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows], lag_ratio=0.15), m.FadeIn(query),
            ),
        )
        self.say(
            "[Rephrase the instruction], [or just reorder the examples], [and the accuracy can change.]",
            swap(wording, reworded),
            m.AnimationGroup(*[r.animate(path_arc=0.6).move_to([x0, slots[s], 0]) for r, s in zip(rows, shuffle)], run_time=1.2),
            m.AnimationGroup(m.Create(ax), m.FadeIn(ticks), m.FadeIn(y_title), m.FadeIn(x_title)),
        )
        self.say(
            "[For example, try all twenty-four orders of the same four examples], on the same test reviews.",
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.12, run_time=2.4),
        )
        self.say(
            "[Here, accuracy runs from fifty-eight], [to eighty-eight percent], [with the very same examples.]",
            m.AnimationGroup(bars[0].animate.set_fill(BAD, opacity=0.95), m.Create(leader), m.FadeIn(low_label, shift=0.1 * m.UP)),
            m.AnimationGroup(bars[-1].animate.set_fill(GOOD, opacity=0.95), m.FadeIn(high_label, shift=0.1 * m.UP)),
            m.FadeIn(sources),
        )
        self.say(
            "[So measure every prompt on labelled examples], [just as we measured models in episode six.]",
            m.FadeIn(advice[0], shift=0.2 * m.UP),
            m.FadeIn(advice[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ finite

    def finite(self):
        self.next_section(
            "7 · Finite",
            notes="Instructions, examples, the input (and the reply) share one context window of tokens; a prompt past the limit does not fit. Many labelled examples: fine-tuning (Episode 29).",
        )
        y, x_start, limit = 0.6, -6.3, 5.4
        bar = card(limit - x_start + 0.2, 1.0, EDGE, CARD, radius=0.15, stroke=2).move_to([(x_start - 0.1 + limit + 0.1) / 2, y, 0]).set_z_index(-1)
        name = text("context window", 22, MUTED, weight=m.SEMIBOLD).next_to(bar, m.UP, buff=0.2).align_to(bar, m.LEFT)
        edge = m.DashedLine([limit, y - 0.75, 0], [limit, y + 0.75, 0], dash_length=0.1).set_stroke(BAD, width=3)
        edge_name = text("limit", 20, BAD, weight=m.SEMIBOLD).next_to(edge, m.UP, buff=0.1)

        widths = {"task": 1.0, "example": 1.1, "input": 0.9}
        gap = 0.08

        def layout(n_examples):
            """x-centres of the blocks: instruction, n examples, the review."""
            xs, x = [], x_start
            for kind in ["task"] + ["example"] * n_examples + ["input"]:
                xs.append(x + widths[kind] / 2)
                x += widths[kind] + gap
            return xs

        def blk(kind, color, fill, x):
            return block(color, fill, widths[kind], 0.66).move_to([x, y, 0]).set_z_index(1)

        three, eight, nine = layout(3), layout(8), layout(9)
        assert eight[-1] + widths["input"] / 2 < limit < nine[-1] + widths["input"] / 2  # eight fit, the ninth pushes the review out
        task = blk("task", *TASK, three[0])
        examples = [blk("example", *EXAMPLE, x) for x in three[1:4]]
        review = blk("input", *INPUT, three[-1])
        more = [blk("example", *EXAMPLE, x) for x in eight[4:9]]
        last = blk("example", *EXAMPLE, nine[9])
        costs = text("every example costs tokens", 22, MUTED).move_to([-3.2, y - 1.05, 0])
        spill = text("✗ doesn't fit", 22, BAD, weight=m.SEMIBOLD).move_to([0, y - 1.05, 0]).align_to([6.45, 0, 0], m.RIGHT)
        tile = card(9.6, 1.6, TEAL, CARD, stroke=2).move_to([0, -2.1, 0])
        words = m.VGroup(text("Thousands of labelled reviews?", 28, INK, weight=m.BOLD), text("Fine-tuning (Episode 29) may be the better tool.", 22, MUTED)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        body = m.VGroup(icon("database", 0.95), words).arrange(m.RIGHT, buff=0.45).move_to(tile)

        self.say(
            "[And everything must fit in the context window.]",
            m.AnimationGroup(
                self.new_header("THE CATCH", "The window is finite"),
                m.FadeIn(bar), m.FadeIn(name), m.Create(edge), m.FadeIn(edge_name),
                m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.RIGHT) for b in (task, *examples, review)], lag_ratio=0.15),
            ),
        )
        self.say(
            "[Every example uses up tokens], [and past the limit, nothing more fits.]",
            m.AnimationGroup(
                m.FadeIn(costs, shift=0.1 * m.UP),
                review.animate.move_to([eight[-1], y, 0]),
                m.LaggedStart(*[m.FadeIn(b, scale=0.8) for b in more], lag_ratio=0.2),
                run_time=1.2,  # done before the next bracket moves the review again
            ),
            m.AnimationGroup(
                m.FadeIn(last, scale=0.8),
                review.animate.move_to([nine[-1], y, 0]).set_stroke(BAD).set_fill(BAD_FILL),
                m.FadeIn(spill, shift=0.1 * m.UP),
            ),
        )
        self.say(
            "[With thousands of labelled reviews, fine-tuning, from episode twenty-nine, may be the better tool.]",
            m.AnimationGroup(m.FadeIn(tile), m.FadeIn(body, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Ask (instructions), show (examples), reason (step by step).")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in (("Ask", TEAL), ("Show", AMBER), ("Reason", VIOLET))]).arrange(m.RIGHT, buff=1.3))
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A good prompt can teach a new task, without changing a single weight.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [ask], [show], [reason.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A good prompt can teach a model a new task, without changing a single weight.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
