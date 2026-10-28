"""AI, Explained · Episode 17 · Information Theory

Running example: guessing the next word of "The cat sat on the ...". For example, the world's
chances for the next word are p = (mat 1/2, sofa 1/4, floor 1/8, roof 1/8), so the surprises
are exactly 1, 2, 3 and 3 bits and the entropy is 1.75 bits. A model believes, for example,
q = (0.56, 0.28, 0.14, 0.02): a little too sure of the first three words, and roof nearly
ruled out.

    surprise          -log2 p(x)                 1, 2, 3, 3 bits; a fair coin 1 bit; 1-in-8 3 bits
    entropy           H(p)    = 1.75 bits        fair coin 1 bit, loaded coin (90% heads) 0.47 bits
    cross-entropy     H(p, q) = 1.94 bits        roof's surprise under q: 5.64 bits
    KL divergence     D(p||q) = 0.19 bits        swapped, D(q||p) = 0.11 bits (not symmetric)
    perplexity        2^1.75 = 3.36, 2^1.94 = 3.83 (= e^1.34 nats)
    a language model  chances 0.02, 0.20, 0.50, 0.70, 0.56 for "cat sat on the mat":
                      1.43 nats per token on average, perplexity e^1.43 = 4.2

Every number on screen and in the narration is computed below and asserted, so the film cannot
drift from its words; a seeded sample of 200,000 words checks that the formulas really are
average surprises. The distributions are made up for teaching (said "for example").

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

WORDS = ("mat", "sofa", "floor", "roof")
FRACS = ("1/2", "1/4", "1/8", "1/8")
P = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])  # the world: how often each word really comes next
Q = np.array([0.56, 0.28, 0.14, 0.02])  # a model's beliefs (for example)


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cross_entropy(p, q):
    return float(-(np.asarray(p) * np.log2(q)).sum())


def kl(p, q):
    return cross_entropy(p, q) - entropy(p)


def coin_entropy(b):
    return entropy([b, 1 - b])


S_P = -np.log2(P)  # surprise of each word under the world's chances
S_Q = -np.log2(Q)  # ... under the model's
H_P, H_PQ = entropy(P), cross_entropy(P, Q)
KL_PQ, KL_QP = kl(P, Q), kl(Q, P)
PPL_P, PPL_Q = 2**H_P, 2**H_PQ
NATS_PQ = H_PQ * np.log(2)
H_FAIR, H_LOADED = coin_entropy(0.5), coin_entropy(0.9)

assert np.isclose(P.sum(), 1) and np.isclose(Q.sum(), 1)
assert np.allclose(S_P, [1, 2, 3, 3])  # "mat is one bit, sofa two, floor and roof three each"
assert -np.log2(1 / 2) == 1 and -np.log2(1 / 8) == 3  # "a fair coin is one bit", "one in eight is three bits"
assert H_P == 0.5 * 1 + 0.25 * 2 + 0.125 * 3 + 0.125 * 3 == 1.75  # "one point seven five bits"
assert H_FAIR == 1.0 and 0.45 < H_LOADED < 0.5 and round(H_LOADED, 2) == 0.47  # "just under half a bit"
assert all(coin_entropy(b) < 1 for b in np.linspace(0, 1, 101) if abs(b - 0.5) > 1e-9)  # the peak is at one half
assert 5.5 < S_Q[3] < 5.7 and round(S_Q[3], 2) == 5.64  # roof under q: "over five and a half bits"
assert all(S_Q[:3] < S_P[:3])  # the model is less surprised by the other three words
assert round(H_PQ, 2) == 1.94 and H_PQ > H_P  # "about one point nine four bits, more than one point seven five"
assert round(KL_PQ, 2) == 0.19 and round(KL_QP, 2) == 0.11  # "zero point one nine", swapped "zero point one one"
assert KL_PQ > 0 and KL_QP > 0 and abs(kl(P, P)) < 1e-12  # never negative, zero when q = p
assert round(PPL_P, 1) == 3.4 and round(PPL_P, 2) == 3.36 and 3 < PPL_P < 4  # "about three point four"
assert round(PPL_Q, 1) == 3.8 and round(PPL_Q, 2) == 3.83  # "about three point eight"
assert round(NATS_PQ, 2) == 1.34 and np.isclose(np.exp(NATS_PQ), PPL_Q)  # e^(nats) = 2^(bits)

# the formulas really are average surprises: draw many words from p (seeded)
_draws = np.random.default_rng(17).choice(len(P), size=200_000, p=P)
assert abs(S_P[_draws].mean() - H_P) < 0.01 and abs(S_Q[_draws].mean() - H_PQ) < 0.01

# a language model reading "The cat sat on the mat": its chance for each real next word (for example)
TOKENS = ("The", "cat", "sat", "on", "the", "mat")
CHANCES = np.array([0.02, 0.20, 0.50, 0.70, 0.56])
NATS = -np.log(CHANCES)
AVG_NATS = float(NATS.mean())
PPL_TEXT = float(np.exp(AVG_NATS))
assert CHANCES[-1] == Q[0]  # the same model's chance for "mat"
assert round(AVG_NATS, 2) == 1.43 and round(PPL_TEXT, 1) == 4.2  # "about four point two"
assert np.isclose(PPL_TEXT, 2 ** (AVG_NATS / np.log(2)))

# ---------------------------------------------------------------------------- layout

ROW_Y = (1.45, 0.55, -0.35, -1.25)  # the word table (game and surprise)
CHIP_X, BAR_X0, BAR_W, BAR_H = -5.75, -4.85, 5.2, 0.42
BITS_X = -0.7
CH_X0, CH_W = -6.6, 7.2  # the bar chart: width = chance, height = surprise
GAP = 0.06


def bits_words(b):
    return f"{b:g} bit" if b == 1 else f"{b:g} bits"


def chart_bars(heights, unit, base, color, fill, opacity=1.0):
    """One bar per word: as wide as its chance (of CH_W), as tall as `heights` bits."""
    bars, x = m.VGroup(), CH_X0
    for p, h in zip(P, heights):
        w = CH_W * p
        bar = m.Rectangle(width=w - GAP, height=max(h * unit, 0.002)).set_stroke(color, width=2.5).set_fill(fill, opacity=opacity)
        bars.add(bar.move_to([x + w / 2, base + h * unit / 2, 0]))
        x += w
    return bars


def level_bars(height, unit, base, color, opacity=0.28):
    """The same four columns, all at one height: the average."""
    cols, x = m.VGroup(), CH_X0
    for p in P:
        w = CH_W * p
        col = m.Rectangle(width=w - GAP, height=height * unit).set_stroke(color, width=0).set_fill(color, opacity=opacity)
        cols.add(col.move_to([x + w / 2, base + height * unit / 2, 0]))
        x += w
    return cols


def chart_labels(base):
    words, fracs, x = m.VGroup(), m.VGroup(), CH_X0
    for p, w, f in zip(P, WORDS, FRACS):
        cx = x + CH_W * p / 2
        words.add(text(w, 24, INK, weight=m.SEMIBOLD).move_to([cx, base - 0.33, 0]))
        fracs.add(code(f, 22, TEAL).move_to([cx, base - 0.72, 0]))
        x += CH_W * p
    on_baseline(words)
    return words, fracs


def coin(radius=0.5, letter="H"):
    rim = m.Circle(radius=radius).set_stroke(AMBER, width=4).set_fill(AMBER_FILL, opacity=1)
    inner = m.Circle(radius=radius * 0.78).set_stroke(AMBER, width=2, opacity=0.6)
    face = text(letter, int(radius * 62), AMBER, weight=m.BOLD).move_to(rim)
    return m.VGroup(rim, inner, face)


def cell(label="", side=0.5, size=22):
    box = m.RoundedRectangle(corner_radius=0.08, width=side, height=side).set_stroke(SKY, width=2.5).set_fill(SKY_FILL, opacity=1)
    if not label:
        return m.VGroup(box)
    return m.VGroup(box, text(label, size, INK, weight=m.BOLD).move_to(box))


def legend_row(formula, words, color):
    return m.VGroup(tex(formula, 34, color), text(words, 22, MUTED)).arrange(m.RIGHT, buff=0.25)


def cite(words):
    return text(words, 20, MUTED)


# ---------------------------------------------------------------------------- the episode


class Ep17InformationTheory(Episode):
    number = 17

    def lesson(self):
        self.hook()
        self.game()
        self.surprise()
        self.entropy()
        self.coins()
        self.cross_entropy()
        self.divergence()
        self.perplexity()
        self.language_models()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Guessing the next word: some answers surprise you more than others.")
        sentence = text("The cat sat on the", 60, INK, weight=m.SEMIBOLD)
        blank = m.VGroup(card(1.9, 0.95, MUTED, CARD, radius=0.2, stroke=2.5))
        blank.add(text("?", 46, MUTED, weight=m.BOLD).move_to(blank[0]))
        row = m.VGroup(sentence, blank).arrange(m.RIGHT, buff=0.35).move_to([0, 1.6, 0])
        blank.set_y(sentence[0].get_bottom()[1] + 0.36)
        self.sentence = row

        def outcome(word, b, caption, x):
            box = card(5.6, 2.0, EDGE, CARD, stroke=2).move_to([x, -0.75, 0])
            w = chip(word, SKY, SKY_FILL, width=1.7, height=0.8, size=32).move_to(box.get_left() + np.array([1.45, 0.22, 0]))
            cap = text(caption, 22, MUTED).next_to(w, m.DOWN, buff=0.22)
            track = m.Rectangle(width=2.4, height=0.3).set_stroke(EDGE, width=2).set_fill(BG, opacity=1)
            track.move_to(box.get_right() + np.array([-1.6, -0.1, 0]))
            name = text("surprise", 22, AMBER, weight=m.SEMIBOLD).next_to(track, m.UP, buff=0.2).align_to(track, m.LEFT)
            fill = m.Rectangle(width=2.4 * b / 3, height=0.3, stroke_width=0).set_fill(AMBER, opacity=0.9)
            fill.move_to(track.get_left(), aligned_edge=m.LEFT).set_z_index(1)
            value = text(bits_words(b), 22, AMBER, weight=m.BOLD).next_to(track, m.DOWN, buff=0.2).align_to(track, m.RIGHT)
            return m.VGroup(box, w, cap, track, name), fill, value

        mat, mat_fill, mat_bits = outcome("mat", 1, "hardly surprising", -3.15)
        roof, roof_fill, roof_bits = outcome("roof", 3, "more surprising", 3.15)
        today = m.VGroup(*[pill(s, AMBER, AMBER_FILL, 22) for s in ("entropy", "cross-entropy", "KL divergence", "perplexity")])
        today.arrange(m.RIGHT, buff=0.35).move_to([0, -2.95, 0])

        self.say(
            "So... [guess the next word]: [the cat sat on the...]",
            m.FadeIn(sentence, shift=0.3 * m.UP),
            m.Succession(m.FadeIn(blank, scale=0.8), m.Indicate(blank, color=SKY, scale_factor=1.1)),
        )
        self.say(
            "[If it's mat, you're hardly surprised.] [If it's roof, you're more surprised.]",
            m.AnimationGroup(m.FadeIn(mat, shift=0.2 * m.UP), m.GrowFromEdge(mat_fill, m.LEFT)),
            m.AnimationGroup(m.FadeIn(roof, shift=0.2 * m.UP), m.GrowFromEdge(roof_fill, m.LEFT)),
        )
        self.say(
            "[Information theory turns that feeling into a number], [and it's at the heart of how language models learn.]",
            m.AnimationGroup(m.FadeIn(mat_bits, scale=1.2), m.FadeIn(roof_bits, scale=1.2)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.2 * m.UP) for t in today], lag_ratio=0.25),
        )
        self.wait(0.3)
        self.play(*[m.FadeOut(mob) for mob in (mat, mat_fill, mat_bits, roof, roof_fill, roof_bits, today)], run_time=0.7)

    # ------------------------------------------------------------------ the game: a distribution

    def game(self):
        self.next_section(
            "The game",
            notes="For example: p = mat 1/2, sofa 1/4, floor 1/8, roof 1/8 (illustrative chances, chosen so the surprises are whole bits).",
        )
        chips = m.VGroup(*[chip(w, SKY, SKY_FILL, width=1.45, height=0.62, size=26).move_to([CHIP_X, y, 0]) for w, y in zip(WORDS, ROW_Y)])
        bars = m.VGroup(
            *[
                m.Rectangle(width=BAR_W * p, height=BAR_H, stroke_width=0).set_fill(TEAL, opacity=0.85).move_to([BAR_X0, y, 0], aligned_edge=m.LEFT)
                for p, y in zip(P, ROW_Y)
            ]
        )
        fracs = m.VGroup(*[code(f, 26, TEAL).next_to(b, m.RIGHT, buff=0.22) for f, b in zip(FRACS, bars)])
        head = text("how often it comes next", 22, TEAL, weight=m.SEMIBOLD).move_to([0, 2.2, 0]).align_to([BAR_X0, 0, 0], m.LEFT)
        total = tex(r"\tfrac{1}{2} + \tfrac{1}{4} + \tfrac{1}{8} + \tfrac{1}{8} = 1", 42, TEAL)
        total.move_to([0, -2.25, 0]).align_to([BAR_X0, 0, 0], m.LEFT)
        name = pill("p: a probability distribution", TEAL, TEAL_FILL, 22).move_to([0, -3.1, 0]).align_to([BAR_X0, 0, 0], m.LEFT)

        # eight typical endings: four mats, two sofas, a floor and a roof
        target = self.sentence.copy().scale(0.62).move_to([3.7, 1.75, 0])
        endings = m.VGroup(*[chip(w, SKY, SKY_FILL, width=1.25, height=0.58, size=22) for w in ("mat",) * 4 + ("sofa",) * 2 + ("floor", "roof")])
        endings.arrange_in_grid(rows=2, cols=4, buff=(0.2, 0.25)).move_to([3.7, -0.35, 0])
        ending_head = text("eight typical endings, for example", 22, MUTED).next_to(endings, m.UP, buff=0.35)
        groups = (endings[:4], endings[4:6], endings[6:])

        self.say(
            "[Say we know, for example, how often each word comes next.]",
            m.AnimationGroup(
                self.new_header("THE GAME", "Guess the next word"),
                m.Transform(self.sentence, target),
                m.FadeIn(ending_head),
                m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.RIGHT) for c in chips], lag_ratio=0.15),
                m.FadeIn(head),
            ),
        )
        self.say(
            "[Mat comes up half the time], [sofa a quarter of the time], [floor and roof one time in eight each.]",
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(fracs[0]), m.LaggedStart(*[m.FadeIn(c, scale=0.7) for c in groups[0]], lag_ratio=0.15)),
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(fracs[1]), m.LaggedStart(*[m.FadeIn(c, scale=0.7) for c in groups[1]], lag_ratio=0.2)),
            m.AnimationGroup(*[m.GrowFromEdge(b, m.LEFT) for b in bars[2:]], *[m.FadeIn(f) for f in fracs[2:]], *[m.FadeIn(c, scale=0.7) for c in groups[2]]),
        )
        self.say(
            "[Together, they add up to one]: [a probability distribution, which we'll call p.]",
            m.Write(total),
            m.FadeIn(name, shift=0.2 * m.UP),
        )
        self.table = dict(chips=chips, bars=bars, fracs=fracs, head=head)
        self.wait(0.3)
        self.play(*[m.FadeOut(mob) for mob in (total, name, ending_head, endings, self.sentence)], run_time=0.7)

    # ------------------------------------------------------------------ surprise

    def surprise(self):
        self.next_section("Surprise", notes="surprise = -log2 p, in bits: 1/2 -> 1 bit (a fair coin), 1/8 -> 3 bits (three halvings).")
        formula = m.MathTex(r"\text{surprise}(x)", "=", "-", r"\log_2", r"p(x)", font_size=46, color=INK)
        formula[0].set_color(AMBER)
        formula[4].set_color(TEAL)
        formula.move_to([3.7, 2.05, 0])
        unit = text("measured in bits", 22, MUTED).next_to(formula, m.DOWN, buff=0.25)

        # a fair coin: two options, one question
        the_coin = coin(0.45).move_to([1.15, 0.2, 0])
        sides = m.VGroup(cell("H", 0.72, 26), cell("T", 0.72, 26)).arrange(m.RIGHT, buff=0.14).move_to([2.75, 0.2, 0])
        half = code("1/2", 22, TEAL).next_to(sides[0], m.DOWN, buff=0.15)
        one = m.VGroup(text("1 bit", 30, AMBER, weight=m.BOLD), text("1 question", 22, MUTED)).arrange(m.DOWN, buff=0.1)
        one.move_to([5.5, 0.2, 0])

        # one in eight: three halvings
        cells = m.VGroup(*[cell("", 0.46) for _ in range(8)]).arrange(m.RIGHT, buff=0.09).move_to([2.95, -1.25, 0])
        steps = m.VGroup(*[code(s, 24, MUTED) for s in ("8", "→ 4", "→ 2", "→ 1")]).arrange(m.RIGHT, buff=0.18)
        steps.next_to(cells, m.DOWN, buff=0.3)
        three = m.VGroup(text("3 bits", 30, AMBER, weight=m.BOLD), text("3 questions", 22, MUTED)).arrange(m.DOWN, buff=0.1)
        three.move_to([6.0, -1.25, 0])
        answer = 5
        keep = [range(4, 8), range(4, 6), range(5, 6)]
        alive = set(range(8))
        halvings = []
        for k in keep:
            gone = [cells[i] for i in sorted(alive - set(k))]
            alive = set(k)
            halvings.append(m.AnimationGroup(*[c.animate.set_opacity(0.18) for c in gone], run_time=0.6))
        halvings = [m.AnimationGroup(h, m.FadeIn(s, shift=0.1 * m.RIGHT)) for h, s in zip(halvings, steps[1:])]
        chain = tex(r"1 \;\to\; \tfrac{1}{2} \;\to\; \tfrac{1}{4} \;\to\; \tfrac{1}{8}", 34, TEAL)
        chain_note = text("three halvings: three bits", 22, MUTED)
        log_note = m.VGroup(chain, chain_note).arrange(m.RIGHT, buff=0.35).move_to([3.7, -2.95, 0])

        pills = m.VGroup(
            *[pill(bits_words(b), AMBER, AMBER_FILL, 22).move_to([BITS_X, y, 0]) for b, y in zip(S_P, ROW_Y)]
        )
        pills_head = text("surprise", 22, AMBER, weight=m.SEMIBOLD).move_to([BITS_X, 2.2, 0])

        # the curve: surprise against chance
        ax = axes([0, 1.12, 0.25], [0, 4.4, 1], 5.5, 3.3).move_to([3.85, -1.3, 0])
        ticks = m.VGroup(
            *[code(s, 20, MUTED).next_to(ax.c2p(v, 0), m.DOWN, buff=0.24) for s, v in (("1/8", 1 / 8), ("1/4", 1 / 4), ("1/2", 1 / 2), ("1", 1))],
            *[code(str(k), 20, MUTED).next_to(ax.c2p(0, k), m.LEFT, buff=0.15) for k in (1, 2, 3)],
        )
        x_name = text("chance p", 22, TEAL).move_to([ax.c2p(0.55, 0)[0], -3.52, 0])
        y_name = text("surprise in bits", 22, AMBER).next_to(ax.c2p(0, 4.4), m.UP, buff=0.15).align_to(ax.c2p(0, 0), m.LEFT)
        curve = ax.plot(lambda p: -np.log2(p), x_range=[0.05, 1.0]).set_stroke(AMBER, width=4.5)
        marks = m.VGroup(*[m.Dot(ax.c2p(p, -np.log2(p)), radius=0.08, color=SKY).set_z_index(3) for p in (1 / 2, 1 / 4, 1 / 8)])
        guides = m.VGroup(
            *[m.DashedLine(ax.c2p(0, -np.log2(p)), ax.c2p(p, -np.log2(p)), dash_length=0.08).set_stroke(FAINT, width=2) for p in (1 / 2, 1 / 4, 1 / 8)]
        )
        names = m.VGroup(
            text("mat", 22, SKY, weight=m.SEMIBOLD).next_to(marks[0], m.UR, buff=0.08),
            text("sofa", 22, SKY, weight=m.SEMIBOLD).next_to(marks[1], m.UR, buff=0.08),
            text("floor, roof", 22, SKY, weight=m.SEMIBOLD).next_to(marks[2], m.RIGHT, buff=0.2).shift(0.12 * m.UP),
        )
        chance = m.ValueTracker(0.5)
        rider = m.always_redraw(lambda: m.Dot(ax.c2p(chance.get_value(), -np.log2(chance.get_value())), radius=0.12, color=AMBER).set_z_index(4))
        certain = text("certain: 0 bits", 22, AMBER, weight=m.BOLD).move_to(ax.c2p(0.9, 0.85))

        self.say(
            "[Information theory measures surprise in bits]: [minus the log, base two, of the probability.]",
            m.AnimationGroup(self.new_header("SURPRISE", "How unexpected is an outcome?"), m.FadeIn(unit)),
            m.Write(formula),
        )
        self.say(
            "[Take a fair coin.] [Heads has a one in two chance]: [that's one bit, the answer to one yes or no question.]",
            m.AnimationGroup(m.FadeIn(the_coin, scale=0.8), m.FadeIn(sides, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Indicate(sides[0], color=AMBER, scale_factor=1.15), m.FadeIn(half)),
            m.AnimationGroup(sides[1].animate.set_opacity(0.18), m.FadeIn(one, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[A one in eight chance is three bits]: [three yes or no questions], [each one cutting the options in half: eight, four, two, one.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in cells], lag_ratio=0.08), m.FadeIn(steps[0])),
            m.FadeIn(three, shift=0.2 * m.LEFT),
            m.Succession(*halvings, m.Indicate(cells[answer], color=AMBER, scale_factor=1.3)),
        )
        self.say(
            "[That's what the log counts]: [how many halvings it takes to reach the chance.] [The minus sign just keeps it positive.]",
            m.Indicate(formula[3], color=AMBER, scale_factor=1.2),
            m.FadeIn(log_note, shift=0.2 * m.UP),
            m.Indicate(formula[2], color=AMBER, scale_factor=1.6),
        )
        t = self.table
        self.say(
            "[So for our words]: [mat is one bit], [sofa is two], [floor and roof are three each.]",
            m.FadeIn(pills_head),
            m.AnimationGroup(m.FadeIn(pills[0], shift=0.2 * m.LEFT), m.Indicate(t["chips"][0], color=SKY, scale_factor=1.08)),
            m.AnimationGroup(m.FadeIn(pills[1], shift=0.2 * m.LEFT), m.Indicate(t["chips"][1], color=SKY, scale_factor=1.08)),
            m.AnimationGroup(*[m.FadeIn(p, shift=0.2 * m.LEFT) for p in pills[2:]]),
        )
        self.say(
            "[The rarer the word, the bigger the surprise], [and a word you're certain of carries no surprise at all.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (the_coin, sides, half, one, cells, steps, three, log_note)],
                m.Create(ax), m.FadeIn(x_name), m.FadeIn(y_name), m.FadeIn(ticks),
                m.Create(curve, run_time=1.4),
                m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in marks], lag_ratio=0.3),
                m.Create(guides),
                m.FadeIn(names),
            ),
            m.AnimationGroup(m.FadeIn(rider, scale=0.5), chance.animate(run_time=1.6).set_value(1.0), m.FadeIn(certain, shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.stop(rider)
        self.clear()

    # ------------------------------------------------------------------ entropy: the average surprise

    def entropy(self):
        self.next_section("Entropy", notes="H(p) = -sum p log2 p = 1/2*1 + 1/4*2 + 1/8*3 + 1/8*3 = 1.75 bits. Shannon, 1948, 'A Mathematical Theory of Communication'.")
        unit, base = 1.1, -2.45
        bars = chart_bars(S_P, unit, base, AMBER, AMBER_FILL)
        bars.set_z_index(1)
        floor_line = m.Line([CH_X0 - 0.1, base, 0], [CH_X0 + CH_W + 0.1, base, 0]).set_stroke(FAINT, width=3)
        words, fracs = chart_labels(base)
        tops = m.VGroup(*[text(bits_words(b), 22, AMBER, weight=m.BOLD).next_to(bar, m.UP, buff=0.15) for b, bar in zip(S_P, bars)])
        wide = m.VGroup(*[m.Line(bar.get_corner(m.DL) + 0.02 * m.UP, bar.get_corner(m.DR) + 0.02 * m.UP).set_stroke(TEAL, width=6) for bar in bars]).set_z_index(2)
        level = level_bars(H_P, unit, base, AMBER)
        line_y = base + H_P * unit
        avg_line = m.DashedLine([CH_X0, line_y, 0], [CH_X0 + CH_W, line_y, 0], dash_length=0.12).set_stroke(AMBER, width=4).set_z_index(3)
        avg_label = text("average: 1.75 bits", 26, AMBER, weight=m.BOLD).move_to([CH_X0 + 1.75, line_y + 0.32, 0])

        formula = m.MathTex(r"H(p)", "=", "-", r"\sum_x", r"p(x)", r"\log_2 p(x)", font_size=46, color=INK)
        formula[0].set_color(AMBER)
        formula[4].set_color(TEAL)
        formula[5].set_color(AMBER)
        formula.move_to([3.85, 1.85, 0])
        source = cite("Shannon, 1948").next_to(formula, m.DOWN, buff=0.22).align_to(formula, m.RIGHT)
        legend = m.VGroup(
            legend_row(r"p(x)", "how often word x comes up", TEAL),
            legend_row(r"-\log_2 p(x)", "its surprise", AMBER),
        ).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([3.75, 0.2, 0])
        work = tex(r"\tfrac{1}{2}\cdot 1 + \tfrac{1}{4}\cdot 2 + \tfrac{1}{8}\cdot 3 + \tfrac{1}{8}\cdot 3", 38, INK).move_to([3.75, -1.35, 0])
        result = tex(r"= 1.75 \text{ bits}", 44, AMBER).next_to(work, m.DOWN, buff=0.35)

        self.say(
            "[So how surprised are we, on average?]",
            m.AnimationGroup(self.new_header("ENTROPY", "The average surprise"), m.Create(floor_line), m.FadeIn(words), m.FadeIn(fracs)),
        )
        self.say(
            "[Give each word a bar]: [as wide as its chance], [and as tall as its surprise.]",
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.2),
            m.AnimationGroup(m.Create(wide), m.Indicate(fracs, color=TEAL, scale_factor=1.15)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tops], lag_ratio=0.2),
        )
        self.say(
            "[Now level the bars out.] [The height you get is the average surprise]: [one point seven five bits.]",
            m.AnimationGroup(
                *[m.TransformFromCopy(b, c) for b, c in zip(bars, level)],
                bars.animate.set_fill(opacity=0.0).set_stroke(opacity=0.5),
                run_time=1.6,
            ),
            m.Create(avg_line),
            m.FadeIn(avg_label, shift=0.15 * m.UP),
        )
        self.say(
            "[That average is called entropy]: [minus the sum, over every word, of p times log p.]",
            m.FadeIn(source),
            m.Write(formula, run_time=1.6),
        )
        self.say(
            "[Each term is a word's chance], [times its surprise]: [a half times one, plus a quarter times two, plus two eighths times three.]",
            m.AnimationGroup(m.Indicate(formula[4], color=TEAL, scale_factor=1.2), m.FadeIn(legend[0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Indicate(m.VGroup(formula[2], formula[5]), color=AMBER, scale_factor=1.15), m.FadeIn(legend[1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Write(work, run_time=2.4), m.FadeIn(result, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ a fair coin and a loaded coin

    def coins(self):
        self.next_section("Fair and loaded coins", notes="H(b) = -b log2 b - (1-b) log2 (1-b): 1 bit at b = 0.5, 0.47 bits at b = 0.9, 0 at b = 0 or 1.")
        b = m.ValueTracker(0.5)
        the_coin = coin(0.55).move_to([-4.7, 1.55, 0])
        lx, bw = -5.05, 2.4
        rows = m.VGroup(
            text("heads", 24, INK, weight=m.SEMIBOLD).move_to([-6.15, 0.35, 0]),
            text("tails", 24, INK, weight=m.SEMIBOLD).move_to([-6.15, -0.35, 0]),
        )
        rows[1].align_to(rows[0], m.LEFT)
        tracks = m.VGroup(
            *[m.Rectangle(width=bw, height=0.3).set_stroke(EDGE, width=2).set_fill(CARD, opacity=1).move_to([lx, y, 0], aligned_edge=m.LEFT) for y in (0.35, -0.35)]
        )

        def side_bar(chance_of, y):
            return m.always_redraw(
                lambda: m.Rectangle(width=max(0.02, bw * chance_of()), height=0.3, stroke_width=0).set_fill(TEAL, opacity=0.9).move_to([lx, y, 0], aligned_edge=m.LEFT).set_z_index(1)
            )

        def side_value(chance_of, y):
            return m.always_redraw(lambda: code(f"{100 * chance_of():.0f}%", 24, TEAL).move_to([lx + bw + 0.55, y, 0]))

        heads_bar, tails_bar = side_bar(b.get_value, 0.35), side_bar(lambda: 1 - b.get_value(), -0.35)
        heads_val, tails_val = side_value(b.get_value, 0.35), side_value(lambda: 1 - b.get_value(), -0.35)
        h_name = text("entropy", 26, AMBER, weight=m.BOLD).move_to([-4.65, -1.45, 0])
        h_val = m.always_redraw(lambda: text(f"{coin_entropy(b.get_value()):.2f} bits", 44, AMBER, weight=m.BOLD).move_to([-4.65, -2.2, 0]))

        ax = axes([0, 1.15, 0.25], [0, 1.2, 0.25], 6.6, 4.3).move_to([2.85, -0.55, 0])
        ticks = m.VGroup(
            code("0", 20, MUTED).next_to(ax.c2p(0, 0), m.DOWN, buff=0.18),
            code("1/2", 20, MUTED).next_to(ax.c2p(0.5, 0), m.DOWN, buff=0.18),
            code("1", 20, MUTED).next_to(ax.c2p(1, 0), m.DOWN, buff=0.18),
            code("1 bit", 20, MUTED).next_to(ax.c2p(0, 1), m.LEFT, buff=0.15),
        )
        x_name = text("chance of heads", 22, TEAL).next_to(ax.c2p(0.5, 0), m.DOWN, buff=0.6)
        y_name = text("entropy", 22, AMBER).next_to(ax.c2p(0, 1.2), m.RIGHT, buff=0.2)
        hill = ax.plot(coin_entropy, x_range=[0.0005, 0.9995]).set_stroke(AMBER, width=4.5)
        dot = m.always_redraw(lambda: m.Dot(ax.c2p(b.get_value(), coin_entropy(b.get_value())), radius=0.12, color=AMBER).set_z_index(4))
        drop = m.always_redraw(
            lambda: m.DashedLine(ax.c2p(b.get_value(), 0), ax.c2p(b.get_value(), max(coin_entropy(b.get_value()), 0.02)), dash_length=0.08).set_stroke(MUTED, width=2.5)
        )
        peak = text("hardest to predict", 22, AMBER, weight=m.BOLD).next_to(ax.c2p(0.5, 1), m.UP, buff=0.25)
        loaded = pill("90% heads: 0.47 bits", AMBER, AMBER_FILL, 20).next_to(ax.c2p(0.9, coin_entropy(0.9)), m.LEFT, buff=0.3).shift(0.4 * m.UP)

        self.say(
            "[Now take a coin.] [A fair coin has one bit of entropy], [the most a coin can have.]",
            m.AnimationGroup(
                self.new_header("ENTROPY", "A fair coin and a loaded coin"),
                m.FadeIn(the_coin, scale=0.8),
                m.FadeIn(rows), m.FadeIn(tracks), m.FadeIn(heads_bar), m.FadeIn(tails_bar), m.FadeIn(heads_val), m.FadeIn(tails_val),
            ),
            m.AnimationGroup(m.FadeIn(h_name), m.FadeIn(h_val), m.Create(ax), m.FadeIn(ticks), m.FadeIn(x_name), m.FadeIn(y_name), m.FadeIn(drop), m.FadeIn(dot, scale=0.5)),
            m.Flash(ax.c2p(0.5, 1), color=AMBER, flash_radius=0.35),
        )
        self.say(
            "[Load it to land heads nine times in ten], [and the entropy drops to just under half a bit.]",
            m.AnimationGroup(b.animate(run_time=2.0).set_value(0.9), m.Rotate(the_coin, angle=2 * m.PI, run_time=2.0)),
            m.AnimationGroup(m.Indicate(h_val, color=AMBER, scale_factor=1.1), m.FadeIn(loaded, shift=0.15 * m.UP)),
        )
        self.say(
            "[Sweep the chance of heads from zero to one], [and the entropy traces a hill]: [highest at one half, where the flip is hardest to predict.]",
            m.Succession(m.FadeOut(loaded, run_time=0.3), b.animate(run_time=0.7).set_value(0.0)),
            m.AnimationGroup(m.Create(hill, rate_func=m.linear, run_time=2.6), b.animate(run_time=2.6, rate_func=m.linear).set_value(1.0)),
            m.AnimationGroup(b.animate(run_time=1.2).set_value(0.5), m.FadeIn(peak, shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.stop(heads_bar, tails_bar, heads_val, tails_val, h_val, dot, drop)
        self.clear()

    # ------------------------------------------------------------------ cross-entropy

    def cross_entropy(self):
        self.next_section(
            "Cross-entropy",
            notes="A model's chances, for example q = 0.56, 0.28, 0.14, 0.02. H(p, q) = -sum p log2 q = 1.94 bits; roof costs 5.64 bits.",
        )
        unit, base = 0.72, -2.6
        bars = chart_bars(S_P, unit, base, AMBER, AMBER_FILL)
        bars.set_z_index(1)
        q_bars = chart_bars(S_Q, unit, base, VIOLET, VIOLET_FILL).set_z_index(1)
        floor_line = m.Line([CH_X0 - 0.1, base, 0], [CH_X0 + CH_W + 0.1, base, 0]).set_stroke(FAINT, width=3)
        words, fracs = chart_labels(base)
        h_y = base + H_P * unit
        h_line = m.DashedLine([CH_X0, h_y, 0], [CH_X0 + CH_W, h_y, 0], dash_length=0.12).set_stroke(AMBER, width=4).set_z_index(3)
        h_label = text("entropy: 1.75 bits", 22, AMBER, weight=m.BOLD).move_to([CH_X0 + 1.75, h_y - 0.27, 0]).set_z_index(3)
        x_y = base + H_PQ * unit
        level = level_bars(H_PQ, unit, base, VIOLET, opacity=0.3)
        x_line = m.DashedLine([CH_X0, x_y, 0], [CH_X0 + CH_W, x_y, 0], dash_length=0.12).set_stroke(VIOLET, width=4).set_z_index(3)
        x_label = text("cross-entropy: 1.94 bits", 22, VIOLET, weight=m.BOLD).move_to([CH_X0 + 1.75, x_y + 0.3, 0]).set_z_index(3)
        roof_label = text("5.64 bits", 24, VIOLET, weight=m.BOLD).next_to(q_bars[3].get_corner(m.UL), m.LEFT, buff=0.18).shift(0.3 * m.DOWN)

        # the table: world p, model q, the model's surprise
        cols = (1.5, 3.05, 4.5, 5.98)
        rows_y = (0.35, -0.3, -0.95, -1.6)
        heads = m.VGroup(
            text("word", 22, MUTED, weight=m.SEMIBOLD).move_to([cols[0], 1.0, 0]),
            text("world p", 22, TEAL, weight=m.BOLD).move_to([cols[1], 1.0, 0]),
            text("model q", 22, VIOLET, weight=m.BOLD).move_to([cols[2], 1.0, 0]),
            tex(r"-\log_2 q", 28, VIOLET).move_to([cols[3], 1.0, 0]),
        )
        rule = m.Line([0.9, 0.68, 0], [6.75, 0.68, 0]).set_stroke(EDGE, width=2)
        names = m.VGroup(*[text(w, 24, INK, weight=m.SEMIBOLD).move_to([cols[0], y, 0]) for w, y in zip(WORDS, rows_y)])
        ps = m.VGroup(*[code(f, 24, TEAL).move_to([cols[1], y, 0]) for f, y in zip(FRACS, rows_y)])
        qs = m.VGroup(*[code(f"{q:.2f}", 24, VIOLET).move_to([cols[2], y, 0]) for q, y in zip(Q, rows_y)])
        sq = m.VGroup(*[code(f"{s:.2f}", 24, VIOLET).move_to([cols[3], y, 0]) for s, y in zip(S_Q, rows_y)])
        note = text("q: an example model's chances", 22, MUTED).move_to([3.7, -2.35, 0])

        formula = m.MathTex(r"H(p, q)", "=", "-", r"\sum_x", r"p(x)", r"\log_2 q(x)", font_size=44, color=INK)
        formula[0].set_color(VIOLET)
        formula[4].set_color(TEAL)
        formula[5].set_color(VIOLET)
        formula.move_to([3.75, 2.0, 0])

        self.say(
            "[Now a model tries to guess the word], [with its own chances, which we'll call q.]",
            m.AnimationGroup(
                self.new_header("CROSS-ENTROPY", "Believing q, living in p"),
                m.Create(floor_line), m.FadeIn(words), m.FadeIn(fracs), m.FadeIn(bars), m.Create(h_line), m.FadeIn(h_label),
            ),
            m.AnimationGroup(m.FadeIn(heads), m.Create(rule), m.FadeIn(names), m.FadeIn(ps), m.LaggedStart(*[m.FadeIn(q, shift=0.15 * m.LEFT) for q in qs], lag_ratio=0.15), m.FadeIn(note)),
        )
        self.say(
            "[Say it's a little too sure of mat, sofa and floor], [and gives roof just two chances in a hundred.]",
            m.LaggedStart(*[m.Indicate(q, color=VIOLET, scale_factor=1.2) for q in qs[:3]], lag_ratio=0.25),
            m.Indicate(qs[3], color=BAD, scale_factor=1.35),
        )
        self.say(
            "[Cross-entropy is your average surprise when you believe q], [but the words really come from p.]",
            m.Write(formula, run_time=1.6),
            m.Indicate(formula[4], color=TEAL, scale_factor=1.2),
        )
        self.say(
            "[Each word is still weighted by its real chance], [but its surprise now comes from the model.]",
            m.AnimationGroup(m.Indicate(fracs, color=TEAL, scale_factor=1.15), m.Indicate(ps, color=TEAL, scale_factor=1.15)),
            m.AnimationGroup(
                m.Indicate(m.VGroup(formula[2], formula[5]), color=VIOLET, scale_factor=1.12),
                *[m.Transform(a, b) for a, b in zip(bars, q_bars)],
                m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.LEFT) for s in sq], lag_ratio=0.15),
                run_time=1.8,
            ),
        )
        self.say(
            "[So when roof turns up, the model is shocked]: [over five and a half bits.]",
            m.Indicate(bars[3], color=BAD, scale_factor=1.05),
            m.AnimationGroup(m.FadeIn(roof_label, shift=0.15 * m.UP), m.Indicate(sq[3], color=BAD, scale_factor=1.3)),
        )
        self.say(
            "[On average, that's about one point nine four bits], [more than the one point seven five of the true chances.]",
            m.AnimationGroup(
                *[m.TransformFromCopy(b, c) for b, c in zip(bars, level)],
                bars.animate.set_fill(opacity=0.0).set_stroke(opacity=0.5),
                m.Create(x_line),
                m.FadeIn(x_label, shift=0.15 * m.UP),
                run_time=1.8,
            ),
            m.Indicate(h_label, color=AMBER, scale_factor=1.15),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ KL divergence

    def divergence(self):
        self.next_section(
            "KL divergence",
            notes="D_KL(p||q) = H(p, q) - H(p) = 0.19 bits >= 0; zero only if q = p; swapped, D_KL(q||p) = 0.11 bits. Kullback & Leibler, 1951.",
        )
        V, x0 = 3.0, -3.2
        rows_y = (0.4, -0.7)
        formula = m.MathTex(r"D_{\mathrm{KL}}(p \,\|\, q)", "=", r"H(p, q)", "-", r"H(p)", r"\ge 0", font_size=48, color=INK)
        formula[0].set_color(BAD)
        formula[2].set_color(VIOLET)
        formula[4].set_color(AMBER)
        formula.move_to([0, 1.9, 0])
        source = cite("Kullback & Leibler, 1951").next_to(formula, m.DOWN, buff=0.2).align_to(formula, m.RIGHT)
        names = m.VGroup(
            text("entropy", 26, AMBER, weight=m.BOLD).move_to([0, rows_y[0], 0]),
            text("cross-entropy", 26, VIOLET, weight=m.BOLD).move_to([0, rows_y[1], 0]),
        )
        for n in names:
            n.align_to([x0 - 0.3, 0, 0], m.RIGHT)
        h_bar = m.Rectangle(width=H_P * V, height=0.62).set_stroke(AMBER, width=2.5).set_fill(AMBER_FILL, opacity=1)
        h_bar.move_to([x0, rows_y[0], 0], aligned_edge=m.LEFT)
        h_val = text("1.75 bits", 26, AMBER, weight=m.BOLD).next_to(h_bar, m.RIGHT, buff=0.25)
        shared = m.Rectangle(width=H_P * V, height=0.62).set_stroke(VIOLET, width=2.5).set_fill(VIOLET_FILL, opacity=1)
        shared.move_to([x0, rows_y[1], 0], aligned_edge=m.LEFT)
        edge_x = x0 + H_P * V
        guide = m.DashedLine([edge_x, rows_y[0] - 0.31, 0], [edge_x, rows_y[1] + 0.31, 0], dash_length=0.08).set_stroke(MUTED, width=2.5)

        t = m.ValueTracker(0.0)  # 0: the model's q, 1: q equal to p

        def q_now():
            a = t.get_value()
            return (1 - a) * Q + a * P

        def gap():
            return max(kl(P, q_now()), 0.0)

        def make_extra():
            w = max(gap() * V, 0.004)
            return m.Rectangle(width=w, height=0.62).set_stroke(BAD, width=2.5).set_fill(BAD_FILL, opacity=1).move_to([edge_x, rows_y[1], 0], aligned_edge=m.LEFT)

        extra = m.always_redraw(make_extra)
        x_val = m.always_redraw(lambda: text(f"{H_P + gap():.2f} bits", 26, VIOLET, weight=m.BOLD).move_to([edge_x + gap() * V + 0.25, rows_y[1], 0], aligned_edge=m.LEFT))

        def make_tag():
            w = gap() * V
            y = rows_y[1] - 0.48
            mark = m.VMobject().set_points_as_corners([[edge_x, y + 0.12, 0], [edge_x, y, 0], [edge_x + w, y, 0], [edge_x + w, y + 0.12, 0]])
            mark.set_stroke(BAD, width=3)
            words = text(f"KL ≈ {gap():.2f} bits", 24, BAD, weight=m.BOLD).move_to([edge_x + w / 2, y - 0.33, 0])
            return m.VGroup(mark, words)

        tag = m.always_redraw(make_tag)
        price = text("extra bits per word, paid for wrong beliefs", 22, MUTED).move_to([0, -1.95, 0]).align_to([x0, 0, 0], m.LEFT)

        one = m.VGroup(tex(r"D_{\mathrm{KL}}(p \,\|\, q) \approx 0.19", 38, BAD), text("bits", 24, MUTED)).arrange(m.RIGHT, buff=0.2)
        two = m.VGroup(tex(r"D_{\mathrm{KL}}(q \,\|\, p) \approx 0.11", 38, BAD), text("bits", 24, MUTED)).arrange(m.RIGHT, buff=0.2)
        swap = m.VGroup(one, two).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([0, -2.65, 0]).align_to([x0, 0, 0], m.LEFT)
        not_sym = pill("not symmetric", BAD, BAD_FILL, 24).move_to([4.6, -2.65, 0])

        self.say(
            "[That extra is called the KL divergence]: [cross-entropy, minus entropy.]",
            m.AnimationGroup(
                self.new_header("KL DIVERGENCE", "The price of wrong beliefs"),
                m.FadeIn(names), m.FadeIn(h_bar), m.FadeIn(h_val), m.FadeIn(shared), m.FadeIn(extra), m.FadeIn(x_val), m.Create(guide),
            ),
            m.AnimationGroup(m.Write(formula[:5], run_time=1.4), m.FadeIn(source)),
        )
        self.say(
            "[Here it's about zero point one nine bits]: [the price of the model's wrong beliefs.]",
            m.FadeIn(tag, shift=0.15 * m.DOWN),
            m.FadeIn(price, shift=0.15 * m.UP),
        )
        self.say(
            "[It's never negative], [and it's zero only when the model's chances match the world's exactly.]",
            m.FadeIn(formula[5], shift=0.2 * m.LEFT),
            t.animate(run_time=2.6).set_value(1.0),
        )
        self.say(
            "[But it isn't symmetric.] [Swap p and q], [and you get about zero point one one bits instead.]",
            m.AnimationGroup(t.animate(run_time=1.4).set_value(0.0), m.FadeOut(price)),
            m.FadeIn(one, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(two, shift=0.15 * m.UP), m.FadeIn(not_sym, scale=1.15)),
        )
        self.wait(0.3)
        self.stop(extra, x_val, tag)
        self.clear()

    # ------------------------------------------------------------------ perplexity

    def perplexity(self):
        self.next_section(
            "Perplexity",
            notes="perplexity = 2^H (H in bits): a fair coin 2, our sentence 2^1.75 = 3.36, the model 2^1.94 = 3.83. Jelinek et al., 1977.",
        )
        slot, x0, y0 = 1.5, -6.1, -0.05
        h = m.ValueTracker(1.0)
        formula = m.MathTex(r"\text{perplexity}", "=", r"2^{H}", font_size=52, color=INK)
        formula[0].set_color(SKY)
        formula[2].set_color(AMBER)
        formula.move_to([-1.4, 1.95, 0])
        h_note = text("H: the entropy in bits", 22, MUTED).next_to(formula, m.RIGHT, buff=0.6)
        slots = m.VGroup(*[m.Rectangle(width=slot, height=0.95).set_stroke(EDGE, width=2.5).set_fill(CARD, opacity=1) for _ in range(5)])
        slots.arrange(m.RIGHT, buff=0).move_to([x0, y0, 0], aligned_edge=m.LEFT)
        counts = m.VGroup(*[code(str(k), 20, MUTED).move_to([x0 + slot * k, y0 - 0.75, 0]) for k in range(1, 6)])
        choices = text("equally likely choices", 22, SKY).move_to([x0 + 2.5 * slot, y0 - 1.2, 0])

        def make_fill():
            w = slot * 2 ** h.get_value()
            return m.Rectangle(width=w, height=0.95, stroke_width=0).set_fill(SKY, opacity=0.6).move_to([x0, y0, 0], aligned_edge=m.LEFT).set_z_index(1)

        fill = m.always_redraw(make_fill)
        edge = m.always_redraw(lambda: m.Line([x0 + slot * 2 ** h.get_value(), y0 - 0.55, 0], [x0 + slot * 2 ** h.get_value(), y0 + 0.5, 0]).set_stroke(SKY, width=5).set_z_index(2))
        rx = 2.75
        r_names = m.VGroup(
            text("entropy", 24, AMBER, weight=m.SEMIBOLD).move_to([0, 0.35, 0]),
            text("perplexity", 24, SKY, weight=m.SEMIBOLD).move_to([0, -0.45, 0]),
        )
        for r in r_names:
            r.align_to([rx, 0, 0], m.LEFT)
        cross_name = text("cross-entropy", 24, VIOLET, weight=m.SEMIBOLD).move_to(r_names[0]).align_to([rx, 0, 0], m.LEFT)
        h_val = m.always_redraw(lambda: text(f"{h.get_value():.2f} bits", 28, VIOLET if h.get_value() > H_P + 0.02 else AMBER, weight=m.BOLD).move_to([6.6, 0.35, 0], aligned_edge=m.RIGHT))
        p_val = m.always_redraw(lambda: text(f"{2 ** h.get_value():.2f}", 28, SKY, weight=m.BOLD).move_to([6.6, -0.45, 0], aligned_edge=m.RIGHT))
        who = [text(s, 28, INK, weight=m.BOLD).move_to([rx, 1.05, 0], aligned_edge=m.LEFT) for s in ("A fair coin", "Our sentence", "The model")]
        tick_marks = m.VGroup()
        for value, color in ((1.0, MUTED), (H_P, AMBER), (H_PQ, VIOLET)):
            x = x0 + slot * 2**value
            tick_marks.add(m.Triangle(fill_color=color, fill_opacity=1, stroke_width=0).scale(0.12).rotate(m.PI).move_to([x, y0 + 0.68, 0]))
        effective = pill("perplexity: the effective number of equally likely choices", SKY, SKY_FILL, 22).move_to([-1.0, -2.35, 0])
        model_note = text("for a model, H is its cross-entropy", 22, MUTED).move_to([-1.0, -3.15, 0])

        self.say(
            "[Bits are hard to picture, so we often turn them into choices]: [perplexity is two to the power of the entropy.]",
            m.AnimationGroup(self.new_header("PERPLEXITY", "Bits as choices"), m.FadeIn(slots), m.FadeIn(counts), m.FadeIn(choices)),
            m.AnimationGroup(m.Write(formula), m.FadeIn(h_note)),
        )
        self.say(
            "[A fair coin has one bit]: [two to the one is two], [a choice between two equally likely sides.]",
            m.AnimationGroup(m.FadeIn(who[0], shift=0.15 * m.UP), m.FadeIn(r_names), m.FadeIn(h_val)),
            m.AnimationGroup(m.FadeIn(p_val), m.FadeIn(fill), m.FadeIn(edge)),
            m.AnimationGroup(m.FadeIn(tick_marks[0], shift=0.1 * m.UP), m.Indicate(m.VGroup(slots[0], slots[1]), color=SKY, scale_factor=1.04)),
        )
        self.say(
            "[Our sentence has one point seven five bits]: [a perplexity of about three point four], [like picking among three or four equally likely words.]",
            m.AnimationGroup(m.FadeTransform(who[0], who[1]), h.animate(run_time=1.6).set_value(H_P)),
            m.AnimationGroup(m.Indicate(p_val, color=SKY, scale_factor=1.15), m.FadeIn(tick_marks[1], shift=0.1 * m.UP)),
            m.FadeIn(effective, shift=0.2 * m.UP),
        )
        self.say(
            "[The model's one point nine four bits give about three point eight]: [it's more perplexed than it needs to be.]",
            m.AnimationGroup(
                m.FadeTransform(who[1], who[2]), m.FadeTransform(r_names[0], cross_name),
                h.animate(run_time=1.4).set_value(H_PQ), m.FadeIn(model_note),
            ),
            m.AnimationGroup(m.FadeIn(tick_marks[2], shift=0.1 * m.UP), m.Indicate(tick_marks[1], color=AMBER, scale_factor=1.5)),
        )
        self.wait(0.3)
        self.stop(fill, edge, h_val, p_val)
        self.clear()

    # ------------------------------------------------------------------ language models

    def language_models(self):
        self.next_section(
            "Language models",
            notes="For example, chances 0.02, 0.20, 0.50, 0.70, 0.56 for the real next words: mean -ln = 1.43 nats per token; perplexity e^1.43 = 4.2. Training lowers cross-entropy; H of the text is fixed, so it lowers the KL gap.",
        )
        chips = m.VGroup(*[chip(w, SKY if k else EDGE, SKY_FILL if k else CARD, width=1.45, height=0.72, size=28) for k, w in enumerate(TOKENS)])
        chips.arrange(m.RIGHT, buff=0.32).move_to([0, 1.8, 0])
        on_baseline(m.VGroup(*[c[1] for c in chips]))
        xs = [c.get_x() for c in chips]
        chance_name = text("model's chance", 22, VIOLET, weight=m.SEMIBOLD).move_to([xs[0], 0.95, 0])
        chance_vals = m.VGroup(*[code(f"{c:.2f}", 24, VIOLET).move_to([x, 0.95, 0]) for c, x in zip(CHANCES, xs[1:])])
        base, unit = -2.5, 0.62
        bars = m.VGroup(
            *[
                m.Rectangle(width=0.9, height=n * unit).set_stroke(AMBER, width=2.5).set_fill(AMBER_FILL, opacity=1).move_to([x, base + n * unit / 2, 0])
                for n, x in zip(NATS, xs[1:])
            ]
        )
        floor_line = m.Line([xs[1] - 0.75, base, 0], [xs[-1] + 0.75, base, 0]).set_stroke(FAINT, width=3)
        nat_vals = m.VGroup(*[code(f"{n:.2f}", 22, AMBER).next_to(b, m.UP, buff=0.12) for n, b in zip(NATS, bars)])
        nat_name = m.VGroup(text("surprise", 22, AMBER, weight=m.SEMIBOLD), text("in nats", 22, AMBER)).arrange(m.DOWN, buff=0.08).move_to([xs[0], -1.4, 0])
        avg_y = base + AVG_NATS * unit
        avg_line = m.DashedLine([xs[1] - 0.6, avg_y, 0], [xs[-1] + 0.6, avg_y, 0], dash_length=0.12).set_stroke(AMBER, width=4).set_z_index(3)
        avg_label = text("average: 1.43 nats", 22, AMBER, weight=m.BOLD).move_to([xs[4] + 0.85, avg_y + 0.3, 0])
        result = tex(r"\text{perplexity} = e^{1.43} \approx 4.2", 40, SKY).move_to([-0.6, -3.15, 0])
        source = cite("Jelinek et al., 1977").move_to([4.6, -3.2, 0])

        # training: the text's entropy is fixed, the gap shrinks
        g = m.ValueTracker(2.0)
        sx0, sy, fixed_w = -5.6, -0.75, 6.2
        fixed = m.Rectangle(width=fixed_w, height=0.75).set_stroke(AMBER, width=2.5).set_fill(AMBER_FILL, opacity=1).move_to([sx0, sy, 0], aligned_edge=m.LEFT)
        fixed_name = text("the text's own entropy: fixed", 24, AMBER, weight=m.BOLD).next_to(fixed, m.UP, buff=0.25)
        gap_bar = m.always_redraw(
            lambda: m.Rectangle(width=max(g.get_value(), 0.01), height=0.75).set_stroke(BAD, width=2.5).set_fill(BAD_FILL, opacity=1).move_to([sx0 + fixed_w, sy, 0], aligned_edge=m.LEFT)
        )
        gap_name = m.always_redraw(lambda: text("KL divergence: shrinks", 24, BAD, weight=m.BOLD).next_to([sx0 + fixed_w + g.get_value(), sy, 0], m.RIGHT, buff=0.3))

        def make_total():
            right = sx0 + fixed_w + g.get_value()
            y = sy - 0.6
            mark = m.VMobject().set_points_as_corners([[sx0, y + 0.14, 0], [sx0, y, 0], [right, y, 0], [right, y + 0.14, 0]]).set_stroke(VIOLET, width=3)
            words = text("the model's loss: cross-entropy", 24, VIOLET, weight=m.BOLD).move_to([(sx0 + right) / 2, y - 0.38, 0])
            return m.VGroup(mark, words)

        total = m.always_redraw(make_total)

        self.say(
            "[A language model plays this guessing game at every token]: [it gives a chance to each real next word, given the words before it.]",
            m.AnimationGroup(self.new_header("LANGUAGE MODELS", "Perplexity in practice"), m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in chips], lag_ratio=0.12)),
            m.AnimationGroup(m.FadeIn(chance_name), m.LaggedStart(*[m.FadeIn(v, shift=0.15 * m.DOWN) for v in chance_vals], lag_ratio=0.2)),
        )
        self.say(
            "[Average its surprise using the natural log, in units called nats], [and e to that average is its perplexity]: [here, about four point two.]",
            m.AnimationGroup(
                m.Create(floor_line), m.FadeIn(nat_name),
                m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(b, m.DOWN), m.FadeIn(v)) for b, v in zip(bars, nat_vals)], lag_ratio=0.2),
                m.Create(avg_line), m.FadeIn(avg_label),
            ),
            m.AnimationGroup(m.Write(result), m.FadeIn(source)),
            m.Circumscribe(result, color=SKY, buff=0.12),
        )
        self.say(
            "[Training pushes that number down.] [The text's own entropy can't change], [so what training really shrinks is the KL divergence.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (chance_name, chance_vals, bars, nat_vals, nat_name, floor_line, avg_line, avg_label, result, source)],
                m.FadeIn(fixed), m.FadeIn(fixed_name), m.FadeIn(gap_bar), m.FadeIn(gap_name), m.FadeIn(total),
            ),
            m.Indicate(fixed_name, color=AMBER, scale_factor=1.08),
            g.animate(run_time=2.4).set_value(0.45),
        )
        self.wait(0.3)
        self.stop(gap_bar, gap_name, total)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Surprise, average, gap.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Surprise", "Average", "Gap"), (AMBER, TEAL, BAD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        subs = m.VGroup(
            *[text(s, 22, MUTED).move_to([w.get_x(), min(v.get_bottom()[1] for v in words) - 0.35, 0]) for s, w in zip(("−log p", "entropy", "KL divergence"), words)]
        )
        line = subs[1][0].get_bottom()[1]
        for sub in subs:  # one baseline: line up the bottoms of the letters without descenders
            sitting = [g for g, c in zip(sub, sub.text.replace(" ", "")) if c not in "gjpqy−"]
            sub.shift((line - min(g.get_bottom()[1] for g in sitting)) * m.UP)
        tagline = text("Information is surprise. A good model is rarely surprised by real text.", 30, MUTED).move_to([0, -1.9, 0])

        self.say(
            "[To sum up]: [surprise], [average], [gap.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Information is surprise, and a good model is one that real text rarely surprises.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, subs, tagline)], run_time=0.9)


