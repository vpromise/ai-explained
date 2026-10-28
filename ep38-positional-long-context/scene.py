"""AI, Explained · Episode 38 · Positional Encodings & Long Context

Running example: a 100-page contract (about 65,000 tokens at ~650 tokens a page) read by a model
trained on 4,096-token sequences, so the contract is 16 times longer than anything it trained on.

Arc: attention is order-blind (Buyer pays Seller = Seller pays Buyer, computed) → absolute
positions: sinusoids (Vaswani et al., 2017) as clock hands, and learned tables that end at their
last row → RoPE (Su et al., 2021): rotate each pair of query/key dimensions by position × θ_i →
the score depends only on the offset (computed: same score 40 pages later) → past the training
length the slow pairs reach angles never seen → position interpolation (Chen et al., 2023) →
NTK-aware scaling (bloc97, 2023) and YaRN (Peng et al., 2023), per-pair scale factors computed →
lost in the middle (Liu et al., 2023), schematic → recap.

Colours, all film long: words / inputs and the query SKY · the key AMBER · positions, angles and
clock hands PINK (also the level accent) · seen in training GOOD · never seen BAD · position
interpolation VIOLET · NTK-aware BLUE · YaRN TEAL.

Every number the narration quotes is computed below and asserted. The RoPE geometry is the real
one for a 128-dimensional head with base 10,000 (64 pairs, θ_i = 10000^(−2i/128)); the query and
key vectors and the toy attention layer are seeded random examples. The lost-in-the-middle curve
is a schematic of the shape Liu et al. report, not their data.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

BLUE_FILL = "#142040"

# ---------------------------------------------------------------------------- the numbers

# the contract and the model
PAGES, TOKENS, PER_PAGE = 100, 65_000, 650
assert PAGES * PER_PAGE == TOKENS  # "roughly sixty-five thousand tokens"
L_TRAIN, SCALE = 4096, 16
L_NEW = L_TRAIN * SCALE  # 65,536: "sixteen times longer"
assert abs(L_NEW - TOKENS) / TOKENS < 0.01
SHIFT = 40 * PER_PAGE  # "forty pages later" = 26,000 tokens

# RoPE in a typical head: d = 128, base 10,000, 64 pairs
D, BASE = 128, 10000.0
THETA = BASE ** (-np.arange(0, D, 2) / D)
WAVE = 2 * np.pi / THETA  # tokens per full turn of each pair
assert len(THETA) == 64 and THETA[0] == 1.0
NEVER = int((WAVE > L_TRAIN).sum())
assert NEVER == 18  # "eighteen of the sixty-four pairs never finish a single turn"
SLOW = float(THETA[-1])
SLOW_TRAINED = L_TRAIN * SLOW  # radians the slowest pair turns in training
assert round(np.degrees(SLOW_TRAINED)) == 27  # "the slowest moves just twenty-seven degrees"
FAST_TURNS = L_TRAIN / WAVE[0]
assert 100 < FAST_TURNS < 1000 and round(FAST_TURNS) == 652  # "hundreds of times"
assert round(WAVE[-1]) == 54410 and round(WAVE[0], 1) == 6.3
assert round(np.degrees(THETA[0])) == 57 and np.degrees(THETA[0] / SCALE) < 4  # "under four degrees ... fifty-seven"
DIAL_THETAS = (THETA[0], THETA[16], THETA[32], THETA[48])  # the four clock hands
assert np.allclose(DIAL_THETAS, (1, 0.1, 0.01, 0.001))


def rope(x, pos):
    """Rotate every pair (x[2i], x[2i+1]) by pos * THETA[i]."""
    pairs = x.reshape(-1, 2)
    a = pos * THETA
    c, s = np.cos(a), np.sin(a)
    return np.stack([pairs[:, 0] * c - pairs[:, 1] * s, pairs[:, 0] * s + pairs[:, 1] * c], 1).ravel()


_rng = np.random.default_rng(9)
Q = _rng.normal(size=D)
K = 0.6 * Q + 0.8 * _rng.normal(size=D)


def score(mq, nk):
    return float(rope(Q, mq) @ rope(K, nk) / np.sqrt(D))


M_PAYS, N_BUYER = 12, 10  # "pays", and "Buyer" two words earlier
NEAR = score(M_PAYS, N_BUYER)
FAR = score(M_PAYS + SHIFT, N_BUYER + SHIFT)
OTHER = score(40, N_BUYER)
assert abs(NEAR - FAR) < 1e-9  # the same score forty pages later
assert round(NEAR, 2) == 7.76 and round(FAR, 2) == 7.76  # "seven point seven six ... again"
assert round(OTHER, 2) == 5.24 and abs(OTHER - NEAR) > 1  # "a different offset gives a different score"
Q_FAST, Q_SLOW, K_SLOW = Q[:2], Q[-2:], K[-2:]  # pair 1 for the rotation demo; pair 64 for the offset demo


def slow_dot(shift):
    return float(rope(Q, M_PAYS + shift)[-2:] @ rope(K, N_BUYER + shift)[-2:])


assert abs(slow_dot(0) - slow_dot(SHIFT)) < 1e-9

# order-blind attention: one toy layer (no positions, no mask), seeded
WORDS = ("Buyer", "pays", "Seller")
CLAUSE_A, CLAUSE_B = ("Buyer", "pays", "Seller"), ("Seller", "pays", "Buyer")
_r2 = np.random.default_rng(38)
EMB = {w: _r2.normal(size=6) for w in WORDS}
WQ, WK, WV = (_r2.normal(size=(6, 6)) / np.sqrt(6) for _ in range(3))


def attend(X):
    s = (X @ WQ) @ (X @ WK).T / np.sqrt(6)
    a = np.exp(s - s.max(1, keepdims=True))
    return (a / a.sum(1, keepdims=True)) @ (X @ WV)


def sinusoid(pos, d=6):
    i = np.arange(d // 2)
    out = np.zeros(d)
    out[0::2], out[1::2] = np.sin(pos / 10000 ** (2 * i / d)), np.cos(pos / 10000 ** (2 * i / d))
    return out


XA = np.array([EMB[w] for w in CLAUSE_A])
XB = np.array([EMB[w] for w in CLAUSE_B])
OUT_A, OUT_B = attend(XA), attend(XB)
SHUFFLE = np.eye(3)[[2, 1, 0]]
assert np.allclose(XB, SHUFFLE @ XA) and np.allclose(attend(SHUFFLE @ XA), SHUFFLE @ attend(XA))  # Attn(PX) = P Attn(X)
for _i, _w in enumerate(CLAUSE_A):
    assert np.allclose(OUT_A[_i], OUT_B[CLAUSE_B.index(_w)])  # "each word comes out exactly the same"
_PE = np.array([sinusoid(p) for p in range(3)])
assert not np.allclose(attend(XA + _PE)[0], attend(XB + _PE)[2])  # with positions added, they differ

# context extension: how much each pair's rotation speed is scaled (s = 16)
PI_MULT = np.full(64, 1 / SCALE)
NTK_BASE = BASE * SCALE ** (D / (D - 2))
NTK_MULT = NTK_BASE ** (-np.arange(0, D, 2) / D) / THETA
assert np.isclose(NTK_MULT[0], 1) and np.isclose(NTK_MULT[-1], 1 / SCALE)  # "fast pairs barely change ... slowest the full sixteen"
ALPHA, BETA = 1, 32  # YaRN's ramp (Peng et al., 2023, the values used for LLaMA)
RATIO = L_TRAIN / WAVE  # turns per training window
GAMMA = np.clip((RATIO - ALPHA) / (BETA - ALPHA), 0, 1)
YARN_MULT = (1 - GAMMA) / SCALE + GAMMA
UNTOUCHED, INTERPOLATED = int((GAMMA == 1).sum()), int((GAMMA == 0).sum())
assert UNTOUCHED == 21 and INTERPOLATED == NEVER  # fast pairs alone, the never-turning pairs fully interpolated
CROSS = 64 * np.log10(L_TRAIN / (2 * np.pi)) / 4  # pair index where one turn = 4,096 tokens
YARN_TEMP = (0.1 * np.log(SCALE) + 1) ** 2  # logits multiplied by this: a slightly sharper softmax
assert 1 < YARN_TEMP < 2


def lost(x):
    """Schematic U shape (after Liu et al., 2023): best at the start, good at the end, worst mid-way."""
    return 0.56 + 0.8 * (x - 0.52) ** 2 + 0.03 * (1 - x)


assert lost(0) > lost(1) > lost(0.5)

# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def cite(s):
    return text(s, 20, MUTED)


def unit(a):
    return np.array([np.cos(a), np.sin(a), 0.0])


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


def arc(center, radius, a0, a1, color, width=9, opacity=1.0):
    span = a1 - a0
    mob = m.Arc(radius=radius, start_angle=a0, angle=max(span, 1e-3), arc_center=np.array(center, dtype=float))
    mob.set_fill(opacity=0).set_stroke(color, width=width, opacity=opacity if span > 1e-3 else 0)
    return mob


def vec(origin, v, color, width=6):
    return m.Arrow(origin, origin + v, buff=0, stroke_width=width, max_tip_length_to_length_ratio=0.22).set_color(color)


class Dial:
    """A clock face: a ring, a hub, hands at angles (radians, counter-clockwise from the right)."""

    def __init__(self, center, radius):
        self.c = np.array([center[0], center[1], 0.0])
        self.r = radius
        self.ring = m.Circle(radius=radius).set_stroke(EDGE, width=3).set_fill(opacity=0).move_to(self.c)
        self.hub = m.Dot(self.c, radius=0.07, color=INK).set_z_index(4)

    def at(self, angle, frac=1.0):
        return self.c + frac * self.r * unit(angle)

    def hand(self, angle, color=PINK, width=6, frac=0.86, opacity=1.0):
        return m.Line(self.c, self.at(angle, frac)).set_stroke(color, width=width, opacity=opacity).set_z_index(3)

    def arc(self, a0, a1, color, width=9, frac=1.0, opacity=1.0):
        return arc(self.c, self.r * frac, a0, a1, color, width, opacity)


def page_card(w, h, title=None, rows=7):
    """A contract page: a card, ruled text lines, a signature line."""
    box = card(w, h, EDGE, CARD, radius=0.12 if w > 1 else 0.07, stroke=2)
    parts = [box]
    top, left, right = box.get_top()[1], box.get_left()[0], box.get_right()[0]
    pad = 0.12 * w
    first = top - (0.75 if title else 0.18 * h)
    if title:
        parts.append(text(title, 20, MUTED, weight=m.BOLD).move_to([box.get_x(), top - 0.38, 0]))
    step = (first - (box.get_bottom()[1] + 0.3 * h)) / max(rows - 1, 1)
    for k in range(rows):
        end = right - pad - (0.35 * w if k % 3 == 2 else 0)
        parts.append(m.Line([left + pad, first - k * step, 0], [end, first - k * step, 0]).set_stroke(FAINT, width=4 if w > 1 else 2.5))
    sig_y = box.get_bottom()[1] + 0.12 * h
    parts.append(m.Line([right - pad - 0.4 * w, sig_y, 0], [right - pad, sig_y, 0]).set_stroke(PINK, width=4 if w > 1 else 2.5))
    return m.VGroup(*parts)


def mini_bars(values, color, center, scale):
    """A word's output vector as a little bar chart around a baseline."""
    bars = m.VGroup()
    n = len(values)
    for k, v in enumerate(values):
        h = max(abs(v) * scale, 0.03)
        x = center[0] + (k - (n - 1) / 2) * 0.2
        y = center[1] + (h / 2 if v > 0 else -h / 2)
        bars.add(m.Rectangle(width=0.14, height=h).set_stroke(width=0).set_fill(color, opacity=0.95).move_to([x, y, 0]))
    base = m.Line([center[0] - 0.62, center[1], 0], [center[0] + 0.62, center[1], 0]).set_stroke(FAINT, width=2)
    return m.VGroup(base, bars)


def row_of(*parts, buff=0.2):
    return m.VGroup(*parts).arrange(m.RIGHT, buff=buff)


# ---------------------------------------------------------------------------- the episode


class Ep38PositionalLongContext(Episode):
    number = 38

    def lesson(self):
        self.hook()
        self.problem()
        self.absolute()
        self.rotary()
        self.offset()
        self.limit()
        self.interpolation()
        self.scaling()
        self.middle()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes="A 100-page contract: ~500 words a page, ~1.3 tokens a word, so roughly 65,000 tokens "
            "(illustrative estimate). Two questions: word order, and reading past the training length.",
        )
        stack = m.VGroup(*[page_card(2.3, 3.0) for _ in range(4)], page_card(2.3, 3.0, title="CONTRACT"))
        for k, p in enumerate(stack):
            p.move_to([-3.75 + 0.17 * k, 0.55 - 0.14 * k, 0])
        pages = text("100 pages", 66, INK, weight=m.BOLD).move_to([2.6, 1.05, 0])
        count = m.DecimalNumber(TOKENS, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=44, color=PINK)
        room = code("65,000", 44, PINK)  # the counter's final width, to place "tokens" after it
        tokens = row_of(text("≈", 44, PINK, weight=m.BOLD), room, text("tokens", 32, MUTED), buff=0.3)
        tokens.move_to([2.6, -0.1, 0])
        count.move_to(room).align_to(room, m.LEFT)
        tokens.remove(room)
        tokens.insert(1, count)
        on_baseline(m.VGroup(tokens[1], tokens[2]))
        tally = m.ValueTracker(0)
        count.set_value(0)
        count.add_updater(lambda d: d.set_value(int(round(tally.get_value()))))
        questions = m.VGroup(
            pill("1 · What order are the words in?", PINK, PINK_FILL, 24),
            pill("2 · Can it read past its training length?", PINK, PINK_FILL, 24),
        ).arrange(m.RIGHT, buff=0.45).move_to([0, -2.75, 0])

        self.say(
            "So... [imagine handing a language model a hundred-page contract.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.UP) for p in stack], lag_ratio=0.15), m.FadeIn(pages, shift=0.2 * m.UP)),
        )
        self.say(
            "[That's roughly sixty-five thousand tokens.]",
            m.AnimationGroup(m.FadeIn(tokens), tally.animate(run_time=1.6, rate_func=m.smooth).set_value(TOKENS)),
        )
        self.stop(count)
        self.say(
            "To read it, the model needs two things: [it must know the order of the words], [and it must read further than it was trained to.]",
            m.FadeIn(questions[0], shift=0.2 * m.UP),
            m.FadeIn(questions[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the problem

    def problem(self):
        self.next_section(
            "Order-blind attention",
            notes="One toy attention layer (6 dimensions, seeded random weights, no positions, no causal mask). "
            "Clauses 'Buyer pays Seller' and 'Seller pays Buyer': every word's output is identical (asserted), "
            "since Attn(PX) = P·Attn(X) for any row permutation P. With sinusoidal positions added the outputs differ (asserted).",
        )
        xs = (-4.7, -2.85, -1.0)
        ya, yb = 1.5, -0.95
        scale = 0.36 / float(np.abs(np.vstack([OUT_A, OUT_B])).max())

        def chips(words, y):
            return m.VGroup(*[chip(w, SKY, SKY_FILL, width=1.55, height=0.68, size=28).move_to([x, y, 0]) for w, x in zip(words, xs)])

        def links(row):
            tops = [c.get_top() + 0.07 * m.UP for c in row]
            return m.VGroup(*[m.ArcBetweenPoints(tops[i], tops[j], angle=-1.0).set_stroke(MUTED, width=2.5, opacity=0.8) for i, j in ((0, 1), (1, 2), (0, 2))])

        def outputs(out, y):
            return m.VGroup(*[mini_bars(v, SKY, (x, y), scale) for v, x in zip(out, xs)])

        row_a, row_b = chips(CLAUSE_A, ya), chips(CLAUSE_B, yb)
        arcs_a, arcs_b = links(row_a), links(row_b)
        bars_a, bars_b = outputs(OUT_A, ya - 0.98), outputs(OUT_B, yb - 0.98)
        labels = m.VGroup(
            text("clause 1", 20, MUTED).move_to([-6.25, ya, 0]),
            text("output", 20, MUTED).move_to([-6.25, ya - 0.98, 0]),
            text("clause 2", 20, MUTED).move_to([-6.25, yb, 0]),
            text("output", 20, MUTED).move_to([-6.25, yb - 0.98, 0]),
        )
        tint = {"Buyer": BLUE, "pays": MUTED, "Seller": LIME}

        def frames(words, y):
            return m.VGroup(
                *[m.RoundedRectangle(corner_radius=0.1, width=1.45, height=0.9).set_stroke(tint[w], width=3).set_fill(opacity=0).move_to([x, y, 0]) for w, x in zip(words, xs)]
            )

        frames_a, frames_b = frames(CLAUSE_A, ya - 0.98), frames(CLAUSE_B, yb - 0.98)
        same = text("each word: the same output in both clauses", 22, INK, weight=m.SEMIBOLD).move_to([-2.85, -3.0, 0])

        law = tex(r"\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)", 46).move_to([3.75, 1.3, 0])
        terms = m.VGroup(
            row_of(tex("X", 36, SKY), text("the word vectors, one row each", 22, MUTED), buff=0.3),
            row_of(tex("P", 36, PINK), text("a shuffle of the rows", 22, MUTED), buff=0.3),
        ).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([3.75, 0.05, 0])
        terms[1][0].set_x(terms[0][0].get_x())
        terms[1][1].align_to(terms[0][1], m.LEFT)
        aside = text("no position signal, no causal mask", 20, MUTED).move_to([3.75, -0.9, 0])
        verdict = pill("Who pays whom? Invisible.", BAD, BAD_FILL, 26).move_to([3.75, -2.0, 0])

        self.say(
            "[First, the problem.] [Take a clause: Buyer pays Seller.]",
            self.new_header("THE PROBLEM", "Attention is blind to order"),
            m.AnimationGroup(m.FadeIn(labels[0]), m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in row_a], lag_ratio=0.25)),
        )
        self.say(
            "[Self-attention mixes each word with all the others], [and turns it into a new vector.]",
            m.Create(arcs_a, run_time=1.2),
            m.AnimationGroup(m.FadeIn(labels[1]), m.LaggedStart(*[m.FadeIn(b, shift=0.15 * m.DOWN) for b in bars_a], lag_ratio=0.2)),
        )
        self.say(
            "[Now swap the parties: Seller pays Buyer.] [Same words, opposite obligation.]",
            m.AnimationGroup(m.FadeIn(labels[2]), *[m.FadeIn(row_b[CLAUSE_B.index(w)], target_position=row_a[i], path_arc=0.6) for i, w in enumerate(CLAUSE_A)]),
            m.AnimationGroup(m.Create(arcs_b), m.FadeIn(labels[3]), m.LaggedStart(*[m.FadeIn(b, shift=0.15 * m.DOWN) for b in bars_b], lag_ratio=0.2)),
        )
        self.say(
            "[Yet without any position signal, each word comes out exactly the same in both clauses.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.Create(frames_a[i]), m.Create(frames_b[CLAUSE_B.index(w)])) for i, w in enumerate(CLAUSE_A)], lag_ratio=0.4),
                m.FadeIn(same, shift=0.15 * m.UP),
            ),
        )
        self.say(
            "[In symbols]: [shuffle the input rows with P, and the outputs are shuffled the same way.]",
            m.Write(law, run_time=1.3),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in terms], lag_ratio=0.4), m.FadeIn(aside)),
        )
        self.say("[So on its own, the model can't tell who pays whom.]", m.FadeIn(verdict, scale=1.2))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ absolute positions

    def absolute(self):
        self.next_section(
            "Absolute positions",
            notes="Sinusoidal encodings (Vaswani et al., 2017): PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...), "
            "added to the token embedding. The four dials are pairs 1, 17, 33, 49 of a 128-wide model: 1, 0.1, 0.01, 0.001 "
            "radians per token. Learned absolute tables: BERT (Devlin et al., 2019, 512 positions), GPT-2 (Radford et al., 2019, "
            "1,024); our illustrative model has 4,096 rows.",
        )
        word = chip("pays", SKY, SKY_FILL, width=1.35, height=0.62, size=26)
        where = chip("position 12", PINK, PINK_FILL, width=2.2, height=0.62, size=26)
        stamp = row_of(word, text("+", 34, INK, weight=m.BOLD), where, buff=0.3).move_to([-3.45, 1.95, 0])

        f_sin = tex(r"\mathrm{PE}_{\text{pos},\,2i} = \sin\!\big(\text{pos} / 10000^{2i/d}\big)", 30)
        f_cos = tex(r"\mathrm{PE}_{\text{pos},\,2i+1} = \cos\!\big(\text{pos} / 10000^{2i/d}\big)", 30)
        pe = m.VGroup(f_sin, f_cos).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([-3.45, 0.95, 0])
        legend = text("pos: position  ·  i: which pair  ·  d: vector width", 20, MUTED).move_to([-3.45, 0.12, 0])

        dials = [Dial((x, -0.95), 0.5) for x in (-5.55, -4.15, -2.75, -1.35)]
        pos = m.ValueTracker(0)
        hands = [m.always_redraw(lambda d=d, w=w: d.hand(pos.get_value() * w, PINK, width=5)) for d, w in zip(dials, DIAL_THETAS)]
        faces = m.VGroup(*[m.VGroup(d.ring, d.hub) for d in dials])
        speeds = m.VGroup(*[code(s, 22, MUTED).move_to([d.c[0], -1.72, 0]) for d, s in zip(dials, ("1", "0.1", "0.01", "0.001"))])
        speeds.add(text("radians per token", 20, MUTED).move_to([-3.45, -2.08, 0]))
        readout_n = m.DecimalNumber(60, num_decimal_places=0, mob_class=Mono, font_size=28, color=PINK)
        readout = row_of(code("pos =", 26, PINK), readout_n, buff=0.18).move_to([-3.45, -2.68, 0])
        readout_n.set_value(0)
        readout_n.add_updater(lambda d: d.set_value(int(round(pos.get_value()))))
        vaswani = cite("Vaswani et al., 2017").move_to([-3.45, -3.25, 0])

        table = card(5.4, 4.75, EDGE, CARD, stroke=2).move_to([3.65, -0.45, 0])
        t_title = text("Learned table", 28, INK, weight=m.BOLD).move_to([3.65, 1.5, 0])
        t_sub = text("one vector per position (BERT, GPT-2)", 20, MUTED).move_to([3.65, 1.05, 0])
        rng = np.random.default_rng(5)
        rows = m.VGroup()
        for label, y in (("pos 1", 0.45), ("pos 2", -0.05), ("pos 3", -0.55), ("pos 4,096", -1.45)):
            cells = m.VGroup(*[m.Square(0.32).set_stroke(EDGE, width=1.5).set_fill(mix(CARD, PINK, v), opacity=1) for v in rng.uniform(0.15, 0.9, 8)])
            cells.arrange(m.RIGHT, buff=0.06).move_to([4.45, y, 0])
            rows.add(m.VGroup(code(label, 20, PINK).next_to(cells, m.LEFT, buff=0.3), cells))
        for r in rows:
            r[0].align_to(rows[3][0], m.RIGHT)
        dots = text("⋮", 26, MUTED).move_to([rows[0][1].get_x(), -0.98, 0])
        empty = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.08, width=rows[0][1].width, height=0.42).set_stroke(BAD, width=2), num_dashes=28)
        empty.move_to([rows[0][1].get_x(), -2.2, 0])
        missing = m.VGroup(code("pos 4,097", 20, BAD).next_to(empty, m.LEFT, buff=0.3).align_to(rows[3][0], m.RIGHT), empty)
        no_row = text("no row", 22, BAD, weight=m.BOLD).move_to(empty).set_z_index(2)
        learned_cite = cite("Devlin et al., 2019 · Radford et al., 2019").move_to([3.65, -3.25, 0])

        self.say(
            "[The original Transformer fixed this by adding a position signal to each word's vector.]",
            m.AnimationGroup(
                self.new_header("ABSOLUTE POSITIONS", "Stamp each word with its address"),
                m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.UP) for p in stamp], lag_ratio=0.3),
                m.FadeIn(vaswani),
            ),
        )
        self.say(
            "[Each pair of dimensions gets a sine and a cosine of the position], [like a clock hand turning at its own speed.]",
            m.AnimationGroup(m.Write(pe, run_time=1.6), m.FadeIn(legend)),
            m.AnimationGroup(m.FadeIn(faces), *[m.FadeIn(h) for h in hands], m.FadeIn(speeds), m.FadeIn(readout)),
        )
        self.say(
            "[As we read along], [fast hands tell neighbouring words apart, while slow hands change only over many pages.]",
            pos.animate(run_time=4.0, rate_func=m.linear).set_value(60),
            m.AnimationGroup(m.Circumscribe(m.VGroup(dials[0].ring, speeds[0]), color=PINK, buff=0.08), m.Circumscribe(m.VGroup(dials[3].ring, speeds[3]), color=PINK, buff=0.08)),
        )
        self.stop(*hands, readout_n)
        self.say(
            "[Many later models simply learned one vector per position instead, stored in a table.]",
            m.AnimationGroup(
                m.FadeIn(table),
                m.FadeIn(t_title, shift=0.15 * m.UP),
                m.FadeIn(t_sub),
                m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.LEFT) for r in rows[:3]], m.FadeIn(dots), m.FadeIn(rows[3], shift=0.15 * m.LEFT), lag_ratio=0.25),
                m.FadeIn(learned_cite),
            ),
        )
        self.say(
            "[But a table has a last row.] [Train on four thousand and ninety-six positions, and the next one has no vector at all.]",
            m.Indicate(rows[3][0], color=PINK, scale_factor=1.15),
            m.AnimationGroup(m.FadeIn(missing, shift=0.15 * m.DOWN), m.FadeIn(no_row, scale=1.3)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ RoPE: rotate

    def rotary(self):
        self.next_section(
            "RoPE: rotate",
            notes="RoPE (Su et al., 2021, RoFormer): split q and k into pairs; rotate pair i at position m by m·θ_i, "
            "θ_i = 10000^(−2i/d). The arrow is pair 1 (θ = 1 radian) of a seeded random query; m runs 0 → 3.",
        )
        centre = np.array([-4.0, -0.75, 0.0])
        self.centre = centre
        ring = m.Circle(radius=1.9).set_stroke(EDGE, width=2.5).set_fill(opacity=0).move_to(centre)
        cross = m.VGroup(
            m.Line(centre + 2.2 * m.LEFT, centre + 2.2 * m.RIGHT).set_stroke(FAINT, width=2),
            m.Line(centre + 2.2 * m.DOWN, centre + 2.2 * m.UP).set_stroke(FAINT, width=2),
        )
        self.plane = m.VGroup(ring, cross)

        values = Q[:6]
        boxes = m.VGroup()
        for k in range(3):
            words = code(f"{values[2 * k]:.1f} {values[2 * k + 1]:.1f}".replace("-", "−"), 22, INK)
            box = card(words.width + 0.32, 0.58, PINK if k == 0 else EDGE, PINK_FILL if k == 0 else CARD, radius=0.14, stroke=2)
            boxes.add(m.VGroup(box, words.move_to(box)))
        more = m.VGroup(*[m.Dot(radius=0.04, color=MUTED) for _ in range(3)]).arrange(m.RIGHT, buff=0.1)
        vector_row = row_of(tex("q =", 34, SKY), *boxes, more, buff=0.18).move_to([0, 1.95, 0]).align_to([-6.55, 0, 0], m.LEFT)

        scale = 2.0
        start = float(np.arctan2(Q_FAST[1], Q_FAST[0]))
        length = scale * float(np.linalg.norm(Q_FAST))
        mpos = m.ValueTracker(0)
        still = vec(centre, length * unit(start), SKY)
        ghost = vec(centre, length * unit(start), SKY).set_opacity(0.25)
        turning = m.always_redraw(lambda: vec(centre, length * unit(start + mpos.get_value()), SKY))
        swept = m.always_redraw(lambda: arc(centre, 0.62, start, start + mpos.get_value(), PINK, width=5))
        m_num = m.DecimalNumber(3, num_decimal_places=0, mob_class=Mono, font_size=30, color=PINK)
        m_read = row_of(tex("m =", 34, PINK), m_num, buff=0.15).move_to([-0.85, 0.85, 0])
        m_num.set_value(0)
        m_num.add_updater(lambda d: d.set_value(int(round(mpos.get_value()))))
        angle_label = m.always_redraw(
            lambda: tex(r"m\theta", 30, PINK).move_to(centre + 0.98 * unit(start + mpos.get_value() / 2)).set_opacity(1 if mpos.get_value() > 0.4 else 0)
        )

        rot = tex(r"R(m\theta) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}", 36).move_to([3.55, 1.35, 0])
        terms = m.VGroup(
            row_of(tex("m", 34, PINK), text("the word's position", 22, MUTED), buff=0.3),
            row_of(tex(r"\theta", 34, PINK), text("this pair's angle per step", 22, MUTED), buff=0.3),
        ).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([3.3, 0.0, 0])
        speeds = tex(r"\theta_i = 10000^{-2i/d}", 36, INK).move_to([3.55, -1.25, 0])
        speeds_note = text("fast pairs to slow pairs, the sinusoids' frequencies", 20, MUTED).next_to(speeds, m.DOWN, buff=0.22)
        su = cite("Su et al., 2021").move_to([3.55, -2.75, 0])
        self.rope_parts = dict(vector_row=vector_row, rot=rot, terms=terms, speeds=speeds, speeds_note=speeds_note, su=su, m_read=m_read)

        self.say(
            "[Rotary position embedding, or RoPE, takes another route]: [it rotates the query and key vectors instead of adding to them.]",
            m.AnimationGroup(self.new_header("ROTARY", "RoPE: rotate instead of add"), m.FadeIn(su)),
            m.Create(self.plane, run_time=1.2),
        )
        self.say(
            "[Split a vector into pairs of numbers], [and read each pair as an arrow in a plane.]",
            m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in vector_row], lag_ratio=0.15),
            m.FadeIn(still, target_position=boxes[0], scale=0.6),
        )
        self.add(ghost)
        self.remove(still)
        self.add(turning, swept, angle_label)
        self.say(
            "[A word at position m has its arrow turned by m times an angle, theta.] [Here, three steps of one radian.]",
            m.AnimationGroup(m.Write(rot, run_time=1.6), m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in terms], lag_ratio=0.4), m.FadeIn(m_read)),
            mpos.animate(run_time=2.4, rate_func=m.smooth).set_value(3),
        )
        self.stop(turning, swept, angle_label, m_num)
        self.say(
            "[Each pair has its own theta, from fast to slow, with the same frequencies as the sinusoids.]",
            m.AnimationGroup(m.Write(speeds), m.FadeIn(speeds_note, shift=0.15 * m.UP)),
        )
        self.rope_demo = m.VGroup(ghost, turning, swept, angle_label)
        self.wait(0.3)

    # ------------------------------------------------------------------ only the offset matters

    def offset(self):
        self.next_section(
            "Only the offset matters",
            notes=f"<R(mθ)q, R(nθ)k> = <q, R((n−m)θ)k>. Plane: pair 64 (θ ≈ {SLOW:.6f}) of seeded random q, k; "
            f"'pays' at m = 12, 'Buyer' at n = 10, slid by 26,000 tokens (40 pages at 650 tokens a page). "
            f"Full head (d = 128, scores q·k/√d): (12, 10) → {NEAR:.4f}; (26,012, 26,010) → {FAR:.4f}; (40, 10) → {OTHER:.4f}.",
        )
        parts = self.rope_parts
        centre = self.centre
        scale = 1.0
        shift = m.ValueTracker(0)
        q_start, k_start = (float(np.arctan2(v[1], v[0])) for v in (Q_SLOW, K_SLOW))
        q_len, k_len = scale * float(np.linalg.norm(Q_SLOW)), scale * float(np.linalg.norm(K_SLOW))

        def q_angle():
            return q_start + (M_PAYS + shift.get_value()) * SLOW

        def k_angle():
            return k_start + (N_BUYER + shift.get_value()) * SLOW

        q_arrow = m.always_redraw(lambda: vec(centre, q_len * unit(q_angle()), SKY))
        k_arrow = m.always_redraw(lambda: vec(centre, k_len * unit(k_angle()), AMBER))
        q_tag = m.always_redraw(lambda: tex("q", 34, SKY).move_to(centre + (q_len + 0.3) * unit(q_angle())))
        k_tag = m.always_redraw(lambda: tex("k", 34, AMBER).move_to(centre + (k_len + 0.3) * unit(k_angle())))
        between = m.always_redraw(lambda: arc(centre, 0.42, q_angle(), q_angle() + ((k_angle() - q_angle()) % (2 * np.pi)), INK, width=4))
        q_ghost = vec(centre, q_len * unit(q_angle()), SKY).set_opacity(0.25)
        k_ghost = vec(centre, k_len * unit(k_angle()), AMBER).set_opacity(0.25)

        m_num = m.DecimalNumber(M_PAYS + SHIFT, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=26, color=SKY)
        n_num = m.DecimalNumber(N_BUYER + SHIFT, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=26, color=AMBER)
        m_row = row_of(text("pays", 24, SKY, weight=m.SEMIBOLD), tex("m =", 32, SKY), m_num, buff=0.15)
        n_row = row_of(text("Buyer", 24, AMBER, weight=m.SEMIBOLD), tex("n =", 32, AMBER), n_num, buff=0.15)
        positions = m.VGroup(m_row, n_row).arrange(m.RIGHT, buff=0.7).move_to([-3.45, 1.95, 0])
        on_baseline(m.VGroup(m_row[0], m_row[2], n_row[0], n_row[2]))
        m_num.set_value(M_PAYS)
        n_num.set_value(N_BUYER)
        m_num.add_updater(lambda d: d.set_value(int(round(M_PAYS + shift.get_value()))))
        n_num.add_updater(lambda d: d.set_value(int(round(N_BUYER + shift.get_value()))))
        which = text("pair 64 of 64: the slowest", 20, MUTED).move_to([-4.0, -3.3, 0])

        dot_num = m.DecimalNumber(slow_dot(0), num_decimal_places=2, mob_class=Mono, font_size=36, color=INK)
        dot_num.add_updater(lambda d: d.set_value(slow_dot(shift.get_value())))
        dot_box = m.VGroup(text("this pair's", 20, MUTED), text("dot product", 20, MUTED), dot_num).arrange(m.DOWN, buff=0.12).move_to([-0.7, -0.45, 0])
        later = m.VGroup(text("+ 40 pages", 26, PINK, weight=m.BOLD), text("26,000 tokens", 20, MUTED)).arrange(m.DOWN, buff=0.1).move_to([-0.7, 1.05, 0])

        top = tex(r"\langle R(m\theta)\,q,\; R(n\theta)\,k \rangle", 36).move_to([3.4, 1.75, 0])
        bottom = tex(r"= \langle q,\; R\big((n-m)\theta\big)\,k \rangle", 36).next_to(top, m.DOWN, buff=0.28).align_to(top, m.LEFT).shift(0.35 * m.RIGHT)

        panel = card(5.6, 2.55, EDGE, CARD, stroke=2).move_to([3.55, -1.25, 0])
        p_title = text("full head: 128 dimensions", 24, INK, weight=m.BOLD).move_to([3.55, -0.3, 0])
        cols = (1.85, 3.45, 5.3)
        heads = m.VGroup(*[text(s, 24, MUTED, weight=m.SEMIBOLD).move_to([x, -0.78, 0]) for s, x in zip(("pays m", "Buyer n", "score"), cols)])
        rows = []
        for k, (mq, nk, sc, color) in enumerate(((M_PAYS, N_BUYER, NEAR, INK), (M_PAYS + SHIFT, N_BUYER + SHIFT, FAR, GOOD), (40, N_BUYER, OTHER, AMBER))):
            y = -1.25 - 0.47 * k
            rows.append(
                m.VGroup(
                    code(f"{mq:,}", 24, SKY).move_to([cols[0], y, 0]),
                    code(f"{nk:,}", 24, AMBER).move_to([cols[1], y, 0]),
                    code(f"{sc:.2f}", 24, color).move_to([cols[2], y, 0]),
                )
            )
        point = pill("the score depends only on n − m", GOOD, GOOD_FILL, 22).move_to([3.55, -3.2, 0])

        self.say(
            "[Now the clever part.] [Here's the query for pays, and the key for Buyer, two words earlier.]",
            m.AnimationGroup(
                self.new_header("RELATIVE", "Only the offset matters"),
                *[m.FadeOut(parts[k]) for k in ("vector_row", "rot", "terms", "speeds", "speeds_note", "m_read")],
                m.FadeOut(self.rope_demo),
                m.FadeOut(parts["su"]),
            ),
            m.AnimationGroup(m.GrowArrow(q_arrow), m.GrowArrow(k_arrow), m.FadeIn(q_tag), m.FadeIn(k_tag), m.FadeIn(positions), m.FadeIn(which)),
        )
        self.say(
            "[Their attention score is the dot product of the two rotated arrows.]",
            m.AnimationGroup(m.Create(between), m.FadeIn(dot_box, shift=0.15 * m.UP)),
        )
        self.add(q_ghost, k_ghost)
        self.say(
            "[Slide the clause forty pages later.] [Both arrows turn by the same extra angle], [so the angle between them, and the score, stay the same.]",
            m.FadeIn(later, shift=0.15 * m.DOWN),
            shift.animate(run_time=3.2, rate_func=m.smooth).set_value(SHIFT),
            m.Circumscribe(dot_box, color=GOOD, buff=0.12),
        )
        self.stop(q_arrow, k_arrow, q_tag, k_tag, between, m_num, n_num, dot_num)
        self.say(
            "Rotations compose: [turning q by m and k by n] [is the same as turning k alone by n minus m.]",
            m.Write(top, run_time=1.4),
            m.Succession(m.Write(bottom, run_time=1.4), m.Circumscribe(bottom, color=PINK, buff=0.1)),
        )
        self.say(
            "[Check a full head of a hundred and twenty-eight dimensions.] [Near the start, the score is seven point seven six.] [Forty pages later: seven point seven six again.]",
            m.AnimationGroup(m.FadeIn(panel), m.FadeIn(p_title), m.FadeIn(heads)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.LEFT) for c in rows[0]], lag_ratio=0.3),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.LEFT) for c in rows[1]], lag_ratio=0.3),
        )
        self.say(
            "[A different offset gives a different score.] [Attention now sees distance, not address.]",
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.LEFT) for c in rows[2]], lag_ratio=0.3),
            m.FadeIn(point, scale=1.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ past the training length

    def limit(self):
        self.next_section(
            "Past the training length",
            notes=f"Illustrative model trained on 4,096 tokens (as Llama 2); contract 65,536 = 16 × 4,096. Head d = 128, base 10,000: "
            f"pair 1 turns once every {WAVE[0]:.1f} tokens ({FAST_TURNS:.0f} turns in 4,096); pair 64 once every {WAVE[-1]:,.0f} tokens "
            f"({np.degrees(SLOW_TRAINED):.1f}° in 4,096, {np.degrees(L_NEW * SLOW):.0f}° by 65,536). {NEVER} of 64 pairs never finish a turn. "
            "Extrapolating RoPE past the trained length can give catastrophically large attention scores (Chen et al., 2023).",
        )
        x0, x1, by = -6.0, 6.0, 1.75

        def bx(p):
            return x0 + (x1 - x0) * p / L_NEW

        contract = m.Rectangle(width=x1 - x0, height=0.32).set_stroke(BAD, width=2).set_fill(BAD_FILL, opacity=1).move_to([0, by, 0])
        trained = m.Rectangle(width=bx(L_TRAIN) - x0, height=0.32).set_stroke(GOOD, width=2).set_fill(GOOD, opacity=0.85)
        trained.move_to([(x0 + bx(L_TRAIN)) / 2, by, 0]).set_z_index(1)
        t_label = text("trained on 4,096 tokens", 22, GOOD, weight=m.SEMIBOLD).move_to([0, by + 0.45, 0]).align_to([x0, 0, 0], m.LEFT)
        c_label = text("our contract: 16 × longer", 22, BAD, weight=m.SEMIBOLD).move_to([0, by + 0.45, 0]).align_to([x1, 0, 0], m.RIGHT)
        pos = m.ValueTracker(0)
        self.pos, self.bx, self.by = pos, bx, by
        tick = m.always_redraw(lambda: m.Line([bx(pos.get_value()), by - 0.27, 0], [bx(pos.get_value()), by + 0.27, 0]).set_stroke(PINK, width=5).set_z_index(3))
        tick_num = m.DecimalNumber(0, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=22, color=PINK)
        tick_num.add_updater(lambda d: d.set_value(int(round(pos.get_value()))).move_to([bx(pos.get_value()), by - 0.5, 0]))

        fast, slow = Dial((-3.6, -0.9), 0.95), Dial((3.6, -0.9), 0.95)
        self.fast, self.slow = fast, slow
        titles = m.VGroup(
            text("fastest pair", 26, INK, weight=m.BOLD).move_to([-3.6, 0.82, 0]),
            text("slowest pair", 26, INK, weight=m.BOLD).move_to([3.6, 0.82, 0]),
        )
        subs = m.VGroup(
            text("one turn every 6.3 tokens", 20, MUTED).move_to([-3.6, 0.42, 0]),
            text("one turn every 54,410 tokens", 20, MUTED).move_to([3.6, 0.42, 0]),
        )
        spin = m.ValueTracker(0)
        fast_ring = fast.arc(0, 2 * np.pi - 1e-3, GOOD, width=9)
        fast_hand = m.always_redraw(lambda: fast.hand(spin.get_value()))
        fast_note = text("≈ 652 turns during training", 22, GOOD, weight=m.SEMIBOLD).move_to([-3.6, -2.25, 0])

        def slow_angle():
            return pos.get_value() * SLOW

        slow_seen = m.always_redraw(lambda: slow.arc(0, min(slow_angle(), SLOW_TRAINED), GOOD, width=9))
        slow_new = m.always_redraw(lambda: slow.arc(SLOW_TRAINED, min(slow_angle(), 2 * np.pi), BAD, width=9))
        slow_hand = m.always_redraw(lambda: slow.hand(slow_angle()))
        slow_note = text("27° during training", 22, GOOD, weight=m.SEMIBOLD).move_to([3.6, -2.25, 0])
        never = text("18 of 64 pairs never finish a turn", 20, MUTED).move_to([3.6, -2.65, 0])
        broken = m.VGroup(pill("new angles: attention scores can blow up", BAD, BAD_FILL, 22), cite("Chen et al., 2023")).arrange(m.RIGHT, buff=0.3)
        broken.move_to([0, -3.3, 0])
        self.limit_parts = dict(
            contract=contract, trained=trained, c_label=c_label, tick=tick, tick_num=tick_num, fast_ring=fast_ring, fast_hand=fast_hand,
            fast_note=fast_note, slow_seen=slow_seen, slow_new=slow_new, slow_hand=slow_hand, broken=broken, slow_note=slow_note, never=never,
        )

        self.say(
            "[Now, the length problem.] [Say the model was trained on four thousand tokens], [and our contract is sixteen times longer.]",
            self.new_header("THE LIMIT", "Past the training length"),
            m.AnimationGroup(m.FadeIn(trained), m.FadeIn(t_label)),
            m.AnimationGroup(m.GrowFromEdge(contract, m.LEFT), m.FadeIn(c_label)),
        )
        self.say(
            "[The fastest pairs are fine]: [they spun around hundreds of times during training.]",
            m.AnimationGroup(m.FadeIn(titles[0]), m.FadeIn(subs[0]), m.Create(fast.ring), m.FadeIn(fast.hub), m.FadeIn(fast_hand)),
            m.AnimationGroup(spin.animate(run_time=1.8, rate_func=m.smooth).set_value(4 * np.pi), m.Create(fast_ring, run_time=1.8), m.FadeIn(fast_note)),
        )
        self.stop(fast_hand)
        self.add(slow_new, tick, tick_num)
        self.say(
            "[But in a typical head, eighteen of the sixty-four pairs never finish a single turn.] [The slowest moves just twenty-seven degrees.]",
            m.AnimationGroup(m.FadeIn(titles[1]), m.FadeIn(subs[1]), m.Create(slow.ring), m.FadeIn(slow.hub), m.FadeIn(slow_hand), m.FadeIn(slow_seen), m.FadeIn(never)),
            m.AnimationGroup(pos.animate(run_time=2.0, rate_func=m.smooth).set_value(L_TRAIN), m.FadeIn(slow_note)),
        )
        self.say(
            "[Read on through the contract, and it swings into angles the model has never seen.] [Attention scores can blow up, and the output falls apart.]",
            pos.animate(run_time=3.5, rate_func=m.smooth).set_value(L_NEW),
            m.FadeIn(broken, shift=0.2 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ position interpolation

    def interpolation(self):
        self.next_section(
            "Position interpolation",
            notes="Position interpolation (Chen et al., 2023): m' = m·L/L' = m/16, so every position lands inside the trained range; "
            "they extended LLaMA (2,048) to up to 32,768 tokens with a short fine-tune (within ~1,000 steps). Cost: adjacent tokens' "
            "angle in the fastest pair shrinks from 1 radian (57.3°) to 1/16 radian (3.6°).",
        )
        p = self.limit_parts
        bx, by, pos, fast = self.bx, self.by, self.pos, self.fast
        squeezed = m.Rectangle(width=bx(L_TRAIN) - bx(0), height=0.32).set_stroke(GOOD, width=2).set_fill(GOOD_FILL, opacity=1)
        squeezed.move_to([(bx(0) + bx(L_TRAIN)) / 2, by, 0])
        c_new = text("all 65,536 positions, squeezed into 4,096", 22, VIOLET, weight=m.SEMIBOLD).move_to(p["c_label"]).align_to(p["c_label"], m.RIGHT)

        rule = tex(r"m' = m \cdot \frac{L}{L'}", 44).move_to([0, -0.45, 0])
        legend = m.VGroup(
            row_of(tex("L", 30, GOOD), code("= 4,096 trained", 20, MUTED), buff=0.18),
            row_of(tex("L'", 30, BAD), code("= 65,536 needed", 20, MUTED), buff=0.18),
        ).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([0, -1.55, 0])
        chen = cite("Chen et al., 2023").move_to([0, 0.42, 0])
        result = pill("LLaMA: 2,048 → 32,768 tokens after a short fine-tune", VIOLET, VIOLET_FILL, 22).move_to([0, -3.3, 0])

        gap = m.ValueTracker(THETA[0])
        nxt = m.always_redraw(lambda: fast.hand(gap.get_value(), PINK, width=5, frac=0.86, opacity=0.6))
        wedge = m.always_redraw(lambda: fast.arc(0, gap.get_value(), INK, width=4, frac=0.45))
        deg57 = text("57°", 24, INK, weight=m.BOLD).move_to(fast.at(0.5, 1.4))
        deg36 = text("3.6°", 24, BAD, weight=m.BOLD).move_to(fast.at(0.08, 1.45))
        fast_new = text("neighbouring words, one step apart", 22, MUTED, weight=m.SEMIBOLD).move_to(p["fast_note"])

        self.say(
            "[The first fix is position interpolation]: [divide every position by sixteen.]",
            m.AnimationGroup(self.new_header("FIX 1", "Position interpolation"), m.FadeOut(p["broken"]), m.FadeIn(chen)),
            m.AnimationGroup(m.Write(rule), m.FadeIn(legend, shift=0.15 * m.UP)),
        )
        self.say(
            "[The whole contract now fits inside the trained range], [so the slow hand stays on familiar angles.]",
            m.AnimationGroup(m.Transform(p["contract"], squeezed), m.FadeTransform(p["c_label"], c_new), run_time=1.6),
            pos.animate(run_time=3.0, rate_func=m.smooth).set_value(L_NEW / SCALE),
        )
        self.stop(p["tick"], p["tick_num"], p["slow_seen"], p["slow_new"], p["slow_hand"])
        self.say(
            "[With a short fine-tune, Chen and colleagues stretched Llama from two thousand to thirty-two thousand tokens.]",
            m.FadeIn(result, shift=0.2 * m.UP),
        )
        self.add(wedge)
        self.say(
            "[The catch: fast pairs get squeezed too.] [Neighbouring words end up under four degrees apart, instead of fifty-seven.]",
            m.AnimationGroup(m.FadeOut(p["fast_ring"]), m.FadeIn(nxt), m.FadeIn(deg57), m.FadeTransform(p["fast_note"], fast_new)),
            m.Succession(gap.animate(run_time=1.8, rate_func=m.smooth).set_value(THETA[0] / SCALE), m.FadeTransform(deg57, deg36, run_time=0.6)),
        )
        self.stop(nxt, wedge)
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ NTK-aware and YaRN

    def scaling(self):
        self.next_section(
            "NTK-aware scaling and YaRN",
            notes=f"Per-pair speed multipliers for s = 16, d = 128, base 10,000, computed here. PI: all ×1/16. NTK-aware (bloc97, 2023, "
            f"via Peng et al., 2023): base b' = b·s^(d/(d−2)) = {NTK_BASE:,.0f}, so pair 1 ×1 and pair 64 ×1/16 exactly. YaRN (Peng et al., "
            f"2023): ramp on turns-per-window r = L/λ with α = 1, β = 32: {UNTOUCHED} fast pairs untouched, {INTERPOLATED} slow pairs fully "
            f"interpolated, plus attention logits × (0.1·ln s + 1)² = {YARN_TEMP:.2f}. YaRN extended Llama 2 (4k) to 64k and 128k contexts "
            "with a short fine-tune. Facts checked 2026-10-21 against the arXiv abstracts.",
        )
        lift = 4.6  # chart height = log2(multiplier) + lift, so the x-axis sits below every curve
        chart = axes([0, 70, 8], [0, 5.4, 1], 7.2, 4.0).move_to([-2.15, -0.5, 0])

        def at(i, log_mult):
            return chart.c2p(i, log_mult + lift)

        y_ticks = m.VGroup(*[text(s, 22, MUTED).next_to(at(0, v), m.LEFT, buff=0.18) for s, v in (("×1", 0), ("×1/4", -2), ("×1/16", -4))])
        grid = m.VGroup(*[m.DashedLine(at(0, v), at(64, v), dash_length=0.06).set_stroke(FAINT, width=1.5, opacity=0.6) for v in (0, -2, -4)])
        y_title = text("rotation speed after scaling", 22, MUTED).next_to(chart.y_axis.get_end(), m.RIGHT, buff=0.2)
        x_ticks = m.VGroup(
            text("pair 1", 20, MUTED).next_to(chart.c2p(0, 0), m.DOWN, buff=0.15),
            text("pair 64", 20, MUTED).next_to(chart.c2p(63, 0), m.DOWN, buff=0.15).align_to(chart.c2p(63, 0) + 0.12 * m.RIGHT, m.RIGHT),
        )
        x_title = text("fast pairs  →  slow pairs", 22, MUTED).next_to(chart.c2p(32, 0), m.DOWN, buff=0.15)
        cross = m.DashedLine(chart.c2p(CROSS, 0), at(CROSS, 0.35), dash_length=0.08).set_stroke(MUTED, width=2.5)
        cross_label = m.VGroup(text("one turn =", 20, MUTED), text("4,096 tokens", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        cross_label.next_to(at(CROSS, -0.6), m.RIGHT, buff=0.15)

        def curve(mult, color, a=0, b=64):
            return polyline([at(i, np.log2(v)) for i, v in zip(range(a, b), mult[a:b])], color, width=5)

        pi_curve = curve(PI_MULT, VIOLET)
        ntk_curve = curve(NTK_MULT, BLUE)
        first_ramp = int(np.argmax(GAMMA < 1))
        last_ramp = int(np.argmax(GAMMA == 0))
        yarn_parts = (curve(YARN_MULT, TEAL, 0, first_ramp), curve(YARN_MULT, TEAL, first_ramp - 1, last_ramp + 1), curve(YARN_MULT, TEAL, last_ramp, 64))
        for c in yarn_parts:
            c.set_stroke(width=6).set_z_index(2)

        def key(color, label):
            return row_of(m.Line(m.ORIGIN, 0.5 * m.RIGHT).set_stroke(color, width=6), text(label, 24, color, weight=m.SEMIBOLD), buff=0.2)

        keys = m.VGroup(key(VIOLET, "position interpolation"), key(BLUE, "NTK-aware"), key(TEAL, "YaRN")).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        keys.move_to([4.6, 1.3, 0]).align_to([2.6, 0, 0], m.LEFT)
        rule = tex(r"b' = b \cdot s^{d/(d-2)}", 40).move_to([4.65, -0.35, 0])
        rule_terms = code("b = 10,000  s = 16  d = 128", 20, MUTED).next_to(rule, m.DOWN, buff=0.25)
        cites = m.VGroup(cite("NTK-aware: bloc97, 2023"), cite("YaRN: Peng et al., 2023")).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        cites.move_to([4.65, -2.0, 0]).align_to(keys, m.LEFT)
        sharper = pill("+ a slightly sharper softmax", TEAL, TEAL_FILL, 22)
        reach = pill("Llama 2: 4k → 128k tokens", GOOD, GOOD_FILL, 22)
        m.VGroup(sharper, reach).arrange(m.RIGHT, buff=0.5).move_to([0, -3.35, 0])

        self.say(
            "[A smarter idea: scale each pair differently.] [Here's every pair's new speed.] [Interpolation slows them all by sixteen.]",
            self.new_header("FIX 2", "NTK-aware scaling and YaRN"),
            m.AnimationGroup(m.Create(chart), m.FadeIn(grid), m.FadeIn(y_ticks), m.FadeIn(y_title), m.FadeIn(x_ticks), m.FadeIn(x_title), m.Create(cross), m.FadeIn(cross_label)),
            m.AnimationGroup(m.FadeIn(keys[0], shift=0.15 * m.LEFT), m.Create(pi_curve, run_time=1.4)),
        )
        self.say(
            "[A trick inspired by neural tangent kernels raises the rotation base instead]: [fast pairs barely change, and only the slowest slow down the full sixteen times.]",
            m.AnimationGroup(m.Write(rule), m.FadeIn(rule_terms), m.FadeIn(keys[1], shift=0.15 * m.LEFT), m.FadeIn(cites[0])),
            m.Create(ntk_curve, run_time=2.4),
        )
        self.say(
            "[A method called Yarn goes further]: [it leaves fast pairs alone], [fully interpolates the slow ones], [and ramps in between.]",
            m.AnimationGroup(m.FadeIn(keys[2], shift=0.15 * m.LEFT), m.FadeIn(cites[1])),
            m.Create(yarn_parts[0], run_time=1.0),
            m.Create(yarn_parts[2], run_time=1.0),
            m.Create(yarn_parts[1], run_time=1.2),
        )
        self.say("[It also sharpens attention slightly, to suit the longer context.]", m.FadeIn(sharper, shift=0.2 * m.UP))
        self.say(
            "[With a short fine-tune, Llama 2 went from four thousand to a hundred and twenty-eight thousand tokens.]",
            m.FadeIn(reach, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ lost in the middle

    def middle(self):
        self.next_section(
            "Lost in the middle",
            notes="Liu et al., 2023 (TACL 2024): multi-document QA and key-value retrieval; accuracy is highest when the relevant "
            "passage is at the start or end of the input and drops in the middle. The curve here is a schematic of that U shape, "
            "not their numbers.",
        )
        xs = np.linspace(-5.0, 5.0, 11)
        strip = m.VGroup(*[page_card(0.62, 0.82, rows=4).move_to([x, 1.55, 0]) for x in xs])
        page_labels = m.VGroup(*[text(s, 20, MUTED).move_to([xs[k], 0.72, 0]) for s, k in (("page 1", 0), ("page 50", 5), ("page 100", 10))])
        chart = axes([0, 1.04, 0.25], [0.45, 0.95, 0.1], 10.4, 2.7).move_to([0.2, -1.6, 0])

        def at(x):
            return chart.c2p(x, lost(x))

        u_curve = polyline([at(x) for x in np.linspace(0, 1, 160)], PINK, width=5)
        y_title = text("accuracy", 22, MUTED).next_to(chart.c2p(0, 0.85), m.RIGHT, buff=0.18)
        x_title = text("where the answer sits in the input", 22, MUTED).next_to(chart.c2p(0.5, 0.45), m.DOWN, buff=0.22)
        source = cite("schematic, after Liu et al., 2023").move_to([4.6, -3.35, 0])
        marker = m.RoundedRectangle(corner_radius=0.1, width=0.84, height=1.02).set_stroke(GOOD, width=4).set_fill(opacity=0).move_to(strip[0])
        ends = m.VGroup(*[m.Dot(at(x), radius=0.1, color=GOOD).set_z_index(3) for x in (0, 1)])
        x_min = 0.5
        dip = m.Dot(at(x_min), radius=0.11, color=BAD).set_z_index(3)
        dip_label = text("lost in the middle", 24, BAD, weight=m.BOLD).next_to(dip, m.UP, buff=0.3)
        ring = m.RoundedRectangle(corner_radius=0.1, width=0.84, height=1.02).set_stroke(BAD, width=4).set_fill(opacity=0).move_to(strip[5])
        clause = text("your key clause?", 22, BAD, weight=m.BOLD).move_to([xs[5], 2.42, 0])

        self.say(
            "[One last catch]: [fitting the whole contract is not the same as using it well.]",
            self.new_header("THE CATCH", "Long context is not good use of it"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in strip], lag_ratio=0.06), m.FadeIn(page_labels)),
        )
        self.say(
            "[Liu and colleagues moved the passage holding the answer through a long input], [and measured accuracy at each spot.]",
            m.AnimationGroup(m.Create(chart), m.FadeIn(y_title), m.FadeIn(x_title), m.FadeIn(source), m.FadeIn(marker)),
            m.AnimationGroup(
                marker.animate(run_time=3.0, rate_func=m.linear).move_to(strip[10]),
                m.Create(u_curve, run_time=3.0, rate_func=m.linear),
            ),
        )
        self.say(
            "[Accuracy was best near the start and the end], [and sagged in the middle.]",
            m.AnimationGroup(m.FadeOut(marker), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in ends], lag_ratio=0.4)),
            m.AnimationGroup(m.FadeIn(dip, scale=0.4), m.FadeIn(dip_label, shift=0.15 * m.DOWN)),
        )
        self.say(
            "[So a key clause on page fifty is the one most likely to be missed.]",
            m.AnimationGroup(m.Create(ring), m.FadeIn(clause, shift=0.15 * m.DOWN)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Order, rotate, stretch.")
        halo = glow(PINK, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Order", "Rotate", "Stretch"), (SKY, PINK, TEAL))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Rotations make attention relative; rescaling them reads further.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [order], [rotate], [stretch.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[In short: rotations make attention relative, and rescaling them lets a model read further.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
