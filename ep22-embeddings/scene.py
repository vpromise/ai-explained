"""AI, Explained · Episode 22 · Embeddings

Running example: king − man + woman. Words become points in space:

- one-hot vectors: one slot per word, huge, and every pair is equally far apart (cosine 0);
- dense embeddings: a few hundred ordinary numbers, similar words get similar vectors;
- the embedding table: one row per word; a lookup is a row, the same as one-hot × table;
- word2vec (Mikolov et al., 2013): learn the vectors by predicting a word's neighbours;
  a tiny skip-gram model is trained for real below, and its words drift together on screen;
- vector arithmetic: king − man + woman lands next to queen on a toy 2-D map; in real models
  only approximately, and the three input words are left out of the search (Nissim et al., 2020);
- embeddings of images, users and products, and the token embeddings every language model starts with.

Everything plotted is computed here: the character codes, the one-hot cosines, the dense
300-number vectors (illustrative, seeded), the lookup as a matrix product, the toy map's
arithmetic and nearest word, and the toy skip-gram training (seeded, full-batch gradient
descent). The narration's claims about them are asserted, so the scene cannot drift from its words.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- colours

LIME_FILL = "#1E2A10"
WORD = {
    "king": (SKY, SKY_FILL),
    "queen": (PINK, PINK_FILL),  # king and queen as in Episode 9
    "man": (TEAL, TEAL_FILL),
    "woman": (AMBER, AMBER_FILL),
    "apple": (LIME, LIME_FILL),
    "banana": (LIME, LIME_FILL),
}
VOCAB = list(WORD)
STEP = VIOLET  # the step "woman − man"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------- the data

# the hook: to a computer, a word is character codes
CODES = {w: [ord(c) for c in w] for w in ("king", "queen")}
assert CODES["king"] == [107, 105, 110, 103] and CODES["queen"] == [113, 117, 101, 101, 110]

# one-hot: one slot per word
ONE_HOT = {w: row for w, row in zip(VOCAB, np.eye(len(VOCAB)))}
assert cosine(ONE_HOT["king"], ONE_HOT["queen"]) == 0.0 == cosine(ONE_HOT["king"], ONE_HOT["apple"])  # "zero ... and zero"
assert all(np.isclose(np.linalg.norm(ONE_HOT[a] - ONE_HOT[b]), np.sqrt(2)) for a in VOCAB for b in VOCAB if a != b)  # equally far apart

# dense: 300 numbers per word (illustrative: built from shared random "themes", seeded)
DIM = 300
_rng = np.random.default_rng(2022)
_PERSON, _ROYAL, _MALE, _FEMALE, _FRUIT = _rng.normal(0, 1, (5, DIM))


def _dense(*themes):
    return 0.3 * (sum(themes) + 0.35 * _rng.normal(0, 1, DIM))


DENSE = {
    "king": _dense(_PERSON, _ROYAL, 0.4 * _MALE),
    "queen": _dense(_PERSON, _ROYAL, 0.4 * _FEMALE),
    "man": _dense(_PERSON, 0.4 * _MALE),
    "woman": _dense(_PERSON, 0.4 * _FEMALE),
    "apple": _dense(_FRUIT),
    "banana": _dense(_FRUIT),
}
TABLE = np.array([DENSE[w] for w in VOCAB])  # the embedding table: one row per word
COS_KQ, COS_KA = cosine(DENSE["king"], DENSE["queen"]), cosine(DENSE["king"], DENSE["apple"])
assert TABLE.shape == (6, 300)  # "300 numbers"
assert round(COS_KQ, 1) == 0.9 and abs(COS_KA) < 0.1  # "about zero point nine", "close to zero"
assert np.allclose(ONE_HOT["king"] @ TABLE, TABLE[0])  # the lookup is one-hot × table
assert all(np.allclose(ONE_HOT[w] @ TABLE, DENSE[w]) for w in VOCAB)

# the toy 2-D map (illustrative numbers, chosen by hand)
MAP = {
    "king": np.array([2.0, 4.8]),
    "queen": np.array([3.6, 4.65]),
    "man": np.array([1.2, 2.0]),
    "woman": np.array([2.6, 1.6]),
    "apple": np.array([5.5, 1.1]),
    "banana": np.array([6.0, 1.8]),
}
RESULT = MAP["king"] - MAP["man"] + MAP["woman"]
STEP_VEC = MAP["woman"] - MAP["man"]


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def nearest(point, skip=()):
    return min((w for w in VOCAB if w not in skip), key=lambda w: dist(MAP[w], point))


assert np.allclose(RESULT, [3.4, 4.4])  # the arithmetic shown, coordinate by coordinate
assert nearest(RESULT) == "queen" == nearest(RESULT, skip=("king", "man", "woman"))  # "right next to queen"
assert dist(RESULT, MAP["queen"]) < 0.35 < 1.0 < dist(RESULT, MAP["king"])
for _w, _partner in (("king", "queen"), ("queen", "king"), ("man", "woman"), ("woman", "man"), ("apple", "banana"), ("banana", "apple")):
    assert nearest(MAP[_w], skip=(_w,)) == _partner  # "king near queen, man near woman", the fruit together
assert cosine(STEP_VEC, MAP["queen"] - MAP["king"]) > 0.98  # "a very similar step"

# word2vec, tiny and real: skip-gram (predict the neighbours within two words) with a full
# softmax, 4 numbers per word, full-batch gradient descent from seeded random vectors
CORPUS = [
    "the wise king wore a crown",
    "the wise queen wore a crown",
    "the king sat on the throne",
    "the queen sat on the throne",
    "the king ruled the land",
    "the queen ruled the land",
    "the old king rode a horse",
    "the young queen read a letter",
    "she ate a ripe apple",
    "he ate a ripe banana",
    "a sweet apple fell from the tree",
    "a sweet banana fell from the tree",
    "a crisp green apple",
    "a soft yellow banana",
]
SHOWN = ["king", "queen", "apple", "banana"]
W2V_SEED, W2V_DIM, W2V_STEPS, W2V_RATE = 6, 4, 3000, 2.0


def skipgram(seed=W2V_SEED, dim=W2V_DIM, steps=W2V_STEPS, rate=W2V_RATE, window=2):
    vocab = sorted({w for s in CORPUS for w in s.split()})
    ix = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(vocab)))  # how often each word has each neighbour
    for s in CORPUS:
        words = s.split()
        for i, centre in enumerate(words):
            for j in range(max(0, i - window), min(len(words), i + window + 1)):
                if j != i:
                    counts[ix[centre], ix[words[j]]] += 1
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, (len(vocab), dim))  # the word vectors (drawn)
    U = rng.normal(0, 0.3, (len(vocab), dim))  # the neighbour vectors
    per_word = counts.sum(1, keepdims=True)
    keep = set(np.round(np.geomspace(1, steps + 1, 70)).astype(int) - 1)  # dense snapshots early on
    snaps, at, losses = [], [], []
    for t in range(steps + 1):
        scores = W @ U.T
        scores -= scores.max(1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(1, keepdims=True)
        if t in keep:
            snaps.append(W[[ix[w] for w in SHOWN]].copy())
            at.append(t)
            losses.append(float(-(counts * np.log(p)).sum() / counts.sum()))
        grad = (per_word * p - counts) / counts.sum()  # d(cross-entropy)/d(scores)
        W, U = W - rate * grad @ U, U - rate * grad.T @ W
    return np.array(snaps), np.array(at), np.array(losses)


W2V_SNAPS, W2V_AT, W2V_LOSS = skipgram()
_final = W2V_SNAPS[-1]
_centre = _final.mean(0)
_, _, _vt = np.linalg.svd(_final - _centre)
W2V_2D = (W2V_SNAPS - _centre) @ _vt[:2].T  # flattened to the final vectors' two main directions (PCA)


def _closest(points, k):
    return min((j for j in range(len(SHOWN)) if j != k), key=lambda j: np.linalg.norm(points[j] - points[k]))


_K, _Q, _A, _B = range(4)
assert W2V_AT[-1] == W2V_STEPS and W2V_LOSS[-1] < 0.6 * W2V_LOSS[0]  # it really trained
assert _closest(W2V_2D[0], _K) == _A and _closest(W2V_SNAPS[0], _K) == _A  # "king even begins right next to apple"
assert _closest(W2V_2D[-1], _K) == _Q and _closest(W2V_2D[-1], _A) == _B  # "drift together"
assert _closest(W2V_SNAPS[-1], _K) == _Q and _closest(W2V_SNAPS[-1], _A) == _B  # also in all four numbers
_cross = min(np.linalg.norm(W2V_2D[-1][i] - W2V_2D[-1][j]) for i in (_K, _Q) for j in (_A, _B))
assert np.linalg.norm(W2V_2D[-1][_K] - W2V_2D[-1][_Q]) < 0.3 * _cross and np.linalg.norm(W2V_2D[-1][_A] - W2V_2D[-1][_B]) < 0.3 * _cross
assert cosine(_final[_K], _final[_Q]) > 0.8 and cosine(_final[_A], _final[_B]) > 0.95 and cosine(_final[_K], _final[_A]) < 0.2

# ---------------------------------------------------------------------------- helpers


def ep_recolored(name, color, fill):
    """kit.recolored(), but written as assets/ep22-<name>-<color>.svg (this episode only adds
    ep22- files to the shared assets)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep22-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def num(v):
    return f"{v:.2f}".replace("-", "−")


def op(symbol, size=44, color=INK, width=5):
    """+, − and = drawn as strokes: a typeset minus is too thin for the layout checker."""
    h, g = 0.0046 * size, 0.0022 * size
    bars = {
        "−": [((-h, 0), (h, 0))],
        "+": [((-h, 0), (h, 0)), ((0, -h), (0, h))],
        "=": [((-h, g), (h, g)), ((-h, -g), (h, -g))],
    }[symbol]
    return m.VGroup(*[m.Line([a[0], a[1], 0], [b[0], b[1], 0]) for a, b in bars]).set_stroke(color, width=width)


def dots3(color=MUTED, radius=0.032):
    """A horizontal ellipsis made of dots (a typeset one is too small for the layout checker)."""
    return m.VGroup(*[m.Dot(radius=radius, color=color) for _ in range(3)]).arrange(m.RIGHT, buff=0.07)


def word_chip(w, width=None, height=0.62, size=26):
    color, fill = WORD[w]
    return chip(w, color, fill, width=width or 0.42 + 0.21 * len(w) * size / 26, height=height, size=size)


def word_label(w, size=24):
    return text(w, size, WORD[w][0], weight=m.SEMIBOLD)


def hbracket(group, color, lip=0.1, pad=0.12, extra=0.08):
    """Square brackets around a horizontal row of numbers."""
    top, bottom = group.get_top()[1] + extra, group.get_bottom()[1] - extra

    def side(x, d):
        pts = [[x + d * lip, top, 0], [x, top, 0], [x, bottom, 0], [x + d * lip, bottom, 0]]
        return m.VMobject().set_points_as_corners(pts).set_stroke(color, width=3)

    return m.VGroup(side(group.get_left()[0] - pad, 1), side(group.get_right()[0] + pad, -1))


def row_vector(values, color, size=24, xs=None):
    """First four numbers, an ellipsis, the last one, in brackets; at given x positions (or arranged)."""
    shown = [num(v) for v in values[:4]] + ["…", num(values[-1])]
    nums = m.VGroup(*[code(s, size, color) if s != "…" else dots3() for s in shown])
    if xs is None:
        nums.arrange(m.RIGHT, buff=0.34)
    else:
        for n, x in zip(nums, xs):
            n.set_x(x)
    for n in nums[1:]:  # one baseline
        n.align_to(nums[0], m.DOWN)
    nums[4].set_y(nums[0].get_y() - 0.06)
    return m.VGroup(nums, hbracket(nums, color))


class ToyMap:
    """The toy 2-D map: two made-up dimensions, the six words as dots."""

    def __init__(self, origin=(-6.2, -3.15, 0), x_len=6.0, y_len=5.0, x_max=6.6, y_max=5.6, with_axes=True):
        self.ax = axes([0, x_max, 1], [0, y_max, 1], x_len, y_len)
        self.ax.shift(np.array(origin) - self.ax.c2p(0, 0))
        self.frame = m.VGroup(self.ax)
        if with_axes:
            self.frame.add(
                text("dimension 1", 20, MUTED).next_to(self.ax.c2p(x_max / 2, 0), m.DOWN, buff=0.3),
                text("dimension 2", 20, MUTED).next_to(self.ax.c2p(0, y_max), m.RIGHT, buff=0.2),
            )
        self.dots = {w: m.Dot(self.p(MAP[w]), radius=0.11, color=WORD[w][0]).set_z_index(3) for w in VOCAB}
        sides = {"king": m.LEFT, "queen": m.RIGHT, "man": m.DOWN, "woman": m.DOWN, "apple": m.LEFT, "banana": m.UP}
        self.labels = {w: word_label(w).next_to(self.dots[w], sides[w], buff=0.16) for w in VOCAB}

    def p(self, xy):
        return self.ax.c2p(*xy)

    def pair(self, w):
        return m.VGroup(self.dots[w], self.labels[w])


def tick_row(rows, x0, y0, dx, dy):
    """Lay a list of rows (lists of mobjects) out on a grid of centres."""
    for r, row in enumerate(rows):
        for c, mob in enumerate(row):
            mob.move_to([x0 + c * dx, y0 - r * dy, 0])


# ---------------------------------------------------------------------------- the episode


class Ep22Embeddings(Episode):
    number = 22

    def lesson(self):
        self.hook()
        self.one_hot()
        self.dense()
        self.table()
        self.word2vec()
        self.analogy()
        self.beyond()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="king − man + woman = ? To a computer, a word is just character codes; embeddings make it a point.")
        halo = glow(VIOLET, radius=3.6).move_to(1.3 * m.UP)
        mark = chip("?", INK, CARD, width=1.0, height=0.8, size=34)
        parts = [word_chip("king", height=0.8, size=34), op("−"), word_chip("man", height=0.8, size=34), op("+"),
                 word_chip("woman", height=0.8, size=34), op("="), mark]
        eq = m.VGroup(*parts).arrange(m.RIGHT, buff=0.3).move_to([0, 1.3, 0])
        answer = word_chip("queen", height=0.8, size=34).move_to(mark).align_to(mark, m.LEFT)

        def code_row(w, y):
            word = text(f'"{w}"', 30, WORD[w][0], weight=m.SEMIBOLD)
            word.move_to([0, y, 0]).align_to([-1.7, 0, 0], m.RIGHT)
            arrow_ = arrow([-1.45, y, 0], [-0.45, y, 0], MUTED, buff=0)
            nums = code("  ".join(str(c) for c in CODES[w]), 30, INK).move_to([0, y, 0]).align_to([-0.2, 0, 0], m.LEFT)
            return m.VGroup(word, arrow_, nums)

        rows = m.VGroup(code_row("king", -0.45), code_row("queen", -1.3))
        rows.set_x(0)
        nothing = pill("no hint that king and queen are related", BAD, BAD_FILL, 22).move_to([0, -2.5, 0])

        mid = np.array([3.65, 2.85])
        spots = {w: np.array([0.0, -1.35, 0]) + 0.72 * np.array([*(MAP[w] - mid), 0]) for w in VOCAB}
        dots = m.VGroup(*[m.Dot(spots[w], radius=0.11, color=WORD[w][0]) for w in VOCAB])
        sides = {"king": m.LEFT, "queen": m.RIGHT, "man": m.LEFT, "woman": m.RIGHT, "apple": m.LEFT, "banana": m.RIGHT}
        labels = m.VGroup(*[word_label(w).next_to(d, sides[w], buff=0.16) for w, d in zip(VOCAB, dots)])
        links = m.VGroup(*[
            m.DashedLine(spots[a], spots[b], dash_length=0.07, buff=0.16).set_stroke(MUTED, width=2.5)
            for a, b in (("king", "queen"), ("man", "woman"), ("apple", "banana"))
        ])

        self.say(
            "So... [what is king, minus man, plus woman?]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.LaggedStart(*[m.FadeIn(p, shift=0.25 * m.UP) for p in parts], lag_ratio=0.15)),
        )
        self.say(
            "[You'd probably say queen.] [But to a computer, a word is just a string of letters], [with no meaning at all.]",
            m.FadeTransform(mark, answer),
            m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.UP) for r in rows], lag_ratio=0.4),
            m.FadeIn(nothing, shift=0.2 * m.UP),
        )
        self.say(
            "[Embeddings fix that.] [They turn every word into a point in space], [where meaning becomes geometry.]",
            m.AnimationGroup(m.FadeOut(rows), m.FadeOut(nothing), m.FadeOut(halo), m.VGroup(eq[:-1], answer).animate.move_to([0, 2.15, 0]).scale(0.85)),
            m.LaggedStart(*[m.FadeIn(m.VGroup(d, l), scale=0.6) for d, l in zip(dots, labels)], lag_ratio=0.15),
            m.LaggedStart(*[m.Create(l) for l in links], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ one-hot

    def one_hot(self):
        self.next_section(
            "One-hot vectors",
            notes="Six-word toy vocabulary. Every pair of one-hot vectors: dot product 0, cosine 0, distance √2. "
            "Real vocabularies: tens of thousands of words or more (50,000 shown as an example).",
        )
        ys = [1.45 - 0.66 * r for r in range(6)]
        xs = [-4.15 + 0.6 * c for c in range(6)]
        chips = [word_chip(w, width=1.75, height=0.56, size=24).move_to([-5.55, y, 0]) for w, y in zip(VOCAB, ys)]
        slots = m.VGroup(*[text(str(c + 1), 20, MUTED).move_to([x, 2.12, 0]) for c, x in enumerate(xs)])
        slot_word = text("slot", 20, MUTED).move_to([-5.55, 2.12, 0])
        cells = m.VGroup(*[m.Square(side_length=0.5).set_stroke(EDGE, width=2).move_to([x, y, 0]) for y in ys for x in xs])
        rows = []
        for r, w in enumerate(VOCAB):
            color, fill = WORD[w]
            ones, zeros = m.VGroup(), m.VGroup()
            for c, x in enumerate(xs):
                if ONE_HOT[w][c] == 1:
                    sq = m.Square(side_length=0.5).set_stroke(color, width=2.5).set_fill(fill, opacity=1).move_to([x, ys[r], 0])
                    ones.add(sq, code("1", 24, color).move_to(sq))
                else:
                    zeros.add(code("0", 24, FAINT).move_to([x, ys[r], 0]))
            rows.append((ones, zeros))

        def problem(title, y, height):
            box = card(5.7, height, BAD, CARD, stroke=2).move_to([3.75, y, 0])
            head = text(title, 26, BAD, weight=m.BOLD).move_to(box.get_top() + 0.42 * m.DOWN).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            return box, head

        box1, head1 = problem("Problem 1 · huge", 1.15, 2.0)
        big1 = text("a 50,000-word vocabulary?", 24, INK).next_to(head1, m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        big2 = text("50,000 numbers per word, all but one 0", 22, MUTED).next_to(big1, m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        box2, head2 = problem("Problem 2 · no similarity", -1.35, 2.5)
        cos1 = code("cos(king, queen) = 0", 24, INK, t2c={"king": SKY, "queen": PINK, "= 0": BAD})
        cos2 = code("cos(king, apple) = 0", 24, INK, t2c={"king": SKY, "apple": LIME, "= 0": BAD})
        cos1.next_to(head2, m.DOWN, buff=0.32, aligned_edge=m.LEFT)
        cos2.next_to(cos1, m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        ep9 = text("cosine similarity · Episode 9", 20, MUTED).next_to(cos2, m.DOWN, buff=0.24, aligned_edge=m.LEFT)

        self.say(
            "[The simplest way to turn words into numbers is called one-hot.]",
            m.AnimationGroup(
                self.new_header("ONE-HOT VECTORS", "One slot per word"),
                m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.RIGHT) for c in chips], lag_ratio=0.1),
            ),
        )
        self.say(
            "[Give every word in the vocabulary its own slot.] [King gets a one in its own slot], [and zeros everywhere else.]",
            m.AnimationGroup(m.FadeIn(slots), m.FadeIn(slot_word), m.LaggedStart(*[m.Create(c) for c in cells], lag_ratio=0.01)),
            m.AnimationGroup(m.FadeIn(rows[0][0], scale=1.3), m.Indicate(chips[0], color=SKY, scale_factor=1.08)),
            m.LaggedStart(*[m.FadeIn(z, shift=0.1 * m.DOWN) for z in rows[0][1]], lag_ratio=0.1),
        )
        self.say(
            "[Every other word works the same way.]",
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(o, scale=1.3), m.FadeIn(z)) for o, z in rows[1:]], lag_ratio=0.2),
        )
        self.say(
            "[Problem one: they're huge.] [A real vocabulary has tens of thousands of words], [so each vector has tens of thousands of numbers, almost all zero.]",
            m.AnimationGroup(m.FadeIn(box1), m.FadeIn(head1, shift=0.15 * m.UP)),
            m.FadeIn(big1, shift=0.15 * m.UP),
            m.FadeIn(big2, shift=0.15 * m.UP),
        )
        self.say(
            "[Problem two: every word is equally far from every other.]",
            m.AnimationGroup(m.FadeIn(box2), m.FadeIn(head2, shift=0.15 * m.UP)),
        )
        self.say(
            "[Measure them with cosine similarity, from episode nine]: [king and queen score zero], [and so do king and apple.]",
            m.FadeIn(ep9, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(cos1, shift=0.15 * m.UP), m.Indicate(m.VGroup(chips[0], chips[1]), color=INK, scale_factor=1.06)),
            m.AnimationGroup(m.FadeIn(cos2, shift=0.15 * m.UP), m.Indicate(m.VGroup(chips[0], chips[4]), color=INK, scale_factor=1.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ dense

    def dense(self):
        self.next_section(
            "Dense embeddings",
            notes=f"Illustrative 300-number vectors (seeded numpy, built from shared themes): cos(king, queen) = {COS_KQ:.3f}, "
            f"cos(king, apple) = {COS_KA:.3f}. word2vec's Google News vectors have 300 numbers per word (Mikolov et al., 2013). "
            "The 2-D map is a hand-made toy.",
        )
        shown = ("king", "queen", "apple")
        ys = (1.35, 0.25, -0.85)
        xs = [-3.95 + 1.02 * c for c in range(6)]
        chips = {w: word_chip(w, width=1.75, height=0.56, size=24).move_to([-5.6, y, 0]) for w, y in zip(shown, ys)}
        vecs = {w: row_vector(DENSE[w], WORD[w][0], 22, xs).set_y(y) for w, y in zip(shown, ys)}
        for w in shown:  # centre each row on its chip
            vecs[w].shift((chips[w].get_y() - vecs[w][0][0].get_y()) * m.UP)
        size_pill = pill("word2vec: 300 per word", VIOLET, VIOLET_FILL, 20).move_to([0, ys[0], 0]).align_to([2.35, 0, 0], m.LEFT)
        reads = {
            "queen": code(f"cos with king = {num(COS_KQ)}", 24, GOOD, t2c={"king": SKY}).move_to([4.65, ys[1], 0]),
            "apple": code(f"cos with king = {num(COS_KA)}", 24, MUTED, t2c={"king": SKY}).move_to([4.65, ys[2], 0]),
        }
        for r in reads.values():
            r.align_to(size_pill, m.LEFT)
        caveat = text("illustrative numbers · the first four and the last of 300 shown", 20, MUTED).move_to([0, -1.95, 0])

        toy = ToyMap()
        self.toy = toy
        note_box = card(5.9, 2.5, EDGE, CARD, stroke=2).move_to([3.75, 0.85, 0])
        note = m.VGroup(
            text("A toy map", 26, INK, weight=m.BOLD),
            text("two made-up numbers per word", 22, MUTED),
            text("real embeddings: hundreds of", 22, MUTED),
            text("dimensions, none with a name", 22, MUTED),
        ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        note.move_to(note_box).align_to(note_box, m.LEFT).shift(0.4 * m.RIGHT)

        def ring(words, color):
            a, b = (toy.p(MAP[w]) for w in words)
            return m.DashedLine(a, b, dash_length=0.08, buff=0.18).set_stroke(color, width=3)

        rings = [ring(("king", "queen"), INK), ring(("man", "woman"), INK), ring(("apple", "banana"), LIME)]
        close = pill("similar words, close together", GOOD, GOOD_FILL, 22).move_to([3.75, -1.15, 0])

        self.say(
            "[A dense embedding does the opposite.] [Each word gets a short list of ordinary numbers], [often a few hundred.]",
            self.new_header("DENSE EMBEDDINGS", "A few hundred numbers per word"),
            m.AnimationGroup(m.FadeIn(chips["king"], shift=0.2 * m.RIGHT), m.FadeIn(vecs["king"], shift=0.2 * m.RIGHT)),
            m.FadeIn(size_pill, shift=0.2 * m.LEFT),
        )
        self.say(
            "[The numbers are learned], [so that words used in similar ways get similar vectors.]",
            m.AnimationGroup(*[m.FadeIn(m.VGroup(chips[w], vecs[w]), shift=0.2 * m.RIGHT) for w in ("queen", "apple")], m.FadeIn(caveat)),
            m.AnimationGroup(m.Indicate(vecs["king"][0], color=SKY, scale_factor=1.05), m.Indicate(vecs["queen"][0], color=PINK, scale_factor=1.05)),
        )
        self.say(
            "[For example, king and queen might have a cosine similarity of about zero point nine], [while king and apple are close to zero.]",
            m.FadeIn(reads["queen"], shift=0.2 * m.LEFT),
            m.FadeIn(reads["apple"], shift=0.2 * m.LEFT),
        )
        self.say(
            "[Picture each vector as a point in space.] [Here's a toy map with just two numbers per word], [so we can draw it.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (*chips.values(), *vecs.values(), size_pill, *reads.values(), caveat)],
                m.Create(toy.ax),
            ),
            m.LaggedStart(*[m.FadeIn(toy.pair(w), scale=0.6) for w in VOCAB], lag_ratio=0.15),
            m.AnimationGroup(m.FadeIn(toy.frame[1:]), m.FadeIn(note_box), m.FadeIn(note, shift=0.15 * m.UP)),
        )
        self.say(
            "[Similar words sit close together]: [king near queen, man near woman], [and the fruit off on their own.]",
            m.FadeIn(close, shift=0.2 * m.UP),
            m.AnimationGroup(m.Create(rings[0]), m.Create(rings[1])),
            m.Create(rings[2]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the embedding table

    def table(self):
        self.next_section(
            "The embedding table",
            notes="The table holds the six illustrative 300-number vectors as rows. One-hot × table = the row (asserted). "
            "Frameworks read the row directly (an index), which is the same result without the multiplications.",
        )
        ys = [1.2 - 0.58 * r for r in range(6)]
        xs = [-3.5 + 1.0 * c for c in range(6)]
        labels = m.VGroup(*[word_label(w, 24).move_to([-4.9, y, 0]) for w, y in zip(VOCAB, ys)])
        for lab in labels:
            lab.align_to([-4.45, 0, 0], m.RIGHT)
        cells = m.VGroup()
        for w, y in zip(VOCAB, ys):
            row = m.VGroup(*[code(s, 22, INK) if s != "…" else dots3() for s in [num(v) for v in DENSE[w][:4]] + ["…", num(DENSE[w][-1])]])
            for n, x in zip(row, xs):
                n.move_to([x, y, 0])
            for n in row[1:]:
                n.align_to(row[0], m.DOWN)
            row[4].set_y(row[0].get_y())
            cells.add(row)
        more = m.VGroup(*[text("⋮", 24, MUTED).move_to([x, ys[-1] - 0.55, 0]) for x in (xs[0], xs[3], xs[5])])
        grid = m.VGroup(cells, more)
        brackets = hbracket(grid, INK, lip=0.14, pad=0.25, extra=0.15)
        rows_note = text("one row per word: tens of thousands of rows", 22, MUTED).move_to([-1.3, -2.85, 0])
        brace = m.Brace(m.VGroup(*[c for c in cells[0]]), m.UP, buff=0.22).set_color(MUTED)
        cols_note = text("one column per dimension: a few hundred", 22, MUTED).next_to(brace, m.UP, buff=0.1)

        onehot = m.VGroup(*[code(str(int(ONE_HOT["king"][r])), 22, SKY if r == 0 else MUTED).move_to([-6.3, y, 0]) for r, y in enumerate(ys)])
        onehot_more = text("⋮", 24, MUTED).move_to([-6.3, ys[-1] - 0.55, 0])
        onehot_br = hbracket(m.VGroup(onehot, onehot_more), SKY, lip=0.08, pad=0.18, extra=0.15)
        onehot_all = m.VGroup(onehot, onehot_more, onehot_br)
        x_name = tex(r"\mathbf{x}", 36, SKY).next_to(onehot_br, m.UP, buff=0.18)

        highlight = m.SurroundingRectangle(m.VGroup(labels[0], cells[0]), color=SKY, buff=0.1, corner_radius=0.1).set_stroke(SKY, width=3)
        formula = tex(r"\mathbf{e}_{\text{king}} = \mathbf{x}_{\text{king}}\, E", 44, INK).move_to([4.55, 1.75, 0])
        terms = m.VGroup(
            m.VGroup(tex(r"\mathbf{x}", 36, SKY), text("king's one-hot vector", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(tex(r"E", 36, INK), text("the embedding table", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(tex(r"\mathbf{e}", 36, VIOLET), text("king's embedding", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([4.55, 0.5, 0])
        for t in terms:  # symbols in one column, words in another
            t[0].set_x(terms[1][0].get_x())
            t[1].align_to(terms[1][1], m.LEFT)
        terms.align_to([2.75, 0, 0], m.LEFT)
        result = vector(VIOLET, [num(v) for v in DENSE["king"][:3]] + [num(DENSE["king"][-1])], 22).move_to([5.3, -1.45, 0])
        result_title = m.VGroup(text("the lookup", 24, VIOLET, weight=m.SEMIBOLD), text("king's row", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        result_title.next_to(result, m.LEFT, buff=0.45)
        weights = pill("weights, learned like any layer", VIOLET, VIOLET_FILL, 20).move_to([4.55, -3.05, 0])

        self.say(
            "[Where do these vectors live? In one big matrix, called the embedding table.]",
            m.AnimationGroup(
                self.new_header("THE EMBEDDING TABLE", "One row per word"),
                m.FadeIn(brackets),
                m.LaggedStart(*[m.FadeIn(m.VGroup(l, c), shift=0.15 * m.RIGHT) for l, c in zip(labels, cells)], lag_ratio=0.12),
                m.FadeIn(more),
            ),
        )
        self.say(
            "[It has one row for every word in the vocabulary], [and one column for every dimension.]",
            m.AnimationGroup(m.FadeIn(rows_note, shift=0.15 * m.UP), m.LaggedStart(*[m.Indicate(l, scale_factor=1.15) for l in labels], lag_ratio=0.1)),
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(cols_note, shift=0.15 * m.DOWN)),
        )
        self.say(
            "[Looking up a word just means reading its row.]",
            m.AnimationGroup(m.Create(highlight), m.FadeIn(result_title), m.TransformFromCopy(cells[0], result)),
        )
        self.say(
            "[In math, the lookup is king's one-hot vector times the table.]",
            m.AnimationGroup(m.Write(formula), m.FadeIn(onehot_all, shift=0.2 * m.RIGHT), m.FadeIn(x_name)),
        )
        self.say(
            "[Each row is multiplied by its entry]: [the one keeps king's row], [and every zero wipes out the rest.]",
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in terms], lag_ratio=0.3),
            m.AnimationGroup(m.Indicate(onehot[0], color=SKY, scale_factor=1.6), m.Indicate(cells[0], color=SKY, scale_factor=1.04)),
            m.AnimationGroup(*[c.animate.set_opacity(0.2) for c in cells[1:]], *[m.Indicate(z, color=INK, scale_factor=1.4) for z in onehot[1:]]),
        )
        self.say(
            "[So the table is just a matrix of weights], [learned by gradient descent, like any other layer.]",
            m.AnimationGroup(*[c.animate.set_opacity(1) for c in cells[1:]], m.Circumscribe(m.VGroup(grid, brackets), color=VIOLET, buff=0.12)),
            m.FadeIn(weights, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ word2vec

    def word2vec(self):
        self.next_section(
            "word2vec",
            notes=f"word2vec: Mikolov et al., 2013 (skip-gram: predict the words around each word). The tiny model is trained "
            f"here for real: {len(CORPUS)} made-up sentences, a window of two, full softmax, {W2V_DIM} numbers per word, "
            f"{W2V_STEPS} full-batch gradient steps from seeded random vectors (seed {W2V_SEED}); loss {W2V_LOSS[0]:.2f} → {W2V_LOSS[-1]:.2f}. "
            "Drawn: the four word vectors at each snapshot, flattened onto the two main directions (PCA) of the final vectors.",
        )
        cite = text("Mikolov et al., 2013", 20, MUTED).move_to([0, 3.2, 0]).align_to([6.75, 0, 0], m.RIGHT)

        def sentence(words, centre_word, y):
            chips = m.VGroup(*[
                chip(w, *(WORD[w] if w in WORD else (EDGE, CARD)), width=0.34 + 0.17 * len(w), height=0.58, size=24)
                for w in words.split()
            ]).arrange(m.RIGHT, buff=0.1).move_to([-3.5, y, 0])
            k = words.split().index(centre_word)
            arcs = m.VGroup()
            for j in range(max(0, k - 2), min(len(chips), k + 3)):
                if j == k:
                    continue
                a, b = chips[k].get_top() + 0.06 * m.UP, chips[j].get_top() + 0.06 * m.UP
                bend = 0.9 if b[0] < a[0] else -0.9
                arcs.add(m.CurvedArrow(a, b, angle=bend * (1 if abs(j - k) == 1 else 1.15), color=WORD[centre_word][0], stroke_width=3, tip_length=0.15))
            lo, hi = max(0, k - 2), min(len(chips) - 1, k + 2)
            window = m.DashedVMobject(
                m.RoundedRectangle(corner_radius=0.18, width=chips[hi].get_right()[0] - chips[lo].get_left()[0] + 0.1, height=1.55),
                num_dashes=60,
            ).set_stroke(MUTED, width=2)
            window.move_to([(chips[hi].get_right()[0] + chips[lo].get_left()[0]) / 2, y + 0.4, 0])
            return chips, arcs, window

        s1, arcs1, win1 = sentence(CORPUS[0], "king", 1.2)
        s2, arcs2, win2 = sentence(CORPUS[1], "queen", -0.8)
        win_label = text("window: two words each side", 20, MUTED).next_to(win1, m.DOWN, buff=0.12).align_to(win1, m.LEFT)
        task = text("predict the neighbours of the middle word", 22, INK).move_to([-3.5, -1.9, 0])
        same = pill("same neighbours → similar vectors", GOOD, GOOD_FILL, 20).move_to([-3.5, -2.6, 0])

        # the tiny trained model, on the right
        panel = card(6.1, 5.75, EDGE, CARD, stroke=2).move_to([3.75, -0.5, 0])
        title = text("a tiny model, trained for real", 24, INK, weight=m.SEMIBOLD).move_to(panel.get_top() + 0.42 * m.DOWN)
        flat = text(f"{W2V_DIM} numbers per word, flattened to 2-D", 20, MUTED).next_to(title, m.DOWN, buff=0.12)
        lo, hi = W2V_2D.reshape(-1, 2).min(0), W2V_2D.reshape(-1, 2).max(0)
        scale = min(4.2 / (hi[0] - lo[0]), 2.6 / (hi[1] - lo[1]))
        centre2 = (lo + hi) / 2
        area = panel.get_center() + 0.3 * m.DOWN

        def at(xy):
            return area + scale * np.array([xy[0] - centre2[0], xy[1] - centre2[1], 0])

        clock = m.ValueTracker(0.0)

        def where(k):
            u = clock.get_value() * (len(W2V_2D) - 1)
            i = int(min(np.floor(u), len(W2V_2D) - 2))
            a = u - i
            return at((1 - a) * W2V_2D[i][k] + a * W2V_2D[i + 1][k])

        dots = []
        for k, w in enumerate(SHOWN):
            d = m.Dot(where(k), radius=0.11, color=WORD[w][0]).set_z_index(3)
            d.add_updater(lambda mob, k=k: mob.move_to(where(k)))
            dots.append(d)

        sides = {"king": m.RIGHT, "queen": m.DOWN, "apple": m.UP, "banana": m.RIGHT}  # clear at the start and at the end

        def label_for(k):
            return word_label(SHOWN[k], 24).next_to(where(k), sides[SHOWN[k]], buff=0.17)

        labels = [m.always_redraw(lambda k=k: label_for(k)) for k in range(len(SHOWN))]
        step = m.DecimalNumber(0, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=22, color=INK)
        step.add_updater(lambda d: d.set_value(float(np.interp(clock.get_value() * (len(W2V_AT) - 1), np.arange(len(W2V_AT)), W2V_AT))))
        step_row = m.VGroup(text("step", 20, MUTED), step).arrange(m.RIGHT, buff=0.18)
        step_row.move_to(panel.get_bottom() + 0.38 * m.UP).align_to(panel, m.LEFT).shift(0.3 * m.RIGHT)
        step.add_updater(lambda d: d.next_to(step_row[0], m.RIGHT, buff=0.18))
        meaning = pill("nobody told it what any word means", VIOLET, VIOLET_FILL, 20).move_to([-3.5, -3.3, 0])

        self.say(
            "[How are these vectors learned?] [The classic recipe is word2vec, from twenty thirteen.]",
            self.new_header("WORD2VEC", "Learn from the neighbours"),
            m.FadeIn(cite, shift=0.15 * m.LEFT),
        )
        self.say(
            "[Slide a small window along ordinary text.]",
            m.Succession(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in s1], lag_ratio=0.1), m.AnimationGroup(m.Create(win1), m.FadeIn(win_label))),
        )
        self.say(
            "[Take the word in the middle], [and train the model to predict its neighbours.]",
            m.Indicate(s1[2], color=SKY, scale_factor=1.15),
            m.AnimationGroup(m.LaggedStart(*[m.Create(a) for a in arcs1], lag_ratio=0.2), m.FadeIn(task, shift=0.15 * m.UP)),
        )
        self.say(
            "[In another sentence, queen turns up with exactly the same neighbours.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in s2], lag_ratio=0.08),
                m.Create(win2),
                m.LaggedStart(*[m.Create(a) for a in arcs2], lag_ratio=0.15),
            ),
        )
        self.say(
            "[For example, here's a tiny version, trained for real on a handful of made-up sentences.]",
            m.AnimationGroup(
                m.FadeIn(panel), m.FadeIn(title), m.FadeIn(flat), m.FadeIn(step_row),
                m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.15), *[m.FadeIn(l) for l in labels],
            ),
        )
        self.say(
            "[The vectors start out random]: [king even begins right next to apple.]",
            m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.6) for d in dots], lag_ratio=0.1),
            m.AnimationGroup(m.Flash(dots[_K].get_center(), color=SKY, flash_radius=0.3), m.Flash(dots[_A].get_center(), color=LIME, flash_radius=0.3)),
        )
        self.say(
            "[As training runs], [words that share neighbours drift together.]",
            clock.animate(run_time=5.0, rate_func=m.smooth).set_value(1.0),
            m.FadeIn(same, shift=0.2 * m.UP),
        )
        for mob in (*dots, *labels, step):
            mob.clear_updaters()
        self.say(
            "[Nobody tells it what the words mean]: it learns from the neighbours alone.",
            m.FadeIn(meaning, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ vector arithmetic

    def analogy(self):
        self.next_section(
            "Vector arithmetic",
            notes=f"Toy map (hand-made): king {MAP['king']}, man {MAP['man']}, woman {MAP['woman']}, queen {MAP['queen']}; "
            f"king − man + woman = {RESULT}, nearest word queen (distance {dist(RESULT, MAP['queen']):.2f}). In real models the result is only "
            "near queen (Mikolov et al., 2013), and the search leaves out the input words; without that rule the answer is usually "
            "one of the inputs, often king (Nissim, van Noord & van der Goot, 2020, Computational Linguistics 46(2)). Checked 2026-10-21.",
        )
        toy = ToyMap()
        formula = m.VGroup(
            text("king", 40, SKY, weight=m.BOLD), op("−", 40), text("man", 40, TEAL, weight=m.BOLD),
            op("+", 40), text("woman", 40, AMBER, weight=m.BOLD),
        ).arrange(m.RIGHT, buff=0.25)
        for w in (formula[2], formula[4]):  # one baseline (k and g reach above and below)
            w.align_to(formula[0], m.DOWN).shift(0.0 * m.UP)
        formula[2].shift((formula[0][0].get_bottom()[1] - formula[2][0].get_bottom()[1]) * m.UP)
        formula[4].shift((formula[0][0].get_bottom()[1] - formula[4][0].get_bottom()[1]) * m.UP)
        for sign in (formula[1], formula[3]):
            sign.set_y(formula[2].get_y())
        formula.move_to([3.85, 1.95, 0])
        start, end = toy.p(MAP["man"]), toy.p(MAP["woman"])
        step = m.Arrow(start, end, buff=0.14, color=STEP, stroke_width=6, max_tip_length_to_length_ratio=0.25).set_z_index(2)
        step_name = text("woman − man", 24, STEP, weight=m.SEMIBOLD).next_to(m.VGroup(toy.dots["man"], toy.dots["woman"]), m.UP, buff=0.2)
        moved = m.Arrow(toy.p(MAP["king"]), toy.p(RESULT), buff=0.14, color=STEP, stroke_width=6, max_tip_length_to_length_ratio=0.25).set_z_index(2)
        landing = m.VGroup(
            m.Circle(radius=0.17).set_stroke(INK, width=3).move_to(toy.p(RESULT)),
            m.Dot(toy.p(RESULT), radius=0.05, color=INK),
        ).set_z_index(4)

        # the arithmetic, coordinate by coordinate
        def vec(v, color):
            return code(f"[{v[0]:.1f}, {v[1]:.1f}]", 28, color)

        lines = [("", "king", MAP["king"]), ("−", "man", MAP["man"]), ("+", "woman", MAP["woman"])]
        rows = m.VGroup()
        for k, (sign, w, v) in enumerate(lines):
            y = 0.85 - 0.62 * k
            rows.add(m.VGroup(
                (op(sign, 30) if sign else m.VGroup()).move_to([1.55, y, 0]),
                word_label(w, 28).move_to([2.0, y, 0]).align_to([1.95, 0, 0], m.LEFT),
                vec(v, WORD[w][0]).move_to([5.0, y, 0]),
            ))
        rule = m.Line([1.35, -1.15, 0], [6.4, -1.15, 0]).set_stroke(EDGE, width=2.5)
        total = m.VGroup(op("=", 30).move_to([1.55, -1.6, 0]), vec(RESULT, INK).move_to([5.0, -1.6, 0]))
        verdict = pill(f"nearest word: queen", GOOD, GOOD_FILL, 24).move_to([3.85, -2.55, 0])

        real_box = card(5.9, 3.65, EDGE, CARD, stroke=2).move_to([3.85, -0.6, 0])
        real_title = text("In real models", 26, INK, weight=m.BOLD)
        real_rows = m.VGroup(
            text("≈ only approximately right", 24, INK),
            text("nearest word by cosine similarity", 24, INK),
            text("skip king, man and woman:", 24, INK),
            text("otherwise, often king wins", 24, MUTED),
        )
        real = m.VGroup(real_title, *real_rows).arrange(m.DOWN, buff=0.24, aligned_edge=m.LEFT)
        real.move_to(real_box).align_to(real_box, m.LEFT).shift(0.45 * m.RIGHT + 0.25 * m.UP)
        real_rows[3].shift(0.1 * m.UP)
        nissim = text("Nissim et al., 2020", 20, MUTED).next_to(real_box.get_corner(m.DR), m.UL, buff=0.22)

        self.say(
            "[Now for the payoff]: [king, minus man, plus woman.]",
            m.AnimationGroup(
                self.new_header("VECTOR ARITHMETIC", "king − man + woman"),
                m.Create(toy.ax), m.FadeIn(toy.frame[1:]),
                m.LaggedStart(*[m.FadeIn(toy.pair(w), scale=0.6) for w in VOCAB], lag_ratio=0.1),
            ),
            m.Write(formula),
        )
        self.say(
            "[First, woman minus man]: [the step that takes you from man to woman.]",
            m.AnimationGroup(m.Indicate(formula[4], color=AMBER), m.Indicate(formula[2], color=TEAL)),
            m.AnimationGroup(m.GrowArrow(step), m.FadeIn(step_name, shift=0.15 * m.UP)),
        )
        self.say(
            "[Now add that same step to king.]",
            m.AnimationGroup(m.Indicate(toy.pair("king"), color=SKY, scale_factor=1.15), m.TransformFromCopy(step, moved, path_arc=-0.5), run_time=1.8),
        )
        self.say(
            "[It lands right next to queen.]",
            m.AnimationGroup(m.FadeIn(landing, scale=1.6), m.Indicate(toy.labels["queen"], color=PINK, scale_factor=1.25)),
        )
        self.say(
            "[In numbers, it's plain arithmetic], [one coordinate at a time.]",
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.LEFT) for r in rows], lag_ratio=0.3),
            m.AnimationGroup(m.Create(rule), m.FadeIn(total, shift=0.15 * m.UP), m.FadeIn(verdict, shift=0.2 * m.UP)),
        )
        self.say(
            "[In real models, with hundreds of dimensions, the match is only approximate.]",
            m.AnimationGroup(*[m.FadeOut(mob) for mob in (rows, rule, total, verdict)], m.FadeIn(real_box), m.FadeIn(real_title), m.FadeIn(real_rows[0], shift=0.15 * m.UP)),
        )
        self.say(
            "[We look for the nearest word by cosine similarity], [and skip the three words we started with]: [otherwise, the answer is often king itself.]",
            m.FadeIn(real_rows[1], shift=0.15 * m.UP),
            m.FadeIn(real_rows[2], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(real_rows[3], shift=0.15 * m.UP), m.FadeIn(nissim), m.Indicate(toy.pair("king"), color=SKY, scale_factor=1.15)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ not just words

    def beyond(self):
        self.next_section(
            "Not just words",
            notes="Images, users and products, and the token embeddings of language models (tokens: Episode 23). "
            "Pictures and positions are illustrative. Bias in embeddings: Caliskan et al., 2017 (Science); Bolukbasi et al., 2016.",
        )
        user = ep_recolored("user", AMBER, AMBER_FILL)
        specs = [("Images", "similar pictures, nearby points", SKY), ("Users and products", "recommend what sits nearby", AMBER), ("Tokens", "where language models start", VIOLET)]
        cards = []
        for x, (title, sub, color) in zip((-4.6, 0, 4.6), specs):
            box = card(4.25, 3.95, color, CARD, stroke=2).move_to([x, -0.1, 0])
            head = m.VGroup(text(title, 28, color, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1)
            head.move_to(box.get_top() + 0.62 * m.DOWN)
            cards.append((box, head))

        def plane(box):
            c = box.get_center() + 0.55 * m.DOWN
            corner = c + np.array([-1.75, -1.15, 0])
            return m.VGroup(
                m.Line(corner, corner + 3.5 * m.RIGHT).set_stroke(FAINT, width=2),
                m.Line(corner, corner + 2.3 * m.UP).set_stroke(FAINT, width=2),
            ), c

        # images
        axes1, c1 = plane(cards[0][0])
        pics = m.VGroup(
            icon("ep22-mountains", 0.62).move_to(c1 + np.array([-1.0, 0.45, 0])),
            icon("ep22-hills", 0.62).move_to(c1 + np.array([-0.05, 0.8, 0])),
            icon("ep22-boat", 0.62).move_to(c1 + np.array([1.1, -0.5, 0])),
        )

        # users and products
        axes2, c2 = plane(cards[1][0])
        person = icon(user, 0.6).move_to(c2 + np.array([-0.8, 0.45, 0]))
        products = m.VGroup(
            icon("ep22-film", 0.55).move_to(c2 + np.array([0.25, 0.75, 0])),
            icon("ep22-bag", 0.55).move_to(c2 + np.array([1.15, -0.55, 0])),
        )
        rec = m.DashedLine(person.get_center(), products[0].get_center(), dash_length=0.06, buff=0.38).set_stroke(AMBER, width=2.5)

        # tokens
        box3 = cards[2][0]
        toks = m.VGroup(*[chip(w, VIOLET, VIOLET_FILL, width=0.34 + 0.17 * len(w), height=0.56, size=24) for w in ("the", "king", "wore")])
        toks.arrange(m.RIGHT, buff=0.15).move_to(box3.get_center() + 0.55 * m.UP)
        mini = m.VGroup(*[
            m.RoundedRectangle(corner_radius=0.05, width=2.6, height=0.2).set_stroke(EDGE, width=1.5).set_fill(CARD, opacity=1)
            for _ in range(5)
        ]).arrange(m.DOWN, buff=0.07).move_to(box3.get_center() + 0.85 * m.DOWN).set_z_index(1)
        picks = [mini[1], mini[3], mini[0]]
        lit = m.VGroup(*[r.copy().set_stroke(VIOLET, width=2).set_fill(VIOLET, opacity=0.55) for r in picks]).set_z_index(2)
        feeds = m.VGroup(*[
            m.Line(t.get_bottom() + 0.05 * m.DOWN, [t.get_x(), r.get_top()[1] + 0.05, 0]).set_stroke(VIOLET, width=2, opacity=0.8)
            for t, r in zip(toks, picks)
        ])
        table_name = text("embedding table", 20, MUTED).next_to(mini, m.DOWN, buff=0.12)
        ep23 = text("tokens: Episode 23", 20, MUTED).next_to(box3, m.DOWN, buff=0.14).align_to(box3, m.RIGHT)

        catch = pill("They learn from human text, and absorb its stereotypes too", BAD, BAD_FILL, 22).move_to([0, -2.88, 0])
        cite = text("Caliskan et al., 2017", 20, MUTED).move_to([0, -3.5, 0])

        self.say("[Embeddings aren't just for words.]", self.new_header("NOT JUST WORDS", "Anything can become a point"))
        self.say(
            "[Pictures can be embedded too], [so that similar images land close together.]",
            m.AnimationGroup(m.FadeIn(cards[0][0]), m.FadeIn(cards[0][1], shift=0.15 * m.UP), m.Create(axes1)),
            m.Succession(m.LaggedStart(*[m.FadeIn(p, scale=0.6) for p in pics], lag_ratio=0.25), m.Circumscribe(m.VGroup(pics[0], pics[1]), color=SKY, buff=0.1)),
        )
        self.say(
            "[Recommendation systems often embed users and products], [then suggest whatever sits nearby.]",
            m.AnimationGroup(m.FadeIn(cards[1][0]), m.FadeIn(cards[1][1], shift=0.15 * m.UP), m.Create(axes2), m.FadeIn(person, scale=0.6), m.FadeIn(products, scale=0.6)),
            m.AnimationGroup(m.Create(rec), m.Indicate(products[0], color=AMBER, scale_factor=1.2)),
        )
        self.say(
            "[And every language model starts with an embedding table]: [one row for each token it reads.]",
            m.AnimationGroup(m.FadeIn(cards[2][0]), m.FadeIn(cards[2][1], shift=0.15 * m.UP), m.FadeIn(mini), m.FadeIn(table_name)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.DOWN) for t in toks], lag_ratio=0.2), m.Create(feeds), m.FadeIn(lit), m.FadeIn(ep23)),
        )
        self.say(
            "[One catch: embeddings learn from human writing], [so they soak up its stereotypes too.]",
            m.FadeIn(catch, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(cite), m.Wiggle(catch, scale_value=1.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Tables, neighbours, geometry.")
        halo = glow(VIOLET, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Tables", "Neighbours", "Geometry"), (SKY, VIOLET, GOOD))]).arrange(m.RIGHT, buff=1.1)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Words become points, and meaning becomes geometry.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [tables], [neighbours], [geometry.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Words become points, and meaning becomes geometry.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
