"""AI, Explained · Episode 20 · Convolutional Networks

Running example: finding the edges in a 16×16 grayscale picture of a house, drawn in code
(brightness 0 = black … 9 = white; no photographs anywhere). Everything on screen is computed
below with numpy:

- a 3×3 vertical-edge Sobel filter and its horizontal twin slid over the image (stride 1, no
  padding: a 14×14 feature map each);
- the worked multiply-and-sum (a sky|wall patch: -4 + 0 + 32 = 28; a flat wall patch: 0);
- the weight counts (one filter: 9; a dense layer from 256 pixels to 196 outputs: 50,176);
- the output sizes for stride 2 (7×7) and padding 1 (16×16);
- ReLU and 2×2 max pooling (14×14 → 7×7);
- the house moved one pixel right: the feature map moves exactly one pixel (translation
  equivariance); one pooled block keeps its maximum, 28, although the values inside it change
  (some invariance, not complete);
- the receptive field of three conv → ReLU → pool blocks (3, 8, 18 pixels).

The asserts tie every number the narration quotes to these computations. As in deep-learning
libraries, the "convolution" does not flip the filter (strictly a cross-correlation); the
screen says so. Colours in this film: TEAL = positive (weights, responses), PINK = negative,
AMBER = the filter and its window, VIOLET = the network (the level's accent).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the data

H = W = 16


def make_house():
    """The running example: brightness 0 (black) to 9 (white), drawn as pixel art."""
    img = np.full((H, W), 1, dtype=int)  # night sky
    img[14:, :] = 5  # ground
    img[8:14, 3:13] = 8  # walls
    for k, r in enumerate(range(3, 8)):  # the roof, a stepped triangle
        img[r, 7 - k : 9 + k] = 4
    img[10:14, 9:11] = 2  # door
    img[9:11, 4:7] = 3  # window
    return img


IMG = make_house()
assert IMG.shape == (16, 16) and IMG.size == 256 and IMG.min() >= 0 and IMG.max() <= 9  # "sixteen by sixteen", "zero … nine"

GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel: vertical edges (Sobel & Feldman, 1968)
GY = GX.T  # the same filter turned on its side (rotated a quarter turn clockwise): horizontal edges
assert (np.rot90(GX, -1) == GY).all() and GX.size == 9


def conv2d(a, k, stride=1, pad=0):
    """Slide k over a (zero padding `pad`), multiply and add: no flip, as in CNN libraries."""
    a = np.pad(a, pad)
    n = k.shape[0]
    rows, cols = (a.shape[0] - n) // stride + 1, (a.shape[1] - n) // stride + 1
    out = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = int((a[i * stride : i * stride + n, j * stride : j * stride + n] * k).sum())
    return out


def maxpool(a, size=2):
    h, w = a.shape[0] // size, a.shape[1] // size
    return a[: h * size, : w * size].reshape(h, size, w, size).max(axis=(1, 3))


FX, FY = conv2d(IMG, GX), conv2d(IMG, GY)
assert FX.shape == FY.shape == (14, 14)
VMAX = int(max(np.abs(FX).max(), np.abs(FY).max()))
assert VMAX == 28

# one step, worked: rows 11–13, columns 2–4 (sky | wall | wall), then a flat patch of wall
EDGE_AT, FLAT_AT = (11, 2), (11, 5)
PATCH, FLAT = IMG[11:14, 2:5], IMG[11:14, 5:8]
assert PATCH.tolist() == [[1, 8, 8]] * 3  # "dark on the left, bright on the right"
assert (PATCH * GX).sum(axis=0).tolist() == [-4, 0, 32] and FX[EDGE_AT] == 28  # "minus four … plus thirty-two … twenty-eight"
assert (FLAT == 8).all() and (FLAT * GX).sum(axis=0).tolist() == [-32, 0, 32] and FX[FLAT_AT] == 0  # "cancel: zero"

# the slide: first along row 11 (the worked row), then every other row, top to bottom
ORDER = [(11, j) for j in range(14)] + [(i, j) for i in range(14) for j in range(14) if i != 11]
assert len(ORDER) == FX.size == 196

# weight sharing
DENSE = IMG.size * FX.size
assert DENSE == 50_176 and DENSE > 50_000  # "over fifty thousand" vs "just nine"

# stride and padding
S2, P1 = conv2d(IMG, GX, stride=2), conv2d(IMG, GX, pad=1)
assert S2.shape == (7, 7) and P1.shape == (16, 16)  # "half as wide: seven by seven"; "keeps the image's size"

# ReLU, then 2×2 max pooling
RX = np.maximum(FX, 0)
PX = maxpool(RX)
POOL_AT = (4, 0)  # the block of rows 8–9, columns 0–1
assert PX.shape == (7, 7) and RX[8:10, 0:2].tolist() == [[0, 28], [0, 28]] and PX[POOL_AT] == 28  # "zero, twenty-eight, zero and twenty-eight"

# the house moves one pixel right
IMG2 = np.roll(IMG, 1, axis=1)
assert (IMG[:, 0] == IMG[:, -1]).all()  # the column that wraps round is plain sky and ground: a true shift
FX2 = conv2d(IMG2, GX)
assert (FX2[:, 1:] == FX[:, :-1]).all() and (FX2[:, 0] == 0).all()  # the map moves exactly one pixel: equivariance
RX2 = np.maximum(FX2, 0)
PX2 = maxpool(RX2)
KEEP = (5, 1)  # the block of rows 10–11, columns 2–3
BEFORE, AFTER = RX[10:12, 2:4], RX2[10:12, 2:4]
assert BEFORE.tolist() == [[23, 0], [28, 0]] and AFTER.tolist() == [[28, 23], [28, 28]]  # "the values inside changed"
assert PX[KEEP] == PX2[KEEP] == 28  # "the largest is still twenty-eight"
assert (PX != PX2).any()  # "not complete"


def receptive(layers):
    """How many input pixels across one unit sees, after these layers (3×3 conv, 2×2 pool)."""
    field, jump = 1, 1
    for kind in layers:
        if kind == "conv":
            field += 2 * jump
        else:
            field += jump
            jump *= 2
    return field


RF = [receptive(["conv"]), receptive(["conv", "pool", "conv"]), receptive(["conv", "pool", "conv", "pool", "conv"])]
assert RF == [3, 8, 18] and RF[2] > W  # "three pixels across, then eight, then eighteen: the whole house"

# ImageNet 2012 (ILSVRC-2012), top-5 test error (Krizhevsky et al., 2012)
ALEX, RUNNER_UP = 15.3, 26.2
assert round(ALEX) == 15 and round(RUNNER_UP) == 26  # "about fifteen percent … about twenty-six"

# ---------------------------------------------------------------------------- helpers

DARK = "#0E1426"  # brightness 0
ZERO = "#19213A"  # a computed zero in a feature map


def mix(a, b, t):
    """A colour between two hex colours."""
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


def gray(v):
    return mix(DARK, INK, v / 9)


def heat(v):
    """A response: teal if positive, pink if negative, brighter when stronger."""
    if v == 0:
        return ZERO
    t = min(1.0, abs(v) / VMAX) ** 0.7
    return mix(ZERO, TEAL if v > 0 else PINK, 0.22 + 0.78 * t)


def signed(v):
    return "0" if v == 0 else (f"+{v}" if v > 0 else f"−{-v}")


def sign_color(v):
    return TEAL if v > 0 else (PINK if v < 0 else MUTED)


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


class Board:
    """A grid of square cells, row-major: an image, a filter or a feature map."""

    def __init__(self, values, cell, color_of, center, stroke=BG, width=1.2, show=True):
        self.values = np.asarray(values)
        self.rows, self.cols = self.values.shape
        self.cell = cell
        self.cells = m.VGroup()
        for r in range(self.rows):
            for c in range(self.cols):
                sq = m.Square(side_length=cell).set_stroke(stroke, width=width)
                sq.set_fill(color_of(self.values[r, c]), opacity=1 if show else 0)
                self.cells.add(sq.move_to([c * cell, -r * cell, 0]))
        self.cells.move_to(center)

    def at(self, r, c):
        return self.cells[r * self.cols + c]

    def point(self, r, c):
        """The centre of cell (r, c); any r, c (outside the board too, e.g. padding)."""
        return self.cells[0].get_center() + np.array([c * self.cell, -r * self.cell, 0])

    def window(self, r, c, n=3):
        """The centre of the n×n block whose top-left cell is (r, c)."""
        return self.point(r + (n - 1) / 2, c + (n - 1) / 2)

    def recolor(self, values, color_of):
        return m.AnimationGroup(*[self.at(r, c).animate.set_fill(color_of(v), opacity=1) for (r, c), v in np.ndenumerate(values)])


def digits(board, values, size=22):
    """Each cell's number on top of it: dark on bright cells, light on dark ones."""
    return m.VGroup(*[code(str(v), size, BG if v >= 5 else INK).move_to(board.at(r, c)) for (r, c), v in np.ndenumerate(values)])


def window_frame(cell, n=3, color=AMBER, width=4):
    return m.Square(side_length=n * cell).set_stroke(color, width=width).set_fill(BG, opacity=0).set_z_index(4)


def edge_lines(board, values, color, axis=None, width=4):
    """A segment on every border between two cells that differ: where the brightness jumps.
    axis="v" keeps only vertical borders (left-right jumps), "h" only horizontal ones."""
    segs = m.VGroup()
    h, e = board.cell / 2, 0.012
    for r in range(board.rows):
        for c in range(board.cols):
            p = board.point(r, c)
            if axis != "h" and c + 1 < board.cols and values[r, c] != values[r, c + 1]:
                segs.add(m.Line(p + np.array([h, h + e, 0]), p + np.array([h, -h - e, 0])).set_stroke(color, width=width))
            if axis != "v" and r + 1 < board.rows and values[r, c] != values[r + 1, c]:
                segs.add(m.Line(p + np.array([-h - e, -h, 0]), p + np.array([h + e, -h, 0])).set_stroke(color, width=width))
    return segs.set_z_index(2)


def kernel_mob(k, cell, size, stroke=AMBER):
    """A 3×3 filter: weights in boxes, teal if positive, pink if negative."""
    squares, nums = m.VGroup(), m.VGroup()
    for (r, c), v in np.ndenumerate(k):
        sq = m.Square(side_length=cell).set_stroke(stroke, width=2.5).set_fill(CARD, opacity=1).move_to([(c - 1) * cell, (1 - r) * cell, 0])
        squares.add(sq)
        nums.add(code(signed(int(v)), size, sign_color(int(v)), weight=m.BOLD).move_to(sq))
    return m.VGroup(squares, nums)


def num_grid(values, cell, size, fill_of, ink_of, stroke=EDGE):
    """A small grid of numbers in coloured cells (the zoomed patch, the products)."""
    squares, nums = m.VGroup(), m.VGroup()
    rows, cols = np.asarray(values).shape
    for (r, c), v in np.ndenumerate(values):
        sq = m.Square(side_length=cell).set_stroke(stroke, width=2).set_fill(fill_of(v), opacity=1)
        squares.add(sq.move_to([(c - (cols - 1) / 2) * cell, ((rows - 1) / 2 - r) * cell, 0]))
        nums.add(code(signed(int(v)) if ink_of is sign_color else str(int(v)), size, ink_of(v), weight=m.BOLD).move_to(sq))
    return m.VGroup(squares, nums)


def baseline(words):
    """Line up words on one baseline (glyphs with descenders don't count)."""

    def base(w):
        sitting = [g for g, ch in zip(w, w.text.replace(" ", "").replace("\n", "")) if ch not in "gjpqy,"] or list(w)
        return min(g.get_bottom()[1] for g in sitting)

    for w in words[1:]:
        w.shift((base(words[0]) - base(w)) * m.UP)
    return words


def polyline(points, color, width=6):
    return m.VMobject().set_points_as_corners([np.array([x, y, 0]) for x, y in points]).set_stroke(color, width=width)


def feature_tile(kind, color, size=1.15):
    """A small card with the kind of pattern a layer responds to."""
    box = card(size, size, color, CARD, radius=0.15, stroke=2)
    shapes = {
        "vertical": [[(0, -0.36), (0, 0.36)]],
        "horizontal": [[(-0.36, 0), (0.36, 0)]],
        "corner": [[(-0.3, 0.32), (-0.3, -0.28), (0.32, -0.28)]],
        "roof": [[(-0.38, -0.2), (0, 0.24), (0.38, -0.2)]],
        "house": [[(-0.3, -0.36), (0.3, -0.36), (0.3, 0.04), (0, 0.36), (-0.3, 0.04), (-0.3, -0.36)], [(-0.07, -0.36), (-0.07, -0.12), (0.07, -0.12), (0.07, -0.36)]],
        "tree": [[(-0.3, -0.12), (0.3, -0.12), (0, 0.38), (-0.3, -0.12)], [(0, -0.12), (0, -0.38)]],
    }[kind]
    marks = m.VGroup(*[polyline(p, color, 5) for p in shapes]).move_to(box)
    return m.VGroup(box, marks)


def thumbnail(kind, color, fill):
    """An abstract stand-in for a labelled photo (no real photographs)."""
    box = card(0.95, 0.72, EDGE, CARD, radius=0.1, stroke=2)
    c = box.get_center()
    if kind == 0:  # a hill and a sun
        art = m.VGroup(
            m.Polygon(c + [-0.38, -0.26, 0], c + [0.0, 0.14, 0], c + [0.38, -0.26, 0]).set_stroke(color, width=3).set_fill(fill, opacity=1),
            m.Circle(radius=0.08).set_stroke(AMBER, width=3).set_fill(AMBER_FILL, opacity=1).move_to(c + [0.24, 0.16, 0]),
        )
    elif kind == 1:  # a ball
        art = m.Circle(radius=0.22).set_stroke(color, width=3).set_fill(fill, opacity=1).move_to(c)
    elif kind == 2:  # a box
        art = m.Square(side_length=0.36).set_stroke(color, width=3).set_fill(fill, opacity=1).move_to(c)
    else:  # a fish
        art = m.VGroup(
            m.Ellipse(width=0.46, height=0.26).set_stroke(color, width=3).set_fill(fill, opacity=1).move_to(c + [-0.05, 0, 0]),
            m.Polygon(c + [0.18, 0, 0], c + [0.36, 0.13, 0], c + [0.36, -0.13, 0]).set_stroke(color, width=3).set_fill(fill, opacity=1),
        )
    return m.VGroup(box, art)


# ---------------------------------------------------------------------------- the episode


class Ep20ConvolutionalNetworks(Episode):
    number = 20

    def lesson(self):
        house, extras = self.hook()
        self.problem(house, extras)
        k = self.the_filter()
        self.one_step(k)
        self.feature_map()
        self.sharing()
        self.stride_padding()
        self.pooling()
        self.shifting()
        self.stacking()
        self.breakthrough()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A 16×16 house drawn in code (illustrative); each pixel is a brightness, 0 black to 9 white; edges are where neighbours differ (computed).")
        house = Board(IMG, 0.36, gray, [-2.9, -0.35, 0])
        nums = digits(house, IMG).set_z_index(3)
        size = text("16 × 16 pixels", 32, INK, weight=m.BOLD).move_to([3.6, 1.75, 0])
        scale_vals = np.arange(10)[None, :]
        swatch = Board(scale_vals, 0.42, gray, [3.6, 0.6, 0])
        swatch_nums = digits(swatch, scale_vals).set_z_index(3)
        ends = m.VGroup(
            text("black", 22, MUTED).next_to(swatch.at(0, 0), m.DOWN, buff=0.2),
            text("white", 22, MUTED).next_to(swatch.at(0, 9), m.DOWN, buff=0.2),
        )
        count = text("256 numbers", 30, SKY, weight=m.BOLD).move_to([3.6, -0.95, 0])
        edges = edge_lines(house, IMG, TEAL)
        note = text("edges: where neighbouring pixels differ", 24, TEAL, weight=m.SEMIBOLD).move_to([3.6, -2.35, 0])

        self.say(
            "So... [how does a network see?] [Take this tiny house, drawn in code]: sixteen by sixteen pixels.",
            m.FadeIn(house.cells, lag_ratio=0.004, run_time=1.6),
            m.FadeIn(size, shift=0.2 * m.UP),
        )
        self.say(
            "[To the network, it's just numbers]: [the brightness of each pixel, from zero for black to nine for white.]",
            m.AnimationGroup(m.FadeIn(nums, lag_ratio=0.003, run_time=1.4), m.FadeIn(count, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(swatch.cells, lag_ratio=0.1), m.FadeIn(swatch_nums, lag_ratio=0.1), m.FadeIn(ends)),
        )
        self.say(
            "[Our goal: find its edges], [the places where the brightness suddenly jumps.]",
            m.AnimationGroup(self.new_header("THE GOAL", "Find the edges"), m.Create(edges, lag_ratio=0.01, run_time=1.6)),
            m.FadeIn(note, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        return house, [nums, size, swatch.cells, swatch_nums, ends, count, edges, note]

    # ------------------------------------------------------------------ the problem with dense layers

    def problem(self, house, extras):
        self.next_section("The problem", notes="A dense neuron needs one weight per pixel: 256 here, 1,000,000 for a one-megapixel photo; weights learned at one position don't transfer to another.")
        small = Board(IMG, 0.24, gray, [-4.6, -0.4, 0])
        small.cells.set_z_index(1)
        neuron = m.Circle(radius=0.5).set_stroke(VIOLET, width=4).set_fill(VIOLET_FILL, opacity=1).move_to([0.45, -0.4, 0]).set_z_index(2)
        sigma = tex(r"\Sigma", 44, INK).move_to(neuron).set_z_index(3)
        hub = neuron.get_center()
        lines = m.VGroup(*[m.Line(small.point(r, c), hub).set_stroke(VIOLET, width=1, opacity=0.22) for r in range(H) for c in range(W)])
        lines.set_z_index(-1)
        one = text("one neuron", 26, MUTED).move_to([4.0, 0.95, 0])
        count = text("256 weights", 48, VIOLET, weight=m.BOLD).next_to(one, m.DOWN, buff=0.2)
        mega_box = card(5.5, 1.45, BAD, BAD_FILL).move_to([4.0, -1.55, 0])
        mega = m.VGroup(text("a 1-megapixel photo", 22, MUTED), text("1,000,000 weights per neuron", 24, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.15).move_to(mega_box)

        def spot(r, c, color):
            frame = window_frame(small.cell, color=color, width=3.5).move_to(small.window(r, c))
            bundle = m.VGroup(*[m.Line(small.point(r + i, c + j), hub).set_stroke(color, width=2.5, opacity=0.95) for i in range(3) for j in range(3)])
            return frame, bundle.set_z_index(0)

        frame_a, lines_a = spot(*EDGE_AT, BLUE)
        frame_b, lines_b = spot(8, 5, LIME)  # the window's right side: the same dark → bright edge
        assert FX[8, 5] > 0 and FX[EDGE_AT] > 0
        note = text("the same kind of edge, in two places: two separate sets of weights", 24, INK).move_to([-0.3, -3.25, 0])

        self.say(
            "[Why not use a dense layer, like in episode three?]",
            m.AnimationGroup(
                self.new_header("THE PROBLEM", "A weight for every pixel"),
                m.ReplacementTransform(house.cells, small.cells),
                *[m.FadeOut(mob) for mob in extras],
            ),
        )
        self.say(
            "[Every neuron would connect to every pixel]: [two hundred and fifty-six weights, for just one neuron.]",
            m.AnimationGroup(m.FadeIn(neuron, scale=0.6), m.FadeIn(sigma), m.Create(lines, lag_ratio=0.002, run_time=1.6)),
            m.AnimationGroup(m.FadeIn(one), m.FadeIn(count, shift=0.2 * m.UP)),
        )
        self.say(
            "[For a one-megapixel photo, that's a million weights per neuron.]",
            m.AnimationGroup(m.FadeIn(mega_box, shift=0.2 * m.UP), m.FadeIn(mega, shift=0.2 * m.UP)),
        )
        self.say(
            "[Worse, each position has its own weights]: [learning an edge here] [doesn't help to spot one over there.]",
            m.ShowPassingFlash(lines.copy().set_stroke(VIOLET, width=2, opacity=0.8), time_width=0.6, run_time=1.2),
            m.AnimationGroup(m.Create(frame_a), m.Create(lines_a)),
            m.AnimationGroup(m.Create(frame_b), m.Create(lines_b), m.FadeIn(note, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the filter

    def the_filter(self):
        self.next_section("The filter", notes="A 3×3 filter: 9 weights. The vertical-edge Sobel filter (Sobel & Feldman, 1968).")
        k = kernel_mob(GX, 1.0, 40).move_to([-2.2, -0.45, 0])
        size = text("3 × 3 filter · 9 weights", 28, AMBER, weight=m.BOLD).next_to(k, m.UP, buff=0.4)
        neg = text("negative", 24, PINK, weight=m.SEMIBOLD).next_to(k[0][6], m.DOWN, buff=0.3)
        pos = text("positive", 24, TEAL, weight=m.SEMIBOLD).next_to(k[0][8], m.DOWN, buff=0.3)
        name = text("Finds vertical edges", 30, INK, weight=m.BOLD)
        sub = text("a classic: the Sobel filter", 22, MUTED)
        cite = text("Sobel & Feldman, 1968", 20, MUTED)
        words = m.VGroup(name, sub, cite).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        strip_vals = np.array([[1, 1, 1, 8, 8, 8]])
        strip = Board(strip_vals, 0.42, gray, [0, 0, 0])
        strip_label = text("fires on: dark → bright", 22, MUTED)
        fires = m.VGroup(strip_label, strip.cells).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        right = m.VGroup(words, fires).arrange(m.DOWN, buff=0.6, aligned_edge=m.LEFT).move_to([3.4, -0.45, 0])

        self.say(
            "[Convolution's answer is a filter]: [a tiny three by three grid of weights.]",
            m.AnimationGroup(self.new_header("THE FILTER", "Nine weights, one pattern"), m.FadeIn(k[0], lag_ratio=0.1)),
            m.AnimationGroup(m.FadeIn(k[1], lag_ratio=0.1), m.FadeIn(size, shift=0.2 * m.UP)),
        )
        self.say(
            "[This classic one looks for vertical edges]: [negative weights on the left], [positive weights on the right.]",
            m.AnimationGroup(m.FadeIn(words, shift=0.2 * m.LEFT), m.FadeIn(fires, shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(neg, shift=0.15 * m.UP), *[m.Indicate(k[1][i], color=PINK, scale_factor=1.3) for i in (0, 3, 6)]),
            m.AnimationGroup(m.FadeIn(pos, shift=0.15 * m.UP), *[m.Indicate(k[1][i], color=TEAL, scale_factor=1.3) for i in (2, 5, 8)]),
        )
        self.wait(0.3)
        self.clear(k[0], k[1])
        return k

    # ------------------------------------------------------------------ one step: multiply and add

    def one_step(self, k):
        self.next_section(
            "One step",
            notes="Patch rows 11–13, cols 2–4 = [1 8 8] ×3; × Sobel: columns −4, 0, +32 → 28. Flat wall patch (cols 5–7): −32, 0, +32 → 0. "
            "S(i,j) = Σm Σn I(i+m, j+n)·K(m,n): no flip, i.e. cross-correlation, as in deep-learning libraries.",
        )
        cell, y = 0.66, 0.95
        xs = dict(patch=-2.0, times=-0.62, filter=0.75, step=2.15, products=3.55, sum=5.85)
        house = Board(IMG, 0.2, gray, [-5.2, 0.75, 0])
        win = window_frame(house.cell, width=3.5).move_to(house.window(*EDGE_AT))
        k_target = k.copy().scale(cell / 1.0).move_to([xs["filter"], y, 0])

        def zoom_of(values):
            return num_grid(values, cell, 26, gray, lambda v: BG if v >= 5 else INK).move_to([xs["patch"], y, 0])

        def products_of(values):
            return num_grid(values * GX, cell, 24, lambda v: CARD, sign_color).move_to([xs["products"], y, 0])

        def sums_of(values, products):
            cols = (values * GX).sum(axis=0)
            return m.VGroup(*[code(signed(int(v)), 26, sign_color(int(v)), weight=m.BOLD).move_to([products[0][j].get_x(), y - 1.35, 0]) for j, v in enumerate(cols)])

        zoom = zoom_of(PATCH)
        products = products_of(PATCH)
        sums = sums_of(PATCH, products)
        times = tex(r"\times", 44, INK).move_to([xs["times"], y, 0])
        equals = arrow([xs["step"] - 0.27, y, 0], [xs["step"] + 0.27, y, 0], MUTED, buff=0.02)
        to_sum = arrow([xs["products"] + 1.13, y, 0], [xs["products"] + 1.67, y, 0], MUTED, buff=0.02)
        total = pill("28", TEAL, TEAL_FILL, 34).move_to([xs["sum"], y, 0])
        total_zero = pill("0", MUTED, CARD, 34).move_to(total)
        labels = m.VGroup(
            text("patch", 24, INK, weight=m.SEMIBOLD).move_to([xs["patch"], y + 1.4, 0]),
            text("filter", 24, AMBER, weight=m.SEMIBOLD).move_to([xs["filter"], y + 1.4, 0]),
            text("pixel × weight", 24, INK, weight=m.SEMIBOLD).move_to([xs["products"], y + 1.4, 0]),
            text("sum", 24, TEAL, weight=m.SEMIBOLD).move_to([xs["sum"], y + 1.4, 0]),
        )
        baseline(labels)
        dark = text("dark", 22, MUTED).move_to([zoom[0][0].get_x(), y - 1.35, 0])
        bright = text("bright", 22, INK).move_to([zoom[0][2].get_x(), y - 1.35, 0])
        baseline([dark, bright])
        strong = text("a strong edge", 22, TEAL, weight=m.SEMIBOLD).next_to(total, m.DOWN, buff=0.25)
        none = text("no edge", 22, MUTED, weight=m.SEMIBOLD).next_to(total, m.DOWN, buff=0.25)

        flat_zoom, flat_products = zoom_of(FLAT), products_of(FLAT)
        flat_sums = sums_of(FLAT, flat_products)

        parts = [(r"S(i,j)", TEAL), ("=", INK), (r"\sum_{m}\sum_{n}", INK), (r"I(i+m,\, j+n)", INK), (r"\cdot", INK), (r"K(m,n)", AMBER)]
        formula = m.MathTex(*[p for p, _ in parts], font_size=40, color=INK)
        for part, (_, c) in zip(formula, parts):
            part.set_color(c)
        formula.move_to([-2.3, -2.25, 0])
        below = formula.get_bottom()[1] - 0.3
        terms = baseline([
            text("output", 22, TEAL, weight=m.SEMIBOLD).move_to([formula[0].get_x(), below, 0]),
            text("sum over 3 × 3", 22, MUTED, weight=m.SEMIBOLD).move_to([formula[2].get_x(), below, 0]),
            text("pixel", 22, INK, weight=m.SEMIBOLD).move_to([formula[3].get_x(), below, 0]),
            text("weight", 22, AMBER, weight=m.SEMIBOLD).move_to([formula[5].get_x(), below, 0]),
        ])
        caveat = m.VGroup(text("Strictly a cross-correlation:", 20, MUTED), text("libraries don't flip the filter", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        caveat.move_to([4.7, -2.4, 0])

        self.say(
            "[Lay it on a patch of the image], [multiply each pixel by the weight on top], [and add all nine up.]",
            m.AnimationGroup(
                self.new_header("ONE STEP", "Multiply, then add"),
                m.FadeIn(house.cells),
                m.Transform(k[0], k_target[0]),
                m.Transform(k[1], k_target[1]),
                m.FadeIn(times),
                m.FadeIn(labels[0]),
                m.FadeIn(labels[1]),
                m.Create(win),
                m.TransformFromCopy(m.VGroup(*[house.at(11 + i, 2 + j) for i in range(3) for j in range(3)]), zoom[0]),
                m.FadeIn(zoom[1]),
            ),
            m.AnimationGroup(m.GrowArrow(equals), m.FadeIn(labels[2]), m.FadeIn(products[0], lag_ratio=0.08), m.FadeIn(products[1], lag_ratio=0.08)),
            m.AnimationGroup(m.GrowArrow(to_sum), m.FadeIn(labels[3]), m.Circumscribe(products[0], color=TEAL, buff=0.06)),
        )
        self.say(
            "[This patch is dark on the left and bright on the right.]",
            m.AnimationGroup(m.FadeIn(dark, shift=0.15 * m.UP), m.FadeIn(bright, shift=0.15 * m.UP)),
        )
        self.say(
            "[The left column adds up to minus four], [the right column to plus thirty-two], [so the sum is twenty-eight]: a strong edge.",
            m.AnimationGroup(m.FadeIn(sums[0], shift=0.15 * m.DOWN), m.FadeIn(sums[1], shift=0.15 * m.DOWN), *[m.Indicate(products[1][i], color=PINK, scale_factor=1.25) for i in (0, 3, 6)]),
            m.AnimationGroup(m.FadeIn(sums[2], shift=0.15 * m.DOWN), *[m.Indicate(products[1][i], color=TEAL, scale_factor=1.25) for i in (2, 5, 8)]),
            m.AnimationGroup(m.FadeIn(total, scale=1.3), m.FadeIn(strong, shift=0.15 * m.UP)),
        )
        self.say(
            "[On a flat patch, the minuses and pluses cancel]: [zero, no edge.]",
            m.AnimationGroup(
                win.animate.move_to(house.window(*FLAT_AT)),
                m.FadeOut(dark),
                m.FadeOut(bright),
                m.Transform(zoom[0], flat_zoom[0]),
                m.FadeTransform(zoom[1], flat_zoom[1]),
                m.FadeTransform(products[1], flat_products[1]),
                *[m.FadeTransform(a, b) for a, b in zip(sums, flat_sums)],
            ),
            m.AnimationGroup(m.FadeTransform(total, total_zero), m.FadeTransform(strong, none)),
        )
        self.say(
            "[In symbols]: [the output at each position] [is the sum over the window] [of each pixel times its weight.]",
            m.Write(formula),
            m.AnimationGroup(m.Indicate(formula[0], color=TEAL), m.FadeIn(terms[0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Indicate(formula[2], color=INK), m.FadeIn(terms[1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(terms[2], shift=0.15 * m.UP), m.FadeIn(terms[3], shift=0.15 * m.UP), m.FadeIn(caveat)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ slide: the feature map

    def feature_map(self):
        self.next_section("The feature map", notes="The filter slid over all 14×14 positions (row 11 first, then the rest); the vertical and the horizontal Sobel maps, computed.")
        house = Board(IMG, 0.26, gray, [-4.45, -0.5, 0])
        fmap = Board(FX, 0.28, heat, [4.25, -0.5, 0], stroke=EDGE, width=1, show=False)
        fmap.cells.set_z_index(1)
        k = kernel_mob(GX, 0.5, 22).move_to([0, 0.95, 0])
        k_label = text("filter", 22, AMBER, weight=m.SEMIBOLD).next_to(k, m.UP, buff=0.2)
        kind = text("vertical edges", 22, MUTED).next_to(k, m.DOWN, buff=0.2)
        kind_h = text("horizontal edges", 22, MUTED).move_to(kind)
        flow = arrow([-1.6, -0.55, 0], [1.6, -0.55, 0], MUTED)
        img_label = text("image", 24, MUTED, weight=m.SEMIBOLD).move_to([-4.45, 2.2, 0])
        map_label = text("feature map", 24, TEAL, weight=m.SEMIBOLD).move_to([4.25, 2.2, 0])
        maps_label = text("feature maps: one per filter", 24, TEAL, weight=m.SEMIBOLD).move_to(map_label)
        win = window_frame(house.cell).move_to(house.window(*ORDER[0]))
        out_win = window_frame(fmap.cell, n=1, width=3.5).move_to(fmap.at(*ORDER[0]))
        value = m.DecimalNumber(0, num_decimal_places=0, mob_class=Mono, font_size=28, color=INK)
        readout = m.VGroup(code("sum =", 26, MUTED), value).arrange(m.RIGHT, buff=0.2).move_to([0, -1.25, 0])
        pos = m.ValueTracker(0)
        shown = {"n": -1}

        def current():
            return min(int(pos.get_value()), len(ORDER) - 1)

        def reveal(group):
            n = current()
            if n == shown["n"]:
                return
            shown["n"] = n
            for idx, (i, j) in enumerate(ORDER):
                fmap.at(i, j).set_fill(heat(FX[i, j]), opacity=1 if idx <= n else 0)

        vertical = edge_lines(house, IMG, TEAL, axis="v", width=4)
        horizontal = edge_lines(house, IMG, TEAL, axis="h", width=4)
        legend = m.VGroup(
            m.VGroup(m.Square(side_length=0.3).set_stroke(width=0).set_fill(TEAL, opacity=1), text("dark → bright", 22, INK)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Square(side_length=0.3).set_stroke(width=0).set_fill(PINK, opacity=1), text("bright → dark", 22, INK)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.5).move_to([4.25, -3.05, 0])
        strong = [fmap.at(i, j) for (i, j), v in np.ndenumerate(FX) if v >= 20]
        weak = [fmap.at(i, j) for (i, j), v in np.ndenumerate(FX) if v <= -20]
        shadows = m.VGroup(*[
            m.Rectangle(width=14 * fmap.cell, height=14 * fmap.cell).set_stroke(EDGE, width=2).set_fill(CARD, opacity=1).move_to(fmap.cells).shift(s * 0.16 * (m.LEFT + m.UP))
            for s in (3, 2, 1)
        ]).set_z_index(0)
        learned = pill("Filters are learned by gradient descent, not designed by hand", VIOLET, VIOLET_FILL, 22).move_to([0, -3.2, 0])

        self.say(
            "[Now let's slide the filter across the whole image.]",
            m.AnimationGroup(
                self.new_header("THE FEATURE MAP", "Slide the filter everywhere"),
                m.FadeIn(house.cells), m.FadeIn(img_label), m.FadeIn(k), m.FadeIn(k_label), m.FadeIn(kind),
                m.GrowArrow(flow), m.FadeIn(fmap.cells), m.FadeIn(map_label), m.FadeIn(readout),
                m.FadeIn(win, scale=1.3), m.FadeIn(out_win, scale=1.3),
            ),
        )
        fmap.cells.add_updater(reveal)
        win.add_updater(lambda f: f.move_to(house.window(*ORDER[current()])))
        out_win.add_updater(lambda f: f.move_to(fmap.at(*ORDER[current()])))
        value.add_updater(lambda d: d.set_value(int(FX[ORDER[current()]])))
        self.say(
            "[One pixel at a time, it writes each sum into a new grid.]",
            pos.animate(run_time=4.2, rate_func=m.linear).set_value(13.999),
        )
        self.say("[Row by row, it sweeps the whole image.]", pos.animate(run_time=3.4, rate_func=m.linear).set_value(len(ORDER) - 0.001))
        self.stop(fmap.cells, win, out_win, value)
        self.say(
            "[The result is a feature map.] [It lights up exactly where the vertical edges are.]",
            m.AnimationGroup(m.FadeOut(win), m.FadeOut(out_win), m.FadeOut(readout), m.Circumscribe(fmap.cells, color=TEAL, buff=0.08)),
            m.AnimationGroup(m.Create(vertical, lag_ratio=0.02)),
        )
        self.say(
            "[Positive where dark turns bright], [negative where bright turns dark.]",
            m.AnimationGroup(m.FadeIn(legend[0], shift=0.15 * m.UP), *[m.Indicate(c, color=TEAL, scale_factor=1.35) for c in strong]),
            m.AnimationGroup(m.FadeIn(legend[1], shift=0.15 * m.UP), *[m.Indicate(c, color=PINK, scale_factor=1.35) for c in weak]),
        )
        turned = []
        for idx, (r, c) in enumerate(np.ndindex(3, 3)):
            spot = k[0][3 * c + (2 - r)].get_center()  # (r, c) → (c, 2 − r): a quarter turn clockwise
            assert GY[c, 2 - r] == GX[r, c]
            turned.append(m.Transform(k[1][idx], k[1][idx].copy().move_to(spot), path_arc=-np.pi / 2))
        self.say(
            "[Turn the filter on its side], [and it finds horizontal edges instead]: [the roof, the window, the ground.]",
            m.AnimationGroup(
                m.FadeOut(vertical),
                *turned,
                m.FadeTransform(kind, kind_h),
            ),
            fmap.recolor(FY, heat),
            m.Create(horizontal, lag_ratio=0.02),
        )
        self.say(
            "[A real layer has many filters, each making its own feature map], [and it learns their weights by gradient descent.]",
            m.AnimationGroup(m.FadeOut(horizontal), m.FadeIn(shadows, lag_ratio=0.3), m.FadeTransform(map_label, maps_label)),
            m.AnimationGroup(m.FadeOut(legend), m.FadeIn(learned, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ weight sharing

    def sharing(self):
        self.next_section("Weight sharing", notes="One 3×3 filter: 9 weights at every position. A dense layer from 256 pixels to the 196 outputs: 256 × 196 = 50,176 weights.")
        house = Board(IMG, 0.22, gray, [-4.9, 0.25, 0])
        house.cells.set_z_index(1)
        spots = [(1, 5), (5, 9), (8, 1), (10, 8), (11, 11)]
        frames = m.VGroup(*[window_frame(house.cell, width=3).move_to(house.window(r, c)) for r, c in spots])
        k = kernel_mob(GX, 0.38, 20).move_to([-1.35, 0.25, 0])
        links = m.VGroup(*[m.Line(f.get_center(), k.get_left()).set_stroke(AMBER, width=2, opacity=0.7) for f in frames]).set_z_index(0)
        caption = text("the same 9 weights at every position", 24, AMBER, weight=m.SEMIBOLD).move_to([-3.4, -1.95, 0])

        x0, full = 0.9, 5.6

        def row(title, sub, y, width, color, value):
            words = m.VGroup(text(title, 26, INK, weight=m.BOLD), text(sub, 22, MUTED)).arrange(m.RIGHT, buff=0.3)
            baseline(words)
            words.move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT)
            bar = m.Rectangle(width=max(width, 0.05), height=0.36, stroke_width=0).set_fill(color, opacity=1)
            bar.move_to([x0, y - 0.6, 0], aligned_edge=m.LEFT)
            number = code(value, 26, INK, weight=m.BOLD).move_to([0, y - 1.15, 0]).align_to([x0, 0, 0], m.LEFT)
            return words, bar, number

        dense = row("Dense layer", "256 pixels × 196 outputs", 1.75, full, BAD, "50,176 weights")
        conv = row("Convolution", "one 3 × 3 filter, shared", -0.35, full * GX.size / DENSE, GOOD, "9 weights")
        still = pill("16 × 16 image or 4,000 × 3,000 photo: still 9 weights", GOOD, GOOD_FILL, 22).move_to([0, -3.2, 0])

        self.say(
            "[Notice: the same nine weights were used at every position.] [That's called weight sharing.]",
            m.AnimationGroup(
                self.new_header("WEIGHT SHARING", "Nine weights, reused everywhere"),
                m.FadeIn(house.cells), m.FadeIn(k), m.Create(frames, lag_ratio=0.2), m.Create(links, lag_ratio=0.2),
            ),
            m.AnimationGroup(m.FadeIn(caption, shift=0.15 * m.UP), m.Circumscribe(k, color=AMBER, buff=0.08)),
        )
        self.say(
            "[A dense layer from our pixels to this map would need over fifty thousand weights.]",
            m.AnimationGroup(m.FadeIn(dense[0], shift=0.15 * m.UP), m.GrowFromEdge(dense[1], m.LEFT), m.FadeIn(dense[2])),
        )
        self.say(
            "[The filter needs just nine], [whether the image is sixteen pixels wide or four thousand.]",
            m.AnimationGroup(m.FadeIn(conv[0], shift=0.15 * m.UP), m.GrowFromEdge(conv[1], m.LEFT), m.FadeIn(conv[2])),
            m.FadeIn(still, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ stride and padding

    def stride_padding(self):
        self.next_section("Stride and padding", notes="Output sizes, computed: stride 1, no padding: 14×14; stride 2: 7×7; padding 1: 16×16 (the real maps are shown).")
        house = Board(IMG, 0.3, gray, [-3.6, -0.5, 0])
        ring_cells, ring_nums = m.VGroup(), m.VGroup()
        for r in range(-1, H + 1):
            for c in range(-1, W + 1):
                if r in (-1, H) or c in (-1, W):
                    sq = m.Square(side_length=house.cell).set_stroke(EDGE, width=1.2).set_fill(DARK, opacity=1).move_to(house.point(r, c))
                    ring_cells.add(sq)
                    ring_nums.add(code("0", 20, MUTED).move_to(sq))
        ring_nums.set_z_index(3)
        win = window_frame(house.cell).move_to(house.window(0, 0))
        centre = [3.7, -0.5, 0]
        out14 = Board(FX, 0.22, heat, centre, stroke=EDGE, width=1)
        out7 = Board(S2, 0.22, heat, centre, stroke=EDGE, width=1)
        out16 = Board(P1, 0.22, heat, centre, stroke=EDGE, width=1)
        setting = text("stride 1 · no padding", 24, AMBER, weight=m.SEMIBOLD).move_to([3.7, 2.1, 0])
        setting2 = text("stride 2 · no padding", 24, AMBER, weight=m.SEMIBOLD).move_to(setting)
        setting3 = text("stride 1 · padding 1", 24, AMBER, weight=m.SEMIBOLD).move_to(setting)
        size = text("14 × 14", 30, INK, weight=m.BOLD).move_to([3.7, -2.75, 0])
        size2 = text("7 × 7", 30, INK, weight=m.BOLD).move_to(size)
        size3 = text("16 × 16", 30, INK, weight=m.BOLD).move_to(size)
        flow = arrow([-0.6, -0.5, 0], [1.6, -0.5, 0], MUTED)

        def hop(r, c, t=0.45):
            return win.animate(run_time=t).move_to(house.window(r, c))

        self.say(
            "[Two settings decide the map's size.] [Stride is how far the filter jumps each step.]",
            m.AnimationGroup(
                self.new_header("STRIDE AND PADDING", "Step size and borders"),
                m.FadeIn(house.cells), m.GrowArrow(flow), m.FadeIn(out14.cells), m.FadeIn(setting), m.FadeIn(size),
            ),
            m.Succession(m.FadeIn(win, scale=1.3, run_time=0.4), hop(0, 1), hop(0, 2), hop(0, 3)),
        )
        self.say(
            "[With a stride of two, it skips every other position], [so the map is half as wide: seven by seven.]",
            m.AnimationGroup(m.FadeTransform(setting, setting2), m.Succession(hop(0, 0, 0.3), hop(0, 2), hop(0, 4), hop(0, 6))),
            m.AnimationGroup(m.ReplacementTransform(out14.cells, out7.cells), m.FadeTransform(size, size2)),
        )
        self.say(
            "[Padding adds a border of zeros], [so the filter also fits over the edge pixels, and the map keeps the image's size.]",
            m.AnimationGroup(m.FadeTransform(setting2, setting3), m.FadeIn(ring_cells, lag_ratio=0.01), m.FadeIn(ring_nums, lag_ratio=0.01)),
            m.AnimationGroup(hop(-1, -1, 0.6), m.ReplacementTransform(out7.cells, out16.cells), m.FadeTransform(size2, size3)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ pooling

    def pooling(self):
        self.next_section("Pooling", notes="ReLU then 2×2 max pooling of the vertical-edge map, computed: 14×14 → 7×7; the block of rows 8–9, columns 0–1 holds 0, 28, 0, 28 → 28.")
        left = Board(FX, 0.3, heat, [-3.4, -0.4, 0], stroke=EDGE, width=1)
        right = Board(PX, 0.6, heat, [3.4, -0.4, 0], stroke=EDGE, width=1.5, show=False)
        left_label = text("feature map", 24, TEAL, weight=m.SEMIBOLD).move_to([-3.4, 2.15, 0])
        relu_label = text("after ReLU", 24, TEAL, weight=m.SEMIBOLD).move_to(left_label)
        right_label = text("pooled map", 24, TEAL, weight=m.SEMIBOLD).move_to([3.4, 2.15, 0])
        flow = arrow([-1.0, -0.4, 0], [1.0, -0.4, 0], MUTED)
        flow_label = text("2 × 2 max", 22, AMBER, weight=m.SEMIBOLD).next_to(flow, m.UP, buff=0.15)
        x0, y0 = left.cells.get_left()[0], left.cells.get_top()[1]
        x1, y1 = left.cells.get_right()[0], left.cells.get_bottom()[1]
        blocks = m.VGroup(
            *[m.Line([x0 + 2 * k * left.cell, y0, 0], [x0 + 2 * k * left.cell, y1, 0]) for k in range(1, 7)],
            *[m.Line([x0, y0 - 2 * k * left.cell, 0], [x1, y0 - 2 * k * left.cell, 0]) for k in range(1, 7)],
        ).set_stroke(AMBER, width=2, opacity=0.7).set_z_index(2)
        r, c = POOL_AT
        block = window_frame(left.cell, n=2).move_to(left.window(2 * r, 2 * c, n=2))
        target = window_frame(right.cell, n=1).move_to(right.at(r, c))
        vals = RX[2 * r : 2 * r + 2, 2 * c : 2 * c + 2].ravel()
        callout = code(f"max({', '.join(str(v) for v in vals)}) = {PX[POOL_AT]}", 26, INK).move_to([0, -3.2, 0])
        size_l = text("after ReLU · 14 × 14", 24, TEAL, weight=m.SEMIBOLD).move_to(left_label)
        size_r = text("pooled map · 7 × 7", 24, TEAL, weight=m.SEMIBOLD).move_to(right_label)
        negatives = [left.at(i, j) for (i, j), v in np.ndenumerate(FX) if v < 0]
        survivors = [right.at(i, j) for (i, j), v in np.ndenumerate(PX) if v >= 20]

        self.say(
            "[First, as in episode three], [a ReLU sets the negative values to zero.]",
            m.AnimationGroup(self.new_header("POOLING", "Keep the strongest response"), m.FadeIn(left.cells), m.FadeIn(left_label)),
            m.AnimationGroup(*[cell.animate.set_fill(ZERO, opacity=1) for cell in negatives], m.FadeTransform(left_label, relu_label)),
        )
        self.say(
            "[Then pooling shrinks the map]: [two by two max pooling keeps only the largest value in each block.]",
            m.AnimationGroup(m.Create(blocks, lag_ratio=0.05), m.GrowArrow(flow), m.FadeIn(flow_label), m.FadeIn(right.cells), m.FadeIn(right_label)),
            m.LaggedStart(*[right.at(i, j).animate.set_fill(heat(v), opacity=1) for (i, j), v in np.ndenumerate(PX)], lag_ratio=0.04),
        )
        self.say(
            "[Here, zero, twenty-eight, zero and twenty-eight become just twenty-eight.]",
            m.AnimationGroup(m.LaggedStart(m.Create(block), m.Create(target), lag_ratio=0.6), m.FadeIn(callout, shift=0.15 * m.UP)),
        )
        self.say(
            "[The map is now a quarter of the size], [but the strong edges survive.]",
            m.AnimationGroup(m.FadeTransform(relu_label, size_l), m.FadeTransform(right_label, size_r)),
            m.AnimationGroup(*[m.Indicate(cell, color=TEAL, scale_factor=1.2) for cell in survivors]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ shifting: equivariance and invariance

    def shifting(self):
        self.next_section(
            "Shifting",
            notes="The house moved one pixel right (computed): the feature map moves exactly one pixel (equivariance); "
            "the pooled block of rows 10–11, columns 2–3 goes from 23, 0, 28, 0 to 28, 23, 28, 28: max 28 both times; other pooled cells do change.",
        )
        y = 0.45
        house = Board(IMG, 0.2, gray, [-4.9, y, 0])
        fmap = Board(RX, 0.22, heat, [-0.2, y, 0], stroke=EDGE, width=1)
        pooled = Board(PX, 0.44, heat, [4.7, y, 0], stroke=EDGE, width=1.2)
        labels = m.VGroup(
            text("image", 24, MUTED, weight=m.SEMIBOLD).move_to([-4.9, y + 1.95, 0]),
            text("feature map", 24, TEAL, weight=m.SEMIBOLD).move_to([-0.2, y + 1.95, 0]),
            text("pooled", 24, TEAL, weight=m.SEMIBOLD).move_to([4.7, y + 1.95, 0]),
        )
        baseline(labels)
        flows = m.VGroup(arrow([-3.6, y, 0], [-1.95, y, 0], MUTED), arrow([1.55, y, 0], [2.95, y, 0], MUTED))

        def marker(board, col, r0, r1):
            x = board.point(0, col)[0] - board.cell / 2
            top, bottom = board.point(r0, 0)[1] + board.cell / 2, board.point(r1, 0)[1] - board.cell / 2
            return m.DashedLine([x, top + 0.15, 0], [x, bottom - 0.15, 0], dash_length=0.08).set_stroke(INK, width=2.5, opacity=0.9).set_z_index(3)

        marks = m.VGroup(marker(house, 3, 8, 13), marker(fmap, 1, 6, 13))
        equi = pill("translation equivariance: shift the image, the map shifts too", TEAL, TEAL_FILL, 22).move_to([0, -2.15, 0])
        inv = pill("pooling: some translation invariance, small shifts matter less", AMBER, AMBER_FILL, 22).move_to([0, -3.0, 0])

        r, c = KEEP
        block = window_frame(fmap.cell, n=2, width=3).move_to(fmap.window(2 * r, 2 * c, n=2))
        cell_ring = window_frame(pooled.cell, n=1, width=3.5).move_to(pooled.at(r, c))

        def mini(values, label, x):
            g = num_grid(values, 0.5, 22, heat, lambda v: BG if v >= 14 else INK)
            name = text(label, 22, MUTED, weight=m.SEMIBOLD)
            result = text(f"max {int(values.max())}", 24, AMBER, weight=m.BOLD)
            return m.VGroup(name, g, result).arrange(m.RIGHT, buff=0.3).move_to([x, -2.55, 0])

        before, after = mini(BEFORE, "before", -3.2), mini(AFTER, "after", 2.2)
        assert BEFORE.max() == AFTER.max() == 28

        self.say(
            "[Now, what if the house] [moves one pixel to the right?]",
            m.AnimationGroup(
                self.new_header("SHIFTING", "Move the house, watch the maps"),
                m.FadeIn(house.cells), m.FadeIn(fmap.cells), m.FadeIn(pooled.cells), m.FadeIn(labels), m.FadeIn(flows), m.FadeIn(marks),
            ),
            house.recolor(IMG2, gray),
        )
        self.say(
            "[The feature map moves one pixel too]: [the same responses, in a new place.] [That's called translation equivariance.]",
            fmap.recolor(RX2, heat),
            m.Indicate(marks[1], color=TEAL, scale_factor=1.0),
            m.FadeIn(equi, shift=0.2 * m.UP),
        )
        self.say(
            "[After pooling, look at this block.] [The values inside changed], [but the largest is still twenty-eight.]",
            m.AnimationGroup(pooled.recolor(PX2, heat), m.Create(block), m.Create(cell_ring), m.FadeOut(equi)),
            m.AnimationGroup(m.FadeIn(before, shift=0.15 * m.UP), m.FadeIn(after[0], shift=0.15 * m.UP), m.FadeIn(after[1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(after[2], scale=1.3), m.Indicate(cell_ring, color=AMBER, scale_factor=1.15)),
        )
        self.say(
            "[Small shifts matter less after pooling]: [it adds some translation invariance, though not complete.]",
            m.AnimationGroup(m.FadeOut(before), *[m.FadeOut(x) for x in after], m.FadeIn(equi, shift=0.2 * m.UP)),
            m.FadeIn(inv, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ stacking layers

    def stacking(self):
        self.next_section("Stacking layers", notes="conv → ReLU → pool, three times; receptive fields 3, 8, 18 pixels (computed). Feature hierarchy as in Episode 3 (illustrative tiles).")
        top = 1.65
        house = Board(IMG, 0.17, gray, [-5.15, -0.55, 0])
        input_label = text("input", 24, MUTED, weight=m.SEMIBOLD).move_to([-5.15, top, 0])
        colors = [(SKY, SKY_FILL), (VIOLET, VIOLET_FILL), (GOOD, GOOD_FILL)]
        xs = [-1.85, 1.35, 4.55]
        blocks = m.VGroup()
        for x, (color, fill) in zip(xs, colors):
            box = card(2.7, 0.85, color, fill, radius=0.2)
            blocks.add(m.VGroup(box, text("conv · ReLU · pool", 20, INK, weight=m.SEMIBOLD).move_to(box)).move_to([x, top, 0]))
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(blocks, blocks[1:])])
        feed = arrow(input_label.get_right(), blocks[0].get_left(), MUTED, buff=0.15)
        kinds = [("vertical", "horizontal"), ("corner", "roof"), ("house", "tree")]
        names = ["edges", "corners and shapes", "objects"]
        columns = m.VGroup()
        for x, (color, _), pair, name in zip(xs, colors, kinds, names):
            tiles = m.VGroup(*[feature_tile(kd, color) for kd in pair]).arrange(m.RIGHT, buff=0.3)
            columns.add(m.VGroup(tiles, text(name, 24, color, weight=m.BOLD)).arrange(m.DOWN, buff=0.3).move_to([x, -0.4, 0]))
        for col in columns[1:]:
            col[1].align_to(columns[0][1], m.DOWN)
        callback = pill("as in Episode 3: simple patterns build complex ones", VIOLET, VIOLET_FILL, 22).move_to([1.35, -2.55, 0])
        centre = house.cells.get_center()
        fields = m.VGroup(*[
            m.Square(side_length=n * house.cell).set_stroke(color, width=4).set_fill(BG, opacity=0).move_to(centre).set_z_index(3)
            for n, (color, _) in zip(RF, colors)
        ])
        sees = m.VGroup(*[
            text(f"layer {i + 1} sees {n} × {n}", 22, color, weight=m.SEMIBOLD) for i, (n, (color, _)) in enumerate(zip(RF, colors))
        ]).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([-5.15, -3.0, 0])

        self.say(
            "[A convolutional network, or CNN, stacks these blocks]: [convolution, ReLU, pooling, again and again.]",
            m.AnimationGroup(self.new_header("STACKING LAYERS", "Edges, then shapes, then objects"), m.FadeIn(house.cells), m.FadeIn(input_label), m.GrowArrow(feed), m.FadeIn(blocks[0], shift=0.2 * m.UP)),
            m.LaggedStart(m.GrowArrow(links[0]), m.FadeIn(blocks[1], shift=0.2 * m.UP), m.GrowArrow(links[1]), m.FadeIn(blocks[2], shift=0.2 * m.UP), lag_ratio=0.35),
        )
        self.say(
            "[The first layer finds edges.] [The next combines them into corners and simple shapes.]",
            m.FadeIn(columns[0], shift=0.2 * m.UP),
            m.FadeIn(columns[1], shift=0.2 * m.UP),
        )
        self.say(
            "[Deeper layers build whole objects, like a house], [just like the features we met in episode three.]",
            m.FadeIn(columns[2], shift=0.2 * m.UP),
            m.FadeIn(callback, shift=0.2 * m.UP),
        )
        self.say(
            "[And each layer sees a wider area of the image]: [three pixels across], [then eight], [then eighteen: the whole house.]",
            m.Indicate(house.cells, color=INK, scale_factor=1.04),
            m.AnimationGroup(m.Create(fields[0]), m.FadeIn(sees[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(fields[1]), m.FadeIn(sees[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(fields[2]), m.FadeIn(sees[2], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the breakthrough

    def breakthrough(self):
        self.next_section(
            "The breakthrough",
            notes="ILSVRC-2012: about 1.2 million training images, 1,000 classes. AlexNet's top-5 test error 15.3% vs 26.2% for the second-best entry "
            "(Krizhevsky, Sutskever & Hinton, NeurIPS 2012). Thumbnails are abstract stand-ins, not dataset images.",
        )
        rng = np.random.default_rng(20)
        palette = [(SKY, SKY_FILL), (GOOD, GOOD_FILL), (PINK, PINK_FILL), (VIOLET, VIOLET_FILL), (AMBER, AMBER_FILL), (TEAL, TEAL_FILL)]
        thumbs = m.VGroup(*[thumbnail(int(rng.integers(0, 4)), *palette[int(rng.integers(0, len(palette)))]) for _ in range(12)])
        thumbs.arrange_in_grid(rows=3, cols=4, buff=(0.15, 0.15)).move_to([-3.6, 0.8, 0])
        facts = m.VGroup(
            text("The ImageNet challenge", 30, INK, weight=m.BOLD),
            text("about 1.2 million training photos", 24, MUTED),
            text("1,000 categories", 24, MUTED),
        ).arrange(m.DOWN, buff=0.16).next_to(thumbs, m.DOWN, buff=0.45)

        x0, per = 1.0, 0.17
        title = text("top-5 error · lower is better", 22, MUTED, weight=m.SEMIBOLD).move_to([0, 1.95, 0]).align_to([x0, 0, 0], m.LEFT)

        def bar_row(name, value, color, y):
            label = text(name, 24, INK, weight=m.BOLD).move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT)
            bar = m.Rectangle(width=value * per, height=0.45, stroke_width=0).set_fill(color, opacity=1).move_to([x0, y - 0.55, 0], aligned_edge=m.LEFT)
            number = code(f"{value}%", 26, INK, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.2)
            return label, bar, number

        alex = bar_row("AlexNet · a convolutional network", ALEX, VIOLET, 1.15)
        other = bar_row("next best entry", RUNNER_UP, FAINT, -0.25)
        meaning = text("top-5: the true label isn't in its 5 best guesses", 20, MUTED).move_to([0, -1.5, 0]).align_to([x0, 0, 0], m.LEFT)
        cite = text("Krizhevsky et al., 2012", 20, MUTED).move_to([0, -1.95, 0]).align_to([x0, 0, 0], m.LEFT)
        boom = pill("a turning point for deep learning", VIOLET, VIOLET_FILL, 24).move_to([0, -3.15, 0])

        self.say(
            "[In twenty twelve, a convolutional network called AlexNet won the ImageNet challenge]: [over a million photos, in a thousand categories.]",
            m.AnimationGroup(self.new_header("THE BREAKTHROUGH", "ImageNet, 2012"), m.FadeIn(thumbs, lag_ratio=0.08)),
            m.FadeIn(facts, shift=0.2 * m.UP, lag_ratio=0.2),
        )
        self.say(
            "[Its top five error rate was about fifteen percent], [against about twenty-six percent for the next best entry.]",
            m.AnimationGroup(m.FadeIn(title), m.FadeIn(alex[0]), m.GrowFromEdge(alex[1], m.LEFT), m.FadeIn(alex[2]), m.FadeIn(cite)),
            m.AnimationGroup(m.FadeIn(other[0]), m.GrowFromEdge(other[1], m.LEFT), m.FadeIn(other[2]), m.FadeIn(meaning)),
        )
        self.say(
            "[That win helped spark the deep learning boom.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(alex[1], alex[2]), color=VIOLET, buff=0.1), m.FadeIn(boom, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Filter, slide, pool.")
        halo = glow(VIOLET, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Filter", "Slide", "Pool"), (AMBER, TEAL, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Small shared filters, stacked deep: that's how networks see.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [filter], [slide], [pool.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Small filters, shared across the image and stacked deep: that's how networks see.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
