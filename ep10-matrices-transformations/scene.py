"""AI, Explained · Episode 10 · Matrices as Transformations

Running example: a grid in the plane, turned and stretched by one matrix,
W = [[1, -1], [1, 1]] (an eighth of a turn plus a stretch by the square root of two).
Three moves (a quarter turn, a stretch, a shear); the columns of W are where the basis
vectors land; y = Wx worked through for x = (3, 1) -> y = (2, 4); shapes (an m x n matrix
times a length-n vector gives length m; 64 pixels -> 16 numbers is a 16 x 64 matrix, 1,024
weights); composing (S R is not R S); a batch of twelve points as the columns of X, Y = WX;
and a layer, ReLU(Wx + b), whose fold is what keeps two layers from collapsing into one.

Every grid picture is the real map: grid lines, arrows and points are sent through the
stage's current matrix (plus bias and ReLU fold) every frame and clipped to the stage. The
numbers the narration quotes are asserted below. Illustrative (made up for teaching): the
2 x 3 example, the 16 x 64 weights and the twelve batch points (seeded numpy), the bias.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------- the numbers

I2 = np.eye(2)
W = np.array([[1.0, -1.0], [1.0, 1.0]])
TURN = float(np.degrees(np.arctan2(W[1, 0], W[0, 0])))
GROW = float(np.hypot(W[0, 0], W[1, 0]))
assert np.isclose(TURN, 45.0) and np.isclose(GROW, np.sqrt(2))  # "an eighth of a turn, and stretches it"
assert np.allclose(W, GROW * rotation(np.pi / 4))
E1, E2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
assert np.allclose(W @ E1, [1, 1]) and np.allclose(W @ E1, W[:, 0])  # "lands at one, one: the first column"
assert np.allclose(W @ E2, [-1, 1]) and np.allclose(W @ E2, W[:, 1])  # "minus one, one: the second column"

X = np.array([3.0, 1.0])  # "x equals three, one"
Y = W @ X
assert np.allclose(Y, [2, 4])  # "y is two, four"
assert np.allclose(X[0] * W[:, 0] + X[1] * W[:, 1], Y)  # "three of the first column, plus one of the second"
assert (W[0, 0], X[0], W[0, 1], X[1], Y[0]) == (1, 3, -1, 1, 2)  # "one times three, minus one times one, is two"
assert (W[1, 0], X[0], W[1, 1], X[1], Y[1]) == (1, 3, 1, 1, 4)  # "one times three, plus one times one, is four"

QUARTER = np.round(rotation(np.pi / 2)) + 0.0
assert np.allclose(QUARTER, rotation(np.pi / 2)) and np.allclose(QUARTER, [[0, -1], [1, 0]])  # "a quarter turn"
WIDE = np.diag([2.0, 0.5])  # "twice as wide, and half as tall"
SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])

# shapes (illustrative numbers)
SMALL_W = np.array([[1, 0, 2], [-1, 3, 1]])
SMALL_X = np.array([2, 1, 1])
SMALL_Y = SMALL_W @ SMALL_X
assert SMALL_W.shape == (2, 3)  # "two rows and three columns"
assert SMALL_X.shape == (3,) and SMALL_Y.shape == (2,)  # "takes in three numbers, and gives back two"
assert list(SMALL_Y) == [4, 2]

SEVEN = [  # the handwritten "7" of Episode 3 (its 8 x 8 pattern, redrawn here)
    "01111110",
    "00000110",
    "00001100",
    "00001000",
    "00011000",
    "00010000",
    "00110000",
    "00100000",
]
_rng = np.random.default_rng(7)
DIGIT = np.array([float(_rng.uniform(0.8, 1.0)) if bit == "1" else 0.0 for row in SEVEN for bit in row])
PIXELS, HIDDEN = 64, 16
assert DIGIT.size == PIXELS  # "the sixty-four pixels"
LAYER = np.random.default_rng(10).normal(0.0, 1.0, (HIDDEN, PIXELS))  # illustrative weights
assert LAYER.shape == (16, 64) and LAYER.size == 1024  # "sixteen by sixty-four", "one thousand and twenty-four weights"
LAYER_OUT = LAYER @ DIGIT
assert LAYER_OUT.shape == (16,)  # "turns them into sixteen numbers"

# composing two moves
R = QUARTER
S = np.diag([2.0, 1.0])  # "stretch twice as wide"
SR, RS = S @ R, R @ S
assert np.allclose(SR, [[0, -2], [1, 0]]) and np.allclose(RS, [[0, -1], [2, 0]])
assert not np.allclose(SR, RS)  # "S times R is not R times S"


# a batch of twelve points (illustrative): spread out, away from the centre
def _batch(seed, n=12):
    return np.round(np.random.default_rng(seed).uniform(-1.6, 1.6, (2, n)), 1)


def _fine(pts):
    gaps = np.linalg.norm(pts[:, :, None] - pts[:, None, :], axis=0)[np.triu_indices(pts.shape[1], 1)]
    return gaps.min() > 0.6 and np.linalg.norm(pts, axis=0).min() > 0.45


BATCH_SEED = next(s for s in range(10_000) if _fine(_batch(s)))
BATCH = _batch(BATCH_SEED)
OUT = W @ BATCH
assert BATCH.shape == (2, 12)  # "twelve points", "two by twelve"
assert W.shape == (2, 2) and OUT.shape == (2, 12)  # "two by two, times two by twelve, gives two by twelve"
assert all(np.allclose(OUT[:, k], W @ BATCH[:, k]) for k in range(12))  # "each column of the result is W times one input"
BIAS = np.array([1.0, 0.5])  # illustrative


# ---------------------------------------------------------------------------- helpers

MINUS = "−"
MAIN = dict(center=(-3.35, -0.5), width=6.4, height=5.9, unit=0.66)


def num(v, places=None):
    """A number as the screen shows it: a real minus sign, no trailing zeros unless `places`."""
    if isinstance(v, str):
        return v
    v = float(v)
    body = f"{abs(v):g}" if places is None else f"{abs(v):.{places}f}"
    return (MINUS if v < 0 and float(body) != 0 else "") + body


class Mat(m.VGroup):
    """A matrix (or a column vector) in the series' monospace font between thin brackets,
    entries right-aligned in their columns, one colour per column."""

    def __init__(self, rows, size=36, colors=None, color=INK, gap=0.42, row_gap=0.22, bracket_color=INK, places=None, pad=0.22):
        rows = [list(r) for r in rows]
        nr, nc = len(rows), len(rows[0])
        cols = colors or [color] * nc
        digit = code("0", size).height

        def entry(v, c):
            if v == "⋯":  # drawn, not typeset: the glyph is too thin to read
                return m.VGroup(*[m.Dot(radius=0.032 * size / 22, color=cols[c]) for _ in range(3)]).arrange(m.RIGHT, buff=0.06 * size / 22)
            return code(num(v, places), size, cols[c])

        cells = [[entry(v, c) for c, v in enumerate(row)] for row in rows]
        widths = [max(cells[r][c].width for r in range(nr)) for c in range(nc)]
        right = np.cumsum(widths) + gap * np.arange(nc)
        for r in range(nr):
            base = -r * (digit + row_gap)
            for c in range(nc):
                cell = cells[r][c]
                if isinstance(cell, m.Text):
                    cell.move_to([right[c] - cell.width / 2, base + cell.height / 2, 0])
                else:  # an ellipsis: centred in its column, at mid-digit height
                    cell.move_to([right[c] - widths[c] / 2, base + digit / 2, 0])
        entries = m.VGroup(*[cell for row in cells for cell in row])
        top, bottom = digit + 0.13, -(nr - 1) * (digit + row_gap) - 0.13

        def bracket(x, lip):
            corners = [[x + lip, top, 0], [x, top, 0], [x, bottom, 0], [x + lip, bottom, 0]]
            return m.VMobject().set_points_as_corners(corners).set_stroke(bracket_color, width=3)

        lb = bracket(entries.get_left()[0] - pad, 0.12)
        rb = bracket(entries.get_right()[0] + pad, -0.12)
        super().__init__(lb, entries, rb)
        self.cells, self.entries, self.brackets = cells, entries, m.VGroup(lb, rb)

    def col(self, c):
        return [row[c] for row in self.cells]

    def row(self, r):
        return list(self.cells[r])


def equals(width=0.36, color=INK, stroke=4):
    """An equals sign drawn as two strokes (a lone typeset "=" is too short for the layout check)."""
    return m.VGroup(*[m.Line([-width / 2, y, 0], [width / 2, y, 0]).set_stroke(color, width=stroke) for y in (0.065, -0.065)])


def underline(mob, color=INK, buff=0.09, width=3):
    return m.Line([mob.get_left()[0] - 0.03, mob.get_bottom()[1] - buff, 0], [mob.get_right()[0] + 0.03, mob.get_bottom()[1] - buff, 0]).set_stroke(color, width=width)


def baseline_row(words):
    """Words in a row on one baseline, measured on glyphs without descenders."""
    def foot(w):
        sitting = [g for g, c in zip(w, w.text.replace(" ", "")) if c not in "gjpqy,"] or list(w)
        return min(g.get_bottom()[1] for g in sitting)

    for w in words[1:]:
        w.shift((foot(words[0]) - foot(w)) * m.UP)
    return words


def each(make, mobs):
    """One animation per mobject, together (so no new group joins the scene)."""
    return m.AnimationGroup(*[make(mob) for mob in mobs])


def polyline(points, color, width=3.0, opacity=1.0):
    return m.VMobject().set_points_as_corners([[x, y, 0] for x, y in points]).set_stroke(color, width=width, opacity=opacity)


def clip(p, q, box):
    """Liang-Barsky: the part of the segment p -> q inside box (x0, x1, y0, y1), as (t0, t1)."""
    x0, x1, y0, y1 = box
    d = q - p
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-d[0], p[0] - x0), (d[0], x1 - p[0]), (-d[1], p[1] - y0), (d[1], y1 - p[1])):
        if abs(pk) < 1e-12:
            if qk < 0:
                return None
        elif pk < 0:
            t0 = max(t0, qk / pk)
        else:
            t1 = min(t1, qk / pk)
    return (t0, t1) if t0 < t1 else None


def clipped(points, box):
    """A polyline (N x 2 screen points) cut to the box: the runs that stay inside."""
    runs, run = [], []
    for p, q in zip(points[:-1], points[1:]):
        t = clip(p, q, box)
        if t is None:
            if run:
                runs.append(run)
                run = []
            continue
        a, b = p + t[0] * (q - p), p + t[1] * (q - p)
        if run and t[0] == 0.0:
            run.append(b)
        else:
            if run:
                runs.append(run)
            run = [a, b]
        if t[1] < 1.0:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def tag(label, color, size=24, math=False):
    """A label on a small opaque patch, so grid lines pass behind it."""
    words = tex(label, size + 10, color) if math else code(label, size, color)
    box = card(words.width + 0.28, words.height + 0.22, BG, BG, radius=0.1, stroke=0)
    box.set_fill(BG, opacity=0.92)
    return m.VGroup(box, words.move_to(box)).set_z_index(4)


def outline(mobs, color, buff=0.1, width=3):
    group = m.VGroup(*mobs) if isinstance(mobs, (list, tuple)) else mobs
    return m.RoundedRectangle(
        corner_radius=0.12, width=group.width + 2 * buff, height=group.height + 2 * buff,
        stroke_color=color, stroke_width=width, fill_opacity=0,
    ).move_to(group)


def formula(parts, colors, size=56):
    f = m.MathTex(*parts, font_size=size, color=INK)
    for i, c in colors.items():
        f[i].set_color(c)
    return f


class Stage:
    """A clipped window onto the plane. Its grid, square, arrows and dots are redrawn every
    frame from the stage's state: a matrix M, a shift b and a ReLU fold r in [0, 1], each
    point p going to (1 - r) q + r max(q, 0) with q = M p + b. Animations change the state;
    `plan_*` is the state the scene will be in once the animations built so far have played
    (live mobjects are first drawn from it, so they can be introduced mid-film)."""

    def __init__(self, center, width, height, unit, band=0.0, span=12):
        cx, cy = center
        self.frame = card(width, height, EDGE, CARD, radius=0.2, stroke=2).move_to([cx, cy, 0])
        self.frame.set_fill(CARD, opacity=0.75).set_z_index(-3)
        pad = 0.07
        self.box = (cx - width / 2 + pad, cx + width / 2 - pad, cy - height / 2 + pad, cy + height / 2 - pad - band)
        self.o = np.array([(self.box[0] + self.box[1]) / 2, (self.box[2] + self.box[3]) / 2])
        self.u, self.span = unit, span
        self.m = [m.ValueTracker(v) for v in (1.0, 0.0, 0.0, 1.0)]
        self.b = [m.ValueTracker(0.0), m.ValueTracker(0.0)]
        self.r = m.ValueTracker(0.0)
        self.plan_M, self.plan_b, self.plan_r = I2.copy(), np.zeros(2), 0.0
        self.last = None
        self.live = []
        self.base = self.lines(I2, np.zeros(2), 0.0, EDGE, 1.6, 1.0, 2.6, 1.0, axis_color=FAINT).set_z_index(-2)

    # ---- state and geometry

    def state(self):
        M = np.array([[t.get_value() for t in self.m[:2]], [t.get_value() for t in self.m[2:]]])
        return M, np.array([t.get_value() for t in self.b]), self.r.get_value()

    def planned(self):
        return self.plan_M.copy(), self.plan_b.copy(), self.plan_r

    @staticmethod
    def send(pts, M, b, r):
        q = np.asarray(pts, float) @ M.T + b
        return (1 - r) * q + r * np.maximum(q, 0.0) if r > 0 else q

    def screen(self, q):
        return self.o + self.u * np.asarray(q, float)

    def point(self, q, M=None, b=None, r=None):
        """Where grid point q is drawn under a state (default: the planned one)."""
        M = self.plan_M if M is None else np.asarray(M, float)
        b = self.plan_b if b is None else np.asarray(b, float)
        r = self.plan_r if r is None else r
        x, y = self.screen(self.send(np.atleast_2d(q), M, b, r))[0]
        return np.array([x, y, 0.0])

    def lines(self, M, b, r, color, width=2.0, opacity=0.55, axis_width=3.5, axis_opacity=0.95, axis_color=None):
        group = m.VGroup()
        n = self.span
        for k in range(-n, n + 1):
            for start, step in (((k, -n), (0, 1)), ((-n, k), (1, 0))):
                A, D = np.array(start, float), np.array(step, float)
                ts = [0.0, 2.0 * n]
                if r > 0:  # the fold bends a line exactly where a coordinate crosses zero
                    q0, dq = M @ A + b, M @ D
                    ts += [-q0[i] / dq[i] for i in range(2) if abs(dq[i]) > 1e-9 and 0 < -q0[i] / dq[i] < 2 * n]
                pts = A + np.outer(np.sort(ts), D)
                for run in clipped(self.screen(self.send(pts, M, b, r)), self.box):
                    axis = k == 0
                    group.add(polyline(run, (axis_color or color) if axis else color, axis_width if axis else width, axis_opacity if axis else opacity))
        return group

    def snapshot(self, M, color=AMBER, width=4):
        """The grid under M, as a still picture (for a flash)."""
        return self.lines(np.asarray(M, float), np.zeros(2), 0.0, color, width, 1.0, width, 1.0)

    # ---- live mobjects

    def _live(self, build, z):
        mob = build(*self.planned()).set_z_index(z)
        mob.add_updater(lambda mob: mob.become(build(*self.state()).set_z_index(z)))
        self.live.append(mob)
        return mob

    def grid(self):
        return self._live(lambda M, b, r: self.lines(M, b, r, BLUE), -1)

    def square(self, color=AMBER):
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)

        def build(M, b, r):
            pts = self.screen(self.send(corners, M, b, r))
            return m.Polygon(*[[x, y, 0] for x, y in pts]).set_stroke(color, width=3).set_fill(color, opacity=0.22)

        return self._live(build, 0)

    def arrow(self, a, b, color, width=6):
        ends = np.array([a, b], float)

        def build(M, bb, r):
            p, q = self.screen(self.send(ends, M, bb, r))
            return m.Arrow(
                [*p, 0], [*q, 0], buff=0, color=color, stroke_width=width,
                max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=14,
            )

        return self._live(build, 1)

    def dots(self, pts, color, radius=0.08):
        pts = np.asarray(pts, float)

        def build(M, b, r):
            return m.VGroup(*[m.Dot([x, y, 0], radius=radius, color=color) for x, y in self.screen(self.send(pts, M, b, r))])

        return self._live(build, 2)

    def stop(self):
        for mob in self.live:
            mob.clear_updaters()

    # ---- animations of the state

    def _go(self, path, run_time):
        def update(_, alpha):
            for t, v in zip(self.m, path(alpha).flat):
                t.set_value(float(v))

        return m.UpdateFromAlphaFunc(self.m[0], update, run_time=run_time)

    def apply(self, P, turn=False, run_time=1.6):
        """Apply P after what is there; `turn` for P = s x rotation: turn and grow together."""
        A, P = self.plan_M.copy(), np.asarray(P, float)
        if turn:
            s, a = float(np.hypot(P[0, 0], P[1, 0])), float(np.arctan2(P[1, 0], P[0, 0]))
            assert np.allclose(P, s * rotation(a))

            def path(t):
                return (1 + t * (s - 1)) * rotation(t * a) @ A
        else:

            def path(t):
                return ((1 - t) * I2 + t * P) @ A

        self.plan_M, self.last = P @ A, path
        return self._go(path, run_time)

    def to(self, M, run_time=1.2):
        A, B = self.plan_M.copy(), np.asarray(M, float)

        def path(t):
            return (1 - t) * A + t * B

        self.plan_M, self.last = B.copy(), path
        return self._go(path, run_time)

    def rewind(self, run_time=0.9):
        """Undo the last move, back along its own path."""
        path, self.last = self.last, None
        self.plan_M = path(0.0)
        return self._go(lambda t: path(1 - t), run_time)

    def slide(self, b, run_time=1.4):
        start, end = self.plan_b.copy(), np.asarray(b, float)
        self.plan_b = end

        def update(_, t):
            for tr, v in zip(self.b, (1 - t) * start + t * end):
                tr.set_value(float(v))

        return m.UpdateFromAlphaFunc(self.b[0], update, run_time=run_time)

    def fold(self, r=1.0, run_time=2.0):
        start = self.plan_r
        self.plan_r = r
        return m.UpdateFromAlphaFunc(self.r, lambda _, t: self.r.set_value(start + t * (r - start)), run_time=run_time)


# ---------------------------------------------------------------------------- the episode


class Ep10MatricesTransformations(Episode):
    number = 10

    def lesson(self):
        self.hook()
        self.moves()
        self.insight()
        self.product()
        self.shapes()
        self.compose()
        self.batch()
        self.layer()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A layer moves every point at once; the recipe for the move is a matrix, one number per connection.")
        st = Stage(**MAIN)
        grid = st.grid()
        ins = [m.Circle(radius=0.3).set_stroke(SKY, width=3).set_fill(SKY_FILL, opacity=1).move_to([2.25, y, 0]) for y in (1.95, 0.65)]
        outs = [m.Circle(radius=0.3).set_stroke(GOOD, width=3).set_fill(GOOD_FILL, opacity=1).move_to([4.95, y, 0]) for y in (1.95, 0.65)]
        edges = [[m.Line(ins[j].get_center(), outs[i].get_center(), buff=0.34).set_stroke(AMBER, width=3) for j in range(2)] for i in range(2)]
        names = m.VGroup(
            *[tex(f"x_{j + 1}", 32, SKY).next_to(ins[j], m.LEFT, buff=0.2) for j in range(2)],
            *[tex(f"y_{i + 1}", 32, GOOD).next_to(outs[i], m.RIGHT, buff=0.2) for i in range(2)],
        )
        caption = text("one layer", 22, MUTED).move_to([3.6, -0.05, 0])
        w_name = formula(["W", "="], {0: AMBER}, 48)
        w_mat = Mat(W, 40)
        m.VGroup(w_name, w_mat).arrange(m.RIGHT, buff=0.3).move_to([3.6, -1.55, 0])
        note = pill("a matrix: one number per connection", AMBER, AMBER_FILL, 22).move_to([3.6, -2.95, 0])

        self.say(
            "So... [what does one layer of a neural network actually do] to its input?",
            m.AnimationGroup(
                m.FadeIn(st.frame), m.FadeIn(st.base), m.FadeIn(grid),
                m.LaggedStart(*[m.Create(e) for row in edges for e in row], lag_ratio=0.15),
                *[m.FadeIn(c, scale=0.6) for c in ins + outs],
                m.FadeIn(names),
                m.FadeIn(caption),
            ),
        )
        self.say(
            "[It moves every point in space at once]: [turning the grid], [and stretching it.]",
            m.LaggedStart(*[m.ShowPassingFlash(line, time_width=0.6) for line in st.snapshot(I2, SKY, 4)], lag_ratio=0.02, run_time=1.6),
            st.apply(rotation(np.pi / 4), turn=True, run_time=1.6),
            st.apply(GROW * I2, run_time=1.4),
        )
        assert np.allclose(st.plan_M, W)
        self.say(
            "[And the recipe for that move is a small table of numbers, one per connection]: [a matrix.]",
            m.AnimationGroup(
                m.FadeIn(w_name),
                m.FadeIn(w_mat.brackets),
                m.LaggedStart(*[m.TransformFromCopy(edges[i][j], w_mat.cells[i][j]) for i in range(2) for j in range(2)], lag_ratio=0.2),
            ),
            m.FadeIn(note, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        st.stop()
        self.clear()

    # ------------------------------------------------------------------ three moves

    def moves(self):
        self.next_section("Three moves", notes="A quarter turn [[0,-1],[1,0]], a stretch [[2,0],[0,0.5]], a shear [[1,1],[0,1]]; lines stay straight, parallel, evenly spaced; the origin stays: a linear map.")
        st = self.stage = Stage(**MAIN)
        grid, square = st.grid(), st.square()
        mats = [Mat(M, 44).move_to([3.6, 1.3, 0]) for M in (QUARTER, WIDE, SHEAR)]
        captions = [text(s, 28, INK, weight=m.SEMIBOLD).move_to([3.6, -0.05, 0]) for s in ("a quarter turn", "a stretch", "a shear")]
        facts = m.VGroup(
            text("lines stay straight, parallel, evenly spaced", 22, MUTED),
            text("and the center stays put", 22, MUTED),
        ).arrange(m.DOWN, buff=0.18).move_to([3.6, -1.5, 0])
        linear = pill("a linear map", AMBER, AMBER_FILL, 26).move_to([3.6, -2.75, 0])

        self.say(
            "[Picture a flat grid, with one small square marked on it.]",
            m.AnimationGroup(
                self.new_header("THREE MOVES", "A matrix moves the whole grid"),
                m.FadeIn(st.frame), m.FadeIn(st.base), m.FadeIn(grid), m.FadeIn(square),
            ),
        )
        self.say(
            "[This matrix] [turns the whole grid a quarter turn.]",
            m.AnimationGroup(m.FadeIn(mats[0], shift=0.2 * m.UP), m.FadeIn(captions[0], shift=0.2 * m.UP)),
            st.apply(QUARTER, turn=True, run_time=1.8),
        )
        self.say(
            "[This one stretches it]: [twice as wide, and half as tall.]",
            m.AnimationGroup(st.rewind(0.8), m.FadeTransform(mats[0], mats[1]), m.FadeTransform(captions[0], captions[1])),
            st.apply(WIDE, run_time=1.8),
        )
        self.say(
            "[And this one shears it], [like a deck of cards pushed sideways.]",
            m.AnimationGroup(st.rewind(0.8), m.FadeTransform(mats[1], mats[2]), m.FadeTransform(captions[1], captions[2])),
            st.apply(SHEAR, run_time=1.8),
        )
        flash = st.snapshot(SHEAR, AMBER, 4)
        self.say(
            "[In every case, lines stay straight, parallel and evenly spaced], [and the center stays put.] [That's called a linear map.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.ShowPassingFlash(line, time_width=0.6) for line in flash], lag_ratio=0.03, run_time=2.2),
                m.FadeIn(facts[0], shift=0.15 * m.UP),
            ),
            m.AnimationGroup(m.Flash(st.point((0, 0)), color=AMBER, flash_radius=0.4), m.FadeIn(facts[1], shift=0.15 * m.UP)),
            m.FadeIn(linear, scale=1.15),
        )
        self.wait(0.3)
        self.grid, self.square = grid, square
        self.right_side = [mats[2], captions[2], facts, linear]

    # ------------------------------------------------------------------ the key insight

    def insight(self):
        self.next_section("The key insight", notes="W = [[1,-1],[1,1]]: an eighth of a turn and a stretch by sqrt 2. The basis vectors land on its columns, (1,1) and (-1,1).")
        st = self.stage
        header = m.AnimationGroup(
            self.new_header("THE KEY INSIGHT", "Columns are where the arrows land"),
            *[m.FadeOut(mob) for mob in self.right_side],
            st.rewind(1.0),
        )
        e1, e2 = st.arrow((0, 0), (1, 0), TEAL), st.arrow((0, 0), (0, 1), PINK)
        e1_tag = tag("e_1", TEAL, math=True).move_to(st.point((1, 0), I2) + np.array([0.15, -0.42, 0]))
        e2_tag = tag("e_2", PINK, math=True).move_to(st.point((0, 1), I2) + np.array([-0.5, 0.12, 0]))
        w_name = formula(["W", "="], {0: AMBER}, 50)
        w_mat = Mat(W, 44, gap=0.6, pad=0.3)
        w_row = m.VGroup(w_name, w_mat).arrange(m.RIGHT, buff=0.3).move_to([3.6, 1.3, 0])
        box1, box2 = outline(w_mat.col(0), TEAL, 0.1), outline(w_mat.col(1), PINK, 0.1)
        land1 = tag("(1, 1)", TEAL).move_to(st.point((1, 0), W) + np.array([0.82, 0.12, 0]))
        land2 = tag(f"({MINUS}1, 1)", PINK).move_to(st.point((0, 1), W) + np.array([-0.95, 0.12, 0]))
        rule = pill("columns = where the basis vectors land", AMBER, AMBER_FILL, 22).move_to([3.6, -0.75, 0])
        then = text("know those two, and you know every point", 22, MUTED).move_to([3.6, -1.5, 0])

        self.say(
            "[Now meet the two basis vectors]: [one step to the right], [and one step up.]",
            header,
            m.AnimationGroup(m.GrowArrow(e1), m.FadeIn(e1_tag)),
            m.AnimationGroup(m.GrowArrow(e2), m.FadeIn(e2_tag)),
        )
        self.say(
            "[Let's apply our matrix W], [which turns the grid an eighth of a turn, and stretches it.]",
            m.AnimationGroup(m.FadeIn(w_row, shift=0.2 * m.UP), m.FadeOut(e1_tag), m.FadeOut(e2_tag)),
            st.apply(W, turn=True, run_time=2.4),
        )
        self.say(
            "[The first basis vector lands at one, one]: [exactly the first column of W.]",
            m.FadeIn(land1, scale=0.8),
            m.AnimationGroup(each(lambda c: c.animate.set_color(TEAL), w_mat.col(0)), m.Create(box1)),
        )
        self.say(
            "[The second lands at minus one, one]: [the second column.]",
            m.FadeIn(land2, scale=0.8),
            m.AnimationGroup(each(lambda c: c.animate.set_color(PINK), w_mat.col(1)), m.Create(box2)),
        )
        self.say(
            "[So the columns of a matrix are where the basis vectors land.] [Know those two, and you know where every point goes.]",
            m.FadeIn(rule, shift=0.2 * m.UP),
            m.AnimationGroup(
                m.LaggedStart(*[m.ShowPassingFlash(line, time_width=0.6) for line in st.snapshot(W, AMBER, 4)], lag_ratio=0.03, run_time=2.2),
                m.FadeIn(then, shift=0.15 * m.UP),
            ),
        )
        self.wait(0.3)
        self.e1, self.e2 = e1, e2
        self.right_side = [w_row, box1, box2, land1, land2, rule, then]

    # ------------------------------------------------------------------ y = Wx

    def product(self):
        self.next_section("y = Wx", notes="x = (3, 1): three of e1 plus one of e2, so y = 3 x column 1 + 1 x column 2 = (2, 4); row by row 1x3 - 1x1 = 2, 1x3 + 1x1 = 4.")
        st = self.stage
        for mob in (self.e1, self.e2, self.square):
            mob.clear_updaters()
        header = m.AnimationGroup(
            self.new_header("MATRIX TIMES VECTOR", "y = Wx, with real numbers"),
            *[m.FadeOut(mob) for mob in (*self.right_side, self.e1, self.e2, self.square)],
            st.rewind(1.0),
        )
        top = formula(["y", "=", "W", "x"], {0: GOOD, 2: AMBER, 3: SKY}, 60).move_to([3.6, 2.0, 0])
        y_col = Mat([[Y[0]], [Y[1]]], 36, color=GOOD)
        w_m = Mat(W, 36, colors=[TEAL, PINK], gap=0.5, pad=0.3)
        x_col = Mat([[X[0]], [X[1]]], 36, color=SKY, pad=0.3)
        eq = equals()
        m.VGroup(y_col, eq, w_m, x_col).arrange(m.RIGHT, buff=0.3).move_to([3.6, 0.5, 0])

        def worked(r):
            terms = [f"{num(abs(W[r, c]) if c else W[r, c])}×{num(X[c])}" for c in range(2)]
            lhs = terms[0] + (f" {MINUS} " if W[r, 1] < 0 else " + ") + terms[1] + " ="
            return m.VGroup(code(lhs, 26, INK), code(num(Y[r]), 26, GOOD)).arrange(m.RIGHT, buff=0.2)

        sums = m.VGroup(worked(0), worked(1)).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([3.6, -1.3, 0])
        assert sums[0][0].text == f"1×3 {MINUS} 1×1 =" and sums[1][0].text == "1×3 + 1×1 ="
        weighted = pill("each output: a weighted sum", AMBER, AMBER_FILL, 22).move_to([3.6, -2.85, 0])
        row_box = [outline(w_m.row(r), INK, 0.09) for r in range(2)]
        x_box = outline(x_col.entries, SKY, 0.09)

        steps = [st.arrow((k, 0), (k + 1, 0), TEAL, 6) for k in range(3)] + [st.arrow((3, 0), (3, 1), PINK, 6)]
        x_arrow = st.arrow((0, 0), X, SKY, 7)
        x_tag = tag("(3, 1)", SKY).move_to(st.point(X, I2) + np.array([0.0, 0.48, 0]))
        tip = st.point(X, W)
        y_arrow = m.Arrow(st.point((0, 0), W), tip, buff=0, color=GOOD, stroke_width=7, max_tip_length_to_length_ratio=0.3).set_z_index(3)
        y_tag = tag("(2, 4)", GOOD).move_to(tip + np.array([-0.9, 0.0, 0]))
        corners = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]
        pulses = [
            m.Line(st.point(a, W), st.point(b, W)).set_stroke(TEAL if k < 3 else PINK, width=12)
            for k, (a, b) in enumerate(zip(corners, corners[1:]))
        ]

        self.say(
            "[Let's send one vector through.] [Take x equals three, one]: [three steps right, one step up.]",
            m.AnimationGroup(header, m.Write(top)),
            m.AnimationGroup(m.FadeIn(y_col.brackets), m.FadeIn(eq), m.FadeIn(w_m), m.FadeIn(x_col)),
            m.AnimationGroup(m.GrowArrow(x_arrow), m.FadeIn(x_tag)),
        )
        self.say(
            "[That's three of the first basis vector, plus one of the second.]",
            m.Succession(*[m.GrowArrow(a, run_time=0.5) for a in steps]),
        )
        self.say(
            "[Apply W, and the recipe still holds]: [three of the first column, plus one of the second.]",
            m.AnimationGroup(st.apply(W, turn=True, run_time=2.2), m.FadeOut(x_tag)),
            m.AnimationGroup(
                m.Succession(*[m.ShowPassingFlash(p, time_width=0.9, run_time=0.55) for p in pulses]),
                m.Succession(
                    each(lambda c: m.Indicate(c, color=TEAL, scale_factor=1.25), w_m.col(0)),
                    each(lambda c: m.Indicate(c, color=PINK, scale_factor=1.25), w_m.col(1)),
                ),
            ),
        )
        self.say(
            "[Row by row: one times three, minus one times one], [is two.]",
            m.AnimationGroup(m.Create(row_box[0]), m.Create(x_box), m.FadeIn(sums[0][0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(sums[0][1], scale=1.3), m.FadeIn(y_col.cells[0][0], scale=1.3)),
        )
        self.say(
            "[And one times three, plus one times one], [is four.]",
            m.AnimationGroup(m.ReplacementTransform(row_box[0], row_box[1]), m.FadeIn(sums[1][0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(sums[1][1], scale=1.3), m.FadeIn(y_col.cells[1][0], scale=1.3)),
        )
        self.say(
            "[So y is two, four], [exactly where our arrow landed.]",
            m.AnimationGroup(m.FadeOut(row_box[1]), m.FadeOut(x_box), each(lambda c: m.Indicate(c, color=GOOD, scale_factor=1.3), [y_col.cells[0][0], y_col.cells[1][0]])),
            m.AnimationGroup(m.GrowArrow(y_arrow), m.FadeIn(y_tag), m.Flash(tip, color=GOOD, flash_radius=0.4)),
        )
        self.say(
            "[Each output is a weighted sum of the inputs], [just like a neuron from episode three.]",
            each(lambda s: m.Indicate(s, color=INK, scale_factor=1.06), [sums[0], sums[1]]),
            m.FadeIn(weighted, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        st.stop()
        self.clear()

    # ------------------------------------------------------------------ shapes

    def shapes(self):
        self.next_section("Shapes", notes="An m x n matrix times a length-n vector gives length m. Illustrative: the 2 x 3 example; a 64 -> 16 layer (16 x 64 = 1,024 seeded weights) applied to Episode 3's digit.")
        sw = Mat(SMALL_W, 36, color=AMBER)
        sx = Mat([[v] for v in SMALL_X], 36, color=SKY)
        eq = equals()
        sy = Mat([[v] for v in SMALL_Y], 36, color=GOOD)
        example = m.VGroup(sw, sx, eq, sy).arrange(m.RIGHT, buff=0.35).move_to([0, 1.25, 0])
        low = example.get_bottom()[1] - 0.38
        t_w = text(f"{SMALL_W.shape[0]} × {SMALL_W.shape[1]}", 26, AMBER, weight=m.SEMIBOLD).move_to([sw.get_x(), low, 0])
        t_x = text(f"{SMALL_X.shape[0]}", 26, SKY, weight=m.SEMIBOLD).move_to([sx.get_x(), low, 0])
        t_y = text(f"{SMALL_Y.shape[0]}", 26, GOOD, weight=m.SEMIBOLD).move_to([sy.get_x(), low, 0])

        def column(sym, shape, color):
            big = tex(sym, 64, color)
            small = m.MathTex(*shape, font_size=38, color=MUTED)
            return m.VGroup(big, small).arrange(m.DOWN, buff=0.22)

        cw, cx, cy = column("W", ["m", r"\times", "n"], AMBER), column("x", ["n"], SKY), column("y", ["m"], GOOD)
        eq2 = equals(0.42)
        rule = m.VGroup(cw, cx, eq2, cy).arrange(m.RIGHT, buff=0.55, aligned_edge=m.UP).move_to([0, -1.65, 0])
        for c in (cx, cy):
            c[1].align_to(cw[1], m.DOWN)
        eq2.set_y(cw[0].get_y())
        inner = [underline(cw[1][2]), underline(cx[1][0]), underline(glyphs(t_w, "3")), underline(t_x)]
        bad = pill("2 × 3  times  2:  undefined", BAD, BAD_FILL, 22).move_to([0, -3.3, 0])

        # the 64 -> 16 layer, drawn to scale: one cell per number
        cell = 0.065
        grid = m.VGroup()
        for k, v in enumerate(DIGIT):
            sq = m.Square(side_length=0.2).set_stroke(EDGE, width=1.2).set_fill(INK, opacity=v)
            grid.add(sq.move_to([(k % 8) * 0.2, -(k // 8) * 0.2, 0]))
        grid.move_to([-3.39, -0.45, 0])
        grid_label = text(f"{PIXELS} pixels", 22, MUTED).next_to(grid, m.DOWN, buff=0.3)
        x_cells = m.VGroup(*[m.Rectangle(width=0.26, height=cell).set_stroke(width=0).set_fill(SKY, opacity=0.1 + 0.9 * v) for v in DIGIT])
        x_cells.arrange(m.DOWN, buff=0).move_to([2.9, -0.45, 0])
        lo, hi = LAYER.min(), LAYER.max()
        rows = m.VGroup()
        for r in range(HIDDEN):
            rows.add(m.VGroup(*[m.Square(side_length=cell).set_stroke(width=0).set_fill(AMBER, opacity=0.08 + 0.92 * (w - lo) / (hi - lo)) for w in LAYER[r]]).arrange(m.RIGHT, buff=0))
        rows.arrange(m.DOWN, buff=0).move_to([0.39, -0.45, 0])
        ylo, yhi = LAYER_OUT.min(), LAYER_OUT.max()
        y_cells = m.VGroup(*[m.Rectangle(width=0.26, height=cell).set_stroke(width=0).set_fill(GOOD, opacity=0.1 + 0.9 * (v - ylo) / (yhi - ylo)) for v in LAYER_OUT])
        y_cells.arrange(m.DOWN, buff=0).move_to([4.06, -0.45, 0])
        eq3 = equals().move_to([3.48, -0.45, 0])
        frames = [outline(g, c, 0.04, 2) for g, c in ((rows, AMBER), (x_cells, SKY), (y_cells, GOOD))]
        for f in frames:
            f.set_z_index(1)
        labels = [
            text(f"W: {HIDDEN} × {PIXELS}", 24, AMBER, weight=m.SEMIBOLD).next_to(rows, m.UP, buff=0.3),
            text(f"x: {PIXELS}", 24, SKY, weight=m.SEMIBOLD).next_to(x_cells, m.UP, buff=0.25),
            text(f"y: {HIDDEN}", 24, GOOD, weight=m.SEMIBOLD).next_to(y_cells, m.UP, buff=0.3),
        ]
        labels[2].align_to(labels[0], m.DOWN)
        weights = pill(f"{HIDDEN} × {PIXELS} = {LAYER.size:,} weights", AMBER, AMBER_FILL, 24).move_to([0.39, -3.25, 0])

        self.say(
            "[A matrix doesn't have to be square.] [This one, for example, has two rows and three columns.]",
            m.AnimationGroup(self.new_header("SHAPES", "Rows out, columns in"), m.FadeIn(sw, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(t_w, shift=0.15 * m.UP), m.Succession(*[each(lambda c: m.Indicate(c, color=INK, scale_factor=1.2), sw.row(r)) for r in range(2)])),
        )
        self.say(
            "[It takes in three numbers], [and gives back two.]",
            m.AnimationGroup(m.FadeIn(sx, shift=0.2 * m.LEFT), m.FadeIn(t_x)),
            m.AnimationGroup(m.FadeIn(eq), m.FadeIn(sy, shift=0.2 * m.LEFT), m.FadeIn(t_y)),
        )
        self.say(
            "[The rule: an m by n matrix], [times a vector of length n], [gives a vector of length m.]",
            m.FadeIn(cw, shift=0.2 * m.UP),
            m.FadeIn(cx, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(eq2), m.FadeIn(cy, shift=0.2 * m.UP)),
        )
        self.say(
            "[The two inner numbers must match], [or the product isn't defined.]",
            m.LaggedStart(*[m.Create(b) for b in inner], lag_ratio=0.2),
            m.FadeIn(bad, shift=0.2 * m.UP),
        )
        self.say(
            "[For example, take the sixty-four pixels of the digit from episode three], [as one long column.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (sw, sx, eq, sy, t_w, t_x, t_y, cw, cx, eq2, cy, bad, *inner)],
                m.FadeIn(grid),
                m.FadeIn(grid_label),
            ),
            m.AnimationGroup(
                m.LaggedStart(*[m.TransformFromCopy(sq, c) for sq, c in zip(grid, x_cells)], lag_ratio=0.02),
                m.FadeIn(frames[1]),
                m.FadeIn(labels[1]),
            ),
        )
        self.say(
            "[A layer that turns them into sixteen numbers is a sixteen by sixty-four matrix]: [one thousand and twenty-four weights.]",
            m.Succession(
                m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r) for r in rows], lag_ratio=0.12), m.FadeIn(frames[0]), m.FadeIn(labels[0]), run_time=2.0),
                m.AnimationGroup(m.FadeIn(eq3), m.FadeIn(y_cells), m.FadeIn(frames[2]), m.FadeIn(labels[2])),
            ),
            m.FadeIn(weights, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ composing

    def compose(self):
        self.next_section("Composing", notes="R = quarter turn, S = stretch x2 wide. S R = [[0,-2],[1,0]] (rotate first) and R S = [[0,-1],[2,0]] (stretch first) differ.")
        panels = []
        for x, title in ((-3.4, "Rotate (R), then stretch (S)"), (3.4, "Stretch (S), then rotate (R)")):
            st = Stage(center=(x, 0.55), width=6.3, height=4.1, unit=0.5, band=0.62)
            heading = text(title, 24, INK, weight=m.SEMIBOLD).move_to([x, 0.55 + 2.05 - 0.36, 0])
            parts = (st.grid(), st.square(), st.arrow((0, 0), (1, 0), TEAL, 5), st.arrow((0, 0), (0, 1), PINK, 5))
            panels.append((st, heading, parts))
        (left, l_head, l_parts), (right, r_head, r_parts) = panels

        def product(name, M, x):
            label = formula(list(name) + ["="], {}, 40)
            return m.VGroup(label, Mat(M, 32)).arrange(m.RIGHT, buff=0.25).move_to([x, -2.3, 0])

        sr, rs = product("SR", SR, -3.4), product("RS", RS, 3.4)
        law = formula(["S", "(", "R", "x", ")", "=", "(", "S", "R", ")", "x"], {3: SKY, 10: SKY}, 42).move_to([0, -3.3, 0])
        differ = formula(["S", "R", r"\neq", "R", "S"], {2: BAD}, 46).move_to([0, -3.3, 0])

        def shape(st, M):
            return m.Polygon(*[st.point(p, M) for p in ((0, 0), (1, 0), (1, 1), (0, 1))])

        def show(st, heading, parts):
            return m.AnimationGroup(m.FadeIn(st.frame), m.FadeIn(st.base), m.FadeIn(heading), *[m.FadeIn(p) for p in parts])

        self.say(
            "[What if we apply one matrix, and then another?] [Say, rotate a quarter turn], [then stretch twice as wide.]",
            m.AnimationGroup(self.new_header("COMPOSING", "One move after another"), show(left, l_head, l_parts)),
            left.apply(R, turn=True, run_time=1.6),
            left.apply(S, run_time=1.4),
        )
        self.say(
            "[Together, the two steps are a single matrix]: [the product, S times R.]",
            m.Succession(left.to(I2, 0.7), left.to(SR, 1.5)),
            m.FadeIn(sr, shift=0.2 * m.UP),
        )
        self.say(
            "[The matrix next to x acts first], [so we read the product from right to left.]",
            m.Write(law),
            m.Succession(
                each(lambda p: m.Indicate(p, color=AMBER, scale_factor=1.4), [law[2], law[8]]),
                each(lambda p: m.Indicate(p, color=AMBER, scale_factor=1.4), [law[0], law[7]]),
            ),
        )
        self.say(
            "[Now swap the order]: [stretch first], [then rotate.]",
            show(right, r_head, r_parts),
            right.apply(S, run_time=1.4),
            right.apply(R, turn=True, run_time=1.6),
        )
        self.say(
            "[A different result.] [Order matters]: [S times R is not R times S.]",
            m.AnimationGroup(
                m.FadeIn(rs, shift=0.2 * m.UP),
                m.Circumscribe(shape(left, SR), color=AMBER, buff=0.12),
                m.Circumscribe(shape(right, RS), color=AMBER, buff=0.12),
            ),
            m.AnimationGroup(m.Circumscribe(sr, color=INK, buff=0.1), m.Circumscribe(rs, color=INK, buff=0.1)),
            m.FadeTransform(law, differ),
        )
        self.wait(0.3)
        left.stop()
        right.stop()
        self.clear()

    # ------------------------------------------------------------------ batches

    def batch(self):
        self.next_section("Batches", notes=f"Twelve illustrative points (numpy seed {BATCH_SEED}) as the columns of X (2 x 12); Y = WX (2 x 12). Code libraries often store inputs as rows instead (X W^T): the same products.")
        st = self.stage = Stage(**MAIN)
        grid = st.grid()
        points = st.dots(BATCH.T, SKY)
        top = formula(["Y", "=", "W", "X"], {0: GOOD, 2: AMBER, 3: SKY}, 56).move_to([3.6, 2.0, 0])
        shown = [0, 1, 2, None, 11]

        def table(values, color):
            rows = [[("⋯" if k is None else values[r, k]) for k in shown] for r in range(2)]
            return Mat(rows, 22, color=color, gap=0.3, places=1)

        xm, ym = table(BATCH, SKY), table(OUT, GOOD)
        x_name, y_name = formula(["X", "="], {0: SKY}, 44), formula(["Y", "="], {0: GOOD}, 44)
        m.VGroup(x_name, xm).arrange(m.RIGHT, buff=0.25).move_to([3.6, 0.6, 0])
        m.VGroup(y_name, ym).arrange(m.RIGHT, buff=0.25).move_to([3.6, -0.95, 0])
        ym.shift((xm.get_x() - ym.get_x()) * m.RIGHT)
        y_name.next_to(ym, m.LEFT, buff=0.25)
        sizes = m.VGroup(
            text("2 × 2", 26, AMBER, weight=m.SEMIBOLD), text("times", 24, MUTED),
            text("2 × 12", 26, SKY, weight=m.SEMIBOLD), text("gives", 24, MUTED),
            text("2 × 12", 26, GOOD, weight=m.SEMIBOLD),
        ).arrange(m.RIGHT, buff=0.25).move_to([3.6, -2.45, 0])
        baseline_row(sizes)
        boxes = [outline(xm.col(0), SKY, 0.08), outline(ym.col(0), GOOD, 0.08)]
        x_cols = [m.VGroup(*xm.col(c)) for c in range(len(shown))]
        moves = []  # copies of the live dots fly into X's columns (the hidden ones into the ellipsis)
        for k in range(12):
            slot = shown.index(k) if k in shown else 3
            if slot == 3:
                moves.append(m.FadeOut(points[k].copy(), target_position=x_cols[3].get_center(), scale=0.5))
            else:
                moves.append(m.TransformFromCopy(points[k], x_cols[slot]))

        self.say(
            "[Networks rarely see one input at a time.] [Here's a batch of twelve points], for example.",
            m.AnimationGroup(self.new_header("BATCHES", "Many inputs, one multiplication"), m.FadeIn(st.frame), m.FadeIn(st.base), m.FadeIn(grid)),
            m.FadeIn(points, scale=0.6),
        )
        self.say(
            "[Stack them side by side, as the twelve columns of one matrix, X.]",
            m.AnimationGroup(m.FadeIn(x_name), m.FadeIn(xm.brackets), m.FadeIn(x_cols[3]), m.LaggedStart(*moves, lag_ratio=0.1)),
        )
        self.say(
            "[Now one multiplication, W times X], [moves every point at once.]",
            m.Write(top),
            st.apply(W, turn=True, run_time=2.4),
        )
        self.say(
            "[Each column of the result is W times one input.] [The shapes still line up]: [two by two, times two by twelve, gives two by twelve.]",
            m.AnimationGroup(m.FadeIn(y_name), m.FadeIn(ym, shift=0.2 * m.UP), m.Create(boxes[0]), m.Create(boxes[1])),
            m.AnimationGroup(m.FadeOut(boxes[0]), m.FadeOut(boxes[1])),
            m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.UP) for s in sizes], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.points = points
        self.right_side = [top, x_name, xm.brackets, *x_cols, y_name, ym, sizes]

    # ------------------------------------------------------------------ the layer

    def layer(self):
        self.next_section("A layer", notes="y = ReLU(Wx + b) with the illustrative bias b = (1, 0.5): W turns and stretches, b slides, ReLU folds negative coordinates to zero. Without the fold, W2(W1x) = (W2W1)x: two layers collapse into one (Episode 3's straight line).")
        st = self.stage
        header = m.AnimationGroup(
            self.new_header("A LAYER", "Multiply, add, fold"),
            *[m.FadeOut(mob) for mob in self.right_side],
            st.rewind(1.0),
        )
        top = formula(["y", "=", r"\mathrm{ReLU}", "(", "W", "x", "+", "b", ")"], {0: GOOD, 2: LIME, 4: AMBER, 5: SKY, 7: VIOLET}, 54).move_to([3.6, 2.0, 0])
        rows = m.VGroup(
            m.VGroup(formula(["W", "x"], {0: AMBER, 1: SKY}, 40), text("turns and stretches the grid", 22, MUTED)),
            m.VGroup(formula(["+", "b"], {1: VIOLET}, 40), text("slides it over", 22, MUTED)),
            m.VGroup(formula([r"\mathrm{ReLU}"], {0: LIME}, 40), text("folds negatives up to zero", 22, MUTED)),
        )
        for row in rows:
            row.arrange(m.RIGHT, buff=0.35)
        rows.arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT).move_to([3.5, 0.45, 0])
        column = max(r[0].get_right()[0] for r in rows) + 0.35
        for row in rows:
            row[1].shift((column - row[1].get_left()[0]) * m.RIGHT)
        note = text("no fold: two layers collapse into one", 22, MUTED).move_to([3.6, -1.3, 0])
        collapse = tex(r"W_2(W_1x) = (W_2W_1)\,x", 40).move_to([3.6, -2.05, 0])
        callback = pill("why Episode 3 needed an activation", AMBER, AMBER_FILL, 22).move_to([3.6, -3.0, 0])
        moved = st.point((0, 0), W, BIAS, 0.0)
        centre = m.Dot(moved, radius=0.09, color=VIOLET).set_z_index(3)

        self.say(
            "[Now we can say exactly what a layer does.] [First, W times x turns and stretches the space.]",
            m.AnimationGroup(header, m.Write(top)),
            m.AnimationGroup(
                st.apply(W, turn=True, run_time=2.2),
                m.FadeIn(rows[0], shift=0.15 * m.UP),
                each(lambda p: m.Indicate(p, color=AMBER, scale_factor=1.3), [top[4], top[5]]),
            ),
        )
        self.say(
            "[Then the bias, b, slides everything over], [so the center can move too.]",
            m.AnimationGroup(st.slide(BIAS, 1.6), m.FadeIn(rows[1], shift=0.15 * m.UP), each(lambda p: m.Indicate(p, color=VIOLET, scale_factor=1.3), [top[6], top[7]])),
            m.AnimationGroup(m.FadeIn(centre, scale=0.5), m.Flash(moved, color=VIOLET, flash_radius=0.4)),
        )
        self.say(
            "[Finally, the activation, ReLU, folds every negative number up to zero.]",
            m.AnimationGroup(st.fold(1.0, 2.6), m.FadeIn(rows[2], shift=0.15 * m.UP), m.Indicate(top[2], color=LIME, scale_factor=1.2)),
        )
        self.say(
            "[Without that fold, two layers would collapse into one]: [their matrices just multiply.]",
            m.FadeIn(note, shift=0.15 * m.UP),
            m.Write(collapse),
        )
        self.say(
            "[That's why, in episode three, layers without an activation only ever drew a straight line.]",
            m.AnimationGroup(m.FadeIn(callback, shift=0.2 * m.UP), m.Circumscribe(rows[2], color=LIME, buff=0.1)),
        )
        self.wait(0.3)
        st.stop()
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Columns, shapes, layers.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Columns", "Shapes", "Layers"), (TEAL, AMBER, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A layer turns and stretches space, then folds it.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [columns], [shapes], [layers.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A layer turns and stretches space with a matrix, then folds it with an activation.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
