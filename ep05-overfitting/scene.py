"""AI, Explained · Episode 5 · Overfitting & Generalization

Running example: twenty noisy points sampled from a smooth curve, fitted with polynomials of
degree 1 (underfit), 3 (good fit) and 15 (wiggly overfit), and fifteen held-out test points.
Every curve on screen is computed here with numpy from fixed seeds: least-squares fits
(np.polyfit), training and test errors (mean squared error), ridge regression for the
regularization demo, and gradient descent (in closed form, from zero weights) for early
stopping. Inputs x run from -1 to 1; the hidden curve is y = -sin(pi x), noise sd 0.3.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the data

SEED, N, NOISE, N_TEST = 3297, 20, 0.3, 15
SAMPLE_SEEDS = (4, 19, 65, 74, 118, 165)  # six fresh samples for the bias-variance picture


def truth(x):
    return -np.sin(np.pi * x)


def sample(seed, n=N, n_test=N_TEST):
    """n training points (spread out, at least a little apart) and n_test held-out points."""
    rng = np.random.default_rng(seed)
    while True:
        u = np.sort(rng.uniform(0.03, 0.97, n))
        if np.diff(u).min() > 0.018:
            break
    x = 2 * u - 1
    y = truth(x) + NOISE * rng.normal(0, 1, n)
    xt = 2 * rng.uniform(0.08, 0.92, n_test) - 1
    yt = truth(xt) + rng.normal(0, NOISE, n_test)
    return x, y, xt, yt


def polyfit(x, y, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # degree 15 is poorly conditioned; the fit is still exact least squares
        return np.polyfit(x, y, d)


def mse(c, x, y):
    return float(np.mean((np.polyval(c, x) - y) ** 2))


X, Y, XT, YT = sample(SEED)
FITS = {d: polyfit(X, Y, d) for d in range(1, 16)}
TRAIN = {d: mse(FITS[d], X, Y) for d in FITS}
TEST = {d: mse(FITS[d], XT, YT) for d in FITS}
BEST = min(TEST, key=TEST.get)  # 3


def _miss(d, i):
    return abs(float(np.polyval(FITS[d], XT[i])) - YT[i])


HERO = max(range(N_TEST), key=lambda i: _miss(15, i) if _miss(3, i) < 0.15 else 0.0)  # degree 15's worst miss

# more data: a hundred more points from the same curve, the same noise
_rng = np.random.default_rng(SEED + 1)
XM = _rng.uniform(X[0], X[-1], 100)
YM = truth(XM) + NOISE * _rng.normal(0, 1, 100)
MORE = polyfit(np.concatenate([X, XM]), np.concatenate([Y, YM]), 15)

# regularization: minimise  mean squared error + lam * (sum of squared weights), degree 15
PHI = np.vander(X, 16)


def ridge(lam):
    return np.linalg.solve(PHI.T @ PHI / N + lam * np.eye(16), PHI.T @ Y / N)


RIDGE_FROM, RIDGE_TO = -18.0, -3.0  # log10 of the penalty strength

# early stopping: plain gradient descent on the same loss (no penalty), from zero weights,
# step 1/L; after t steps the weights are exactly sum_k (1 - (1 - eta e_k)^t) (u_k . y / s_k) v_k
_U, _S, _VT = np.linalg.svd(PHI, full_matrices=False)
_EIG = 2 * _S**2 / N
_ETA = 1 / _EIG.max()
_PROJ = _U.T @ Y


def descent(t):
    return _VT.T @ ((1 - (1 - _ETA * _EIG) ** t) * _PROJ / _S)


LOG_T = np.linspace(0, 14, 141)
GD_TRAIN = np.array([mse(descent(10**lt), X, Y) for lt in LOG_T])
GD_HELD = np.array([mse(descent(10**lt), XT, YT) for lt in LOG_T])
GD_STOP = float(LOG_T[int(np.argmin(GD_HELD))])

OTHERS = [sample(s)[:2] for s in SAMPLE_SEEDS]


# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """An icon in another accent, like kit.recolored(), but saved as assets/ep05-….svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    stem = name if name.startswith("ep05-") else f"ep05-{name}"
    out = f"{stem}-{color.lstrip('#').lower()}"
    svg = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{out}.svg"
    if not path.exists() or path.read_text() != svg:
        path.write_text(svg)
    return out


def mix(a, b, t):
    """A colour between two hex colours."""
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


class Plot:
    """Quiet axes for inputs x in [-1, 1] (vertical axis at the left edge) and outputs y in
    [-y_max, y_max]; graphs are cut where they leave the box (wild fits run off the chart)."""

    def __init__(self, width, height, center, y_max=2.0, labels=True):
        self.ax = axes([0, 2.1, 0.25], [-y_max, y_max, 1], width, height).move_to(center)
        self.y_max = y_max
        self.frame = m.VGroup(self.ax)
        if labels:
            self.frame.add(
                text("x", 24, MUTED).next_to(self.ax.x_axis.get_end(), m.DOWN, buff=0.15),
                text("y", 24, MUTED).next_to(self.ax.y_axis.get_end(), m.RIGHT, buff=0.18),
            )

    def p(self, x, y):
        return self.ax.c2p(x + 1, y)

    def _cross(self, xa, ya, xb, yb):
        edge = self.y_max if max(ya, yb) > self.y_max else -self.y_max
        t = (edge - ya) / (yb - ya)
        return self.p(xa + t * (xb - xa), edge)

    def graph(self, fn, x0, x1, color, width=4, opacity=1.0, n=500):
        xs = np.linspace(x0, x1, n)
        ys = np.asarray(fn(xs), dtype=float)
        pieces, run = [], []
        for i, (x, y) in enumerate(zip(xs, ys)):
            if abs(y) <= self.y_max:
                if not run and i > 0:
                    run.append(self._cross(xs[i - 1], ys[i - 1], x, y))
                run.append(self.p(x, y))
            elif run:
                run.append(self._cross(xs[i - 1], ys[i - 1], x, y))
                pieces.append(run)
                run = []
        if run:
            pieces.append(run)
        curves = [polyline(r, color, width, opacity) for r in pieces if len(r) > 1]
        return curves[0] if len(curves) == 1 else m.VGroup(*curves)

    def fit(self, coefs, color, width=4, x0=None, x1=None, **kw):
        return self.graph(lambda xs: np.polyval(coefs, xs), X[0] if x0 is None else x0, X[-1] if x1 is None else x1, color, width, **kw)

    def dots(self, xs, ys, color=SKY, radius=0.085):
        return m.VGroup(*[m.Dot(self.p(x, y), radius=radius, color=color) for x, y in zip(xs, ys)])

    def rings(self, xs, ys, color=PINK, radius=0.1):
        return m.VGroup(
            *[m.Circle(radius=radius).set_stroke(color, width=3.5).set_fill(BG, opacity=1).move_to(self.p(x, y)) for x, y in zip(xs, ys)]
        )


def blend(d):
    """The fitted curve for a degree between two whole degrees (for a smooth sweep)."""
    lo = int(np.clip(np.floor(d), 1, 15))
    hi = min(lo + 1, 15)
    a = float(np.clip(d - lo, 0, 1))
    return lambda xs: (1 - a) * np.polyval(FITS[lo], xs) + a * np.polyval(FITS[hi], xs)


def regime(d):
    """Amber while too simple, green around the sweet spot, red as it overfits."""
    if d < 3:
        return mix(AMBER, GOOD, d - 2)
    return mix(GOOD, BAD, (d - 7) / 5)


def swatch(kind, color, width=0.95, height=0.42):
    xs = np.linspace(-0.5, 0.5, 120)
    ys = {"line": 0.7 * xs, "s": 0.5 * np.sin(2 * np.pi * xs) * 0.9, "wild": 0.5 * np.sin(9 * np.pi * xs) * (0.4 + 1.2 * np.abs(xs))}[kind]
    return polyline([[x * width, y * height, 0] for x, y in zip(xs, ys)], color, width=4)


# ---------------------------------------------------------------------------- the episode


class Ep05Overfitting(Episode):
    number = 5

    def lesson(self):
        self.hook()
        self.idea()
        self.example()
        self.fitting()
        self.test()
        self.big_picture()
        self.bias_variance()
        self.fixes()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Perfect on the training data is not the same as a good model.")
        halo = glow(GOOD, radius=3.4).move_to(1.0 * m.UP)
        score = text("100%", 150, GOOD, weight=m.BOLD).move_to([0, 1.05, 0])
        caption = text("correct on its training examples", 30, MUTED).move_to([0, -0.35, 0])
        mark = text("?", 150, INK, weight=m.BOLD)
        total = score.width + 0.3 + mark.width
        mark.move_to([total / 2 - mark.width / 2, score.get_y(), 0]).align_to(score, m.DOWN)
        shift = (-total / 2 + score.width / 2) * m.RIGHT
        memo = pill("memorized the answers", BAD, BAD_FILL, 28)
        learn = pill("learned the pattern", GOOD, GOOD_FILL, 28)
        either = text("or", 28, MUTED)
        choice = m.VGroup(memo, either, learn).arrange(m.RIGHT, buff=0.45).move_to([0, -2.05, 0])

        self.say(
            "So... [imagine a model that gets every single training example right.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(score, shift=0.3 * m.UP), m.FadeIn(caption), run_time=1.2),
        )
        self.say(
            "[Is that a great model?] [Not necessarily.]",
            m.AnimationGroup(score.animate.shift(shift), m.FadeIn(mark, shift=0.3 * m.LEFT)),
            m.Wiggle(m.VGroup(score, mark), scale_value=1.04),
        )
        self.say(
            "[It might have simply memorized the answers], [instead of learning the pattern behind them.]",
            m.FadeIn(memo, shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(either), m.FadeIn(learn, shift=0.3 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the idea: two students

    def idea(self):
        self.next_section("The idea", notes="Two students: one memorizes last year's answers, one understands them.")

        def panel(x, color, fill, name, how, new_score, new_color, new_fill):
            box = card(6.1, 4.6, color, CARD, stroke=2).move_to([x, -0.35, 0])
            who = icon(variant("ep05-student", color, fill), 1.3)
            title = text(name, 30, color, weight=m.BOLD)
            desc = text(how, 22, MUTED)
            words = m.VGroup(title, desc).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
            top = m.VGroup(who, words).arrange(m.RIGHT, buff=0.4)
            top.move_to(box.get_top() + 1.0 * m.DOWN).align_to(box, m.LEFT).shift(0.45 * m.RIGHT)
            rule = m.Line(box.get_left() + 0.35 * m.RIGHT, box.get_right() + 0.35 * m.LEFT).set_stroke(EDGE, width=2)
            rule.set_y(box.get_top()[1] - 1.95)
            rows = []
            for k, (label, score, c, f) in enumerate((("Last year's paper", "100%", GOOD, GOOD_FILL), ("New questions", new_score, new_color, new_fill))):
                y = box.get_top()[1] - 2.65 - 0.95 * k
                lab = text(label, 26, INK).move_to([0, y, 0]).align_to(box, m.LEFT).shift(0.5 * m.RIGHT)
                tag = pill(score, c, f, 26).move_to([0, y, 0]).align_to(box, m.RIGHT).shift(0.5 * m.LEFT)
                rows.append((lab, tag))
            return m.VGroup(box, rule), who, title, desc, rows

        left = panel(-3.35, BAD, BAD_FILL, "The memorizer", "learns last year's\nanswers by heart", "35%", BAD, BAD_FILL)
        right = panel(3.35, GOOD, GOOD_FILL, "The understander", "learns why each\nanswer is right", "90%", GOOD, GOOD_FILL)
        general = pill("Generalization: doing well on examples you've never seen", SKY, SKY_FILL, 24).move_to([0, -3.3, 0])

        self.say(
            "[Think of two students preparing for an exam.]",
            m.AnimationGroup(
                self.new_header("THE IDEA", "Memorizing versus understanding"),
                *[m.FadeIn(p[0]) for p in (left, right)],
                *[m.FadeIn(p[1], scale=0.8) for p in (left, right)],
                *[m.FadeIn(p[2], shift=0.2 * m.UP) for p in (left, right)],
            ),
        )
        self.say("[The first one memorizes the answers to last year's paper.]", m.AnimationGroup(m.FadeIn(left[3], shift=0.2 * m.UP), m.Indicate(left[1], color=BAD, scale_factor=1.08)))
        self.say("[The second one learns why each answer is right.]", m.AnimationGroup(m.FadeIn(right[3], shift=0.2 * m.UP), m.Indicate(right[1], color=GOOD, scale_factor=1.08)))
        self.say(
            "[On last year's paper, they both score a hundred percent.]",
            m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(p[4][0][0]), m.FadeIn(p[4][0][1], scale=1.3)) for p in (left, right)]),
        )
        self.say(
            "[But the real exam has new questions.] [The memorizer is lost], [while the student who understands does fine.]",
            m.AnimationGroup(*[m.FadeIn(p[4][1][0], shift=0.2 * m.UP) for p in (left, right)]),
            m.Succession(m.FadeIn(left[4][1][1], scale=1.3), m.Wiggle(left[4][1][1], scale_value=1.1)),
            m.FadeIn(right[4][1][1], scale=1.3),
        )
        self.say(
            "[Doing well on examples you've never seen is called generalization.] [It's what we really want from a model.]",
            m.FadeIn(general, shift=0.2 * m.UP),
            m.Circumscribe(general, color=SKY, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the running example

    def example(self):
        self.next_section("The example", notes="Twenty noisy points from a smooth curve; the polynomial model.")
        plot = Plot(6.2, 4.6, [-3.75, -0.4, 0])
        self.plot = plot
        hidden = m.DashedVMobject(plot.graph(truth, -1, 1, MUTED, width=3), num_dashes=70)
        hidden_label = text("the hidden pattern", 22, MUTED).move_to(plot.p(0.5, -1.5))
        dots = plot.dots(X, truth(X))
        dots.set_z_index(3)
        self.dots = dots

        formula = tex(r"\hat{y} = w_0 + w_1 x + w_2 x^2 + \cdots + w_d\, x^d", 36, INK).move_to([3.4, 1.85, 0])
        caption = text("a polynomial: powers of x, each with a weight", 22, MUTED).next_to(formula, m.DOWN, buff=0.25)
        degree_note = m.VGroup(text("d", 26, AMBER, weight=m.BOLD, slant=m.ITALIC), text("the degree: the highest power", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        degree_note.next_to(caption, m.DOWN, buff=0.3)
        rows = m.VGroup()
        for kind, label, desc, color in (("line", "degree 1", "a straight line", AMBER), ("s", "degree 3", "a gentle S-shaped curve", GOOD), ("wild", "degree 15", "a very bendy curve", BAD)):
            row = m.VGroup(swatch(kind, color), text(label, 26, color, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.RIGHT, buff=0.3)
            rows.add(row)
        rows.arrange(m.DOWN, buff=0.32, aligned_edge=m.LEFT).move_to([3.4, -0.85, 0])
        for row in rows[1:]:  # labels in one column
            row[1].align_to(rows[0][1], m.LEFT)
            row[2].next_to(row[1], m.RIGHT, buff=0.3)
        squares = pill("training: the smallest squared error", VIOLET, VIOLET_FILL, 24).move_to([3.4, -2.7, 0])
        self.right_side = m.VGroup(formula, caption, degree_note, rows, squares)

        self.say(
            "[Let's watch this happen with real numbers.] [Here's a smooth curve: the hidden pattern.]",
            m.AnimationGroup(self.new_header("THE EXAMPLE", "Noisy points from a smooth curve"), m.Create(plot.frame)),
            m.AnimationGroup(m.Create(hidden, run_time=1.6), m.FadeIn(hidden_label)),
        )
        self.say(
            "[We take twenty points from it], [each nudged by a bit of random noise, like real measurements.]",
            m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.08),
            m.LaggedStart(*[d.animate.move_to(plot.p(x, y)) for d, x, y in zip(dots, X, Y)], lag_ratio=0.05, run_time=2.0),
        )
        self.say(
            "[Then we hide the curve.] [A model only ever gets to see the points.]",
            m.AnimationGroup(m.FadeOut(hidden), m.FadeOut(hidden_label)),
            m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.6) for d in dots], lag_ratio=0.06),
        )
        self.say(
            "[Our model is a polynomial]: [a sum of powers of x, each with its own weight.]",
            m.Write(formula, run_time=1.2),
            m.FadeIn(caption, shift=0.15 * m.UP),
        )
        self.say(
            "[The highest power is called the degree.] [The higher the degree, the more weights, and the more the curve can bend.]",
            m.FadeIn(degree_note, shift=0.15 * m.UP),
            m.LaggedStart(*[m.AnimationGroup(m.Create(r[0]), m.FadeIn(r[1:], shift=0.15 * m.RIGHT)) for r in rows], lag_ratio=0.35),
        )
        self.say(
            "[Training picks the weights that make the squared error on these points as small as possible.]",
            m.FadeIn(squares, shift=0.2 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ three fits

    def scoreboard(self):
        box = card(5.9, 3.55, EDGE, CARD).move_to([3.85, -0.35, 0])
        xs = [1.7, 3.15, 4.6, 5.95]
        top = box.get_top()[1] - 0.6

        def head(lines, color):
            return m.VGroup(*[text(s, 24, color, weight=m.SEMIBOLD) for s in lines]).arrange(m.DOWN, buff=0.08)

        heads = m.VGroup(head(["degree"], MUTED), head(["training", "error"], SKY), head(["test", "error"], PINK), head(["verdict"], MUTED))
        for h, x in zip(heads, xs):
            h.move_to([x, top, 0])
        heads[0].align_to(heads[1], m.DOWN)
        heads[3].align_to(heads[1], m.DOWN)
        rule = m.Line([box.get_left()[0] + 0.3, top - 0.52, 0], [box.get_right()[0] - 0.3, top - 0.52, 0]).set_stroke(EDGE, width=2)
        rows = {}
        for k, (d, color, fill) in enumerate(((1, AMBER, AMBER_FILL), (3, GOOD, GOOD_FILL), (15, BAD, BAD_FILL))):
            y = top - 1.1 - 0.68 * k
            rows[d] = dict(
                chip=chip(str(d), color, fill, width=0.85, height=0.56, size=26).move_to([xs[0], y, 0]),
                train=code(f"{TRAIN[d]:.3f}", 26, SKY).move_to([xs[1], y, 0]),
                test=code(f"{TEST[d]:.3f}", 26, PINK).move_to([xs[2], y, 0]),
                y=y,
            )
        verdicts = {
            1: text("underfit", 24, AMBER, weight=m.BOLD),
            3: text("good fit", 24, GOOD, weight=m.BOLD),
            15: text("overfit", 24, BAD, weight=m.BOLD),
        }
        for d, v in verdicts.items():
            v.move_to([xs[3], rows[d]["y"], 0])
        best = text("best?", 24, INK, weight=m.BOLD).move_to([xs[3], rows[15]["y"], 0])
        return box, heads, rule, rows, verdicts, best

    def fitting(self):
        self.next_section("Three models", notes="Degree 1, 3 and 15 fitted by least squares; training error only.")
        plot, dots = self.plot, self.dots
        c1, c3, c15 = (plot.fit(FITS[d], c, width=4.5) for d, c in ((1, AMBER), (3, GOOD), (15, BAD)))
        self.curves = {1: c1, 3: c3, 15: c15}
        box, heads, rule, rows, verdicts, best = self.scoreboard()
        self.board = dict(rows=rows, verdicts=verdicts, best=best)
        gaps = m.VGroup(*[m.Line(plot.p(x, y), plot.p(x, float(np.polyval(FITS[1], x)))).set_stroke(AMBER, width=2.5, opacity=0.75) for x, y in zip(X, Y)])
        under = pill("Underfitting: too simple for the pattern", AMBER, AMBER_FILL, 24).move_to([-3.4, -3.3, 0])

        self.say(
            "[Let's fit three models.] [First, degree one: a straight line.]",
            m.AnimationGroup(
                self.new_header("FITTING", "Three models, one set of points"),
                m.FadeOut(self.right_side),
                m.FadeIn(box),
                m.FadeIn(heads),
                m.Create(rule),
                m.LaggedStart(*[m.FadeIn(rows[d]["chip"], shift=0.15 * m.RIGHT) for d in rows], lag_ratio=0.2),
            ),
            m.AnimationGroup(m.Create(c1, run_time=1.2), m.FadeIn(rows[1]["train"])),
        )
        self.say(
            "[It can't bend, so it misses the shape.] [That's underfitting: a model too simple for the pattern.]",
            m.Create(gaps),
            m.AnimationGroup(m.FadeIn(verdicts[1], shift=0.15 * m.LEFT), m.FadeIn(under, shift=0.2 * m.UP)),
        )
        self.say(
            "[Degree three can bend a little], [and it follows the pattern nicely.]",
            m.AnimationGroup(m.FadeOut(gaps), m.FadeOut(under), c1.animate.set_stroke(opacity=0.3), m.Create(c3, run_time=1.4)),
            m.FadeIn(rows[3]["train"]),
        )
        self.say(
            "[Degree fifteen is far more flexible.] [It twists and turns to pass close to every point.]",
            m.AnimationGroup(c3.animate.set_stroke(opacity=0.3), m.Indicate(rows[15]["chip"], color=BAD, scale_factor=1.15)),
            m.AnimationGroup(m.Create(c15, run_time=2.6), m.FadeIn(rows[15]["train"])),
        )
        self.say(
            "[Its training error is the smallest of the three: almost zero.] [So it must be the best model, right?]",
            m.Circumscribe(rows[15]["train"], color=SKY, buff=0.1),
            m.FadeIn(best, scale=1.3),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the test

    def test(self):
        self.next_section("The test", notes="Fifteen held-out points: degree 15 has the lowest training error and the highest test error.")
        plot, dots, rows, verdicts, best = self.plot, self.dots, self.board["rows"], self.board["verdicts"], self.board["best"]
        c1, c3, c15 = (self.curves[d] for d in (1, 3, 15))
        rings = plot.rings(XT, YT)
        rings.set_z_index(3)
        legend = m.VGroup(
            m.VGroup(m.Dot(radius=0.085, color=SKY), text("training points", 22, MUTED)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Circle(radius=0.1).set_stroke(PINK, width=3.5), text("new test points", 22, MUTED)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.5).move_to([-3.4, 2.3, 0])
        hx, hy = XT[HERO], YT[HERO]
        p3, p15 = float(np.polyval(FITS[3], hx)), float(np.polyval(FITS[15], hx))
        on3 = m.Dot(plot.p(hx, p3), radius=0.075, color=GOOD).set_z_index(4)
        on15 = m.Dot(plot.p(hx, p15), radius=0.075, color=BAD).set_z_index(4)
        miss = m.DashedLine(plot.p(hx, hy), plot.p(hx, p15), dash_length=0.09).set_stroke(BAD, width=3.5).set_z_index(2)
        ring = m.Circle(radius=0.24).set_stroke(INK, width=3).move_to(plot.p(hx, hy)).set_z_index(4)
        miss3, miss15 = f"degree 3 is off by {abs(p3 - hy):.2f}", f"degree 15 by {abs(p15 - hy):.2f}"
        note = text(f"this point: {miss3}, {miss15}", 24, INK, t2c={miss3: GOOD, miss15: BAD}).move_to([-3.0, -3.3, 0])
        over = pill("Overfitting: it memorized the noise", BAD, BAD_FILL, 24).move_to([-3.4, -3.3, 0])

        self.say(
            "[Here's the real test]: [fifteen new points from the same curve, held back from training.]",
            self.new_header("THE TEST", "New points the models have never seen"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, scale=0.3) for r in rings], lag_ratio=0.08), m.FadeIn(legend)),
        )
        self.say(
            "[Look at this one.] [Degree three predicts it well], [but degree fifteen misses it badly.]",
            m.AnimationGroup(m.Create(ring), m.Indicate(rings[HERO], color=INK, scale_factor=1.6)),
            m.AnimationGroup(c3.animate.set_stroke(opacity=1), m.FadeIn(on3, scale=0.4)),
            m.AnimationGroup(m.FadeIn(on15, scale=0.4), m.Create(miss), m.FadeIn(note, shift=0.2 * m.UP)),
        )
        self.say(
            "[Across all fifteen new points], [degree three does best], [and degree fifteen does worst, even worse than the straight line.]",
            m.LaggedStart(*[m.FadeIn(rows[d]["test"], shift=0.15 * m.LEFT) for d in rows], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(verdicts[3], shift=0.15 * m.LEFT), m.Indicate(rows[3]["test"], color=GOOD, scale_factor=1.15)),
            m.AnimationGroup(m.Circumscribe(rows[15]["test"], color=BAD, buff=0.1), m.Indicate(rows[1]["test"], color=AMBER, scale_factor=1.1)),
        )
        self.say(
            "[It memorized the noise in its training points, instead of learning the curve.] [That's overfitting.]",
            m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.6) for d in dots], lag_ratio=0.05),
            m.AnimationGroup(m.FadeTransform(best, verdicts[15]), m.FadeTransform(note, over)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section("The big picture", notes="Every degree 1-15: training error falls, test error is U-shaped; pick with validation data.")
        plot = Plot(5.7, 3.5, [-3.6, -0.3, 0])
        dots = plot.dots(X, Y, radius=0.07)
        dots.set_z_index(3)
        degree = m.ValueTracker(1.0)
        curve = m.always_redraw(lambda: plot.graph(blend(degree.get_value()), X[0], X[-1], regime(degree.get_value()), width=4))
        number = m.DecimalNumber(1, num_decimal_places=0, mob_class=Mono, font_size=34, color=INK)
        number.add_updater(lambda d: d.set_value(int(round(degree.get_value()))))
        readout = m.VGroup(text("degree", 30, MUTED, weight=m.SEMIBOLD), number).arrange(m.RIGHT, buff=0.25).move_to([-3.6, 2.05, 0])

        chart = axes([0, 16.5, 1], [0, 0.55, 0.1], 5.4, 3.5).move_to([3.5, -0.3, 0])
        ticks = m.VGroup(*[text(str(k), 22, MUTED).next_to(chart.c2p(k, 0), m.DOWN, buff=0.28) for k in (1, 5, 10, 15)])
        x_label = text("degree", 24, MUTED).next_to(chart.c2p(8, 0), m.DOWN, buff=0.62)
        y_label = text("error", 24, MUTED).next_to(chart.y_axis.get_end(), m.RIGHT, buff=0.18)
        legend = m.VGroup(
            m.VGroup(m.Line(m.ORIGIN, 0.45 * m.RIGHT).set_stroke(SKY, width=5), text("training error", 22, SKY, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Line(m.ORIGIN, 0.45 * m.RIGHT).set_stroke(PINK, width=5), text("test error", 22, PINK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.5).move_to([3.6, 2.05, 0])

        def trail():
            d = degree.get_value()
            whole = [k for k in range(1, 16) if k <= d]
            points = [chart.c2p(k, TRAIN[k]) for k in whole]
            if d < 15 and d > whole[-1]:
                k = whole[-1]
                a = d - k
                points.append(chart.c2p(d, (1 - a) * TRAIN[k] + a * TRAIN[k + 1]))
            if len(points) < 2:
                points.append(points[-1] + 0.001 * m.RIGHT)
            return polyline(points, SKY, width=4.5)

        train_line = m.always_redraw(trail)
        test_down = polyline([chart.c2p(k, TEST[k]) for k in range(1, BEST + 1)], PINK, width=4.5)
        test_up = polyline([chart.c2p(k, TEST[k]) for k in range(BEST, 16)], PINK, width=4.5)
        under = text("underfitting", 24, AMBER, weight=m.BOLD).move_to(chart.c2p(3.3, 0.42))
        over = text("overfitting", 24, BAD, weight=m.BOLD).move_to(chart.c2p(10.6, 0.42))
        spot = m.Dot(chart.c2p(BEST, TEST[BEST]), radius=0.11, color=GOOD).set_z_index(3)
        spot_label = text("sweet spot", 24, GOOD, weight=m.BOLD).move_to(chart.c2p(BEST, 0.27))
        pointer = arrow(spot_label.get_bottom(), spot.get_top(), GOOD, buff=0.08)
        best_tick = text(str(BEST), 22, GOOD, weight=m.BOLD).next_to(chart.c2p(BEST, 0), m.DOWN, buff=0.28)
        practice = m.VGroup(
            pill("choose the degree with a validation set", SKY, SKY_FILL, 22),
            pill("keep the test set for a final check", PINK, PINK_FILL, 22),
        ).arrange(m.RIGHT, buff=0.4).move_to([0, -3.4, 0])

        self.say(
            "[Now let's try every degree, from one to fifteen.]",
            m.AnimationGroup(
                self.new_header("THE BIG PICTURE", "Training error versus test error"),
                m.Create(plot.frame),
                m.FadeIn(dots),
                m.FadeIn(curve),
                m.FadeIn(readout),
                m.Create(chart),
                m.FadeIn(ticks),
                m.FadeIn(x_label),
                m.FadeIn(y_label),
                m.FadeIn(legend),
            ),
        )
        self.add(train_line)
        self.say(
            "[As the degree grows, the training error keeps falling.]",
            degree.animate(run_time=4.5, rate_func=m.linear).set_value(15),
        )
        self.stop(train_line)
        self.say("[But the test error drops at first], [then turns back up.]", m.Create(test_down, run_time=0.8), m.Create(test_up, run_time=2.2))
        self.say(
            "[On the left, models are too simple: underfitting.] [On the right, they're too flexible: overfitting.]",
            m.FadeIn(under, shift=0.2 * m.DOWN),
            m.FadeIn(over, shift=0.2 * m.DOWN),
        )
        self.say(
            "[The sweet spot is the bottom of the test curve]: [here, degree three.]",
            m.AnimationGroup(m.FadeIn(spot, scale=0.4), m.GrowArrow(pointer), m.FadeIn(spot_label, shift=0.15 * m.DOWN), m.FadeIn(best_tick)),
            degree.animate(run_time=2.0).set_value(BEST),
        )
        self.say(
            "[In practice, we choose the degree with a separate validation set], [and save the test set for one final check.]",
            m.FadeIn(practice[0], shift=0.2 * m.UP),
            m.FadeIn(practice[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.stop(curve, number)
        self.clear()

    # ------------------------------------------------------------------ bias and variance

    def bias_variance(self):
        self.next_section("Bias and variance", notes="Six fresh samples: degree 1 fits agree (high bias), degree 15 fits disagree (high variance).")

        def panel(x, d, color, fill, title, verdict):
            box = card(6.2, 4.6, color, CARD, stroke=2).move_to([x, -0.35, 0])
            heading = text(title, 26, color, weight=m.BOLD).move_to(box.get_top() + 0.45 * m.DOWN)
            plot = Plot(5.0, 2.6, box.get_center() + 0.12 * m.DOWN, labels=False)
            hidden = m.DashedVMobject(plot.graph(truth, -1, 1, MUTED, width=2.5), num_dashes=50)
            fits = m.VGroup(*[plot.fit(polyfit(xs, ys, d), color, width=3, opacity=0.85, x0=xs[0], x1=xs[-1]) for xs, ys in OTHERS])
            tag = pill(verdict, color, fill, 22).next_to(box.get_bottom(), m.UP, buff=0.25)
            return m.VGroup(box, heading, plot.frame, hidden), fits, tag

        left, lines, bias = panel(-3.4, 1, AMBER, AMBER_FILL, "Degree 1 · six samples", "high bias: the same mistake every time")
        right, wiggles, variance = panel(3.4, 15, BAD, BAD_FILL, "Degree 15 · six samples", "high variance: different every time")
        between = pill("A good model: flexible enough for the pattern, not for the noise", GOOD, GOOD_FILL, 22).move_to([0, -3.3, 0])

        self.say(
            "[Imagine drawing six fresh samples from the same curve, and fitting a model to each one.]",
            m.AnimationGroup(self.new_header("BIAS AND VARIANCE", "Two ways to be wrong"), m.FadeIn(left), m.FadeIn(right)),
        )
        self.say(
            "[Straight lines come out almost the same every time], [and they all miss the bend.] [That's called high bias.]",
            m.LaggedStart(*[m.Create(c) for c in lines], lag_ratio=0.25),
            m.Indicate(left[3], color=INK, scale_factor=1.0),
            m.FadeIn(bias, shift=0.2 * m.UP),
        )
        self.say(
            "[Degree fifteen gives a wildly different curve every time.] [That's called high variance.]",
            m.LaggedStart(*[m.Create(c) for c in wiggles], lag_ratio=0.25, run_time=2.6),
            m.FadeIn(variance, shift=0.2 * m.UP),
        )
        self.say(
            "[A good model sits in between]: [flexible enough for the pattern, but not so flexible that it chases the noise.]",
            m.FadeIn(between, shift=0.2 * m.UP),
            m.Circumscribe(between, color=GOOD, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the fixes

    def fixes(self):
        self.next_section("The fixes", notes="More data, a simpler model, regularization (ridge), early stopping (gradient descent, log time axis).")
        plot = Plot(6.4, 4.4, [-3.45, -0.55, 0])
        dots = plot.dots(X, Y, radius=0.08)
        dots.set_z_index(3)
        extra = plot.dots(XM, YM, radius=0.05)
        extra.set_opacity(0.55)
        extra.set_z_index(1)
        current = plot.fit(FITS[15], BAD, width=4.5)

        items = (
            ("database", "More data", "harder to memorize 120 points than 20"),
            ("ep05-curve", "A simpler model", "fewer weights, fewer ways to memorize"),
            ("ep05-weight", "Regularization", "a penalty on big weights"),
            ("ep05-stop", "Early stopping", "stop when held-out error rises"),
        )
        rows = m.VGroup()
        for k, (name, title, desc) in enumerate(items):
            words = m.VGroup(text(f"{k + 1} · {title}", 28, INK, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            rows.add(m.VGroup(icon(variant(name, GOOD, GOOD_FILL), 0.72), words).arrange(m.RIGHT, buff=0.35))
        rows.arrange(m.DOWN, buff=0.45, aligned_edge=m.LEFT).move_to([3.75, -0.45, 0])
        column = max(r[0].get_right()[0] for r in rows) + 0.35
        for row in rows:  # icons on one centre line, words in one column
            row[0].set_x(rows[0][0].get_x())
            row[1].shift((column - row[1].get_left()[0]) * m.RIGHT)

        lam = m.ValueTracker(RIDGE_FROM)

        def ridged():
            v = lam.get_value()
            return plot.fit(ridge(10**v), mix(BAD, GOOD, (v - RIDGE_FROM) / (RIDGE_TO - RIDGE_FROM) * 1.6), width=4.5)

        smooth = m.always_redraw(ridged)
        biggest = m.DecimalNumber(float(np.abs(ridge(10**RIDGE_FROM)).max()), num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=30, color=VIOLET)
        biggest.add_updater(lambda d: d.set_value(float(np.abs(ridge(10 ** lam.get_value())).max())))
        readout = m.VGroup(text("largest weight", 24, MUTED, weight=m.SEMIBOLD), biggest).arrange(m.RIGHT, buff=0.25)
        readout.move_to([-2.4, 2.15, 0])
        penalty = tex(r"\text{loss} = \text{error} + \lambda \sum_j w_j^2", 36, INK).move_to([-3.45, -3.2, 0])

        chart = axes([0, 14.6, 1], [0, 0.55, 0.1], 6.2, 4.2).move_to([-3.45, -0.55, 0])
        t_label = text("training time", 24, MUTED).next_to(chart.c2p(10.5, 0), m.DOWN, buff=0.3)
        e_label = text("error", 24, MUTED).next_to(chart.y_axis.get_end(), m.RIGHT, buff=0.18)
        gd_train = polyline([chart.c2p(t, e) for t, e in zip(LOG_T, GD_TRAIN)], SKY, width=4.5)
        gd_held = polyline([chart.c2p(t, e) for t, e in zip(LOG_T, GD_HELD)], PINK, width=4.5)
        train_tag = text("training", 22, SKY, weight=m.SEMIBOLD).move_to(chart.c2p(12.4, 0.07))
        held_tag = text("held-out", 22, PINK, weight=m.SEMIBOLD).move_to(chart.c2p(11.2, 0.45))
        stop_line = m.DashedLine(chart.c2p(GD_STOP, 0), chart.c2p(GD_STOP, 0.4), dash_length=0.08).set_stroke(GOOD, width=3.5)
        stop_label = text("stop here", 22, GOOD, weight=m.BOLD).next_to(stop_line, m.UP, buff=0.12)

        self.say(
            "[So how do we stop a model from memorizing?] Here are four common fixes.",
            m.AnimationGroup(
                self.new_header("THE FIXES", "How to stop memorizing"),
                m.Create(plot.frame),
                m.FadeIn(dots),
                m.Create(current, run_time=1.5),
            ),
        )
        self.say(
            "[First, more data.] [With a hundred and twenty points, degree fifteen can't chase them all], [and it settles close to the real curve.]",
            m.FadeIn(rows[0], shift=0.2 * m.LEFT),
            m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in extra], lag_ratio=0.02),
            m.Transform(current, plot.fit(MORE, GOOD, width=4.5)),
        )
        self.say(
            "[Second, a simpler model], [with fewer weights, and fewer ways to memorize.]",
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.2 * m.LEFT), m.FadeOut(extra)),
            m.Transform(current, plot.fit(FITS[3], GOOD, width=4.5)),
        )
        self.say(
            "[Third, regularization]: [we add a penalty for big weights to the training loss.]",
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.2 * m.LEFT), m.Transform(current, plot.fit(FITS[15], BAD, width=4.5))),
            m.Write(penalty),
        )
        self.remove(current)
        self.add(smooth)
        self.say(
            "[The wiggly curve needs weights in the tens of thousands.] [As the penalty grows, the weights shrink, and the curve calms down.]",
            m.FadeIn(readout, shift=0.2 * m.DOWN),
            lam.animate(run_time=3.5, rate_func=m.smooth).set_value(RIDGE_TO),
        )
        self.stop(smooth, biggest)
        self.say(
            "[Fourth, early stopping]: [watch the error on held-out data during training], [and stop when it starts to rise.]",
            m.AnimationGroup(m.FadeIn(rows[3], shift=0.2 * m.LEFT), *[m.FadeOut(mob) for mob in (plot.frame, dots, smooth, readout, penalty)]),
            m.AnimationGroup(
                m.Create(chart),
                m.FadeIn(t_label),
                m.FadeIn(e_label),
                m.Create(gd_train, run_time=2.0),
                m.Create(gd_held, run_time=2.0),
                m.FadeIn(train_tag),
                m.FadeIn(held_tag),
            ),
            m.AnimationGroup(m.Create(stop_line), m.FadeIn(stop_label, shift=0.15 * m.DOWN)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Fit, test, simplify.")
        halo = glow(GOOD, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Fit", "Test", "Simplify"), (SKY, PINK, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A good model learns the pattern, not the noise.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [fit], [test], [simplify.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A good model learns the pattern, not the noise.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
