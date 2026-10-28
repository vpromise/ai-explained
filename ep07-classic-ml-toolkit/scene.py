"""AI, Explained · Episode 7 · The Classic ML Toolkit

Running example: sorting fruit by two features, weight (grams) and redness (0 to 10), into
apples and oranges. One seeded dataset, one mystery fruit, five classic models:

- linear regression: a line predicts an orange's juice from its weight (recalls Episode 2);
- logistic regression: a score, an S-curve, a probability, a threshold, a straight boundary;
- a decision tree: two yes/no questions cut the chart into labelled boxes;
- k-nearest neighbours: the five closest fruit vote on the mystery fruit;
- k-means: the same fruit without labels; centroids assign and move until nothing changes.

Everything shown is computed below with numpy: the least-squares line, a logistic regression
fitted by gradient descent, the best Gini splits, the k-NN vote and every k-means step. The
narration quotes those numbers, and the asserts keep them honest. Distances (k-NN, k-means)
are measured on the chart, where both features are scaled to the axes, as features are scaled
in practice.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

APPLE, APPLE_FILL = BAD, BAD_FILL  # in this film red means "apple", never "mistake"
ORANGE, ORANGE_FILL = AMBER, AMBER_FILL
GROUP_A, GROUP_B = BLUE, LIME  # k-means groups: not label colours on purpose

# ---------------------------------------------------------------------------- the data

W_MIN, W_MAX, R_MIN, R_MAX = 100.0, 250.0, 0.0, 10.0
XL, YL = 5.8, 4.2  # the fruit chart's axis lengths: distances are measured in these units
CHART_ORIGIN = np.array([-6.1, -2.55, 0.0])  # where (100 g, redness 0) sits on screen


def make_fruit(seed=8):
    """12 apples (9 red, 3 green) and 12 oranges: [weight g, redness 0–10]; label 1 = orange."""
    rng = np.random.default_rng(seed)
    red = np.c_[rng.normal(152, 14, 9), rng.normal(7.3, 0.9, 9)]
    green = np.c_[rng.normal(134, 10, 3), rng.normal(3.0, 0.6, 3)]
    oranges = np.c_[rng.normal(187, 14, 12), rng.normal(4.2, 0.8, 12)]
    fruit = np.r_[red, green, oranges]
    fruit[:, 0] = np.clip(np.round(fruit[:, 0]), 112, 238)
    fruit[:, 1] = np.clip(np.round(fruit[:, 1], 1), 0.8, 9.6)
    return fruit, np.r_[np.zeros(12), np.ones(12)]


FRUIT, IS_ORANGE = make_fruit()
MYSTERY = np.array([168.0, 5.8])
SHOW_APPLE, SHOW_ORANGE = 0, 20  # the two fruit read out as numbers


def on_chart(p):
    """[weight, redness] → chart units (the axes' scale), for distances."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    return np.c_[(p[:, 0] - W_MIN) / (W_MAX - W_MIN) * XL, (p[:, 1] - R_MIN) / (R_MAX - R_MIN) * YL]


def off_chart(c):
    c = np.atleast_2d(c)
    return np.c_[W_MIN + c[:, 0] / XL * (W_MAX - W_MIN), R_MIN + c[:, 1] / YL * (R_MAX - R_MIN)]


# linear regression: juice (ml) from weight (g), 10 oranges, least squares
_jr = np.random.default_rng(5)
JUICE_W = np.sort(np.round(_jr.uniform(140, 232, 10)))
JUICE_ML = np.round(0.45 * JUICE_W - 4 + _jr.normal(0, 4.0, 10), 1)
JUICE_K, JUICE_B = np.polyfit(JUICE_W, JUICE_ML, 1)
JUICE_NEW = 200.0
JUICE_PRED = JUICE_K * JUICE_NEW + JUICE_B
assert round(JUICE_PRED) == 84  # "about eighty-four millilitres"


# logistic regression: gradient descent on standardised features, a little L2
def fit_logistic(x, y, l2=0.01, steps=4000, rate=0.5):
    mu, sd = x.mean(0), x.std(0)
    z = (x - mu) / sd
    w, b = np.zeros(2), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        w -= rate * (z.T @ (p - y) / len(y) + l2 * w)
        b -= rate * np.mean(p - y)
    return w / sd, b - np.sum(w * mu / sd)  # back to raw units: score = a·[weight, redness] + c


LOGIT_A, LOGIT_C = fit_logistic(FRUIT, IS_ORANGE)


def score(p):
    return float(np.asarray(p) @ LOGIT_A + LOGIT_C)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Z_MYSTERY = score(MYSTERY)
P_MYSTERY = sigmoid(Z_MYSTERY)
assert 0 < Z_MYSTERY < 1 and round(100 * P_MYSTERY) == 58  # "just above zero", "fifty-eight percent"
assert np.all((FRUIT @ LOGIT_A + LOGIT_C > 0) == (IS_ORANGE == 1))  # the boundary sorts every fruit


# a decision tree of depth two, by the best Gini split at each node
def gini(y):
    return 0.0 if len(y) == 0 else 2 * y.mean() * (1 - y.mean())


def best_split(x, y):
    best = None
    for f in range(2):
        values = np.unique(x[:, f])
        for lo, hi in zip(values, values[1:]):
            cut = (lo + hi) / 2
            left, right = y[x[:, f] <= cut], y[x[:, f] > cut]
            impurity = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if best is None or impurity < best[0] - 1e-12:
                best = (impurity, f, lo, hi)
    return best


_, ROOT_F, ROOT_LO, ROOT_HI = best_split(FRUIT, IS_ORANGE)
ROOT_T = 150.0  # shown as "150 g": it splits the fruit exactly as the learned cut (midway) does
assert ROOT_F == 0 and ROOT_LO < ROOT_T < ROOT_HI
_heavy = FRUIT[:, 0] > ROOT_T
assert IS_ORANGE[~_heavy].sum() == 0  # light: all apples
_, NEXT_F, NEXT_LO, NEXT_HI = best_split(FRUIT[_heavy], IS_ORANGE[_heavy])
NEXT_T = 6.0  # "redness above six?"
assert NEXT_F == 1 and NEXT_LO < NEXT_T < NEXT_HI
_red = FRUIT[:, 1] > NEXT_T
assert IS_ORANGE[_heavy & _red].sum() == 0 and IS_ORANGE[_heavy & ~_red].all()
assert MYSTERY[0] > ROOT_T and MYSTERY[1] <= NEXT_T  # the mystery fruit lands with the oranges

# k-nearest neighbours, k = 5, straight-line distance on the chart
K = 5
_d = np.linalg.norm(on_chart(FRUIT) - on_chart(MYSTERY), axis=1)
NEAR = np.argsort(_d)[:K]
NEAR_RADIUS = (_d[NEAR[-1]] + np.sort(_d)[K]) / 2  # a circle between the 5th and 6th fruit
assert IS_ORANGE[NEAR].sum() == 3  # "three oranges, two apples"


# k-means, k = 2, on the unlabelled fruit
def kmeans(points, start, limit=20):
    """Returns the centroids before each step and the assignment of each step (chart units)."""
    centres, history, assignments = start.copy(), [start.copy()], []
    for _ in range(limit):
        nearest = np.argmin(np.linalg.norm(points[:, None] - centres[None], axis=2), axis=1)
        if assignments and np.array_equal(nearest, assignments[-1]):
            assignments.append(nearest)
            break
        assignments.append(nearest)
        centres = np.array([points[nearest == j].mean(0) for j in range(len(centres))])
        history.append(centres.copy())
    return history, assignments


KM_START = on_chart([[125.0, 1.5], [145.0, 1.0]])  # dropped anywhere: both low on the chart
KM_CENTRES, KM_STEPS = kmeans(on_chart(FRUIT), KM_START)
assert len(KM_STEPS) == 3 and len(KM_CENTRES) == 3  # assign, move, assign, move, nothing changes
assert np.array_equal(KM_STEPS[-1], IS_ORANGE.astype(int))  # group 1 = the apples, group 2 = the oranges

# ---------------------------------------------------------------------------- helpers


def chart(x_range, y_range, x_ticks, y_ticks, x_title, y_title):
    """Quiet axes at the left of the stage, a few tick numbers, and the two titles."""
    ax = axes(x_range, y_range, XL, YL)
    ax.shift(CHART_ORIGIN - ax.c2p(x_range[0], y_range[0]))
    xt = m.VGroup(*[text(f"{v:g}", 20, MUTED).next_to(ax.c2p(v, y_range[0]), m.DOWN, buff=0.18) for v in x_ticks])
    yt = m.VGroup(*[text(f"{v:g}", 20, MUTED).next_to(ax.c2p(x_range[0], v), m.LEFT, buff=0.18) for v in y_ticks])
    xl = text(x_title, 22, MUTED).move_to([ax.c2p((x_range[0] + x_range[1]) / 2, 0)[0], CHART_ORIGIN[1] - 0.72, 0])
    yl = text(y_title, 22, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.2)
    return ax, m.VGroup(xt, yt), xl, yl


def dot(point, color, radius=0.085):
    return m.Dot(point, radius=radius, color=color).set_z_index(2)


def mystery_marker(point):
    return m.VGroup(
        m.Circle(radius=0.2).set_stroke(INK, width=3).set_fill(BG, opacity=0.0).move_to(point),
        m.Dot(point, radius=0.1, color=INK),
    ).set_z_index(3)


def diamond(color, size=0.32):
    shape = m.Square(side_length=size).rotate(np.pi / 4)
    return shape.set_fill(color, opacity=1).set_stroke(INK, width=2.5).set_z_index(4)


def rect_on(ax, w0, w1, r0, r1, color, opacity=0.12):
    corners = [ax.c2p(w0, r0), ax.c2p(w1, r0), ax.c2p(w1, r1), ax.c2p(w0, r1)]
    return m.Polygon(*corners).set_stroke(width=0).set_fill(color, opacity=opacity).set_z_index(-1)


def half_plane(ax, a, c, positive, color, opacity=0.12):
    """The part of the chart where a·[w, r] + c is > 0 (positive) or < 0: a clipped polygon."""
    box = [(W_MIN, R_MIN), (W_MAX, R_MIN), (W_MAX, R_MAX), (W_MIN, R_MAX)]
    side = lambda p: (p[0] * a[0] + p[1] * a[1] + c) * (1 if positive else -1)  # noqa: E731
    out = []
    for p, q in zip(box, box[1:] + box[:1]):
        sp, sq = side(p), side(q)
        if sp >= 0:
            out.append(p)
        if sp * sq < 0:
            t = sp / (sp - sq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return m.Polygon(*[ax.c2p(*p) for p in out]).set_stroke(width=0).set_fill(color, opacity=opacity).set_z_index(-1)


def boundary_ends(a, c):
    """Where the line a·[w, r] + c = 0 crosses the chart's edges."""
    hits = []
    for w in (W_MIN, W_MAX):
        r = -(a[0] * w + c) / a[1]
        if R_MIN <= r <= R_MAX:
            hits.append((w, r))
    for r in (R_MIN, R_MAX):
        w = -(a[1] * r + c) / a[0]
        if W_MIN <= w <= W_MAX:
            hits.append((w, r))
    return sorted(hits)[:2]


def pictogram(kind):
    """A tiny picture of each model, about 1.7 × 1.0."""
    g = m.VGroup()
    if kind == "line":
        pts = [(-0.7, -0.36), (-0.4, -0.08), (-0.1, -0.22), (0.2, 0.16), (0.48, 0.04), (0.72, 0.36)]
        g.add(m.Line([-0.82, -0.44, 0], [0.82, 0.42, 0]).set_stroke(SKY, width=4))
        g.add(*[m.Dot([x, y, 0], radius=0.07, color=ORANGE) for x, y in pts])
    elif kind == "sigmoid":
        xs = np.linspace(-0.82, 0.82, 50)
        curve = m.VMobject().set_points_smoothly([[x, -0.42 + 0.84 / (1 + np.exp(-8 * x)), 0] for x in xs])
        g.add(curve.set_stroke(SKY, width=4))
        g.add(*[m.Dot([x, -0.42, 0], radius=0.07, color=APPLE) for x in (-0.72, -0.5, -0.28)])
        g.add(*[m.Dot([x, 0.42, 0], radius=0.07, color=ORANGE) for x in (0.28, 0.5, 0.72)])
    elif kind == "tree":
        root = m.RoundedRectangle(corner_radius=0.08, width=0.6, height=0.32).set_stroke(SKY, width=3).set_fill(SKY_FILL, 1)
        left = m.RoundedRectangle(corner_radius=0.08, width=0.5, height=0.3).set_stroke(APPLE, width=3).set_fill(APPLE_FILL, 1)
        right = m.RoundedRectangle(corner_radius=0.08, width=0.5, height=0.3).set_stroke(ORANGE, width=3).set_fill(ORANGE_FILL, 1)
        root.move_to([0, 0.33, 0]); left.move_to([-0.55, -0.33, 0]); right.move_to([0.55, -0.33, 0])
        links = m.VGroup(
            m.Line(root.get_bottom(), left.get_top()).set_stroke(MUTED, width=3),
            m.Line(root.get_bottom(), right.get_top()).set_stroke(MUTED, width=3),
        )
        g.add(links, root, left, right)
    elif kind == "knn":
        ring = m.DashedVMobject(m.Circle(radius=0.5), num_dashes=22).set_stroke(MUTED, width=2.5)
        near = [((0.3, 0.27), ORANGE), ((-0.34, 0.16), APPLE), ((0.12, -0.36), ORANGE)]
        far = [((-0.78, -0.3), APPLE), ((0.8, -0.08), ORANGE)]
        g.add(ring)
        g.add(*[m.Line([0, 0, 0], [x, y, 0]).set_stroke(INK, width=2, opacity=0.7) for (x, y), _ in near])
        g.add(*[m.Dot([x, y, 0], radius=0.07, color=c) for (x, y), c in near + far])
        g.add(m.Dot([0, 0, 0], radius=0.09, color=INK))
    elif kind == "kmeans":
        offsets = [(-0.18, 0.1), (0.14, 0.2), (0.0, -0.15), (-0.2, -0.18), (0.2, -0.06)]
        for (cx, cy), color in (((-0.45, 0.12), GROUP_A), ((0.45, -0.12), GROUP_B)):
            g.add(*[m.Dot([cx + dx, cy + dy, 0], radius=0.065, color=color) for dx, dy in offsets])
            g.add(diamond(color, 0.2).set_stroke(INK, width=1.5).move_to([cx, cy, 0]).set_z_index(0))
    return g


TOOLS = [
    ("line", "Linear", "regression", "predict a number"),
    ("sigmoid", "Logistic", "regression", "yes or no, with a probability"),
    ("tree", "Decision", "tree", "rules you can read"),
    ("knn", "k-nearest", "neighbours", "label by similar examples"),
    ("kmeans", "k-means", "clustering", "find groups without labels"),
]


def tool_card(kind, first, second, width=2.45, height=2.35):
    box = card(width, height, EDGE, CARD)
    pic = pictogram(kind).move_to(box.get_center() + 0.42 * m.UP)
    name = m.VGroup(text(first, 26, INK, weight=m.BOLD), text(second, 22, MUTED)).arrange(m.DOWN, buff=0.08)
    name.move_to(box.get_center() + 0.62 * m.DOWN)
    return m.VGroup(box, pic, name)


def ep_recolored(name, color, fill):
    """kit.recolored(), but the variant is written as assets/ep07-<name>-<color>.svg (this
    episode only adds ep07- files to the shared assets)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep07-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


def fact_tile(name, title, desc, color, width=6.0, height=1.6):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 26, INK, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
    glyph = icon(name, 0.85).move_to(box.get_left() + 0.85 * m.RIGHT)  # a fixed slot: the words line up across tiles
    words.move_to(box).align_to(box.get_left() + 1.65 * m.RIGHT, m.LEFT)
    return m.VGroup(box, m.VGroup(glyph, words))


# ---------------------------------------------------------------------------- the episode


class Ep07ClassicMLToolkit(Episode):
    number = 7

    def lesson(self):
        self.hook()
        self.linear()
        self.task()
        self.logistic()
        self.tree()
        self.neighbours()
        self.clusters()
        self.which()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Apple or orange? Five classic models, all on one example.")
        apple = icon("ep07-apple", 1.9).move_to([-2.3, 1.75, 0])
        orange = icon("ep07-orange", 1.9).move_to([2.3, 1.75, 0])
        mark = text("?", 80, SKY, weight=m.BOLD).move_to([0, 1.75, 0])
        cards = m.VGroup(*[tool_card(kind, a, b) for kind, a, b, _ in TOOLS]).arrange(m.RIGHT, buff=0.24).move_to([0, -1.2, 0])
        perks = m.VGroup(
            pill("fast", GOOD, GOOD_FILL, 22), pill("easy to understand", GOOD, GOOD_FILL, 22), pill("hard to beat", GOOD, GOOD_FILL, 22)
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -3.15, 0])

        self.say(
            "[Apple, or orange?] [You can tell at a glance. But how would a machine decide?]",
            m.AnimationGroup(m.FadeIn(apple, shift=0.3 * m.RIGHT), m.FadeIn(orange, shift=0.3 * m.LEFT)),
            m.FadeIn(mark, scale=0.6),
        )
        self.say(
            "[Long before deep learning, machine learning had a toolkit of simple models.]",
            m.LaggedStart(*[m.FadeIn(c, shift=0.3 * m.UP) for c in cards], lag_ratio=0.2),
        )
        self.say(
            "[They're fast, easy to understand], [and on the right problem, very hard to beat.] Let's meet all five.",
            m.LaggedStart(m.FadeIn(perks[0], shift=0.2 * m.UP), m.FadeIn(perks[1], shift=0.2 * m.UP), lag_ratio=0.4),
            m.FadeIn(perks[2], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 linear regression

    def linear(self):
        self.next_section(
            "Linear regression",
            notes=f"Least-squares line through 10 oranges: juice = {JUICE_K:.3f}·weight {JUICE_B:+.2f}; 200 g → {JUICE_PRED:.1f} ml.",
        )
        ax, ticks, xl, yl = chart([120, 240, 20], [40, 110, 10], (120, 160, 200, 240), (40, 70, 100), "weight (g)", "juice (ml)")
        points = m.VGroup(*[dot(ax.c2p(w, j), ORANGE) for w, j in zip(JUICE_W, JUICE_ML)])
        line = m.Line(ax.c2p(124, JUICE_K * 124 + JUICE_B), ax.c2p(238, JUICE_K * 238 + JUICE_B)).set_stroke(SKY, width=5)
        line.set_z_index(1)
        formula = tex(r"\text{juice} = w \times \text{weight} + b", 40, INK).move_to([3.65, 1.45, 0])
        learned = code(f"learned:  w = {JUICE_K:.2f},  b = {JUICE_B:.1f}".replace("-", "−"), 22, MUTED).next_to(formula, m.DOWN, buff=0.35)
        recall = pill("the same line as Episode 2's house prices", SKY, SKY_FILL, 20).next_to(learned, m.DOWN, buff=0.45)
        at = ax.c2p(JUICE_NEW, JUICE_PRED)
        marker = m.DashedLine(ax.c2p(JUICE_NEW, 40), at, dash_length=0.1).set_stroke(MUTED, width=2.5)
        across = m.DashedLine(at, ax.c2p(120, JUICE_PRED), dash_length=0.1).set_stroke(MUTED, width=2.5)
        guess = dot(at, GOOD, 0.12).set_z_index(3)
        answer = pill(f"≈ {JUICE_PRED:.0f} ml", GOOD, GOOD_FILL, 24).move_to(at + np.array([1.25, -0.75, 0]))
        kind = pill("linear regression predicts a number", SKY, SKY_FILL, 22).move_to([3.65, -1.2, 0])

        self.say(
            "[Let's warm up with a model you already know.]",
            m.AnimationGroup(
                self.new_header("MODEL 1 · LINEAR REGRESSION", "A line through the points"),
                m.Create(ax), m.FadeIn(ticks), m.FadeIn(xl), m.FadeIn(yl),
            ),
        )
        self.say(
            "[Say we want to predict how much juice an orange gives], [from its weight.]",
            m.LaggedStart(*[m.FadeIn(p, scale=0.3) for p in points], lag_ratio=0.12),
            m.Indicate(xl, color=SKY, scale_factor=1.15),
        )
        self.say(
            "[Linear regression finds the line with the smallest squared error], [just like the house prices in episode two.]",
            m.AnimationGroup(m.Create(line), m.Write(formula)),
            m.AnimationGroup(m.FadeIn(learned, shift=0.15 * m.UP), m.FadeIn(recall, shift=0.2 * m.UP)),
        )
        self.say(
            "[A new orange weighs two hundred grams?] [The line predicts about eighty-four millilitres of juice.]",
            m.Create(marker),
            m.AnimationGroup(m.FadeIn(guess, scale=0.4), m.Create(across), m.FadeIn(answer, shift=0.2 * m.UP)),
        )
        self.say("[Linear regression predicts a number.] But sorting fruit needs a category.", m.FadeIn(kind, shift=0.2 * m.UP))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the task: fruit as points

    def task(self):
        self.next_section("Fruit as points", notes="24 fruit: weight (g) and redness (0–10). Mystery fruit: 168 g, redness 5.8.")
        ax, ticks, xl, yl = chart([100, 250, 25], [0, 10, 2], (100, 150, 200, 250), (0, 5, 10), "weight (g)", "redness")
        self.ax = ax
        here = on_chart(FRUIT[0])[0]
        assert np.allclose(ax.c2p(*FRUIT[0]) - ax.c2p(W_MIN, R_MIN), [here[0], here[1], 0])  # chart units = distances
        colours = [ORANGE if o else APPLE for o in IS_ORANGE]
        self.dots = [dot(ax.c2p(*p), c) for p, c in zip(FRUIT, colours)]
        self.mystery = mystery_marker(ax.c2p(*MYSTERY))
        self.chart_parts = [ax, ticks, xl, yl]

        def row(glyph, p, color, y):
            box = card(5.6, 1.05, color, CARD, radius=0.2, stroke=2)
            words = text(f"{p[0]:.0f} g   ·   redness {p[1]:.1f}", 26, INK, weight=m.SEMIBOLD, t2c={f"{p[0]:.0f} g": color, f"{p[1]:.1f}": color})
            body = m.VGroup(glyph, words).arrange(m.RIGHT, buff=0.4).move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            return m.VGroup(box, body).move_to([3.65, y, 0])

        unknown = m.VGroup(m.Circle(radius=0.36).set_stroke(INK, width=3), text("?", 34, INK, weight=m.BOLD))
        rows = [
            row(icon("ep07-apple", 0.72), FRUIT[SHOW_APPLE], APPLE, 1.45),
            row(icon("ep07-orange", 0.72), FRUIT[SHOW_ORANGE], ORANGE, 0.15),
            row(unknown, MYSTERY, INK, -1.15),
        ]
        self.rows = rows
        rest_apples = [d for i, d in enumerate(self.dots) if not IS_ORANGE[i] and i != SHOW_APPLE]
        rest_oranges = [d for i, d in enumerate(self.dots) if IS_ORANGE[i] and i != SHOW_ORANGE]
        self.tag = text("?", 30, INK, weight=m.BOLD).next_to(self.mystery, m.UR, buff=0.04)

        self.say(
            "[Now for the main task: apple, or orange?]",
            m.AnimationGroup(self.new_header("THE TASK", "Apple or orange?"), m.Create(ax), m.FadeIn(ticks), m.FadeIn(xl), m.FadeIn(yl)),
        )
        self.say(
            "[We describe each fruit with two numbers]: [its weight, and how red it is, from zero to ten.]",
            m.FadeIn(rows[0], shift=0.2 * m.LEFT),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.2 * m.LEFT), m.Indicate(xl, color=SKY, scale_factor=1.15), m.Indicate(yl, color=SKY, scale_factor=1.15)),
        )
        self.say(
            "[So every fruit becomes a point.] [Apples are lighter, and most are red.] [Oranges are heavier, and less red.]",
            m.AnimationGroup(
                m.TransformFromCopy(rows[0][1][0], self.dots[SHOW_APPLE]),
                m.TransformFromCopy(rows[1][1][0], self.dots[SHOW_ORANGE]),
            ),
            m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in rest_apples], lag_ratio=0.12),
            m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in rest_oranges], lag_ratio=0.1),
        )
        self.say(
            "[And here's a mystery fruit.] Let's see what each model makes of it.",
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.2 * m.LEFT), m.FadeIn(self.mystery, scale=1.6), m.FadeIn(self.tag)),
        )
        self.wait(0.3)

    def keep(self):
        """What stays on stage from one model to the next: the chart, the fruit, the mystery fruit."""
        return [*self.chart_parts, *self.dots, self.mystery]

    # ------------------------------------------------------------------ 2 logistic regression

    def logistic(self):
        self.next_section(
            "Logistic regression",
            notes=f"score = {LOGIT_A[0]:.3f}·weight {LOGIT_A[1]:+.3f}·redness {LOGIT_C:+.2f}; mystery: z = {Z_MYSTERY:.2f}, p = {P_MYSTERY:.3f}.",
        )
        ax = self.ax
        formula = tex(r"z = w_1 \cdot \text{weight} + w_2 \cdot \text{redness} + b", 34, INK).move_to([3.65, 2.0, 0])
        learned = code(
            f"learned: w1={LOGIT_A[0]:.2f}  w2={LOGIT_A[1]:.2f}  b={LOGIT_C:.1f}".replace("-", "−"), 20, MUTED
        ).next_to(formula, m.DOWN, buff=0.3)

        sx = axes([0, 16, 2], [0, 1, 0.5], 5.0, 2.5)
        sx.shift(np.array([1.25, -2.75, 0]) - sx.c2p(0, 0))
        curve = sx.plot(lambda u: sigmoid(u - 8), x_range=[0, 16]).set_stroke(SKY, width=5)
        s_title = text("score z", 24, MUTED).next_to(sx.c2p(16, 0), m.DOWN, buff=0.22).shift(0.3 * m.LEFT)
        mid = text("0", 20, MUTED).next_to(sx.c2p(8, 0), m.DOWN, buff=0.18)
        p_title = text("P(orange)", 22, MUTED).next_to(sx.y_axis.get_end(), m.RIGHT, buff=0.2)
        one = text("1", 20, MUTED).next_to(sx.c2p(0, 1), m.LEFT, buff=0.18)
        zero = text("0", 20, MUTED).next_to(sx.c2p(0, 0), m.LEFT, buff=0.18)
        half = text("0.5", 20, MUTED).next_to(sx.c2p(0, 0.5), m.LEFT, buff=0.18)
        squash = tex(r"p = \frac{1}{1 + e^{-z}}", 34, INK).move_to([5.25, 0.35, 0])
        u = Z_MYSTERY + 8
        rise = m.DashedLine(sx.c2p(u, 0), sx.c2p(u, P_MYSTERY), dash_length=0.08).set_stroke(INK, width=2.5)
        on_curve = mystery_marker(sx.c2p(u, P_MYSTERY)).scale(0.8)
        verdict = pill(f"{100 * P_MYSTERY:.0f}% orange", ORANGE, ORANGE_FILL, 24).move_to([5.3, -2.1, 0])
        cut = m.DashedLine(sx.c2p(0, 0.5), sx.c2p(16, 0.5), dash_length=0.12).set_stroke(GOOD, width=2.5)
        cut_name = text("threshold", 20, GOOD, weight=m.SEMIBOLD).next_to(sx.c2p(16, 0.5), m.UP, buff=0.12).shift(0.55 * m.LEFT)

        (w0, r0), (w1, r1) = boundary_ends(LOGIT_A, LOGIT_C)
        boundary = m.Line(ax.c2p(w0, r0), ax.c2p(w1, r1)).set_stroke(GOOD, width=4).set_z_index(1)
        top = max((w0, r0), (w1, r1), key=lambda p: p[1])
        boundary_name = text("decision boundary", 22, GOOD, weight=m.SEMIBOLD).next_to(ax.c2p(*top), m.UP, buff=0.2)
        tints = m.VGroup(half_plane(ax, LOGIT_A, LOGIT_C, False, APPLE), half_plane(ax, LOGIT_A, LOGIT_C, True, ORANGE))
        self.overlays = [formula, learned, sx, curve, s_title, mid, p_title, one, zero, half, squash, rise, on_curve, verdict, cut, cut_name, boundary, boundary_name, tints]

        self.say(
            "[Model two is logistic regression.] Despite its name, it sorts things into categories.",
            m.AnimationGroup(
                self.new_header("MODEL 2 · LOGISTIC REGRESSION", "From a score to a probability"),
                *[m.FadeOut(r, shift=0.2 * m.RIGHT) for r in self.rows], m.FadeOut(self.tag),
            ),
        )
        self.say(
            "[First, it computes a score]: [a weighted sum of its weight and redness, plus a bias.]",
            m.Write(formula),
            m.FadeIn(learned, shift=0.15 * m.UP),
        )
        self.say(
            "[Then an S-shaped curve turns the score into a probability], [between zero and one.]",
            m.AnimationGroup(m.Create(sx), m.FadeIn(s_title), m.FadeIn(p_title), m.Create(curve, run_time=2.0)),
            m.AnimationGroup(m.FadeIn(one), m.FadeIn(zero), m.FadeIn(mid), m.Write(squash)),
        )
        self.say(
            "[Our mystery fruit scores just above zero], [so it's fifty-eight percent likely to be an orange]: a close call.",
            m.AnimationGroup(m.Indicate(self.mystery, color=INK, scale_factor=1.4), m.Create(rise), m.FadeIn(on_curve, scale=1.5)),
            m.FadeIn(verdict, shift=0.2 * m.UP),
        )
        self.say(
            "[We set a threshold at one half]: [above it, we call it an orange.]",
            m.AnimationGroup(m.Create(cut), m.FadeIn(half), m.FadeIn(cut_name)),
            m.Indicate(verdict, color=ORANGE, scale_factor=1.12),
        )
        self.say(
            "[On the chart, that threshold is a straight line]: [the decision boundary.]",
            m.Create(boundary),
            m.AnimationGroup(m.FadeIn(tints), m.FadeIn(boundary_name, shift=0.15 * m.DOWN)),
        )
        self.wait(0.3)
        self.clear(*self.keep())

    # ------------------------------------------------------------------ 3 decision tree

    def tree(self):
        self.next_section(
            "Decision tree",
            notes=f"Best Gini splits: weight between {ROOT_LO:.0f} and {ROOT_HI:.0f} g (shown as 150), then redness between {NEXT_LO} and {NEXT_HI} (shown as 6).",
        )
        ax = self.ax

        def question(words, width):
            box = card(width, 0.82, SKY, SKY_FILL, radius=0.2)
            return m.VGroup(box, text(words, 24, INK, weight=m.SEMIBOLD).move_to(box))

        def leaf(name, color, fill):
            return pill(name, color, fill, 24)

        root = question("weight > 150 g?", 3.0).move_to([3.65, 1.75, 0])
        light = leaf("apple", APPLE, APPLE_FILL).move_to([1.9, 0.05, 0])
        ask = question("redness > 6?", 2.6).move_to([4.95, 0.05, 0])
        big_red = leaf("apple", APPLE, APPLE_FILL).move_to([3.95, -1.75, 0])
        orange = leaf("orange", ORANGE, ORANGE_FILL).move_to([5.85, -1.75, 0])

        def branch(a, b, word, corner):
            """A link and its answer, placed beside the link's middle, clear of the line."""
            line = m.Line(a.get_bottom(), b.get_top(), buff=0.06).set_stroke(MUTED, width=3)
            label = text(word, 24, MUTED, weight=m.SEMIBOLD).next_to(line.point_from_proportion(0.5), corner, buff=0.1)
            return line, label

        no1, no1_t = branch(root, light, "no", m.DR)  # long links: answers inside the fork
        yes1, yes1_t = branch(root, ask, "yes", m.DL)
        yes2, yes2_t = branch(ask, big_red, "yes", m.UL)  # short links: answers outside
        no2, no2_t = branch(ask, orange, "no", m.UR)

        cut1 = m.Line(ax.c2p(ROOT_T, R_MIN), ax.c2p(ROOT_T, R_MAX)).set_stroke(SKY, width=4).set_z_index(1)
        cut2 = m.Line(ax.c2p(ROOT_T, NEXT_T), ax.c2p(W_MAX, NEXT_T)).set_stroke(SKY, width=4).set_z_index(1)
        box_light = rect_on(ax, W_MIN, ROOT_T, R_MIN, R_MAX, APPLE)
        box_red = rect_on(ax, ROOT_T, W_MAX, NEXT_T, R_MAX, APPLE)
        box_orange = rect_on(ax, ROOT_T, W_MAX, R_MIN, NEXT_T, ORANGE)
        readable = pill("learned from the data · readable by anyone", GOOD, GOOD_FILL, 20).move_to([3.65, -3.05, 0])
        self.overlays = [root, light, ask, big_red, orange, no1, no1_t, yes1, yes1_t, yes2, yes2_t, no2, no2_t, cut1, cut2, box_light, box_red, box_orange, readable]

        self.say(
            "[Model three is a decision tree]: a flowchart of yes or no questions.",
            self.new_header("MODEL 3 · DECISION TREE", "A flowchart of questions"),
        )
        self.say(
            "[First question: does it weigh more than one hundred and fifty grams?] [If not, it's an apple.]",
            m.AnimationGroup(m.FadeIn(root, shift=0.2 * m.DOWN), m.Create(cut1)),
            m.AnimationGroup(m.Create(no1), m.FadeIn(no1_t), m.FadeIn(light, shift=0.2 * m.DOWN), m.FadeIn(box_light)),
        )
        self.say(
            "[If it does, a second question]: [is its redness above six?]",
            m.AnimationGroup(m.Create(yes1), m.FadeIn(yes1_t)),
            m.AnimationGroup(m.FadeIn(ask, shift=0.2 * m.DOWN), m.Create(cut2)),
        )
        self.say(
            "[Each answer leads to a box with a label]: [big red apples up here], [and oranges down here.]",
            m.AnimationGroup(m.Create(yes2), m.FadeIn(yes2_t), m.Create(no2), m.FadeIn(no2_t)),
            m.AnimationGroup(m.FadeIn(big_red, shift=0.2 * m.DOWN), m.FadeIn(box_red)),
            m.AnimationGroup(m.FadeIn(orange, shift=0.2 * m.DOWN), m.FadeIn(box_orange)),
        )
        self.say(
            "[Our mystery fruit is heavy, but not very red], [so the tree says: orange.]",
            m.AnimationGroup(
                m.Indicate(self.mystery, color=INK, scale_factor=1.4),
                m.ShowPassingFlash(m.Line(yes1.get_start(), yes1.get_end()).set_stroke(SKY, width=8), time_width=0.8),
                m.Indicate(root, color=SKY, scale_factor=1.06),
            ),
            m.AnimationGroup(
                m.ShowPassingFlash(m.Line(no2.get_start(), no2.get_end()).set_stroke(ORANGE, width=8), time_width=0.8),
                m.Circumscribe(orange, color=ORANGE, buff=0.08),
            ),
        )
        self.say(
            "[The tree learns these questions from the data], [and anyone can read its rules.]",
            m.AnimationGroup(m.Indicate(cut1, color=SKY), m.Indicate(cut2, color=SKY)),
            m.FadeIn(readable, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear(*self.keep())

    # ------------------------------------------------------------------ 4 k-nearest neighbours

    def neighbours(self):
        near = ", ".join(f"{'orange' if IS_ORANGE[i] else 'apple'} {FRUIT[i][0]:.0f} g/{FRUIT[i][1]}" for i in NEAR)
        self.next_section("k-nearest neighbours", notes=f"k = 5 nearest on the chart: {near}. Vote 3–2: orange.")
        ax = self.ax
        centre = ax.c2p(*MYSTERY)
        links = m.VGroup()
        rings = m.VGroup()
        for i in NEAR:
            color = ORANGE if IS_ORANGE[i] else APPLE
            links.add(m.DashedLine(centre, self.dots[i].get_center(), dash_length=0.08, buff=0.2).set_stroke(color, width=3).set_z_index(1))
            rings.add(m.Circle(radius=0.17).set_stroke(color, width=3).move_to(self.dots[i]).set_z_index(2))
        reach = m.DashedVMobject(m.Circle(radius=NEAR_RADIUS), num_dashes=48).set_stroke(MUTED, width=2).move_to(centre)
        k_pill = pill("k = 5", SKY, SKY_FILL, 26).move_to([3.65, 1.95, 0])
        distance = tex(r"d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}", 36, INK).move_to([3.65, 0.85, 0])
        scaled = text("straight-line distance on the chart", 20, MUTED).next_to(distance, m.DOWN, buff=0.25)

        def tally(name, color, count, label):
            icons = m.VGroup(*[icon(name, 0.6) for _ in range(count)]).arrange(m.RIGHT, buff=0.15)
            return m.VGroup(icons, text(label, 24, color, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.35)

        votes = m.VGroup(tally("ep07-orange", ORANGE, 3, "3 oranges"), tally("ep07-apple", APPLE, 2, "2 apples"))
        votes.arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
        ballot = card(5.0, votes.height + 0.6, EDGE, CARD, radius=0.2, stroke=2).move_to([3.65, -1.35, 0])
        votes.move_to(ballot)
        winner = pill("orange wins, 3 to 2", ORANGE, ORANGE_FILL, 24).move_to([3.65, -3.0, 0])
        self.overlays = [links, rings, reach, k_pill, distance, scaled, ballot, votes, winner]

        self.say(
            "[Model four is k nearest neighbours.] [It learns no formula at all]: it just remembers every fruit.",
            self.new_header("MODEL 4 · K-NEAREST NEIGHBOURS", "Ask the closest fruit"),
            m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.6) for d in self.dots], lag_ratio=0.04),
        )
        self.say(
            "[To label the mystery fruit, it finds the k closest fruit.] [Here, k is five.]",
            m.Broadcast(m.Circle(radius=NEAR_RADIUS).set_stroke(INK, width=2), focal_point=centre, n_mobs=3),
            m.AnimationGroup(m.Create(reach), m.LaggedStart(*[m.Create(l) for l in links], lag_ratio=0.15), m.FadeIn(rings), m.FadeIn(k_pill, scale=1.2)),
        )
        self.say(
            "[Closeness is just the straight-line distance on the chart.]",
            m.AnimationGroup(m.Write(distance), m.FadeIn(scaled, shift=0.15 * m.UP)),
        )
        self.say(
            "[Then they vote]: [three oranges, two apples.] [Orange wins.]",
            m.FadeIn(ballot),
            m.LaggedStart(*[m.FadeIn(v, shift=0.2 * m.RIGHT) for v in votes], lag_ratio=0.4),
            m.FadeIn(winner, scale=1.2),
        )
        self.wait(0.3)
        self.clear(*self.keep())

    # ------------------------------------------------------------------ 5 k-means

    def clusters(self):
        moves = " → ".join(f"({c[0][0]:.0f} g/{c[0][1]:.1f}, {c[1][0]:.0f} g/{c[1][1]:.1f})" for c in (off_chart(h) for h in KM_CENTRES))
        self.next_section("k-means", notes=f"k = 2, centroids {moves}; the third assignment changes nothing.")
        ax = self.ax
        origin = ax.c2p(W_MIN, R_MIN)

        def at(c):  # chart units → screen
            return origin + np.array([c[0], c[1], 0])

        groups = (GROUP_A, GROUP_B)
        centroids = [diamond(groups[j]).move_to(at(KM_CENTRES[0][j])) for j in range(2)]
        supervised = pill("supervised: learns from labelled examples", SKY, SKY_FILL, 20).move_to([3.65, 2.0, 0])
        unsupervised = pill("unsupervised: no labels at all", PINK, PINK_FILL, 20).move_to([3.65, 1.3, 0])
        title = text("k-means,  k = 2", 26, INK, weight=m.BOLD)
        steps = m.VGroup(
            text("1   each fruit joins its nearest centroid", 22, INK),
            text("2   each centroid moves to its group's average", 22, INK),
            text("↻   repeat until nothing changes", 22, MUTED),
        ).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        panel = m.VGroup(title, steps).arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT).move_to([3.65, -0.9, 0])
        note = text("It finds the groups. We give them names.", 24, INK, weight=m.SEMIBOLD).move_to([3.65, -2.75, 0])

        def assign(step):
            return m.AnimationGroup(*[d.animate.set_color(groups[g]) for d, g in zip(self.dots, KM_STEPS[step])])

        def spokes(step, centres):
            return m.VGroup(
                *[
                    m.Line(d.get_center(), at(centres[g])).set_stroke(groups[g], width=1.5, opacity=0.45).set_z_index(1)
                    for d, g in zip(self.dots, KM_STEPS[step])
                ]
            )

        first_spokes = spokes(0, KM_CENTRES[0])
        trails = m.VGroup(
            *[
                m.DashedLine(at(a[j]), at(b[j]), dash_length=0.08).set_stroke(groups[j], width=2, opacity=0.7).set_z_index(1)
                for a, b in zip(KM_CENTRES, KM_CENTRES[1:])
                for j in range(2)
            ]
        )

        def move(step):
            return m.AnimationGroup(*[c.animate.move_to(at(KM_CENTRES[step][j])) for j, c in enumerate(centroids)])

        found = m.VGroup(
            icon("ep07-apple", 0.6).move_to(ax.c2p(112, 5.0)),
            icon("ep07-orange", 0.6).move_to(ax.c2p(230, 4.4)),
        )
        self.overlays = [*centroids, supervised, unsupervised, panel, note, first_spokes, trails, found]

        self.say(
            "[So far, every fruit came with a label.] [That's called supervised learning.]",
            m.AnimationGroup(self.new_header("MODEL 5 · K-MEANS", "Groups without labels"), m.FadeOut(self.mystery)),
            m.FadeIn(supervised, shift=0.2 * m.UP),
        )
        self.say(
            "[But what if nobody labelled the fruit?]",
            m.AnimationGroup(*[d.animate.set_color(MUTED) for d in self.dots], m.FadeIn(unsupervised, shift=0.2 * m.UP)),
        )
        self.say(
            "[k means clustering finds groups on its own.] [Choose k, say two, and drop two centroids anywhere.]",
            m.FadeIn(title, shift=0.2 * m.UP),
            m.LaggedStart(*[m.FadeIn(c, scale=2.0) for c in centroids], lag_ratio=0.4),
        )
        self.say(
            "[Step one: each fruit joins its nearest centroid.]",
            m.AnimationGroup(m.FadeIn(steps[0], shift=0.15 * m.UP), m.Create(first_spokes), assign(0)),
        )
        self.say(
            "[Step two: each centroid moves to the middle of its group.]",
            m.AnimationGroup(m.FadeIn(steps[1], shift=0.15 * m.UP), m.FadeOut(first_spokes), m.Create(trails[0:2]), move(1), run_time=1.6),
        )
        self.say(
            "[Then repeat], [until nothing changes.]",
            m.AnimationGroup(m.FadeIn(steps[2], shift=0.15 * m.UP), m.Succession(assign(1), m.AnimationGroup(m.Create(trails[2:4]), move(2))), run_time=2.2),
            m.AnimationGroup(*[m.Flash(at(KM_CENTRES[-1][j]), color=groups[j], flash_radius=0.4) for j in range(2)]),
        )
        self.say(
            "[Without a single label, it found the apples and the oranges.] [It just never learns what to call them.]",
            m.LaggedStart(*[m.FadeIn(f, scale=0.6) for f in found], lag_ratio=0.3),
            m.FadeIn(note, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ when to use which

    def which(self):
        self.next_section("When to use which", notes="Small tabular data: trees and linear models often beat deep nets (e.g. Grinsztajn et al., 2022).")
        rows = m.VGroup()
        for kind, first, second, use in TOOLS:
            pic = pictogram(kind).scale(0.62)
            frame = card(1.35, 0.85, EDGE, CARD, radius=0.15, stroke=2)
            pic.move_to(frame)
            name = "k-nearest neighbours" if kind == "knn" else ("k-means" if kind == "kmeans" else f"{first} {second}")
            words = m.VGroup(text(name, 24, INK, weight=m.BOLD), text(use, 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            rows.add(m.VGroup(m.VGroup(frame, pic), words).arrange(m.RIGHT, buff=0.3))
        rows.arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([-3.55, -0.45, 0])
        tiles = m.VGroup(
            fact_tile(ep_recolored("database", GOOD, GOOD_FILL), "A small table of data?", "Trees and linear models often\nbeat deep networks.", GOOD),
            fact_tile(ep_recolored("search", SKY, SKY_FILL), "Must explain a decision?", "A simple model can show\nexactly why.", SKY),
            fact_tile("llm", "Images, sound or text?", "That's where deep\nnetworks shine.", VIOLET),
        ).arrange(m.DOWN, buff=0.28).move_to([3.5, -0.45, 0])

        self.say("[So, which tool should you reach for?]", self.new_header("WHEN TO USE WHICH", "Start simple"))
        self.say(
            "[A line for numbers], [logistic regression for yes or no], [a tree for readable rules], "
            "[neighbours for similar cases], [and k means for groups without labels.]",
            *[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in rows],
        )
        self.say(
            "[With a small table of data like ours], trees and linear models often beat deep networks, and train in seconds.",
            m.FadeIn(tiles[0], shift=0.2 * m.LEFT),
        )
        self.say("[When you need to explain a decision], a simple model can show you exactly why.", m.FadeIn(tiles[1], shift=0.2 * m.LEFT))
        self.say("[Deep networks shine on huge amounts of images, sound and text.]", m.FadeIn(tiles[2], shift=0.2 * m.LEFT))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Lines, splits, distances.")
        halo = glow(SKY, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Lines", "Splits", "Distances"), (SKY, GOOD, VIOLET))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Start simple: often, the simplest model is the right model.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [lines], [splits], [distances.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Start simple. Often, the simplest model is the right model.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
