"""AI, Explained · Episode 19 · Activation Functions

Running example: a gradient travelling back from the loss through ten layers, and fading.
The film shows why the shape of a nonlinearity matters: each layer multiplies the gradient by
its activation's slope (the chain rule, Episode 12), so the slope decides what survives.

- sigmoid σ(x) = 1 / (1 + e^-x), slope σ(1 − σ): at most 0.25 (at x = 0), ~0.007 at x = 5;
- tanh, slope 1 − tanh²: at most 1 (four times the sigmoid's best), < 0.01 at x = 3;
- ten sigmoid layers, every weight set to 1 (to isolate the slopes), best case 0.25 per layer:
  0.25¹⁰ ≈ 0.00000095 of the gradient reaches layer 1 (less than one millionth);
- ReLU max(0, x), slope 1 for positive inputs (the whole gradient survives) and 0 for
  negative ones (the path is blocked); a dead ReLU: 24 example inputs (seeded, illustrative)
  pushed below zero by one big update, so output, slope and gradient are all zero;
- GELU x·Φ(x): ≈ ReLU for large |x|, smooth, slope from −0.13 to 1.13, 0.13 at x = −0.5.

Every curve is computed here with numpy (derivatives checked against central differences),
and every number the narration quotes is asserted, so the film cannot drift from its words.

Sources (checked 2026-10): vanishing gradients, Hochreiter 1991 (diploma thesis) and Bengio,
Simard & Frasconi 1994; ReLU, Nair & Hinton 2010, Glorot, Bordes & Bengio 2011; GELU,
Hendrycks & Gimpel 2016 (arXiv:1606.08415), used in GPT (Radford et al. 2018) and BERT
(Devlin et al. 2018).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- colours

SIG_C, TANH_C, RELU_C, GELU_C = BLUE, TEAL, LIME, PINK  # one category colour per function
RELU_FILL = "#1E2A0F"
GRAD_C = AMBER  # gradients and slopes (the series' convention, Episodes 2, 12, 13)
NEURON_C, NEURON_FILL = VIOLET, VIOLET_FILL

# ---------------------------------------------------------------------------- the functions


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def d_tanh(x):
    return 1.0 - np.tanh(np.asarray(x, dtype=float)) ** 2


def relu(x):
    return np.maximum(0.0, np.asarray(x, dtype=float))


def d_relu(x):
    return (np.asarray(x, dtype=float) > 0).astype(float)


_erf = np.vectorize(math.erf)


def Phi(x):
    """The standard normal CDF: the chance a bell-curve random number is below x."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def phi(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


def gelu(x):
    return np.asarray(x, dtype=float) * Phi(x)


def d_gelu(x):
    return Phi(x) + np.asarray(x, dtype=float) * phi(x)


# the derivatives really are the slopes: central differences
_XS = np.linspace(-4, 4, 161)
_H = 1e-5
for _f, _df in ((sigmoid, d_sigmoid), (np.tanh, d_tanh), (gelu, d_gelu)):
    assert np.allclose((_f(_XS + _H) - _f(_XS - _H)) / (2 * _H), _df(_XS), atol=1e-5)
_OFF = _XS[np.abs(_XS) > 0.01]
assert np.allclose((relu(_OFF + _H) - relu(_OFF - _H)) / (2 * _H), d_relu(_OFF))
assert np.allclose(np.tanh(_XS), 2 * sigmoid(2 * _XS) - 1)  # "the same S-shape, stretched"

# ---------------------------------------------------------------------------- the numbers

GRID = np.linspace(-8, 8, 160001)
SIG_MAX = float(d_sigmoid(GRID).max())
assert float(d_sigmoid(0.0)) == 0.25 and abs(SIG_MAX - 0.25) < 1e-12  # "peaks in the middle at only a quarter"
SIG_AT_5 = float(d_sigmoid(5.0))
assert SIG_AT_5 < 0.01  # "far out on either side it's almost flat" (readout 0.0066)
TANH_MAX = float(d_tanh(GRID).max())
assert float(d_tanh(0.0)) == 1.0 and TANH_MAX == 1.0  # "the slope is a full one"
assert TANH_MAX / 0.25 == 4  # "four times the sigmoid's best"
TANH_AT_3 = float(d_tanh(3.0))
assert TANH_AT_3 < 0.01  # "fades to almost nothing" (readout 0.0099)

DEPTH = 10
GRADS = 0.25 ** np.arange(DEPTH + 1)  # the gradient after j bends: j = 0 at the loss, j = 10 at layer 1
assert GRADS[1] == 0.25 and GRADS[2] == 1 / 16  # "to a quarter, a sixteenth"
assert GRADS[-1] < 1e-6 and round(math.log10(GRADS[-1])) == -6  # "less than one millionth", ≈ 10⁻⁶
assert f"{GRADS[-1]:.8f}" == "0.00000095"
assert np.prod(d_relu(np.full(DEPTH, 1.0))) == 1.0  # ReLU, positive inputs: "the whole gradient"
assert float(d_relu(-0.5)) == 0.0  # "a negative input has a slope of zero"
DEAD = 6  # the layer whose neuron gets a negative input

# a dead ReLU: one neuron's input for 24 training examples (illustrative), then one big update
_rng = np.random.default_rng(19)
Z_BEFORE = _rng.normal(0.9, 0.7, 24)
UPDATE = -3.3
Z_AFTER = Z_BEFORE + UPDATE
assert (Z_BEFORE > 0).sum() == 23 and Z_AFTER.max() < 0  # "below zero for every example"
assert np.all(relu(Z_AFTER) == 0) and np.all(d_relu(Z_AFTER) == 0)  # "outputs zero, its slope is zero"

GELU_LO, GELU_HI = float(d_gelu(GRID).min()), float(d_gelu(GRID).max())
assert (round(GELU_LO, 2), round(GELU_HI, 2)) == (-0.13, 1.13)
assert float(Phi(2.6)) > 0.99 and float(Phi(-2.6)) < 0.01  # "nearly one", "nearly zero"
assert abs(float(gelu(2.6)) - 2.6) < 0.02 and abs(float(gelu(-2.6))) < 0.02  # "acts like ReLU"
assert round(float(d_gelu(-0.5)), 2) == 0.13  # "slightly negative inputs still pass a little gradient"
assert float(d_gelu(0.0)) == 0.5

# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def formula(*parts, size=44):
    """A MathTex from (latex, colour) pairs, one part each."""
    t = m.MathTex(*[p for p, _ in parts], font_size=size, color=INK)
    for part, (_, c) in zip(t, parts):
        part.set_color(c)
    return t


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


class Plot:
    """Quiet axes (axes cross at the origin); graphs are cut where they leave the box."""

    def __init__(self, x_range, y_range, width, height, center, label="x", x_step=1, y_step=0.5, small=False):
        (self.x0, self.x1), (self.y0, self.y1) = x_range, y_range
        if small:  # thin axes, small tips, no ticks
            self.ax = m.Axes(
                x_range=[self.x0, self.x1, x_step], y_range=[self.y0, self.y1, y_step], x_length=width, y_length=height,
                axis_config={"color": FAINT, "stroke_width": 2, "include_tip": True, "tip_length": 0.1, "include_ticks": False},
            ).move_to(center)
        else:
            self.ax = axes([self.x0, self.x1, x_step], [self.y0, self.y1, y_step], width, height).move_to(center)
        self.end = self.x1 - 0.3 * (self.x1 - self.x0) / width  # stop short of the arrow tip
        self.frame = m.VGroup(self.ax)
        if label:
            self.frame.add(text(label, 26, MUTED).next_to(self.ax.x_axis.get_end(), m.DOWN, buff=0.15))

    def p(self, x, y):
        return np.array(self.ax.c2p(float(x), float(y)))

    def _cross(self, xa, ya, xb, yb):
        edge = self.y1 if max(ya, yb) > self.y1 else self.y0
        t = (edge - ya) / (yb - ya)
        return self.p(xa + t * (xb - xa), edge)

    def graph(self, fn, color, width=4.5, x0=None, x1=None, n=600, opacity=1.0):
        xs = np.linspace(self.x0 if x0 is None else x0, self.end if x1 is None else x1, n)
        ys = np.asarray(fn(xs), dtype=float)
        pieces, run = [], []
        for i, (x, y) in enumerate(zip(xs, ys)):
            if self.y0 <= y <= self.y1:
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

    def steps(self, color, width=4.5, opacity=1.0):
        """ReLU's slope: 0 for negative inputs, 1 for positive ones (two pieces)."""
        return m.VGroup(
            m.Line(self.p(self.x0, 0), self.p(0, 0)).set_stroke(color, width=width, opacity=opacity),
            m.Line(self.p(0, 1), self.p(self.end, 1)).set_stroke(color, width=width, opacity=opacity),
        )

    def hline(self, y, color=FAINT, opacity=1.0):
        return m.DashedLine(self.p(self.x0, y), self.p(self.end, y), dash_length=0.1).set_stroke(color, width=2.5, opacity=opacity)


def legend(*items):
    """Short coloured strokes with names, in one row."""
    row = m.VGroup()
    for color, label, opacity in items:
        row.add(
            m.VGroup(
                m.Line(m.ORIGIN, 0.45 * m.RIGHT).set_stroke(color, width=5, opacity=opacity),
                text(label, 22, color if opacity == 1 else MUTED, weight=m.SEMIBOLD),
            ).arrange(m.RIGHT, buff=0.15)
        )
    return row.arrange(m.RIGHT, buff=0.5)


def readouts(rows, width=4.3, row_h=0.54):
    """Live numbers in a card: rows of (label, colour, getter, decimals, baseline glyph)."""
    box = card(width, row_h * len(rows) + 0.4, EDGE, CARD)
    labels, values = m.VGroup(), []
    top = box.get_top()[1] - 0.2 - row_h / 2
    for k, (label, color, get, places, anchor) in enumerate(rows):
        base = top - k * row_h - 0.11
        lab = text(label, 24, color, weight=m.SEMIBOLD).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
        lab.shift((base - lab[anchor].get_bottom()[1]) * m.UP)
        num = m.DecimalNumber(get(), num_decimal_places=places, mob_class=Mono, font_size=28, color=color)
        num.align_to(box, m.LEFT).shift(2.05 * m.RIGHT)
        num.shift((base - num[-1].get_bottom()[1]) * m.UP)
        num.add_updater(lambda d, g=get: d.set_value(g()))
        labels.add(lab)
        values.append(num)
    return m.VGroup(box, labels, *values), values


PLOT_C = np.array([-3.3, -0.45, 0])
PLOT_W, PLOT_H = 6.3, 4.3
RIGHT_X = 3.65


class Stage:
    """One activation on the left plot: its curve, its slope, a dot riding the curve with a tangent,
    and live readouts (x, output, slope) for the right column."""

    def __init__(self, fn, dfn, color, x_range, y_range, start, y_step=0.5, extra_rows=()):
        self.fn, self.dfn, self.color = fn, dfn, color
        self.plot = plot = Plot(x_range, y_range, PLOT_W, PLOT_H, PLOT_C, y_step=y_step)
        self.curve = plot.graph(fn, color, width=5)
        self.slope = plot.graph(dfn, GRAD_C, width=4) if dfn is not d_relu else plot.steps(GRAD_C, width=4)
        self.x = m.ValueTracker(start)
        x = self.xv
        self.dot = m.always_redraw(lambda: m.Dot(plot.p(x(), fn(x())), radius=0.11, color=color).set_z_index(6))
        self.slope_dot = m.always_redraw(lambda: m.Dot(plot.p(x(), dfn(x())), radius=0.085, color=GRAD_C).set_z_index(6))
        self.tangent = m.always_redraw(self._tangent)
        extra = [(label, c, (lambda f=f: float(f(x()))), places, anchor) for label, c, f, places, anchor in extra_rows]
        rows = [("x", INK, x, 2, 0), *extra, ("output", color, lambda: float(fn(x())), 3, 0), ("slope", GRAD_C, lambda: float(dfn(x())), 4, 0)]
        self.card, self.numbers = readouts(rows)
        self.card.move_to([RIGHT_X, -0.95 - 0.5 * len(extra_rows), 0])

    def xv(self):
        return float(self.x.get_value())

    def _tangent(self):
        x = self.xv()
        y, s = float(self.fn(x)), float(self.dfn(x))
        p0 = self.plot.p(x, y)
        v = self.plot.p(x + 1, y + s) - p0
        v = v / np.linalg.norm(v)
        return m.Line(p0 - 0.7 * v, p0 + 0.7 * v).set_stroke(GRAD_C, width=4).set_z_index(5)

    def live(self):
        return [self.dot, self.slope_dot, self.tangent, *self.numbers]


def mini_curve(kind, color, width=0.4, height=0.28, stroke=3):
    """A tiny activation drawn inside a neuron: "s" (sigmoid) or "r" (ReLU)."""
    us = np.linspace(-1, 1, 41)
    vs = sigmoid(5 * us) - 0.5 if kind == "s" else relu(us) - 0.5
    return polyline([[u * width / 2, v * height, 0] for u, v in zip(us, vs)], color, width=stroke)


# the ten-layer row: layer k at LAYER_X[k-1]; the gradient after j bends sits at BAR_X[j]
LAYER_X = [-5.9 + 1.12 * k for k in range(DEPTH)]
LOSS_X = 5.75
BAR_X = [LOSS_X] + [LAYER_X[DEPTH - j] for j in range(1, DEPTH + 1)]
PULSE_X = [LOSS_X - 0.8] + BAR_X[1:]


def layer_row(y, kind=None, color=INK):
    neurons = m.VGroup(
        *[m.Circle(radius=0.3).set_stroke(NEURON_C, width=3).set_fill(NEURON_FILL, opacity=1).move_to([x, y, 0]) for x in LAYER_X]
    )
    marks = m.VGroup(*[mini_curve(kind, color).move_to(n) for n in neurons]) if kind else m.VGroup()
    box = card(1.1, 0.7, BAD, BAD_FILL, radius=0.2).move_to([LOSS_X, y, 0])
    loss = m.VGroup(box, text("loss", 24, INK, weight=m.SEMIBOLD).move_to(box))
    links = m.VGroup(
        *[m.Line(a.get_right(), b.get_left()).set_stroke(FAINT, width=2.5) for a, b in zip(neurons, neurons[1:])],
        m.Line(neurons[-1].get_right(), box.get_left()).set_stroke(FAINT, width=2.5),
    )
    links.set_z_index(-1)
    numbers = m.VGroup(*[text(str(k + 1), 20, MUTED).next_to(n, m.DOWN, buff=0.16) for k, n in enumerate(neurons)])
    row = SimpleNamespace(neurons=neurons, marks=marks, loss=loss, links=links, numbers=numbers, y=y)
    row.all = m.VGroup(links, neurons, marks, loss, numbers)
    return row


def show_row(row, lag=0.06):
    return m.AnimationGroup(
        m.Create(row.links),
        m.LaggedStart(*[m.FadeIn(m.VGroup(n, k) if len(row.marks) else n, scale=0.6) for n, k in zip(row.neurons, row.marks if len(row.marks) else row.neurons)], lag_ratio=lag),
        m.FadeIn(row.numbers),
        m.FadeIn(row.loss, scale=0.8),
    )


def pulse_dot(r, y):
    """The gradient as a travelling dot, dimmer the more sigmoid bends (0.25 each) it has passed."""
    r = float(np.clip(r, 0, DEPTH))
    j = min(int(r), DEPTH - 1)
    a = r - j
    x = PULSE_X[j] + (PULSE_X[j + 1] - PULSE_X[j]) * a
    o = float(np.clip(1 + math.log10(0.25**r) / 6.5, 0.1, 1.0))
    return m.Dot([x, y, 0], radius=0.07 + 0.13 * o, color=GRAD_C).set_opacity(o).set_z_index(6)


def bar(x, base, height, color=GRAD_C, width=0.42):
    rect = m.Rectangle(width=width, height=max(height, 0.004), stroke_width=0).set_fill(color, opacity=0.9)
    return rect.move_to([x, base, 0], aligned_edge=m.DOWN)


def stop(*mobs):
    for mob in mobs:
        mob.clear_updaters()


# ---------------------------------------------------------------------------- the episode


class Ep19ActivationFunctions(Episode):
    number = 19

    def lesson(self):
        self.hook()
        self.bend()
        self.sigmoid()
        self.tanh()
        self.vanishing()
        self.relu()
        self.relu_depth()
        self.gelu()
        self.big_picture()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A gradient fades as it travels back through ten layers (the dot's brightness follows 0.25 per layer, the sigmoid's best case).")
        row = layer_row(0.15)
        reach = m.ValueTracker(0.0)
        pulse = m.always_redraw(lambda: pulse_dot(reach.get_value(), row.y))
        grad_word = text("gradient", 26, GRAD_C, weight=m.BOLD).move_to([4.0, 1.5, 0])
        grad_arrow = arrow([5.1, 1.0, 0], [2.9, 1.0, 0], GRAD_C, buff=0)
        faint = text("almost nothing arrives", 24, BAD, weight=m.SEMIBOLD).move_to([LAYER_X[1], 1.2, 0])
        marks = m.VGroup(*[mini_curve("s", INK).move_to(n) for n in row.neurons])
        name = pill("the activation function", NEURON_C, NEURON_FILL, 26).move_to([0, -1.75, 0])

        self.say(
            "So... [here's a deep network: ten layers, one after another].",
            m.AnimationGroup(
                m.Create(row.links),
                m.LaggedStart(*[m.FadeIn(n, scale=0.5) for n in row.neurons], lag_ratio=0.12),
                m.FadeIn(row.numbers),
                m.FadeIn(row.loss, scale=0.8),
            ),
        )
        self.say(
            "[In training, a signal flows backward from the loss]: [the gradient, telling each layer how to change.]",
            m.AnimationGroup(m.Indicate(row.loss, color=BAD, scale_factor=1.1), m.Succession(m.FadeIn(pulse, scale=0.5), reach.animate(run_time=1.0).set_value(1))),
            m.AnimationGroup(m.FadeIn(grad_word, shift=0.2 * m.LEFT), m.GrowArrow(grad_arrow)),
        )
        self.say(
            "[But as it travels back, it can fade], [until the first layers get almost nothing.]",
            reach.animate(run_time=2.8, rate_func=m.linear).set_value(DEPTH),
            m.AnimationGroup(m.FadeIn(faint, shift=0.2 * m.DOWN), *[n.animate.set_stroke(BAD) for n in row.neurons[:3]]),
        )
        stop(pulse)
        self.say(
            "[What decides how much survives?] [A small curve inside every neuron]: [its activation function.]",
            m.AnimationGroup(m.FadeOut(faint), m.FadeOut(pulse), *[n.animate.set_stroke(NEURON_C) for n in row.neurons[:3]]),
            m.LaggedStart(*[m.FadeIn(k, scale=0.5) for k in marks], lag_ratio=0.1),
            m.FadeIn(name, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the bend: where the slope enters

    def bend(self):
        self.next_section("The bend", notes="One neuron: weighted sum z, activation a = f(z), loss. Backward: dL/dz = f'(z) · dL/da (the chain rule, Episode 12). The box holds a sigmoid; the arrow's width follows its slope.")
        y, back_y = -0.35, -2.05
        ins = m.VGroup(
            *[m.Circle(radius=0.28).set_stroke(SKY, width=3).set_fill(SKY_FILL, opacity=1).move_to([-6.1, yy, 0]) for yy in (0.75, -0.35, -1.45)]
        )
        ssum = m.Circle(radius=0.5).set_stroke(NEURON_C, width=3.5).set_fill(NEURON_FILL, opacity=1).move_to([-4.0, y, 0])
        sym = tex(r"\Sigma", 40).move_to(ssum)
        in_links = m.VGroup(
            *[m.Line(c.get_right(), ssum.get_center() + 0.5 * m.normalize(c.get_center() - ssum.get_center())).set_stroke(MUTED, width=2.5) for c in ins]
        )
        box = card(2.5, 2.0, NEURON_C, NEURON_FILL).move_to([-1.0, y, 0])
        box_title = text("activation", 22, NEURON_C, weight=m.SEMIBOLD).next_to(box, m.UP, buff=0.15)
        mini = Plot((-5, 5), (-0.15, 1.15), 2.0, 1.35, box.get_center(), label=None, x_step=1, y_step=0.5, small=True)
        mini_curve_ = mini.graph(sigmoid, SIG_C, width=4)
        z0 = m.ValueTracker(-1.2)

        def mini_dot():
            z = float(z0.get_value())
            return m.Dot(mini.p(z, sigmoid(z)), radius=0.075, color=INK).set_z_index(6)

        def mini_tangent():
            z = float(z0.get_value())
            p0 = mini.p(z, sigmoid(z))
            v = mini.p(z + 1, sigmoid(z) + d_sigmoid(z)) - p0
            v = v / np.linalg.norm(v)
            return m.Line(p0 - 0.3 * v, p0 + 0.3 * v).set_stroke(GRAD_C, width=4).set_z_index(5)

        def back_c():
            s = float(d_sigmoid(z0.get_value()))
            a = m.Arrow([-2.35, back_y, 0], [-3.75, back_y, 0], buff=0, color=GRAD_C, stroke_width=0.6 + 29.6 * s,
                        max_tip_length_to_length_ratio=0.3, max_stroke_width_to_length_ratio=20)
            return a.set_opacity(0.35 + 0.65 * s / 0.25)

        dot, tangent, back = m.always_redraw(mini_dot), m.always_redraw(mini_tangent), m.always_redraw(back_c)
        z_arrow = arrow(ssum.get_right(), box.get_left(), MUTED)
        z_label = tex("z", 40).next_to(z_arrow, m.UP, buff=0.12)
        out = m.Circle(radius=0.42).set_stroke(NEURON_C, width=3.5).set_fill(NEURON_FILL, opacity=1).move_to([1.9, y, 0])
        a_sym = tex("a", 44).move_to(out)
        a_arrow = arrow(box.get_right(), out.get_left(), MUTED)
        loss_box = card(1.3, 0.8, BAD, BAD_FILL, radius=0.22).move_to([4.6, y, 0])
        loss = m.VGroup(loss_box, text("loss", 26, INK, weight=m.SEMIBOLD).move_to(loss_box))
        l_arrow = arrow(out.get_right(), loss_box.get_left(), MUTED)
        back_a = m.Arrow([4.25, back_y, 0], [0.35, back_y, 0], buff=0, color=GRAD_C, stroke_width=8,
                         max_tip_length_to_length_ratio=0.08, max_stroke_width_to_length_ratio=20)
        lab_a = tex(r"\partial L / \partial a", 30).move_to([2.3, back_y - 0.5, 0])
        lab_z = tex(r"\partial L / \partial z", 30).move_to([-3.05, back_y - 0.5, 0])
        slope_num = m.DecimalNumber(float(d_sigmoid(z0.get_value())), num_decimal_places=2, mob_class=Mono, font_size=26, color=GRAD_C)
        times = m.VGroup(text("× slope", 22, GRAD_C, weight=m.SEMIBOLD), slope_num).arrange(m.RIGHT, buff=0.15).move_to([-1.0, back_y, 0])
        slope_num.add_updater(lambda d: d.set_value(float(d_sigmoid(z0.get_value()))))
        rule = formula((r"\frac{\partial L}{\partial z}", INK), ("=", INK), ("f'(z)", GRAD_C), (r"\cdot", INK), (r"\frac{\partial L}{\partial a}", INK), size=44)
        rule.move_to([0.6, 1.95, 0])

        self.say(
            "[As we saw in episode three, a neuron adds up its weighted inputs], [then bends the sum with an activation function.]",
            m.AnimationGroup(
                self.new_header("THE SETUP", "Every gradient passes through a slope"),
                m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in ins], lag_ratio=0.2),
                m.Create(in_links),
                m.FadeIn(m.VGroup(ssum, sym), scale=0.7),
                m.GrowArrow(z_arrow),
                m.FadeIn(z_label),
            ),
            m.AnimationGroup(
                m.FadeIn(box),
                m.FadeIn(box_title),
                m.Create(mini.frame),
                m.Create(mini_curve_),
                m.GrowArrow(a_arrow),
                m.FadeIn(m.VGroup(out, a_sym), scale=0.7),
                m.GrowArrow(l_arrow),
                m.FadeIn(loss, scale=0.8),
            ),
        )
        self.say(
            "[In training, the gradient comes back through that bend], [and gets multiplied by its slope.]",
            m.AnimationGroup(m.GrowArrow(back_a), m.FadeIn(lab_a)),
            m.AnimationGroup(m.FadeIn(dot, scale=0.5), m.FadeIn(tangent), m.FadeIn(times), m.FadeIn(back), m.FadeIn(lab_z)),
        )
        self.say(
            "[It's the chain rule from episode twelve]: [the gradient before the bend] [is the slope], [times the gradient after it.]",
            m.Write(rule),
            m.AnimationGroup(m.Indicate(rule[0], color=GRAD_C, scale_factor=1.15), m.Indicate(lab_z, color=GRAD_C)),
            m.AnimationGroup(m.Indicate(rule[2], color=GRAD_C, scale_factor=1.2), m.Indicate(times, color=GRAD_C)),
            m.AnimationGroup(m.Indicate(rule[4], color=GRAD_C, scale_factor=1.15), m.Indicate(lab_a, color=GRAD_C)),
        )
        self.say(
            "[So a steep slope passes the gradient on], [and a flat slope chokes it off.]",
            z0.animate(run_time=1.4).set_value(0.0),
            z0.animate(run_time=1.8).set_value(4.0),
        )
        self.wait(0.3)
        stop(dot, tangent, back, slope_num)
        self.clear()

    # ------------------------------------------------------------------ sigmoid

    def sigmoid(self):
        self.next_section("Sigmoid", notes="σ(x) = 1/(1+e^-x); slope σ(1−σ), at most 0.25 at x = 0, 0.0066 at x = 5 (saturation).")
        st = Stage(sigmoid, d_sigmoid, SIG_C, (-6, 6), (-0.25, 1.25), start=0.0, y_step=0.25)
        plot = st.plot
        one = plot.hline(1.0)
        one_label = text("1", 20, MUTED).move_to(plot.p(-5.4, 1.0) + 0.24 * m.UP)
        leg = legend((SIG_C, "sigmoid σ(x)", 1), (GRAD_C, "its slope σ′(x)", 1)).move_to([PLOT_C[0], 2.2, 0])
        f1 = formula((r"\sigma(x)", SIG_C), ("=", INK), (r"\frac{1}{1 + ", INK), (r"e^{-x}", INK), ("}", INK), size=48).move_to([RIGHT_X, 1.6, 0])
        note1 = text("squashes any input into (0, 1)", 22, MUTED).move_to([RIGHT_X, 0.62, 0])
        f2 = formula((r"\sigma'(x)", GRAD_C), ("=", INK), (r"\sigma(x)", SIG_C), (r"\bigl(1 - \sigma(x)\bigr)", INK), size=44).move_to([RIGHT_X, 1.6, 0])
        note2 = text("the slope, from the output itself", 22, MUTED).move_to([RIGHT_X, 0.62, 0])
        quarter = plot.hline(0.25, GRAD_C, opacity=0.6)
        quarter_label = text("max 0.25", 20, GRAD_C, weight=m.SEMIBOLD).move_to(plot.p(4.4, 0.25) + 0.24 * m.UP)
        sat = m.VGroup(
            text("saturated", 22, MUTED).move_to(plot.p(-4.2, 0.47)),
            text("saturated", 22, MUTED).move_to(plot.p(4.2, 0.62)),
        )
        best = pill("slope ≤ 0.25, everywhere", GRAD_C, AMBER_FILL, 24).move_to([RIGHT_X, -2.65, 0])

        self.say(
            "[The classic choice is the sigmoid.] [It squashes any number into the range from zero to one.]",
            m.AnimationGroup(self.new_header("SIGMOID", "Squash every input into zero to one"), m.Create(plot.frame), m.FadeIn(one), m.FadeIn(one_label), m.FadeIn(leg[0])),
            m.AnimationGroup(m.Create(st.curve, run_time=1.6), m.Write(f1), m.FadeIn(note1)),
        )
        self.say(
            "[For large inputs, the exponential on the bottom vanishes], [so the output creeps up toward one.]",
            m.AnimationGroup(m.Indicate(f1[3], color=SIG_C, scale_factor=1.3), m.FadeIn(st.card), m.FadeIn(st.dot, scale=0.5), m.FadeIn(st.tangent)),
            st.x.animate(run_time=2.0).set_value(5.5),
        )
        self.say(
            "[For very negative inputs, it explodes], [and the output sinks toward zero.]",
            m.Indicate(f1[3], color=BAD, scale_factor=1.3),
            st.x.animate(run_time=2.4).set_value(-5.5),
        )
        self.say(
            "[Its slope has a neat form]: [the sigmoid, times one minus itself.]",
            m.AnimationGroup(m.FadeTransform(f1, f2), m.FadeTransform(note1, note2)),
            m.AnimationGroup(m.Create(st.slope, run_time=1.6), m.FadeIn(leg[1]), m.FadeIn(st.slope_dot, scale=0.5)),
        )
        self.say(
            "[The slope peaks in the middle at only a quarter], [and far out on either side it's almost flat: the neuron saturates.]",
            m.AnimationGroup(st.x.animate(run_time=1.6).set_value(0.0), m.Create(quarter), m.FadeIn(quarter_label), m.FadeIn(best, shift=0.2 * m.UP)),
            m.AnimationGroup(st.x.animate(run_time=2.0).set_value(5.0), m.FadeIn(sat)),
        )
        self.wait(0.3)
        stop(*st.live())
        self.clear()

    # ------------------------------------------------------------------ tanh

    def tanh(self):
        self.next_section("Tanh", notes="tanh(x) = 2σ(2x) − 1; slope 1 − tanh², 1 at x = 0 (four times 0.25), 0.0099 at x = 3.")
        st = Stage(np.tanh, d_tanh, TANH_C, (-4, 4), (-1.5, 1.5), start=-2.5)
        plot = st.plot
        lines = m.VGroup(plot.hline(1.0), plot.hline(-1.0))
        labels = m.VGroup(
            text("1", 20, MUTED).move_to(plot.p(-3.6, 1.0) + 0.24 * m.UP),
            text("−1", 20, MUTED).move_to(plot.p(3.0, -1.0) + 0.24 * m.DOWN),
        )
        leg = legend((TANH_C, "tanh(x)", 1), (GRAD_C, "its slope", 1)).move_to([PLOT_C[0], 2.2, 0])
        rule = formula((r"\mathrm{tanh}'(x)", GRAD_C), ("=", INK), ("1 -", INK), (r"\tanh^2(x)", TANH_C), size=48).move_to([RIGHT_X, 1.6, 0])
        note = text("range −1 to 1, centred on zero", 22, MUTED).move_to([RIGHT_X, 0.62, 0])
        ghost = m.DashedVMobject(plot.graph(d_sigmoid, GRAD_C, width=3, opacity=0.55), num_dashes=60)
        ghost_label = text("sigmoid's slope", 20, MUTED).move_to(plot.p(2.0, -0.32))
        four = pill("peak slope 1: four times the sigmoid's", GRAD_C, AMBER_FILL, 22).move_to([RIGHT_X, -2.65, 0])
        sat = m.VGroup(
            text("saturates", 22, MUTED).move_to(plot.p(-3.0, -0.6)),
            text("saturates", 22, MUTED).move_to(plot.p(3.0, 0.6)),
        )

        self.say(
            "[Its cousin, the hyperbolic tangent, has the same S-shape], [stretched to run from minus one to one.]",
            m.AnimationGroup(self.new_header("TANH", "The same S, from minus one to one"), m.Create(plot.frame), m.FadeIn(lines), m.FadeIn(leg[0])),
            m.AnimationGroup(m.Create(st.curve, run_time=1.6), m.FadeIn(labels), m.FadeIn(note), m.FadeIn(st.card), m.FadeIn(st.dot, scale=0.5), m.FadeIn(st.tangent)),
        )
        self.say(
            "[Its slope is one minus its own value, squared.]",
            m.AnimationGroup(m.Write(rule), m.Create(st.slope, run_time=1.6), m.FadeIn(leg[1]), m.FadeIn(st.slope_dot, scale=0.5)),
        )
        self.say(
            "[In the middle, its value is zero, so the slope is a full one]: [four times the sigmoid's best.]",
            st.x.animate(run_time=2.0).set_value(0.0),
            m.AnimationGroup(m.Create(ghost), m.FadeIn(ghost_label), m.FadeIn(four, shift=0.2 * m.UP)),
        )
        self.say(
            "[But toward either end, its value nears plus or minus one], [so the slope fades to almost nothing. It saturates too.]",
            st.x.animate(run_time=2.0).set_value(3.0),
            m.FadeIn(sat),
        )
        self.wait(0.3)
        stop(*st.live())
        self.clear()

    # ------------------------------------------------------------------ vanishing gradients

    def vanishing(self):
        self.next_section(
            "Vanishing gradients",
            notes="Ten sigmoid layers, one neuron each, every weight set to 1 (illustrative, to isolate the slopes), best case slope 0.25: "
            "0.25^10 = 0.00000095 reaches layer 1. Bars are linear in the gradient. Hochreiter 1991; Bengio, Simard & Frasconi 1994.",
        )
        ny, yb, tall = -1.85, -1.3, 3.3
        row = layer_row(ny, "s", SIG_C)
        slots = m.VGroup(*[m.Line([x - 0.3, yb, 0], [x + 0.3, yb, 0]).set_stroke(FAINT, width=2.5) for x in BAR_X])
        bars = [bar(BAR_X[j], yb, tall * GRADS[j]) for j in range(DEPTH + 1)]
        vals = [code(s, 20, GRAD_C).next_to(bars[j], m.UP, buff=0.1) for j, s in enumerate(("1", "0.25", "0.063", "0.016"))]
        reach = m.ValueTracker(0.0)
        pulse = m.always_redraw(lambda: pulse_dot(reach.get_value(), ny))
        value = m.DecimalNumber(1.0, num_decimal_places=8, mob_class=Mono, font_size=36, color=GRAD_C)
        value.add_updater(lambda d: d.set_value(0.25 ** reach.get_value()))
        readout = m.VGroup(text("gradient", 22, MUTED, weight=m.SEMIBOLD), value).arrange(m.DOWN, buff=0.12).move_to([2.75, 1.55, 0])

        chain = formula(
            (r"\frac{\partial L}{\partial z_1}", INK), ("=", INK), (r"\sigma'(z_1)\,", GRAD_C), (r"\sigma'(z_2)\,", GRAD_C),
            (r"\cdots\,", GRAD_C), (r"\sigma'(z_{10})\,", GRAD_C), (r"\frac{\partial L}{\partial a_{10}}", INK), size=38,
        ).move_to([-2.5, 1.8, 0])
        brace = m.Brace(chain[2:6], m.DOWN, buff=0.1).set_color(GRAD_C)
        brace.shift((chain.get_bottom()[1] - 0.12 - brace.get_top()[1]) * m.UP)
        ten = text("ten slopes", 22, GRAD_C, weight=m.SEMIBOLD).next_to(brace, m.DOWN, buff=0.1)
        each = text("ten slopes, each at most 0.25", 22, GRAD_C, weight=m.SEMIBOLD).next_to(brace, m.DOWN, buff=0.1)
        weights = text("every weight set to 1, to isolate the slopes", 20, MUTED).move_to([-2.5, 0.2, 0])
        million = pill("less than one millionth", BAD, BAD_FILL, 22).move_to([-4.55, -0.55, 0])
        million_arrow = arrow([LAYER_X[0], million.get_bottom()[1], 0], [LAYER_X[0], yb + 0.08, 0], BAD, buff=0.05)
        barely = text("barely learn", 22, BAD, weight=m.SEMIBOLD).move_to([(LAYER_X[0] + LAYER_X[3]) / 2, -2.9, 0])
        problem = pill("the vanishing gradient problem", NEURON_C, NEURON_FILL, 24).move_to([1.2, -3.02, 0])
        cite = text("Hochreiter, 1991 · Bengio et al., 1994", 20, MUTED).next_to(problem, m.DOWN, buff=0.12)

        self.say(
            "[Now stack ten sigmoid layers], [and follow the gradient back from the loss.]",
            m.AnimationGroup(self.new_header("VANISHING GRADIENTS", "A quarter, ten times over"), show_row(row), m.FadeIn(slots)),
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.DOWN), m.FadeIn(vals[0]), m.FadeIn(readout), m.FadeIn(pulse, scale=0.5)),
        )
        self.say(
            "[With every weight set to one], [each layer multiplies the gradient by its slope]: [ten slopes in all.]",
            m.FadeIn(weights, shift=0.15 * m.UP),
            m.Write(chain),
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(ten)),
        )
        self.say(
            "[Even at its very best, each slope is only a quarter.]",
            m.AnimationGroup(m.FadeTransform(ten, each), m.LaggedStart(*[m.Indicate(p, color=GRAD_C, scale_factor=1.2) for p in chain[2:6]], lag_ratio=0.2)),
        )
        steps = [m.AnimationGroup(m.GrowFromEdge(bars[j], m.DOWN), *([m.FadeIn(vals[3])] if j == 3 else []), run_time=3.4 / 8) for j in range(3, DEPTH + 1)]
        self.say(
            "[So the gradient shrinks]: [to a quarter], [a sixteenth], [and on and on, layer after layer.]",
            m.Indicate(readout, color=GRAD_C, scale_factor=1.08),
            m.AnimationGroup(reach.animate(run_time=0.8).set_value(1), m.GrowFromEdge(bars[1], m.DOWN, run_time=0.8), m.FadeIn(vals[1], run_time=0.8)),
            m.AnimationGroup(reach.animate(run_time=0.8).set_value(2), m.GrowFromEdge(bars[2], m.DOWN, run_time=0.8), m.FadeIn(vals[2], run_time=0.8)),
            m.AnimationGroup(reach.animate(run_time=3.4, rate_func=m.linear).set_value(DEPTH), m.Succession(*steps)),
        )
        stop(pulse, value)
        self.say(
            "[By the first layer, less than one millionth of it is left.]",
            m.AnimationGroup(m.FadeIn(million, shift=0.2 * m.DOWN), m.GrowArrow(million_arrow), m.Circumscribe(readout, color=GRAD_C, buff=0.12)),
        )
        self.say(
            "[The early layers barely learn.] [This is the vanishing gradient problem.]",
            m.AnimationGroup(*[n.animate.set_stroke(FAINT) for n in row.neurons[:4]], *[k.animate.set_stroke(opacity=0.35) for k in row.marks[:4]], m.FadeIn(barely)),
            m.AnimationGroup(m.FadeIn(problem, shift=0.2 * m.UP), m.FadeIn(cite)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ ReLU

    def relu(self):
        self.next_section("ReLU", notes="ReLU(x) = max(0, x); slope 1 for x > 0, 0 for x < 0. Nair & Hinton 2010; Glorot, Bordes & Bengio 2011.")
        st = Stage(relu, d_relu, RELU_C, (-3, 3), (-0.5, 3.0), start=0.6)
        plot = st.plot
        leg = legend((RELU_C, "ReLU(x)", 1), (GRAD_C, "its slope", 1)).move_to([PLOT_C[0], 2.2, 0])
        rule = formula((r"\mathrm{ReLU}(x)", RELU_C), ("=", INK), (r"\max(0,\, x)", INK), size=48).move_to([RIGHT_X, 1.6, 0])
        note = text("keep positives, zero the negatives", 22, MUTED).move_to([RIGHT_X, 0.62, 0])
        cite = text("Nair & Hinton, 2010 · Glorot et al., 2011", 20, MUTED).move_to([RIGHT_X, -3.35, 0])
        rule_pill = pill("slope: exactly 1 or 0", GRAD_C, AMBER_FILL, 24).move_to([RIGHT_X, -2.65, 0])

        self.say(
            "[A much simpler curve helped deep networks take off]: [ReLU.]",
            m.AnimationGroup(self.new_header("ReLU", "Keep the positives, zero the rest"), m.Create(plot.frame), m.FadeIn(leg[0])),
            m.AnimationGroup(m.Create(st.curve, run_time=1.2), m.Write(rule), m.FadeIn(cite)),
        )
        self.say(
            "[It passes positive inputs straight through], [and turns negative ones into zero.]",
            m.Succession(
                m.AnimationGroup(m.FadeIn(st.card), m.FadeIn(note), m.FadeIn(st.dot, scale=0.5), m.FadeIn(st.tangent), run_time=0.6),
                st.x.animate(run_time=1.4).set_value(2.4),
            ),
            st.x.animate(run_time=2.0).set_value(-2.4),
        )
        self.say(
            "[So its slope is exactly one for every positive input], [and zero for every negative one.]",
            m.AnimationGroup(m.Create(st.slope[1]), m.FadeIn(leg[1]), m.FadeIn(st.slope_dot, scale=0.5), st.x.animate(run_time=1.6).set_value(1.6)),
            m.AnimationGroup(m.Create(st.slope[0]), st.x.animate(run_time=1.6).set_value(-1.6), m.FadeIn(rule_pill, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        stop(*st.live())
        self.clear()

    # ------------------------------------------------------------------ ReLU in depth, dead ReLUs

    def relu_depth(self):
        self.next_section("ReLU in depth", notes="The same ten layers with ReLU and positive inputs: every slope 1, the gradient arrives whole. One negative input (layer 6) blocks this path.")
        ny, yb, tall = -1.85, -1.3, 2.6
        row = layer_row(ny, "r", RELU_C)
        slots = m.VGroup(*[m.Line([x - 0.3, yb, 0], [x + 0.3, yb, 0]).set_stroke(FAINT, width=2.5) for x in BAR_X])
        bars = [bar(BAR_X[j], yb, tall) for j in range(DEPTH + 1)]
        whole = tex(r"1 \times 1 \times \cdots \times 1 = 1", 40, GRAD_C).move_to([-1.6, 2.0, 0])
        blocked = tex(r"1 \times \cdots \times 0 \times \cdots = 0", 40, GRAD_C).move_to([-1.6, 2.0, 0])
        positive = text("every input positive: every slope 1", 22, MUTED).move_to([3.9, 2.0, 0])
        dead_neuron = row.neurons[DEAD - 1]
        negative = text("input below zero: slope 0", 22, BAD, weight=m.SEMIBOLD).move_to([dead_neuron.get_x(), -2.95, 0])
        cut = range(DEPTH + 1 - DEAD, DEPTH + 1)  # bars of layers DEAD … 1

        self.say(
            "[Back in our ten layers, with positive inputs, every slope is one], [and the whole gradient reaches the first layer.]",
            m.AnimationGroup(
                self.new_header("ReLU IN DEPTH", "A slope of one keeps the gradient whole"),
                show_row(row),
                m.FadeIn(slots),
                m.FadeIn(positive),
                m.GrowFromEdge(bars[0], m.DOWN),
            ),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars[1:]], lag_ratio=0.3), m.Write(whole)),
        )
        self.say(
            "[But a negative input has a slope of zero], [and the gradient along this path stops dead.]",
            m.AnimationGroup(dead_neuron.animate.set_stroke(BAD).set_fill(BAD_FILL), m.FadeIn(negative, shift=0.15 * m.UP), m.FadeOut(positive)),
            m.AnimationGroup(*[m.Transform(bars[j], bar(BAR_X[j], yb, 0.0)) for j in cut], m.FadeTransform(whole, blocked)),
        )
        self.wait(0.3)
        self.clear()
        self.dead()

    def dead(self):
        self.next_section("Dead ReLUs", notes="24 example inputs to one neuron (seeded, illustrative): 23 positive; one big update (−3.3) pushes all below zero, so output, slope and gradient are 0 for every example.")
        lo, hi, left, right, ly = -4.0, 3.0, -6.3, -0.7, -0.75
        unit = (right - left) / (hi - lo)

        def sx(z):
            return left + (z - lo) * unit

        zero = sx(0.0)
        number_line = m.Line([left, ly, 0], [right, ly, 0]).set_stroke(MUTED, width=2.5)
        ticks = m.VGroup(*[m.Line([sx(k), ly - 0.08, 0], [sx(k), ly + 0.08, 0]).set_stroke(MUTED, width=2.5) for k in range(-4, 4)])
        tick_labels = m.VGroup(*[text(f"{k}".replace("-", "−"), 20, MUTED).next_to([sx(k), ly - 0.1, 0], m.DOWN, buff=0.12) for k in (-4, -2, 0, 2)])
        dead_zone = m.Rectangle(width=zero - left, height=2.15, stroke_width=0).set_fill(BAD_FILL, opacity=0.8).move_to([(left + zero) / 2, ly + 1.075, 0])
        live_zone = m.Rectangle(width=right - zero, height=2.15, stroke_width=0).set_fill(RELU_FILL, opacity=0.8).move_to([(zero + right) / 2, ly + 1.075, 0])
        for z in (dead_zone, live_zone):
            z.set_z_index(-2)
        zone_labels = m.VGroup(
            text("ReLU outputs 0", 22, BAD, weight=m.SEMIBOLD).move_to([(left + zero) / 2, ly + 1.82, 0]),
            text("passes through", 22, RELU_C, weight=m.SEMIBOLD).move_to([(zero + right) / 2, ly + 1.82, 0]),
        )
        title = text("one neuron's input, for 24 examples", 24, INK, weight=m.SEMIBOLD).move_to([(left + right) / 2, 1.95, 0])
        jitter = np.random.default_rng(91).uniform(0.3, 1.35, len(Z_BEFORE))
        dots = m.VGroup(*[m.Dot([sx(z), ly + j, 0], radius=0.09, color=RELU_C) for z, j in zip(Z_BEFORE, jitter)])
        push = arrow([sx(1.2), ly - 0.95, 0], [sx(-2.1), ly - 0.95, 0], BAD)
        push_label = text("one big update", 22, BAD, weight=m.SEMIBOLD).next_to(push, m.DOWN, buff=0.12)

        nx, ny = 2.6, 0.45
        neuron = m.Circle(radius=0.6).set_stroke(BAD, width=3.5).set_fill(BAD_FILL, opacity=1).move_to([nx, ny, 0])
        mark = mini_curve("r", FAINT, width=0.6, height=0.42).move_to(neuron)
        out_arrow = arrow(neuron.get_right(), [nx + 2.1, ny, 0], MUTED, buff=0.1)
        out_label = on_baseline(m.VGroup(text("output", 22, MUTED), text("0", 30, BAD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18)).next_to(out_arrow, m.UP, buff=0.14)
        back = m.Arrow([nx + 2.1, ny - 1.25, 0], [nx - 0.6, ny - 1.25, 0], buff=0, color=FAINT, stroke_width=5, max_tip_length_to_length_ratio=0.12)
        c0 = back.get_center() + 0.45 * m.UP
        cross = m.VGroup(
            m.Line(c0 + np.array([-0.17, -0.17, 0]), c0 + np.array([0.17, 0.17, 0])).set_stroke(BAD, width=6),
            m.Line(c0 + np.array([-0.17, 0.17, 0]), c0 + np.array([0.17, -0.17, 0])).set_stroke(BAD, width=6),
        )
        back_label = m.VGroup(text("slope 0", 22, GRAD_C, weight=m.SEMIBOLD), text("no gradient, no update", 22, MUTED)).arrange(m.DOWN, buff=0.1)
        back_label.next_to(back, m.DOWN, buff=0.2)
        verdict = pill("a dead ReLU", BAD, BAD_FILL, 28).move_to([nx + 0.75, -2.75, 0])

        self.say(
            "[Now imagine a big update pushes a neuron's input] [below zero for every example.]",
            m.AnimationGroup(
                self.new_header("DEAD ReLUs", "Stuck at zero"),
                m.FadeIn(dead_zone),
                m.FadeIn(live_zone),
                m.FadeIn(zone_labels),
                m.Create(number_line),
                m.FadeIn(ticks),
                m.FadeIn(tick_labels),
                m.FadeIn(title),
                m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.04),
            ),
            m.AnimationGroup(
                m.GrowArrow(push),
                m.FadeIn(push_label),
                *[d.animate(run_time=1.6).shift(UPDATE * unit * m.RIGHT).set_color(BAD) for d in dots],
            ),
        )
        self.say(
            "[It outputs zero, its slope is zero, so no gradient reaches its weights.] [It may never recover: a dead ReLU.]",
            m.AnimationGroup(
                m.FadeIn(neuron, scale=0.7),
                m.FadeIn(mark),
                m.GrowArrow(out_arrow),
                m.FadeIn(out_label),
                m.GrowArrow(back),
                m.FadeIn(cross, scale=1.3),
                m.FadeIn(back_label),
            ),
            m.Succession(m.FadeIn(verdict, scale=1.2), m.Wiggle(verdict, scale_value=1.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ GELU

    def gelu(self):
        self.next_section(
            "GELU",
            notes="GELU(x) = x·Φ(x), Φ the standard normal CDF (Hendrycks & Gimpel 2016); ≈ ReLU for large |x|; slope 0.5 at 0, 0.13 at −0.5, range −0.13 … 1.13. "
            "Used in GPT (Radford et al. 2018) and BERT (Devlin et al. 2018).",
        )
        st = Stage(gelu, d_gelu, GELU_C, (-3, 3), (-0.5, 3.0), start=0.8, extra_rows=(("Φ(x)", GELU_C, Phi, 3, 2),))
        plot = st.plot
        relu_ghost = plot.graph(relu, RELU_C, width=3.5, opacity=0.4)
        relu_slope_ghost = plot.steps(RELU_C, width=3, opacity=0.4)
        leg = m.VGroup(*legend((RELU_C, "ReLU", 0.4), (GELU_C, "GELU", 1), (GRAD_C, "slope", 1))).move_to([PLOT_C[0], 2.2, 0])
        rule = formula((r"\mathrm{GELU}(x)", GELU_C), ("=", INK), ("x", INK), (r"\cdot", INK), (r"\Phi(x)", GELU_C), size=48).move_to([RIGHT_X, 1.85, 0])
        bell = Plot((-3.5, 3.5), (0, 0.45), 3.6, 1.15, [RIGHT_X, 0.88, 0], label=None, x_step=1, y_step=0.45, small=True)
        bell_curve = bell.graph(phi, MUTED, width=3)

        def shade():
            xv = max(st.xv(), bell.x0 + 1e-3)
            ts = np.linspace(bell.x0, xv, 60)
            points = [bell.p(bell.x0, 0), *[bell.p(t, phi(t)) for t in ts], bell.p(xv, 0)]
            return m.Polygon(*points, stroke_width=0).set_fill(GELU_C, opacity=0.45)

        def marker():
            xv = st.xv()
            return m.Line(bell.p(xv, 0), bell.p(xv, phi(xv))).set_stroke(GELU_C, width=3.5)

        area, mark = m.always_redraw(shade), m.always_redraw(marker)
        bell_label = text("Φ(x): the shaded area", 20, MUTED).move_to([RIGHT_X, 0.1, 0])
        used = pill("inside BERT and GPT-style Transformers", GELU_C, PINK_FILL, 22).move_to([PLOT_C[0], -3.3, 0])
        cite = text("Hendrycks & Gimpel, 2016", 20, MUTED).move_to([RIGHT_X, -3.35, 0])

        self.say(
            "[Many Transformers use a smoother cousin of ReLU]: [GELU, the Gaussian error linear unit.]",
            m.AnimationGroup(self.new_header("GELU", "A smooth ReLU"), m.Create(plot.frame), m.Create(relu_ghost), m.FadeIn(leg[0])),
            m.AnimationGroup(m.Create(st.curve, run_time=1.6), m.FadeIn(leg[1]), m.Write(rule)),
        )
        self.say(
            "[It multiplies its input] [by the chance that a bell-curve random number falls below that input.]",
            m.AnimationGroup(m.Indicate(rule[2], color=INK, scale_factor=1.4), m.FadeIn(st.card), m.FadeIn(st.dot, scale=0.5), m.FadeIn(st.tangent)),
            m.AnimationGroup(
                m.Indicate(rule[4], color=GELU_C, scale_factor=1.2),
                m.Create(bell.frame),
                m.Create(bell_curve),
                m.FadeIn(area),
                m.FadeIn(mark),
                m.FadeIn(bell_label),
            ),
        )
        self.say(
            "[For large positive inputs, that chance is nearly one, so it acts like ReLU.] [For very negative inputs, it's nearly zero.]",
            st.x.animate(run_time=2.2).set_value(2.6),
            st.x.animate(run_time=2.4).set_value(-2.6),
        )
        self.say(
            "[In between, it bends smoothly, so its slope never jumps], [and slightly negative inputs still pass a little gradient.]",
            m.AnimationGroup(m.Create(st.slope, run_time=1.8), m.FadeIn(relu_slope_ghost), m.FadeIn(leg[2]), m.FadeIn(st.slope_dot, scale=0.5), st.x.animate(run_time=1.8).set_value(0.0)),
            m.AnimationGroup(st.x.animate(run_time=1.2).set_value(-0.5), m.Indicate(st.numbers[-1], color=GRAD_C, scale_factor=1.25)),
        )
        self.say(
            "[You'll find it inside BERT and GPT-style Transformers.]",
            m.AnimationGroup(m.FadeIn(used, shift=0.2 * m.UP), m.FadeIn(cite)),
        )
        self.wait(0.3)
        stop(*st.live(), area, mark)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section("The big picture", notes="All four with their slopes; slope ranges computed. Callback: ten sigmoid layers 0.00000095, ten ReLU layers (positive inputs) 1.")
        specs = (
            ("sigmoid", SIG_C, sigmoid, d_sigmoid, (-6, 6), (-0.5, 1.5), 2, "slope 0 to 0.25", "saturates", BAD, ("probabilities", "and gates")),
            ("tanh", TANH_C, np.tanh, d_tanh, (-4, 4), (-1.5, 1.5), 1, "slope 0 to 1", "saturates", BAD, ("classic recurrent", "networks")),
            ("ReLU", RELU_C, relu, d_relu, (-3, 3), (-0.5, 2.5), 1, "slope 0 or 1", "can die", AMBER, ("the simple", "default")),
            ("GELU", GELU_C, gelu, d_gelu, (-3, 3), (-0.5, 2.5), 1, f"slope {GELU_LO:.2f} to {GELU_HI:.2f}".replace("-", "−"), "smooth", GOOD, ("Transformers:", "BERT, GPT")),
        )
        cards, verdicts, uses = [], [], []
        for k, (name, color, fn, dfn, xr, yr, xs, slope_txt, verdict, vcolor, use) in enumerate(specs):
            x = -4.95 + 3.3 * k
            box = card(3.05, 4.55, color, CARD, stroke=2).move_to([x, -0.3, 0])
            top = box.get_top()[1]
            title = text(name, 28, color, weight=m.BOLD).move_to([x, top - 0.42, 0])
            plot = Plot(xr, yr, 2.45, 1.6, [x, top - 1.65, 0], label=None, x_step=xs, y_step=0.5, small=True)
            curve = plot.graph(fn, color, width=3.5)
            slope = plot.graph(dfn, GRAD_C, width=2.5) if dfn is not d_relu else plot.steps(GRAD_C, width=2.5)
            slope_line = text(slope_txt, 22, GRAD_C, weight=m.SEMIBOLD).move_to([x, top - 2.82, 0])
            cards.append(m.VGroup(box, title, plot.frame, curve, slope, slope_line))
            verdicts.append(text(verdict, 24, vcolor, weight=m.BOLD).move_to([x, top - 3.35, 0]))
            uses.append(m.VGroup(*[text(s, 20, MUTED) for s in use]).arrange(m.DOWN, buff=0.08).move_to([x, top - 3.95, 0]))
        base = cards[0][1][0].get_bottom()[1]
        for c in cards[1:]:
            c[1].shift((base - c[1][0].get_bottom()[1]) * m.UP)
        sig_pill = pill("10 sigmoid layers: 0.00000095 left", BAD, BAD_FILL, 22)
        relu_pill = pill("10 ReLU layers, inputs > 0: all of it", GOOD, GOOD_FILL, 22)
        m.VGroup(sig_pill, relu_pill).arrange(m.RIGHT, buff=0.5).move_to([0, -3.3, 0])

        self.say(
            "[Here are all four, side by side, each with its slope.]",
            m.AnimationGroup(self.new_header("THE BIG PICTURE", "Four curves, four slopes"), m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in cards], lag_ratio=0.25)),
        )
        self.say(
            "[Sigmoid and the hyperbolic tangent saturate, so deep stacks of them fade.] [They live on where outputs must be squashed, like probabilities and gates.]",
            m.AnimationGroup(*[m.FadeIn(v, shift=0.15 * m.UP) for v in verdicts[:2]]),
            m.AnimationGroup(*[m.FadeIn(u, shift=0.15 * m.UP) for u in uses[:2]]),
        )
        self.say(
            "[ReLU is the fast, simple default], [and its smooth cousin, GELU, runs inside Transformers.]",
            m.AnimationGroup(m.FadeIn(verdicts[2], shift=0.15 * m.UP), m.FadeIn(uses[2], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(verdicts[3], shift=0.15 * m.UP), m.FadeIn(uses[3], shift=0.15 * m.UP)),
        )
        self.say(
            "[In our ten layers, sigmoid passed on less than a millionth of the gradient], [and ReLU, with positive inputs, passed on all of it.]",
            m.AnimationGroup(m.FadeIn(sig_pill, shift=0.2 * m.UP), m.Indicate(cards[0][0], color=SIG_C, scale_factor=1.03)),
            m.AnimationGroup(m.FadeIn(relu_pill, shift=0.2 * m.UP), m.Indicate(cards[2][0], color=RELU_C, scale_factor=1.03)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Shape, slope, signal.")
        halo = glow(VIOLET, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Shape", "Slope", "Signal"), (VIOLET, GRAD_C, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("The shape sets the slope; the slopes decide if the gradient survives.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [shape], [slope], [signal.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[The shape of each bend sets its slope, and the slopes decide whether the signal survives the trip back.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
