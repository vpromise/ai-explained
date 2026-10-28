"""AI, Explained · Episode 25 · Residuals & Normalization

Running example: a 100-layer tower. Why a gradient fades (or explodes) through a hundred plain
layers; the skip connection, y = x + f(x); its derivative 1 + f'(x), the gradient highway; a
measured experiment (seeded numpy) with a hundred random layers, plain versus residual;
LayerNorm on one token's four numbers; post-norm versus pre-norm, and RMSNorm; the modern
Transformer block, stacked a hundred times.

The experiment is a genuine simulation, computed below: width 64, 100 layers, each layer
tanh(W x) with W ~ N(0, 0.9²/64), seed 25; the same weights for both towers. The gradient is a
random unit vector at the top, sent down through the layers' Jacobians; "size of x" is the
root mean square of the signal, relative to the input's. The narration's numbers are asserted.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

GRAD = AMBER  # the gradient, wherever it flows

# ---------------------------------------------------------------------------- the experiment

WIDTH, DEPTH, SCALE, SEED = 64, 100, 0.9, 25


def towers():
    """Two towers of the same 100 random layers, plain (x <- f(x)) and residual (x <- x + f(x)).
    Returns {kind: (size of x at layers 0..100, relative to the input; gradient size at layers
    0..100, the top's = 1)}."""
    rng = np.random.default_rng(SEED)
    weights = [rng.normal(0, SCALE / np.sqrt(WIDTH), (WIDTH, WIDTH)) for _ in range(DEPTH)]
    x0 = rng.normal(0, 1, WIDTH)
    g_top = rng.normal(0, 1, WIDTH)
    g_top /= np.linalg.norm(g_top)
    out = {}
    for kind in ("plain", "residual"):
        x, xs, jacobians = x0, [x0], []
        for w in weights:
            a = np.tanh(w @ x)
            jac = (1 - a**2)[:, None] * w
            if kind == "residual":
                x, jac = x + a, np.eye(WIDTH) + jac
            else:
                x = a
            xs.append(x)
            jacobians.append(jac)
        size = np.array([np.sqrt(np.mean(v**2)) for v in xs]) / np.sqrt(np.mean(x0**2))
        g, grads = g_top, [1.0]
        for jac in reversed(jacobians):
            g = jac.T @ g
            grads.append(float(np.linalg.norm(g)))
        out[kind] = (size, np.array(grads[::-1]))
    return out


TOWERS = towers()
PLAIN_SIZE, PLAIN_GRAD = TOWERS["plain"]
RES_SIZE, RES_GRAD = TOWERS["residual"]

# what the narration says about them
assert 7.5e-6 <= PLAIN_GRAD[0] < 8.5e-6  # "about eight millionths of it reach the bottom"
assert RES_GRAD.min() >= 0.999  # "with skip connections, it never fades"
assert RES_GRAD[0] > 10  # "it even grows"
assert PLAIN_SIZE[-1] < 1e-3  # "in the plain tower, it dies out too"
assert 8.5 <= RES_SIZE[-1] < 9.5  # "about nine times bigger by the top"

# the thought experiment: a factor per layer, a hundred times
SHRINK, GROW = 0.9**DEPTH, 1.1**DEPTH
assert 35_000 < 1 / SHRINK < 40_000  # "shrunk almost forty thousand times"
assert 13_000 < GROW < 14_000  # "almost fourteen thousand times its size"

# LayerNorm on one token's four numbers (illustrative)
TOKEN = np.array([11.0, -3.0, 5.0, 3.0])
MU, SD = float(TOKEN.mean()), float(TOKEN.std())
assert (MU, SD) == (4.0, 5.0)  # "subtract their mean, which is four" · "standard deviation, which is five"
CENTRED = TOKEN - MU
NORMED = CENTRED / SD
assert abs(NORMED.mean()) < 1e-12 and abs(NORMED.std() - 1) < 1e-12  # "a mean of zero, and a spread of one"
GAMMA = np.array([1.0, 0.5, 2.0, 1.0])
BETA = np.array([0.0, 0.0, 0.1, 0.5])
OUT = GAMMA * NORMED + BETA


# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def mix(a, b, t):
    """A colour between two hex colours."""
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


SLAB = mix(VIOLET_FILL, VIOLET, 0.32)  # an unlit layer


def num(v):
    """A number for a chip: no trailing zeros, a real minus sign."""
    s = f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else f"{int(round(v))}"
    return s.replace("-", "−")


def tag(label, color, fill, size=22):
    """A pill whose height and baseline don't depend on its letters (so tags in a row line up)."""
    words = text(label, size, color, weight=m.BOLD)
    height = text("Hgy", size, weight=m.BOLD).height + 0.3
    box = card(words.width + 0.55, height, color, fill, radius=height / 2, stroke=2)
    words.move_to(box)
    sitting = [g for g, c in zip(words, label.replace(" ", "")) if c not in "gjpqy,"] or list(words)
    baseline = box.get_center()[1] - 0.449 * size / 48 / 2 * 1.45
    words.shift((baseline - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    return m.VGroup(box, words)


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners([np.array(p, dtype=float) for p in points]).set_stroke(color, width=width, opacity=opacity)


def formula(*parts, size=44):
    """A MathTex from (latex, colour) pairs, one part each."""
    t = m.MathTex(*[p for p, _ in parts], font_size=size, color=INK)
    for part, (_, c) in zip(t, parts):
        part.set_color(c)
    return t


def tower(cx, bottom, top, width, n=DEPTH, color=SLAB):
    """n thin slabs, bottom to top: tower[0] is layer 1."""
    pitch = (top - bottom) / n
    return m.VGroup(
        *[
            m.Rectangle(width=width, height=pitch * 0.66, stroke_width=0).set_fill(color, opacity=1).move_to([cx, bottom + pitch * (i + 0.5), 0])
            for i in range(n)
        ]
    )


def plus_node(center, color, r=0.26):
    center = np.array(center, dtype=float)
    ring = m.Circle(radius=r).set_stroke(color, width=3).set_fill(BG, opacity=1).move_to(center)
    a = 0.55 * r
    bar = m.Line(center + [-a, 0, 0], center + [a, 0, 0]).set_stroke(color, width=3)
    post = m.Line(center + [0, -a, 0], center + [0, a, 0]).set_stroke(color, width=3)
    return m.VGroup(ring, bar, post)


def block(label, color, fill, w, h, size=24):
    box = card(w, h, color, fill, radius=0.14, stroke=2.5)
    return m.VGroup(box, text(label, size, INK, weight=m.SEMIBOLD).move_to(box))


def tip_line(points, color, width=4, tip=0.24):
    """A path through `points` with an arrowhead ending exactly at the last one."""
    pts = [np.array(p, dtype=float) for p in points]
    along = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
    side = np.array([-along[1], along[0], 0])
    base = pts[-1] - tip * along
    path = polyline([*pts[:-1], base + 0.02 * along], color, width)
    head = m.Polygon(pts[-1], base + 0.55 * tip * side, base - 0.55 * tip * side, stroke_width=0, fill_color=color, fill_opacity=1)
    return m.VGroup(path, head)


def flash(path, color, width=9, time_width=0.6, **kw):
    return m.ShowPassingFlash(path.copy().set_stroke(color, width=width, opacity=1), time_width=time_width, **kw)


class VPlot:
    """A plot standing beside the tower: layers 0..100 run up, values run right."""

    def __init__(self, x0, x1, y0, y1, vmin, vmax):
        self.x0, self.x1, self.y0, self.y1, self.vmin, self.vmax = x0, x1, y0, y1, vmin, vmax
        self.frame = m.VGroup(
            m.Line([x0, y0, 0], [x1 + 0.15, y0, 0]).set_stroke(FAINT, width=2.5),
            m.Line([x0, y0, 0], [x0, y1 + 0.15, 0]).set_stroke(FAINT, width=2.5),
        )

    def p(self, v, layer):
        return np.array([self.x0 + (v - self.vmin) / (self.vmax - self.vmin) * (self.x1 - self.x0), self.y0 + layer / DEPTH * (self.y1 - self.y0), 0])

    def ticks(self, values, labels):
        group = m.VGroup()
        for v, lab in zip(values, labels):
            x = self.p(v, 0)[0]
            group.add(m.Line([x, self.y0, 0], [x, self.y0 - 0.1, 0]).set_stroke(FAINT, width=2.5))
            group.add(lab.next_to([x, self.y0 - 0.1, 0], m.DOWN, buff=0.12))
        return group

    def curve(self, values, color, downward=False, width=5):
        layers = range(DEPTH, -1, -1) if downward else range(DEPTH + 1)
        return polyline([self.p(float(np.clip(values[k], self.vmin, self.vmax)), k) for k in layers], color, width)


# ---------------------------------------------------------------------------- the episode


class Ep25ResidualsNormalization(Episode):
    number = 25

    def lesson(self):
        self.hook()
        self.problem()
        self.skip()
        self.highway()
        self.experiment()
        self.layernorm()
        self.placement()
        self.big_picture()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A 100-layer tower. Plain deep nets degrade: He et al. 2015, Fig. 1 (CIFAR-10, a plain 56-layer net has higher training error than a 20-layer one).")
        slabs = tower(-3.4, -3.15, 1.95, 1.7)
        self.tower = slabs
        brace = m.Brace(slabs, m.LEFT, buff=0.15).set_color(MUTED)
        count = text("100", 26, INK, weight=m.BOLD).next_to(brace, m.LEFT, buff=0.12)
        hero = m.VGroup(text("100 layers", 66, INK, weight=m.BOLD), text("stacked one on another", 26, MUTED)).arrange(m.DOWN, buff=0.2)
        hero.move_to([2.6, 1.15, 0])
        deeper = text("more layers → richer features → a better model?", 26, INK, weight=m.SEMIBOLD).move_to([2.6, -0.25, 0])
        result = m.VGroup(
            text("A plain 56-layer network trained worse", 24, INK),
            text("than a plain 20-layer one.", 24, INK),
            text("He et al., 2015", 20, MUTED),
        ).arrange(m.DOWN, buff=0.12)
        result_box = card(result.width + 0.8, result.height + 0.55, BAD, BAD_FILL, stroke=2).move_to([2.6, -1.55, 0])
        result.move_to(result_box)
        fixes = m.VGroup(tag("skip connections", SKY, SKY_FILL, 26), tag("normalization", AMBER, AMBER_FILL, 26)).arrange(m.RIGHT, buff=0.5)
        fixes.move_to([2.6, -3.0, 0])
        self.hook_items = m.VGroup(brace, count, hero, deeper, result_box, result, fixes)

        self.say(
            "So... [what happens when you stack a hundred layers]?",
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.DOWN) for s in slabs], lag_ratio=0.03, run_time=2.6),
                m.FadeIn(hero, shift=0.2 * m.UP),
                m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(count), lag_ratio=0.6),
            ),
        )
        self.say(
            "[Deeper networks can learn richer features], [so more layers should mean a better model.]",
            m.FadeIn(deeper, shift=0.2 * m.UP),
            m.LaggedStart(*[s.animate(rate_func=m.there_and_back).set_fill(VIOLET) for s in slabs], lag_ratio=0.01, run_time=1.6),
        )
        self.say(
            "[But for years, very deep networks were hard to train.] [Adding layers could even make them worse.]",
            m.FadeIn(result_box, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(result, shift=0.1 * m.UP), m.Wiggle(result_box, scale_value=1.04)),
        )
        self.say(
            "[Two simple ideas fixed that]: [skip connections], [and normalization.] Let's see how.",
            m.Indicate(slabs, color=VIOLET, scale_factor=1.0),
            m.FadeIn(fixes[0], shift=0.2 * m.UP),
            m.FadeIn(fixes[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the problem

    def problem(self):
        self.next_section("The problem", notes="Backprop multiplies the gradient by one factor per layer (episode 12). 0.9^100 = 1/37,649; 1.1^100 = 13,781. Exact arithmetic, a thought experiment.")
        slabs = self.tower
        top = slabs[-1].get_top()
        loss = pill("loss", BAD, BAD_FILL, 22).next_to(slabs, m.UP, buff=0.18)
        down = arrow(slabs.get_corner(m.UR) + [0.45, 0, 0], slabs.get_corner(m.DR) + [0.45, 0, 0], GRAD, buff=0)
        down_label = text("gradient", 22, GRAD, weight=m.SEMIBOLD).next_to(down, m.RIGHT, buff=0.15).shift(1.4 * m.UP)
        tops = text("layer 100", 20, MUTED).next_to(slabs[-1], m.LEFT, buff=0.25)
        bottoms = text("layer 1", 20, MUTED).next_to(slabs[0], m.LEFT, buff=0.25)
        marks_at = (86, 64, 42, 20)
        marks = m.VGroup(*[text("× ?", 24, MUTED, weight=m.BOLD).next_to(slabs[i], m.LEFT, buff=0.25) for i in marks_at])
        nines = m.VGroup(*[text("× 0.9", 24, BAD, weight=m.BOLD).move_to(mk) for mk in marks])

        k = m.ValueTracker(0.0)
        x_col = 3.0
        case1 = pill("× 0.9 at every layer", BAD, BAD_FILL, 24).move_to([x_col, 1.65, 0])
        left1 = m.DecimalNumber(1.0, num_decimal_places=5, mob_class=Mono, font_size=40, color=BAD)
        left1.add_updater(lambda d: d.set_value(0.9 ** k.get_value()))
        row1 = m.VGroup(text("gradient left:", 26, MUTED, weight=m.SEMIBOLD), left1).arrange(m.RIGHT, buff=0.3).move_to([x_col, 0.9, 0])
        cap1 = text(f"after 100 layers: {1 / SHRINK:,.0f}× smaller", 22, MUTED).move_to([x_col, 0.3, 0])

        j = m.ValueTracker(0.0)
        case2 = pill("× 1.1 at every layer", AMBER, AMBER_FILL, 24).move_to([x_col, -0.4, 0])
        left2 = m.DecimalNumber(1.0, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=40, color=AMBER)
        left2.add_updater(lambda d: d.set_value(1.1 ** j.get_value()))
        row2 = m.VGroup(text("gradient:", 26, MUTED, weight=m.SEMIBOLD), left2).arrange(m.RIGHT, buff=0.3).move_to([x_col, -1.15, 0])
        cap2 = text(f"after 100 layers: {GROW:,.0f}× bigger", 22, MUTED).move_to([x_col, -1.75, 0])
        low = m.Brace(slabs[:30], m.RIGHT, buff=0.15).set_color(BAD).next_to(down, m.RIGHT, buff=0.15).align_to(slabs[0], m.DOWN)
        whisper = text("the bottom layers hear almost nothing", 24, INK, weight=m.SEMIBOLD).next_to(low, m.RIGHT, buff=0.2)

        def lit(depth):  # brightness on a log scale: 1 at the top, ~0 at a hundred-thousandth
            return mix(SLAB, GRAD, 1 + np.log10(0.9**depth) / 5)

        self.say(
            "[Training sends a gradient backwards], [from the loss at the top, down through every layer.]",
            m.AnimationGroup(
                self.new_header("THE PROBLEM", "A signal fading through a hundred layers"),
                m.FadeOut(self.hook_items),
                slabs.animate.set_fill(SLAB),
                m.FadeIn(tops),
                m.FadeIn(bottoms),
                m.FadeIn(loss, shift=0.2 * m.DOWN),
            ),
            m.AnimationGroup(m.GrowArrow(down), m.FadeIn(down_label)),
        )
        self.say("[On the way, each layer multiplies it by some factor.]", m.LaggedStart(*[m.FadeIn(mk, shift=0.15 * m.RIGHT) for mk in marks], lag_ratio=0.25))
        self.say(
            "[Say that factor is zero point nine.] [After a hundred layers, the gradient has shrunk almost forty thousand times.]",
            m.AnimationGroup(m.FadeIn(case1, shift=0.2 * m.UP), m.FadeIn(row1), *[m.FadeTransform(a, b) for a, b in zip(marks, nines)]),
            m.AnimationGroup(
                k.animate(run_time=4.0, rate_func=m.linear).set_value(DEPTH),
                m.LaggedStart(*[slabs[DEPTH - 1 - d].animate.set_fill(lit(d)) for d in range(DEPTH)], lag_ratio=0.03, run_time=4.0),
                m.Succession(m.Wait(3.6), m.FadeIn(cap1, shift=0.15 * m.UP, run_time=0.6)),
            ),
        )
        self.say(
            "[If it's one point one instead], [it explodes, to almost fourteen thousand times its size.]",
            m.AnimationGroup(m.FadeIn(case2, shift=0.2 * m.UP), m.FadeIn(row2)),
            m.AnimationGroup(j.animate(run_time=3.0, rate_func=m.linear).set_value(DEPTH), m.Succession(m.Wait(2.6), m.FadeIn(cap2, shift=0.15 * m.UP, run_time=0.6))),
        )
        self.stop(left1, left2)
        self.say(
            "[It's like a whisper passed down a line of a hundred people.] [The bottom layers hear almost nothing], so they barely learn.",
            flash(m.Line(down.get_start(), down.get_end()), GRAD, time_width=0.5, run_time=2.0),
            m.AnimationGroup(m.GrowFromCenter(low), m.FadeIn(whisper, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the skip connection

    def skip(self):
        self.next_section("Skip connections", notes="y = x + f(x). ResNet: He, Zhang, Ren & Sun, Deep Residual Learning for Image Recognition (2015, CVPR 2016), 152 layers on ImageNet.")
        y = 1.0
        x_box = card(1.1, 0.85, SKY, SKY_FILL, radius=0.16).move_to([-5.6, y, 0])
        x_lab = tex("x", 46, SKY).move_to(x_box)
        f_box = card(2.5, 1.25, VIOLET, VIOLET_FILL, radius=0.18).move_to([-1.1, y, 0])
        f_lab = m.VGroup(tex("f", 50, VIOLET), text("a layer", 22, MUTED)).arrange(m.DOWN, buff=0.08).move_to(f_box)
        plus = plus_node([2.35, y, 0], GOOD)
        out_box = card(2.4, 0.85, GOOD, GOOD_FILL, radius=0.16).move_to([5.1, y, 0])
        out_lab = tex("x + f(x)", 40, INK).move_to(out_box)
        a1 = arrow(x_box.get_right(), f_box.get_left(), MUTED, buff=0.1)
        a2 = arrow(f_box.get_right(), plus.get_left(), MUTED, buff=0.08)
        a3 = arrow(plus.get_right(), out_box.get_left(), MUTED, buff=0.08)
        fork = np.array([-4.2, y, 0])
        skip_pts = [fork, [-4.2, 2.3, 0], [2.35, 2.3, 0], [2.35, y + 0.27, 0]]
        skip_path = tip_line(skip_pts, SKY, width=5)
        skip_dot = m.Dot(fork, radius=0.08, color=SKY)
        skip_label = text("skip connection", 24, SKY, weight=m.SEMIBOLD).move_to([-1.1, 2.0, 0])
        self.diagram = m.VGroup(x_box, x_lab, f_box, f_lab, plus, out_box, out_lab, a1, a2, a3)
        self.skip_path, self.skip_pts, self.skip_dot, self.skip_label = skip_path, skip_pts, skip_dot, skip_label
        self.f_box = f_box

        eq = formula(("y", GOOD), ("=", INK), ("x", SKY), ("+", INK), ("f(x)", VIOLET), size=56).move_to([0, -0.75, 0])
        self.eq = eq
        say_x = text("the input, passed straight through", 22, SKY)
        say_x.move_to([0, -1.75, 0]).align_to([eq[2].get_right()[0] + 0.2, 0, 0], m.RIGHT)
        say_f = text("the change the layer computes", 22, VIOLET)
        say_f.move_to([0, -1.75, 0]).align_to([eq[4].get_left()[0] - 0.05, 0, 0], m.LEFT)
        under = eq.get_bottom()[1] - 0.08
        link_x = m.Line([eq[2].get_x(), under, 0], [eq[2].get_x(), say_x.get_top()[1] + 0.08, 0]).set_stroke(SKY, width=2.5)
        link_f = m.Line([eq[4].get_x(), under, 0], [eq[4].get_x(), say_f.get_top()[1] + 0.08, 0]).set_stroke(VIOLET, width=2.5)
        self.term_notes = m.VGroup(say_x, say_f, link_x, link_f)
        residual = pill("it learns only a small correction: the residual", VIOLET, VIOLET_FILL, 22).move_to([0, -3.0, 0])
        resnet = m.VGroup(pill("ResNet: 152 layers deep", GOOD, GOOD_FILL, 24), text("He et al., 2015", 20, MUTED)).arrange(m.RIGHT, buff=0.35).move_to([0, -3.0, 0])
        self.resnet = resnet

        self.say(
            "[The fix is surprisingly simple.] [Instead of replacing its input, each layer adds to it.]",
            m.AnimationGroup(self.new_header("THE FIX", "Skip connections: add, don't replace"), m.FadeIn(x_box), m.FadeIn(x_lab), m.GrowArrow(a1), m.FadeIn(f_box), m.FadeIn(f_lab)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(plus, scale=0.6), m.GrowArrow(a3), m.FadeIn(out_box), m.FadeIn(out_lab)),
        )
        self.say(
            "[The input takes a shortcut around the layer]: [that's the skip connection.]",
            m.AnimationGroup(m.FadeIn(skip_dot, scale=0.5), m.Create(skip_path, run_time=1.4)),
            m.AnimationGroup(m.FadeIn(skip_label, shift=0.15 * m.DOWN), flash(skip_path[0], SKY, run_time=1.2)),
        )
        self.say(
            "[So the output is x plus f of x]: [the input, passed straight through], [plus the change the layer computes.]",
            m.Write(eq),
            m.AnimationGroup(m.Create(link_x), m.FadeIn(say_x, shift=0.15 * m.DOWN), m.Indicate(eq[2], color=SKY, scale_factor=1.3)),
            m.AnimationGroup(m.Create(link_f), m.FadeIn(say_f, shift=0.15 * m.DOWN), m.Indicate(eq[4], color=VIOLET, scale_factor=1.2)),
        )
        self.say(
            "[The layer only has to learn a small correction.] [If it has nothing to add, it can output almost zero], and pass x along.",
            m.FadeIn(residual, shift=0.2 * m.UP),
            m.AnimationGroup(
                m.Succession(f_box.animate(run_time=0.8).set_fill(BG), m.Wait(0.6), f_box.animate(run_time=0.8).set_fill(VIOLET_FILL)),
                flash(skip_path[0], SKY, run_time=1.4),
            ),
        )
        self.say(
            "[In twenty fifteen, this trick let researchers train a network a hundred and fifty-two layers deep.]",
            m.FadeTransform(residual, resnet),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the gradient highway

    def highway(self):
        self.next_section("Gradient highway", notes="d/dx (x + f(x)) = 1 + f'(x): the skip passes the gradient through unchanged (in many dimensions: I + J_f).")
        deriv = formula((r"\frac{dy}{dx}", GOOD), ("=", INK), ("1", SKY), ("+", INK), ("f'(x)", VIOLET), size=56).move_to([0, -2.15, 0])
        back_skip = polyline(list(reversed(self.skip_pts)), GRAD, width=8)
        back_main = polyline([[2.35, 1.0, 0], [-4.2, 1.0, 0]], GRAD, width=4)

        slabs = tower(-2.6, -3.0, 2.0, 1.7)
        lane_x = slabs.get_right()[0] + 0.55
        lane = m.Line([lane_x, slabs.get_top()[1], 0], [lane_x, slabs.get_bottom()[1], 0]).set_stroke(GOOD, width=6)
        ties = m.VGroup(*[m.Line([slabs.get_right()[0] + 0.05, slabs[i].get_y(), 0], [lane_x, slabs[i].get_y(), 0]).set_stroke(GOOD, width=2, opacity=0.7) for i in range(4, DEPTH, 10)])
        lane_label = m.VGroup(
            text("skip connections line up", 26, GOOD, weight=m.SEMIBOLD),
            text("into a highway for the gradient", 26, GOOD, weight=m.SEMIBOLD),
        ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([2.9, 0.6, 0])
        arrives = text("it reaches layer 1 at full strength", 24, INK).next_to(lane_label, m.DOWN, buff=0.6).align_to(lane_label, m.LEFT)
        tops = text("layer 100", 20, MUTED).next_to(slabs[-1], m.LEFT, buff=0.25)
        bottoms = text("layer 1", 20, MUTED).next_to(slabs[0], m.LEFT, buff=0.25)

        self.say(
            "[Now follow the gradient.] [The derivative of x plus f of x is one, plus f prime.]",
            m.AnimationGroup(self.new_header("GRADIENT FLOW", "The skip is a highway"), m.FadeOut(self.term_notes), m.FadeOut(self.resnet)),
            m.Write(deriv),
        )
        self.say(
            "[That one comes from the skip]: [the gradient passes straight through, untouched.]",
            m.AnimationGroup(m.Circumscribe(deriv[2], color=SKY, buff=0.12), m.Indicate(deriv[2], color=SKY, scale_factor=1.4), m.Indicate(self.eq[2], color=SKY, scale_factor=1.3)),
            m.ShowPassingFlash(back_skip, time_width=0.5, run_time=1.6),
        )
        self.say(
            "[The layer adds its own part on top], [so even when that part is tiny, nothing fades.]",
            m.AnimationGroup(m.Indicate(deriv[4], color=VIOLET, scale_factor=1.2), m.ShowPassingFlash(back_main.set_stroke(opacity=0.5), time_width=0.5, run_time=1.4)),
            m.AnimationGroup(m.ShowPassingFlash(polyline(list(reversed(self.skip_pts)), GRAD, width=8), time_width=0.5, run_time=1.4), m.Indicate(deriv[2], color=SKY, scale_factor=1.4)),
        )
        everything = [mob for mob in self.mobjects[1:] if mob is not self.head]
        self.say(
            "[Stack a hundred blocks, and the skips line up into a highway], [carrying the gradient all the way down.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in everything],
                m.LaggedStart(*[m.FadeIn(s) for s in slabs], lag_ratio=0.01, run_time=1.2),
                m.Create(lane),
                m.Create(ties),
                m.FadeIn(tops),
                m.FadeIn(bottoms),
                m.FadeIn(lane_label, shift=0.2 * m.LEFT),
            ),
            m.AnimationGroup(
                m.ShowPassingFlash(lane.copy().set_stroke(GRAD, width=12), time_width=0.4, run_time=2.0),
                m.LaggedStart(*[slabs[DEPTH - 1 - d].animate.set_fill(mix(SLAB, GRAD, 0.85)) for d in range(DEPTH)], lag_ratio=0.02, run_time=2.0),
                m.Succession(m.Wait(1.4), m.FadeIn(arrives, shift=0.15 * m.UP, run_time=0.6)),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the experiment

    def experiment(self):
        self.next_section(
            "The experiment",
            notes="Illustrative simulation (seeded numpy): width 64, 100 layers of tanh(Wx), W ~ N(0, 0.81/64), seed 25, the same weights in both towers. "
            f"Plain: gradient at layer 1 = {PLAIN_GRAD[0]:.2e}, size of x at the top = {PLAIN_SIZE[-1]:.1e}. "
            f"Residual: gradient at layer 1 = {RES_GRAD[0]:.1f} (never below 1), size of x at the top = {RES_SIZE[-1]:.2f}.",
        )
        y0, y1 = -2.55, 1.75
        slabs = tower(-6.1, y0, y1, 0.5)
        tops = text("layer 100", 20, MUTED).next_to(slabs, m.UP, buff=0.15)
        bottoms = text("layer 1", 20, MUTED).next_to(slabs, m.DOWN, buff=0.15)

        grad = VPlot(-4.95, -0.95, y0, y1, -6, 2)
        size = VPlot(0.9, 4.9, y0, y1, 0, 10)
        g_ticks = grad.ticks([-6, -4, -2, 0, 2], [tex(s, 28, MUTED) for s in ("10^{-6}", "10^{-4}", "10^{-2}", "1", "10^{2}")])
        s_ticks = size.ticks([0, 5, 10], [text(s, 22, MUTED) for s in ("0", "5", "10")])
        g_ref = m.DashedLine(grad.p(0, 0), grad.p(0, DEPTH), dash_length=0.08).set_stroke(FAINT, width=2)
        s_ref = m.DashedLine(size.p(1, 0), size.p(1, DEPTH), dash_length=0.08).set_stroke(FAINT, width=2)

        def title(words, direction, x):
            label = text(words, 24, INK, weight=m.SEMIBOLD)
            head = m.Arrow(m.ORIGIN, 0.5 * direction, buff=0, color=GRAD if direction[1] < 0 else SKY, stroke_width=5, max_tip_length_to_length_ratio=0.45)
            return m.VGroup(label, head).arrange(m.RIGHT, buff=0.2).move_to([x, 2.3, 0])

        g_title = title("gradient, flowing down", m.DOWN, (grad.x0 + grad.x1) / 2)
        s_title = title("size of x, flowing up", m.UP, (size.x0 + size.x1) / 2)
        legend = m.VGroup(
            m.VGroup(m.Line(m.ORIGIN, 0.5 * m.RIGHT).set_stroke(BAD, width=5), text("plain tower", 22, BAD, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Line(m.ORIGIN, 0.5 * m.RIGHT).set_stroke(GOOD, width=5), text("with skip connections", 22, GOOD, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.6).move_to([0.4, -3.5, 0])

        g_plain = grad.curve(np.log10(PLAIN_GRAD), BAD, downward=True)
        g_res = grad.curve(np.log10(RES_GRAD), GOOD, downward=True)
        s_plain = size.curve(PLAIN_SIZE, BAD)
        s_res = size.curve(RES_SIZE, GOOD)
        end_plain = code(f"{PLAIN_GRAD[0]:.6f}", 24, BAD).move_to(grad.p(-4.15, 8), aligned_edge=m.LEFT)
        end_res = code(f"×{RES_GRAD[0]:.0f}", 24, GOOD).next_to(grad.p(np.log10(RES_GRAD[0]), 8), m.RIGHT, buff=0.2)
        top_res = code(f"×{RES_SIZE[-1]:.1f}", 24, GOOD).next_to(size.p(RES_SIZE[-1], DEPTH), m.RIGHT, buff=0.18)
        dead = text("dies out", 22, BAD, weight=m.SEMIBOLD).move_to(size.p(2.4, 88), aligned_edge=m.LEFT)

        self.say(
            "[Let's measure it.] [We build a tower of a hundred random layers, for example], [once plain, and once with skip connections.]",
            self.new_header("THE EXPERIMENT", "A hundred random layers, measured"),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(s) for s in slabs], lag_ratio=0.01, run_time=1.2),
                m.FadeIn(tops),
                m.FadeIn(bottoms),
                m.Create(grad.frame),
                m.Create(size.frame),
                m.FadeIn(g_ticks),
                m.FadeIn(s_ticks),
                m.Create(g_ref),
                m.Create(s_ref),
                m.FadeIn(g_title),
                m.FadeIn(s_title),
            ),
            m.FadeIn(legend, shift=0.15 * m.UP),
        )
        self.say(
            "[We send a gradient down from the top.] [In the plain tower, it fades away]: [about eight millionths of it reach the bottom.]",
            m.Indicate(g_title, color=GRAD, scale_factor=1.06),
            m.Create(g_plain, run_time=2.4),
            m.FadeIn(end_plain, shift=0.15 * m.UP),
        )
        self.say(
            "[With skip connections, it never fades.] [It even grows, as each layer adds its own part.]",
            m.Create(g_res, run_time=2.4),
            m.FadeIn(end_res, shift=0.15 * m.LEFT),
        )
        self.say(
            "[Now the signal going up.] [In the plain tower, it dies out too.]",
            m.Indicate(s_title, color=SKY, scale_factor=1.06),
            m.AnimationGroup(m.Create(s_plain, run_time=2.0), m.FadeIn(dead, shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[With skips, it survives], [but it keeps growing: about nine times bigger by the top.]",
            m.Create(s_res, run_time=2.4),
            m.AnimationGroup(m.FadeIn(top_res, shift=0.15 * m.LEFT), m.Circumscribe(top_res, color=GOOD, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ LayerNorm

    def layernorm(self):
        self.next_section("LayerNorm", notes="LayerNorm (Ba, Kiros & Hinton, 2016): per token, across its features, subtract the mean, divide by the standard deviation (real code adds a tiny epsilon), then a learned scale and shift per feature. The four numbers, gamma and beta are illustrative. (ResNets for images used batch normalization, Ioffe & Szegedy 2015; Transformers use layer normalization.)")
        eq = formula(
            ("y", GOOD), ("=", INK), (r"\gamma", VIOLET), (r"\cdot", INK), (r"\frac{", INK), ("x", SKY), ("-", INK), (r"\mu", AMBER), ("}{", INK), (r"\sigma", PINK), ("}", INK), ("+", INK), (r"\beta", VIOLET),
            size=50,
        ).move_to([-0.85, 1.95, 0])
        numer = m.VGroup(eq[5], eq[6], eq[7])
        denom = m.VGroup(eq[9], eq[10])
        outer = m.VGroup(eq[0], eq[1], eq[2], eq[3], eq[11], eq[12])

        cx, rows_y = -0.85, (0.75, -0.3, -1.35, -2.4)

        def row(values, color, fill, y):
            chips = m.VGroup(*[chip(num(v), color, fill, width=1.3, height=0.62, size=26) for v in values]).arrange(m.RIGHT, buff=0.22)
            return chips.move_to([cx, y, 0])

        r1 = row(TOKEN, SKY, SKY_FILL, rows_y[0])
        r2 = row(CENTRED, AMBER, AMBER_FILL, rows_y[1])
        r3 = row(NORMED, PINK, PINK_FILL, rows_y[2])
        r4 = row(OUT, GOOD, GOOD_FILL, rows_y[3])
        lab_x = r1.get_left()[0] - 0.35
        labels = [
            tex("x", 40, SKY),
            formula(("x", SKY), ("-", INK), (r"\mu", AMBER), size=38),
            formula((r"\div", INK), (r"\sigma", PINK), size=38),
            tex("y", 40, GOOD),
        ]
        for lab, yy in zip(labels, rows_y):
            lab.move_to([0, yy, 0]).align_to([lab_x, 0, 0], m.RIGHT)
        side_x = r1.get_right()[0] + 0.45
        stats = m.VGroup(
            m.VGroup(text("mean", 26, AMBER, weight=m.SEMIBOLD), tex(rf"\mu = {num(MU)}", 34, AMBER)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(text("std", 26, PINK, weight=m.SEMIBOLD), tex(rf"\sigma = {num(SD)}", 34, PINK)).arrange(m.RIGHT, buff=0.15),
        )
        stats[0].move_to([0, rows_y[0] + 0.24, 0]).align_to([side_x, 0, 0], m.LEFT)
        stats[1].move_to([0, rows_y[0] - 0.26, 0]).align_to([side_x, 0, 0], m.LEFT)
        steady = pill("mean 0 · spread 1", GOOD, GOOD_FILL, 22).move_to([0, rows_y[2], 0]).align_to([side_x, 0, 0], m.LEFT)
        learned = m.VGroup(
            code("γ = " + "  ".join(num(v) for v in GAMMA), 22, VIOLET),
            code("β = " + "  ".join(num(v) for v in BETA), 22, VIOLET),
        ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        learned.move_to([0, rows_y[3], 0]).align_to([side_x, 0, 0], m.LEFT)
        token_box = m.SurroundingRectangle(r1, buff=0.14, corner_radius=0.15).set_stroke(SKY, width=2.5)
        per_token = pill("one token at a time, across its own features", SKY, SKY_FILL, 22).move_to([cx, -3.35, 0])

        self.say(
            "[To keep the numbers in check, we use layer normalization.] [Take one token's vector: four numbers, for example.]",
            self.new_header("LAYERNORM", "Put every token back on one scale"),
            m.AnimationGroup(m.FadeIn(labels[0]), m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in r1], lag_ratio=0.2)),
        )
        self.say(
            "[First, subtract their mean, which is four.]",
            m.AnimationGroup(
                m.Write(numer),
                m.FadeIn(stats[0], shift=0.15 * m.LEFT),
                m.FadeIn(labels[1]),
                *[m.TransformFromCopy(a, b) for a, b in zip(r1, r2)],
            ),
        )
        self.say(
            "[Then divide by their standard deviation, which is five.]",
            m.AnimationGroup(
                m.Write(denom),
                m.FadeIn(stats[1], shift=0.15 * m.LEFT),
                m.FadeIn(labels[2]),
                *[m.TransformFromCopy(a, b) for a, b in zip(r2, r3)],
            ),
        )
        self.say(
            "[Now they have a mean of zero, and a spread of one], [however big they were before.]",
            m.FadeIn(steady, shift=0.15 * m.LEFT),
            m.AnimationGroup(m.Indicate(r1, color=SKY, scale_factor=1.05), m.Indicate(r3, color=PINK, scale_factor=1.05)),
        )
        self.say(
            "[Finally, a learned scale and shift, one pair per feature], [let the network pick the range it likes.]",
            m.AnimationGroup(m.FadeIn(outer, shift=0.1 * m.UP), m.FadeIn(learned, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(labels[3]), *[m.TransformFromCopy(a, b) for a, b in zip(r3, r4)]),
        )
        self.say(
            "[Each token is normalized on its own, across its features.]",
            m.AnimationGroup(m.Create(token_box), m.FadeIn(per_token, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ pre-norm vs post-norm

    def placement(self):
        self.next_section(
            "Pre-norm",
            notes="Post-norm, x <- Norm(x + f(x)): the original Transformer (Vaswani et al., 2017). Pre-norm, x <- x + f(Norm(x)): GPT-2 onwards and most modern LLMs. "
            "Deep post-norm stacks need learning-rate warm-up (Xiong et al., 2020). RMSNorm: Zhang & Sennrich, 2019. Facts as of 2026-10.",
        )

        def panel(x, title, sub, cite, color, fill, pre):
            box = card(6.2, 5.3, color, CARD, stroke=2).move_to([x, -0.6, 0])
            heading = text(title, 28, color, weight=m.BOLD).move_to(box.get_top() + 0.4 * m.DOWN)
            subline = text(sub, 22, INK).next_to(heading, m.DOWN, buff=0.1)
            cite_t = text(cite, 20, MUTED).next_to(subline, m.DOWN, buff=0.1)
            px, bx = x - 1.35, x + 1.15
            y_in, y_fork, y_out = -2.0, -1.75, 0.55
            norm_main = None
            if pre:
                nrm = block("Norm", AMBER, AMBER_FILL, 1.5, 0.5, 22).move_to([bx, -1.25, 0])
                lay = block("layer f", VIOLET, VIOLET_FILL, 1.5, 0.5, 22).move_to([bx, -0.55, 0])
                y_plus = 0.0
                branch = m.VGroup(
                    polyline([[px, y_fork, 0], [bx, y_fork, 0], [bx, nrm.get_bottom()[1], 0]], MUTED, 3),
                    polyline([[bx, nrm.get_top()[1], 0], [bx, lay.get_bottom()[1], 0]], MUTED, 3),
                    tip_line([[bx, lay.get_top()[1], 0], [bx, y_plus, 0], [px + 0.26, y_plus, 0]], MUTED, 3, tip=0.2),
                )
                boxes = m.VGroup(nrm, lay)
                main = tip_line([[px, y_in, 0], [px, y_plus - 0.25, 0]], GOOD, 5, tip=0.2)
                main_top = tip_line([[px, y_plus + 0.25, 0], [px, y_out, 0]], GOOD, 5, tip=0.2)
            else:
                lay = block("layer f", VIOLET, VIOLET_FILL, 1.5, 0.5, 22).move_to([bx, -1.2, 0])
                y_plus = -0.65
                nrm = block("Norm", AMBER, AMBER_FILL, 1.5, 0.5, 22).move_to([px, -0.05, 0])
                norm_main = nrm
                branch = m.VGroup(
                    polyline([[px, y_fork, 0], [bx, y_fork, 0], [bx, lay.get_bottom()[1], 0]], MUTED, 3),
                    tip_line([[bx, lay.get_top()[1], 0], [bx, y_plus, 0], [px + 0.26, y_plus, 0]], MUTED, 3, tip=0.2),
                )
                boxes = m.VGroup(lay, nrm)
                main = tip_line([[px, y_in, 0], [px, y_plus - 0.25, 0]], BAD, 5, tip=0.2)
                main_top = m.VGroup(
                    polyline([[px, y_plus + 0.25, 0], [px, nrm.get_bottom()[1], 0]], BAD, 5),
                    tip_line([[px, nrm.get_top()[1], 0], [px, y_out, 0]], BAD, 5, tip=0.2),
                )
            plus = plus_node([px, y_plus, 0], INK, r=0.25)
            x_in = tex("x", 36, SKY).next_to([px, y_in, 0], m.DOWN, buff=0.08)
            parts = dict(box=box, heading=heading, sub=subline, branch=branch, boxes=boxes, main=main, main_top=main_top, plus=plus, x_in=x_in, cite=cite_t, norm_main=norm_main, px=px, y_in=y_in, y_out=y_out)
            parts["all"] = m.VGroup(box, heading, subline, branch, boxes, main, main_top, plus, x_in, cite_t)
            return parts

        post = panel(-3.4, "Post-norm", "normalize after adding", "original Transformer · Vaswani et al., 2017", BAD, BAD_FILL, pre=False)
        pre = panel(3.4, "Pre-norm", "normalize before each layer", "most modern language models", GOOD, GOOD_FILL, pre=True)
        bottom = post["box"].get_bottom()[1]
        hard = m.VGroup(
            text("hard to train deep: needs a slow warm-up", 22, BAD, weight=m.SEMIBOLD),
            text("Xiong et al., 2020", 20, MUTED),
        ).arrange(m.DOWN, buff=0.08).move_to([post["box"].get_x(), bottom + 0.55, 0])
        clean = text("the highway carries only additions", 22, GOOD, weight=m.SEMIBOLD).move_to([pre["box"].get_x(), bottom + 0.55, 0])
        highway_pre = polyline([[pre["px"], pre["y_out"], 0], [pre["px"], pre["y_in"], 0]], GRAD, 10)
        highway_post = polyline([[post["px"], post["y_out"], 0], [post["px"], post["y_in"], 0]], GRAD, 10)

        rms_box = card(6.2, 5.3, AMBER, CARD, stroke=2).move_to([-3.4, -0.6, 0])
        rms_title = text("RMSNorm: a lighter norm", 28, AMBER, weight=m.BOLD).move_to(rms_box.get_top() + 0.4 * m.DOWN)
        rms_eq = formula(("y", GOOD), ("=", INK), (r"\gamma", VIOLET), (r"\cdot", INK), (r"\frac{", INK), ("x", SKY), ("}{", INK), (r"\sqrt{\tfrac{1}{n}\sum_i x_i^2}", AMBER), ("}", INK), size=44)
        rms_eq.move_to(rms_box.get_center() + 0.35 * m.UP)
        rms_notes = m.VGroup(
            text("no mean to subtract", 22, INK),
            text("divide by the root mean square", 22, AMBER),
            text("then a learned scale", 22, VIOLET),
        ).arrange(m.DOWN, buff=0.14).next_to(rms_eq, m.DOWN, buff=0.4)
        rms_cite = text("Zhang & Sennrich, 2019", 20, MUTED).move_to(rms_box.get_bottom() + 0.38 * m.UP)

        def show(p):
            return m.AnimationGroup(
                m.FadeIn(p["box"]), m.FadeIn(p["heading"], shift=0.15 * m.UP), m.FadeIn(p["sub"]),
                m.FadeIn(p["x_in"]), m.Create(p["main"]), m.FadeIn(p["plus"]), m.Create(p["branch"]),
                m.FadeIn(p["boxes"]), m.Create(p["main_top"]), m.FadeIn(p["cite"]),
            )

        self.say(
            "[Where should the normalization go?] [The original Transformer, in twenty seventeen, put it after the addition]: [that's post-norm.]",
            self.new_header("PRE-NORM", "Where the normalization goes"),
            show(post),
            m.Indicate(post["norm_main"], color=AMBER, scale_factor=1.1),
        )
        self.say(
            "[That puts a normalization right on the highway], [so the gradient must pass through one at every layer.]",
            m.Circumscribe(post["norm_main"], color=BAD, buff=0.1),
            m.ShowPassingFlash(highway_post, time_width=0.5, run_time=1.6),
        )
        self.say(
            "[Deep post-norm stacks are harder to train], [and usually need a slow warm-up of the learning rate.]",
            m.FadeIn(hard[0], shift=0.1 * m.UP),
            m.FadeIn(hard[1]),
        )
        self.say(
            "[Most modern language models normalize before each layer instead]: [pre-norm.] [The highway stays clean.]",
            show(pre),
            m.Indicate(pre["heading"], color=GOOD, scale_factor=1.1),
            m.AnimationGroup(m.ShowPassingFlash(highway_pre, time_width=0.5, run_time=1.6), m.FadeIn(clean, shift=0.15 * m.UP)),
        )
        self.say(
            "[Many also use a lighter version, root mean square norm]: [it skips the mean, and divides by the root mean square.]",
            m.AnimationGroup(m.FadeOut(post["all"]), m.FadeOut(hard), m.FadeIn(rms_box), m.FadeIn(rms_title, shift=0.15 * m.UP), m.Write(rms_eq), m.FadeIn(rms_cite)),
            m.LaggedStart(*[m.FadeIn(n, shift=0.1 * m.UP) for n in rms_notes], lag_ratio=0.35),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section("The big picture", notes="The pre-norm Transformer block: x + Attention(Norm(x)), then x + FeedForward(Norm(x)) (episode 24). Stacked a hundred times.")
        px, bx = -4.6, -2.1
        y_in, y_out = -3.05, 2.2
        n1 = block("Norm", AMBER, AMBER_FILL, 2.3, 0.5, 22).move_to([bx, -2.0, 0])
        at = block("Attention", SKY, SKY_FILL, 2.3, 0.5, 22).move_to([bx, -1.3, 0])
        n2 = block("Norm", AMBER, AMBER_FILL, 2.3, 0.5, 22).move_to([bx, 0.25, 0])
        ff = block("Feed-forward", VIOLET, VIOLET_FILL, 2.3, 0.5, 22).move_to([bx, 0.95, 0])
        p1 = plus_node([px, -0.55, 0], INK, r=0.24)
        p2 = plus_node([px, 1.7, 0], INK, r=0.24)

        def branch(fork_y, boxes, plus_y):
            lo, hi = boxes
            return m.VGroup(
                polyline([[px, fork_y, 0], [bx, fork_y, 0], [bx, lo.get_bottom()[1], 0]], MUTED, 3),
                polyline([[bx, lo.get_top()[1], 0], [bx, hi.get_bottom()[1], 0]], MUTED, 3),
                tip_line([[bx, hi.get_top()[1], 0], [bx, plus_y, 0], [px + 0.26, plus_y, 0]], MUTED, 3),
            )

        b1 = branch(-2.6, (n1, at), -0.55)
        b2 = branch(-0.05, (n2, ff), 1.7)
        highway = m.VGroup(
            polyline([[px, y_in, 0], [px, -0.79, 0]], GOOD, 5),
            polyline([[px, -0.31, 0], [px, 1.46, 0]], GOOD, 5),
            tip_line([[px, 1.94, 0], [px, y_out, 0]], GOOD, 5),
        )
        x_in = tex("x", 34, SKY).next_to([px, y_in, 0], m.LEFT, buff=0.15)
        frame = card(5.0, 5.55, EDGE, CARD, stroke=2).move_to([-3.25, -0.4, 0]).set_z_index(-1)
        frame_label = text("one block", 22, MUTED, weight=m.SEMIBOLD).next_to(frame.get_corner(m.UR), m.DL, buff=0.2)
        steps = [(n1, at, p1, b1), (n2, ff, p2, b2)]

        slabs = tower(3.1, -2.9, 2.0, 1.3)
        lane_x = slabs.get_left()[0] - 0.4  # the highway on the left, as in the block
        lane = m.Line([lane_x, slabs.get_top()[1], 0], [lane_x, slabs.get_bottom()[1], 0]).set_stroke(GOOD, width=5)
        stack_arrow = arrow([frame.get_right()[0], -0.45, 0], [lane_x - 0.05, -0.45, 0], MUTED, buff=0.25)
        times = text("× 100 blocks", 26, INK, weight=m.BOLD).next_to(stack_arrow, m.UP, buff=0.2)
        reached = pill("✓ reaches layer 1", GOOD, GOOD_FILL, 22).next_to(slabs, m.RIGHT, buff=0.3).align_to(slabs, m.DOWN)

        def run(k):
            lo, hi, plus, br = steps[k]
            return m.Succession(
                m.ShowPassingFlash(br.copy().set_stroke(INK, width=6), time_width=0.6, run_time=0.9),
                m.AnimationGroup(m.Indicate(lo, color=AMBER, scale_factor=1.06), run_time=0.5),
                m.AnimationGroup(m.Indicate(hi, color=hi[0].get_stroke_color(), scale_factor=1.06), run_time=0.5),
                m.Indicate(plus, color=GOOD, scale_factor=1.3, run_time=0.5),
            )

        self.say(
            "[Put it together, and you get the block inside a modern Transformer], [like the ones from episode twenty-four.]",
            m.AnimationGroup(
                self.new_header("THE BIG PICTURE", "Normalize, compute, add"),
                m.FadeIn(frame), m.FadeIn(frame_label), m.FadeIn(x_in), m.Create(highway),
                m.Create(b1), m.Create(b2), *[m.FadeIn(b, scale=0.9) for b in (n1, at, n2, ff)], m.FadeIn(p1), m.FadeIn(p2),
            ),
            m.ShowPassingFlash(polyline([[px, y_in, 0], [px, y_out, 0]], GOOD, 10), time_width=0.5, run_time=1.4),
        )
        self.say("[Normalize, attend, add.] [Normalize, feed forward, add.]", run(0), run(1))
        self.say(
            "[Stack a hundred of these blocks], [and the gradient still reaches the very first layer.]",
            m.AnimationGroup(m.GrowArrow(stack_arrow), m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.DOWN) for s in slabs], lag_ratio=0.01, run_time=1.4), m.Create(lane), m.FadeIn(times)),
            m.AnimationGroup(
                m.ShowPassingFlash(lane.copy().set_stroke(GRAD, width=12), time_width=0.4, run_time=1.8),
                m.LaggedStart(*[slabs[DEPTH - 1 - d].animate.set_fill(mix(SLAB, GRAD, 0.85)) for d in range(DEPTH)], lag_ratio=0.02, run_time=1.8),
                m.Succession(m.Wait(1.3), m.FadeIn(reached, shift=0.15 * m.LEFT, run_time=0.5)),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Skip, normalize, stack.")
        halo = glow(VIOLET, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Skip", "Normalize", "Stack"), (SKY, AMBER, VIOLET))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Small changes, on one steady scale, a hundred layers deep.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [skip], [normalize], [stack.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say(
            "[Let every layer add a small change], [keep the numbers on one scale], [and a hundred layers can learn together.]",
            m.Write(tagline),
            m.Indicate(words[1], color=AMBER, scale_factor=1.08),
            m.Indicate(words[2], color=VIOLET, scale_factor=1.08),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
