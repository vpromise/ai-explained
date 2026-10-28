"""AI, Explained · Episode 11 · Derivatives & Gradients

Running example: a hiker in thick fog on the side of a valley, who can only feel the slope
under their feet. The valley is an illustrative surface, made up for teaching:
height f(x, y) = x²/2 + y² (x: how far east, y: how far north). Its gradient (x, 2y) is
computed analytically and checked against finite differences below.

The hiker stands at (3, 2): height 8.5, slope 3 facing east, 4 facing north, gradient (3, 4),
steepest slope 5. The side view is the east-west slice f(x, 2) = x²/2 + 4, whose rise over a
run h from x = 3 is exactly 3 + h/2 (a run of 2 rises 8: slope 4; then 3.5, 3.25, … → 3).
Descent: eight steps of (x, y) ← (x, y) − 0.3·∇f take the height from 8.5 to about 0.01.
Every number the narration quotes is asserted below, so the picture can't drift from the words.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the valley


def f(x, y):
    return 0.5 * x**2 + y**2


def grad(x, y):
    return np.array([x, 2.0 * y])


HX, HY = 3.0, 2.0  # where the hiker stands
HEIGHT = f(HX, HY)
G = grad(HX, HY)
STEEP = float(np.hypot(*G))
G_ANGLE = float(np.arctan2(G[1], G[0]))


def side(x):
    """The east-west slice through the hiker (y held at 2): the side view."""
    return f(x, HY)


def north(y):
    """The north-south slice through the hiker (x held at 3)."""
    return f(HX, y)


def secant(h):
    return (side(HX + h) - side(HX)) / h


def numeric(fn, t, eps=1e-5):
    return (fn(t + eps) - fn(t - eps)) / (2 * eps)


def slope_toward(theta):
    """How fast the height rises per unit walked in direction theta (the directional derivative)."""
    return float(G @ [np.cos(theta), np.sin(theta)])


# the numbers the narration says
assert HEIGHT == 8.5 and tuple(G) == (3.0, 4.0) and STEEP == 5.0  # "three east, four north", "five"
assert side(HX + 2) - side(HX) == 8 and secant(2) == 4  # "a step of two, the ground rises eight: a slope of four"
assert secant(1) == 3.5 and secant(0.5) == 3.25  # "three and a half", "three and a quarter"
assert abs(secant(1e-3) - 3) < 1e-3 and abs(numeric(side, HX) - 3) < 1e-6  # "exactly three"
assert abs(numeric(north, HY) - 4) < 1e-6  # "its partial derivative is four"
_rng = np.random.default_rng(11)  # the analytic gradient matches finite differences anywhere
for _x, _y in _rng.uniform(-4, 4, (25, 2)):
    _fd = [numeric(lambda t: f(t, _y), _x), numeric(lambda t: f(_x, t), _y)]
    assert np.allclose(_fd, grad(_x, _y), atol=1e-6)
_TH = np.linspace(0, 2 * np.pi, 36001)
_E = 1e-6
_DIR = (f(HX + _E * np.cos(_TH), HY + _E * np.sin(_TH)) - f(HX - _E * np.cos(_TH), HY - _E * np.sin(_TH))) / (2 * _E)
assert abs(_DIR[0] - 3) < 1e-6 and abs(_DIR[9000] - 4) < 1e-6  # facing east: three; north: four
assert abs(_DIR.max() - 5) < 1e-6 and abs(_TH[_DIR.argmax()] - G_ANGLE) < 1e-3  # the steepest way is the gradient's: five
assert np.allclose(_DIR, [slope_toward(a) for a in _TH], atol=1e-6)
_TAN = np.array([G[1], -G[0]]) / STEEP  # along the contour through the hiker, the height doesn't change
assert abs(numeric(lambda t: f(HX + t * _TAN[0], HY + t * _TAN[1]), 0.0)) < 1e-6

ETA, STEPS = 0.3, 8  # illustrative step size
PATH = [np.array([HX, HY])]
for _ in range(STEPS):
    PATH.append(PATH[-1] - ETA * grad(*PATH[-1]))
assert all(f(*a) > f(*b) for a, b in zip(PATH, PATH[1:]))  # every step goes down
assert f(*PATH[-1]) < 0.02  # "after eight steps ... from eight and a half to almost zero"

RINGS = [HEIGHT * k / 5 for k in range(1, 6)]  # contour heights; the last one runs through the hiker

# ---------------------------------------------------------------------------- the map (top view)

MAP_C = np.array([-3.0, -0.55, 0.0])
MAP_S = 0.66  # frame units per map unit, the same both ways so right angles look right
MAP_X, MAP_Y = 4.6, 3.1


def mp(x, y):
    return MAP_C + MAP_S * np.array([x, y, 0.0])


def coords(point):
    return (np.asarray(point)[:2] - MAP_C[:2]) / MAP_S


# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def number(value, color, size=32, places=2, **kw):
    return m.DecimalNumber(value, num_decimal_places=places, mob_class=Mono, font_size=size, color=color, edge_to_fix=m.RIGHT, **kw)


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


def fog(width, height, peak=0.3, layers=16, color=MUTED):
    """A soft bank of fog: nested ellipses, each faint, so the edge has no hard line."""
    return m.VGroup(
        *[
            m.Ellipse(width=width * (1 - 0.85 * i / layers), height=height * (1 - 0.85 * i / layers), stroke_width=0).set_fill(
                color, opacity=peak / layers
            )
            for i in range(layers)
        ]
    )


def marker(at, scale=1.0):
    """The hiker, seen as a point: a dot in a ring."""
    ring = m.Circle(radius=0.2 * scale).set_stroke(SKY, width=3).set_fill(BG, opacity=0.0)
    return m.VGroup(ring, m.Dot(radius=0.09 * scale, color=SKY)).move_to(at).set_z_index(6)


def tip_arrow(start, end, color, width=5, ratio=0.3):
    return m.Arrow(start, end, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=ratio)


def baseline(words):
    """The bottom of a Text's glyphs that don't descend below the line."""
    sitting = [g for g, c in zip(words, words.text.replace(" ", "")) if c not in "gjpqy,"] or list(words)
    return min(g.get_bottom()[1] for g in sitting)


def terrain_y(x):
    """The hook's mountainside, in frame units: a valley at the left of the hiker."""
    return -3.0 + 0.08 * (x + 1.0) ** 2 + 0.18 * np.sin(1.2 * x + 0.3)


# ---------------------------------------------------------------------------- the episode


class Ep11DerivativesGradients(Episode):
    number = 11

    def lesson(self):
        self.hook()
        self.slope()
        self.derivative()
        self.valley_map()
        self.partials()
        self.gradient()
        self.descent()
        self.big_picture()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A hiker in fog can only feel the slope underfoot. Illustrative scene.")
        xs = np.linspace(-7.1, 7.1, 180)
        ground = m.VGroup(
            m.Polygon(*[[x, terrain_y(x), 0] for x in xs], [7.1, -3.98, 0], [-7.1, -3.98, 0], stroke_width=0, fill_color=TEAL_FILL, fill_opacity=1),
            polyline([[x, terrain_y(x), 0] for x in xs], TEAL, width=3),
        )
        hx = 4.2
        hiker = icon("ep11-hiker", 1.15)  # front foot on the ground, back foot lifted mid-step
        hiker.move_to([hx, terrain_y(hx + 0.14) + hiker.height / 2 - 0.02, 0]).set_z_index(3)
        banks = m.VGroup(
            fog(10.0, 2.6, 0.5).move_to([-1.2, -2.5, 0]),
            fog(5.6, 1.6, 0.38).move_to([-4.25, -0.7, 0]),
            fog(5.5, 1.3, 0.32).move_to([1.2, -1.4, 0]),
            fog(3.6, 1.1, 0.28).move_to([5.25, 1.3, 0]),
        ).set_z_index(1)
        thick = fog(7.0, 1.8, 0.35).move_to([-1.0, -2.0, 0]).set_z_index(1)
        slope_here = numeric(terrain_y, hx)
        u = np.array([1.0, slope_here, 0]) / np.hypot(1.0, slope_here)
        feet = np.array([hx, terrain_y(hx), 0])
        tilt = m.Line(feet - 0.95 * u, feet + 0.95 * u).shift(0.03 * m.DOWN).set_stroke(AMBER, width=6).set_z_index(4)
        tilt_label = text("feel the tilt", 24, AMBER, weight=m.SEMIBOLD).next_to(hiker, m.LEFT, buff=0.3).shift(0.35 * m.UP)
        down = tip_arrow([hx - 0.5, terrain_y(hx - 0.5) + 0.42, 0], [hx - 1.65, terrain_y(hx - 1.65) + 0.42, 0], GOOD).set_z_index(4)
        down_label = text("down", 24, GOOD, weight=m.SEMIBOLD).next_to(down.get_end(), m.LEFT, buff=0.18)

        question = text("Which way is down?", 52, INK, weight=m.BOLD).move_to([-2.4, 2.3, 0])
        names = m.VGroup(
            pill("one direction: a derivative", AMBER, AMBER_FILL, 26),
            pill("many directions: a gradient", AMBER, AMBER_FILL, 26),
        ).arrange(m.DOWN, buff=0.3).move_to([-2.4, 1.0, 0])

        self.say(
            "So... [imagine you're hiking high on a mountain], [when a thick fog rolls in.]",
            m.AnimationGroup(m.FadeIn(ground, shift=0.2 * m.UP), m.FadeIn(hiker, shift=0.3 * m.DOWN), run_time=1.2),
            m.LaggedStart(*[m.FadeIn(b, shift=0.6 * m.RIGHT) for b in banks], lag_ratio=0.25, run_time=2.0),
        )
        self.say(
            "[Which way is down? You can't see the valley.] [But you can feel the ground under your feet]: [which way it tilts, and how steeply.]",
            m.AnimationGroup(m.FadeIn(question, shift=0.2 * m.DOWN), m.FadeIn(thick, shift=0.4 * m.RIGHT)),
            m.Create(tilt),
            m.AnimationGroup(m.FadeIn(tilt_label), tilt.animate(rate_func=m.there_and_back).set_stroke(width=10)),
        )
        self.say(
            "[That tilt is enough to find the way down.] [In math, it's called a derivative], [and across many directions at once, a gradient.]",
            m.AnimationGroup(m.GrowArrow(down), m.FadeIn(down_label)),
            m.FadeIn(names[0], shift=0.2 * m.UP),
            m.FadeIn(names[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the slope: rise over run

    def side_view(self):
        """The east-west slice through the hiker on axes at the left, and the secant over a run h."""
        ax = axes([-1, 6, 1], [0, 18, 3], 7.0, 5.2).move_to([-2.75, -0.65, 0])
        self.ax = ax
        xs = np.linspace(-1, 5.25, 160)
        ground = m.Polygon(
            *[ax.c2p(x, side(x)) for x in xs], ax.c2p(5.25, 0), ax.c2p(-1, 0),
            stroke_width=0, fill_color=TEAL_FILL, fill_opacity=0.9,
        ).set_z_index(-1)
        curve = polyline([ax.c2p(x, side(x)) for x in xs], TEAL, width=4.5)
        x_label = text("x · east", 22, MUTED).next_to(ax.x_axis.get_end(), m.DOWN, buff=0.3).shift(0.4 * m.LEFT)
        y_label = text("height", 22, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.18)
        self.P = ax.c2p(HX, HEIGHT)
        who = marker(self.P)
        who_label = text("hiker", 22, SKY, weight=m.SEMIBOLD).next_to(who, m.LEFT, buff=0.18).shift(0.3 * m.UP)
        self.h = m.ValueTracker(2.0)

        def make_secant():
            h = self.h.get_value()
            P, Q = np.array(self.P), np.array(ax.c2p(HX + h, side(HX + h)))
            corner = np.array(ax.c2p(HX + h, HEIGHT))
            d = Q - P
            length = float(np.linalg.norm(d))
            u = d / length
            line = m.Line(P - 1.3 * u, P + max(1.3, length + 0.7) * u).set_stroke(AMBER, width=4.5)
            fade = float(np.clip((h - 0.45) / 0.35, 0, 1))
            run = m.Line(P, corner).set_stroke(BLUE, width=4, opacity=fade)
            rise = m.Line(corner, Q).set_stroke(TEAL, width=4, opacity=fade)
            q = m.Dot(Q, radius=0.08, color=AMBER).set_opacity(fade)
            words = float(np.clip((h - 1.1) / 0.4, 0, 1))  # the words leave before the triangle gets small
            run_l = text("run", 26, BLUE, weight=m.SEMIBOLD).next_to(run, m.DOWN, buff=0.12)
            rise_l = text("rise", 26, TEAL, weight=m.SEMIBOLD).next_to(corner + 0.3 * (Q - corner), m.LEFT, buff=0.12)
            if words == 0:  # hidden: parked in an empty corner of the plot (always_redraw keeps the same parts)
                run_l.move_to(ax.c2p(0.2, 15.5))
                rise_l.next_to(run_l, m.RIGHT, buff=0.3)
            run_l.set_opacity(words)
            rise_l.set_opacity(words)
            return m.VGroup(line, run, rise, q, run_l, rise_l).set_z_index(3)

        self.make_secant = make_secant
        self.secant = m.always_redraw(make_secant)
        return ax, ground, curve, x_label, y_label, who, who_label

    def slope(self):
        self.next_section("The slope", notes="Rise over run on the east-west slice f(x, 2) = x²/2 + 4: a run of 2 rises 8, slope 4.")
        ax, ground, curve, x_label, y_label, who, who_label = self.side_view()
        self.who = who

        rule = text("slope = rise ÷ run", 36, INK, weight=m.BOLD, t2c={"slope": AMBER, "rise": TEAL, "run": BLUE}).move_to([3.95, 2.1, 0])
        board = card(4.7, 2.3, EDGE, CARD).move_to([3.95, 0.5, 0])
        rows = []
        for k, (label, color) in enumerate((("run", BLUE), ("rise", TEAL), ("slope", AMBER))):
            y = board.get_top()[1] - 0.48 - 0.67 * k
            name = text(label, 28, color, weight=m.SEMIBOLD).move_to([0, y, 0]).align_to(board, m.LEFT).shift(0.45 * m.RIGHT)
            rows.append(name)
        run_v = number(2.0, BLUE).move_to([5.75, rows[0].get_y(), 0], aligned_edge=m.RIGHT)
        rise_v = number(8.0, TEAL).move_to([5.75, rows[1].get_y(), 0], aligned_edge=m.RIGHT)
        slope_v = number(4.0, AMBER).move_to([5.75, rows[2].get_y(), 0], aligned_edge=m.RIGHT)
        run_v.add_updater(lambda d: d.set_value(self.h.get_value()))
        rise_v.add_updater(lambda d: d.set_value(side(HX + self.h.get_value()) - HEIGHT))
        slope_v.add_updater(lambda d: d.set_value(secant(self.h.get_value())))
        self.readouts = (run_v, rise_v, slope_v)
        self.board = m.VGroup(board, *rows, run_v, rise_v, slope_v)
        self.rule = rule
        average = text("an average over the whole step", 22, MUTED).next_to(board, m.DOWN, buff=0.3)
        between = polyline([ax.c2p(x, side(x)) for x in np.linspace(HX, HX + 2, 40)], INK, width=7)

        self.say(
            "[Let's look at the ground from the side], along the east-west path the hiker is on.",
            m.AnimationGroup(
                self.new_header("THE SLOPE", "Rise over run"),
                m.Create(ax),
                m.FadeIn(ground),
                m.Create(curve),
                m.FadeIn(x_label),
                m.FadeIn(y_label),
                m.FadeIn(who, scale=0.5),
                m.FadeIn(who_label),
            ),
        )
        static = self.make_secant()  # the same picture as the live secant at h = 2, built up piece by piece
        line, run, rise, q, run_l, rise_l = static
        self.say(
            "[A slope is rise over run]: [walk a little way east], [and see how much higher you get.]",
            m.Write(rule),
            m.AnimationGroup(m.Create(run), m.FadeIn(run_l)),
            m.AnimationGroup(m.Create(rise), m.FadeIn(rise_l), m.FadeIn(q, scale=0.5)),
        )
        self.say(
            "[Take a step of two], [and the ground rises eight]: [a slope of four.]",
            m.AnimationGroup(m.FadeIn(board), m.FadeIn(rows[0]), m.FadeIn(run_v), m.Indicate(run, color=BLUE, scale_factor=1.05)),
            m.AnimationGroup(m.FadeIn(rows[1]), m.FadeIn(rise_v), m.Indicate(rise, color=TEAL, scale_factor=1.05)),
            m.AnimationGroup(m.Create(line), m.FadeIn(rows[2]), m.FadeIn(slope_v), m.Indicate(slope_v, color=AMBER, scale_factor=1.2)),
        )
        self.say(
            "[But that's an average over the whole step.] [The ground right under your feet could be steeper, or flatter.]",
            m.AnimationGroup(m.FadeIn(average, shift=0.15 * m.UP), m.ShowPassingFlash(between, time_width=0.8, run_time=1.4)),
            m.Indicate(who, color=SKY, scale_factor=1.5),
        )
        self.average = average
        self.remove(*static)
        self.add(self.secant)
        self.wait(0.3)

    # ------------------------------------------------------------------ the derivative: shrink the step

    def derivative(self):
        self.next_section("The derivative", notes="Shrink the run: 4 → 3.5 → 3.25 → … → 3, the tangent's slope. f'(x) as a limit.")
        ax = self.ax
        limit = m.MathTex("f'(x)", "=", r"\lim_{h \to 0}", r"\frac{", r"f(x+h) - f(x)", "}{", "h", "}", font_size=40, color=INK)
        limit[0].set_color(AMBER)
        limit[4].set_color(TEAL)
        limit[6].set_color(BLUE)
        limit.move_to([3.95, -1.85, 0])
        caption = text("the derivative: the slope of the tangent", 22, MUTED).next_to(limit, m.DOWN, buff=0.3)
        P = np.array(self.P)
        u = np.array(ax.c2p(HX + 1, HEIGHT + 3)) - P
        u /= np.linalg.norm(u)
        tangent_label = text("tangent", 24, AMBER, weight=m.SEMIBOLD).next_to(P + 1.3 * u, m.DR, buff=0.05)

        y_arrows = P[1] + 1.3
        px = P[0]
        down = tip_arrow([px - 0.3, y_arrows, 0], [px - 1.35, y_arrows, 0], GOOD, width=5)
        up = tip_arrow([px + 0.3, y_arrows, 0], [px + 1.0, y_arrows, 0], AMBER, width=5)
        down_l = text("west: down", 22, GOOD, weight=m.SEMIBOLD).next_to(down, m.UP, buff=0.12)
        up_l = text("east: up", 22, AMBER, weight=m.SEMIBOLD).next_to(up, m.UP, buff=0.12)
        plus = text("slope +3", 26, AMBER, weight=m.BOLD).next_to(m.VGroup(down_l, up_l), m.UP, buff=0.15)

        self.say(
            "[So let's shrink the step.] [A step of one gives three and a half.] [Half a step: three and a quarter.]",
            m.AnimationGroup(self.new_header("THE DERIVATIVE", "The slope at a single point"), m.FadeOut(self.average)),
            self.h.animate(run_time=1.2).set_value(1.0),
            self.h.animate(run_time=1.2).set_value(0.5),
        )
        self.say(
            "[As the step shrinks toward zero], [the line settles on one slope: exactly three.]",
            self.h.animate(run_time=2.4, rate_func=m.smooth).set_value(0.001),
            m.AnimationGroup(m.FadeIn(tangent_label), m.Indicate(self.readouts[2], color=AMBER, scale_factor=1.25)),
        )
        self.say(
            "[That limit is the derivative]: [the slope of the tangent line, right where you stand.]",
            m.Write(limit, run_time=1.6),
            m.AnimationGroup(m.FadeIn(caption, shift=0.15 * m.UP), m.Indicate(self.who, color=SKY, scale_factor=1.5)),
        )
        self.say(
            "[On top, the rise], [underneath, the run, h], [and we let h shrink to zero.]",
            m.Indicate(limit[4], color=TEAL, scale_factor=1.15),
            m.Indicate(limit[6], color=BLUE, scale_factor=1.4),
            m.Indicate(limit[2], color=AMBER, scale_factor=1.15),
        )
        self.say(
            "[Plus three means the ground rises toward the east], [so the way down is west.]",
            m.AnimationGroup(m.FadeIn(plus, shift=0.15 * m.DOWN), m.GrowArrow(up), m.FadeIn(up_l)),
            m.AnimationGroup(m.GrowArrow(down), m.FadeIn(down_l)),
        )
        self.wait(0.3)
        self.stop(self.secant, *self.readouts)
        self.clear()

    # ------------------------------------------------------------------ the map: the valley from above

    def build_map(self):
        corner = mp(-MAP_X, -MAP_Y)
        x_axis = tip_arrow(corner, mp(MAP_X + 0.15, -MAP_Y), FAINT, width=2.5, ratio=0.04)
        y_axis = tip_arrow(corner, mp(-MAP_X, MAP_Y + 0.15), FAINT, width=2.5, ratio=0.05)
        self.x_name = text("x · east", 22, MUTED).next_to(x_axis.get_end(), m.DOWN, buff=0.18)
        self.y_name = text("y · north", 22, MUTED).next_to(y_axis.get_end(), m.UP, buff=0.12)
        self.frame_axes = m.VGroup(x_axis, y_axis)
        self.rings = m.VGroup(
            *[
                m.Ellipse(width=2 * np.sqrt(2 * c) * MAP_S, height=2 * np.sqrt(c) * MAP_S)
                .set_stroke(TEAL, width=2.5, opacity=0.35 + 0.65 * (k + 1) / len(RINGS))
                .move_to(MAP_C)
                for k, c in enumerate(RINGS)
            ]
        )
        self.floor = m.Dot(mp(0, 0), radius=0.09, color=GOOD).set_z_index(4)
        self.floor_label = text("valley floor", 22, GOOD, weight=m.SEMIBOLD).next_to(self.floor, m.DOWN, buff=0.16)
        self.P = mp(HX, HY)
        self.who = marker(self.P)

    def valley_map(self):
        self.next_section(
            "The valley from above",
            notes="Contours of the illustrative surface f(x, y) = x²/2 + y²; the side view was the slice y = 2.",
        )
        self.build_map()
        compass = m.VGroup(
            *[tip_arrow(self.P + 0.28 * d, self.P + 0.85 * d, MUTED, width=4, ratio=0.35) for d in (m.RIGHT, m.LEFT, m.UP, m.DOWN)]
        )
        formula = tex(r"f(x, y) = \tfrac{1}{2}x^2 + y^2", 46, INK).move_to([3.95, 1.75, 0])
        legend = m.VGroup(
            m.VGroup(tex("x", 34, BLUE), text("how far east", 24, MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(tex("y", 34, PINK), text("how far north", 24, MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(tex("f", 34, TEAL), text("the height there", 24, MUTED)).arrange(m.RIGHT, buff=0.3),
        ).arrange(m.DOWN, buff=0.32, aligned_edge=m.LEFT).move_to([3.95, 0.05, 0])
        for row in legend[1:]:
            row[1].align_to(legend[0][1], m.LEFT)
        example = text("an example valley", 22, MUTED).next_to(legend, m.DOWN, buff=0.45)
        self.slice_x = m.DashedLine(mp(-MAP_X, HY), mp(MAP_X, HY), dash_length=0.12).set_stroke(BLUE, width=3).set_z_index(2)

        self.say(
            "[But a mountain isn't one path.] [The hiker can step east or west, north or south.]",
            m.AnimationGroup(
                self.new_header("THE MAP", "The valley from above"),
                m.Create(self.frame_axes),
                m.FadeIn(self.x_name),
                m.FadeIn(self.y_name),
                m.FadeIn(self.who, scale=0.5),
            ),
            m.LaggedStart(*[m.GrowArrow(a) for a in compass], lag_ratio=0.2),
        )
        self.say(
            "[Here's the valley from above, with contour lines, like a hiking map]: [each ring joins points at the same height.]",
            m.AnimationGroup(
                m.FadeOut(compass),
                m.LaggedStart(*[m.Create(r) for r in self.rings], lag_ratio=0.2, run_time=2.2),
                m.FadeIn(self.floor, scale=0.4),
                m.FadeIn(self.floor_label),
            ),
            m.ShowPassingFlash(self.rings[-1].copy().set_stroke(TEAL, width=8, opacity=1), time_width=0.6, run_time=1.6),
        )
        self.say(
            "[For example, say the height is one half x squared, plus y squared], [where x is how far east you are, and y how far north.]",
            m.Write(formula, run_time=1.5),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.UP) for r in legend], lag_ratio=0.3), m.FadeIn(example)),
        )
        self.say(
            "[Our side view was one slice of this map]: [the east-west line through the hiker.]",
            m.Create(self.slice_x),
            m.Indicate(self.who, color=SKY, scale_factor=1.5),
        )
        self.formula, self.legend, self.example = formula, legend, example
        self.wait(0.3)

    # ------------------------------------------------------------------ partial derivatives

    def inset(self, center, title, color, fn, lo, hi, at, formula):
        """A small side view: the slice fn over [lo, hi], the hiker at `at`, its tangent, the partial."""
        box = card(5.6, 2.6, color, CARD, stroke=2).move_to(center)
        left, bottom = box.get_left()[0] + 0.35, box.get_bottom()[1] + 0.28
        sx, sy = 0.74, 0.17  # the same scale in both insets, so the steeper one looks steeper

        def q(t, height):
            return np.array([left + (t - lo) * sx, bottom + (height - 4.0) * sy, 0.0])

        curve = polyline([q(t, fn(t)) for t in np.linspace(lo, hi, 120)], TEAL, width=4)
        here = q(at, fn(at))
        u = np.array([sx, numeric(fn, at) * sy, 0.0])
        u /= np.linalg.norm(u)
        tangent = m.Line(here - 0.7 * u, here + 0.7 * u).set_stroke(color, width=4.5)
        dot = m.Dot(here, radius=0.09, color=SKY).set_z_index(3)
        head = text(title, 24, color, weight=m.SEMIBOLD)
        head.move_to(box.get_top() + 0.38 * m.DOWN).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
        eq = m.MathTex(*formula, font_size=40, color=color).move_to([box.get_right()[0] - 0.95, box.get_y() - 0.1, 0])
        return m.VGroup(box, head, curve, dot), tangent, eq

    def partials(self):
        self.next_section("Partial derivatives", notes="Slices through the hiker: ∂f/∂x = 3 (y fixed at 2), ∂f/∂y = 4 (x fixed at 3).")
        caption = tex(r"f(x, y) = \tfrac{1}{2}x^2 + y^2", 34, MUTED).move_to([-3.0, -3.38, 0])
        east, east_t, east_eq = self.inset(
            [3.95, 1.05, 0], "Walk east · y fixed at 2", BLUE, side, -1.0, 3.6, HX, (r"\frac{\partial f}{\partial x}", "=", "3")
        )
        nor, nor_t, nor_eq = self.inset(
            [3.95, -1.75, 0], "Walk north · x fixed at 3", PINK, north, -1.6, 3.0, HY, (r"\frac{\partial f}{\partial y}", "=", "4")
        )
        self.slice_y = m.DashedLine(mp(HX, -MAP_Y), mp(HX, MAP_Y), dash_length=0.12).set_stroke(PINK, width=3).set_z_index(2)
        walk_x = m.DoubleArrow(self.P + 0.75 * m.LEFT, self.P + 0.75 * m.RIGHT, buff=0, color=BLUE, stroke_width=5, tip_length=0.18).set_z_index(5)
        walk_y = m.DoubleArrow(self.P + 0.75 * m.DOWN, self.P + 0.75 * m.UP, buff=0, color=PINK, stroke_width=5, tip_length=0.18).set_z_index(5)

        self.say(
            "[Its slope, three, is the partial derivative of f with respect to x]: [x changes, while y stays fixed.]",
            m.AnimationGroup(
                self.new_header("PARTIAL DERIVATIVES", "One direction at a time"),
                m.FadeOut(self.legend),
                m.FadeOut(self.example),
                m.ReplacementTransform(self.formula, caption),
                m.FadeIn(east, shift=0.2 * m.LEFT),
                m.Create(east_t),
                m.FadeIn(east_eq),
            ),
            m.AnimationGroup(m.GrowFromCenter(walk_x), m.Indicate(east[1], color=BLUE, scale_factor=1.08)),
        )
        self.say(
            "[The curly d means partial]: [only one input moves, and the others stay frozen.]",
            m.Indicate(east_eq[0], color=BLUE, scale_factor=1.3),
            m.ShowPassingFlash(m.Line(mp(-MAP_X, HY), mp(MAP_X, HY)).set_stroke(BLUE, width=8), time_width=0.5, run_time=1.6),
        )
        self.say(
            "[Now hold x fixed, and walk north instead.] [This slice is steeper]: [its partial derivative is four.]",
            m.AnimationGroup(m.FadeOut(walk_x), m.Create(self.slice_y), m.GrowFromCenter(walk_y)),
            m.AnimationGroup(m.FadeIn(nor, shift=0.2 * m.LEFT), m.Create(nor_t)),
            m.AnimationGroup(m.FadeIn(nor_eq), m.Indicate(nor_t, color=PINK, scale_factor=1.15)),
        )
        self.caption = caption
        self.panel = m.VGroup(east, east_t, east_eq, nor, nor_t, nor_eq)
        self.walk_y = walk_y
        self.wait(0.3)

    # ------------------------------------------------------------------ the gradient

    def gradient(self):
        self.next_section(
            "The gradient",
            notes="∇f = (3, 4) at the hiker: steepest ascent, slope 5 (checked over every direction), at right angles to the contour.",
        )
        P = np.array(self.P)
        k = 0.45  # map units per unit of gradient
        corner = mp(HX + k * G[0], HY)
        tip = mp(HX + k * G[0], HY + k * G[1])
        part_x = tip_arrow(P, corner, BLUE, width=5, ratio=0.25).set_z_index(5)
        part_y = tip_arrow(corner, tip, PINK, width=5, ratio=0.25).set_z_index(5)
        three = text("3", 26, BLUE, weight=m.BOLD).next_to(part_x, m.UP, buff=0.1).shift(0.1 * m.RIGHT)
        four = text("4", 26, PINK, weight=m.BOLD).next_to(part_y, m.RIGHT, buff=0.12)
        arrow_g = tip_arrow(P, tip, AMBER, width=6, ratio=0.22).set_z_index(5)
        self.arrow_g = arrow_g

        formula = m.MathTex(
            r"\nabla f", "=", r"\Big(", r"\frac{\partial f}{\partial x}", ",", r"\frac{\partial f}{\partial y}", r"\Big)", "=", r"(3,\ 4)",
            font_size=44, color=INK,
        ).move_to([3.95, 1.75, 0])
        formula[0].set_color(AMBER)
        formula[3].set_color(BLUE)
        formula[5].set_color(PINK)
        formula[8].set_color(AMBER)
        nabla = m.VGroup(tex(r"\nabla", 30, AMBER), text("is called “nabla”", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        nabla.next_to(formula, m.DOWN, buff=0.35)

        theta = m.ValueTracker(0.0)

        def make_facing():
            a = theta.get_value()
            return tip_arrow(P, P + 1.15 * np.array([np.cos(a), np.sin(a), 0.0]), SKY, width=5, ratio=0.25).set_z_index(5)

        facing = m.always_redraw(make_facing)
        read_v = number(3.0, SKY, size=36)
        read_v.add_updater(lambda d: d.set_value(slope_toward(theta.get_value())))
        read = m.VGroup(text("slope facing this way", 24, MUTED), read_v).arrange(m.RIGHT, buff=0.35).move_to([3.95, -0.3, 0])
        steepest = pill("steepest ascent: a slope of 5", AMBER, AMBER_FILL, 24).move_to([3.95, -1.3, 0])
        square = pill("at right angles to the contours", TEAL, TEAL_FILL, 24).move_to([3.95, -2.2, 0])
        lengths = pill("longer where the ground is steeper", AMBER, AMBER_FILL, 24).move_to([3.95, -3.1, 0])

        t = np.array([-G[1], G[0], 0.0]) / STEEP
        g = np.array([G[0], G[1], 0.0]) / STEEP
        a = 0.36
        right = polyline([P + a * t, P + a * t + a * g, P + a * g], INK, width=3.5).set_z_index(5)
        along = m.DashedLine(P - 0.9 * t, P + 0.9 * t, dash_length=0.08).set_stroke(TEAL, width=3).set_z_index(4)

        field = m.VGroup()
        outer, inner = m.VGroup(), m.VGroup()
        for gx in range(-4, 5):
            for gy in range(-2, 3):
                if (gx, gy) in ((0, 0), (int(HX), int(HY))) or (abs(gx) <= 1 and gy == -1):  # the hiker, the floor and its label
                    continue
                v = grad(gx, gy)
                size = float(np.hypot(*v))
                length = float(np.clip(0.11 * size, 0.16, 0.62))
                u = np.array([v[0], v[1], 0.0]) / size
                c = mp(gx, gy)
                arr = tip_arrow(c - 0.5 * length * u, c + 0.5 * length * u, AMBER, width=3, ratio=0.38)
                arr.set_opacity(0.8).set_z_index(1)
                field.add(arr)
                (outer if size >= 4.5 else inner if size <= 2.3 else m.VGroup()).add(arr)
        self.field = field

        self.say(
            "[Stack the two partial derivatives together], [and you get the gradient], [written with an upside-down triangle called nabla.]",
            m.AnimationGroup(
                self.new_header("THE GRADIENT", "Both slopes in one arrow"),
                m.FadeOut(self.panel),
                m.FadeOut(self.slice_x),
                m.FadeOut(self.slice_y),
                m.FadeOut(self.walk_y),
            ),
            m.Write(formula, run_time=1.6),
            m.AnimationGroup(m.FadeIn(nabla, shift=0.15 * m.UP), m.Indicate(formula[0], color=AMBER, scale_factor=1.3)),
        )
        self.say(
            "[Three east], [and four north], [make one arrow.]",
            m.AnimationGroup(m.GrowArrow(part_x), m.FadeIn(three)),
            m.AnimationGroup(m.GrowArrow(part_y), m.FadeIn(four)),
            m.AnimationGroup(m.GrowArrow(arrow_g), m.Indicate(formula[8], color=AMBER, scale_factor=1.25)),
        )
        self.say(
            "[Let's turn the hiker to face different directions.] [Facing east, the ground rises three per step.] [Facing north, four.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(part_x, part_y, three, four)), m.FadeIn(facing), m.FadeIn(read, shift=0.15 * m.UP)),
            m.Indicate(read_v, color=SKY, scale_factor=1.25),
            theta.animate(run_time=1.6).set_value(np.pi / 2),
        )
        self.say(
            "[In between, along the gradient], [it rises five per step]: [steeper than any other way.]",
            theta.animate(run_time=1.2).set_value(G_ANGLE),
            m.Indicate(read_v, color=AMBER, scale_factor=1.3),
            theta.animate(run_time=3.2, rate_func=m.smooth).set_value(G_ANGLE + 2 * np.pi),
        )
        self.stop(facing)
        self.say(
            "[So the gradient points in the direction of steepest ascent], [and its length, five, tells you how steep.]",
            m.AnimationGroup(m.FadeOut(facing), m.Indicate(arrow_g, color=AMBER, scale_factor=1.12)),
            m.FadeIn(steepest, shift=0.15 * m.UP),
        )
        self.say(
            "[It also crosses the contour line at a right angle], [pointing straight up the slope.]",
            m.AnimationGroup(m.Create(along), m.Create(right), m.FadeIn(square, shift=0.15 * m.UP)),
            m.ShowPassingFlash(m.Line(P, tip).set_stroke(INK, width=10), time_width=0.6, run_time=1.2),
        )
        self.say(
            "[Here are the gradients all over the map]: [long where the ground is steep], [short near the valley floor.]",
            m.AnimationGroup(m.FadeOut(along), m.FadeOut(right), m.LaggedStart(*[m.GrowArrow(a) for a in field], lag_ratio=0.03, run_time=2.0)),
            m.AnimationGroup(m.FadeIn(lengths, shift=0.15 * m.UP), m.Indicate(outer, color=AMBER, scale_factor=1.12)),
            m.Indicate(inner, color=GOOD, scale_factor=1.3),
        )
        self.stop(read_v)
        self.panel = m.VGroup(formula, nabla, read, steepest, square, lengths)
        self.wait(0.3)

    # ------------------------------------------------------------------ gradient descent

    def descent(self):
        self.next_section(
            "Gradient descent",
            notes="Eight steps of (x, y) ← (x, y) − 0.3∇f (an illustrative step size): height 8.5 → about 0.01. Callback to Episode 2.",
        )
        P = np.array(self.P)
        k = 0.45

        def downhill(at):
            v = grad(*at)
            return tip_arrow(mp(*at), mp(at[0] - k * v[0], at[1] - k * v[1]), GOOD, width=6, ratio=0.22).set_z_index(5)

        first = downhill(PATH[0])
        minus = m.VGroup(m.MathTex(r"-\nabla f", font_size=48, color=GOOD), text("points downhill", 26, MUTED)).arrange(m.RIGHT, buff=0.3)
        minus.move_to([3.95, 1.75, 0])
        rule = m.MathTex("(x, y)", r"\leftarrow", "(x, y)", "-", r"\eta", r"\,\nabla f", font_size=44, color=INK).move_to([3.95, 1.75, 0])
        rule[4].set_color(GOOD)
        rule[5].set_color(AMBER)
        eta = m.VGroup(tex(r"\eta", 32, GOOD), text("a small step size (the learning rate)", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        eta.next_to(rule, m.DOWN, buff=0.35)
        nearby = m.DashedVMobject(m.Circle(radius=0.62).move_to(P), num_dashes=24).set_stroke(SKY, width=2.5).set_z_index(5)
        nearby_l = text("the slope holds only nearby", 22, SKY, weight=m.SEMIBOLD).next_to(nearby, m.UP, buff=0.12).shift(0.6 * m.LEFT)

        height_v = number(HEIGHT, TEAL, size=36)
        height_v.add_updater(lambda d: d.set_value(f(*coords(self.who.get_center()))))
        height = m.VGroup(text("height", 26, TEAL, weight=m.SEMIBOLD), height_v).arrange(m.RIGHT, buff=0.35).move_to([3.95, 0.05, 0])

        points = [mp(*p) for p in PATH]
        trail = [m.Line(a, b).set_stroke(GOOD, width=4).set_z_index(4) for a, b in zip(points, points[1:])]
        stops = [m.Dot(p, radius=0.055, color=GOOD).set_z_index(4) for p in points[1:]]
        second = downhill(PATH[1])
        flat = m.VGroup(text("at the floor:", 24, GOOD, weight=m.SEMIBOLD), m.MathTex(r"\nabla f = 0", font_size=34, color=GOOD)).arrange(m.RIGHT, buff=0.25)
        flat.move_to([3.95, -0.75, 0])

        def step(i, run_time=1.0):
            return m.AnimationGroup(self.who.animate.move_to(points[i + 1]), m.Create(trail[i]), m.FadeIn(stops[i]), run_time=run_time)

        # Episode 2's ball in its loss valley
        box = card(5.4, 2.0, EDGE, CARD, stroke=2).move_to([3.95, -2.3, 0])
        box_title = text("Episode 2: a ball in the loss valley", 22, MUTED).move_to(box.get_top() + 0.36 * m.DOWN)
        base = box.get_bottom()[1] + 0.25
        mid = box.get_x()

        def bowl(u):
            return np.array([mid + 1.6 * u, base + 0.95 * u**2, 0.0])

        valley2 = polyline([bowl(u) for u in np.linspace(-1, 1, 60)], BAD, width=4)
        roll = m.ValueTracker(-0.92)
        ball = m.always_redraw(lambda: m.Dot(bowl(roll.get_value()) + 0.11 * m.UP, radius=0.11, color=SKY))

        self.say(
            "[To go down, the hiker walks the opposite way]: [along minus the gradient.]",
            m.AnimationGroup(
                self.new_header("GRADIENT DESCENT", "Walk against the gradient"),
                m.FadeOut(self.panel),
                m.FadeOut(self.field),
            ),
            m.AnimationGroup(m.GrowArrow(first), m.FadeIn(minus, shift=0.15 * m.UP), self.arrow_g.animate.set_opacity(0.35)),
        )
        self.say(
            "[Each step moves the position against the gradient], [scaled by a small step size, called eta.]",
            m.AnimationGroup(m.FadeOut(minus, shift=0.15 * m.UP), m.Write(rule, run_time=1.5)),
            m.AnimationGroup(m.FadeIn(eta, shift=0.15 * m.UP), m.Indicate(rule[4], color=GOOD, scale_factor=1.5)),
        )
        self.say(
            "[Small, because in the fog, the slope only holds close to your feet.]",
            m.AnimationGroup(m.Create(nearby), m.FadeIn(nearby_l), m.FadeOut(self.arrow_g)),
        )
        self.say(
            "[Step], [feel the slope again], [step again.]",
            m.AnimationGroup(
                m.FadeOut(nearby), m.FadeOut(nearby_l), m.FadeOut(first),
                m.FadeIn(height, shift=0.15 * m.UP), step(0),
            ),
            m.GrowArrow(second),
            m.AnimationGroup(m.FadeOut(second), step(1)),
        )
        self.say(
            "[After eight steps, the height falls from eight and a half to almost zero]: [the valley floor, where the ground is flat, and the gradient is zero.]",
            m.Succession(*[step(i, run_time=0.5) for i in range(2, STEPS)]),
            m.AnimationGroup(m.Flash(mp(0, 0), color=GOOD, flash_radius=0.35), m.FadeIn(flat, shift=0.15 * m.UP), m.Indicate(height_v, color=GOOD, scale_factor=1.2)),
        )
        self.stop(height_v)
        self.say(
            "[Sound familiar? It's the ball from episode two, rolling down into the valley], [now with two directions instead of one.]",
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(box_title), m.Create(valley2), m.FadeIn(ball), roll.animate(run_time=2.4, rate_func=m.smooth).set_value(0.0)),
            m.AnimationGroup(m.Indicate(self.x_name, color=BLUE, scale_factor=1.25), m.Indicate(self.y_name, color=PINK, scale_factor=1.25)),
        )
        self.stop(ball)
        self.panel = m.VGroup(rule, eta, height, box, box_title, valley2, ball)
        self.flat = flat
        self.wait(0.3)

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section("The big picture", notes="In ML the height is the loss and the position is the parameters; one partial derivative per parameter.")
        w1 = tex("w_1", 34, BLUE).move_to(self.x_name)
        w2 = tex("w_2", 34, PINK).move_to(self.y_name)
        loss_caption = m.VGroup(text("loss", 30, TEAL, weight=m.SEMIBOLD), tex("L(w_1, w_2)", 36, INK)).arrange(m.RIGHT, buff=0.2)
        loss_caption.move_to(self.caption)
        rows = m.VGroup()
        for k, (a, color, b) in enumerate((("height", TEAL, "the loss"), ("position", SKY, "the parameters"))):
            y = 2.05 - 0.6 * k
            rows.add(
                m.VGroup(
                    text(a, 26, color, weight=m.SEMIBOLD).move_to([0, y, 0], aligned_edge=m.RIGHT).set_x(2.75, direction=m.RIGHT),
                    text("→", 26, MUTED).move_to([3.15, y, 0]),
                    text(b, 26, INK).move_to([0, y, 0]).set_x(3.55, direction=m.LEFT),
                )
            )
        for row in rows:  # the right-hand words sit on the left-hand word's baseline
            row[2].shift((baseline(row[0]) - baseline(row[2])) * m.UP)
        vec = m.MathTex(
            r"\nabla L = \begin{pmatrix} \partial L / \partial w_1 \\ \partial L / \partial w_2 \\ \vdots \\ \partial L / \partial w_n \end{pmatrix}",
            font_size=38, color=INK,
        ).move_to([3.95, -0.6, 0])
        vec[0][:2].set_color(AMBER)
        billions = text("n: billions, in a large model", 22, MUTED).next_to(vec, m.DOWN, buff=0.25)
        recipe = pill("find the gradient, step against it", GOOD, GOOD_FILL, 24).move_to([3.95, -3.15, 0])
        later = pill("next: the chain rule · Episode 12", AMBER, AMBER_FILL, 24).move_to([3.95, -3.15, 0])
        lowest = text("lowest loss", 22, GOOD, weight=m.SEMIBOLD).move_to(self.floor_label)
        path_line = polyline([mp(*p) for p in PATH], GOOD, width=9)

        self.say(
            "[In machine learning, the height is the loss], [and the hiker's position is the model's parameters.]",
            m.AnimationGroup(
                self.new_header("THE BIG PICTURE", "Hiking down the loss"),
                m.FadeOut(self.panel),
                m.FadeOut(self.flat),
                m.FadeIn(rows[0], shift=0.15 * m.UP),
                m.FadeTransform(self.caption, loss_caption),
                m.FadeTransform(self.floor_label, lowest),
            ),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.15 * m.UP), m.FadeTransform(self.x_name, w1), m.FadeTransform(self.y_name, w2)),
        )
        self.say(
            "[Our map has just two parameters.] [A large model has billions], [so its gradient is a list of billions of partial derivatives, one per parameter.]",
            m.AnimationGroup(m.Indicate(w1, color=BLUE, scale_factor=1.4), m.Indicate(w2, color=PINK, scale_factor=1.4)),
            m.FadeIn(billions, shift=0.15 * m.UP),
            m.Write(vec, run_time=1.8),
        )
        self.say(
            "[The recipe stays the same]: [find the gradient, and step against it.]",
            m.FadeIn(recipe, shift=0.15 * m.UP),
            m.ShowPassingFlash(path_line, time_width=0.7, run_time=1.6),
        )
        self.say(
            "[Working out billions of slopes at once takes one more idea: the chain rule.]",
            m.FadeTransform(recipe, later),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Slope, gradient, downhill.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Slope", "Gradient", "Downhill"), (BLUE, AMBER, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("The gradient points uphill. To learn, step the other way.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [feel the slope], [build the gradient], [step downhill.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[The gradient points uphill. To learn, step the other way.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
