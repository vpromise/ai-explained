"""AI, Explained · Episode 13 · Optimizers: SGD, Momentum, Adam

Running example: a ball in a long, narrow valley — the loss over two parameters,
L(θ) = ½(0.02·θ₁² + 2·θ₂²), an ill-conditioned quadratic (curvature ratio 100, minimum at 0).
Every path on screen is computed below with numpy from the same start (-10, 1.5):

- plain gradient descent, learning rate 0.95 (just under the limit 2/2 = 1 for the steep
  walls): it zig-zags across and crawls along the floor; 1.05 bounces out of the valley;
- momentum as a running average, m ← βm + (1-β)g, θ ← θ - ηm, β = 0.9, η = 2
  (the same as "heavy ball" momentum with step 0.2);
- Adam (Kingma & Ba, 2014) with β₁ = 0.9, β₂ = 0.999, ε = 1e-8, η = 0.5.

Each optimizer has its own learning rate (picked by hand to suit it); "the bottom" means
staying within 0.3 of the minimum. The mini-batch noise is illustrative: the loss is the
average of 512 per-example bowls ½(θ - cᵢ)ᵀH(θ - cᵢ) whose centres scatter (seeded), so a
mini-batch of 32 gives an exact, noisy gradient whose average over all examples is the true
one. The narration quotes the computed numbers; the asserts keep them honest.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# colours: one per idea, for the whole film
GD_C, GD_FILL = SKY, SKY_FILL  # plain gradient descent and SGD
MOM_C, MOM_FILL = LIME, "#1C2A0E"  # momentum: the running average m
SIZE_C, SIZE_FILL = PINK, PINK_FILL  # adaptive step sizes: the running average v
ADAM_C, ADAM_FILL = VIOLET, VIOLET_FILL  # Adam = momentum + adaptive steps
GRAD_C = AMBER  # gradients

# ---------------------------------------------------------------------------- the valley

A_ALONG, B_ACROSS = 0.02, 2.0  # curvature along the floor and across the walls
H = np.array([A_ALONG, B_ACROSS])
START = np.array([-10.0, 1.5])
STEPS = 60  # the length of every race on screen
ARRIVE_R = 0.3  # "at the bottom": within this distance of the minimum, for good

ETA_GD, ETA_BIG = 0.95, 1.05
ETA_MOM, BETA = 2.0, 0.9
ETA_ADAM, BETA1, BETA2, EPS = 0.5, 0.9, 0.999, 1e-8


def grad(w):
    return H * np.asarray(w, dtype=float)


def run(kind, steps, eta, start=START):
    """The path (steps + 1 points) of one optimizer, and its running averages m per step."""
    w, m_avg, v_avg = start.astype(float).copy(), np.zeros(2), np.zeros(2)
    path, ms = [w.copy()], []
    for t in range(1, steps + 1):
        g = grad(w)
        if kind == "gd":
            w = w - eta * g
        elif kind == "momentum":
            m_avg = BETA * m_avg + (1 - BETA) * g
            w = w - eta * m_avg
        else:  # adam
            m_avg = BETA1 * m_avg + (1 - BETA1) * g
            v_avg = BETA2 * v_avg + (1 - BETA2) * g * g
            m_hat, v_hat = m_avg / (1 - BETA1**t), v_avg / (1 - BETA2**t)
            w = w - eta * m_hat / (np.sqrt(v_hat) + EPS)
        path.append(w.copy())
        ms.append(m_avg.copy())
    return np.array(path), np.array(ms)


def arrival(kind, eta):
    """The first step after which the ball stays within ARRIVE_R of the minimum."""
    dist = np.linalg.norm(run(kind, 1000, eta)[0], axis=1)
    inside = dist < ARRIVE_R
    return next(i for i in range(len(dist)) if inside[i:].all())


GD_PATH, _ = run("gd", STEPS, ETA_GD)
BIG_PATH, _ = run("gd", 5, ETA_BIG)
MOM_PATH, MOM_M = run("momentum", STEPS, ETA_MOM)
ADAM_PATH, _ = run("adam", STEPS, ETA_ADAM)
ARRIVE_GD, ARRIVE_MOM, ARRIVE_ADAM = arrival("gd", ETA_GD), arrival("momentum", ETA_MOM), arrival("adam", ETA_ADAM)

assert 0.28 < np.linalg.norm(GD_PATH[STEPS]) / np.linalg.norm(START) < 0.36  # "about a third of the way to go"
assert np.abs(BIG_PATH[:, 1]).max() > 2.1  # "right out of the valley" (the map ends at ±2.1)
assert ARRIVE_MOM < 60 and ARRIVE_ADAM < 60  # "settle at the bottom in under sixty steps"
assert 175 <= ARRIVE_GD <= 185  # "about a hundred and eighty"

# the zig-zag's first six downhill steps: across, they flip every step; along, they agree
SIX = -np.array([grad(w) for w in GD_PATH[:6]])
assert all(np.sign(SIX[k, 1]) == (-1) ** (k + 1) for k in range(6))  # down, up, down, …
assert np.all(SIX[:, 0] > 0)  # always towards the minimum
assert abs(SIX[:, 1].mean()) < 0.1 * np.abs(SIX[:, 1]).mean()  # "they cancel out"

G0 = grad(START)
RATIO = abs(G0[1]) / abs(G0[0])
assert round(RATIO) == 15  # "fifteen times bigger"
ADAM_FIRST = ADAM_PATH[1] - START
assert np.allclose(np.abs(ADAM_FIRST), ETA_ADAM, rtol=1e-6)  # Adam's first step: the same size both ways
assert round(1 / (1 - BETA1)) == 10 and round(1 / (1 - BETA2)) == 1000  # "about ten steps", "about a thousand"

# bias correction for a steady gradient of 1: the raw average starts at 0.1, the corrected one at 1
BIAS_T = np.arange(0, 31)
BIAS_RAW = 1 - BETA1**BIAS_T
assert abs(BIAS_RAW[1] - 0.1) < 1e-12

# ---------------------------------------------------------------------------- mini-batches

N_EX, BATCH, SIGMA_U = 512, 32, 0.15
_rng = np.random.default_rng(13)
U = _rng.normal(0, SIGMA_U, (N_EX, 2))
U -= U.mean(0)  # example i's own bowl has its lowest point at U_i / H; on average, at 0


def batch_grad(w, idx):
    """The exact gradient of the mean loss over the examples idx."""
    return H * np.asarray(w, dtype=float) - U[idx].mean(0)


assert np.allclose(batch_grad(START, np.arange(N_EX)), grad(START))  # all examples: the true gradient
FLOOR = GD_PATH[STEPS]  # where the zig-zag left the ball
BATCHES = [_rng.choice(N_EX, BATCH, replace=False) for _ in range(12)]
FAN = np.array([-batch_grad(FLOOR, b) for b in BATCHES])  # downhill, as each mini-batch sees it
TRUE_DOWN = -grad(FLOOR)


def angle(v):
    return np.degrees(np.arctan2(v[1], v[0]))


FAN_SPREAD = max(abs(angle(f) - angle(TRUE_DOWN)) for f in FAN)
assert FAN_SPREAD > 20  # each one is visibly off …
assert abs(angle(FAN.mean(0)) - angle(TRUE_DOWN)) < 10  # … "but on average, they point the right way"

SGD_PATH = [FLOOR.copy()]
for _ in range(16):
    SGD_PATH.append(SGD_PATH[-1] - ETA_GD * batch_grad(SGD_PATH[-1], _rng.choice(N_EX, BATCH, replace=False)))
SGD_PATH = np.array(SGD_PATH)
assert np.abs(SGD_PATH[:, 1]).max() < 1.0 and SGD_PATH[-1, 0] > FLOOR[0]  # wobbly, but in the valley and onwards

# the hook: one bowl, then the same bowl made gentle; the same learning rate on both
HOOK_C, HOOK_FLAT, HOOK_ETA, HOOK_X0 = 0.3, 0.045, 0.9167, -2.7
HOOK_HOPS = [HOOK_X0 * (1 - 2 * HOOK_C * HOOK_ETA) ** k for k in range(4)]
HOOK_CRAWL = [HOOK_X0 * (1 - 2 * HOOK_FLAT * HOOK_ETA) ** k for k in range(9)]
assert 1 - 2 * HOOK_FLAT * HOOK_ETA > 0.9  # each gentle step covers under a tenth of the rest

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


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


def math(s, size=40, color=INK, colors=None):
    return m.MathTex(s, font_size=size, color=color, tex_to_color_map=colors or {})


def ball(color, radius=0.13):
    core = m.Circle(radius=radius).set_fill(color, opacity=1).set_stroke(BG, width=2.5)
    shine = m.Circle(radius=radius * 0.3).set_fill(INK, opacity=0.85).set_stroke(width=0)
    shine.move_to(core.get_center() + radius * np.array([-0.35, 0.35, 0]))
    return m.VGroup(core, shine).set_z_index(8)


def ray(start, end, color, width=5, opacity=1.0, tip=0.3):
    a = m.Arrow(start, end, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=tip)
    return a.set_opacity(opacity)


def at(path, t):
    """The point t steps along a path (between steps: on the straight hop)."""
    k = int(np.clip(np.floor(t), 0, len(path) - 1))
    if k >= len(path) - 1:
        return path[-1]
    return path[k] + (t - k) * (path[k + 1] - path[k])


def write_png(path, rgb):
    """An RGB uint8 array as a PNG file (stdlib only); written only when it changes."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    if not path.exists() or path.read_bytes() != png:
        path.write_bytes(png)
    return path


def hex_rgb(c):
    return np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


class Valley:
    """The loss over two parameters as a map, seen from above: darker = lower loss. Drawn as
    one image (contour bands and lines computed from the loss), so labels can sit on it."""

    W1, W2, S = (-11.0, 2.5), (-2.1, 2.1), 0.94
    CENTER = np.array([0.0, 0.55, 0.0])
    ACROSS = (0.17, 0.38, 0.62, 0.9, 1.22, 1.58, 1.98, 2.45, 3.0)  # contour half-widths across the floor
    HIGH, LOW, LINE = "#2E3D63", "#080C18", "#5B6E9C"
    IMAGE = Path(__file__).resolve().parent / "valley-map.png"

    def __init__(self):
        mid = np.array([(self.W1[0] + self.W1[1]) / 2, (self.W2[0] + self.W2[1]) / 2, 0])
        self.o = self.CENTER - self.S * mid
        self.width = (self.W1[1] - self.W1[0]) * self.S
        self.height = (self.W2[1] - self.W2[0]) * self.S
        self.render()

    def p(self, w1, w2=None):
        if w2 is None:
            w1, w2 = w1
        return self.o + self.S * np.array([w1, w2, 0.0])

    def inside(self, w):
        return self.W1[0] <= w[0] <= self.W1[1] and self.W2[0] <= w[1] <= self.W2[1]

    def color(self, k):
        """The colour of band k: 0 = outside every contour (high) … len(ACROSS) = the floor."""
        return mix(self.HIGH, self.LOW, k / len(self.ACROSS))

    def render(self, scale=2):
        per_w = self.S * 135 * scale  # pixels per unit of θ
        w, h = int(round(self.width * 135 * scale)), int(round(self.height * 135 * scale))
        x = self.W1[0] + (np.arange(w) + 0.5) / per_w
        y = self.W2[1] - (np.arange(h) + 0.5) / per_w
        X, Y = np.meshgrid(x, y)
        k = np.sqrt(A_ALONG / B_ACROSS)
        r = np.sqrt(Y**2 + (k * X) ** 2)  # the half-width across of the contour through each point
        slope = np.sqrt((k * k * X) ** 2 + Y**2) / np.maximum(r, 1e-9)  # |grad r|
        outside, line = np.zeros_like(r), np.zeros_like(r)
        for c in self.ACROSS:
            d = (r - c) / np.maximum(slope, 1e-9) * per_w  # signed distance to the contour, in pixels
            outside += np.clip(d + 0.5, 0, 1)
            line = np.maximum(line, np.clip(1.1 * scale + 0.5 - np.abs(d), 0, 1))
        n = len(self.ACROSS)
        lo, hi = hex_rgb(self.LOW), hex_rgb(self.HIGH)
        rgb = lo + (hi - lo) * (outside / n)[..., None]
        rgb += (hex_rgb(self.LINE) - rgb) * (0.5 * line)[..., None]
        write_png(self.IMAGE, np.clip(np.round(rgb), 0, 255).astype(np.uint8))

    def bands(self):
        image = m.ImageMobject(str(self.IMAGE))
        image.height = self.height
        image.move_to(self.o + self.S * np.array([(self.W1[0] + self.W1[1]) / 2, (self.W2[0] + self.W2[1]) / 2, 0]))
        frame = m.Rectangle(width=self.width, height=self.height).move_to(image.get_center()).set_stroke(EDGE, width=2).set_fill(opacity=0)
        return image.set_z_index(-5), frame.set_z_index(-4)

    def segment(self, a, b):
        """The part of the hop a → b inside the map."""
        if self.inside(b):
            return b
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.inside(a + mid * (b - a)) else (lo, mid)
        return a + lo * (b - a)


def walked(valley, path, t, color, width=3.5, dots=True, opacity=1.0):
    """The path walked after t steps, with a small dot at every step."""
    k = int(np.clip(np.floor(t), 0, len(path) - 1))
    pts = [valley.p(q) for q in path[: k + 1]]
    if t > k + 1e-6 and k < len(path) - 1:
        pts.append(valley.p(at(path, t)))
    if len(pts) < 2:
        pts.append(pts[-1] + 0.001 * m.RIGHT)
    group = m.VGroup(polyline(pts, color, width, opacity))
    if dots:
        group.add(*[m.Dot(valley.p(q), radius=0.034, color=color).set_opacity(opacity) for q in path[1 : k + 1]])
    return group.set_z_index(4)


def trail(valley, path, tracker, color, **kw):
    """The path walked so far, following the tracker."""
    return m.always_redraw(lambda: walked(valley, path, tracker.get_value(), color, **kw))


def rider(valley, path, tracker, color):
    """A ball that follows the tracker along a path."""
    b = ball(color).move_to(valley.p(path[0]))
    b.add_updater(lambda mob: mob.move_to(valley.p(at(path, tracker.get_value()))))
    return b


def counter(tracker, position):
    number = m.DecimalNumber(0, num_decimal_places=0, mob_class=Mono, font_size=30, color=INK)
    number.add_updater(lambda d: d.set_value(int(np.floor(tracker.get_value() + 1e-6))))
    row = m.VGroup(text("step", 24, MUTED, weight=m.SEMIBOLD), number).arrange(m.RIGHT, buff=0.2)
    return row.move_to(position)


def stop(*mobs):
    for mob in mobs:
        mob.clear_updaters()


def legend_row(symbol, words, size=22):
    return m.VGroup(symbol, text(words, size, MUTED)).arrange(m.RIGHT, buff=0.25)


# ---------------------------------------------------------------------------- the episode


class Ep13Optimizers(Episode):
    number = 13

    def lesson(self):
        self.V = Valley()
        self.hook()
        self.valley()
        self.descent()
        self.minibatches()
        self.momentum()
        self.adaptive()
        self.adam()
        self.race()
        self.practice()
        self.recap()

    def column(self, *mobs, x=-6.35):
        """Left-align a column of formula and notes at x."""
        for mob in mobs:
            mob.align_to([x, 0, 0], m.LEFT)
        return mobs

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Gradient descent from Episode 2; on a gentle slope its steps shrink; three upgrades.")
        sx, base = 1.15, -0.55

        def bowl_point(x, c):
            return np.array([sx * x, base + c * (sx * x) ** 2, 0.0])

        def bowl(c):
            xs = np.linspace(-2.75, 2.75, 120)
            return polyline([bowl_point(x, c) for x in xs], MUTED, width=4)

        def on(x, c):
            return bowl_point(x, c) + 0.15 * m.UP

        curve = bowl(HOOK_C)
        b = ball(GD_C, 0.15).move_to(on(HOOK_HOPS[0], HOOK_C))
        hops = [
            m.MoveAlongPath(b, m.ArcBetweenPoints(on(x0, HOOK_C), on(x1, HOOK_C), angle=-0.9), run_time=0.55)
            for x0, x1 in zip(HOOK_HOPS, HOOK_HOPS[1:])
        ]
        name = pill("gradient descent", GD_C, GD_FILL, 26).move_to([0, -1.35, 0])
        flat = bowl(HOOK_FLAT)
        crawl = [
            m.MoveAlongPath(b, m.ArcBetweenPoints(on(x0, HOOK_FLAT), on(x1, HOOK_FLAT), angle=-0.6), run_time=0.4)
            for x0, x1 in zip(HOOK_CRAWL, HOOK_CRAWL[1:])
        ]
        slow = text("tiny steps: a crawl", 24, BAD, weight=m.SEMIBOLD).move_to([0, 0.35, 0])
        lead = text("Three upgrades:", 30, INK, weight=m.BOLD)
        chips = [chip(w, c, f, width=wd, size=28) for w, c, f, wd in (("mini-batches", GD_C, GD_FILL, 2.9), ("momentum", MOM_C, MOM_FILL, 2.6), ("Adam", ADAM_C, ADAM_FILL, 1.6))]
        row = m.VGroup(lead, *chips).arrange(m.RIGHT, buff=0.4).move_to([0, -2.75, 0])
        lead.align_to(chips[0][1], m.DOWN)

        self.say(
            "[In episode two, a model learned by walking downhill on its loss], [one small step at a time]: [gradient descent.]",
            m.AnimationGroup(m.Create(curve), m.FadeIn(b, scale=0.5)),
            m.Succession(*hops),
            m.FadeIn(name, shift=0.2 * m.UP),
        )
        self.say(
            "[But where the slope is gentle, the steps shrink], [and learning slows to a crawl.]",
            m.AnimationGroup(m.Transform(curve, flat), b.animate.move_to(on(HOOK_CRAWL[0], HOOK_FLAT))),
            m.AnimationGroup(m.Succession(*crawl), m.FadeIn(slow, shift=0.2 * m.UP)),
        )
        self.say(
            "[In this episode, three upgrades]: [mini-batches], [momentum], [and Adam.]",
            m.FadeIn(lead, shift=0.2 * m.UP),
            *[m.FadeIn(c, shift=0.2 * m.UP) for c in chips],
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the valley

    def valley(self):
        self.next_section(
            "The valley",
            notes="Illustrative loss: L = ½(0.02·θ₁² + 2·θ₂²), curvature ratio 100, minimum at 0. Bands are contours; darker = lower.",
        )
        V = self.V
        self.map_bands, self.map_frame = V.bands()
        swatches = m.VGroup(*[m.Square(side_length=0.32).set_fill(V.color(k), opacity=1).set_stroke(EDGE, width=1.5) for k in (0, 2, 4, 6, 8)])
        swatches.arrange(m.RIGHT, buff=0.08)
        key = m.VGroup(text("higher loss", 22, MUTED), swatches, text("lower loss", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        key.move_to([0, -2.15, 0])
        floor = m.Line(V.p(-11, 0), V.p(2.5, 0)).set_stroke(GRAD_C, width=9)
        walls = m.DoubleArrow(V.p(-1.2, -1.95), V.p(-1.2, 1.95), buff=0, color=INK, stroke_width=4, max_tip_length_to_length_ratio=0.08)
        walls_label = text("steep walls", 24, INK, weight=m.SEMIBOLD).next_to(walls, m.RIGHT, buff=0.2).shift(1.1 * m.UP)
        along = m.Arrow(V.p(-9.0, 0), V.p(-2.4, 0), buff=0, color=INK, stroke_width=4, max_tip_length_to_length_ratio=0.06)
        along_label = text("a gently sloping floor", 24, INK, weight=m.SEMIBOLD).next_to(along, m.UP, buff=0.16)
        self.marks = m.VGroup(walls, walls_label, along, along_label)
        self.start_ball = ball(GD_C).move_to(V.p(START))
        lowest = m.VGroup(
            m.Circle(radius=0.16).set_stroke(INK, width=3).move_to(V.p(0, 0)),
            m.Dot(V.p(0, 0), radius=0.06, color=INK),
        ).set_z_index(3)
        lowest_label = text("lowest point", 22, INK, weight=m.SEMIBOLD).next_to(lowest, m.DOWN, buff=0.7)
        self.lowest, self.lowest_label = lowest, lowest_label
        self.key = key

        self.say(
            "[Picture the loss over two parameters as a map, seen from above]: [the darker the color, the lower the loss.]",
            m.AnimationGroup(
                self.new_header("THE PROBLEM", "A ball in a narrow valley"),
                m.FadeIn(self.map_bands, run_time=1.2),
                m.Create(self.map_frame),
            ),
            m.FadeIn(key, shift=0.2 * m.UP),
        )
        self.say(
            "[Take this valley, for example: long and narrow.] [Its walls are steep], [but its floor slopes gently.]",
            m.ShowPassingFlash(floor, time_width=0.6, run_time=1.5),
            m.AnimationGroup(m.GrowFromCenter(walls), m.FadeIn(walls_label, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.GrowArrow(along), m.FadeIn(along_label, shift=0.15 * m.DOWN)),
        )
        self.say(
            "[We drop a ball here], [and want it to reach the lowest point, over here.]",
            m.FadeIn(self.start_ball, scale=0.4),
            m.AnimationGroup(m.FadeIn(lowest, scale=0.5), m.FadeIn(lowest_label, shift=0.15 * m.UP)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ plain gradient descent

    def descent(self):
        self.next_section(
            "Gradient descent",
            notes="Learning rate 0.95: the step across lands on the other wall every time (factor −0.9); along the floor the distance shrinks 1.9% a step. After 60 steps 31% of the distance is left. Learning rate 1.05 diverges.",
        )
        V = self.V
        up = ray(V.p(START), V.p(START + 0.18 * G0), GRAD_C, tip=0.35)
        up_label = text("gradient: uphill", 22, GRAD_C, weight=m.SEMIBOLD).next_to(up, m.RIGHT, buff=0.2)
        step = ray(V.p(START), V.p(START - ETA_GD * G0), GD_C, tip=0.12)
        step_label = text("the step", 22, GD_C, weight=m.SEMIBOLD).next_to(step.get_center(), m.RIGHT, buff=0.3)

        rule = math(r"\theta \leftarrow \theta - \eta\, g", 48, INK, {"g": GRAD_C}).move_to([-4.6, -2.45, 0])
        terms = m.VGroup(
            legend_row(math(r"\theta", 32, INK), "the parameters: where the ball is"),
            legend_row(math(r"\eta", 32, INK), "the learning rate: how big a step"),
            legend_row(math(r"g", 32, GRAD_C), "the gradient: which way is uphill"),
        ).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        for row in terms:
            row[1].next_to(terms[0][0], m.RIGHT, buff=0.3).set_y(row[0].get_y())
            row[1].align_to(terms[0][1], m.LEFT)
        terms.move_to([0.6, -2.5, 0]).align_to([-2.3, 0, 0], m.LEFT)

        t = m.ValueTracker(0)
        path = trail(V, GD_PATH, t, GD_C)
        self.remove(self.start_ball)
        rolling = rider(V, GD_PATH, t, GD_C)
        steps = counter(t, [5.35, 2.17, 0])
        end = V.p(GD_PATH[STEPS])
        gap = m.DashedLine(end + 0.2 * m.RIGHT, V.p(0, 0) + 0.24 * m.LEFT, dash_length=0.1).set_stroke(BAD, width=3.5)
        gap_label = text("a third of the way to go", 22, BAD, weight=m.SEMIBOLD).next_to(gap, m.UP, buff=0.32).shift(0.3 * m.LEFT)
        big = [START]
        for a, b in zip(BIG_PATH, BIG_PATH[1:]):
            big.append(V.segment(a, b))
            if not V.inside(b):
                break
        bounce = m.DashedVMobject(polyline([V.p(q) for q in big], BAD, width=4), num_dashes=46).set_z_index(5)
        too_big = pill("learning rate too big: it flies out", BAD, BAD_FILL, 22).move_to([4.25, -2.45, 0])
        done = walked(V, GD_PATH, STEPS, GD_C)

        self.say(
            "[The gradient at the ball points straight uphill]: [here, mostly up the wall.]",
            m.AnimationGroup(
                self.new_header("GRADIENT DESCENT", "Always straight downhill", GD_C, GD_FILL),
                m.FadeOut(self.marks),
                m.FadeOut(self.key),
                m.GrowArrow(up),
                m.FadeIn(up_label, shift=0.15 * m.LEFT),
            ),
            m.Indicate(up, color=GRAD_C, scale_factor=1.3),
        )
        self.add(rolling)
        self.say(
            "[So gradient descent steps the opposite way], [scaled by the learning rate.]",
            m.AnimationGroup(m.GrowArrow(step), m.FadeIn(step_label, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.Write(rule), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.UP) for r in terms], lag_ratio=0.3)),
        )
        self.say(
            "[But downhill points across the valley, not along it.]",
            m.Indicate(m.VGroup(step, step_label), color=GD_C, scale_factor=1.08),
        )
        self.add(path)
        self.say(
            "[So the ball zig-zags from wall to wall], [and only creeps along the floor.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (up, up_label, step, step_label)],
                m.FadeIn(steps),
                t.animate(run_time=3.6, rate_func=m.linear).set_value(14),
            ),
            t.animate(run_time=4.2, rate_func=m.linear).set_value(STEPS),
        )
        stop(path, rolling, steps[1])
        self.remove(path)
        self.add(done)
        self.say(
            "[After sixty steps], [it still has about a third of the way to go.]",
            m.Indicate(steps, color=GD_C, scale_factor=1.15),
            m.AnimationGroup(m.Create(gap), m.FadeIn(gap_label, shift=0.15 * m.UP)),
        )
        self.say(
            "[A bigger learning rate won't fix it]: [the ball bounces higher and higher, right out of the valley.]",
            m.AnimationGroup(m.FadeOut(terms), m.FadeIn(too_big, shift=0.2 * m.UP), m.Transform(done, walked(V, GD_PATH, STEPS, GD_C, opacity=0.25))),
            m.Create(bounce, run_time=2.2),
        )
        self.wait(0.3)
        self.gd = dict(path=done, ball=rolling, steps=steps)
        self.play(
            *[m.FadeOut(mob) for mob in (gap, gap_label, bounce, too_big, steps, rule)],
            m.Transform(done, walked(V, GD_PATH, STEPS, GD_C)),
            run_time=0.7,
        )

    # ------------------------------------------------------------------ mini-batches

    def minibatches(self):
        self.next_section(
            "Mini-batches",
            notes="Illustrative data: 512 examples, each with its own bowl (seeded); mini-batches of 32. The arrows are downhill directions at the ball, drawn longer than a step. SGD: Robbins & Monro, 1951.",
        )
        V = self.V
        cells = m.VGroup(*[m.Square(side_length=0.08).set_fill(FAINT, opacity=1).set_stroke(width=0) for _ in range(N_EX)])
        cells.arrange_in_grid(rows=16, cols=32, buff=0.03).move_to([-4.75, -2.6, 0])
        chosen = m.VGroup(*[cells[i] for i in BATCHES[0]])
        square = lambda c: m.Square(side_length=0.2).set_fill(c, opacity=1).set_stroke(width=0)  # noqa: E731
        all_label = legend_row(square(FAINT), "all 512 training examples")
        some_label = legend_row(square(GD_C), "a random 32: one mini-batch")
        labels = m.VGroup(all_label, some_label).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([0, -2.3, 0]).align_to([-2.65, 0, 0], m.LEFT)
        millions = text("in real training: often millions", 22, MUTED).next_to(labels, m.DOWN, buff=0.3).align_to(labels, m.LEFT)

        def swatch(color, width, opacity=1.0):
            return ray(m.ORIGIN, 0.6 * m.RIGHT, color, width=width, opacity=opacity, tip=0.35)

        arrows_key = m.VGroup(
            legend_row(swatch(GRAD_C, 6), "the true gradient"),
            legend_row(swatch(GRAD_C, 3, 0.55), "one per mini-batch"),
            legend_row(swatch(INK, 3.5), "their average"),
        ).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([0, -2.5, 0]).align_to([2.75, 0, 0], m.LEFT)

        here = V.p(FLOOR)
        k = 1.05 / (V.S * np.linalg.norm(TRUE_DOWN))  # display scale: the true arrow 1.05 long
        noisy = m.VGroup(*[ray(here, V.p(FLOOR + k * f), GRAD_C, width=3, opacity=0.55, tip=0.18) for f in FAN]).set_z_index(6)
        true = ray(here, V.p(FLOOR + k * TRUE_DOWN), GRAD_C, width=6, tip=0.25).set_z_index(7)
        mean = ray(here, V.p(FLOOR + k * FAN.mean(0)), INK, width=3.5, tip=0.25).set_z_index(7)

        sgd = m.VGroup(pill("SGD", GD_C, GD_FILL, 24), text("stochastic gradient descent", 22, INK), text("Robbins & Monro, 1951", 20, MUTED))
        sgd.arrange(m.DOWN, buff=0.18).move_to([4.6, -2.5, 0])
        t = m.ValueTracker(0)
        wobble = trail(V, SGD_PATH, t, GD_C, width=3)
        ball_now = self.gd["ball"]

        self.say(
            "[There's a second problem.] [The true gradient averages over every training example], [and there could be millions.]",
            self.new_header("MINI-BATCHES", "A gradient from a small sample", GD_C, GD_FILL),
            m.AnimationGroup(m.FadeIn(cells, lag_ratio=0.002, run_time=1.4), m.FadeIn(all_label, shift=0.15 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[c.animate(rate_func=m.there_and_back).set_fill(GRAD_C) for c in cells], lag_ratio=0.002, run_time=1.4), m.FadeIn(millions)),
        )
        self.say(
            "[So each step uses a small random sample instead, a mini-batch], [of, for example, thirty-two examples.]",
            m.FadeIn(some_label, shift=0.15 * m.UP),
            m.LaggedStart(*[c.animate.set_fill(GD_C) for c in chosen], lag_ratio=0.04, run_time=1.2),
        )
        self.say(
            "[Each mini-batch gives a slightly different gradient]: [a noisy estimate of the true one.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in noisy], lag_ratio=0.12, run_time=1.8), m.FadeIn(arrows_key[1])),
            m.AnimationGroup(m.GrowArrow(true), m.FadeIn(arrows_key[0])),
        )
        self.say(
            "[Each one is a little off], [but on average, they point the right way.]",
            m.LaggedStart(*[m.Indicate(a, color=GRAD_C, scale_factor=1.05) for a in noisy], lag_ratio=0.08),
            m.AnimationGroup(m.TransformFromCopy(noisy, mean), m.FadeIn(arrows_key[2])),
        )
        self.add(wobble)
        ball_now.add_updater(lambda mob: mob.move_to(V.p(at(SGD_PATH, t.get_value()))))
        self.say(
            "[Gradient descent with mini-batches is called stochastic gradient descent, or SGD.] [Its steps are cheap, but wobbly.]",
            m.AnimationGroup(m.FadeOut(arrows_key), m.LaggedStart(*[m.FadeIn(x, shift=0.15 * m.UP) for x in sgd], lag_ratio=0.25)),
            m.AnimationGroup(m.FadeOut(m.VGroup(noisy, true, mean)), t.animate(run_time=3.0, rate_func=m.linear).set_value(len(SGD_PATH) - 1)),
        )
        stop(wobble, ball_now)
        self.wait(0.3)
        self.play(*[m.FadeOut(mob) for mob in (cells, labels, millions, sgd, wobble, ball_now)], run_time=0.7)

    # ------------------------------------------------------------------ momentum

    def momentum(self):
        self.next_section(
            "Momentum",
            notes="m ← βm + (1−β)g, θ ← θ − ηm with β = 0.9, η = 2 (heavy-ball momentum, Polyak 1964). Right: the zig-zag's first six steps, across and along. The lime arrow on the ball is its next step.",
        )
        V = self.V
        cite = text("Polyak, 1964", 20, MUTED).move_to([0, -1.75, 0])
        formula = math(r"m \leftarrow \beta\, m + (1-\beta)\, g", 46, INK, {"m": MOM_C, "g": GRAD_C}).move_to([0, -2.25, 0])
        what = text("m: a running average of gradients", 22, MUTED).move_to([0, -2.8, 0])
        keep = text("β = 0.9: keep 90% of m, add 10% of the new g", 22, INK, t2c={"β = 0.9": MOM_C}).move_to([0, -3.22, 0])
        self.column(cite, formula, what, keep)

        def signs(row_y, vectors, axis, label, verdict, color):
            name = text(label, 24, MUTED, weight=m.SEMIBOLD).move_to([0.75, row_y, 0])
            arrows = m.VGroup()
            for j, v in enumerate(vectors):
                c = np.array([1.6 + 0.55 * j, row_y, 0])
                d = (np.sign(v) * 0.24) * (m.UP if axis == 1 else m.RIGHT)
                arrows.add(ray(c - d, c + d, GRAD_C, width=4, tip=0.4))
            result = text(verdict, 24, color, weight=m.BOLD).move_to([5.5, row_y, 0])
            return m.VGroup(name, arrows, result)

        across = signs(-2.25, SIX[:, 1], 1, "across", "cancel out", MUTED)
        along = signs(-3.05, SIX[:, 0], 0, "along", "add up", MOM_C)
        steps_note = text("the zig-zag's first six steps", 20, MUTED).move_to([2.95, -1.75, 0])

        t = m.ValueTracker(0)
        path = trail(V, MOM_PATH, t, MOM_C)
        rolling = rider(V, MOM_PATH, t, MOM_C)

        def velocity():
            s = t.get_value()
            k = int(np.clip(np.floor(s), 0, STEPS - 1))
            here = V.p(at(MOM_PATH, s))
            nxt = V.p(MOM_PATH[k + 1]) - V.p(MOM_PATH[k])
            hidden = np.linalg.norm(nxt) < 0.12 or s >= STEPS - 0.5
            if hidden:  # same shape, invisible: become() between different shapes nests copies
                nxt = 0.3 * m.RIGHT
            return ray(here, here + nxt, MOM_C, width=4, tip=0.3, opacity=0.0 if hidden else 1.0).set_z_index(7)

        arrow_v = m.always_redraw(velocity)
        steps = counter(t, [5.35, 2.17, 0])

        self.say(
            "[The first upgrade is momentum.] [Picture a heavy ball that keeps rolling the way it was already going.]",
            m.AnimationGroup(
                self.new_header("UPGRADE 1 · MOMENTUM", "A heavy ball keeps rolling", MOM_C, MOM_FILL),
                m.Transform(self.gd["path"], walked(V, GD_PATH, STEPS, GD_C, opacity=0.22)),
                m.FadeOut(self.lowest_label),
            ),
            m.FadeIn(rolling, scale=0.4),
        )
        self.say(
            "[Instead of following only the latest gradient], [it steps along a running average of recent ones.]",
            m.Write(formula),
            m.AnimationGroup(m.FadeIn(what, shift=0.15 * m.UP), m.FadeIn(cite)),
        )
        self.say(
            "[With the usual setting], [each step keeps ninety percent of the old direction, and adds ten percent of the new gradient.]",
            m.Indicate(formula, color=MOM_C, scale_factor=1.06),
            m.FadeIn(keep, shift=0.15 * m.UP),
        )
        self.say(
            "[Across the valley, the gradients keep flipping, so they cancel out.] [Along the floor, they agree, so they add up.]",
            m.AnimationGroup(m.FadeIn(steps_note), m.FadeIn(across[0]), m.LaggedStart(*[m.GrowArrow(a) for a in across[1]], lag_ratio=0.2), m.FadeIn(across[2], shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(along[0]), m.LaggedStart(*[m.GrowArrow(a) for a in along[1]], lag_ratio=0.2), m.FadeIn(along[2], shift=0.15 * m.LEFT)),
        )
        self.add(path, arrow_v)
        self.say(
            "[The ball builds up speed along the floor], [and settles at the bottom in under sixty steps.]",
            m.AnimationGroup(m.FadeIn(steps), t.animate(run_time=3.4, rate_func=m.linear).set_value(30)),
            t.animate(run_time=3.0, rate_func=m.linear).set_value(STEPS),
        )
        stop(path, rolling, arrow_v, steps[1])
        self.wait(0.3)
        self.play(*[m.FadeOut(mob) for mob in (formula, what, keep, cite, across, along, steps_note, path, rolling, arrow_v, steps)], run_time=0.7)

    # ------------------------------------------------------------------ adaptive step sizes

    def adaptive(self):
        self.next_section(
            "Adaptive steps",
            notes="v ← β₂v + (1−β₂)g², step ∝ g/√v, as in RMSprop (Tieleman & Hinton, 2012, Coursera lecture 6.5). At the start |g| across = 3.0, along = 0.2; on the first step g/√v is exactly ±1 in both directions.",
        )
        V = self.V
        plain = ray(V.p(START), V.p(START - ETA_GD * G0), GD_C, width=4, opacity=0.7, tip=0.12)
        plain_label = text("plain step", 22, GD_C, weight=m.SEMIBOLD).next_to(plain.get_center(), m.RIGHT, buff=0.3)
        scaled = ray(V.p(START), V.p(START + ADAM_FIRST), SIZE_C, width=6, tip=0.3).set_z_index(7)
        scaled_label = text("scaled step", 22, SIZE_C, weight=m.BOLD).next_to(scaled.get_end(), m.RIGHT, buff=0.2)
        b = ball(SIZE_C).move_to(V.p(START))

        cite = text("as in RMSprop · Tieleman & Hinton, 2012", 20, MUTED).move_to([0, -1.75, 0])
        formula = math(r"v \leftarrow \beta_2\, v + (1-\beta_2)\, g^2", 46, INK, {"v": SIZE_C, "g": GRAD_C}).move_to([0, -2.3, 0])
        what = text("v: the typical size of g, squared", 22, MUTED).move_to([0, -2.85, 0])
        divide = text("then divide each step by √v", 22, INK, t2c={"√v": SIZE_C}).move_to([0, -3.25, 0])
        self.column(cite, formula, what, divide)

        left, unit = 1.75, 1.0
        ys = (-2.25, -3.0)

        def bar(value, y, color):
            return m.Rectangle(width=max(unit * value, 0.04), height=0.34).set_fill(color, opacity=1).set_stroke(width=0).move_to([left, y, 0], aligned_edge=m.LEFT)

        def number(s, b):
            return code(s, 24, INK).next_to(b, m.RIGHT, buff=0.18)

        names = m.VGroup(text("across", 24, MUTED, weight=m.SEMIBOLD).move_to([0.95, ys[0], 0]), text("along", 24, MUTED, weight=m.SEMIBOLD).move_to([0.95, ys[1], 0]))
        before = [bar(abs(G0[1]), ys[0], GRAD_C), bar(abs(G0[0]), ys[1], GRAD_C)]
        before_n = [number(f"{abs(G0[1]):.1f}", before[0]), number(f"{abs(G0[0]):.1f}", before[1])]
        after = [bar(1.0, ys[0], SIZE_C), bar(1.0, ys[1], SIZE_C)]
        after_n = [number("1.0", after[0]), number("1.0", after[1])]
        title = text("gradient size at the start", 20, MUTED).move_to([3.2, -1.75, 0])
        title_after = text("after dividing by √v", 20, SIZE_C).move_to([3.2, -1.75, 0])
        ratio = text(f"{round(RATIO)}×", 30, GRAD_C, weight=m.BOLD).next_to(before_n[0], m.RIGHT, buff=0.45)

        self.say(
            "[The second upgrade: give every parameter its own step size.]",
            m.AnimationGroup(
                self.new_header("UPGRADE 2 · ADAPTIVE STEPS", "Every parameter, its own step size", SIZE_C, SIZE_FILL),
                m.FadeOut(self.gd["path"]),
                m.FadeIn(b, scale=0.4),
                m.GrowArrow(plain),
                m.FadeIn(plain_label),
            ),
        )
        self.say(
            "[At the start, the gradient across the valley is fifteen times bigger than the one along it.]",
            m.AnimationGroup(m.FadeIn(names), m.FadeIn(title), *[m.GrowFromEdge(x, m.LEFT) for x in before], *[m.FadeIn(n) for n in before_n], m.FadeIn(ratio, scale=1.3)),
        )
        self.say(
            "[So we keep a running average of each parameter's squared gradients], [called v.]",
            m.AnimationGroup(m.Write(formula), m.FadeIn(cite)),
            m.FadeIn(what, shift=0.15 * m.UP),
        )
        self.say(
            "[Dividing each step by the square root of v evens things out]: [smaller steps across, bigger steps along.]",
            m.AnimationGroup(
                m.FadeIn(divide, shift=0.15 * m.UP),
                m.FadeOut(ratio),
                m.FadeTransform(title, title_after),
                *[m.Transform(x, y) for x, y in zip(before, after)],
                *[m.Transform(x, y) for x, y in zip(before_n, after_n)],
            ),
            m.AnimationGroup(m.GrowArrow(scaled), m.FadeIn(scaled_label), plain.animate.set_opacity(0.35), plain_label.animate.set_opacity(0.5)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ Adam

    def adam(self):
        self.next_section(
            "Adam",
            notes="Adam: Kingma & Ba, 2014 (arXiv 1412.6980; ICLR 2015). Bias correction: for a steady gradient of 1, m after t steps is 1 − 0.9^t (0.1 after one step); m̂ is 1 from the start. A running average with β remembers about 1/(1−β) steps: 10 for β₁ = 0.9, 1,000 for β₂ = 0.999.",
        )
        rows = [
            (math(r"m \leftarrow \beta_1\, m + (1-\beta_1)\, g", 42, INK, {"m": MOM_C, "g": GRAD_C}), "momentum: a running average of gradients"),
            (math(r"v \leftarrow \beta_2\, v + (1-\beta_2)\, g^2", 42, INK, {"v": SIZE_C, "g": GRAD_C}), "size: a running average of squared gradients"),
            (math(r"\hat{m} = \frac{m}{1-\beta_1^{\,t}} \qquad \hat{v} = \frac{v}{1-\beta_2^{\,t}}", 38, INK), "bias correction (t counts the steps)"),
            (math(r"\theta \leftarrow \theta - \eta\, \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}", 44, INK), "the step: direction from m, size from v"),
        ]
        ys = (1.95, 0.6, -0.85, -2.5)
        lines = m.VGroup()
        for (f, note), y in zip(rows, ys):
            f.move_to([-3.6, y, 0]).align_to([-6.5, 0, 0], m.LEFT)
            label = text(note, 22, MUTED).next_to(f, m.DOWN, buff=0.2).align_to(f, m.LEFT)
            lines.add(m.VGroup(f, label))

        chart = axes([0, 30, 5], [0, 1.2, 0.5], 4.4, 2.0).move_to([3.4, 0.9, 0])
        chart_title = text("for a steady gradient g = 1:", 22, MUTED).next_to(chart, m.UP, buff=0.45)
        raw = polyline([chart.c2p(t, v) for t, v in zip(BIAS_T, BIAS_RAW)], MUTED, width=4)
        fixed = polyline([chart.c2p(t, 1.0) for t in BIAS_T[1:]], MOM_C, width=4.5)
        raw_label = text("m starts too small", 22, MUTED).move_to(chart.c2p(16, 0.42))
        fixed_label = m.VGroup(math(r"\hat{m}", 30, MOM_C), text("is right from step 1", 22, MOM_C)).arrange(m.RIGHT, buff=0.15)
        fixed_label.next_to(chart.c2p(15, 1.0), m.UP, buff=0.15)
        t_label = text("step t", 22, MUTED).next_to(chart.x_axis.get_end(), m.RIGHT, buff=0.15)

        def setting(label, color, fill, note):
            return m.VGroup(pill(label, color, fill, 24), text(note, 20, MUTED)).arrange(m.DOWN, buff=0.16)

        defaults = m.VGroup(
            setting("β₁ = 0.9", MOM_C, MOM_FILL, "about 10 steps"),
            setting("β₂ = 0.999", SIZE_C, SIZE_FILL, "about 1,000 steps"),
            setting("ε = 10⁻⁸", MUTED, CARD, "tiny"),
        ).arrange(m.RIGHT, buff=0.4, aligned_edge=m.UP).move_to([3.4, -1.55, 0])
        defaults_label = text("the usual defaults: how far back each looks", 22, MUTED).next_to(defaults, m.UP, buff=0.25)
        cite = text("Kingma & Ba, 2014", 22, MUTED).move_to([3.4, -2.95, 0])
        eps = lines[3][0]

        self.say(
            "[Put momentum and adaptive steps together, and you get Adam], [published in twenty fourteen.]",
            self.new_header("ADAM", "Momentum plus adaptive steps", ADAM_C, ADAM_FILL),
            m.FadeIn(cite, shift=0.15 * m.UP),
        )
        self.say(
            "[First, m: a running average of the gradients. That's momentum.]",
            m.AnimationGroup(m.Write(lines[0][0]), m.FadeIn(lines[0][1], shift=0.15 * m.UP)),
        )
        self.say(
            "[Second, v: a running average of the squared gradients, which measures their size.]",
            m.AnimationGroup(m.Write(lines[1][0]), m.FadeIn(lines[1][1], shift=0.15 * m.UP)),
        )
        self.say(
            "[Both start at zero, so early on they come out too small.] [Dividing by one minus beta to the power t corrects that.]",
            m.AnimationGroup(m.Create(chart), m.FadeIn(chart_title), m.FadeIn(t_label), m.Create(raw, run_time=1.6), m.FadeIn(raw_label)),
            m.AnimationGroup(m.Write(lines[2][0]), m.FadeIn(lines[2][1], shift=0.15 * m.UP), m.Create(fixed, run_time=1.2), m.FadeIn(fixed_label)),
        )
        self.say(
            "[Then the step: the learning rate, times m hat, divided by the square root of v hat.]",
            m.AnimationGroup(m.Write(lines[3][0]), m.FadeIn(lines[3][1], shift=0.15 * m.UP)),
        )
        self.say(
            "[Epsilon is a tiny number that avoids dividing by zero.]",
            m.AnimationGroup(m.Indicate(eps, color=ADAM_C, scale_factor=1.06), m.FadeIn(defaults[2], shift=0.15 * m.UP)),
        )
        self.say(
            "[With the usual settings, m averages over about the last ten steps], [and v over about a thousand.]",
            m.AnimationGroup(m.FadeIn(defaults_label), m.FadeIn(defaults[0], shift=0.15 * m.UP)),
            m.FadeIn(defaults[1], shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ the race

    def race(self):
        self.next_section(
            "The race",
            notes=f"Same start, 60 steps each. Stays within 0.3 of the minimum from step: momentum {ARRIVE_MOM}, Adam {ARRIVE_ADAM}, gradient descent {ARRIVE_GD}. Learning rates 0.95 / 2 / 0.5, each picked to suit its optimizer.",
        )
        V = self.V
        bands, frame = V.bands()
        ring = m.DashedVMobject(m.Circle(radius=ARRIVE_R * V.S).move_to(V.p(0, 0)).set_stroke(INK, width=2.5), num_dashes=18).set_z_index(3)
        lowest = m.Dot(V.p(0, 0), radius=0.06, color=INK).set_z_index(3)
        t = m.ValueTracker(0)
        runs = ((GD_PATH, GD_C, "gradient descent"), (MOM_PATH, MOM_C, "momentum"), (ADAM_PATH, ADAM_C, "Adam"))
        paths = [trail(V, p, t, c, width=3.5, dots=(c == GD_C)) for p, c, _ in runs]
        balls = [rider(V, p, t, c) for p, c, _ in runs]
        keys = m.VGroup(*[m.VGroup(m.Line(m.ORIGIN, 0.5 * m.RIGHT).set_stroke(c, width=6), text(n, 24, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2) for _, c, n in runs])
        keys.arrange(m.RIGHT, buff=0.8).move_to([0, -2.05, 0])
        steps = counter(t, [5.35, 2.17, 0])
        results = m.VGroup(
            pill(f"momentum: {ARRIVE_MOM} steps", MOM_C, MOM_FILL, 22),
            pill(f"Adam: {ARRIVE_ADAM} steps", ADAM_C, ADAM_FILL, 22),
            pill(f"gradient descent: {ARRIVE_GD} steps", GD_C, GD_FILL, 22),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -3.0, 0])
        results_note = text("steps until it stays inside the dashed ring", 20, MUTED).next_to(results, m.DOWN, buff=0.18)

        self.say(
            "[Now let's race all three from the same start], [each with a learning rate that suits it.]",
            m.AnimationGroup(
                self.new_header("THE RACE", "Same valley, same start"),
                m.FadeIn(bands),
                m.Create(frame),
                m.FadeIn(ring),
                m.FadeIn(lowest),
                *[m.FadeIn(b, scale=0.4) for b in balls],
            ),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(k, shift=0.15 * m.UP) for k in keys], lag_ratio=0.25), m.FadeIn(steps)),
        )
        self.add(*paths)
        self.say(
            "[Gradient descent zig-zags], [momentum swings from side to side], [and Adam heads diagonally down, then runs along the floor.]",
            t.animate(run_time=2.0, rate_func=m.linear).set_value(14),
            t.animate(run_time=2.0, rate_func=m.linear).set_value(30),
            t.animate(run_time=3.2, rate_func=m.linear).set_value(STEPS),
        )
        stop(*paths, *balls, steps[1])
        self.say(
            "[Momentum and Adam settle at the bottom in under sixty steps.] [Plain gradient descent would need about a hundred and eighty.]",
            m.AnimationGroup(m.FadeIn(results[0], shift=0.15 * m.UP), m.FadeIn(results[1], shift=0.15 * m.UP), m.Indicate(ring, color=INK)),
            m.AnimationGroup(m.FadeIn(results[2], shift=0.15 * m.UP), m.FadeIn(results_note)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ in practice

    def practice(self):
        self.next_section(
            "In practice",
            notes="AdamW: Loshchilov & Hutter, Decoupled Weight Decay Regularization (ICLR 2019; arXiv 2017), the common default for large language models. Adam keeps m and v: two extra numbers per parameter; SGD with momentum keeps one. SGD with momentum remains common for image classifiers. Facts as of 2026.",
        )
        xs = (-4.45, 0.0, 4.45)

        def frame(x, color):
            return card(4.0, 4.1, color, CARD, stroke=2).move_to([x, -0.45, 0])

        def column(box, *parts, top=0.45, buff=0.22):
            group = m.VGroup(*parts).arrange(m.DOWN, buff=buff)
            return group.move_to(box.get_top() + (top + group.height / 2) * m.DOWN)

        b1 = frame(xs[0], ADAM_C)
        adamw = column(b1, text("AdamW", 34, ADAM_C, weight=m.BOLD), text("Adam, plus weight decay", 22, INK), text("applied separately", 22, INK))
        decay = column(b1, text("weight decay: a gentle pull", 22, MUTED), text("of every weight towards zero", 22, MUTED), top=2.05, buff=0.12)
        cite1 = text("Loshchilov & Hutter, 2019", 20, MUTED).move_to(b1.get_bottom() + 0.4 * m.UP)

        b2 = frame(xs[1], BAD)
        catch = text("The catch: memory", 28, BAD, weight=m.BOLD).move_to(b2.get_top() + 0.6 * m.DOWN)

        def grid(color, label):
            dots = m.VGroup(*[m.Dot(radius=0.06, color=color) for _ in range(16)]).arrange_in_grid(rows=4, cols=4, buff=0.12)
            return m.VGroup(dots, label).arrange(m.DOWN, buff=0.2)

        grids = m.VGroup(grid(INK, math(r"\theta", 38, INK)), grid(MOM_C, math(r"m", 38, MOM_C)), grid(SIZE_C, math(r"v", 38, SIZE_C)))
        grids.arrange(m.RIGHT, buff=0.45, aligned_edge=m.UP).move_to(b2.get_center() + 0.05 * m.DOWN)
        extra = column(b2, text("two extra numbers", 22, INK), text("for every parameter", 22, INK), top=2.95, buff=0.1)

        b3 = frame(xs[2], GD_C)
        classic = column(b3, text("SGD + momentum", 28, GD_C, weight=m.BOLD), text("still a strong choice", 22, INK), text("for many image models", 22, INK))
        lighter = column(b3, text("one running average,", 22, MUTED), text("not two: less memory", 22, MUTED), top=2.05, buff=0.12)

        self.say(
            "[In practice, large models are usually trained with a variant called Adam W.]",
            m.AnimationGroup(self.new_header("IN PRACTICE", "What people actually use"), m.FadeIn(b1), m.FadeIn(adamw, shift=0.15 * m.UP), m.FadeIn(cite1)),
        )
        self.say(
            "[It applies weight decay separately]: [a gentle pull of every weight towards zero, which we'll meet in episode twenty-six.]",
            m.Indicate(adamw[1:], color=ADAM_C, scale_factor=1.05),
            m.FadeIn(decay, shift=0.15 * m.UP),
        )
        self.say(
            "[The catch: Adam stores two extra numbers for every parameter], [so it needs more memory.]",
            m.AnimationGroup(m.FadeIn(b2), m.FadeIn(catch), m.FadeIn(grids[0]), m.FadeIn(grids[1:], shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(extra, shift=0.15 * m.UP), m.Indicate(grids[1:], color=INK, scale_factor=1.06)),
        )
        self.say(
            "[And plain SGD with momentum is still a strong choice], [for example for many image models.]",
            m.AnimationGroup(m.FadeIn(b3), m.FadeIn(classic, shift=0.15 * m.UP)),
            m.FadeIn(lighter, shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Sample, smooth, scale.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Sample", "Smooth", "Scale"), (GD_C, MOM_C, SIZE_C))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.45 * m.UP)
        subs = m.VGroup(*[text(s, 24, MUTED) for s in ("mini-batches", "momentum", "adaptive steps")])
        for sub, w in zip(subs, words):
            sub.next_to(w, m.DOWN, buff=0.3)
        on_baseline(subs)
        tagline = text("Cheap steps, a steady direction, and the right size for every parameter.", 28, INK).move_to([0, -1.9, 0])

        self.say(
            "[To sum up]: [sample], [smooth], [scale.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Cheap steps from mini-batches, a steady direction from momentum, and the right step size for every parameter.]",
            m.Write(tagline),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, subs, tagline)], run_time=0.9)
