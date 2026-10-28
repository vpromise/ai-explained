"""AI, Explained · Episode 34 · VAEs & GANs

Running example: generating cartoon faces. Every face is drawn here in code (no photographs)
from two numbers: z1, how much it smiles, and z2, how open its eyes are, so a two-number latent
space can be shown as a grid of faces. The same drawer stands in for a trained decoder and
generator: the faces illustrate what those models do, they are not the outputs of trained
networks. The episode's hero face has the code (0.8, -0.3).

Computed here (numpy, fixed seeds) and asserted where the narration quotes them:
- the encoder's input is a 32 x 32 picture: 1,024 pixels;
- the reparameterisation example: z = mu + sigma * eps = 0.8 + 0.3 * (-1.2) = 0.44;
- the KL divergence of the cloud N(0.8, 0.3^2) from the prior N(0, 1): 1.07 nats, closed form;
  it falls to 0 when the cloud becomes the prior;
- the GAN picture: one feature (the smile) of real faces, N(0.6, 0.45^2), illustrative, and a
  generator whose spread moves onto it; the optimal discriminator D*(x) = p_data / (p_data + p_g)
  (Goodfellow et al., 2014, Prop. 1), which becomes 1/2 everywhere when the spreads match;
- the "blurrier" VAE-style picture is the pixel average of 64 faces whose codes are drawn
  around the hero's; the "sharp" GAN-style picture is one face.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

HERO = (0.8, -0.3)  # the running example's code: smile 0.8, eyes -0.3
PIX = 32  # the encoder's input picture is PIX x PIX
assert PIX * PIX == 1024  # "a thousand and twenty-four pixels"

MU, SIGMA, EPS = 0.8, 0.3, -1.2  # along the smile direction
Z_SAMPLE = MU + SIGMA * EPS
assert round(Z_SAMPLE, 2) == 0.44  # "a code of zero point four four"
SIGMA2 = 0.25  # the cloud's spread along the eyes direction (picture only)


def kl_normal(mu, sigma):
    """KL( N(mu, sigma^2) || N(0, 1) ) in nats."""
    return 0.5 * (mu**2 + sigma**2 - 1 - np.log(sigma**2))


KL_START = float(kl_normal(MU, SIGMA))
assert round(KL_START, 2) == 1.07  # "costs about one point zero seven"
assert abs(kl_normal(0.0, 1.0)) < 1e-12  # "the cost falls to zero"
assert kl_normal(0.4, 0.65) > 0 and kl_normal(-0.3, 1.4) > 0  # never negative


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


REAL_MU, REAL_SD = 0.6, 0.45  # one feature of real faces: the smile (illustrative)
FAKE_MU, FAKE_SD = -1.5, 0.35  # where the generator starts


def fake_at(t):
    """The generator's spread of smiles at training progress t in [0, 1]."""
    return FAKE_MU + (REAL_MU - FAKE_MU) * t, FAKE_SD + (REAL_SD - FAKE_SD) * t


def d_star(x, t):
    """The best possible discriminator: p_data / (p_data + p_g)."""
    mg, sg = fake_at(t)
    real, fake = gauss(x, REAL_MU, REAL_SD), gauss(x, mg, sg)
    return real / (real + fake + 1e-300)


_XS = np.linspace(-3, 3, 601)
assert d_star(REAL_MU, 0) > 0.99 and d_star(FAKE_MU, 0) < 0.01  # "real where real faces are", "fake where the fakes are"
assert np.allclose(d_star(_XS, 1.0), 0.5)  # "one half, everywhere"

rng = np.random.default_rng(34)
HOOK_CODES = [HERO, (-1.6, 1.2), (1.9, 1.5), (-0.4, -1.7), (1.3, -1.4), (-1.9, -0.6), (0.2, 1.9), (-0.9, 0.4)]
SAMPLES = [(MU + SIGMA * a, HERO[1] + SIGMA2 * b) for a, b in rng.normal(0, 1, (6, 2)).clip(-1.6, 1.6)]
NEW_CODES = [tuple(c) for c in rng.normal(0, 1, (6, 2)).clip(-2.1, 2.1)]
COLLAPSE = [(1.5 + 0.05 * a, 0.9 + 0.05 * b) for a, b in rng.normal(0, 1, (6, 2))]
REAL_ROW = [(-1.5, 1.0), (1.8, -0.4), (0.1, 1.9), (-0.7, -1.6), (1.1, 0.8), (-1.9, -0.2)]
BLUR_CODES = np.array(HERO) + np.array([0.75, 0.75]) * rng.normal(0, 1, (64, 2))
assert len(BLUR_CODES) == 64

# ---------------------------------------------------------------------------- faces


def face(z1, z2, size=1.2, color=VIOLET, fill=VIOLET_FILL):
    """A cartoon face from two numbers: z1 = smile (-2.5 frown … 2.5 grin), z2 = eyes (sleepy … wide)."""
    r = size / 2
    w = max(1.6, 2.6 * size)
    head = m.Circle(radius=r).set_stroke(color, width=w).set_fill(fill, opacity=1)
    t = float(np.clip((z2 + 2.5) / 5, 0, 1))
    rx, ry = r * (0.10 + 0.04 * t), r * (0.035 + 0.15 * t)
    eyes = m.VGroup(
        *[m.Ellipse(width=2 * rx, height=2 * ry).set_stroke(width=0).set_fill(INK, opacity=1).move_to([s * 0.36 * r, 0.14 * r, 0]) for s in (-1, 1)]
    )
    c = 0.16 * float(np.clip(z1, -2.5, 2.5))
    xs = np.linspace(-1, 1, 25)
    mouth = m.VMobject().set_points_smoothly([[0.38 * r * x, r * (-0.38 + c * (x * x - 0.5)), 0] for x in xs])
    mouth.set_stroke(INK, width=w * 1.1)
    return m.VGroup(head, eyes, mouth)


def garbled(size=1.2, color=BAD, fill=BAD_FILL, seed=5):
    """What a decoder draws from a code it never learned: a lumpy head, odd eyes, a scribble."""
    r = size / 2
    w = max(1.6, 2.6 * size)
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rr = r * (1 + 0.09 * np.sin(3 * a + seed) + 0.06 * np.sin(5 * a + 2 * seed))
    head = m.VMobject().set_points_smoothly([[x, y, 0] for x, y in zip(rr * np.cos(a), rr * np.sin(a))] + [[rr[0], 0, 0]])
    head.set_stroke(color, width=w).set_fill(fill, opacity=1)
    eyes = m.VGroup(
        m.Ellipse(width=0.42 * r, height=0.2 * r).set_stroke(width=0).set_fill(INK, opacity=1).rotate(0.5).move_to([-0.34 * r, 0.32 * r, 0]),
        m.Ellipse(width=0.12 * r, height=0.1 * r).set_stroke(width=0).set_fill(INK, opacity=1).move_to([0.44 * r, -0.02 * r, 0]),
    )
    xs = np.linspace(-0.42, 0.36, 9)
    ys = [-0.42 + 0.12 * (-1) ** k + 0.08 * x for k, x in enumerate(xs)]
    mouth = m.VMobject().set_points_as_corners([[x * r, y * r, 0] for x, y in zip(xs, ys)]).set_stroke(INK, width=w)
    return m.VGroup(head, eyes, mouth)


def rough(size=1.2, color=VIOLET, fill=VIOLET_FILL):
    """A fake halfway through training: nearly a face."""
    g = face(0.3, 0.6, size, color, fill)
    g[1][0].scale(1.35).shift(0.04 * size * m.UP)
    g[2].rotate(-0.18)
    return g


def _rgb(hexcode):
    return np.array([int(hexcode[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def face_pixels(z1, z2, n, color=SKY, fill=SKY_FILL, line=0.085):
    """The same face drawn as n x n RGBA pixels (anti-aliased, transparent outside the head)."""
    span = 1.12
    s = (np.arange(n) + 0.5) / n * 2 * span - span
    x, y = np.meshgrid(s, -s)
    px = 2 * span / n
    rad = np.sqrt(x * x + y * y)

    def cover(d):
        return np.clip(0.5 - d / px, 0, 1)[..., None]

    alpha = cover(rad - 1 - line / 2)
    rgb = np.broadcast_to(_rgb(fill), (n, n, 3)).copy()
    rgb = rgb * (1 - cover(np.abs(rad - 1) - line / 2)) + _rgb(color) * cover(np.abs(rad - 1) - line / 2)
    t = float(np.clip((z2 + 2.5) / 5, 0, 1))
    rx, ry = 0.10 + 0.04 * t, 0.035 + 0.15 * t
    for side in (-1, 1):
        dx, dy = x - side * 0.36, y - 0.14
        k = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
        grad = np.sqrt((dx / rx**2) ** 2 + (dy / ry**2) ** 2) / np.maximum(k, 1e-9)
        cov = cover((k - 1) / np.maximum(grad, 1e-9))
        rgb = rgb * (1 - cov) + _rgb(INK) * cov
    c = 0.16 * float(np.clip(z1, -2.5, 2.5))
    u = np.linspace(-1, 1, 48)
    mx, my = 0.38 * u, -0.38 + c * (u * u - 0.5)
    dist = np.min(np.sqrt((x[..., None] - mx) ** 2 + (y[..., None] - my) ** 2), axis=-1)
    cov = cover(dist - line * 0.55)
    rgb = rgb * (1 - cov) + _rgb(INK) * cov
    return np.concatenate([rgb, 255 * alpha], axis=-1)


def picture(rgba, height, smooth=True):
    img = m.ImageMobject(np.round(rgba).astype(np.uint8), scale_to_resolution=0, resampling_algorithm=3 if smooth else 0)
    img.height = height
    return img


def pixel_frame(img, color):
    return m.Rectangle(width=img.width + 0.12, height=img.height + 0.12).set_stroke(color, width=2.5).move_to(img)


# ---------------------------------------------------------------------------- other helpers


def variant(name, color, fill):
    """kit.recolored(), written under this episode's prefix: assets/ep34-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    stem = f"ep34-{name}-{color.lstrip('#').lower()}"
    path = ASSETS / f"{stem}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return stem


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


def cite(words):
    return text(words, 20, MUTED)


def funnel(color, fill, label, wide_left=True, width=1.7, wide=1.9, narrow=0.8):
    """An encoder (wide → narrow) or a decoder (narrow → wide) as a trapezoid with its name."""
    a, b = (wide, narrow) if wide_left else (narrow, wide)
    shape = m.Polygon([-width / 2, a / 2, 0], [width / 2, b / 2, 0], [width / 2, -b / 2, 0], [-width / 2, -a / 2, 0])
    shape.set_stroke(color, width=2.5).set_fill(fill, opacity=1)
    name = text(label, 22, INK, weight=m.BOLD).move_to(shape)
    return m.VGroup(shape, name)


def code_box(values, color=AMBER, fill=AMBER_FILL, size=24):
    nums = m.VGroup(*[code(v, size, color) for v in values]).arrange(m.DOWN, buff=0.14)
    box = card(nums.width + 0.45, nums.height + 0.36, color, fill, radius=0.15, stroke=2.5)
    return m.VGroup(box, nums.move_to(box))


class Plane:
    """A latent plane: a card with faint cross axes; codes in [-span, span] on both axes."""

    def __init__(self, side, center, span=2.5, labels=True):
        self.box = card(side, side, EDGE, CARD, radius=0.2, stroke=2).move_to(center)
        self.unit = (side - 0.3) / (2 * span)
        self.center = np.array(center, dtype=float)
        h = side / 2 - 0.12
        self.axes = m.VGroup(
            m.Line(self.center + [-h, 0, 0], self.center + [h, 0, 0]),
            m.Line(self.center + [0, -h, 0], self.center + [0, h, 0]),
        ).set_stroke(FAINT, width=2)
        self.frame = m.VGroup(self.box, self.axes)
        if labels:
            self.frame.add(
                text("smile →", 22, MUTED).next_to(self.box, m.DOWN, buff=0.14),
                m.VGroup(text("eyes", 22, MUTED), text("↑", 22, MUTED)).arrange(m.DOWN, buff=0.06).next_to(self.box, m.LEFT, buff=0.16),
            )

    def p(self, z1, z2):
        return self.center + self.unit * np.array([z1, z2, 0])


def dice(side=0.62, color=AMBER, fill=AMBER_FILL):
    box = card(side, side, color, fill, radius=0.12, stroke=2.5)
    pips = m.VGroup(*[m.Dot(box.get_center() + side * np.array([a, a, 0]), radius=0.05, color=color) for a in (-0.24, 0, 0.24)])
    return m.VGroup(box, pips)


def node_circle(sym, color, fill, radius=0.42, size=40):
    disc = m.Circle(radius=radius).set_stroke(color, width=3).set_fill(fill, opacity=1)
    return m.VGroup(disc, tex(sym, size, INK).move_to(disc))


def bell(color, width=0.8, height=0.4, stroke=3):
    xs = np.linspace(-2.6, 2.6, 60)
    return polyline([[x / 5.2 * width, np.exp(-x * x / 2) * height - height / 2, 0] for x in xs], color, stroke)


# ---------------------------------------------------------------------------- the episode


class Ep34VAEsGANs(Episode):
    number = 34

    def lesson(self):
        self.hook()
        self.latent()
        self.autoencoder()
        self.vae()
        self.reparam()
        self.elbo()
        self.new_faces()
        self.gan()
        self.minimax()
        self.tradeoffs()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Eight cartoon faces, each drawn from two numbers (codes shown). VAEs and GANs.")
        faces, codes = m.VGroup(), m.VGroup()
        for k, (z1, z2) in enumerate(HOOK_CODES):
            x, y = -4.5 + 3.0 * (k % 4), (1.45 if k < 4 else -0.85)
            f = face(z1, z2, 1.45).move_to([x, y, 0])
            faces.add(f)
            codes.add(code(f"({z1:.1f}, {z2:.1f})".replace("-", "−"), 22, AMBER).next_to(f, m.DOWN, buff=0.18))
        vae = m.VGroup(pill("VAEs", TEAL, TEAL_FILL, 26), cite("Kingma & Welling, 2013")).arrange(m.RIGHT, buff=0.25)
        gan = m.VGroup(pill("GANs", PINK, PINK_FILL, 26), cite("Goodfellow et al., 2014")).arrange(m.RIGHT, buff=0.25)
        pair = m.VGroup(vae, gan).arrange(m.RIGHT, buff=1.2).move_to([0, -2.95, 0])
        tagline = text("two of the first deep generative models", 24, MUTED).next_to(pair, m.DOWN, buff=0.22)

        self.say(
            "So... [how does a computer draw a face that nobody has ever drawn?]",
            m.LaggedStart(*[m.FadeIn(f, scale=0.7) for f in faces], lag_ratio=0.15),
        )
        self.say("[Each of these was made from just two numbers.]", m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in codes], lag_ratio=0.1))
        self.say(
            "Today: [VAEs], [and GANs], [two of the first deep generative models.]",
            m.FadeIn(vae, shift=0.2 * m.UP),
            m.FadeIn(gan, shift=0.2 * m.UP),
            m.FadeIn(tagline),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ latent space

    def latent(self):
        self.next_section("Latent space", notes="A face from two numbers (smile, eyes); a live point and face; the plane as a 5 x 5 grid of faces.")
        plane = Plane(4.8, [-3.0, -0.5, 0])
        z1, z2 = m.ValueTracker(0.0), m.ValueTracker(0.0)
        dot = m.Dot(plane.p(0, 0), radius=0.11, color=AMBER).set_z_index(3)
        dot.add_updater(lambda d: d.move_to(plane.p(z1.get_value(), z2.get_value())))
        big = m.always_redraw(lambda: face(z1.get_value(), z2.get_value(), 3.0).move_to([3.4, 0.45, 0]))

        def readout(name, tracker):
            num = m.DecimalNumber(0, num_decimal_places=1, mob_class=Mono, font_size=30, color=AMBER, include_sign=False)
            num.add_updater(lambda d: d.set_value(tracker.get_value()))
            return m.VGroup(text(name, 26, MUTED, weight=m.SEMIBOLD), num).arrange(m.RIGHT, buff=0.25)

        r1, r2 = readout("smile", z1), readout("eyes", z2)
        reads = m.VGroup(r1, r2).arrange(m.RIGHT, buff=0.8).move_to([3.4, -1.75, 0])
        grid = m.VGroup()
        for a in (-2, -1, 0, 1, 2):
            for b in (-2, -1, 0, 1, 2):
                grid.add(face(a, b, 0.74).move_to(plane.p(a, b)))
        grid.set_z_index(2)
        name = pill("latent space", AMBER, AMBER_FILL, 22).next_to(plane.box, m.UP, buff=0.18)
        unknown = m.VGroup(*[text("?", 30, BAD, weight=m.BOLD).move_to(r[1]) for r in (r1, r2)])
        learn = pill("learned from pictures alone", TEAL, TEAL_FILL, 24).move_to([3.4, -3.0, 0])

        self.say(
            "[Our cartoon faces are drawn from two numbers]: [how much the face smiles], [and how open its eyes are.]",
            m.AnimationGroup(
                self.new_header("LATENT SPACE", "Two numbers behind every face"),
                m.FadeIn(plane.frame),
                m.FadeIn(dot, scale=0.4),
                m.FadeIn(big),
                m.FadeIn(reads),
            ),
            m.Succession(
                z1.animate(run_time=0.9).set_value(2.2), z1.animate(run_time=1.2).set_value(-2.2), z1.animate(run_time=0.8).set_value(0.8)
            ),
            m.Succession(
                z2.animate(run_time=0.9).set_value(2.2), z2.animate(run_time=1.2).set_value(-2.2), z2.animate(run_time=0.8).set_value(-0.3)
            ),
        )
        self.say(
            "[Move the point anywhere], and the face changes smoothly.",
            m.Succession(
                *[
                    m.AnimationGroup(z1.animate.set_value(a), z2.animate.set_value(b), run_time=0.85)
                    for a, b in ((1.8, 1.7), (-1.7, 1.2), (-1.4, -1.9), HERO)
                ]
            ),
        )
        self.stop(dot, big, r1[1], r2[1])
        self.say(
            "[This space of hidden numbers is called a latent space], [here drawn as a grid of faces.]",
            m.FadeIn(name, shift=0.15 * m.DOWN),
            m.AnimationGroup(m.FadeOut(dot), m.LaggedStart(*[m.FadeIn(f, scale=0.5) for f in grid], lag_ratio=0.04)),
        )
        self.say(
            "[Real pictures don't come with these numbers.] [A model must discover them by itself.]",
            m.AnimationGroup(*[m.FadeTransform(r[1], u) for r, u in zip((r1, r2), unknown)]),
            m.FadeIn(learn, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ autoencoder

    def autoencoder(self):
        self.next_section("Encoder and decoder", notes="A 32 x 32 face (1,024 pixels) squeezed to a two-number code and rebuilt; a plain autoencoder's codes leave gaps (illustrative clumps); decoding a gap gives nonsense.")
        y = 1.15
        src = picture(face_pixels(*HERO, PIX, SKY, SKY_FILL, line=0.1), 1.9, smooth=False).move_to([-5.45, y, 0])
        src_frame = pixel_frame(src, SKY)
        src_label = text("32 × 32 = 1,024 pixels", 22, SKY, weight=m.SEMIBOLD).next_to(src_frame, m.DOWN, buff=0.2)
        enc = funnel(TEAL, TEAL_FILL, "encoder").move_to([-2.85, y, 0])
        zbox = code_box(["0.8", "−0.3"]).move_to([-0.75, y, 0])
        z_label = text("code", 22, AMBER, weight=m.SEMIBOLD).next_to(zbox, m.DOWN, buff=0.2)
        dec = funnel(VIOLET, VIOLET_FILL, "decoder", wide_left=False).move_to([1.35, y, 0])
        out_rgba = np.mean([face_pixels(HERO[0] + 0.25 * a, HERO[1] + 0.25 * b, PIX, VIOLET, VIOLET_FILL, line=0.1) for a, b in np.random.default_rng(3).normal(0, 1, (12, 2))], axis=0)
        out = picture(out_rgba, 1.9, smooth=False).move_to([4.05, y, 0])
        out_frame = pixel_frame(out, VIOLET)
        out_label = text("rebuilt", 22, VIOLET, weight=m.SEMIBOLD).next_to(out_frame, m.DOWN, buff=0.2)
        a1 = arrow(src_frame.get_right(), enc.get_left(), MUTED)
        a2 = arrow(enc.get_right(), zbox.get_left(), MUTED)
        a3 = arrow(zbox.get_right(), dec.get_left(), MUTED)
        a4 = arrow(dec.get_right(), out_frame.get_left(), MUTED)
        match = pill("training: make the copy match the original, pixel by pixel", GOOD, GOOD_FILL, 22).move_to([0, -0.75, 0])

        plane = Plane(2.7, [-2.55, -2.35, 0], labels=False)
        plane_label = m.VGroup(text("the codes", 22, MUTED), text("it learned", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.RIGHT)
        plane_label.next_to(plane.box, m.LEFT, buff=0.3)
        clumps = ((-1.5, 1.3), (1.4, 1.1), (0.6, -1.5))
        dots = m.VGroup()
        crng = np.random.default_rng(7)
        for cx, cy in clumps:
            for a, b in crng.normal(0, 0.22, (9, 2)):
                dots.add(m.Dot(plane.p(cx + a, cy + b), radius=0.05, color=TEAL))
        gap_at = (-0.6, -0.9)
        ring = m.DashedVMobject(m.Circle(radius=0.36).set_stroke(MUTED, width=2.5), num_dashes=14).move_to(plane.p(*gap_at))
        gap_label = text("gap", 22, MUTED).next_to(ring, m.LEFT, buff=0.1)
        probe = m.Dot(plane.p(*gap_at), radius=0.09, color=AMBER).set_z_index(3)
        junk = garbled(1.7).move_to([2.0, -2.35, 0])
        decode = arrow(plane.box.get_right() + [0.05, 0, 0], junk.get_left(), MUTED, buff=0.2)
        decode_label = text("decode", 22, MUTED).next_to(decode, m.UP, buff=0.1)
        nonsense = pill("✗  nonsense", BAD, BAD_FILL, 24).next_to(junk, m.RIGHT, buff=0.45)

        self.say(
            "[An autoencoder squeezes a picture], [a thousand and twenty-four pixels], [down to a code of just two numbers.]",
            m.AnimationGroup(self.new_header("AUTOENCODER", "Squeeze a face, then rebuild it"), m.FadeIn(src), m.Create(src_frame)),
            m.FadeIn(src_label, shift=0.15 * m.UP),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(enc), m.GrowArrow(a2), m.FadeIn(zbox, scale=0.8), m.FadeIn(z_label)),
        )
        self.say(
            "[A decoder rebuilds the face from that code], [and training makes the copy match the original.]",
            m.AnimationGroup(m.GrowArrow(a3), m.FadeIn(dec), m.GrowArrow(a4), m.FadeIn(out), m.Create(out_frame), m.FadeIn(out_label)),
            m.FadeIn(match, shift=0.2 * m.UP),
        )
        self.say(
            "[But its codes can land anywhere], [with empty gaps in between.]",
            m.AnimationGroup(m.FadeOut(match), m.FadeIn(plane.frame), m.FadeIn(plane_label), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.03)),
            m.AnimationGroup(m.Create(ring), m.FadeIn(gap_label)),
        )
        self.say(
            "[Decode a point from a gap], [and you get nonsense.] So a plain autoencoder can't invent new faces.",
            m.AnimationGroup(m.FadeIn(probe, scale=0.4), m.GrowArrow(decode), m.FadeIn(decode_label)),
            m.Succession(m.AnimationGroup(m.FadeIn(junk, scale=0.8), m.FadeIn(nonsense, shift=0.15 * m.LEFT)), m.Wiggle(junk, scale_value=1.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the VAE

    def vae(self):
        self.next_section("A cloud, not a point", notes="Kingma & Welling, 2013. The encoder outputs a centre mu = (0.8, -0.3) and a spread sigma = (0.3, 0.25); codes sampled from the cloud (seeded) all decode to the hero's kind of face.")
        y = 0.1
        src = face(*HERO, 1.4, SKY, SKY_FILL).move_to([-6.1, y, 0])
        enc = funnel(TEAL, TEAL_FILL, "encoder", width=1.5, wide=1.7, narrow=0.75).move_to([-3.95, y, 0])
        plane = Plane(4.2, [0, y, 0], span=1.6, labels=False)
        plane.box.set_stroke(TEAL)
        center = plane.p(*HERO)
        cloud = m.VGroup(
            *[
                m.Ellipse(width=2 * k * SIGMA * plane.unit, height=2 * k * SIGMA2 * plane.unit).set_stroke(TEAL, width=2, opacity=0.9 if k == 2 else 0.5).set_fill(TEAL, opacity=0.12).move_to(center)
                for k in (2, 1)
            ]
        )
        mu_dot = m.Dot(center, radius=0.08, color=INK).set_z_index(3)
        mu_label = m.VGroup(text("centre", 24, TEAL, weight=m.SEMIBOLD), tex(r"\mu", 36, TEAL)).arrange(m.RIGHT, buff=0.12)
        mu_label.move_to(center + [0, 1.0, 0])
        mu_link = m.Line(mu_label.get_bottom() + [0, -0.06, 0], center + [0, 0.12, 0]).set_stroke(MUTED, width=2)
        sig_line = m.Line(center, center + [2 * SIGMA * plane.unit, 0, 0]).set_stroke(TEAL, width=3)
        sig_label = m.VGroup(text("spread", 24, TEAL, weight=m.SEMIBOLD), tex(r"\sigma", 36, TEAL)).arrange(m.RIGHT, buff=0.12)
        sig_label.move_to(center + [0.1, -1.0, 0])
        samples = m.VGroup(*[m.Dot(plane.p(a, b), radius=0.06, color=AMBER).set_z_index(4) for a, b in SAMPLES])
        dec = funnel(VIOLET, VIOLET_FILL, "decoder", wide_left=False, width=1.5, wide=1.7, narrow=0.75).move_to([3.95, y, 0])
        outs = [face(a, b, 1.4).move_to([6.1, y, 0]) for a, b in SAMPLES]
        e1 = arrow(src.get_right(), enc.get_left(), MUTED)
        e2 = arrow(enc.get_right(), plane.box.get_left(), MUTED)
        e3 = arrow(plane.box.get_right(), dec.get_left(), MUTED)
        e4 = arrow(dec.get_right(), outs[0].get_left(), MUTED)
        source = cite("Kingma & Welling, 2013").move_to([5.3, 3.2, 0])
        region = pill("nearby codes all mean the same kind of face: no gaps", GOOD, GOOD_FILL, 22).move_to([0, -2.9, 0])
        out = outs[0]

        self.say(
            "[A variational autoencoder fixes this.] [Its encoder outputs a cloud, not a point]: [a centre, and a spread.]",
            m.AnimationGroup(
                self.new_header("VAE", "A cloud instead of a point"), m.FadeIn(source), m.FadeIn(src), m.GrowArrow(e1), m.FadeIn(enc), m.FadeIn(plane.frame)
            ),
            m.AnimationGroup(m.GrowArrow(e2), m.FadeIn(cloud, scale=0.3)),
            m.AnimationGroup(m.FadeIn(mu_dot, scale=0.4), m.Create(mu_link), m.FadeIn(mu_label), m.Create(sig_line), m.FadeIn(sig_label)),
        )
        self.say(
            "[We sample a code from the cloud], [and decode it], [so a whole region comes to mean this face.]",
            m.LaggedStart(*[m.FadeIn(s, scale=0.3) for s in samples], lag_ratio=0.2),
            m.Succession(
                m.AnimationGroup(m.GrowArrow(e3), m.FadeIn(dec), m.GrowArrow(e4), m.FadeIn(out, scale=0.8)),
                *[m.Transform(out, o, run_time=0.45) for o in outs[1:]],
            ),
            m.FadeIn(region, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ reparameterisation

    def reparam(self):
        self.next_section(
            "Reparameterisation",
            notes="z = mu + sigma * eps, eps from N(0, 1). Example along the smile direction: 0.8 + 0.3 * (-1.2) = 0.44. Gradients reach mu and sigma; eps needs none.",
        )
        y = 0.75
        # before: cloud → random draw → code → decoder → face
        cloud_card = m.VGroup(card(1.9, 1.1, TEAL, TEAL_FILL), m.VGroup(tex(r"\mu,\ \sigma", 34, INK))).move_to([-4.9, y, 0])
        cloud_card[1].move_to(cloud_card[0])
        die = dice().move_to([-2.4, y, 0])
        die_label = text("random draw", 22, AMBER, weight=m.SEMIBOLD).next_to(die, m.UP, buff=0.18)
        z = node_circle("z", AMBER, AMBER_FILL).move_to([1.0, y, 0])
        dec = funnel(VIOLET, VIOLET_FILL, "decoder", wide_left=False, width=1.5, wide=1.6, narrow=0.7).move_to([3.15, y, 0])
        out = face(Z_SAMPLE, HERO[1], 1.5).move_to([5.55, y, 0])
        b1 = arrow(cloud_card.get_right(), die.get_left(), MUTED)
        b2 = arrow(die.get_right(), z.get_left(), MUTED)
        z_dec = arrow(z.get_right(), dec.get_left(), MUTED)
        dec_out = arrow(dec.get_right(), out.get_left(), MUTED)
        back = m.DashedLine([0.9, y - 1.05, 0], [-4.9, y - 1.05, 0], dash_length=0.12).set_stroke(BAD, width=3)
        back_tip = m.Triangle().scale(0.12).rotate(np.pi / 2).set_stroke(width=0).set_fill(BAD, opacity=1).move_to([-4.95, y - 1.05, 0])
        cross = text("✗", 40, BAD, weight=m.BOLD).move_to([-2.4, y - 1.05, 0]).set_z_index(3)
        cross_bg = m.Circle(radius=0.3).set_stroke(width=0).set_fill(BG, opacity=1).move_to(cross)
        no_grad = text("gradients can't pass a random draw", 22, BAD, weight=m.SEMIBOLD).move_to([-2.0, y - 1.65, 0])

        # after: z = mu + sigma * eps
        formula = m.MathTex("z", "=", r"\mu", "+", r"\sigma", r"\cdot", r"\varepsilon", font_size=60, color=INK).move_to([0, 2.05, 0])
        formula[0].set_color(AMBER)
        formula[2].set_color(TEAL)
        formula[4].set_color(TEAL)
        formula[6].set_color(SKY)
        mu = node_circle(r"\mu", TEAL, TEAL_FILL).move_to([-5.3, y, 0])
        sig = node_circle(r"\sigma", TEAL, TEAL_FILL).move_to([-5.3, -0.9, 0])
        times = node_circle(r"\times", MUTED, CARD, radius=0.32, size=34).move_to([-3.05, -0.9, 0])
        plus = node_circle("+", MUTED, CARD, radius=0.32, size=34).move_to([-1.0, y, 0])
        eps = node_circle(r"\varepsilon", SKY, SKY_FILL).move_to([-3.05, -2.3, 0])
        eps_bell = bell(SKY, 0.9, 0.42)
        eps_label = text("random, from a standard bell curve", 22, SKY)
        eps_info = m.VGroup(eps_bell, eps_label).arrange(m.RIGHT, buff=0.25).next_to(eps, m.RIGHT, buff=0.3)
        l_mu = arrow(mu.get_right(), plus.get_left(), TEAL)
        l_sig = arrow(sig.get_right(), times.get_left(), TEAL)
        l_eps = arrow(eps.get_top(), times.get_bottom(), SKY)
        l_times = arrow(times.get_right(), plus.get_bottom() + [-0.12, -0.05, 0], MUTED)
        l_plus = arrow(plus.get_right(), z.get_left(), MUTED)
        vals = m.VGroup(
            code("0.8", 24, TEAL).next_to(mu, m.UP, buff=0.15),
            code("0.3", 24, TEAL).next_to(sig, m.DOWN, buff=0.15),
            code("−1.2", 24, SKY).next_to(eps, m.LEFT, buff=0.2),
            code("0.44", 24, AMBER).next_to(z, m.UP, buff=0.15),
        )
        worked = tex(r"0.8 + 0.3 \times (-1.2) = 0.44", 40, INK).move_to([3.4, -1.1, 0])
        grads = m.VGroup(
            text("gradient ✓", 22, GOOD, weight=m.SEMIBOLD).next_to(l_mu, m.UP, buff=0.12),
            text("gradient ✓", 22, GOOD, weight=m.SEMIBOLD).next_to(l_sig, m.DOWN, buff=0.12),
        )
        outside = pill("the randomness sits outside: gradients reach the centre and the spread", GOOD, GOOD_FILL, 22).move_to([0, -3.35, 0])

        def back_pulse(a, b):
            return m.ShowPassingFlash(m.Line(a, b).set_stroke(GOOD, width=9), time_width=0.8)

        self.say(
            "[But sampling is random], [and you can't backpropagate through a random draw.]",
            m.AnimationGroup(
                self.new_header("REPARAMETERISATION", "Move the randomness aside"),
                m.FadeIn(cloud_card), m.GrowArrow(b1), m.FadeIn(die), m.FadeIn(die_label), m.GrowArrow(b2), m.FadeIn(z),
                m.GrowArrow(z_dec), m.FadeIn(dec), m.GrowArrow(dec_out), m.FadeIn(out),
            ),
            m.AnimationGroup(m.Create(back), m.FadeIn(back_tip), m.FadeIn(cross_bg), m.FadeIn(cross, scale=1.4), m.FadeIn(no_grad)),
        )
        self.say(
            "[The trick]: [write the code as the centre, plus the spread times a random number.]",
            m.FadeOut(m.VGroup(cloud_card, b1, die, die_label, b2, back, back_tip, cross, cross_bg, no_grad)),
            m.Write(formula, run_time=1.6),
        )
        self.say(
            "[The random number comes from a standard bell curve], [the spread scales it], [and the centre shifts it.]",
            m.AnimationGroup(m.Circumscribe(formula[6], color=SKY, buff=0.08), m.FadeIn(eps), m.Create(eps_bell), m.FadeIn(eps_label)),
            m.AnimationGroup(m.Circumscribe(formula[4], color=TEAL, buff=0.08), m.FadeIn(sig), m.GrowArrow(l_sig), m.GrowArrow(l_eps), m.FadeIn(times)),
            m.AnimationGroup(m.Circumscribe(formula[2], color=TEAL, buff=0.08), m.FadeIn(mu), m.GrowArrow(l_mu), m.FadeIn(plus), m.GrowArrow(l_times), m.GrowArrow(l_plus)),
        )
        self.say(
            "[Say the centre is zero point eight, the spread zero point three, and the draw minus one point two]: [the code is zero point four four.]",
            m.LaggedStart(*[m.FadeIn(v, shift=0.1 * m.UP) for v in vals[:3]], lag_ratio=0.5),
            m.AnimationGroup(m.FadeIn(vals[3], shift=0.1 * m.UP), m.Write(worked)),
        )
        self.say(
            "[Now the randomness sits outside the network], [so gradients flow back through the centre and the spread.] That's the reparameterisation trick.",
            m.Circumscribe(m.VGroup(eps, eps_info), color=SKY, buff=0.12),
            m.AnimationGroup(
                m.Succession(
                    back_pulse(out.get_left(), dec.get_right()),
                    back_pulse(dec.get_left(), z.get_right()),
                    back_pulse(z.get_left(), plus.get_right()),
                    m.AnimationGroup(back_pulse(plus.get_left(), mu.get_right()), back_pulse(plus.get_bottom(), times.get_right())),
                    back_pulse(times.get_left(), sig.get_right()),
                    run_time=2.4,
                ),
                m.Succession(m.Wait(1.6), m.AnimationGroup(m.FadeIn(grads), m.FadeIn(outside, shift=0.2 * m.UP))),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the ELBO

    def elbo(self):
        self.next_section(
            "The ELBO",
            notes="ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z)) <= log p(x). KL of N(0.8, 0.3^2) from N(0, 1) = 1.07 nats, falling to 0 as the cloud becomes the prior. Clouds packed around zero (illustrative).",
        )
        formula = m.MathTex(
            r"\mathrm{ELBO}", "=", r"\mathbb{E}_{q(z \mid x)}\big[\log p(x \mid z)\big]", "-", r"\mathrm{KL}\big(q(z \mid x)\,\|\,p(z)\big)", font_size=42, color=INK
        ).move_to([0, 2.05, 0])
        formula[2].set_color(VIOLET)
        formula[4].set_color(AMBER)
        brace1 = m.Brace(formula[2], m.DOWN, buff=0.12).set_color(VIOLET)
        brace2 = m.Brace(formula[4], m.DOWN, buff=0.12).set_color(AMBER)
        label1 = text("reconstruction", 24, VIOLET, weight=m.BOLD).next_to(brace1, m.DOWN, buff=0.1)
        label2 = text("KL to the prior", 24, AMBER, weight=m.BOLD).next_to(brace2, m.DOWN, buff=0.1)

        # reconstruction: the original and the rebuilt face
        left_c = np.array([-4.0, -1.45, 0])
        orig = face(*HERO, 1.6, SKY, SKY_FILL).move_to(left_c + [-1.25, 0.15, 0])
        rebuilt = face(Z_SAMPLE, HERO[1], 1.6).move_to(left_c + [1.25, 0.15, 0])
        approx = tex(r"\approx", 48, MUTED).move_to(left_c + [0, 0.15, 0])
        recon_note = text("likely to rebuild the original?", 22, MUTED).move_to(left_c + [0, -1.15, 0])
        recon = m.VGroup(orig, approx, rebuilt, recon_note)

        # KL: the encoder's cloud against the prior, along the smile direction
        ax = axes([-3, 3, 1], [0, 1.45, 0.5], 5.6, 2.9).move_to([3.15, -1.4, 0])
        ticks = m.VGroup(*[code(f"{v}".replace("-", "−"), 20, MUTED).next_to(ax.c2p(v, 0), m.DOWN, buff=0.15) for v in (-2, 0, 2)])
        mu_t, sd_t = m.ValueTracker(MU), m.ValueTracker(SIGMA)
        prior = ax.plot(lambda v: gauss(v, 0, 1), x_range=[-3, 3]).set_stroke(AMBER, width=8)
        prior_label = text("prior", 22, AMBER, weight=m.SEMIBOLD).move_to(ax.c2p(-2.15, 0.42))
        q = m.always_redraw(lambda: ax.plot(lambda v: gauss(v, mu_t.get_value(), sd_t.get_value()), x_range=[-3, 3]).set_stroke(TEAL, width=4.5))
        def place_q_label():
            mu, sd = mu_t.get_value(), sd_t.get_value()
            return text("encoder's cloud", 22, TEAL, weight=m.SEMIBOLD).move_to(ax.c2p(mu + 0.6 * sd + 1.15, gauss(mu, mu, sd) + 0.08))

        q_label = m.always_redraw(place_q_label)
        kl_num = m.DecimalNumber(KL_START, num_decimal_places=2, mob_class=Mono, font_size=30, color=AMBER)
        kl_num.add_updater(lambda d: d.set_value(max(float(kl_normal(mu_t.get_value(), sd_t.get_value())), 0.0)))
        kl_read = m.VGroup(text("KL =", 26, AMBER, weight=m.BOLD), kl_num, text("nats", 22, MUTED)).arrange(m.RIGHT, buff=0.18)
        kl_read.move_to([3.15, 0.45, 0])
        kl_read[2].align_to(kl_read[0], m.DOWN)
        params = code(f"centre {MU}, spread {SIGMA}", 22, TEAL).next_to(ax, m.DOWN, buff=0.5)

        # packing the clouds
        plane = Plane(3.0, [-4.0, -1.45, 0], labels=False)
        prng = np.random.default_rng(11)
        spots = [(-1.6, 1.4), (1.5, 1.3), (0.9, -1.5), (-1.4, -1.2), (0.1, 0.3)]
        loose = m.VGroup()
        packed = m.VGroup()
        for (cx, cy), (a, b) in zip(spots, prng.normal(0, 0.35, (5, 2))):
            loose.add(m.Ellipse(width=0.22, height=0.18).set_stroke(TEAL, width=2).set_fill(TEAL, opacity=0.2).move_to(plane.p(cx, cy)))
            packed.add(m.Ellipse(width=0.85, height=0.72).set_stroke(TEAL, width=2).set_fill(TEAL, opacity=0.12).move_to(plane.p(a, b)))
        prior_ring = m.DashedVMobject(m.Circle(radius=2 * plane.unit).set_stroke(AMBER, width=2.5), num_dashes=30).move_to(plane.p(0, 0))
        packed_note = text("no gaps", 22, GOOD, weight=m.SEMIBOLD).next_to(plane.box, m.RIGHT, buff=0.2)
        bound = m.VGroup(tex(r"\mathrm{ELBO} \le \log p(x)", 36, INK), text("the evidence lower bound", 22, MUTED)).arrange(m.RIGHT, buff=0.35).move_to([0, -3.45, 0])

        self.say(
            "[A variational autoencoder is trained to maximise one number]: [the ELBO.]",
            self.new_header("THE ELBO", "What a VAE maximises"),
            m.Write(formula, run_time=1.8),
        )
        self.say(
            "[The first term is the reconstruction]: [how likely the decoder makes the original face], [from a code sampled from the cloud.]",
            m.AnimationGroup(m.GrowFromCenter(brace1), m.FadeIn(label1)),
            m.AnimationGroup(m.FadeIn(orig, shift=0.2 * m.RIGHT), m.FadeIn(approx), m.FadeIn(rebuilt, shift=0.2 * m.LEFT), m.FadeIn(recon_note)),
            m.Circumscribe(formula[2][0:7], color=TEAL, buff=0.06),
        )
        self.say(
            "[The second is the KL divergence from episode seventeen]: [how far the encoder's cloud is from the prior], [a standard bell curve.]",
            m.AnimationGroup(m.GrowFromCenter(brace2), m.FadeIn(label2)),
            m.AnimationGroup(m.Create(ax), m.FadeIn(ticks), m.FadeIn(q), m.FadeIn(q_label)),
            m.AnimationGroup(m.Create(prior.set_z_index(-1)), m.FadeIn(prior_label)),
        )
        self.say(
            "[For example, a cloud at zero point eight, with a spread of zero point three], [costs about one point zero seven.]",
            m.FadeIn(params, shift=0.1 * m.UP),
            m.FadeIn(kl_read, shift=0.15 * m.DOWN),
        )
        self.say(
            "[Pull the cloud onto the prior], and the cost falls to zero.",
            m.AnimationGroup(mu_t.animate.set_value(0.0), sd_t.animate.set_value(1.0), m.FadeOut(params), run_time=2.6),
        )
        self.stop(q, kl_num, q_label)
        self.say(
            "[So the KL term packs the clouds of all the faces around zero], [leaving no gaps.]",
            m.AnimationGroup(m.FadeOut(recon), m.FadeIn(plane.frame), m.FadeIn(loose), m.Create(prior_ring)),
            m.AnimationGroup(*[m.Transform(a, b) for a, b in zip(loose, packed)], m.FadeIn(packed_note, shift=0.15 * m.LEFT), run_time=1.6),
        )
        self.say(
            "[It's a lower bound]: [the ELBO never exceeds the log probability of the data], so raising it lifts that too.",
            m.FadeIn(bound, shift=0.2 * m.UP),
            m.Circumscribe(bound[0], color=INK, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ new faces

    def new_faces(self):
        self.next_section("New faces", notes="Sampling codes from the prior N(0, I) and decoding them (seeded); a 7 x 7 grid of the latent space, after Kingma & Welling's 2-D manifold figures.")
        plane = Plane(3.6, [-4.3, -0.5, 0], labels=False)
        rings = m.VGroup(
            *[m.DashedVMobject(m.Circle(radius=k * plane.unit).set_stroke(AMBER, width=2.5, opacity=1.0 if k == 1 else 0.6), num_dashes=12 * k + 8).move_to(plane.p(0, 0)) for k in (1, 2)]
        )
        prior_label = text("the prior: a bell curve", 22, AMBER, weight=m.SEMIBOLD).next_to(plane.box, m.UP, buff=0.18)
        dots = m.VGroup(*[m.Dot(plane.p(a, b), radius=0.07, color=AMBER).set_z_index(3) for a, b in NEW_CODES])
        dec = funnel(VIOLET, VIOLET_FILL, "decoder", wide_left=False, width=1.5, wide=1.7, narrow=0.75).move_to([-0.95, -0.5, 0])
        d1 = arrow(plane.box.get_right(), dec.get_left(), MUTED)
        made = m.VGroup(*[face(a, b, 1.45) for a, b in NEW_CODES]).arrange_in_grid(rows=2, buff=(0.5, 0.55)).move_to([3.6, -0.5, 0])
        d2 = arrow(dec.get_right(), made.get_left(), MUTED)
        grid = m.VGroup()
        side = 7
        steps = np.linspace(-2.1, 2.1, side)
        for b in steps[::-1]:
            for a in steps:
                grid.add(face(a, b, 0.72))
        grid.arrange_in_grid(rows=side, buff=0.1).move_to([2.95, -0.55, 0])
        path = [grid[r * side + r] for r in range(side)]
        frame = m.SurroundingRectangle(path[0], buff=0.05, corner_radius=0.1).set_stroke(AMBER, width=3.5)

        self.say(
            "[Once trained, we can drop the encoder.] [Sample codes from the bell curve], [decode them], [and out come new faces.]",
            m.AnimationGroup(self.new_header("NEW FACES", "Sample, then decode"), m.FadeIn(plane.frame), m.Create(rings), m.FadeIn(prior_label)),
            m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.25),
            m.AnimationGroup(m.GrowArrow(d1), m.FadeIn(dec)),
            m.AnimationGroup(m.GrowArrow(d2), m.LaggedStart(*[m.FadeIn(f, scale=0.7) for f in made], lag_ratio=0.2)),
        )
        self.say(
            "[Lay the codes out on a grid], [and one face morphs smoothly into the next.]",
            m.AnimationGroup(m.FadeOut(made), m.FadeOut(d2), m.LaggedStart(*[m.FadeIn(f, scale=0.6) for f in grid], lag_ratio=0.01, run_time=1.4)),
            m.Succession(m.FadeIn(frame), *[frame.animate(run_time=0.4).move_to(p) for p in path[1:]]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the GAN

    def gan(self):
        self.next_section("GANs", notes="Goodfellow et al., 2014. Generator (noise → face) versus discriminator (face → probability it is real); a forger and a detective, trained in turns. Scores illustrative.")
        source = cite("Goodfellow et al., 2014").move_to([5.3, 3.2, 0])
        noise = vector(AMBER, ["0.7", "−1.3", "0.2", "−0.5"], size=22).move_to([-6.1, 0.85, 0])
        noise_label = text("noise", 22, AMBER, weight=m.SEMIBOLD).next_to(noise, m.DOWN, buff=0.2)
        gen = node(variant("sparkle", VIOLET, VIOLET_FILL), "Generator", "draws faces", VIOLET, VIOLET_FILL, width=2.4, height=1.9).move_to([-3.75, 0.85, 0])
        fake_at_ = [-0.95, 0.85, 0]
        fakes = [garbled(1.5, VIOLET, VIOLET_FILL, seed=3), rough(1.5), face(1.2, 0.4, 1.5)]
        for f in fakes:
            f.move_to(fake_at_)
        fake = fakes[0]
        real = face(-0.6, 1.1, 1.5, SKY, SKY_FILL).move_to([-0.95, -1.75, 0])
        real_label = text("real", 22, SKY, weight=m.SEMIBOLD).next_to(real, m.DOWN, buff=0.15)
        fake_label = text("fake", 22, VIOLET, weight=m.SEMIBOLD).next_to(fake, m.UP, buff=0.15)
        disc = node(variant("search", PINK, PINK_FILL), "Discriminator", "real or fake?", PINK, PINK_FILL, width=2.6, height=1.9).move_to([2.25, -0.45, 0])
        g1 = arrow(noise.get_right(), gen.get_left(), MUTED)
        g2 = arrow(gen.get_right(), fake.get_left(), MUTED)
        f1 = arrow(fake.get_right(), disc.get_left() + [0, 0.35, 0], MUTED)
        r1 = arrow(real.get_right(), disc.get_left() + [0, -0.35, 0], MUTED)

        def score(label, value, color):
            return m.VGroup(text(label, 22, color, weight=m.SEMIBOLD), code(value, 26, INK)).arrange(m.RIGHT, buff=0.3)

        chance = text("chance it's real", 22, PINK, weight=m.SEMIBOLD)
        real_score = score("real", "0.90", SKY)
        fake_scores = [score("fake", v, VIOLET) for v in ("0.05", "0.30", "0.50")]
        scores = m.VGroup(chance, real_score, fake_scores[0]).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).next_to(disc, m.RIGHT, buff=0.45)
        fake_scores[0][1].align_to(real_score[1], m.LEFT)
        for s in fake_scores[1:]:
            s.move_to(fake_scores[0], aligned_edge=m.LEFT)
            s[1].align_to(real_score[1], m.LEFT)
        forger = text("the forger", 24, VIOLET, weight=m.BOLD).next_to(gen, m.DOWN, buff=0.25)
        detective = text("the detective", 24, PINK, weight=m.BOLD).next_to(disc, m.DOWN, buff=0.25)
        turns = pill("they train in turns: catch the fakes, then fool the detective", MUTED, CARD, 22).move_to([0, -3.35, 0])

        self.say(
            "[Generative adversarial networks take a different route]: [a game between two networks.]",
            m.AnimationGroup(self.new_header("GANs", "A forger versus a detective"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(gen, shift=0.2 * m.RIGHT), m.FadeIn(disc, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[The generator turns random noise into a face.]",
            m.AnimationGroup(m.FadeIn(noise), m.FadeIn(noise_label), m.GrowArrow(g1), m.GrowArrow(g2), m.FadeIn(fake, scale=0.7), m.FadeIn(fake_label)),
        )
        self.say(
            "[The discriminator sees a face, real or generated], [and says how likely it is to be real.]",
            m.AnimationGroup(m.FadeIn(real, shift=0.2 * m.UP), m.FadeIn(real_label), m.GrowArrow(f1), m.GrowArrow(r1)),
            m.LaggedStart(m.FadeIn(chance), m.FadeIn(real_score, shift=0.15 * m.LEFT), m.FadeIn(fake_scores[0], shift=0.15 * m.LEFT), lag_ratio=0.35),
        )
        self.say(
            "[Like a detective and a forger]: [the detective learns to catch fakes], [the forger learns to fool it.]",
            m.AnimationGroup(m.FadeIn(forger, shift=0.15 * m.UP), m.FadeIn(detective, shift=0.15 * m.UP)),
            m.Circumscribe(disc, color=PINK, buff=0.1),
            m.Circumscribe(gen, color=VIOLET, buff=0.1),
        )
        self.say(
            "[They take turns], [and the fakes get better and better.]",
            m.FadeIn(turns, shift=0.2 * m.UP),
            m.Succession(
                m.AnimationGroup(m.Transform(fake, fakes[1]), m.Transform(fake_scores[0][1], fake_scores[1][1]), run_time=1.0),
                m.AnimationGroup(m.Transform(fake, fakes[2]), m.Transform(fake_scores[0][1], fake_scores[2][1]), run_time=1.0),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the minimax game

    def minimax(self):
        self.next_section(
            "The minimax game",
            notes="min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]. One feature (smile): real N(0.6, 0.45^2), generator from N(-1.5, 0.35^2) onto it (illustrative); the optimal D* = p_data/(p_data + p_g) flattens to 1/2 (Goodfellow et al., 2014).",
        )
        formula = m.MathTex(
            r"\min_G \max_D", r"\;\mathbb{E}_{x}\big[\log D(x)\big]", "+", r"\mathbb{E}_{z}\big[\log\big(1 - D(G(z))\big)\big]", font_size=44, color=INK
        ).move_to([0, 2.0, 0])
        formula[1].set_color(SKY)
        formula[3].set_color(VIOLET)
        brace1 = m.Brace(formula[1], m.DOWN, buff=0.24).set_color(SKY)
        brace2 = m.Brace(formula[3], m.DOWN, buff=0.24).set_color(VIOLET)
        label1 = text("real faces called real", 22, SKY, weight=m.SEMIBOLD).next_to(brace1, m.DOWN, buff=0.08)
        label2 = text("generated faces called fake", 22, VIOLET, weight=m.SEMIBOLD).next_to(brace2, m.DOWN, buff=0.08)
        d_up = text("D: push it up", 22, PINK, weight=m.BOLD)
        g_down = text("G: push it down", 22, VIOLET, weight=m.BOLD)
        roles = m.VGroup(g_down, d_up).arrange(m.RIGHT, buff=0.4).next_to(formula[0], m.DOWN, buff=0.25).align_to(formula, m.LEFT).shift(0.1 * m.LEFT)

        # one feature, two spreads, the best discriminator (axes start at the left: smile + 3)
        x0 = 0.6
        top = axes([0, 6.2, 1], [0, 1.25, 0.5], 9.0, 1.7).move_to([x0, -0.65, 0])
        bot = m.Axes(
            x_range=[0, 6.2, 1], y_range=[0, 1.1, 0.5], x_length=9.0, y_length=1.45,
            axis_config={"color": FAINT, "stroke_width": 2.5, "include_tip": False},
        ).move_to([x0, -2.75, 0])

        def tp(v, y):
            return top.c2p(v + 3, y)

        def bp(v, y):
            return bot.c2p(v + 3, y)

        smile_label = text("smile", 22, MUTED).next_to(tp(2.55, 0), m.DOWN, buff=0.22)
        dens_label = text("how common", 22, MUTED).next_to(tp(-3, 0.6), m.LEFT, buff=0.25)
        d_ticks = m.VGroup(*[code(s, 20, MUTED).next_to(bp(-3, v), m.LEFT, buff=0.15) for s, v in (("0", 0), ("0.5", 0.5), ("1", 1))])
        d_label = tex("D(x)", 30, PINK).next_to(bp(-3, 0.5), m.LEFT, buff=0.75)
        half = m.DashedLine(bp(-3, 0.5), bp(3, 0.5), dash_length=0.1).set_stroke(FAINT, width=2)
        t = m.ValueTracker(0.0)
        real_curve = top.plot(lambda u: gauss(u - 3, REAL_MU, REAL_SD), x_range=[0, 6]).set_stroke(SKY, width=8).set_z_index(-1)
        real_tag = text("real", 22, SKY, weight=m.SEMIBOLD).move_to(tp(1.75, 0.85))
        fake_curve = m.always_redraw(lambda: top.plot(lambda u: gauss(u - 3, *fake_at(t.get_value())), x_range=[0, 6]).set_stroke(VIOLET, width=4))
        fake_tag = m.always_redraw(lambda: text("fakes", 22, VIOLET, weight=m.SEMIBOLD).move_to(tp(fake_at(t.get_value())[0] - 1.0, 0.95)))
        d_curve = m.always_redraw(lambda: bot.plot(lambda u: d_star(u - 3, t.get_value()), x_range=[0, 6]).set_stroke(PINK, width=4.5))
        says_real = text("says real", 22, PINK, weight=m.SEMIBOLD).move_to(bp(1.9, 0.8))
        says_fake = text("says fake", 22, PINK, weight=m.SEMIBOLD).move_to(bp(-1.5, 0.3))
        verdict = m.VGroup(text("D(x) =", 24, PINK, weight=m.BOLD), code("0.50", 26, INK), text("everywhere: a coin flip", 22, MUTED)).arrange(m.RIGHT, buff=0.18)
        verdict.move_to(bp(0.9, 0.82))

        self.say(
            "[Here's the whole game in one formula.]",
            m.AnimationGroup(self.new_header("MINIMAX", "One value, two players"), m.Write(formula, run_time=1.8)),
        )
        self.say(
            "[The first term rewards the discriminator for calling real faces real.]",
            m.AnimationGroup(m.GrowFromCenter(brace1), m.FadeIn(label1, shift=0.1 * m.UP)),
        )
        self.say(
            "[The second rewards it for calling generated faces fake.]",
            m.AnimationGroup(m.GrowFromCenter(brace2), m.FadeIn(label2, shift=0.1 * m.UP)),
        )
        self.say(
            "[The discriminator pushes this value up], [and the generator pushes it down.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(brace1, brace2, label1, label2)), m.FadeIn(d_up, shift=0.1 * m.UP), m.Circumscribe(formula[0][3:], color=PINK, buff=0.06)),
            m.AnimationGroup(m.FadeIn(g_down, shift=0.1 * m.UP), m.Circumscribe(formula[0][:3], color=VIOLET, buff=0.06)),
        )
        self.say(
            "[Picture one feature, say the smile.] [Real faces have this spread], [and the generator's fakes start far off.]",
            m.AnimationGroup(m.Create(top), m.FadeIn(smile_label), m.FadeIn(dens_label)),
            m.AnimationGroup(m.Create(real_curve), m.FadeIn(real_tag)),
            m.AnimationGroup(m.FadeIn(fake_curve), m.FadeIn(fake_tag)),
        )
        self.say(
            "[The best discriminator says real where real faces are common], [and fake where the fakes are.]",
            m.AnimationGroup(m.Create(bot), m.FadeIn(d_ticks), m.FadeIn(d_label), m.Create(half), m.FadeIn(d_curve), m.FadeIn(says_real)),
            m.FadeIn(says_fake, shift=0.1 * m.UP),
        )
        self.say(
            "[As the generator improves, the two spreads match], [and the best the discriminator can do is guess: one half, everywhere.]",
            m.AnimationGroup(t.animate(run_time=3.0).set_value(1.0), m.FadeOut(says_real), m.FadeOut(says_fake)),
            m.FadeIn(verdict, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.stop(fake_curve, fake_tag, d_curve)
        self.clear()

    # ------------------------------------------------------------------ trade-offs

    def tradeoffs(self):
        self.next_section(
            "Trade-offs",
            notes="Mode collapse (illustrative faces). Sharp vs blurrier: the VAE-style picture is the pixel average of 64 faces with codes drawn around the hero's; the GAN-style picture is one face. Both largely overtaken by diffusion for images; latent diffusion (Rombach et al., 2022) compresses pictures with a VAE-style autoencoder.",
        )
        real_row = m.VGroup(*[face(a, b, 1.25, SKY, SKY_FILL) for a, b in REAL_ROW]).arrange(m.RIGHT, buff=0.45).move_to([0.9, 1.25, 0])
        gen_row = m.VGroup(*[face(a, b, 1.25) for a, b in COLLAPSE]).arrange(m.RIGHT, buff=0.45).move_to([0.9, -0.85, 0])
        real_name = text("real faces", 24, SKY, weight=m.BOLD).next_to(real_row, m.LEFT, buff=0.45)
        gen_name = text("generated", 24, VIOLET, weight=m.BOLD).next_to(gen_row, m.LEFT, buff=0.45)
        gen_name.align_to(real_name, m.LEFT)
        collapse = pill("✗  mode collapse: the same few faces, over and over", BAD, BAD_FILL, 22).move_to([0, -2.55, 0])

        n = 120
        sharp = picture(face_pixels(*HERO, n, PINK, PINK_FILL, line=0.07), 2.5).move_to([3.4, 0.95, 0])
        blur_rgba = np.mean([face_pixels(a, b, n, TEAL, TEAL_FILL, line=0.07) for a, b in BLUR_CODES], axis=0)
        blurry = picture(blur_rgba, 2.5).move_to([-3.4, 0.95, 0])
        gan_tag = pill("GAN-style: sharp", PINK, PINK_FILL, 24).next_to(sharp, m.DOWN, buff=0.3)
        vae_tag = pill("VAE-style: blurrier", TEAL, TEAL_FILL, 24).next_to(blurry, m.DOWN, buff=0.3)
        avg_note = text("an average of 64 faces it can't tell apart", 22, MUTED).next_to(vae_tag, m.DOWN, buff=0.2)

        def bullets(items, x):
            rows = m.VGroup(*[text(s, 24, GOOD if s.startswith("✓") else BAD, weight=m.SEMIBOLD) for s in items])
            return rows.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([x, -1.75, 0])

        vae_pros = bullets(["✓  steady training", "✓  a smooth latent space"], -3.4)
        gan_pros = bullets(["✓  sharp pictures", "✗  tricky to train"], 3.4)
        diffusion = pill("for images, both largely overtaken by diffusion models", AMBER, AMBER_FILL, 22).move_to([0, -2.95, 0])
        legacy = m.VGroup(text("many diffusion models still compress pictures with a VAE first", 22, MUTED), cite("Rombach et al., 2022")).arrange(m.RIGHT, buff=0.3)
        legacy.move_to([0, -3.55, 0])

        self.say(
            "[One famous GAN failure is mode collapse]: [the generator finds a few faces that fool the discriminator], [and repeats them.]",
            m.AnimationGroup(self.new_header("TRADE-OFFS", "Sharp but fickle, smooth but blurry"), m.FadeIn(real_name), m.LaggedStart(*[m.FadeIn(f, scale=0.7) for f in real_row], lag_ratio=0.1)),
            m.AnimationGroup(m.FadeIn(gen_name), m.FadeIn(gen_row[0], scale=0.7)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, scale=0.7) for f in gen_row[1:]], lag_ratio=0.2), m.FadeIn(collapse, shift=0.2 * m.UP)),
        )
        self.play(*[m.FadeOut(x) for x in (real_row, gen_row, real_name, gen_name, collapse)], run_time=0.6)
        self.say(
            "[Still, GAN faces tend to be sharp], [while VAE faces come out blurrier]: [the VAE plays it safe, averaging faces it can't tell apart.]",
            m.AnimationGroup(m.FadeIn(sharp, scale=0.9), m.FadeIn(gan_tag, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(blurry, scale=0.9), m.FadeIn(vae_tag, shift=0.15 * m.UP)),
            m.FadeIn(avg_note, shift=0.1 * m.UP),
        )
        self.say(
            "[So VAEs train steadily, with a smooth latent space]; [GANs are sharper, but tricky to train.]",
            m.AnimationGroup(m.FadeOut(avg_note), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.UP) for r in vae_pros], lag_ratio=0.3)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.UP) for r in gan_pros], lag_ratio=0.3),
        )
        self.say(
            "[For images, both have largely been overtaken by diffusion models], [though many of those still use a VAE to compress pictures first.]",
            m.FadeIn(diffusion, shift=0.2 * m.UP),
            m.FadeIn(legacy, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Encode, sample, compete.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Encode", "Sample", "Compete"), (TEAL, AMBER, PINK))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Learn the hidden numbers behind the data, and you can make new data.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [encode], [sample], [compete.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Learn the hidden numbers behind the data, and you can make new data.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
