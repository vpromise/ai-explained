"""AI, Explained · Episode 35 · Diffusion Models

Running example: a picture dissolving into static and back. The picture is drawn here in code
(no photograph): a 64 x 64 sunset over hills, pixel values scaled to [-1, 1].

Everything shown is computed below with seeded numpy:

- The noise schedule of Ho et al., 2020 (DDPM): T = 1000 steps, beta rising linearly from
  0.0001 to 0.02, alpha_bar_t = prod_{s<=t} (1 - beta_s).
- The forward process for real: the hook runs the step-by-step chain
  x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) z_t; the formula section uses the closed form
  x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps with one fixed noise sample.
- The reverse process for real (DDPM ancestral sampling, Ho et al. Algorithm 2) with an exact
  denoiser. For the pictures, the "dataset" is four synthetic pictures, each a narrow Gaussian
  (sd 0.05 per pixel) around it, so E[x_0 | x_t] and the predicted noise are analytic. That
  denoiser can only bring back one of its four pictures (it is not a trained network; the
  narration says so). The hook's reverse starts from the static the forward chain ended on
  (seed chosen so it lands on the sunset); the later run starts from fresh static and lands on
  the moon.
- A 2-D mixture of 74 Gaussians shaped like a sun and a crescent moon: its score at every
  noise level is analytic (a mixture again), so the score arrows, the one-step guess, the
  thousand-step reverse runs and classifier-free guidance are exact computations.
- Guidance toy: the "moon" prompt is deliberately weak, p(x | moon) = 0.75 moon + 0.25 sun
  (illustrative), the unconditional model is 0.5 / 0.5. Guided noise
  eps~ = eps(x) + s (eps(x, c) - eps(x)) (Ho & Salimans, 2022).
- Latent diffusion numbers: Stable Diffusion v1 encodes 512 x 512 x 3 images to 64 x 64 x 4
  latents (Rombach et al., 2022).

Every number the narration quotes is asserted below.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the schedule

T = 1000
BETA = np.linspace(1e-4, 0.02, T)  # beta_1 .. beta_T
ALPHA = 1 - BETA
ABAR = np.concatenate([[1.0], np.cumprod(ALPHA)])  # alpha_bar_0 = 1 .. alpha_bar_T


def signal(t):
    """The picture's weight at step t: sqrt(alpha_bar_t)."""
    return float(np.sqrt(ABAR[int(round(t))]))


def noise_w(t):
    """The noise's weight at step t: sqrt(1 - alpha_bar_t)."""
    return float(np.sqrt(1 - ABAR[int(round(t))]))


assert T == 1000 and BETA[0] == 1e-4 and abs(BETA[-1] - 0.02) < 1e-12  # "one in ten thousand", "two percent", "a thousand steps"
assert round(signal(500), 2) == 0.28  # "at step five hundred ... zero point two eight"
assert signal(1000) < 0.01  # "by the last step, below one percent"
assert all(abs(signal(t) ** 2 + noise_w(t) ** 2 - 1) < 1e-12 for t in range(T + 1))  # "squares add up to one"

# ---------------------------------------------------------------------------- the pictures

N = 64  # pixels per side
SIGMA_D = 0.05  # each training picture is a narrow bell curve around it (a little grain)
EVERY = 5  # keep one frame in five for the films


def _rgb(h):
    return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], float) / 255


def _soft(d, w):
    return 1 / (1 + np.exp(np.clip(d / w, -60, 60)))


def _grid():
    s = (np.arange(N) + 0.5) / N * 2 - 1
    return np.meshgrid(s, -s)  # u to the right, v up


def _ramp(v, stops):
    vs = np.array([s[0] for s in stops])
    cs = np.array([_rgb(s[1]) for s in stops])
    return np.stack([np.interp(v, vs, cs[:, c]) for c in range(3)], axis=-1)


def _paint(img, mask, color):
    return img * (1 - mask[..., None]) + _rgb(color) * mask[..., None]


def sunset():
    u, v = _grid()
    img = _ramp(v, [(-1, "#FBBF24"), (-0.25, "#FBBF24"), (0.05, "#F472B6"), (0.45, "#7E3A8C"), (1, "#24184A")])
    img = _paint(img, _soft(np.hypot(u - 0.3, v + 0.12) - 0.3, 0.02), "#FEF3C7")
    img = _paint(img, _soft(v - (-0.3 + 0.16 * np.cos(2.4 * u + 0.5)), 0.02), "#0F5E58")
    return _paint(img, _soft(v - (-0.6 + 0.2 * np.cos(1.7 * u - 1.4)), 0.02), "#16A394")


def moon():
    u, v = _grid()
    img = _ramp(v, [(-1, "#0B1A36"), (-0.35, "#0B1A36"), (-0.34, "#2A3F78"), (1, "#0A0F24")])
    rng = np.random.default_rng(11)
    for _ in range(14):
        x, y = rng.uniform(-0.95, 0.95), rng.uniform(-0.2, 0.95)
        img = _paint(img, _soft(np.hypot(u - x, v - y) - 0.025, 0.01), "#CBD5E1")
    disc = _soft(np.hypot(u + 0.3, v - 0.42) - 0.28, 0.015)
    bite = _soft(np.hypot(u + 0.17, v - 0.5) - 0.25, 0.015)
    img = _paint(img, disc * (1 - bite), "#F1F5F9")
    sea = _soft(v + 0.35, 0.01)
    img = _paint(img, sea, "#102A4C")
    streak = sea * _soft(np.abs(u + 0.3) - 0.12 * (1 - (v + 1) / 0.65) - 0.03, 0.02) * (np.cos(40 * v) > 0)
    return _paint(img, streak * 0.8, "#93C5FD")


def tree():
    u, v = _grid()
    img = _ramp(v, [(-1, "#BAE6FD"), (0, "#7DD3FC"), (1, "#38BDF8")])
    ground = _soft(v - (-0.5 + 0.06 * np.sin(2.5 * u)), 0.02)
    img = _paint(img, ground, "#4ADE80")
    img = _paint(img, ground * _soft(-(v + 0.75), 0.05) * 0.5, "#15803D")
    img = _paint(img, _soft(np.abs(u) - 0.07, 0.01) * _soft(v - 0.05, 0.01) * _soft(-(v + 0.55), 0.01), "#7C4A1E")
    canopy = np.maximum.reduce([
        _soft(np.hypot(u, v - 0.28) - 0.36, 0.02),
        _soft(np.hypot(u + 0.25, v - 0.1) - 0.25, 0.02),
        _soft(np.hypot(u - 0.25, v - 0.1) - 0.25, 0.02),
    ])
    img = _paint(img, canopy, "#15803D")
    return _paint(img, canopy * _soft(np.hypot(u + 0.1, v - 0.38) - 0.2, 0.06) * 0.7, "#4ADE80")


def mountains():
    u, v = _grid()
    img = _ramp(v, [(-1, "#FDE68A"), (0, "#FCD9B6"), (1, "#A5B4FC")])
    near = _soft(v - (0.5 - 1.2 * np.abs(u + 0.3)), 0.015)
    far = _soft(v - (0.2 - 1.0 * np.abs(u - 0.45)), 0.015)
    img = _paint(img, far, "#64748B")
    img = _paint(img, far * _soft(-(v - 0.02), 0.01), "#F8FAFC")
    img = _paint(img, near, "#334155")
    img = _paint(img, near * _soft(-(v - 0.25), 0.01), "#F1F5F9")
    return _paint(img, _soft(v + 0.62, 0.01), "#1E3A5F")


PICTURES = [sunset(), moon(), tree(), mountains()]
HERO, MOON_PIC = 0, 1
MU = np.stack([2 * p.ravel() - 1 for p in PICTURES])  # the four pictures, pixels in [-1, 1]
assert len(PICTURES) == 4  # "a tiny set of four pictures"


def pix(x):
    """[-1, 1] pixel values → an RGB array to show (clipped, as images are)."""
    return np.round(np.clip((np.asarray(x).reshape(N, N, 3) + 1) / 2, 0, 1) * 255).astype(np.uint8)


def forward_chain(x0, seed):
    """x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) z_t, one fresh z every step."""
    rng = np.random.default_rng(seed)
    x, frames = x0.copy(), [x0.copy()]
    for t in range(1, T + 1):
        x = np.sqrt(ALPHA[t - 1]) * x + np.sqrt(BETA[t - 1]) * rng.normal(size=x.shape)
        if t % EVERY == 0:
            frames.append(x.copy())
    return frames  # frame i = step EVERY * i


def posterior(xt, t):
    """For the four-picture dataset: which picture (responsibilities) and E[x_0 | x_t]."""
    a = ABAR[t]
    var = a * SIGMA_D**2 + 1 - a
    logit = -((xt[None] - np.sqrt(a) * MU) ** 2).sum(1) / (2 * var)
    r = np.exp(logit - logit.max())
    r /= r.sum()
    shrink = SIGMA_D**2 * np.sqrt(a) / var
    return r, r @ (MU + shrink * (xt[None] - np.sqrt(a) * MU))


def reverse(xT, seed):
    """DDPM ancestral sampling with the exact predicted noise; frames, guesses of x_0, responsibilities."""
    rng = np.random.default_rng(seed)
    x = xT.copy()
    r, g = posterior(x, T)
    frames, guesses, resp = [x.copy()], [g], [r]
    for t in range(T, 0, -1):
        r, g = posterior(x, t)
        e = (x - np.sqrt(ABAR[t]) * g) / np.sqrt(1 - ABAR[t])
        x = (x - BETA[t - 1] / np.sqrt(1 - ABAR[t]) * e) / np.sqrt(ALPHA[t - 1])
        if t > 1:
            x = x + np.sqrt(BETA[t - 1]) * rng.normal(size=x.shape)
        if (t - 1) % EVERY == 0:
            r, g = posterior(x, max(t - 1, 1))
            frames.append(x.copy())
            guesses.append(g)
            resp.append(r)
    return frames, guesses, resp  # frame i = step T - EVERY * i


FWD = forward_chain(MU[HERO], seed=7)
HOOK_BACK, _, HOOK_R = reverse(FWD[-1], seed=4)
assert int(np.argmax(HOOK_R[-1])) == HERO  # the reverse from the hook's static lands on the sunset
FWD_PIX = [pix(x) for x in FWD]
HOOK_PIX = [pix(x) for x in HOOK_BACK]

FRESH = np.random.default_rng(21).normal(size=MU.shape[1])  # brand new static
BACK, BACK_GUESS, BACK_R = reverse(FRESH, seed=32)
assert int(np.argmax(BACK_R[-1])) == MOON_PIC  # "this time, we got the moon"
assert BACK_R[0].max() < 0.4  # at the start the guess is undecided: a blurry average
BACK_COMMIT = T - EVERY * next(i for i, r in enumerate(BACK_R) if r.max() > 0.99)
BACK_PIX = [pix(x) for x in BACK]
GUESS_PIX = [pix(g) for g in BACK_GUESS]

EPS = np.random.default_rng(1).normal(size=MU.shape[1])  # the one noise sample of the formula section


def closed_form(t):
    t = int(round(t))
    return pix(signal(t) * MU[HERO] + noise_w(t) * EPS)


# ---------------------------------------------------------------------------- the 2-D world

SIG = 0.06  # each bump's spread


def _sun(c=(-1.25, 0.15)):
    pts = [(0.0, 0.0)]
    pts += [(0.2 * np.cos(a), 0.2 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    pts += [(0.38 * np.cos(a), 0.38 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 13)[:-1] + 0.2]
    for a in np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8:
        pts += [(r * np.cos(a), r * np.sin(a)) for r in (0.62, 0.78)]
    return np.array(pts) + np.array(c)


def _moon(c=(1.3, 0.15), r=0.68, bite=(0.34, 0.14), r2=0.56, step=0.12):
    g = np.arange(-1, 1.001, step / r)
    pts = [(x * r, y * r) for x in g for y in g if np.hypot(x * r, y * r) <= r - 0.04 and np.hypot(x * r - bite[0], y * r - bite[1]) >= r2 + 0.02]
    return np.array(pts) + np.array(c)


SUN_C, MOON_C = _sun(), _moon()
CENTRES = np.concatenate([SUN_C, MOON_C])
IS_MOON = np.r_[np.zeros(len(SUN_C)), np.ones(len(MOON_C))].astype(bool)
assert len(CENTRES) == 74


def mix_weights(moon_share):
    return np.where(IS_MOON, moon_share / IS_MOON.sum(), (1 - moon_share) / (~IS_MOON).sum())


W_ALL = mix_weights(0.5)  # the data: half sun, half moon
W_MOON = mix_weights(0.75)  # the weak "moon" prompt: a quarter of its examples are suns


def _resp(x, t, w):
    a = ABAR[t]
    var = a * SIG**2 + 1 - a
    diff = np.sqrt(a) * CENTRES[None] - x[:, None]
    logit = np.log(w)[None] - (diff**2).sum(-1) / (2 * var)
    r = np.exp(logit - logit.max(1, keepdims=True))
    return r / r.sum(1, keepdims=True), diff, var, a


def score(x, t, w=W_ALL):
    """∇ log p_t(x), exactly, for the noised mixture."""
    r, diff, var, _ = _resp(x, t, w)
    return (r[..., None] * diff).sum(1) / var


def eps2(x, t, w=W_ALL):
    """The ideal noise prediction: the score, flipped and scaled."""
    return -np.sqrt(1 - ABAR[t]) * score(x, t, w)


def guess2(x, t, w=W_ALL):
    """E[x_0 | x_t]: the best one-step jump to a clean point."""
    r, _, var, a = _resp(x, t, w)
    per = CENTRES[None] + SIG**2 * np.sqrt(a) / var * (x[:, None] - np.sqrt(a) * CENTRES[None])
    return (r[..., None] * per).sum(1)


def run2(n, seed, eps_fn):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    path = [x.copy()]
    for t in range(T, 0, -1):
        x = (x - BETA[t - 1] / np.sqrt(1 - ABAR[t]) * eps_fn(x, t)) / np.sqrt(ALPHA[t - 1])
        if t > 1:
            x = x + np.sqrt(BETA[t - 1]) * rng.normal(size=x.shape)
        path.append(x.copy())
    return np.array(path)  # path[i] = the points at step T - i


def lands_on_moon(x):
    return IS_MOON[np.argmin(((x[:, None] - CENTRES[None]) ** 2).sum(-1), 1)]


def guided(s):
    return lambda x, t: eps2(x, t) + s * (eps2(x, t, W_MOON) - eps2(x, t))


_rng = np.random.default_rng(5)
_k = _rng.choice(len(CENTRES), 360, p=W_ALL)
DATA2 = CENTRES[_k] + SIG * _rng.normal(size=(360, 2))
DATA2_MOON = IS_MOON[_k]

N_POINTS = 500
UNCOND = run2(N_POINTS, 1, eps2)
ONE_STEP = guess2(UNCOND[0], T)
_centre = ONE_STEP.mean(0)
assert N_POINTS == 500  # "five hundred points of pure noise"
assert ONE_STEP.std(0).max() < 0.05  # "every point lands on the same spot"
assert SUN_C[:, 0].mean() < _centre[0] < MOON_C[:, 0].mean()  # "between the sun and the moon"
assert np.sqrt(((CENTRES - _centre) ** 2).sum(1)).min() > 5 * SIG  # "where there's no data at all": over five bump widths from any bump
assert np.sqrt(((UNCOND[-1][:, None] - CENTRES[None]) ** 2).sum(-1).min(1)).mean() < 0.1  # it flows into the shapes

N_GUIDE = 400
GUIDE = {s: run2(N_GUIDE, 2, guided(s)) for s in (1, 4)}  # the same static for both scales
SHARE = {s: float(lands_on_moon(p[-1]).mean()) for s, p in GUIDE.items()}
assert 0.70 <= SHARE[1] <= 0.80  # "about three out of four points land on the moon"
assert SHARE[4] >= 0.98  # "nearly every point lands on the moon"

# latent diffusion (Stable Diffusion v1)
PIXELS, LATENTS = 512 * 512 * 3, 64 * 64 * 4
assert 750_000 < PIXELS < 800_000  # "almost eight hundred thousand numbers"
assert PIXELS == 48 * LATENTS  # "sixty-four by sixty-four by four: forty-eight times fewer"

# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def formula(*parts, size=44):
    return m.MathTex(*parts, font_size=size, color=INK)


def picture(arr, height):
    img = m.ImageMobject(arr, scale_to_resolution=0, resampling_algorithm=0)
    img.height = height
    return img


def edge(height, color=EDGE, stroke=2.5):
    return m.Rectangle(width=height, height=height).set_stroke(color, width=stroke).set_fill(BG, opacity=0)


def cite(s):
    return text(s, 20, MUTED)


def mini_bell(color=PINK, width=0.9, height=0.42):
    xs = np.linspace(-1, 1, 60)
    curve = m.VMobject().set_points_as_corners([[x * width / 2, height * np.exp(-8 * x**2) - height / 2, 0] for x in xs])
    base = m.Line([-width / 2, -height / 2, 0], [width / 2, -height / 2, 0]).set_stroke(FAINT, width=2)
    return m.VGroup(base, curve.set_stroke(color, width=3.5))


def unet(color=VIOLET, fill=VIOLET_FILL):
    """A tiny U-Net: blocks shrinking down, then growing back, with skip links across."""
    heights = [0.9, 0.62, 0.36, 0.62, 0.9]
    tops = [0.45, 0.31, 0.18, 0.31, 0.45]
    bars = m.VGroup(*[
        m.RoundedRectangle(corner_radius=0.05, width=0.2, height=h).set_stroke(color, width=2.5).set_fill(fill, opacity=1).move_to([(i - 2) * 0.34, top - h / 2, 0])
        for i, (h, top) in enumerate(zip(heights, tops))
    ])
    skips = m.VGroup(
        m.DashedLine(bars[0].get_top() + 0.12 * m.UP, bars[4].get_top() + 0.12 * m.UP, dash_length=0.06).set_stroke(color, width=2),
        m.DashedLine(bars[1].get_top() + 0.12 * m.UP, bars[3].get_top() + 0.12 * m.UP, dash_length=0.06).set_stroke(color, width=2),
    )
    return m.VGroup(skips, bars)


def funnel(wide, narrow, length, color=BLUE, fill="#13203A", widening=False):
    a, b = (narrow, wide) if widening else (wide, narrow)
    pts = [[-length / 2, a / 2, 0], [length / 2, b / 2, 0], [length / 2, -b / 2, 0], [-length / 2, -a / 2, 0]]
    return m.Polygon(*pts).set_stroke(color, width=2.5).set_fill(fill, opacity=1)


class Plane:
    """A 2-D panel: data coordinates → screen, with points clipped to the panel."""

    def __init__(self, centre, width, height, scale):
        self.c = np.array([centre[0], centre[1], 0.0])
        self.w, self.h, self.s = width, height, scale
        self.box = card(width, height, EDGE, CARD, stroke=2).move_to(self.c).set_z_index(-2)

    def p(self, xy):
        xy = np.atleast_2d(xy)
        x = np.clip(xy[:, 0] * self.s, -self.w / 2 + 0.08, self.w / 2 - 0.08)
        y = np.clip(xy[:, 1] * self.s, -self.h / 2 + 0.08, self.h / 2 - 0.08)
        return np.c_[x + self.c[0], y + self.c[1], np.zeros(len(x))]

    def data(self, opacity=0.9, radius=0.028):
        g = m.VGroup(*[m.Dot(q, radius=radius, color=SKY if mo else AMBER) for q, mo in zip(self.p(DATA2), DATA2_MOON)])
        return g.set_opacity(opacity)

    def swarm(self, start, color=INK, radius=0.032):
        return m.VGroup(*[m.Dot(q, radius=radius, color=color) for q in self.p(start)]).set_z_index(3)

    def follow(self, dots, positions):
        """Move the dots to positions() every frame."""

        def place(g):
            for d, q in zip(g, self.p(positions())):
                d.move_to(q)

        dots.add_updater(place)
        return dots

    def field(self, t, xs=np.arange(-2.75, 2.76, 0.5), ys=np.arange(-2.0, 2.01, 0.5)):
        gx, gy = np.meshgrid(xs, ys)
        pts = np.c_[gx.ravel(), gy.ravel()]
        s = score(pts, int(round(t)))
        size = np.linalg.norm(s, axis=1)
        length = 0.42 * np.tanh(size / 4)
        heads = pts * self.s + s / np.maximum(size, 1e-9)[:, None] * length[:, None]
        arrows = m.VGroup()
        for p0, p1, L in zip(pts, heads, length):
            a = self.c + np.array([p0[0] * self.s, p0[1] * self.s, 0])
            b = self.c + np.array([p1[0], p1[1], 0])
            if L > 0.03:
                arrows.add(m.Arrow(a, b, buff=0, stroke_width=3, max_tip_length_to_length_ratio=0.4, color=VIOLET))
        return arrows.set_z_index(2)


def path_at(path, step):
    """The points of a reverse run at a (fractional) step, by linear interpolation."""
    i = (T - step)
    lo = int(np.clip(np.floor(i), 0, T))
    hi = min(lo + 1, T)
    a = float(np.clip(i - lo, 0, 1))
    return (1 - a) * path[lo] + a * path[hi]


def ease_from(power):
    """A rate function: step = T (1 - p)^power when a tracker goes from T to 0."""
    return lambda p: 1 - (1 - p) ** power


def ease_to(power):
    """A rate function: step = T p^power when a tracker goes from 0 to T."""
    return lambda p: p**power


# ---------------------------------------------------------------------------- the episode


class Ep35Diffusion(Episode):
    number = 35

    def lesson(self):
        self.hook()
        self.forward()
        self.schedule()
        self.training()
        self.the_score()
        self.small_steps()
        self.back_again()
        self.guidance()
        self.latent()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes="A synthetic picture dissolves into static by the real forward chain (1000 steps, linear schedule), then comes back by a real "
            "reverse diffusion run (DDPM sampling with the exact denoiser of a four-picture dataset), starting from the static the forward "
            "chain ended on; its seed was chosen so it lands on this picture. Not the forward film played backwards.",
        )
        c, h = np.array([0, 0.4, 0]), 4.3
        step = m.ValueTracker(0.0)
        phase = {"back": False}

        def frame():
            s = step.get_value()
            arr = HOOK_PIX[int(round((T - s) / EVERY))] if phase["back"] else FWD_PIX[int(round(s / EVERY))]
            return picture(arr, h).move_to(c)

        live = m.always_redraw(frame)
        border = edge(h + 0.05).move_to(c)
        x0, x1, y = -2.15, 2.15, -2.42
        track = m.Line([x0, y, 0], [x1, y, 0]).set_stroke(EDGE, width=4)
        ends = m.VGroup(text("clean", 22, MUTED).next_to(track, m.LEFT, buff=0.25), text("static", 22, MUTED).next_to(track, m.RIGHT, buff=0.25))
        marker = m.Dot([x0, y, 0], radius=0.11, color=PINK).set_z_index(2)
        marker.add_updater(lambda d: d.move_to([x0 + (x1 - x0) * step.get_value() / T, y, 0]))
        fwd = text("forward: add a little noise, a thousand times", 24, PINK, weight=m.SEMIBOLD).move_to([0, -3.15, 0])
        rev = text("reverse: remove noise, step by step", 24, TEAL, weight=m.SEMIBOLD).move_to([0, -3.15, 0])
        honest = text("computed, not the film played backwards", 22, MUTED).move_to([0, -3.6, 0])
        statement = text("Making images by removing noise", 44, INK, weight=m.BOLD).move_to([0, -2.85, 0])

        self.say(
            "So... [here's a picture.] [Now we add a little noise, again and again, until only static is left.]",
            m.AnimationGroup(m.FadeIn(live, scale=0.95), m.FadeIn(border), m.FadeIn(track), m.FadeIn(ends), m.FadeIn(marker), run_time=1.0),
            m.AnimationGroup(m.FadeIn(fwd, shift=0.2 * m.UP), step.animate(run_time=5.0, rate_func=ease_to(1.6)).set_value(T)),
        )
        phase["back"] = True
        self.say(
            "[A diffusion model learns the way back]: [from static, it removes noise, step by step, until a picture appears.]",
            m.AnimationGroup(m.FadeOut(fwd, shift=0.2 * m.UP), m.FadeIn(rev, shift=0.2 * m.UP), marker.animate.set_color(TEAL)),
            m.AnimationGroup(m.FadeIn(honest), step.animate(run_time=5.5, rate_func=ease_from(1.6)).set_value(0)),
        )
        self.stop(live, marker)
        self.say(
            "[That's how image generators like Stable Diffusion work.] [Let's see how, with real numbers.]",
            m.AnimationGroup(*[m.FadeOut(mob) for mob in (track, ends, marker, rev, honest)], m.FadeIn(statement, shift=0.2 * m.UP)),
            m.Circumscribe(border, color=TEAL, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the forward process

    def forward(self):
        self.next_section(
            "Forward process",
            notes="Closed form x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps with one fixed noise sample (seed 1). The two weights' squares sum to 1 "
            "(variance preserving). No learning: the forward process is fixed.",
        )
        t = m.ValueTracker(0.0)
        bc, bh = np.array([-2.35, -0.45, 0]), 3.5
        big = m.always_redraw(lambda: picture(closed_form(t.get_value()), bh).move_to(bc))
        big_edge = edge(bh + 0.05).move_to(bc)
        t_num = m.DecimalNumber(0, num_decimal_places=0, mob_class=Mono, group_with_commas=False, font_size=30, color=INK)
        t_num.add_updater(lambda d: d.set_value(int(round(t.get_value()))))
        t_row = m.VGroup(tex(r"x_t", 40, INK), text("at step", 26, MUTED), t_num).arrange(m.RIGHT, buff=0.22)
        t_row.next_to(big_edge, m.UP, buff=0.25)
        t_num.add_updater(lambda d: d.next_to(t_row[1], m.RIGHT, buff=0.22))

        th = 1.5
        x0_img = m.Group(picture(pix(MU[HERO]), th), edge(th + 0.04)).move_to([-5.55, 0.65, 0])
        eps_img = m.Group(picture(pix(EPS), th), edge(th + 0.04)).move_to([-5.55, -1.75, 0])
        x0_lab = m.VGroup(tex(r"x_0", 34, TEAL), text("clean image", 22, MUTED)).arrange(m.RIGHT, buff=0.15).next_to(x0_img, m.UP, buff=0.15)
        eps_lab = m.VGroup(tex(r"\varepsilon", 40, PINK), text("noise", 22, MUTED)).arrange(m.RIGHT, buff=0.15).next_to(eps_img, m.UP, buff=0.15)
        bell = m.VGroup(mini_bell(), text("bell curve", 20, MUTED)).arrange(m.RIGHT, buff=0.15).next_to(eps_img, m.DOWN, buff=0.18)
        into = m.VGroup(
            arrow(x0_img.get_right(), big_edge.get_left() + 0.6 * m.UP, MUTED),
            arrow(eps_img.get_right(), big_edge.get_left() + 0.6 * m.DOWN, MUTED),
        )

        eq = formula(r"x_t", "=", r"\sqrt{\bar{\alpha}_t}", r"\,x_0", "+", r"\sqrt{1-\bar{\alpha}_t}", r"\,\varepsilon", size=48).move_to([3.55, 1.75, 0])
        eq[2].set_color(TEAL)
        eq[3].set_color(TEAL)
        eq[5].set_color(PINK)
        eq[6].set_color(PINK)
        pic_lab = text("picture weight", 22, TEAL, weight=m.SEMIBOLD).next_to(eq[2:4], m.DOWN, buff=0.22)
        noi_lab = text("noise weight", 22, PINK, weight=m.SEMIBOLD).next_to(eq[5:7], m.DOWN, buff=0.22)

        def bar_row(label, color, fn, y):
            name = text(label, 26, color, weight=m.SEMIBOLD).move_to([1.4, y, 0]).align_to([0.75, 0, 0], m.LEFT)
            track = m.Line([2.75, y, 0], [5.55, y, 0]).set_stroke(EDGE, width=14)
            fill = m.always_redraw(lambda: m.Line([2.75, y, 0], [2.75 + max(2.8 * fn(t.get_value()), 0.01), y, 0]).set_stroke(color, width=14))
            num = m.DecimalNumber(fn(0), num_decimal_places=2, mob_class=Mono, group_with_commas=False, font_size=28, color=color).move_to([6.25, y, 0])
            num.add_updater(lambda d: d.set_value(fn(t.get_value())))
            return m.VGroup(name, track, fill, num)

        rows = [bar_row("picture", TEAL, signal, 0.0), bar_row("noise", PINK, noise_w, -0.85)]

        def squares():
            a, b = signal(t.get_value()), noise_w(t.get_value())
            s = f"{a:.2f}² + {b:.2f}² = {a * a + b * b:.2f}"
            return code(s, 26, INK, t2c={f"{a:.2f}²": TEAL, f"{b:.2f}²": PINK}).move_to([3.6, -1.95, 0])

        check = m.always_redraw(squares)
        fixed = pill("no learning here: a fixed recipe", GOOD, GOOD_FILL, 22).move_to([3.6, -3.05, 0])

        self.say(
            "[First, the easy direction]: destroying a picture. [This is the forward process.]",
            self.new_header("STEP 1 · FORWARD", "Destroying a picture, on purpose"),
            m.AnimationGroup(m.FadeIn(big), m.FadeIn(big_edge), m.FadeIn(t_row)),
        )
        self.say(
            "[Take the clean image], [and draw noise]: [a random number for every pixel, from a bell curve.]",
            m.AnimationGroup(m.FadeIn(x0_img, shift=0.2 * m.RIGHT), m.FadeIn(x0_lab, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(eps_img, shift=0.2 * m.RIGHT), m.FadeIn(eps_lab, shift=0.2 * m.RIGHT)),
            m.FadeIn(bell, shift=0.15 * m.UP),
        )
        self.say(
            "[The noisy image at step t blends the two]: [the picture times one weight], [plus the noise times another.]",
            m.AnimationGroup(m.Write(eq), m.GrowArrow(into[0]), m.GrowArrow(into[1])),
            m.AnimationGroup(m.FadeIn(pic_lab, shift=0.15 * m.UP), m.FadeIn(rows[0])),
            m.AnimationGroup(m.FadeIn(noi_lab, shift=0.15 * m.UP), m.FadeIn(rows[1])),
        )
        self.say(
            "[As t grows, the picture's weight shrinks, and the noise weight grows.]",
            t.animate(run_time=4.5, rate_func=ease_to(1.5)).set_value(T),
        )
        self.say(
            "[The two weights squared always add up to one], [so the pixel values keep the same spread.]",
            m.AnimationGroup(m.FadeIn(check, shift=0.15 * m.UP), t.animate(run_time=1.5).set_value(300)),
            m.Circumscribe(check, color=INK, buff=0.1),
        )
        self.say(
            "[There's no learning here, just a fixed recipe], [and the formula jumps straight to any step.]",
            m.FadeIn(fixed, shift=0.2 * m.UP),
            m.Succession(t.animate(run_time=0.3).set_value(800), m.Wait(0.6), t.animate(run_time=0.3).set_value(120), m.Wait(0.6), t.animate(run_time=0.3).set_value(500)),
        )
        self.wait(0.3)
        self.stop(big, t_num, check, *[r[2] for r in rows], *[r[3] for r in rows])
        self.clear()

    # ------------------------------------------------------------------ the noise schedule

    def schedule(self):
        self.next_section(
            "Noise schedule",
            notes="Linear beta schedule of Ho et al., 2020 (DDPM): beta from 1e-4 to 0.02 over T = 1000 steps. sqrt(abar_500) = 0.28, "
            "sqrt(abar_1000) = 0.0064. Later work uses gentler schedules, e.g. the cosine schedule (Nichol & Dhariwal, 2021).",
        )
        eq = formula(r"\bar{\alpha}_t", "=", r"\prod_{s=1}^{t}", r"(1-\beta_s)", size=44).move_to([-3.6, 1.55, 0])
        eq[0].set_color(TEAL)
        eq[3].set_color(PINK)

        mx, my, mw, mh = 3.35, 1.4, 2.6, 1.0  # the beta mini chart: origin and size, no arrow tips

        def mp(k, b):
            return np.array([mx + mw * k / 1000, my + mh * b / 0.02, 0])

        mini = m.VGroup(m.Line(mp(0, 0), mp(1000, 0)), m.Line(mp(0, 0), mp(0, 0.02))).set_stroke(FAINT, width=2.5)
        beta_line = m.Line(mp(0, 1e-4), mp(1000, 0.02)).set_stroke(PINK, width=4)
        beta_name = m.VGroup(tex(r"\beta_t", 34, PINK), text("noise per step", 22, MUTED)).arrange(m.RIGHT, buff=0.15)
        beta_name.next_to(mini, m.LEFT, buff=0.35).shift(0.15 * m.UP)
        lo = code("0.0001", 20, PINK).next_to(mp(0, 0), m.DOWN, buff=0.14)
        hi = code("0.02", 20, PINK).next_to(mp(1000, 0.02), m.UP, buff=0.1)
        steps = code("1000 steps", 20, MUTED).next_to(mp(1000, 0), m.DOWN, buff=0.14).align_to(mp(1000, 0), m.RIGHT)
        source = cite("linear schedule · Ho et al., 2020").move_to([4.65, 0.72, 0])

        chart = axes([0, 1100, 250], [0, 1.1, 0.5], 11.4, 2.05).move_to([0.25, -0.8, 0])
        ticks = m.VGroup(*[code(str(k), 20, MUTED).next_to(chart.c2p(k, 0), m.DOWN, buff=0.28) for k in (0, 250, 500, 750, 1000)])
        one = code("1", 20, MUTED).next_to(chart.c2p(0, 1), m.LEFT, buff=0.28)
        step_lab = text("step", 22, MUTED).next_to(chart.x_axis.get_end(), m.DOWN, buff=0.2)
        ts = np.arange(0, T + 1, 4)
        teal = m.VMobject().set_points_as_corners([chart.c2p(k, signal(k)) for k in ts]).set_stroke(TEAL, width=5)
        pink = m.VMobject().set_points_as_corners([chart.c2p(k, noise_w(k)) for k in ts]).set_stroke(PINK, width=5)
        teal_lab = text("picture weight", 22, TEAL, weight=m.SEMIBOLD).next_to(chart.c2p(140, signal(140)), m.UR, buff=0.12)
        pink_lab = text("noise weight", 22, PINK, weight=m.SEMIBOLD).next_to(chart.c2p(140, noise_w(140)), m.DR, buff=0.05).shift(0.2 * m.RIGHT)

        thumbs = m.Group()
        for k in (0, 250, 500, 750, 1000):
            th = m.Group(picture(closed_form(k), 0.95), edge(0.99)).move_to([chart.c2p(k, 0)[0], -2.95, 0])
            thumbs.add(th)

        def mark(k, label):
            p = chart.c2p(k, signal(k))
            line = m.DashedLine(chart.c2p(k, 0.04), chart.c2p(k, 1.0), dash_length=0.08).set_stroke(INK, width=2.5)
            dot = m.Dot(p, radius=0.1, color=TEAL).set_z_index(3)
            tag = pill(label, TEAL, TEAL_FILL, 22).next_to(p, m.UR, buff=0.1)
            return m.VGroup(line, dot, tag)

        half = mark(500, f"{signal(500):.2f}")
        last = mark(1000, f"{signal(1000):.3f}")
        last[2].next_to(chart.c2p(1000, 0.3), m.LEFT, buff=0.15)

        self.say(
            "[How fast does the picture fade?] [That's set by the noise schedule.]",
            self.new_header("THE NOISE SCHEDULE", "How fast the picture fades"),
            m.AnimationGroup(m.Create(chart), m.FadeIn(ticks), m.FadeIn(one), m.FadeIn(step_lab)),
        )
        self.say(
            "[Each step mixes in a little noise, beta]: [from one in ten thousand], [up to two percent, over a thousand steps.]",
            m.AnimationGroup(m.FadeIn(beta_name, shift=0.15 * m.UP), m.Create(mini)),
            m.AnimationGroup(m.Create(beta_line), m.FadeIn(lo), m.FadeIn(hi)),
            m.AnimationGroup(m.FadeIn(steps), m.FadeIn(source)),
        )
        self.say(
            "[Alpha bar multiplies what every step keeps]: [one minus beta, step after step.]",
            m.Write(eq),
            m.Circumscribe(eq[3], color=PINK, buff=0.08),
        )
        self.say(
            "[Its square root is the picture's weight, falling step by step], [while the noise weight rises.]",
            m.AnimationGroup(m.Create(teal, run_time=2.2), m.FadeIn(teal_lab), m.LaggedStart(*[m.FadeIn(th, shift=0.15 * m.UP) for th in thumbs], lag_ratio=0.25)),
            m.AnimationGroup(m.Create(pink, run_time=1.6), m.FadeIn(pink_lab)),
        )
        self.say(
            "[At step five hundred, the picture's weight is down to zero point two eight.] [By the last step, it's below one percent.]",
            m.AnimationGroup(m.Create(half[0]), m.FadeIn(half[1], scale=0.4), m.FadeIn(half[2], shift=0.15 * m.UP), m.Circumscribe(thumbs[2], color=TEAL, buff=0.06)),
            m.AnimationGroup(m.Create(last[0]), m.FadeIn(last[1], scale=0.4), m.FadeIn(last[2], shift=0.15 * m.UP), m.Circumscribe(thumbs[4], color=PINK, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ learned denoising

    def training(self):
        self.next_section(
            "Learned denoising",
            notes="DDPM training (Ho et al., 2020): pick x_0, a random step t and fresh noise eps; the network sees x_t and t and predicts eps; "
            "loss = squared error. The network in the paper is a U-Net. The guess slot is drawn empty on purpose: no network is trained here.",
        )
        th = 1.35
        x0_img = m.Group(picture(pix(MU[HERO]), th), edge(th + 0.04)).move_to([-5.7, 0.95, 0])
        eps_img = m.Group(picture(pix(EPS), th), edge(th + 0.04)).move_to([-5.7, -1.35, 0])
        x0_lab = tex(r"x_0", 32, TEAL).next_to(x0_img, m.UP, buff=0.12)
        eps_lab = tex(r"\varepsilon", 40, PINK).next_to(eps_img, m.UP, buff=0.12)
        xh = 1.9
        xt = m.Group(picture(closed_form(500), xh), edge(xh + 0.04)).move_to([-2.85, -0.2, 0])
        xt_lab = tex(r"x_t", 32, INK).next_to(xt, m.DOWN, buff=0.15)
        step_chip = chip("t = 500", INK, CARD, width=1.65, height=0.62, size=24).next_to(xt, m.UP, buff=0.3)
        into = m.VGroup(arrow(x0_img.get_right(), xt.get_left() + 0.45 * m.UP, MUTED), arrow(eps_img.get_right(), xt.get_left() + 0.45 * m.DOWN, MUTED))

        net_box = card(2.7, 2.5, VIOLET, VIOLET_FILL).move_to([0.75, -0.2, 0])
        net_body = m.VGroup(unet(), text("denoising", 24, INK, weight=m.BOLD), text("network", 24, INK, weight=m.BOLD), text("a U-Net", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        net_body[0].shift(0.08 * m.UP)
        net = m.VGroup(net_box, net_body.move_to(net_box))
        feed = m.VGroup(arrow(xt.get_right(), net_box.get_left(), MUTED), arrow(step_chip.get_right(), net_box.get_corner(m.UL) + 0.45 * m.DOWN, MUTED))

        slot = m.DashedVMobject(m.Square(side_length=xh), num_dashes=40).set_stroke(VIOLET, width=3).move_to([3.8, -0.2, 0])
        slot_sym = tex(r"\hat{\varepsilon}", 64, VIOLET).move_to(slot)
        slot_lab = text("its guess of the noise", 22, MUTED).next_to(slot, m.DOWN, buff=0.15)
        out = arrow(net_box.get_right(), slot.get_left(), VIOLET)
        real = m.Group(picture(pix(EPS), 1.2), edge(1.24)).move_to([6.05, -0.2, 0])
        vs = text("vs", 26, MUTED).move_to([5.12, -0.2, 0])
        real_lab = tex(r"\varepsilon", 40, PINK).next_to(real, m.UP, buff=0.12)

        loss = formula(r"\mathcal{L}", "=", r"\big\lVert", r"\varepsilon", "-", r"\hat{\varepsilon}_\theta(x_t, t)", r"\big\rVert^2", size=42).move_to([1.6, -2.75, 0])
        loss[3].set_color(PINK)
        loss[5].set_color(VIOLET)
        loss_lab = text("the loss: squared error", 22, MUTED).next_to(loss, m.LEFT, buff=0.45)
        source = cite("DDPM · Ho et al., 2020").next_to(loss, m.RIGHT, buff=0.5)

        self.say(
            "[Now the hard direction: going back.] [For that, we train a network.]",
            self.new_header("STEP 2 · LEARN", "The network predicts the noise"),
            m.FadeIn(net, shift=0.2 * m.UP),
        )
        self.say(
            "[The training game is simple]: [take an image], [pick a random step], [and add noise with the formula.]",
            m.Wait(0.1),
            m.AnimationGroup(m.FadeIn(x0_img, shift=0.2 * m.RIGHT), m.FadeIn(x0_lab)),
            m.FadeIn(step_chip, shift=0.2 * m.DOWN),
            m.AnimationGroup(m.FadeIn(eps_img, shift=0.2 * m.RIGHT), m.FadeIn(eps_lab), m.GrowArrow(into[0]), m.GrowArrow(into[1]), m.FadeIn(xt), m.FadeIn(xt_lab)),
        )
        self.say(
            "[The network sees the noisy image and the step], [and must guess the noise that was added.]",
            m.AnimationGroup(m.GrowArrow(feed[0]), m.GrowArrow(feed[1]), pulse(feed[0], VIOLET)),
            m.AnimationGroup(m.GrowArrow(out), m.Create(slot), m.FadeIn(slot_sym, scale=0.7), m.FadeIn(slot_lab)),
        )
        self.say(
            "[The loss is the squared difference between the real noise and the guess.]",
            m.AnimationGroup(m.FadeIn(real, target_position=eps_img, path_arc=-0.5), m.FadeIn(real_lab), m.FadeIn(vs), m.Write(loss), m.FadeIn(loss_lab)),
        )
        self.say(
            "[Repeat that millions of times], [and the network gets good at spotting noise.]",
            m.AnimationGroup(*[pulse(a, VIOLET) for a in (*into, feed[0], out)], m.FadeIn(source), lag_ratio=0.3),
            m.Circumscribe(net_box, color=VIOLET, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the score

    def the_score(self):
        self.next_section(
            "The score",
            notes="2-D toy: a mixture of 74 Gaussians (sd 0.06) shaped like a sun and a crescent moon. Noised by the same schedule it stays a "
            "mixture, so the score grad log p_t(x) is exact. Ideal noise prediction = -sqrt(1 - abar_t) * score (Tweedie). Score-based view: "
            "Song & Ermon, 2019; Song et al., 2021. Arrows: direction of the score, length 0.42 tanh(|score| / 4).",
        )
        plane = Plane((-3.05, -0.5), 6.4, 4.7, 1.0)
        self.plane = plane
        dots = plane.data()
        self.data_dots = dots
        legend = m.VGroup(
            text("a known mix of 74 bell curves:", 22, MUTED),
            m.VGroup(m.Dot(radius=0.07, color=AMBER), text("sun", 22, AMBER, weight=m.BOLD)).arrange(m.RIGHT, buff=0.12),
            m.VGroup(m.Dot(radius=0.07, color=SKY), text("moon", 22, SKY, weight=m.BOLD)).arrange(m.RIGHT, buff=0.12),
        ).arrange(m.RIGHT, buff=0.3).move_to([plane.c[0], -3.25, 0])
        self.tags = legend

        level = m.ValueTracker(300.0)
        field = m.always_redraw(lambda: plane.field(level.get_value()))
        eq = formula(r"s(x)", "=", r"\nabla_x", r"\log p_t(x)", size=46).move_to([3.75, 1.8, 0])
        eq[0].set_color(VIOLET)
        terms = m.VGroup(
            m.VGroup(tex(r"\nabla_x", 32, AMBER), text("the uphill direction", 22, MUTED)).arrange(m.RIGHT, buff=0.2),
            m.VGroup(tex(r"\log p_t", 32, AMBER), text("how likely a spot is", 22, MUTED)).arrange(m.RIGHT, buff=0.2),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([3.75, 0.55, 0])
        lvl_num = m.DecimalNumber(300, num_decimal_places=0, mob_class=Mono, group_with_commas=False, font_size=28, color=INK)
        lvl_num.add_updater(lambda d: d.set_value(int(round(level.get_value()))))
        lvl = m.VGroup(text("noise level: step", 24, MUTED), lvl_num).arrange(m.RIGHT, buff=0.2).move_to([3.75, -0.55, 0])
        lvl_num.add_updater(lambda d: d.next_to(lvl[0], m.RIGHT, buff=0.2))
        flip = pill("the noise guess points the other way", VIOLET, VIOLET_FILL, 22).move_to([3.75, -1.6, 0])
        exact = pill("exact for this toy: no network needed", GOOD, GOOD_FILL, 22).move_to([3.75, -2.55, 0])
        source = cite("Song & Ermon, 2019 · Song et al., 2021").move_to([3.75, -3.3, 0])

        probe = np.array([[0.05, -1.2]])
        s0 = score(probe, 300)[0]
        e0 = eps2(probe, 300)[0]
        base = plane.p(probe)[0]
        probe_dot = m.Dot(base, radius=0.08, color=INK).set_z_index(5)
        s_arrow = m.Arrow(base, base + np.r_[0.95 * s0 / np.linalg.norm(s0), 0], buff=0, stroke_width=7, color=VIOLET, max_tip_length_to_length_ratio=0.3).set_z_index(5)
        e_arrow = m.Arrow(base, base + np.r_[0.6 * e0 / np.linalg.norm(e0), 0], buff=0, stroke_width=7, color=PINK, max_tip_length_to_length_ratio=0.4).set_z_index(5)
        halo = m.Circle(radius=0.85).set_stroke(width=0).set_fill(CARD, opacity=0.85).move_to(base).set_z_index(4)

        self.say(
            "[What is the network really learning?] [Let's shrink the problem to two dimensions.]",
            self.new_header("STEP 3 · THE SCORE", "Arrows toward the data"),
            m.FadeIn(plane.box),
        )
        self.say(
            "[Each dot is a data point, in the shape of a sun and a moon.] [They come from a known mix of bell curves], [so everything here is exact.]",
            m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.004),
            m.FadeIn(legend, shift=0.15 * m.UP),
            m.FadeIn(exact, shift=0.2 * m.UP),
        )
        self.say(
            "[The score is the gradient of the log density]: [at every spot, an arrow toward where data is more likely.]",
            m.AnimationGroup(m.Write(eq), m.FadeIn(terms, shift=0.15 * m.UP), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(field), m.FadeIn(lvl)),
        )
        self.say(
            "[Predicting the noise is the same as estimating the score]: [the noise guess points the other way, scaled down.]",
            m.FadeIn(flip, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(halo), m.FadeIn(probe_dot, scale=0.5), m.GrowArrow(s_arrow), m.GrowArrow(e_arrow)),
        )
        self.say(
            "[With lots of noise, the arrows point gently toward the middle.] [As the noise fades, they point sharply at the shapes.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in (halo, probe_dot, s_arrow, e_arrow)], level.animate(run_time=2.4).set_value(T)),
            level.animate(run_time=3.5, rate_func=ease_from(2.0)).set_value(20),
        )
        self.wait(0.3)
        self.stop(field, lvl_num)
        self.play(*[m.FadeOut(mob) for mob in (field, eq, terms, lvl, flip, exact, source)], run_time=0.8)

    # ------------------------------------------------------------------ many small steps

    def small_steps(self):
        self.next_section(
            "Many small steps",
            notes="DDPM ancestral sampling of 500 points with the exact noise prediction, 1000 steps. The one-step jump is E[x_0 | x_T]: all points "
            "land on the mixture's mean, between the shapes. Animation time runs as step = 1000 (1 - p)^1.5.",
        )
        plane = self.plane
        state = {"mode": "jump"}
        jump = m.ValueTracker(0.0)
        step = m.ValueTracker(float(T))

        def where():
            if state["mode"] == "jump":
                a = jump.get_value()
                return (1 - a) * UNCOND[0] + a * ONE_STEP
            return path_at(UNCOND, step.get_value())

        swarm = plane.follow(plane.swarm(UNCOND[0]), where)
        num = m.DecimalNumber(T, num_decimal_places=0, mob_class=Mono, group_with_commas=False, font_size=40, color=INK)
        num.add_updater(lambda d: d.set_value(int(round(step.get_value()))))
        counter = m.VGroup(text("step", 30, MUTED, weight=m.SEMIBOLD), num).arrange(m.RIGHT, buff=0.25).move_to([3.75, 1.7, 0])
        num.add_updater(lambda d: d.next_to(counter[0], m.RIGHT, buff=0.25))
        blob = plane.p(ONE_STEP.mean(0))[0]
        avg = pill("one big step: just the average", BAD, BAD_FILL, 22).move_to([3.75, 0.45, 0])
        ring = m.Circle(radius=0.28).set_stroke(BAD, width=4).move_to(blob).set_z_index(4)
        many = pill("a thousand small steps", GOOD, GOOD_FILL, 22).move_to([3.75, -0.6, 0])
        recipe = m.VGroup(
            text("each step:", 24, INK, weight=m.SEMIBOLD),
            text("move a little along the score", 22, VIOLET),
            text("plus a small kick of fresh noise", 22, PINK),
        ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT).move_to([3.75, -1.95, 0])
        done = text("static → sun and moon", 26, GOOD, weight=m.BOLD).move_to([3.75, -3.2, 0])

        self.say(
            "[Now let's generate.] [Start with five hundred points of pure noise.]",
            m.AnimationGroup(self.new_header("STEP 4 · MANY SMALL STEPS", "From static to shape"), self.data_dots.animate.set_opacity(0.28)),
            m.AnimationGroup(m.FadeIn(swarm, lag_ratio=0.004, run_time=1.5), m.FadeIn(counter)),
        )
        self.say(
            "[What if we jump to a clean answer in one step?] [Every point lands on the average], [between the sun and the moon, where there's no data.]",
            m.Wait(0.1),
            jump.animate(run_time=1.6).set_value(1),
            m.AnimationGroup(m.Create(ring), m.FadeIn(avg, shift=0.2 * m.UP)),
        )
        self.say(
            "[So instead, we take a thousand small steps.] [Each one only undoes a little noise]: [an easy guess.]",
            m.AnimationGroup(jump.animate(run_time=1.2).set_value(0), m.FadeOut(ring), avg.animate.set_opacity(0.45)),
            m.FadeIn(many, shift=0.2 * m.UP),
            m.FadeIn(recipe[0], shift=0.15 * m.UP),
        )
        state["mode"] = "run"
        self.say(
            "[At every step, each point moves a little along the score], [plus a kick of fresh noise], [and the static flows into shape.]",
            m.AnimationGroup(m.FadeIn(recipe[1], shift=0.15 * m.UP), step.animate(run_time=7.5, rate_func=ease_from(1.5)).set_value(0)),
            m.FadeIn(recipe[2], shift=0.15 * m.UP),
            m.FadeIn(done, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.stop(swarm, num)
        self.clear()

    # ------------------------------------------------------------------ back to the picture

    def back_again(self):
        self.next_section(
            "Back to the picture",
            notes=f"Fresh static (seed 21), DDPM sampling with the exact denoiser of a four-picture dataset (sd 0.05 per pixel); reverse seed 32. "
            f"The guess E[x_0 | x_t] is a blend of all four at first and commits (responsibility > 0.99) at step {BACK_COMMIT}. It lands on the moon. "
            "A trained network on a large dataset generalises instead of recalling; occasional memorisation is documented (Carlini et al., 2023).",
        )
        h = 3.1
        lc, rc = np.array([-4.65, -0.3, 0]), np.array([-0.45, -0.3, 0])
        step = m.ValueTracker(float(T))

        def idx():
            return int(round((T - step.get_value()) / EVERY))

        sample = m.always_redraw(lambda: picture(BACK_PIX[idx()], h).move_to(lc))
        guess = m.always_redraw(lambda: picture(GUESS_PIX[idx()], h).move_to(rc))
        frames = m.VGroup(edge(h + 0.05).move_to(lc), edge(h + 0.05, VIOLET).move_to(rc))
        num = m.DecimalNumber(T, num_decimal_places=0, mob_class=Mono, group_with_commas=False, font_size=28, color=INK)
        num.add_updater(lambda d: d.set_value(int(round(step.get_value()))))
        s_lab = m.VGroup(text("the sample at step", 24, MUTED), num).arrange(m.RIGHT, buff=0.18).next_to(frames[0], m.UP, buff=0.22)
        num.add_updater(lambda d: d.next_to(s_lab[0], m.RIGHT, buff=0.18))
        g_lab = text("its guess of the clean image", 24, VIOLET, weight=m.SEMIBOLD).next_to(frames[1], m.UP, buff=0.22)

        th = 1.12
        names = ["sunset", "moon", "tree", "peaks"]
        thumbs = m.Group(*[m.Group(picture(pix(MU[k]), th), edge(th + 0.04)) for k in range(4)])
        thumbs.arrange_in_grid(rows=2, cols=2, buff=(0.35, 0.65)).move_to([4.7, -0.55, 0])
        t_labels = m.VGroup(*[text(nm, 24, MUTED).next_to(tb, m.DOWN, buff=0.16) for nm, tb in zip(names, thumbs)])
        known = text("all it knows: 4 pictures", 24, INK, weight=m.SEMIBOLD).next_to(thumbs, m.UP, buff=0.35)
        chosen = m.SurroundingRectangle(thumbs[MOON_PIC], buff=0.05, color=SKY, stroke_width=4)
        real = pill("a real network learns from millions of images, and makes new ones", VIOLET, VIOLET_FILL, 22).move_to([0, -3.15, 0])

        self.say(
            "[Back to our picture.] [We start from brand new static], [beside the model's guess of the clean image.]",
            self.new_header("BACK TO OUR PICTURE", "From fresh static"),
            m.AnimationGroup(m.FadeIn(sample), m.FadeIn(frames[0]), m.FadeIn(s_lab)),
            m.AnimationGroup(m.FadeIn(guess), m.FadeIn(frames[1]), m.FadeIn(g_lab)),
        )
        self.say(
            "[Now we remove noise, a thousand small steps.] [The guess starts as a blurry average], [then commits, and sharpens.]",
            step.animate(run_time=8.0, rate_func=ease_from(0.55)).set_value(0),
            m.Circumscribe(frames[1], color=VIOLET, buff=0.08),
            m.Wait(0.1),
        )
        self.say(
            "[This time we got the moon]: [this denoiser is exact for a tiny set of four pictures], [so it can only bring one back.]",
            m.Circumscribe(frames[0], color=SKY, buff=0.08),
            m.AnimationGroup(m.FadeIn(known, shift=0.15 * m.DOWN), m.LaggedStart(*[m.FadeIn(tb, scale=0.9) for tb in thumbs], lag_ratio=0.15), m.FadeIn(t_labels)),
            m.Create(chosen),
        )
        self.say(
            "[A real network learns from millions of images], [and makes new pictures, not copies.]",
            m.FadeIn(real, shift=0.2 * m.UP),
            m.Circumscribe(real, color=VIOLET, buff=0.08),
        )
        self.wait(0.3)
        self.stop(sample, guess, num)
        self.clear()

    # ------------------------------------------------------------------ guidance

    def guidance(self):
        self.next_section(
            "Guidance",
            notes=f"Classifier-free guidance (Ho & Salimans, 2022). Toy prompt 'moon' is weak on purpose (illustrative): p(x | moon) = 0.75 moon + "
            f"0.25 sun; unconditional 0.5 / 0.5. Same 400 starting points for both scales. Landed on the moon: s = 1: {SHARE[1]:.1%}, "
            f"s = 4: {SHARE[4]:.1%}. Stable Diffusion's common default scale is 7.5.",
        )
        eq = formula(r"\tilde{\varepsilon}", "=", r"\hat{\varepsilon}(x_t)", "+", "s", r"\,(", r"\hat{\varepsilon}(x_t, c)", "-", r"\hat{\varepsilon}(x_t)", ")", size=44).move_to([0, 2.05, 0])
        eq[2].set_color(MUTED)
        eq[4].set_color(LIME)
        eq[6].set_color(SKY)
        eq[8].set_color(MUTED)
        lab_u = text("guess, no prompt", 22, MUTED).next_to(eq[2], m.DOWN, buff=0.25)
        lab_c = text("guess with prompt c", 22, SKY).next_to(eq[6], m.DOWN, buff=0.25)
        lab_u.shift(0.55 * m.LEFT)
        lab_c.shift(0.75 * m.RIGHT)
        lab_s = text("guidance scale", 22, LIME, weight=m.SEMIBOLD).next_to(eq[4], m.DOWN, buff=0.75)
        tick = m.Line(eq[4].get_bottom() + 0.08 * m.DOWN, lab_s.get_top() + 0.06 * m.UP).set_stroke(LIME, width=2.5)
        lab_s = m.VGroup(tick, lab_s)
        source = cite("classifier-free guidance · Ho & Salimans, 2022").move_to([0, 0.85, 0]).align_to([6.75, 0, 0], m.RIGHT)

        panels = {}
        for s, x in ((1, -3.3), (4, 3.3)):
            plane = Plane((x, -1.6), 6.1, 3.0, 0.7)
            data = plane.data(opacity=0.3, radius=0.024)
            title = m.VGroup(chip("prompt: moon", SKY, SKY_FILL, width=2.35, height=0.5, size=20), text(f"scale s = {s}", 22, LIME, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.3)
            title.move_to([x, 0.25, 0])
            tracker = m.ValueTracker(float(T))
            path = GUIDE[s]
            dots = plane.follow(plane.swarm(GUIDE[s][0], radius=0.026), lambda path=path, tracker=tracker: path_at(path, tracker.get_value()))
            share = text(f"{100 * SHARE[s]:.1f}% land on the moon", 22, INK, weight=m.BOLD).move_to([x, -3.4, 0])
            panels[s] = dict(plane=plane, data=data, title=title, tracker=tracker, dots=dots, share=share)
        less = text("· less variety", 22, MUTED)
        panels[4]["share"].shift(0.8 * m.LEFT)
        less.next_to(panels[4]["share"], m.RIGHT, buff=0.2)

        def show(p):
            return m.AnimationGroup(m.FadeIn(p["plane"].box), m.FadeIn(p["data"]), m.FadeIn(p["title"]), m.FadeIn(p["dots"], lag_ratio=0.004, run_time=1.5))

        def run(p):
            return m.Succession(p["tracker"].animate(run_time=4.0, rate_func=ease_from(1.5)).set_value(0), m.FadeIn(p["share"], scale=1.2))

        self.say(
            "[How do we get the picture we asked for?] [We give the network the prompt, too.]",
            self.new_header("STEP 5 · GUIDANCE", "Following the prompt"),
            m.AnimationGroup(show(panels[1]), show(panels[4])),
        )
        self.say(
            "[During training, the prompt is sometimes left out], [so one network learns both guesses]: [without the prompt, and with it.]",
            m.Wait(0.1),
            m.Wait(0.1),
            m.AnimationGroup(m.FadeIn(eq[2], shift=0.15 * m.UP), m.FadeIn(lab_u), m.FadeIn(eq[6], shift=0.15 * m.UP), m.FadeIn(lab_c)),
        )
        self.say(
            "[Classifier-free guidance starts from the guess without the prompt], [and pushes along the difference the prompt makes], [scaled by a number, s.]",
            m.AnimationGroup(m.FadeIn(eq[0:2], shift=0.15 * m.UP), m.Circumscribe(eq[2], color=MUTED, buff=0.06), m.FadeIn(source)),
            m.FadeIn(m.VGroup(eq[3], eq[5], eq[7], eq[8], eq[9]), shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(eq[4], scale=1.4), m.FadeIn(lab_s, shift=0.15 * m.UP)),
        )
        self.say(
            "[In our toy, the moon prompt is weak: a quarter of its examples are suns.] [At scale one, about three out of four points land on the moon.]",
            m.Circumscribe(panels[1]["title"][0], color=SKY, buff=0.06),
            run(panels[1]),
        )
        self.say(
            "[Turn the scale up to four], [and nearly every point lands on the moon]: [closer to the prompt, with less variety.]",
            m.Circumscribe(panels[4]["title"][1], color=LIME, buff=0.06),
            run(panels[4]),
            m.FadeIn(less, shift=0.15 * m.LEFT),
        )
        self.wait(0.3)
        self.stop(*[p["dots"] for p in panels.values()])
        self.clear()

    # ------------------------------------------------------------------ latent diffusion

    def latent(self):
        self.next_section(
            "Latent diffusion",
            notes="Latent diffusion (Rombach et al., 2022); Stable Diffusion v1: 512 x 512 x 3 = 786,432 numbers → a 64 x 64 x 4 latent = 16,384 "
            "(48 times fewer). The autoencoder is trained first; the prompt enters the denoiser through cross-attention from a CLIP text encoder. "
            "The latent squares are drawn as plain tiles (illustrative), not real latents.",
        )
        y = -0.25
        img_h = 2.0
        big = picture(np.asarray(PICTURES[HERO] * 255, dtype=np.uint8), img_h)
        big.set_resampling_algorithm(3)
        left = m.Group(big, edge(img_h + 0.04)).move_to([-5.45, y, 0])
        right = m.Group(picture(np.asarray(PICTURES[HERO] * 255, dtype=np.uint8), img_h).set_resampling_algorithm(3), edge(img_h + 0.04)).move_to([5.45, y, 0])
        l_lab = m.VGroup(code("512 × 512 × 3", 22, INK), text("786,432 numbers", 22, MUTED)).arrange(m.DOWN, buff=0.1).next_to(left, m.DOWN, buff=0.25)
        enc = m.VGroup(funnel(1.9, 0.9, 1.35), text("encoder", 22, INK, weight=m.SEMIBOLD)).move_to([-2.75, y, 0])
        enc[1].next_to(enc[0], m.DOWN, buff=0.2)
        dec = m.VGroup(funnel(1.9, 0.9, 1.35, widening=True), text("decoder", 22, INK, weight=m.SEMIBOLD)).move_to([2.75, y, 0])
        dec[1].next_to(dec[0], m.DOWN, buff=0.2)
        tiles = m.VGroup(*[m.Square(side_length=0.85).set_stroke(BLUE, width=2.5).set_fill("#13203A", opacity=1).move_to([0.18 * (i - 1.5), y + 0.12 * (1.5 - i), 0]) for i in range(4)])
        latent = m.VGroup(tiles)
        lat_lab = m.VGroup(code("64 × 64 × 4", 22, BLUE), text("16,384 numbers", 22, MUTED)).arrange(m.DOWN, buff=0.1).next_to(tiles, m.DOWN, buff=0.3)
        links = m.VGroup(
            arrow(left.get_right(), enc[0].get_left(), MUTED),
            arrow(enc[0].get_right(), tiles.get_left(), MUTED),
            arrow(tiles.get_right(), dec[0].get_left(), MUTED),
            arrow(dec[0].get_right(), right.get_left(), MUTED),
        )
        loop = m.CurvedArrow(tiles.get_corner(m.UR) + np.array([0.05, 0.1, 0]), tiles.get_corner(m.UL) + np.array([-0.05, 0.1, 0]), angle=2.2)
        loop.set_stroke(TEAL, width=4).set_color(TEAL)
        loop_lab = text("denoise here", 22, TEAL, weight=m.SEMIBOLD).next_to(loop, m.UP, buff=0.12)
        prompt = chip("prompt", SKY, SKY_FILL, width=1.5, height=0.55, size=22).move_to([2.55, 1.55, 0])
        p_link = arrow(prompt.get_left(), loop.get_right() + np.array([0.05, 0.1, 0]), SKY)
        fewer = pill("48 times fewer numbers to denoise", GOOD, GOOD_FILL, 24).move_to([0, -2.6, 0])
        sd = m.VGroup(text("Stable Diffusion", 30, INK, weight=m.BOLD), cite("latent diffusion · Rombach et al., 2022")).arrange(m.RIGHT, buff=0.4).move_to([0, -3.4, 0])

        self.say(
            "[One last trick makes this practical.] [A five hundred and twelve pixel image holds almost eight hundred thousand numbers.]",
            self.new_header("THE BIG PICTURE", "Latent diffusion"),
            m.AnimationGroup(m.FadeIn(left, shift=0.2 * m.RIGHT), m.FadeIn(l_lab, shift=0.15 * m.UP)),
        )
        self.say(
            "[Latent diffusion first squeezes images through an autoencoder], [down to sixty-four by sixty-four by four]: [forty-eight times fewer numbers.]",
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(enc, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.LaggedStart(*[m.FadeIn(tl, scale=0.6) for tl in tiles], lag_ratio=0.2), m.FadeIn(lat_lab)),
            m.FadeIn(fewer, shift=0.2 * m.UP),
        )
        self.say(
            "[The denoising runs in that small space, guided by the prompt], [and a decoder turns the result into pixels.] [That's Stable Diffusion.]",
            m.AnimationGroup(m.Create(loop), m.FadeIn(loop_lab), m.FadeIn(prompt, shift=0.2 * m.LEFT), m.GrowArrow(p_link)),
            m.AnimationGroup(m.GrowArrow(links[2]), m.FadeIn(dec, shift=0.2 * m.RIGHT), m.GrowArrow(links[3]), m.FadeIn(right, shift=0.2 * m.RIGHT)),
            m.FadeIn(sd, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Noise, predict, denoise.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Noise", "Predict", "Denoise"), (PINK, VIOLET, TEAL))]).arrange(m.RIGHT, buff=1.3))
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A picture, found in static, one small step at a time.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [add noise], [learn to predict it], [and remove it.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A diffusion model finds a picture in static, one small step at a time.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
