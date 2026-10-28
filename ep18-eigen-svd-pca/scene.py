"""AI, Explained · Episode 18 · Eigenvectors, SVD & PCA

Running example: compressing a face. The face is drawn here in code (no photograph): a 64 × 64
grayscale image built from soft ellipses and smooth gradients. Its SVD is computed with numpy,
and the rank-1, 5 and 20 reconstructions are shown with their real errors and storage counts.

- Eigenvectors: A = [[1.25, 0.75], [0.75, 1.25]] on a transformed grid (callback to Episode 10):
  eigenvalues 2 (along y = x) and 1/2 (along y = -x), from np.linalg.eigh.
- SVD: the shear [[1, 1], [0, 1]] as turn - stretch - turn (singular values 1.618 and 0.618;
  the signs are chosen so U and V are rotations: in general either may also include a flip).
- PCA: 200 illustrative faces, (width, height) in cm from a seeded generator; centred, the
  covariance's eigenvectors, checked equal to the centred data's right singular vectors.

Error = ||A - A_k|| / ||A|| (Frobenius, relative). Energy = sum of squared pixel values
= sum of squared singular values. Storage of a rank-k copy = k columns + k rows + k values.
Every number the narration quotes is asserted below.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the face

N = 64


def soft(d, w):
    """1 inside (d < 0), 0 outside, with a smooth edge about w wide."""
    return 1 / (1 + np.exp(np.clip(d / w, -60, 60)))


def ellipse(x, y, cx, cy, rx, ry):
    return np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) - 1


def make_face(n=N):
    """A synthetic grayscale face, values 0 (black) to 1 (white)."""
    s = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(s, -s)  # y points up
    img = 0.22 + 0.12 * (y + 1) / 2 - 0.05 * x  # background: a soft gradient
    hair = soft(ellipse(x, y, 0, 0.12, 0.72, 0.86), 0.03)
    img = img * (1 - hair) + 0.10 * hair
    face = soft(ellipse(x, y, 0, -0.02, 0.60, 0.78), 0.025)
    skin = 0.80 - 0.18 * (x + 0.3) ** 2 - 0.12 * (y - 0.1) ** 2 - 0.06 * x  # lit from the upper left
    fringe = soft(-(y - (0.52 - 0.10 * x**2)), 0.03) * hair
    face = face * (1 - fringe)
    img = img * (1 - face) + skin * face
    for side in (-1, 1):
        ex = side * 0.25
        white = soft(ellipse(x, y, ex, 0.16, 0.14, 0.075), 0.02)
        img = img * (1 - white) + 0.95 * white
        iris = soft(ellipse(x, y, ex + 0.01, 0.15, 0.06, 0.06), 0.015)
        img = img * (1 - iris) + 0.12 * iris
        brow = soft(ellipse(x, y, ex, 0.24, 0.16, 0.12), 0.02) * (1 - soft(ellipse(x, y, ex, 0.21, 0.17, 0.11), 0.02))
        brow = brow * (y > 0.26)
        img = img * (1 - brow) + 0.18 * brow
    img = img - 0.14 * np.exp(-(((x - 0.05) / 0.05) ** 2 + ((y + 0.05) / 0.16) ** 2)) * face  # nose shadow
    for side in (-1, 1):
        img = img - 0.25 * np.exp(-(((x - side * 0.06) / 0.03) ** 2 + ((y + 0.19) / 0.02) ** 2))
    smile = soft(ellipse(x, y, 0, -0.30, 0.24, 0.16), 0.02) * (1 - soft(ellipse(x, y, 0, -0.26, 0.25, 0.14), 0.02))
    smile = smile * (y < -0.33)
    img = img * (1 - smile) + 0.25 * smile
    return np.clip(img, 0, 1)


FACE = make_face()
U, S, VT = np.linalg.svd(FACE)
if U[:, 0].sum() < 0:  # the first pair, shown as pictures, reads better positive
    U[:, 0] *= -1
    VT[0] *= -1
ENERGY = np.cumsum(S**2) / np.sum(S**2)
FULL = N * N


def rank_k(k):
    return (U[:, :k] * S[:k]) @ VT[:k]


def error(k):
    return float(np.linalg.norm(FACE - rank_k(k)) / np.linalg.norm(FACE))


def stored(k):
    return k * (N + N + 1)


def share(k):
    return stored(k) / FULL


assert FACE.shape == (64, 64) and FULL == 4096 and len(S) == 64  # "sixty-four by sixty-four", "4,096"
assert np.all(np.diff(S) <= 0)  # "biggest first"
assert np.allclose(U @ np.diag(S) @ VT, FACE)
assert round(100 * ENERGY[0]) == 92 and round(100 * ENERGY[4]) == 99  # "ninety-two", "ninety-nine percent"
assert stored(1) == 129 and stored(5) == 645 and stored(20) == 2580  # "a hundred and twenty-nine"
assert round(100 * share(1)) == 3 and round(100 * share(5)) == 16 and round(100 * share(20)) == 63
assert round(100 * error(1)) == 28 and round(100 * error(5)) == 10 and abs(100 * error(20) - 0.5) < 0.05
for _k in (1, 5, 20):  # Eckart-Young: the truncation's squared error is the dropped energy
    assert np.isclose(np.linalg.norm(FACE - rank_k(_k)) ** 2, np.sum(S[_k:] ** 2))
assert np.all(U[:, 0] > 0) and np.all(VT[0] > 0)  # the first column and row: one sign, a plain plaid

# ---------------------------------------------------------------------------- eigenvectors

EIG_A = np.array([[1.25, 0.75], [0.75, 1.25]])
_vals, _vecs = np.linalg.eigh(EIG_A)
assert np.allclose(_vals, [0.5, 2.0])  # "two times along one diagonal, one half along the other"
assert np.allclose(np.abs(_vecs[:, 1]), [2**-0.5] * 2) and np.isclose(_vecs[0, 0] * _vecs[1, 0], -0.5)
D1 = np.array([1.0, 1.0]) / np.sqrt(2)  # stretched by 2
D2 = np.array([-1.0, 1.0]) / np.sqrt(2)  # stretched by 1/2
assert np.allclose(EIG_A @ D1, 2 * D1) and np.allclose(EIG_A @ D2, 0.5 * D2)
I2 = np.eye(2)


def rotation(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


TURN = np.radians(70)  # the rotation that has no eigenvectors
_w = np.linalg.eigvals(rotation(TURN))
assert np.all(np.abs(_w.imag) > 0.5)  # no real eigenvalues: every direction turns

# ---------------------------------------------------------------------------- the SVD of a shear

SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])
SU, SS, SVT = np.linalg.svd(SHEAR)
if SVT[0, 0] < 0:  # v1 to the right: flipping a pair (u_i, v_i) keeps U Σ Vᵀ the same
    SU[:, 0] *= -1
    SVT[0] *= -1
if np.linalg.det(SVT) < 0:
    SU[:, 1] *= -1
    SVT[1] *= -1
assert np.isclose(np.linalg.det(SU), 1) and np.isclose(np.linalg.det(SVT), 1)  # both pure turns
assert np.allclose(SU @ np.diag(SS) @ SVT, SHEAR)
assert np.allclose(SS, [(1 + 5**0.5) / 2, (5**0.5 - 1) / 2])  # 1.62 and 0.62 on screen
ALPHA = float(np.arctan2(SVT[0, 1], SVT[0, 0]))  # angle of v1
BETA = float(np.arctan2(SU[1, 0], SU[0, 0]))  # angle of u1
assert np.allclose(SVT, rotation(-ALPHA)) and np.allclose(SU, rotation(BETA))
SV1, SV2 = SVT[0].copy(), SVT[1].copy()

# ---------------------------------------------------------------------------- PCA: 200 faces

P_N = 200
_rng = np.random.default_rng(1)
_h = 19 + 1.2 * _rng.normal(size=P_N)
_wd = 14 + 0.8 * (_h - 19) + 0.5 * _rng.normal(size=P_N)
PTS = np.c_[_wd, _h]  # illustrative: face width and height, cm
MEAN = PTS.mean(0)
CENTRED = PTS - MEAN
COV = CENTRED.T @ CENTRED / (P_N - 1)
_lam, _vec = np.linalg.eigh(COV)
LAM = _lam[::-1]
PC1 = _vec[:, 1] * np.sign(_vec[0, 1])
PC2 = np.array([-PC1[1], PC1[0]])
_, _sv, _rv = np.linalg.svd(CENTRED, full_matrices=False)
assert np.allclose(np.abs(_rv @ np.c_[PC1, PC2]), I2, atol=1e-9)  # eigenvectors = right singular vectors
assert np.allclose(_sv**2 / (P_N - 1), LAM)
EXPLAINED = LAM / LAM.sum()
assert P_N == 200 and round(100 * EXPLAINED[0]) == 95  # "two hundred faces", "ninety-five percent"
PC1_ANGLE = float(np.arctan2(PC1[1], PC1[0]))


def spread(theta):
    d = np.array([np.cos(theta), np.sin(theta)])
    return float(d @ COV @ d)


# ---------------------------------------------------------------------------- helpers

DARK_PX = np.array([14.0, 19.0, 34.0])
LIGHT_PX = np.array([236.0, 240.0, 246.0])
AMBER_PX = np.array([251.0, 191.0, 36.0])


def pixels(values):
    """Values 0..1 as the series' cool grey: dark navy-grey to ink."""
    v = np.clip(np.asarray(values, float), 0, 1)
    return np.round(DARK_PX + (LIGHT_PX - DARK_PX) * v[..., None]).astype(np.uint8)


def picture(values, height, signed=False, widen=1, tall=1):
    """A matrix as a picture: every entry one square pixel (nearest resampling)."""
    v = np.asarray(values, float)
    if signed:
        v = 0.5 + 0.5 * v / np.abs(v).max()
    rgb = pixels(v)
    rgb = np.repeat(np.repeat(rgb, tall, axis=0), widen, axis=1)
    img = m.ImageMobject(rgb, scale_to_resolution=0, resampling_algorithm=0)
    img.height = height
    return img


def sigma_picture(height):
    """Σ as a picture: its diagonal in amber (brightness (σ/σ1)^0.3, so small ones show)."""
    rgb = np.tile(DARK_PX, (N, N, 1))
    for i, s in enumerate(S):
        rgb[i, i] = DARK_PX + (AMBER_PX - DARK_PX) * (s / S[0]) ** 0.3
    img = m.ImageMobject(np.round(rgb).astype(np.uint8), scale_to_resolution=0, resampling_algorithm=0)
    img.height = height
    return img


def framed(img, color, stroke=2.5):
    border = m.Rectangle(width=img.width + 0.05, height=img.height + 0.05).set_stroke(color, width=stroke).move_to(img)
    return m.Group(img, border)


def variant(name, color, fill):
    """An icon in another accent, like kit.recolored(), but saved as assets/ep18-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    stem = name if name.startswith("ep18-") else f"ep18-{name}"
    out = f"{stem}-{color.lstrip('#').lower()}"
    svg = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{out}.svg"
    if not path.exists() or path.read_text() != svg:
        path.write_text(svg)
    return out


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def clip(p, q, lim):
    """The part of segment p→q inside the square |x|, |y| ≤ lim (Liang–Barsky), or None."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    t0, t1 = 0.0, 1.0
    for pi, qi in ((-d[0], p[0] + lim), (d[0], lim - p[0]), (-d[1], p[1] + lim), (d[1], lim - p[1])):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 >= t1:
        return None
    return p + t0 * d, p + t1 * d


def row(symbol, words, color=INK, size=24, buff=0.28):
    return m.VGroup(symbol, text(words, size, color if color != INK else MUTED)).arrange(m.RIGHT, buff=buff)


class Lens:
    """A square window onto the plane, drawn through a 2×2 matrix that animations change."""

    def __init__(self, center, scale, half=2.55):
        self.c = np.array([center[0], center[1], 0.0])
        self.s, self.half = scale, half
        self.M = I2.copy()
        self.clock = m.ValueTracker(0.0)
        self.live = []

    @property
    def lim(self):
        return self.half / self.s

    def at(self, p):
        return self.c + self.s * np.array([p[0], p[1], 0.0])

    def frame(self):
        return card(2 * self.half + 0.3, 2 * self.half + 0.3, EDGE, CARD).move_to(self.c).set_z_index(-2)

    def _live(self, build, z):
        mob = build().set_z_index(z)
        mob.add_updater(lambda mob: mob.become(build().set_z_index(z)))
        self.live.append(mob)
        return mob

    def stop(self):
        for mob in self.live:
            mob.clear_updaters()

    def grid(self, step=0.5, reach=8.0):
        def build():
            lines = m.VGroup()
            for k in np.arange(-reach, reach + 1e-9, step):
                axis = abs(k) < 1e-9
                for p, q in (((k, -reach), (k, reach)), ((-reach, k), (reach, k))):
                    seg = clip(self.M @ np.array(p), self.M @ np.array(q), self.lim)
                    if seg is not None:
                        lines.add(
                            m.Line(self.at(seg[0]), self.at(seg[1])).set_stroke(
                                BLUE, width=3 if axis else 1.6, opacity=0.8 if axis else 0.32
                            )
                        )
            return lines

        return self._live(build, -1)

    def static_line(self, direction, color, dashed=True, width=3, opacity=0.75):
        d = np.asarray(direction, float)
        seg = clip(-10 * d, 10 * d, self.lim - 0.05)
        a, b = self.at(seg[0]), self.at(seg[1])
        line = m.DashedLine(a, b, dash_length=0.12) if dashed else m.Line(a, b)
        return line.set_stroke(color, width=width, opacity=opacity).set_z_index(0)

    def arrow(self, v, color, width=6, opacity=None, z=2):
        v = np.asarray(v, float)

        def build():
            tip = self.M @ v
            a = m.Arrow(
                self.at((0, 0)), self.at(tip), buff=0, color=color, stroke_width=width,
                max_tip_length_to_length_ratio=0.25, max_stroke_width_to_length_ratio=14,
            )
            if opacity is not None:
                a.set_opacity(opacity.get_value())
            return a

        return self._live(build, z)

    def ghost(self, v, width=3.5):
        v = np.asarray(v, float)
        a = m.Arrow(
            self.at((0, 0)), self.at(v), buff=0, color=FAINT, stroke_width=width,
            max_tip_length_to_length_ratio=0.25, max_stroke_width_to_length_ratio=14,
        )
        return a.set_z_index(1)

    def loop(self, points, color, width=4, fill=0.0):
        pts = np.asarray(points, float)

        def build():
            q = (self.M @ pts.T).T
            shape = m.VMobject().set_points_as_corners([self.at(p) for p in q] + [self.at(q[0])])
            return shape.set_stroke(color, width=width).set_fill(color, opacity=fill)

        return self._live(build, 1)

    def go(self, path, run_time=1.6, rate_func=m.smooth):
        def update(_, a):
            self.M = np.asarray(path(a), float)

        return m.UpdateFromAlphaFunc(self.clock, update, run_time=run_time, rate_func=rate_func)


def lerp(P, Q):
    return lambda a: (1 - a) * np.asarray(P) + a * np.asarray(Q)


CIRCLE = np.c_[np.cos(np.linspace(0, 2 * np.pi, 120, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 120, endpoint=False))]

# ---------------------------------------------------------------------------- the episode


class Ep18EigenSVDPCA(Episode):
    number = 18

    def lesson(self):
        self.hook()
        self.eigen()
        self.svd()
        self.spectrum()
        self.layers()
        self.compression()
        self.pca()
        self.big_picture()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A 64×64 face drawn in code: 4,096 numbers. How few do we need?")
        face = picture(FACE, 4.2).move_to([-3.1, -0.15, 0])
        border = m.Rectangle(width=face.width + 0.05, height=face.height + 0.05).set_stroke(SKY, width=2.5).move_to(face)
        px = face.height / N
        left, top = face.get_left()[0], face.get_top()[1]
        r0, c0 = 24, 20  # a 4×4 patch over the left eye
        patch_vals = FACE[r0 : r0 + 4, c0 : c0 + 4]
        zoom = m.Square(side_length=4 * px).set_stroke(AMBER, width=3).move_to([left + (c0 + 2) * px, top - (r0 + 2) * px, 0]).set_z_index(3)
        nums = m.VGroup(*[code(f"{v:.2f}", 26, INK) for v in patch_vals.flat]).arrange_in_grid(rows=4, cols=4, buff=(0.42, 0.3))
        panel = card(nums.width + 0.8, nums.height + 0.7, AMBER, CARD).move_to([3.1, 0.75, 0])
        nums.move_to(panel)
        links = m.VGroup(
            m.Line(zoom.get_corner(m.UR), panel.get_corner(m.UL) + 0.12 * m.RIGHT).set_stroke(AMBER, width=2, opacity=0.6),
            m.Line(zoom.get_corner(m.DR), panel.get_corner(m.DL) + 0.12 * m.RIGHT).set_stroke(AMBER, width=2, opacity=0.6),
        )
        size = text("64 × 64 pixels", 24, MUTED).next_to(border, m.DOWN, buff=0.25)
        count = pill("64 × 64 = 4,096 numbers", SKY, SKY_FILL, 28).next_to(panel, m.DOWN, buff=0.4)
        claim = text("Most data is simpler than it looks.", 32, AMBER, weight=m.BOLD).move_to([3.1, -2.25, 0])
        ask = text("How few numbers does this face need?", 24, MUTED).next_to(claim, m.DOWN, buff=0.3)

        self.say(
            "So... [how many numbers does it take to describe a face?]",
            m.AnimationGroup(m.FadeIn(face, scale=0.9), m.Create(border), run_time=1.2),
        )
        self.say(
            "[Every pixel is one number, its brightness.] [This face is sixty-four by sixty-four pixels]: [four thousand and ninety-six numbers.]",
            m.AnimationGroup(m.Create(zoom), m.Create(links), m.FadeIn(panel), m.LaggedStart(*[m.FadeIn(n) for n in nums], lag_ratio=0.05)),
            m.FadeIn(size, shift=0.15 * m.UP),
            m.FadeIn(count, shift=0.2 * m.UP),
        )
        self.say(
            "[But most data is simpler than it looks.] [Let's find out how few numbers this face really needs.]",
            m.FadeIn(claim, shift=0.2 * m.UP),
            m.FadeIn(ask, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ eigenvectors

    def eigen(self):
        self.next_section(
            "Eigenvectors",
            notes="A = [[1.25, .75], [.75, 1.25]]: eigenvalue 2 along y = x, 1/2 along y = −x (np.linalg.eigh). A rotation by 70° has no real eigenvectors. Callback: Episode 10.",
        )
        lens = Lens([-3.45, -0.45], 0.95)
        frame = lens.frame()
        grid = lens.grid()
        dim = m.ValueTracker(1.0)
        greys = [lens.arrow(1.5 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]), MUTED, opacity=dim) for a in (0, 100, 170, 280)]
        ghosts = m.VGroup(*[lens.ghost(1.5 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])) for a in (0, 100, 170, 280, 45, 135)])
        teal = lens.arrow(1.5 * D1, TEAL, z=3)
        pink = lens.arrow(1.5 * D2, PINK, z=3)
        line1 = lens.static_line(D1, TEAL)
        line2 = lens.static_line(D2, PINK)

        x0 = 3.45
        matrix = tex(r"A = \begin{bmatrix} 1.25 & 0.75 \\ 0.75 & 1.25 \end{bmatrix}", 36).move_to([x0, 1.75, 0])
        law = m.MathTex("A", r"\mathbf{v}", "=", r"\lambda", r"\mathbf{v}", font_size=62, color=INK).move_to([x0, 0.55, 0])
        law[3].set_color(AMBER)
        legend = m.VGroup(
            row(tex("A", 34), "the matrix"),
            row(tex(r"\mathbf{v}", 34), "an eigenvector: it keeps its direction"),
            row(tex(r"\lambda", 34, AMBER), "its eigenvalue: how much it stretches"),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([x0, -0.75, 0])
        for r in legend[1:]:
            r[1].align_to(legend[0][1], m.LEFT)
        ones = m.VGroup(
            m.VGroup(m.Line(m.ORIGIN, 0.55 * m.RIGHT).set_stroke(TEAL, width=6), tex(r"\lambda = 2", 34, AMBER), text("twice as long", 24, MUTED)).arrange(m.RIGHT, buff=0.25),
            m.VGroup(m.Line(m.ORIGIN, 0.55 * m.RIGHT).set_stroke(PINK, width=6), tex(r"\lambda = \tfrac{1}{2}", 34, AMBER), text("half as long", 24, MUTED)).arrange(m.RIGHT, buff=0.25),
        ).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([x0, -2.3, 0])
        ones[1][2].align_to(ones[0][2], m.LEFT)
        ones[1][1].align_to(ones[0][1], m.LEFT)
        turn = pill("a rotation: no eigenvectors", BAD, BAD_FILL, 22).move_to([x0, -3.35, 0])

        flashes = m.AnimationGroup(
            *[
                m.ShowPassingFlash(m.Line(lens.at((0, 0)), lens.at(EIG_A @ (1.5 * d))).set_stroke(c, width=12), time_width=0.6, run_time=1.2)
                for d, c in ((D1, TEAL), (D2, PINK))
            ]
        )

        self.say(
            "[First, a callback to episode ten]: [a matrix transforms space, stretching and turning the grid.]",
            m.AnimationGroup(self.new_header("STEP 1 · EIGENVECTORS", "Directions a matrix only stretches"), m.FadeIn(frame), m.FadeIn(grid), m.Write(matrix)),
            lens.go(lerp(I2, EIG_A), run_time=2.8, rate_func=m.there_and_back),
        )
        self.say(
            "[Watch these arrows.] [As the matrix acts], [most of them get knocked off their own line.]",
            m.AnimationGroup(m.FadeIn(ghosts), *[m.FadeIn(a) for a in greys], m.FadeIn(teal), m.FadeIn(pink)),
            lens.go(lerp(I2, EIG_A), run_time=1.8),
            m.Indicate(ghosts[:4], color=INK, scale_factor=1.0),
        )
        self.say(
            "[But two special directions stay on their line]: [the matrix only stretches or squashes them.]",
            m.AnimationGroup(dim.animate.set_value(0.45), m.Create(line1), m.Create(line2)),
            flashes,
        )
        self.say(
            "[These are the eigenvectors.] [In symbols: A times v equals lambda times v.]",
            m.AnimationGroup(*[m.Flash(lens.at(EIG_A @ (1.5 * d)), color=c, flash_radius=0.3) for d, c in ((D1, TEAL), (D2, PINK))]),
            m.Write(law),
        )
        self.say(
            "[A is the matrix, and v is an eigenvector.] [Lambda, the eigenvalue, says how much it stretches.]",
            m.LaggedStart(m.FadeIn(legend[0], shift=0.15 * m.UP), m.FadeIn(legend[1], shift=0.15 * m.UP), lag_ratio=0.5),
            m.AnimationGroup(m.FadeIn(legend[2], shift=0.15 * m.UP), m.Indicate(law[3], color=AMBER, scale_factor=1.25)),
        )
        self.say(
            "[Here, two times along one diagonal], [and one half along the other.]",
            m.FadeIn(ones[0], shift=0.15 * m.UP),
            m.FadeIn(ones[1], shift=0.15 * m.UP),
        )
        self.say(
            "[But not every matrix has them]: [a rotation turns every single direction.]",
            lens.go(lerp(EIG_A, I2), run_time=1.2),
            m.AnimationGroup(lens.go(lambda a: rotation(TURN * a), run_time=2.2), dim.animate(run_time=0.6).set_value(1.0), m.FadeIn(turn, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        lens.stop()
        self.clear()

    # ------------------------------------------------------------------ SVD

    def svd(self):
        self.next_section(
            "SVD: turn, stretch, turn",
            notes="The shear [[1,1],[0,1]]: singular values 1.618 and 0.618 (the golden ratio). Signs chosen so U and V are rotations; in general U and V may also flip.",
        )
        lens = Lens([-3.45, -0.45], 1.2)
        frame = lens.frame()
        axes_x = m.Line(lens.at((-lens.lim + 0.05, 0)), lens.at((lens.lim - 0.05, 0))).set_stroke(FAINT, width=2).set_z_index(0)
        axes_y = m.Line(lens.at((0, -lens.lim + 0.05)), lens.at((0, lens.lim - 0.05))).set_stroke(FAINT, width=2).set_z_index(0)
        disc = lens.loop(CIRCLE, SKY, width=4, fill=0.1)
        a1 = lens.arrow(SV1, TEAL, z=3)
        a2 = lens.arrow(SV2, PINK, z=3)
        target = m.DashedVMobject(
            m.VMobject().set_points_as_corners([lens.at(SHEAR @ p) for p in CIRCLE] + [lens.at(SHEAR @ CIRCLE[0])]).set_stroke(GOOD, width=4),
            num_dashes=60,
        ).set_z_index(4)

        x0 = 3.45
        shear = m.VGroup(tex(r"A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}", 36), text("a shear", 24, MUTED)).arrange(m.RIGHT, buff=0.35).move_to([x0, 1.8, 0])
        formula = m.MathTex("A", "=", "U", r"\Sigma", r"V^{\top}", font_size=70, color=INK).move_to([x0, 0.6, 0])
        formula[2].set_color(VIOLET)
        formula[3].set_color(AMBER)
        formula[4].set_color(VIOLET)
        steps = m.VGroup(
            m.VGroup(text("1", 24, MUTED, weight=m.BOLD), tex(r"V^{\top}", 36, VIOLET), text("turn", 24, MUTED)),
            m.VGroup(text("2", 24, MUTED, weight=m.BOLD), tex(r"\Sigma", 36, AMBER), text("stretch along the axes", 24, MUTED)),
            m.VGroup(text("3", 24, MUTED, weight=m.BOLD), tex("U", 36, VIOLET), text("turn again", 24, MUTED)),
        )
        for k, s in enumerate(steps):
            y = -0.55 - 0.6 * k
            s[0].move_to([x0 - 2.2, y, 0])
            s[1].move_to([x0 - 1.5, y, 0])
            s[2].next_to([x0 - 1.0, y, 0], m.RIGHT, buff=0).align_to(s[0], m.DOWN)
        read = text("read from the right", 22, MUTED).next_to(formula, m.DOWN, buff=0.12).align_to(formula, m.RIGHT)
        check = pill("= exactly what the shear does", GOOD, GOOD_FILL, 22).move_to([x0, -2.5, 0])
        sigmas = m.VGroup(text("singular values:", 24, MUTED), tex(r"\sigma_1 = 1.62", 36, AMBER), tex(r"\sigma_2 = 0.62", 36, AMBER)).arrange(m.RIGHT, buff=0.35).move_to([x0, -3.3, 0])

        def stage1(a):
            return rotation(-ALPHA * a)

        def stage2(a):
            return np.diag([1 + a * (SS[0] - 1), 1 + a * (SS[1] - 1)]) @ rotation(-ALPHA)

        def stage3(a):
            return rotation(BETA * a) @ np.diag(SS) @ rotation(-ALPHA)

        self.say(
            "[The singular value decomposition, or SVD, works for every matrix, of any shape]: [A equals U, sigma, V transpose.]",
            m.AnimationGroup(
                self.new_header("STEP 2 · SVD", "Turn, stretch, turn"),
                m.FadeIn(frame), m.Create(axes_x), m.Create(axes_y), m.FadeIn(disc), m.FadeIn(a1), m.FadeIn(a2), m.FadeIn(shear),
            ),
            m.Write(formula),
        )
        self.say(
            "[Take a shear, and a circle.] [Reading from the right], [V transpose turns it], [sigma stretches it along the axes], [and the last one turns it again.]",
            m.AnimationGroup(m.Indicate(shear[0], color=INK, scale_factor=1.08), m.Indicate(disc, color=SKY, scale_factor=1.0)),
            m.AnimationGroup(m.FadeIn(read, shift=0.1 * m.LEFT), m.Indicate(formula[4], color=VIOLET, scale_factor=1.25)),
            m.AnimationGroup(m.FadeIn(steps[0], shift=0.15 * m.UP), lens.go(stage1, run_time=1.4)),
            m.AnimationGroup(m.FadeIn(steps[1], shift=0.15 * m.UP), lens.go(stage2, run_time=1.6)),
            m.AnimationGroup(m.FadeIn(steps[2], shift=0.15 * m.UP), lens.go(stage3, run_time=1.6)),
        )
        self.say(
            "[Together, that's exactly what the shear does.] [The stretch factors are the singular values], [always sorted, biggest first.]",
            m.AnimationGroup(m.Create(target, run_time=1.2), m.FadeIn(check, shift=0.15 * m.UP)),
            m.FadeIn(sigmas, shift=0.15 * m.UP),
            m.AnimationGroup(m.Indicate(sigmas[1], color=AMBER, scale_factor=1.15), m.Indicate(formula[3], color=AMBER, scale_factor=1.25)),
        )
        self.wait(0.3)
        lens.stop()
        self.clear()

    # ------------------------------------------------------------------ the face's singular values

    def spectrum(self):
        self.next_section(
            "The face's singular values",
            notes="SVD of the 64×64 face (numpy). Energy = sum of squared pixels = Σσ². σ1: 92% of it, σ1..σ5: 99%. Eckart & Young (1936): the truncated SVD is the best rank-k copy in squared error.",
        )
        h = 2.45
        face = framed(picture(FACE, h), SKY)
        eq = tex("=", 96, MUTED)
        u_img = framed(picture(U, h, signed=True), VIOLET)
        s_img = framed(sigma_picture(h), AMBER)
        v_img = framed(picture(VT, h, signed=True), VIOLET)
        xs = (-4.75, -3.05, -1.35, 1.35, 4.05)
        for mob, x in zip((face, eq, u_img, s_img, v_img), xs):
            mob.move_to([x + 0.35, -0.1, 0])
        labels = m.VGroup(
            text("the face", 26, SKY, weight=m.SEMIBOLD).next_to(face, m.DOWN, buff=0.25),
            tex("U", 40, VIOLET).next_to(u_img, m.DOWN, buff=0.2),
            tex(r"\Sigma", 40, AMBER).next_to(s_img, m.DOWN, buff=0.2),
            tex(r"V^{\top}", 40, VIOLET).next_to(v_img, m.DOWN, buff=0.2),
        )
        labels[2:].align_to(labels[1], m.DOWN)
        diag_note = text("64 singular values down its diagonal", 22, MUTED).next_to(labels[2], m.DOWN, buff=0.2)

        # the bar chart
        base, top_h, x_first, gap = -2.0, 3.7, -3.2, 0.148
        bars = m.VGroup(
            *[
                m.Rectangle(width=0.09, height=max(top_h * s / S[0], 0.012)).set_fill(AMBER, opacity=1).set_stroke(width=0).move_to([x_first + i * gap, base, 0], aligned_edge=m.DOWN)
                for i, s in enumerate(S)
            ]
        ).set_z_index(1)
        floor = m.Line([x_first - 0.2, base, 0], [x_first + 63 * gap + 0.2, base, 0]).set_stroke(FAINT, width=2.5)
        ticks = m.VGroup(*[text(str(k), 20, MUTED).next_to([x_first + (k - 1) * gap, base, 0], m.DOWN, buff=0.18) for k in (1, 5, 20, 64)])
        small_face = picture(FACE, 2.3)
        small_face_target = framed(small_face, SKY).move_to([-5.35, 0.0, 0])
        e1 = text(f"first value: {100 * ENERGY[0]:.0f}% of the energy", 28, INK, t2c={f"{100 * ENERGY[0]:.0f}%": AMBER}).move_to([2.9, 1.55, 0])
        e5 = text(f"first five: {100 * ENERGY[4]:.0f}%", 28, INK, t2c={f"{100 * ENERGY[4]:.0f}%": AMBER}).next_to(e1, m.DOWN, buff=0.3).align_to(e1, m.LEFT)
        e_def = text("energy: the sum of all squared pixel values", 20, MUTED).next_to(e5, m.DOWN, buff=0.25).align_to(e1, m.LEFT)
        ring1 = m.SurroundingRectangle(bars[0], buff=0.08, color=AMBER).set_stroke(AMBER, width=2.5)
        ring5 = m.SurroundingRectangle(bars[:5], buff=0.08, color=AMBER).set_stroke(AMBER, width=2.5)
        rank_note = text("rank: how many singular values are not zero", 24, INK).move_to([2.9, -0.55, 0]).align_to(e1, m.LEFT)
        cut_x = x_first + 4.5 * gap
        cut = m.DashedLine([cut_x, base - 0.05, 0], [cut_x, 0.9, 0], dash_length=0.1).set_stroke(GOOD, width=3)
        keep = text("← keep the top few", 24, GOOD, weight=m.BOLD).next_to([cut_x, 0.0, 0], m.RIGHT, buff=0.15)
        drop = text("drop the rest", 24, MUTED, weight=m.SEMIBOLD).move_to([0.6, -1.35, 0])
        best = m.VGroup(
            text("the closest copy of that rank, in squared error", 24, INK),
            text("Eckart & Young, 1936", 20, MUTED),
        ).arrange(m.RIGHT, buff=0.4).move_to([0.6, -3.3, 0])

        self.say(
            "[Now let's do the same to our face.] [It splits into the same three pieces], [with sixty-four singular values down the diagonal.]",
            m.AnimationGroup(self.new_header("THE FACE", "64 singular values, biggest first"), m.FadeIn(face, shift=0.2 * m.UP), m.FadeIn(labels[0])),
            m.AnimationGroup(m.FadeIn(eq), m.LaggedStart(*[m.FadeIn(m.Group(img, lab), shift=0.2 * m.UP) for img, lab in zip((u_img, s_img, v_img), labels[1:])], lag_ratio=0.3)),
            m.AnimationGroup(m.Circumscribe(s_img, color=AMBER, buff=0.08), m.FadeIn(diag_note, shift=0.15 * m.UP)),
        )
        self.say(
            "[Here they are, biggest first.] [Look how fast they fall.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (eq, u_img, v_img, labels, diag_note)],
                m.FadeOut(s_img, shift=0.3 * m.DOWN),
                m.Transform(face, small_face_target),
                m.Create(floor),
                m.FadeIn(ticks),
            ),
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.04, run_time=2.4),
        )
        self.say(
            "[The first one alone holds ninety-two percent of the image's energy], [and the first five, ninety-nine percent.]",
            m.AnimationGroup(m.Create(ring1), m.FadeIn(e1, shift=0.15 * m.UP), m.FadeIn(e_def)),
            m.AnimationGroup(m.ReplacementTransform(ring1, ring5), m.FadeIn(e5, shift=0.15 * m.UP)),
        )
        self.say(
            "[A matrix's rank is how many of these are not zero.] [Keep only the top few], [and you get the closest possible copy of that lower rank.]",
            m.FadeIn(rank_note, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeOut(ring5), m.Create(cut), m.FadeIn(keep), m.FadeIn(drop), bars[5:].animate.set_fill(FAINT)),
            m.FadeIn(best, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ layers

    def layers(self):
        self.next_section("Layers", notes="σ1·u1·v1ᵀ: one column times one row is a plaid layer; the sum of the top k layers is the rank-k copy (k = 1 → 20 shown live).")
        side = 3.0
        lay_c = np.array([-2.55, -0.9, 0])
        layer = framed(picture(np.clip(rank_k(1), 0, 1), side), TEAL).move_to(lay_c)
        col = framed(picture(U[:, :1] / U[:, 0].max(), side, widen=8), TEAL)
        col.move_to(lay_c + (side / 2 + 0.25 + col.width / 2) * m.LEFT)
        rw = picture(VT[:1] / VT[0].max(), 0.375, tall=8)
        rw_f = framed(rw, TEAL).move_to(lay_c + (side / 2 + 0.25 + 0.19) * m.UP)
        corner = np.array([col.get_x(), rw_f.get_y(), 0])
        sig_lab = tex(r"\sigma_1", 34, AMBER)
        sig = m.VGroup(card(0.75, 0.6, AMBER, AMBER_FILL, radius=0.18, stroke=2), sig_lab).move_to(corner)
        sig_lab.move_to(sig[0])
        col_lab = text("a column", 24, TEAL, weight=m.SEMIBOLD).next_to(col, m.LEFT, buff=0.25)
        row_lab = text("a row", 24, TEAL, weight=m.SEMIBOLD).next_to(rw_f, m.RIGHT, buff=0.25)
        lay_lab = text("one layer: a plaid", 24, INK, weight=m.SEMIBOLD).next_to(layer, m.DOWN, buff=0.2)

        k = m.ValueTracker(1.0)
        live_c = np.array([3.75, -0.45, 0])

        def build():
            n = int(round(k.get_value()))
            return framed(picture(np.clip(rank_k(n), 0, 1), 3.4), SKY).move_to(live_c)

        live = m.always_redraw(build)
        counter = m.DecimalNumber(1, num_decimal_places=0, mob_class=Mono, font_size=34, color=GOOD)
        counter.add_updater(lambda d: d.set_value(int(round(k.get_value()))))
        count_row = m.VGroup(text("layers added:", 28, MUTED, weight=m.SEMIBOLD), counter).arrange(m.RIGHT, buff=0.25).move_to(live_c + 2.15 * m.UP)
        plus = arrow([0.0, -0.45, 0], [1.75, -0.45, 0], MUTED)
        plus_lab = text("add them up", 22, MUTED).next_to(plus, m.UP, buff=0.15)

        self.say(
            "[Each singular value comes with a matching column and row.]",
            m.AnimationGroup(
                self.new_header("LAYERS", "One column times one row"),
                m.FadeIn(sig, scale=0.8),
                m.FadeIn(col, shift=0.2 * m.RIGHT),
                m.FadeIn(rw_f, shift=0.2 * m.DOWN),
                m.FadeIn(col_lab),
                m.FadeIn(row_lab),
            ),
        )
        self.say(
            "[Multiply them together, and you get one layer]: [a simple, plaid pattern.]",
            m.AnimationGroup(m.FadeIn(layer, scale=0.6), m.Indicate(sig, color=AMBER, scale_factor=1.15)),
            m.FadeIn(lay_lab, shift=0.15 * m.UP),
        )
        self.say(
            "[Add the layers up, biggest first], [and the face comes back.]",
            m.AnimationGroup(m.GrowArrow(plus), m.FadeIn(plus_lab), m.FadeIn(live), m.FadeIn(count_row)),
            k.animate(run_time=3.6, rate_func=m.linear).set_value(20),
        )
        self.wait(0.4)
        self.stop(live, counter)
        self.clear()

    # ------------------------------------------------------------------ compression

    def compression(self):
        self.next_section(
            "Compression",
            notes="Rank 1: 129 numbers (3%), error 28%. Rank 5: 645 (16%), error 9.7%. Rank 20: 2,580 (63%), error 0.50%. Error = ‖A − A_k‖/‖A‖. A real photo has more fine detail and needs more layers.",
        )
        xs = (-5.03, -1.68, 1.68, 5.03)
        y0 = -0.05
        specs = [
            ("1 layer", 1, GOOD),
            ("5 layers", 5, GOOD),
            ("20 layers", 20, GOOD),
            ("the original", None, SKY),
        ]
        panels = []
        for (title, k, color), x in zip(specs, xs):
            box = card(3.1, 4.55, EDGE, CARD, stroke=2).move_to([x, y0, 0])
            heading = text(title, 26, INK if k else SKY, weight=m.BOLD).move_to([x, y0 + 1.85, 0])
            values = FACE if k is None else np.clip(rank_k(k), 0, 1)
            img = framed(picture(values, 2.45), SKY if k is None else EDGE).move_to([x, y0 + 0.2, 0])
            if k is None:
                store = text("4,096 numbers", 22, INK, weight=m.SEMIBOLD)
                err = text("the reference", 22, MUTED)
            else:
                store = text(f"{stored(k):,} numbers · {100 * share(k):.0f}%", 22, GOOD, weight=m.SEMIBOLD)
                e = 100 * error(k)
                err = text(f"error {e:.0f}%" if e >= 1 else f"error {e:.1f}%", 22, BAD, weight=m.SEMIBOLD)
            store.move_to([x, y0 - 1.38, 0])
            err.move_to([x, y0 - 1.8, 0])
            panels.append(dict(box=box, heading=heading, img=img, store=store, err=err))
        cost = m.MathTex("64", "+", "64", "+", "1", "=", "129", font_size=40, color=INK)
        cost[0].set_color(TEAL)
        cost[2].set_color(TEAL)
        cost[4].set_color(AMBER)
        cost[6].set_color(GOOD)
        cost_lab = text("each layer:", 24, MUTED, weight=m.SEMIBOLD)
        cost_words = text("a column + a row + one singular value", 22, MUTED)
        cost_row = m.VGroup(cost_lab, cost, cost_words).arrange(m.RIGHT, buff=0.35).move_to([0, -3.15, 0])
        cost_lab.align_to(cost_words, m.DOWN)

        def show(p, with_stats=True):
            parts = [m.FadeIn(p["box"]), m.FadeIn(p["heading"], shift=0.15 * m.DOWN), m.FadeIn(p["img"], scale=0.9)]
            if with_stats:
                parts.append(m.FadeIn(p["store"], shift=0.15 * m.UP))
            return m.AnimationGroup(*parts)

        orig = panels[3]
        self.say(
            "[The original stores four thousand and ninety-six numbers.] [Each layer costs a column, a row and one singular value]: [a hundred and twenty-nine.]",
            m.AnimationGroup(self.new_header("COMPRESSION", "Keep the top k layers"), show(orig), m.FadeIn(orig["err"])),
            m.AnimationGroup(m.FadeIn(cost_lab), m.FadeIn(cost[:5]), m.FadeIn(cost_words)),
            m.AnimationGroup(m.FadeIn(cost[5:], shift=0.15 * m.LEFT), m.Indicate(cost[6], color=GOOD, scale_factor=1.2)),
        )
        p1, p5, p20 = panels[:3]
        self.say(
            "[One layer keeps three percent of the numbers], [but it's twenty-eight percent off.]",
            show(p1),
            m.FadeIn(p1["err"], shift=0.15 * m.UP),
        )
        self.say(
            "[Five layers keep sixteen percent], [and it's already clearly a face], [about ten percent off.]",
            show(p5),
            m.Indicate(p5["img"], scale_factor=1.04),
            m.FadeIn(p5["err"], shift=0.15 * m.UP),
        )
        self.say(
            "[Twenty layers keep sixty-three percent], [and the error is just half a percent.]",
            show(p20),
            m.AnimationGroup(m.FadeIn(p20["err"], shift=0.15 * m.UP), m.Circumscribe(m.Group(p20["img"], orig["img"]), color=GOOD, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ PCA

    def pca(self):
        self.next_section(
            "PCA",
            notes="Illustrative data: 200 faces, width and height in cm (seeded numpy). PC1 explains 95% of the variance. Eigenvectors of the covariance = right singular vectors of the centred data (asserted). Pearson 1901; Hotelling 1933.",
        )
        cx, cy, scale = -3.4, -0.35, 0.56
        lo_w, lo_h = 10.0, 15.0  # bottom-left corner of the chart, cm
        mid_w, mid_h = 14.0, 19.0

        def raw(p):
            return np.array([cx + (p[0] - mid_w) * scale, cy + (p[1] - mid_h) * scale, 0.0])

        def centred(p):
            return np.array([cx + p[0] * scale, cy + p[1] * scale, 0.0])

        ax = axes([lo_w, 18.9, 1], [lo_h, 23.9, 1], 8.9 * scale, 8.9 * scale)
        ax.shift(raw((lo_w, lo_h)) - ax.c2p(lo_w, lo_h))
        ticks = m.VGroup(
            *[text(f"{v:g}", 20, MUTED).next_to(raw((v, lo_h)), m.DOWN, buff=0.15) for v in (11, 14, 17)],
            *[text(f"{v:g}", 20, MUTED).next_to(raw((lo_w, v)), m.LEFT, buff=0.15) for v in (16, 19, 22)],
        )
        x_title = text("face width, cm", 22, MUTED).move_to([cx, raw((lo_w, lo_h))[1] - 0.72, 0])
        y_title = text("face height, cm", 22, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.2)
        dots = m.VGroup(*[m.Dot(raw(p), radius=0.045, color=SKY) for p in PTS]).set_z_index(2)
        mean_mark = m.VGroup(
            m.Line(raw(MEAN) + 0.18 * m.LEFT, raw(MEAN) + 0.18 * m.RIGHT),
            m.Line(raw(MEAN) + 0.18 * m.DOWN, raw(MEAN) + 0.18 * m.UP),
        ).set_stroke(AMBER, width=5).set_z_index(4)
        reach = 2.55
        cross = m.VGroup(
            m.Line([cx - reach, cy, 0], [cx + reach, cy, 0]),
            m.Line([cx, cy - reach, 0], [cx, cy + reach, 0]),
        ).set_stroke(FAINT, width=2.5).set_z_index(0)
        theta = m.ValueTracker(np.radians(140))

        def probe_build():
            d = np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])
            return m.Line([cx, cy, 0] - 2.5 * d, [cx, cy, 0] + 2.5 * d).set_stroke(INK, width=3.5, opacity=0.85).set_z_index(3)

        probe = m.always_redraw(probe_build)

        x0 = 3.4
        steps = m.VGroup(
            text("1 · Centre: subtract the average", 26, INK, t2c={"1 ·": MUTED}),
            text("2 · PC1: the direction of most spread", 26, TEAL, t2c={"2 ·": MUTED}),
            text("3 · PC2: at right angles to it", 26, PINK, t2c={"3 ·": MUTED}),
        ).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT).move_to([x0, 1.65, 0])
        track_w = 4.2
        meter_lab = text("spread along the line", 22, MUTED).move_to([x0, -0.1, 0]).align_to(steps, m.LEFT)
        track = card(track_w, 0.26, EDGE, CARD, radius=0.13, stroke=1.5).next_to(meter_lab, m.DOWN, buff=0.18).align_to(steps, m.LEFT)

        def fill_build():
            w = max(track_w * spread(theta.get_value()) / LAM[0], 0.05)
            return m.RoundedRectangle(corner_radius=0.1, width=w, height=0.2).set_fill(AMBER, opacity=1).set_stroke(width=0).move_to(track.get_left() + (0.03 + w / 2) * m.RIGHT).set_z_index(1)

        fill = m.always_redraw(fill_build)
        same = m.VGroup(
            pill("eigenvectors of the covariance", VIOLET, VIOLET_FILL, 22),
            pill("= top singular vectors of the centred data", VIOLET, VIOLET_FILL, 22),
        ).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).move_to([x0, -1.55, 0]).align_to(steps, m.LEFT)
        cite = text("Pearson, 1901 · Hotelling, 1933", 20, MUTED).next_to(same, m.DOWN, buff=0.2).align_to(steps, m.LEFT)
        explained = pill(f"PC1 explains {100 * EXPLAINED[0]:.0f}% of the variance", TEAL, TEAL_FILL, 24).move_to([x0, -3.2, 0]).align_to(steps, m.LEFT)

        pc1_len, pc2_len = 2.5 * np.sqrt(LAM[0]) * scale, 2.5 * np.sqrt(LAM[1]) * scale
        o = np.array([cx, cy, 0.0])
        pc1_arrow = m.Arrow(o, o + pc1_len * np.array([*PC1, 0]), buff=0, color=TEAL, stroke_width=7, max_tip_length_to_length_ratio=0.2).set_z_index(5)
        pc2_arrow = m.Arrow(o, o + pc2_len * np.array([*PC2, 0]), buff=0, color=PINK, stroke_width=7, max_tip_length_to_length_ratio=0.35, max_stroke_width_to_length_ratio=20).set_z_index(5)
        pc1_line = m.DashedLine(o - 2.6 * np.array([*PC1, 0]), o + 2.6 * np.array([*PC1, 0]), dash_length=0.12).set_stroke(TEAL, width=3, opacity=0.8).set_z_index(1)
        projected = m.VGroup(*[m.Dot(centred(float(c @ PC1) * PC1), radius=0.045, color=TEAL) for c in CENTRED]).set_z_index(2)

        self.say(
            "[The same idea finds the shape of a dataset]: [principal component analysis, or PCA.]",
            self.new_header("STEP 3 · PCA", "The directions the data varies most"),
            m.AnimationGroup(m.Create(ax), m.FadeIn(ticks), m.FadeIn(x_title), m.FadeIn(y_title)),
        )
        self.say(
            "[Say we measured two hundred faces, for example their width and height]: [each face becomes one point.]",
            m.Wait(0.1),
            m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.01, run_time=2.0),
        )
        self.say(
            "[First, centre the data]: [subtract the average, so the cloud sits around zero.]",
            m.AnimationGroup(m.FadeIn(steps[0], shift=0.15 * m.UP), m.FadeIn(mean_mark, scale=0.5)),
            m.AnimationGroup(
                m.ReplacementTransform(ax, cross),
                m.FadeOut(ticks),
                m.FadeOut(x_title),
                m.FadeOut(y_title),
                *[d.animate.move_to(centred(c)) for d, c in zip(dots, CENTRED)],
                mean_mark.animate.move_to([cx, cy, 0]),
                run_time=1.8,
            ),
        )
        self.say(
            "[Then turn a line until the points spread out along it the most]: [that's the first principal component.]",
            m.AnimationGroup(m.FadeOut(mean_mark), m.FadeIn(probe), m.FadeIn(meter_lab), m.FadeIn(track), m.FadeIn(fill), theta.animate(run_time=3.2).set_value(PC1_ANGLE)),
            m.AnimationGroup(m.FadeIn(steps[1], shift=0.15 * m.UP), m.GrowArrow(pc1_arrow), m.Create(pc1_line)),
        )
        self.stop(probe, fill)
        self.say(
            "[The second one is at right angles to it], [and catches what's left.]",
            m.AnimationGroup(m.FadeOut(probe), m.FadeIn(steps[2], shift=0.15 * m.UP), m.GrowArrow(pc2_arrow)),
            m.Indicate(pc2_arrow, color=PINK, scale_factor=1.3),
        )
        self.say(
            "[These are the eigenvectors of the data's covariance], [and also the top singular vectors of the centred data.]",
            m.AnimationGroup(m.FadeIn(same[0], shift=0.15 * m.UP), m.FadeIn(cite)),
            m.FadeIn(same[1], shift=0.15 * m.UP),
        )
        self.say(
            "[Here, the first component explains ninety-five percent of the variance.]",
            m.AnimationGroup(m.FadeIn(explained, shift=0.15 * m.UP), m.Indicate(pc1_arrow, color=TEAL, scale_factor=1.1)),
        )
        self.say(
            "[So we can describe each face with one number instead of two], [and lose very little.]",
            m.AnimationGroup(m.FadeOut(pc2_arrow), m.Transform(dots, projected, run_time=2.0)),
            m.Circumscribe(explained, color=TEAL, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section(
            "The big picture",
            notes="Eigenfaces: Turk & Pentland, 1991. PCA maps of embeddings. Low-rank fine-tuning (LoRA): Hu et al., 2021, Episode 41. The catch: linear structure only; PCA depends on the units (standardise features).",
        )
        tiles = m.VGroup(
            tile("ep18-face", "Eigenfaces", "PCA on many face photos\nTurk & Pentland, 1991", SKY),
            tile("ep18-map", "Maps of embeddings", "hundreds of dimensions,\nsqueezed into a picture", GOOD),
            tile(variant("llm", VIOLET, VIOLET_FILL), "Low-rank fine-tuning", "small updates to big models\nEpisode 41 · Hu et al., 2021", VIOLET),
            tile("alert", "The catch", "only flat, straight-line patterns;\nresults depend on the units", BAD),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.45, 0.4)).move_to([0, -0.65, 0])

        self.say(
            "[Low rank shows up all over AI.] [PCA on many face photos gives eigenfaces], [an early way to recognise faces.]",
            self.new_header("THE BIG PICTURE", "Low rank, everywhere"),
            m.FadeIn(tiles[0], shift=0.2 * m.UP),
            m.Indicate(tiles[0][1][0], color=SKY, scale_factor=1.12),
        )
        self.say(
            "[It also squeezes embeddings with hundreds of dimensions into a flat map we can look at.]",
            m.FadeIn(tiles[1], shift=0.2 * m.UP),
        )
        self.say(
            "[And low-rank updates make fine-tuning huge models cheap], [as we'll see in episode forty-one.]",
            m.FadeIn(tiles[2], shift=0.2 * m.UP),
            m.Indicate(tiles[2][1][1][1], color=VIOLET, scale_factor=1.05),
        )
        self.say(
            "[The catch: these methods only find flat, straight-line structure], [and the answer depends on the units you measure in.]",
            m.FadeIn(tiles[3], shift=0.2 * m.UP),
            m.Wiggle(tiles[3][1][0], scale_value=1.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Decompose, sort, keep.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Decompose", "Sort", "Keep"), (VIOLET, AMBER, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        group = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Most data is simpler than it looks: a few directions tell most of the story.", 30, MUTED).next_to(group, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [decompose], [sort], [and keep the top few.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Most data is simpler than it looks: a few directions tell most of the story.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group, tagline)], run_time=0.9)
