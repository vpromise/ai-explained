"""AI, Explained · Episode 36 · Multimodal Models

Running example: matching captions to pictures. Four drawn "photos" (a dog with a ball, a
cat asleep on a sofa, a bicycle on a road, a sailboat at sunset) and their four captions.

- the idea: an image encoder and a text encoder map pictures and captions into one shared
  space; closeness is cosine similarity (Episode 9);
- CLIP (Radford et al., 2021): a batch of N pairs gives an N x N similarity matrix; the N true
  pairs are its diagonal, the rest are free negatives;
- the loss (InfoNCE, van den Oord et al., 2018; CLIP's symmetric version): a softmax over each
  row with a temperature, minus the log of the diagonal entry, averaged over rows and columns;
- training: a tiny toy CLIP, trained right here, pulls the pairs together;
- zero-shot classification by captions ("a photo of a dog."), vision-language models (image
  patches become tokens for a language model), and the catch (counting, word order).

The toy CLIP is real and computed below with seeded numpy: fixed random "features" for each
picture and caption, two linear encoders into a 2-D shared space (unit vectors), trained by
gradient descent on the symmetric InfoNCE loss with temperature 0.1. The grid, the circle, the
loss and the worked example all come from it; the asserts tie the narrated numbers to it.
The zero-shot similarity scores are illustrative ("for example").

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- colours

LIME_FILL = "#1E2A0F"
# the four pairs (category colours, the same all film long)
PAIRS = [
    # key, caption, caption on two lines, colour, fill
    ("dog", "a dog with a ball", ("a dog with", "a ball"), AMBER, AMBER_FILL),
    ("cat", "a cat asleep on a sofa", ("a cat asleep", "on a sofa"), PINK, PINK_FILL),
    ("bike", "a bicycle on a road", ("a bicycle", "on a road"), LIME, LIME_FILL),
    ("boat", "a sailboat at sunset", ("a sailboat", "at sunset"), SKY, SKY_FILL),
]
N = len(PAIRS)
COLORS = [p[3] for p in PAIRS]
FILLS = [p[4] for p in PAIRS]
MODEL, MODEL_FILL = VIOLET, VIOLET_FILL  # the encoders and the language model
SIM = TEAL  # similarity scores (the heat of the grid)
MATCH, MATCH_FILL = GOOD, GOOD_FILL  # the true pairs, the diagonal
LOSS_C = BAD  # the loss

# ---------------------------------------------------------------------------- the toy CLIP

TAU = 0.1  # the temperature of the toy (and of the worked example)
DIM_IN = 6  # the size of each fixed random "feature" vector
LR, STEPS = 0.0005, 2000


def unit(z):
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def softmax_rows(a):
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def clip_loss(s, tau=TAU):
    """CLIP's symmetric InfoNCE: cross-entropy of each row (picture -> caption) and each column
    (caption -> picture), the true pair on the diagonal, averaged."""
    rows = -np.log(np.diag(softmax_rows(s / tau))).mean()
    cols = -np.log(np.diag(softmax_rows(s.T / tau))).mean()
    return 0.5 * (rows + cols)


def loss_and_grads(wi, wt, xi, xt):
    zi, zt = xi @ wi.T, xt @ wt.T
    u, v = unit(zi), unit(zt)
    s = u @ v.T
    pr, pc = softmax_rows(s / TAU), softmax_rows(s.T / TAU).T
    eye = np.eye(N)
    ds = 0.5 / TAU / N * ((pr - eye) + (pc - eye))  # d loss / d similarity

    def through_unit(du, z):
        n = np.linalg.norm(z, axis=1, keepdims=True)
        y = z / n
        return (du - (du * y).sum(1, keepdims=True) * y) / n

    gi = through_unit(ds @ v, zi).T @ xi
    gt = through_unit(ds.T @ u, zt).T @ xt
    return clip_loss(s), u, v, gi, gt


def train(seed=2):
    rng = np.random.default_rng(seed)
    xi, xt = rng.normal(size=(N, DIM_IN)), rng.normal(size=(N, DIM_IN))
    wi, wt = 0.5 * rng.normal(size=(2, DIM_IN)), 0.5 * rng.normal(size=(2, DIM_IN))
    # the hand-written gradient agrees with finite differences
    _, _, _, gi, _ = loss_and_grads(wi, wt, xi, xt)
    bump = np.zeros_like(wi)
    bump[0, 0] = 1e-6
    numeric = (loss_and_grads(wi + bump, wt, xi, xt)[0] - loss_and_grads(wi - bump, wt, xi, xt)[0]) / 2e-6
    assert abs(numeric - gi[0, 0]) < 1e-5
    losses, angles = [], []
    for _ in range(STEPS + 1):
        loss, u, v, gi, gt = loss_and_grads(wi, wt, xi, xt)
        losses.append(loss)
        angles.append(np.r_[np.arctan2(u[:, 1], u[:, 0]), np.arctan2(v[:, 1], v[:, 0])])
        wi, wt = wi - LR * gi, wt - LR * gt
    return np.array(losses), np.unwrap(np.array(angles), axis=0)


LOSSES, ANGLES = train()  # ANGLES[k] = [4 picture angles, 4 caption angles] after k steps
MOTION = np.r_[0.0, np.cumsum(np.abs(np.diff(ANGLES, axis=0)).sum(1))]
MOTION /= MOTION[-1]  # the training film moves at an even visual pace


def angles_at(u):
    """Picture and caption angles at training progress u in [0, 1] (by share of the motion)."""
    k = np.interp(u, MOTION, np.arange(STEPS + 1))
    lo = int(np.floor(k))
    hi = min(lo + 1, STEPS)
    a = ANGLES[lo] + (k - lo) * (ANGLES[hi] - ANGLES[lo])
    return a[:N], a[N:]


def sims_at(u):
    pa, ca = angles_at(u)
    return np.cos(pa[:, None] - ca[None, :])  # unit vectors in 2-D: cosine = cos(angle difference)


S_START = sims_at(0.0)
S_END = sims_at(1.0)
SHOWN = np.round(S_START, 2)  # the grid as shown before training
assert abs(clip_loss(S_START) - LOSSES[0]) < 1e-9 and abs(clip_loss(S_END) - LOSSES[-1]) < 1e-9
assert round(LOSSES[0], 1) == 3.1  # "from about three point one"
assert LOSSES[-1] < 0.005  # "to almost zero"
assert np.argmax(SHOWN[0]) == 2  # "the dog picture even prefers the bicycle caption"
assert all(np.argmax(S_END[i]) == i and np.argmax(S_END[:, i]) == i for i in range(N))  # trained: the diagonal wins
assert np.diag(S_END).min() > 0.8

# the worked example: row 1 (the dog picture), from the numbers on screen
ROW = SHOWN[0]
ROW_P = softmax_rows(ROW[None] / TAU)[0]
ROW_L = -np.log(ROW_P[0])
assert round(ROW_P[0], 3) == 0.105 and round(100 * ROW_P[0]) == 10  # "only about ten percent"
assert round(ROW_L, 1) == 2.3 and round(-np.log(0.105), 1) == 2.3  # "about two point three"
PCTS = [int(round(100 * p)) for p in ROW_P]
assert sum(PCTS) == 100

# CLIP's batch, zero-shot, patches
CLIP_BATCH = 32_768  # Radford et al., 2021
assert CLIP_BATCH**2 > 1e9  # "over a billion scores in every step"
ZS_CLASSES = ["dog", "cat", "bicycle", "boat"]
ZS_SIMS = [0.31, 0.22, 0.17, 0.19]  # illustrative cosine scores for the new picture
assert ZS_CLASSES[int(np.argmax(ZS_SIMS))] == "dog"
IMG, PATCH = 224, 16  # ViT (Dosovitskiy et al., 2020)
assert (IMG // PATCH) ** 2 == 196  # "one hundred and ninety-six"

# ---------------------------------------------------------------------------- helpers


def fmt(v, places=2):
    s = f"{v:.{places}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def mix(a, b, t):
    t = min(1.0, max(0.0, t))
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + t * (y - x)):02X}" for x, y in zip(ca, cb))


HEAT_LO, HEAT_HI = CARD, mix(TEAL_FILL, TEAL, 0.62)


def heat(s):
    """Similarity -1 ... 1 → a single-hue (teal) ramp; the numbers stay in INK."""
    return mix(HEAT_LO, HEAT_HI, (s + 1) / 2)


def photo(key, height):
    return icon(f"ep36-{key}", height)


def stacked(lines, size, color=INK, weight=m.SEMIBOLD):
    """Centred lines on even baselines (descenders don't push a line up), centred on ORIGIN."""
    lead, cap = 0.0155 * size, 0.449 * size / 48
    words = m.VGroup()
    for k, line in enumerate(lines):
        t = text(line, size, color, weight=weight)
        sitting = [g for g, c in zip(t, line.replace(" ", "")) if c not in "gjpqy,"] or list(t)
        t.shift((-k * lead - min(g.get_bottom()[1] for g in sitting)) * m.UP)
        t.set_x(0)
        words.add(t)
    middle = (cap - (len(lines) - 1) * lead) / 2
    return words.shift(-middle * m.UP)


def caption_chip(lines, color=EDGE, fill=CARD, width=None, size=24, height=None):
    words = stacked(lines, size)
    box = card(width or words.width + 0.5, height or 0.0155 * size * (len(lines) - 1) + 0.449 * size / 48 + 0.5, color, fill, radius=0.18, stroke=2.5)
    return m.VGroup(box, words.shift(box.get_center()))


def encoder(word, width=1.9, height=1.35):
    pts = [[-width / 2, height / 2, 0], [width / 2, height / 4, 0], [width / 2, -height / 4, 0], [-width / 2, -height / 2, 0]]
    shape = m.Polygon(*pts).set_stroke(MODEL, width=2.5).set_fill(MODEL_FILL, opacity=1)
    label = m.VGroup(text(word, 22, INK, weight=m.BOLD), text("encoder", 20, MUTED)).arrange(m.DOWN, buff=0.06)
    label.move_to(shape).shift(0.1 * m.LEFT)
    return m.VGroup(shape, label)


def vec2(values, color=INK, size=22):
    nums = m.VGroup(*[code(fmt(v), size, color) for v in values]).arrange(m.DOWN, buff=0.12, aligned_edge=m.RIGHT)
    top, bottom = nums.get_top()[1] + 0.1, nums.get_bottom()[1] - 0.1

    def bracket(x, lip):
        corners = [[x + lip, top, 0], [x, top, 0], [x, bottom, 0], [x + lip, bottom, 0]]
        return m.VMobject().set_points_as_corners(corners).set_stroke(color, width=3)

    return m.VGroup(bracket(nums.get_left()[0] - 0.12, 0.1), nums, bracket(nums.get_right()[0] + 0.12, -0.1))


def on_circle(center, radius, angle):
    return center + radius * np.array([np.cos(angle), np.sin(angle), 0.0])


def pic_marker(color, size=0.24):
    return m.Square(side_length=size).set_fill(color, opacity=1).set_stroke(BG, width=2).set_z_index(3)


def cap_marker(color, radius=0.13):
    return m.Circle(radius=radius).set_fill(color, opacity=1).set_stroke(BG, width=2).set_z_index(4)


def legend(size=22):
    sq = pic_marker(INK, 0.22)
    dot = cap_marker(INK, 0.12)
    row = m.VGroup(
        m.VGroup(sq, text("picture", size, MUTED)).arrange(m.RIGHT, buff=0.15),
        m.VGroup(dot, text("caption", size, MUTED)).arrange(m.RIGHT, buff=0.15),
    ).arrange(m.RIGHT, buff=0.5)
    return row


def cite(words):
    return text(words, 20, MUTED)


def ep_recolored(name, color, fill):
    """kit.recolored(), but written as assets/ep36-<name>-<colour>.svg (episode-owned files)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep36-{name.removeprefix('ep36-')}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


# the grid (sections 3–5)
CW, CH, GAP = 1.62, 0.78, 0.16
COL_X = [-4.65 + k * (CW + GAP) for k in range(N)]
ROW_Y = [1.0 - k * (CH + GAP) for k in range(N)]  # rows 1.0 … −1.82
HEAD_Y = 1.95
THUMB_X = -6.2
PANEL_X = 4.3  # the right-hand panel's centre


def cell_center(i, j):
    return np.array([COL_X[j], ROW_Y[i], 0.0])


# ---------------------------------------------------------------------------- the episode


class Ep36Multimodal(Episode):
    number = 36

    def lesson(self):
        self.hook()
        self.idea()
        self.batch()
        self.loss()
        self.training()
        self.zero_shot()
        self.vision_language()
        self.catch()
        self.recap()

    def fade_live(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Four drawn pictures, four shuffled captions: how can a model match them?")
        xs = [-4.05, -1.35, 1.35, 4.05]
        photos = [photo(p[0], 1.5).move_to([x, 1.2, 0]) for p, x in zip(PAIRS, xs)]
        order = [2, 0, 3, 1]  # caption shown at each slot: bicycle, dog, sailboat, cat
        chips = {}
        for slot, i in enumerate(order):
            chips[i] = caption_chip(PAIRS[i][2], width=2.3, size=24).move_to([xs[slot], -1.45, 0])
        links = [
            m.Line(photos[i].get_bottom(), chips[i].get_top(), buff=0.12).set_stroke(COLORS[i], width=4)
            for i in range(N)
        ]
        today = m.VGroup(
            chip("contrastive learning", TEAL, TEAL_FILL, width=3.9, height=0.75, size=24),
            chip("one shared space", TEAL, TEAL_FILL, width=3.9, height=0.75, size=24),
            chip("models that can see", TEAL, TEAL_FILL, width=3.9, height=0.75, size=24),
        ).arrange(m.RIGHT, buff=0.3).move_to([0, -3.1, 0])

        self.say(
            "So... [here are four pictures], [and four captions, all shuffled.]",
            m.LaggedStart(*[m.FadeIn(p, shift=0.3 * m.DOWN) for p in photos], lag_ratio=0.2),
            m.LaggedStart(*[m.FadeIn(chips[i], shift=0.3 * m.UP) for i in order], lag_ratio=0.2),
        )
        self.say(
            "[You can match them in a second.] [But how can a model learn that a picture and a sentence mean the same thing?]",
            m.LaggedStart(*[m.Create(l) for l in links], lag_ratio=0.25),
            m.AnimationGroup(*[m.AnimationGroup(pulse(l, COLORS[i], 8), chips[i][0].animate.set_stroke(COLORS[i])) for i, l in enumerate(links)]),
        )
        self.say(
            "Today: [contrastive learning], [one shared space for words and images], [and models that can see.]",
            m.FadeIn(today[0], shift=0.3 * m.UP),
            m.FadeIn(today[1], shift=0.3 * m.UP),
            m.FadeIn(today[2], shift=0.3 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the idea: one shared space

    def idea(self):
        pa, ca = angles_at(1.0)
        self.next_section(
            "Two encoders, one space",
            notes="Image encoder and text encoder map into one shared space (CLIP: a few hundred dimensions; "
            f"drawn here in 2-D from the toy model after training). Dog pair cosine {S_END[0, 0]:.2f}.",
        )
        center, radius = np.array([3.3, -0.45, 0.0]), 2.15
        ring = m.Circle(radius=radius).set_stroke(EDGE, width=3).move_to(center)
        cross = m.VGroup(
            m.Line(center + (radius + 0.25) * m.LEFT, center + (radius + 0.25) * m.RIGHT),
            m.Line(center + (radius + 0.25) * m.DOWN, center + (radius + 0.25) * m.UP),
        ).set_stroke(FAINT, width=1.5)
        space_name = text("shared space", 24, MUTED, weight=m.SEMIBOLD).move_to(center + np.array([1.9, 2.6, 0]))

        pics = [pic_marker(COLORS[i]).move_to(on_circle(center, radius, pa[i])) for i in range(N)]
        caps = [cap_marker(COLORS[i]).move_to(on_circle(center, radius, ca[i])) for i in range(N)]
        thumbs = [photo(PAIRS[i][0], 0.6).move_to(on_circle(center, radius + 0.62, pa[i])) for i in range(N)]
        def short_way(a, b):
            return (b - a + np.pi) % (2 * np.pi) - np.pi

        arcs = [
            m.Arc(radius=radius, start_angle=pa[i], angle=short_way(pa[i], ca[i]), arc_center=center).set_stroke(COLORS[i], width=6).set_z_index(2)
            for i in range(N)
        ]
        key = legend().move_to([center[0], -3.3, 0])

        # the pipeline, for the dog
        top_y, low_y = 1.15, -1.75
        dog_photo = photo("dog", 1.3).move_to([-5.45, top_y, 0])
        dog_caption = caption_chip(PAIRS[0][2], AMBER, CARD, width=1.95, size=22).move_to([-5.45, low_y, 0])
        enc_img = encoder("image").move_to([-2.75, top_y, 0])
        enc_txt = encoder("text").move_to([-2.75, low_y, 0])
        v_img = vec2([np.cos(pa[0]), np.sin(pa[0])]).move_to([-0.65, top_y, 0])
        v_txt = vec2([np.cos(ca[0]), np.sin(ca[0])]).move_to([-0.65, low_y, 0])
        a1 = arrow(dog_photo.get_right(), enc_img.get_left(), MUTED)
        a2 = arrow(enc_img.get_right(), v_img.get_left(), MUTED)
        a3 = arrow(dog_caption.get_right(), enc_txt.get_left(), MUTED)
        a4 = arrow(enc_txt.get_right(), v_txt.get_left(), MUTED)
        dims = pill("drawn in 2-D · real models: hundreds of dimensions", EDGE, CARD, 20).move_to([-3.35, -3.2, 0])
        dims[1].set_color(MUTED)
        pipeline = [dog_photo, dog_caption, enc_img, enc_txt, v_img, v_txt, a1, a2, a3, a4, dims]

        # cosine similarity
        formula = m.MathTex("s", "=", r"\cos", r"\theta", font_size=64, color=INK).move_to([-3.6, 1.35, 0])
        formula[0].set_color(SIM)
        formula[3].set_color(SIM)
        terms = m.VGroup(
            m.VGroup(tex("s", 36, SIM), text("similarity, from −1 to 1", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(tex(r"\theta", 36, SIM), text("the angle between two vectors", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
        ).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([-3.6, -1.2, 0])
        phi = m.ValueTracker(ca[0])
        pic_line = m.Line(center, on_circle(center, radius, pa[0])).set_stroke(AMBER, width=5).set_z_index(1)
        probe = m.always_redraw(
            lambda: m.Line(center, on_circle(center, radius, phi.get_value())).set_stroke(INK, width=5).set_z_index(1)
        )

        def angle_arc():
            sweep = phi.get_value() - pa[0]
            if abs(sweep) < 1e-3:
                sweep = 1e-3
            return m.Arc(radius=0.6, start_angle=pa[0], angle=sweep, arc_center=center).set_stroke(SIM, width=4).set_z_index(2)

        arc = m.always_redraw(angle_arc)
        readout = m.always_redraw(
            lambda: m.VGroup(tex("s =", 44, INK), code(fmt(np.cos(phi.get_value() - pa[0])), 40, SIM)).arrange(m.RIGHT, buff=0.25).move_to([-3.6, 0.1, 0])
        )
        center_dot = m.Dot(center, radius=0.07, color=INK).set_z_index(5)

        self.say(
            "[Here's the idea.] [An image encoder turns a picture into a vector]: [a point in space.]",
            m.AnimationGroup(self.new_header("THE IDEA", "Two encoders, one shared space"), m.Create(ring), m.FadeIn(cross), m.FadeIn(space_name)),
            m.AnimationGroup(m.FadeIn(dog_photo, shift=0.2 * m.RIGHT), m.GrowArrow(a1), m.FadeIn(enc_img), m.GrowArrow(a2), m.FadeIn(v_img, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(pics[0], target_position=v_img), m.FadeIn(thumbs[0], target_position=v_img)),
        )
        self.say(
            "[A separate text encoder turns a caption into a vector], [in the very same space.]",
            m.AnimationGroup(m.FadeIn(dog_caption, shift=0.2 * m.RIGHT), m.GrowArrow(a3), m.FadeIn(enc_txt), m.GrowArrow(a4), m.FadeIn(v_txt, shift=0.2 * m.RIGHT)),
            m.FadeIn(caps[0], target_position=v_txt),
        )
        self.say("[Real models use hundreds of dimensions. We'll draw just two.]", m.FadeIn(dims, shift=0.2 * m.UP))
        self.say(
            "[After training, each picture lands near its own caption], [and far from the others.]",
            m.AnimationGroup(
                *[m.FadeOut(p) for p in pipeline],
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(pics[i], scale=0.4), m.FadeIn(thumbs[i], scale=0.6), m.FadeIn(caps[i], scale=0.4)) for i in range(1, N)], lag_ratio=0.3),
            ),
            m.AnimationGroup(m.LaggedStart(*[m.Create(a) for a in arcs], lag_ratio=0.2), m.FadeIn(key, shift=0.2 * m.UP)),
        )
        self.say(
            "[Closeness is cosine similarity, from episode nine]: [the cosine of the angle between two vectors.]",
            m.AnimationGroup(m.Write(formula), m.FadeIn(terms[0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(center_dot), m.Create(pic_line), m.FadeIn(probe), m.FadeIn(arc), m.FadeIn(readout), m.FadeIn(terms[1], shift=0.15 * m.UP)),
        )
        self.say(
            "[Same direction: a score of one.] [At right angles: zero.] [Opposite: minus one.]",
            phi.animate(run_time=1.2).set_value(pa[0]),
            phi.animate(run_time=1.3).set_value(pa[0] + np.pi / 2),
            phi.animate(run_time=1.3).set_value(pa[0] + np.pi),
        )
        self.wait(0.3)
        self.fade_live(probe, arc, readout)
        self.clear()

    # ------------------------------------------------------------------ CLIP: a batch, a grid

    def batch(self):
        self.next_section(
            "A batch, a grid",
            notes="CLIP (Radford et al., 2021): 400 million image-text pairs from the internet; batch size 32,768 "
            f"(32,768² ≈ {CLIP_BATCH**2 / 1e9:.2f} billion scores per step). The grid is the toy model before training "
            f"(seeded, illustrative): dog row {', '.join(fmt(v) for v in SHOWN[0])}.",
        )
        thumbs = [photo(PAIRS[i][0], 0.66).move_to([THUMB_X, ROW_Y[i], 0]) for i in range(N)]
        heads = [caption_chip(PAIRS[j][2], COLORS[j], CARD, width=CW, size=20, height=0.86).move_to([COL_X[j], HEAD_Y, 0]) for j in range(N)]
        rects = [[None] * N for _ in range(N)]
        nums = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                rects[i][j] = m.RoundedRectangle(corner_radius=0.1, width=CW, height=CH).set_stroke(EDGE, width=1.5).set_fill(heat(SHOWN[i, j]), opacity=1).move_to(cell_center(i, j))
                nums[i][j] = code(fmt(SHOWN[i, j]), 24, INK).move_to(cell_center(i, j)).set_z_index(2)
        self.thumbs, self.heads, self.rects, self.nums = thumbs, heads, rects, nums
        diag = m.VGroup(
            *[m.RoundedRectangle(corner_radius=0.12, width=CW + 0.08, height=CH + 0.08).set_stroke(MATCH, width=4).move_to(cell_center(i, i)).set_z_index(3) for i in range(N)]
        )
        self.diag = diag
        off = m.VGroup(*[rects[i][j] for i in range(N) for j in range(N) if i != j])

        big = text("400 million", 56, INK, weight=m.BOLD).move_to([PANEL_X, 1.75, 0])
        big_sub = stacked(("pictures with captions,", "gathered from the web"), 24, MUTED, m.NORMAL).next_to(big, m.DOWN, buff=0.25)
        big_cite = cite("CLIP · Radford et al., 2021").next_to(big_sub, m.DOWN, buff=0.2)
        pairs_pill = pill("4 true pairs", MATCH, MATCH_FILL, 22)
        neg_pill = pill("12 mismatches", EDGE, CARD, 22)
        neg_pill[1].set_color(INK)
        up = text("↑ push these up", 22, MATCH, weight=m.SEMIBOLD)
        down = text("↓ push these down", 22, MUTED, weight=m.SEMIBOLD)
        row1 = m.VGroup(pairs_pill, up).arrange(m.RIGHT, buff=0.3)
        row2 = m.VGroup(neg_pill, down).arrange(m.RIGHT, buff=0.3)
        notes = m.VGroup(row1, row2).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).move_to([-3.0, -3.0, 0]).align_to([-6.6, 0, 0], m.LEFT)
        up.align_to(down, m.LEFT)
        bad_cell = m.RoundedRectangle(corner_radius=0.12, width=CW + 0.08, height=CH + 0.08).set_stroke(LOSS_C, width=4).move_to(cell_center(0, 2)).set_z_index(3)
        batch_card = m.VGroup(
            text("CLIP's batches", 24, MUTED, weight=m.SEMIBOLD),
            text(f"{CLIP_BATCH:,} pairs", 34, INK, weight=m.BOLD),
            text("over a billion scores per step", 24, SIM, weight=m.SEMIBOLD),
        ).arrange(m.DOWN, buff=0.16).move_to([PANEL_X, -1.35, 0])
        self.right = [big, big_sub, big_cite, batch_card]
        self.notes = [notes, bad_cell]

        self.say(
            "[CLIP learned this from four hundred million pictures with captions, gathered from the web.]",
            m.AnimationGroup(self.new_header("CLIP", "Score every picture against every caption"), m.FadeIn(big, shift=0.2 * m.UP), m.FadeIn(big_sub), m.FadeIn(big_cite)),
        )
        self.say(
            "[It trains on batches. Here are four pictures], [and their four captions.]",
            m.LaggedStart(*[m.FadeIn(t, shift=0.2 * m.RIGHT) for t in thumbs], lag_ratio=0.2),
            m.LaggedStart(*[m.FadeIn(h, shift=0.2 * m.DOWN) for h in heads], lag_ratio=0.2),
        )
        self.say(
            "[Score every picture against every caption], and you get a four by four grid.",
            m.LaggedStart(*[m.FadeIn(m.VGroup(rects[i][j], nums[i][j]), scale=0.85) for i in range(N) for j in range(N)], lag_ratio=0.06),
        )
        self.say(
            "[Before training, the scores are noise]: [the dog picture even prefers the bicycle caption.]",
            m.Circumscribe(m.VGroup(*rects[0]), color=INK, buff=0.06),
            m.AnimationGroup(m.Create(bad_cell), m.Circumscribe(heads[2], color=LIME, buff=0.06)),
        )
        self.say(
            "[The four true pairs sit on the diagonal.] [The other twelve cells are mismatches]: free negative examples.",
            m.AnimationGroup(m.FadeOut(bad_cell), m.LaggedStart(*[m.Create(d) for d in diag], lag_ratio=0.2), m.FadeIn(pairs_pill, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(neg_pill, shift=0.2 * m.UP), off.animate(rate_func=m.there_and_back, run_time=1.0).set_stroke(MUTED, width=3)),
        )
        self.say(
            "[Training pushes the diagonal up], [and everything else down.]",
            m.FadeIn(up, shift=0.2 * m.UP),
            m.FadeIn(down, shift=0.2 * m.DOWN),
        )
        self.say(
            "[CLIP used batches of about thirty-two thousand pairs]: [over a billion scores in every step.]",
            m.AnimationGroup(m.FadeIn(batch_card[0]), m.FadeIn(batch_card[1], shift=0.2 * m.UP)),
            m.FadeIn(batch_card[2], shift=0.2 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the loss

    def loss(self):
        self.next_section(
            "The loss",
            notes=f"InfoNCE (van den Oord et al., 2018), CLIP's symmetric form. Dog row / 0.1 → softmax "
            f"{', '.join(f'{p:.3f}' for p in ROW_P)}; loss −log {ROW_P[0]:.3f} = {ROW_L:.3f}. "
            "CLIP learns its temperature (initialised at 0.07). Toy values are illustrative.",
        )
        row_box = m.RoundedRectangle(corner_radius=0.16, width=COL_X[-1] - THUMB_X + CW / 2 + 0.75, height=CH + 0.2)
        row_box.set_stroke(INK, width=3).move_to([(THUMB_X - 0.5 + COL_X[-1] + CW / 2 + 0.25) / 2, ROW_Y[0], 0]).set_z_index(4)
        col_box = m.RoundedRectangle(corner_radius=0.16, width=CW + 0.16, height=HEAD_Y + 0.5 - (ROW_Y[-1] - CH / 2 - 0.13))
        col_box.set_stroke(INK, width=3).move_to([COL_X[0], (HEAD_Y + 0.5 + ROW_Y[-1] - CH / 2 - 0.13) / 2, 0]).set_z_index(4)

        num = m.MathTex(r"\exp(", r"s_{ii}", "/", r"\tau", ")", font_size=38, color=INK)
        den = m.MathTex(r"\sum_{j}", r"\exp(", r"s_{ij}", "/", r"\tau", ")", font_size=38, color=INK)
        num[1].set_color(MATCH)
        den[2].set_color(SIM)
        num[3].set_color(AMBER)
        den[4].set_color(AMBER)
        bar = m.Line([-1.42, 0, 0], [1.42, 0, 0]).set_stroke(INK, width=2.5)
        num.next_to(bar, m.UP, buff=0.12)
        den.next_to(bar, m.DOWN, buff=0.12)
        lhs = m.MathTex(r"L_i", "=", r"-\log", font_size=38, color=INK).next_to(bar, m.LEFT, buff=0.2)
        lhs[0].set_color(LOSS_C)
        formula = m.VGroup(lhs, num, bar, den).move_to([PANEL_X, 1.6, 0])
        source = cite("InfoNCE · van den Oord et al., 2018").next_to(formula, m.DOWN, buff=0.22)
        tau_note = m.VGroup(tex(r"\tau = 0.1", 32, AMBER), text("here · CLIP learns it", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        tau_note.next_to(source, m.DOWN, buff=0.28)

        base, tall = -2.3, 1.45
        xs = [PANEL_X - 1.65 + k * 1.1 for k in range(N)]
        bars = [m.Rectangle(width=0.62, height=max(0.02, tall * p), stroke_width=0).set_fill(COLORS[j], opacity=0.9).move_to([xs[j], base, 0], aligned_edge=m.DOWN) for j, p in enumerate(ROW_P)]
        floor = m.Line([xs[0] - 0.55, base, 0], [xs[-1] + 0.55, base, 0]).set_stroke(FAINT, width=2.5)
        pct = [text(f"{v}%", 22, INK, weight=m.SEMIBOLD).next_to(b, m.UP, buff=0.12) for v, b in zip(PCTS, bars)]
        names = [text(k, 20, MUTED).move_to([x, base - 0.28, 0]) for k, x in zip(("dog", "cat", "bicycle", "boat"), xs)]
        right_ring = m.RoundedRectangle(corner_radius=0.1, width=0.86, height=tall * ROW_P[0] + 0.62).set_stroke(MATCH, width=3)
        right_ring.move_to([xs[0], base + (tall * ROW_P[0] + 0.62) / 2 - 0.08, 0])
        worked = m.MathTex(r"L_1 = -\log 0.105", r"\approx 2.3", font_size=40, color=INK).move_to([PANEL_X, -3.2, 0])
        worked[1].set_color(LOSS_C)
        both = m.VGroup(
            text("rows: picture → caption", 22, INK, weight=m.SEMIBOLD),
            text("columns: caption → picture", 22, INK, weight=m.SEMIBOLD),
            text("CLIP averages both", 22, MUTED),
        ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([-3.0, -3.05, 0]).align_to([-6.6, 0, 0], m.LEFT)
        self.loss_parts = [row_box, col_box, formula, source, tau_note, floor, *bars, *pct, *names, right_ring, worked, both]

        self.say(
            "[For each picture, the loss asks a multiple-choice question]: [which of these captions is mine?]",
            m.AnimationGroup(
                self.new_header("THE LOSS", "InfoNCE: which caption is mine?"),
                *[m.FadeOut(x) for x in self.right], *[m.FadeOut(x) for x in self.notes],
                m.Create(row_box),
            ),
            m.LaggedStart(*[m.Indicate(h, color=COLORS[j], scale_factor=1.08) for j, h in enumerate(self.heads)], lag_ratio=0.15),
        )
        self.say("[Here's the loss for one picture.]", m.AnimationGroup(m.Write(formula, run_time=1.6), m.FadeIn(source)))
        self.say(
            "[On top, the score of the matching pair]; [below, a sum over every caption in the batch.]",
            m.AnimationGroup(m.Circumscribe(num, color=MATCH, buff=0.08), m.Circumscribe(self.rects[0][0], color=MATCH, buff=0.04)),
            m.AnimationGroup(m.Circumscribe(den, color=SIM, buff=0.08), m.Circumscribe(m.VGroup(*self.rects[0]), color=SIM, buff=0.04)),
        )
        self.say(
            "[Each score is divided by a temperature, here zero point one]: [a small temperature sharpens the softmax, as in episode sixteen.]",
            m.AnimationGroup(m.Circumscribe(num[3], color=AMBER, buff=0.06), m.Circumscribe(den[4], color=AMBER, buff=0.06), m.FadeIn(tau_note[0], shift=0.15 * m.UP)),
            m.FadeIn(tau_note[1], shift=0.15 * m.UP),
        )
        self.say(
            "[For the dog picture, the softmax puts only about ten percent on the right caption.]",
            m.AnimationGroup(
                m.Create(floor),
                m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.15),
                m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in (*pct, *names)], lag_ratio=0.06),
                m.Create(right_ring),
            ),
        )
        self.say(
            "[Minus the log of that is the loss]: [about two point three.]",
            m.Write(worked[0]),
            m.FadeIn(worked[1], shift=0.15 * m.LEFT),
        )
        self.say(
            "[CLIP asks the same question down each column, from caption to picture], [and averages both directions.]",
            m.AnimationGroup(m.FadeOut(row_box), m.Create(col_box), m.FadeIn(both[0]), m.FadeIn(both[1])),
            m.FadeIn(both[2], shift=0.15 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ training

    def training(self):
        pa0, ca0 = angles_at(0.0)
        self.next_section(
            "Training",
            notes=f"The toy CLIP trained by gradient descent ({STEPS} steps, learning rate {LR}, temperature {TAU}): "
            f"loss {LOSSES[0]:.2f} → {LOSSES[-1]:.4f}; trained diagonal {', '.join(fmt(v) for v in np.diag(S_END))}.",
        )
        prog = m.ValueTracker(0.0)
        center, radius = np.array([PANEL_X, -0.45, 0.0]), 1.45
        ring = m.Circle(radius=radius).set_stroke(EDGE, width=3).move_to(center)
        cross = m.VGroup(
            m.Line(center + (radius + 0.2) * m.LEFT, center + (radius + 0.2) * m.RIGHT),
            m.Line(center + (radius + 0.2) * m.DOWN, center + (radius + 0.2) * m.UP),
        ).set_stroke(FAINT, width=1.5)
        key = legend(20).move_to([center[0], -3.25, 0])

        def pics_now():
            return angles_at(prog.get_value())

        markers = []
        for i in range(N):
            markers.append(m.always_redraw(lambda i=i: pic_marker(COLORS[i], 0.2).move_to(on_circle(center, radius, pics_now()[0][i]))))
            markers.append(m.always_redraw(lambda i=i: cap_marker(COLORS[i], 0.11).move_to(on_circle(center, radius, pics_now()[1][i]))))
        thumbs = []
        for i in range(N):
            t = photo(PAIRS[i][0], 0.42).move_to(on_circle(center, radius + 0.5, pa0[i]))
            t.add_updater(lambda mob, i=i: mob.move_to(on_circle(center, radius + 0.5, pics_now()[0][i])))
            thumbs.append(t)

        loss_name = text("loss", 26, MUTED, weight=m.SEMIBOLD)
        loss_val = m.always_redraw(
            lambda: code(fmt(clip_loss(sims_at(prog.get_value()))), 36, LOSS_C).next_to(loss_name, m.RIGHT, buff=0.3)
        )
        loss_name.move_to([PANEL_X - 0.75, 2.05, 0])
        final = pill(f"{fmt(LOSSES[0], 1)}  →  almost 0", MATCH, MATCH_FILL, 22).move_to([center[0], -3.25, 0])

        # the grid comes alive: fills and numbers follow the training
        live_nums = []
        for i in range(N):
            for j in range(N):
                self.rects[i][j].add_updater(lambda r, i=i, j=j: r.set_fill(heat(sims_at(prog.get_value())[i, j]), opacity=1))
                live_nums.append(m.always_redraw(lambda i=i, j=j: code(fmt(sims_at(prog.get_value())[i, j]), 24, INK).move_to(cell_center(i, j)).set_z_index(2)))
        old_nums = [self.nums[i][j] for i in range(N) for j in range(N)]
        assert all(fmt(sims_at(0.0)[i, j]) == fmt(SHOWN[i, j]) for i in range(N) for j in range(N))  # a seamless swap
        self.remove(*old_nums)
        self.add(*live_nums)

        self.say(
            "[Now we train.] [Each step nudges both encoders to lower the loss.]",
            m.AnimationGroup(
                self.new_header("TRAINING", "Pull together, push apart"),
                *[m.FadeOut(x) for x in self.loss_parts if x is not self.loss_parts[0]],
                m.Create(ring), m.FadeIn(cross),
            ),
            m.AnimationGroup(*[m.FadeIn(x) for x in markers], *[m.FadeIn(t, scale=0.6) for t in thumbs], m.FadeIn(loss_name), m.FadeIn(loss_val), m.FadeIn(key)),
        )
        self.say(
            "[Matching pairs pull together, mismatches push apart], [and the diagonal lights up.]",
            prog.animate(run_time=4.2, rate_func=m.smooth).set_value(1.0),
            m.LaggedStart(*[m.Circumscribe(d, color=MATCH, buff=0.04) for d in self.diag], lag_ratio=0.25),
        )
        self.say(
            "[The loss falls from about three point one to almost zero.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(loss_name, loss_val), color=LOSS_C, buff=0.12), m.FadeOut(key), m.FadeIn(final, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        live = [*markers, *thumbs, loss_val, *live_nums, *[self.rects[i][j] for i in range(N) for j in range(N)]]
        self.fade_live(*live)
        self.clear()

    # ------------------------------------------------------------------ zero-shot

    def zero_shot(self):
        self.next_section(
            "Zero-shot classification",
            notes="Prompts 'a photo of a {label}.' as in Radford et al., 2021. Scores are illustrative. "
            "CLIP matched the original ResNet-50 on ImageNet zero-shot without using any of its 1.28 million labelled training images (CLIP abstract).",
        )
        new = photo("puppy", 1.9).move_to([-4.75, 0.45, 0])
        new_name = text("a new picture", 22, MUTED).next_to(new, m.DOWN, buff=0.22)
        ys = [1.6, 0.6, -0.4, -1.4]
        prompts = [caption_chip((f"a photo of a {c}.",), COLORS[k], CARD, width=3.7, size=24, height=0.72).move_to([-0.15, y, 0]) for k, (c, y) in enumerate(zip(ZS_CLASSES, ys))]
        links = [m.DashedLine(new.get_right() + 0.1 * m.RIGHT, p.get_left() + 0.08 * m.LEFT, dash_length=0.1).set_stroke(MUTED, width=2) for p in prompts]
        x0, scale = 2.05, 10.0
        bars = [m.Rectangle(width=s * scale, height=0.42, stroke_width=0).set_fill(SIM, opacity=0.85).move_to([x0, y, 0], aligned_edge=m.LEFT) for s, y in zip(ZS_SIMS, ys)]
        vals = [code(fmt(s), 24, INK).next_to(b, m.RIGHT, buff=0.2) for s, b in zip(ZS_SIMS, bars)]
        sim_head = text("similarity", 22, SIM, weight=m.SEMIBOLD).move_to([x0 + 1.0, 2.35, 0])
        win = m.RoundedRectangle(corner_radius=0.2, width=vals[0].get_right()[0] - prompts[0].get_left()[0] + 0.4, height=0.98)
        win.set_stroke(MATCH, width=4).move_to([(vals[0].get_right()[0] + prompts[0].get_left()[0]) / 2, ys[0], 0])
        answer = pill("→ a dog", MATCH, MATCH_FILL, 26).next_to(new_name, m.DOWN, buff=0.3)
        fact = m.VGroup(
            text("No labelled pictures needed: new classes are just new captions.", 24, INK, weight=m.SEMIBOLD),
            m.VGroup(
                text("On ImageNet, as accurate as a classic ResNet-50 trained on its 1.28 million labelled images", 22, MUTED),
            ),
            cite("Radford et al., 2021"),
        ).arrange(m.DOWN, buff=0.14).move_to([0, -2.85, 0])

        self.say("[Here's a bonus: a classifier you never trained.]", self.new_header("ZERO-SHOT", "A classifier written in words"))
        self.say(
            "[Write one caption per class]: [a photo of a dog, a photo of a cat, and so on.]",
            m.FadeIn(prompts[0], shift=0.2 * m.LEFT),
            m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.LEFT) for p in prompts[1:]], lag_ratio=0.3),
        )
        self.say(
            "[For example, take a new picture], [and score it against every caption.]",
            m.AnimationGroup(m.FadeIn(new, shift=0.2 * m.RIGHT), m.FadeIn(new_name)),
            m.AnimationGroup(
                m.LaggedStart(*[m.Create(l) for l in links], lag_ratio=0.15),
                m.FadeIn(sim_head),
                m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in bars], lag_ratio=0.15),
                m.LaggedStart(*[m.FadeIn(v) for v in vals], lag_ratio=0.15),
            ),
        )
        self.say("[The best match is the answer]: [a dog.]", m.Create(win), m.FadeIn(answer, scale=1.2))
        self.say("[New classes just need new captions, not new labelled pictures.]", m.FadeIn(fact[0], shift=0.2 * m.UP))
        self.say(
            "[This way, CLIP matched a classic ImageNet classifier], [without using any of its one point two eight million labelled training images.]",
            m.FadeIn(fact[2], shift=0.15 * m.UP),
            m.FadeIn(fact[1], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ vision-language models

    def vision_language(self):
        self.next_section(
            "Vision-language models",
            notes="ViT (Dosovitskiy et al., 2020): 16×16-pixel patches, a 224×224 image → 14×14 = 196 patches. "
            "LLaVA (Liu et al., 2023): CLIP's ViT-L/14 patch features, a small projection, then a language model. "
            "Drawn here with 3×2 patches for clarity. Flamingo (Alayrac et al., 2022) instead uses cross-attention.",
        )
        pic = photo("dog", 1.8).move_to([-4.9, 1.3, 0])
        cols, rows = 3, 2
        pw, ph = pic.width / cols, pic.height / rows
        corner = pic.get_corner(m.UL)
        patches = [
            m.Rectangle(width=pw, height=ph).set_stroke(INK, width=2.5).set_fill(BG, opacity=0).move_to(corner + np.array([(c + 0.5) * pw, -(r + 0.5) * ph, 0])).set_z_index(2)
            for r in range(rows) for c in range(cols)
        ]
        question = bubble(text("What is the dog doing?", 26, INK, weight=m.SEMIBOLD), SKY, SKY_FILL, tail="left").move_to([-0.55, 1.55, 0])
        fact = m.VGroup(
            text("224 × 224 pixels, 16 × 16 patches", 22, INK, weight=m.SEMIBOLD),
            text("→ 14 × 14 = 196 patches", 24, AMBER, weight=m.BOLD),
        ).arrange(m.DOWN, buff=0.14)
        fact_box = card(fact.width + 0.5, fact.height + 0.4, EDGE, CARD, radius=0.2, stroke=2).move_to([4.35, 1.55, 0])
        fact.move_to(fact_box)
        fact_cite = cite("ViT · Dosovitskiy et al., 2020").next_to(fact_box, m.DOWN, buff=0.18)

        row_y = -0.6
        tok = 0.5
        img_tokens = [
            m.RoundedRectangle(corner_radius=0.08, width=tok, height=tok).set_stroke(AMBER, width=2.5).set_fill(AMBER_FILL, opacity=1).move_to([-6.0 + k * (tok + 0.14) + tok / 2, row_y, 0])
            for k in range(cols * rows)
        ]
        words = ["What", "is", "the", "dog", "doing", "?"]
        txt_tokens = m.VGroup(*[chip(w, SKY, SKY_FILL, width=max(0.62, 0.17 * len(w) + 0.45), height=0.62, size=24) for w in words]).arrange(m.RIGHT, buff=0.12)
        txt_tokens.move_to([0, row_y, 0]).align_to([img_tokens[-1].get_right()[0] + 0.45, 0, 0], m.LEFT)
        img_label = text("image tokens", 22, AMBER, weight=m.SEMIBOLD).next_to(m.VGroup(*img_tokens), m.UP, buff=0.22)
        txt_label = text("question tokens", 22, SKY, weight=m.SEMIBOLD).next_to(txt_tokens, m.UP, buff=0.22)
        steps = m.VGroup(
            text("vision encoder (often CLIP's)", 22, VIOLET, weight=m.SEMIBOLD),
            text("+ projection into word space", 22, SKY, weight=m.SEMIBOLD),
        ).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(m.VGroup(*img_tokens), m.DOWN, buff=0.3).align_to(img_tokens[0], m.LEFT)

        llm = icon("llm", 1.1).move_to([-0.2, -2.35, 0])
        everything = m.VGroup(*img_tokens, txt_tokens)
        feed = arrow([llm.get_x(), row_y - 0.36, 0], llm.get_top() + 0.05 * m.UP, MUTED)
        llm_name = text("language model", 22, MUTED).next_to(llm, m.DOWN, buff=0.15)
        reply = bubble(text("It's leaping to catch a ball.", 26, INK), VIOLET, VIOLET_FILL, tail="left").next_to(llm, m.RIGHT, buff=0.35).shift(0.3 * m.UP)
        reply_words = reply[2]
        llava = cite("e.g. LLaVA · Liu et al., 2023").next_to(llm_name, m.DOWN, buff=0.12)

        self.say(
            "[Matching is great for search. But what if we want to ask about a picture?]",
            m.AnimationGroup(self.new_header("VISION-LANGUAGE MODELS", "Pictures become tokens"), m.FadeIn(pic, shift=0.2 * m.RIGHT), m.FadeIn(question, shift=0.2 * m.UP)),
        )
        self.say(
            "[A vision-language model cuts the picture into patches], [like words in a sentence.]",
            m.LaggedStart(*[m.Create(p) for p in patches], lag_ratio=0.15),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, target_position=p) for t, p in zip(img_tokens, patches)], lag_ratio=0.12), m.FadeIn(img_label)),
        )
        self.say(
            "[For example, a two hundred and twenty-four pixel square picture, in sixteen pixel patches], [makes one hundred and ninety-six of them.]",
            m.AnimationGroup(m.FadeIn(fact_box), m.FadeIn(fact[0]), m.FadeIn(fact_cite)),
            m.FadeIn(fact[1], shift=0.15 * m.UP),
        )
        self.say(
            "[A vision encoder, often CLIP's own, turns each patch into a vector], [and a small projection maps it into the language model's word space.]",
            m.AnimationGroup(m.FadeIn(steps[0]), *[t.animate.set_stroke(VIOLET).set_fill(VIOLET_FILL) for t in img_tokens]),
            m.AnimationGroup(m.FadeIn(steps[1]), *[t.animate.set_stroke(SKY).set_fill(SKY_FILL) for t in img_tokens], img_label.animate.set_color(SKY)),
        )
        self.say(
            "[The language model reads the image tokens, then the question], [and answers in words]:",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, target_position=question) for t in txt_tokens], lag_ratio=0.1), m.FadeIn(txt_label)),
            m.AnimationGroup(m.GrowArrow(feed), m.FadeIn(llm, scale=0.8), m.FadeIn(llm_name), m.FadeIn(llava)),
        )
        self.say(
            "[It's leaping to catch a ball.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(reply[0], reply[1])), m.AddTextLetterByLetter(reply_words, time_per_char=0.035)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section(
            "The catch",
            notes="Counting: 'CLIP also struggles with more abstract and systematic tasks such as counting the number of objects' (Radford et al., 2021). "
            "Word order: CLIP-style models often behave like bags of words (Yuksekgonul et al., 2023).",
        )

        def panel(x, title):
            box = card(6.0, 4.6, BAD, CARD, stroke=2).move_to([x, -0.5, 0])
            head = text(title, 28, INK, weight=m.BOLD).move_to(box.get_top() + 0.55 * m.DOWN)
            return box, head

        box1, head1 = panel(-3.2, "Counting")
        dogs = m.VGroup(*[photo("dog", 0.85) for _ in range(3)]).arrange(m.RIGHT, buff=0.18).move_to([-3.2, 0.2, 0])
        guess = text("two? three? four?", 26, AMBER, weight=m.SEMIBOLD).move_to([-3.2, -0.7, 0])
        desc1 = text("weak at counting objects", 24, MUTED).move_to([-3.2, -1.6, 0])
        cite1 = cite("Radford et al., 2021").move_to(box1.get_bottom() + 0.4 * m.UP)

        box2, head2 = panel(3.2, "Word order")
        c1 = caption_chip(("a dog chasing a cat",), EDGE, CARD, width=4.0, size=24, height=0.7)
        approx = tex(r"\approx", 44, BAD)
        c2 = caption_chip(("a cat chasing a dog",), EDGE, CARD, width=4.0, size=24, height=0.7)
        order = m.VGroup(c1, approx, c2).arrange(m.DOWN, buff=0.1).move_to([3.2, -0.2, 0])
        desc2 = text("can score both almost the same", 24, MUTED).next_to(order, m.DOWN, buff=0.35)
        desc2.match_y(desc1)
        cite2 = cite("Yuksekgonul et al., 2023").move_to(box2.get_bottom() + 0.4 * m.UP)

        self.say(
            "[One catch: CLIP is weak at counting objects],",
            m.AnimationGroup(
                self.new_header("THE CATCH", "What matching misses"),
                m.FadeIn(box1), m.FadeIn(head1), m.LaggedStart(*[m.FadeIn(d, shift=0.2 * m.UP) for d in dogs], lag_ratio=0.2),
                m.FadeIn(guess), m.FadeIn(desc1), m.FadeIn(cite1),
            ),
        )
        self.say(
            "[and it can mix up word order]: [a dog chasing a cat, or a cat chasing a dog.]",
            m.AnimationGroup(m.FadeIn(box2), m.FadeIn(head2), m.FadeIn(desc2), m.FadeIn(cite2)),
            m.LaggedStart(m.FadeIn(c1, shift=0.2 * m.DOWN), m.FadeIn(approx), m.FadeIn(c2, shift=0.2 * m.UP), lag_ratio=0.4),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Encode, compare, contrast.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Encode", "Compare", "Contrast"), (VIOLET, TEAL, GOOD))]).arrange(m.RIGHT, buff=1.2)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Two encoders, one shared space: where pictures and words meet.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [encode], [compare], [contrast.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Two encoders, one shared space: where pictures and words meet.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
