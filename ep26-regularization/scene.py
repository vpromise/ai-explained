"""AI, Explained · Episode 26 · Regularization in Practice

Running example: a small neural network that memorised its training set. The data are tiny
10×10 grayscale pictures of faces drawn in code (no photographs): an 8×8 smile or frown at a
random spot, with random brightness and pixel noise (sd 0.35). 200 training pictures, 20 of
them with a flipped (wrong) label; 400 validation pictures, labelled correctly.

The model: 100 pixels → 256 ReLU → 256 ReLU → 1 logit (91,392 weights), binary cross-entropy,
plain mini-batch gradient descent (batch 20, step 0.05), 300 epochs, the same initial weights
in every run. Five runs, all computed below with seeded numpy when the scene loads (~3 s):

- plain: no regularization;
- weight decay: loss + λ·Σw² with λ = 0.01 (weights only, not biases); with plain gradient
  descent this is exactly "shrink every weight by 2ηλ·w each step";
- dropout (Srivastava et al., 2014): input pixels dropped with p = 0.2, hidden units with
  p = 0.5, in the common "inverted" form (survivors scaled by 1/(1−p) while training, nothing
  dropped at test time; the paper scales the weights by the keep rate at test time instead,
  which is equivalent on average);
- data augmentation: every epoch each training picture is flipped left-right (p 0.5), cropped
  up to one pixel off-centre (pad 1, crop 10×10) and given fresh noise (sd 0.2);
- too much weight decay: λ = 0.1 (the underfitting caveat).
Early stopping reads the plain run's validation curve: its lowest point, patience 20 epochs.

The asserts tie every number the narration quotes to these computations.

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

S, G = 10, 8  # picture and face sizes, pixels
SMILE = ["........", ".X....X.", ".X....X.", "........", "X......X", ".X....X.", "..XXXX..", "........"]
FROWN = SMILE[:4] + ["..XXXX..", ".X....X.", "X......X", "........"]
GLYPHS = [np.array([[c == "X" for c in row] for row in rows], float) for rows in (SMILE, FROWN)]
LABELS = ("smile", "frown")  # label 0, label 1
N_TRAIN, N_VAL, N_WRONG, NOISE = 200, 400, 20, 0.35


def faces(n, rng):
    """n pictures, smiles and frowns alternating: a face at a random spot, brightness 0.6–1, noise."""
    images, labels = np.zeros((n, S, S)), np.arange(n) % 2
    for i in range(n):
        r, c = rng.integers(0, S - G + 1, 2)
        images[i, r : r + G, c : c + G] = GLYPHS[labels[i]] * rng.uniform(0.6, 1.0)
    return images + rng.normal(0, NOISE, images.shape), labels.astype(float)


_rng = np.random.default_rng(26)
XTR, TRUE_TR = faces(N_TRAIN, _rng)
WRONG = np.sort(_rng.choice(N_TRAIN, N_WRONG, replace=False))
YTR = TRUE_TR.copy()
YTR[WRONG] = 1 - YTR[WRONG]  # twenty training labels flipped
XVA, YVA = faces(N_VAL, _rng)


def augment(images, rng):
    """Flip left-right (half the time), crop up to a pixel off-centre, add fresh noise."""
    out = images.copy()
    flip = rng.random(len(out)) < 0.5
    out[flip] = out[flip][:, :, ::-1]
    padded = np.pad(out, ((0, 0), (1, 1), (1, 1)))
    shifts = rng.integers(0, 3, (len(out), 2))
    out = np.stack([p[a : a + S, b : b + S] for p, (a, b) in zip(padded, shifts)])
    return out + rng.normal(0, 0.2, out.shape)


HIDDEN, EPOCHS, BATCH, RATE = (256, 256), 300, 20, 0.05
SIZES = [S * S, *HIDDEN, 1]
N_WEIGHTS = sum(a * b for a, b in zip(SIZES, SIZES[1:]))


def bce(z, y):
    return float(np.mean(np.logaddexp(0, z) - y * z))


def train(decay=0.0, drop_in=0.0, drop_hidden=0.0, augmented=False, seed=0):
    """Mini-batch gradient descent; logs losses and accuracies before training (epoch 0) and after each epoch."""
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(SIZES, SIZES[1:])]
    Ws[-1] *= np.sqrt(0.5)
    bs = [np.zeros(b) for b in SIZES[1:]]

    def logits(X):
        h = X
        for W, b in zip(Ws[:-1], bs[:-1]):
            h = np.maximum(h @ W + b, 0)
        return (h @ Ws[-1] + bs[-1])[:, 0]

    flat_tr, flat_va = XTR.reshape(N_TRAIN, -1), XVA.reshape(N_VAL, -1)
    log = {k: [] for k in ("train", "val", "val_acc", "train_acc", "sumsq")}

    def record():
        zt, zv = logits(flat_tr), logits(flat_va)
        log["train"].append(bce(zt, YTR))
        log["val"].append(bce(zv, YVA))
        log["val_acc"].append(float(np.mean((zv > 0) == (YVA == 1))))
        log["train_acc"].append(float(np.mean((zt > 0) == (YTR == 1))))
        log["sumsq"].append(float(sum((W**2).sum() for W in Ws)))

    record()
    for _ in range(EPOCHS):
        X_all = augment(XTR, rng).reshape(N_TRAIN, -1) if augmented else flat_tr
        order = rng.permutation(N_TRAIN)
        for s in range(0, N_TRAIN, BATCH):
            idx = order[s : s + BATCH]
            h, y = X_all[idx], YTR[idx]
            if drop_in:
                h = h * (rng.random(h.shape) >= drop_in) / (1 - drop_in)
            acts, gates = [h], []
            for W, b in zip(Ws[:-1], bs[:-1]):
                pre = h @ W + b
                keep = (rng.random(pre.shape) >= drop_hidden) / (1 - drop_hidden) if drop_hidden else 1.0
                h = np.maximum(pre, 0) * keep
                gates.append((pre > 0) * keep)
                acts.append(h)
            z = (h @ Ws[-1] + bs[-1])[:, 0]
            g = (1 / (1 + np.exp(-z)) - y)[:, None] / len(y)
            for k in range(len(Ws) - 1, -1, -1):
                gW, gb = acts[k].T @ g + 2 * decay * Ws[k], g.sum(0)  # the gradient of λΣw² is 2λw
                if k:
                    g = (g @ Ws[k].T) * gates[k - 1]
                Ws[k] -= RATE * gW
                bs[k] -= RATE * gb
        record()
    out = {k: np.array(v) for k, v in log.items()}
    out["memorised_wrong"] = int(np.sum((logits(flat_tr)[WRONG] > 0) == (YTR[WRONG] == 1)))
    return out


PLAIN = train()
DECAY = train(decay=0.01)
DROP = train(drop_in=0.2, drop_hidden=0.5)
AUG = train(augmented=True)
HEAVY = train(decay=0.1)
RUNS = (PLAIN, DECAY, DROP, AUG)
EP = np.arange(EPOCHS + 1)


def early_stop(val, patience):
    """The lowest point so far, and the epoch where `patience` epochs have passed without a new one."""
    best, since = 0, 0
    for e in range(1, len(val)):
        if val[e] < val[best]:
            best, since = e, 0
        else:
            since += 1
            if since == patience:
                return best, e
    return best, len(val) - 1


PATIENCE = 20
BEST, STOP = early_stop(PLAIN["val"], PATIENCE)
PLAIN_END, DECAY_END, DROP_END, AUG_END = (float(r["val"][-1]) for r in RUNS)
ES_KEPT = float(PLAIN["val"][BEST])
HEAVY_END = float(HEAVY["val"][-1])

# the narration's numbers
assert N_WEIGHTS == 91_392 and N_WEIGHTS > 90_000  # "over ninety thousand weights"
assert PLAIN["train"][-1] < 0.01  # "almost to zero"
assert PLAIN["train_acc"][-1] == 1.0 and PLAIN["memorised_wrong"] == N_WRONG == 20  # "every training picture right, even the twenty"
assert BEST == int(np.argmin(PLAIN["val"])) == 22 and STOP == 42  # "bottoms out early" (epoch 22 of 300); "twenty passes later"
assert round(ES_KEPT, 1) == 0.4 and round(PLAIN_END, 1) == 0.6  # "about zero point four", "zero point six"
assert round(DECAY_END, 1) == 0.4 and DECAY_END - DECAY["val"].min() < 0.03  # "stops climbing: ends near zero point four"
assert 950 < PLAIN["sumsq"][-1] < 1100 and 18 < DECAY["sumsq"][-1] < 25  # "about a thousand to about twenty"
assert round(DROP_END, 1) == 0.3  # "ends near zero point three"
assert round(AUG_END, 1) == 0.2 and AUG_END < min(DECAY_END, DROP_END)  # "about zero point two, the best so far"
assert AUG["train"][-1] > 0.1  # "the training loss stays well above zero"
assert max(DECAY_END, DROP_END, AUG_END, ES_KEPT) < PLAIN_END  # "every fix beats doing nothing"
assert HEAVY["train"][-1] > 0.68 and abs(HEAVY["val_acc"][-1] - 0.5) < 0.02  # "learned nothing: right only half the time"
assert max(max(r["train"].max(), r["val"].max()) for r in (*RUNS, HEAVY)) < 0.75  # fits the chart


def clarity(img, label):
    """How plainly a picture shows its face: brightest glyph match minus the rest, best offset."""
    g = GLYPHS[label]
    return max(img[r : r + G, c : c + G][g == 1].mean() - img[r : r + G, c : c + G][g == 0].mean() for r in range(3) for c in range(3))


_clear = {i: clarity(XTR[i], int(TRUE_TR[i])) for i in range(N_TRAIN)}
_ok = [i for i in sorted(_clear, key=_clear.get, reverse=True) if i not in WRONG]
_smiles, _frowns = [i for i in _ok if TRUE_TR[i] == 0][:4], [i for i in _ok if TRUE_TR[i] == 1][:3]
_bad = sorted(WRONG, key=lambda i: -_clear[i])
_bad = [next(i for i in _bad if TRUE_TR[i] == 0), next(i for i in _bad if TRUE_TR[i] == 1)]
SHOW = [_smiles[1], _frowns[0], _bad[0], _smiles[2], _frowns[1], _smiles[3], _frowns[2], _bad[1]]  # 8 real training pictures
DEMO = _smiles[0]  # the augmentation example: a clear smile, labelled correctly
assert TRUE_TR[DEMO] == 0 and DEMO not in WRONG and sum(i in WRONG for i in SHOW) == 2

_demo_rng = np.random.default_rng(7)
DEMO_IMG = XTR[DEMO]
CROPPED = np.pad(DEMO_IMG, 1)[1 : 1 + S, 0:S]  # one pixel off-centre (moved right)
NOISIER = DEMO_IMG + _demo_rng.normal(0, 0.2, DEMO_IMG.shape)
VERSIONS = list(augment(np.repeat(DEMO_IMG[None], 4, 0), _demo_rng))  # four epochs' random versions

# the Episode 5 callback: the same twenty points and degree-15 fit as that episode (its seed and recipe)
_r5 = np.random.default_rng(3297)
while True:
    _u = np.sort(_r5.uniform(0.03, 0.97, 20))
    if np.diff(_u).min() > 0.018:
        break
X5 = 2 * _u - 1
Y5 = -np.sin(np.pi * X5) + 0.3 * _r5.normal(0, 1, 20)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # degree 15 is poorly conditioned; still an exact least-squares fit
    FIT15 = np.polyfit(X5, Y5, 15)

# ---------------------------------------------------------------------------- helpers

BLUE_FILL = "#14213D"
PIX_DARK = "#0E1426"
DECAY_C, DROP_C, AUG_C, STOP_C = AMBER, TEAL, BLUE, GOOD
TRAIN_C, VAL_C = SKY, PINK


def variant(name, color, fill, stem):
    """An icon in another accent, like kit.recolored(), but saved as assets/ep26-<stem>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = f"ep26-{stem}-{color.lstrip('#').lower()}"
    svg = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{out}.svg"
    if not path.exists() or path.read_text() != svg:
        path.write_text(svg)
    return out


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def polyline(points, color, width=4.5, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


def face(img, size, color=EDGE, where=None):
    """A picture as S×S squares on a dark tile; where(r, c) → the (row, col) to draw pixel (r, c) at."""
    cell = size / S
    tile_ = m.RoundedRectangle(corner_radius=0.08, width=size + 0.18, height=size + 0.18)
    tile_.set_stroke(color, width=3).set_fill(PIX_DARK, opacity=1)
    pixels = m.VGroup()
    for r in range(S):
        for c in range(S):
            rr, cc = where(r, c) if where else (r, c)
            shade = mix(PIX_DARK, INK, float(np.clip((img[r, c] - 0.12) / 0.78, 0, 1)))
            sq = m.Square(side_length=cell * 0.88).set_stroke(width=0).set_fill(shade, opacity=1)
            pixels.add(sq.move_to([(cc - (S - 1) / 2) * cell, ((S - 1) / 2 - rr) * cell, 0]))
    return m.VGroup(tile_, pixels)


def blended(k):
    """The demo picture's look at moment k of its epoch-by-epoch cycle (holds, then a quick change)."""
    seq = [DEMO_IMG, *VERSIONS, DEMO_IMG]
    i = int(np.clip(np.floor(k), 0, len(seq) - 2))
    a = float(np.clip((k - i - 0.55) / 0.45, 0, 1))
    return (1 - a) * seq[i] + a * seq[i + 1]


def two_part(*pieces):
    """Texts in a row on one baseline: pieces are (string, size, color, font)."""
    row = m.VGroup(*[(code if font == "mono" else text)(s, size, color) for s, size, color, font in pieces])
    return on_baseline(row.arrange(m.RIGHT, buff=0.2))


def end_readout(value, color, label="final validation loss:", ref=PLAIN_END, ref_label="no fix:"):
    mine = two_part((label, 22, MUTED, "sans"), (f"{value:.2f}", 26, color, "mono"))
    other = two_part((ref_label, 22, MUTED, "sans"), (f"{ref:.2f}", 26, MUTED, "mono"))
    other.shift((mine[0][0].get_bottom()[1] - other[0][0].get_bottom()[1]) * m.UP)
    return m.VGroup(mine, other).arrange(m.RIGHT, buff=0.55, aligned_edge=m.DOWN)


def legend(center):
    items = m.VGroup(
        *[
            m.VGroup(m.Line(m.ORIGIN, 0.45 * m.RIGHT).set_stroke(c, width=5), text(s, 22, c, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15)
            for c, s in ((TRAIN_C, "training loss"), (VAL_C, "validation loss"))
        ]
    )
    return items.arrange(m.RIGHT, buff=0.5).move_to(center)


class Chart:
    """Loss (0 … 0.78) against epochs, quiet axes on the right half of the stage."""

    def __init__(self, x_max=340, x_step=100, ticks=(0, 100, 200, 300), center=(3.65, -0.3, 0), width=5.6, height=3.85):
        self.ax = axes([0, x_max, x_step], [0, 0.78, 0.2], width, height).move_to(center)
        ax = self.ax
        xt = m.VGroup(*[text(str(v), 20, MUTED).next_to(ax.c2p(v, 0), m.DOWN, buff=0.2) for v in ticks])
        yt = m.VGroup(*[text(f"{v:.1f}", 20, MUTED).next_to(ax.c2p(0, v), m.LEFT, buff=0.2) for v in (0.2, 0.4, 0.6)])
        xl = text("epoch", 22, MUTED).move_to([ax.c2p(ticks[-1] / 2, 0)[0], xt.get_bottom()[1] - 0.28, 0])
        yl = text("loss", 22, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.18)
        self.frame = m.VGroup(ax, xt, yt, xl, yl)

    def p(self, e, v):
        return self.ax.c2p(e, v)

    def curve(self, values, color, lo=0, hi=EPOCHS, width=4.5):
        return polyline([self.p(e, values[e]) for e in range(lo, hi + 1)], color, width)

    def pair(self, run, hi=EPOCHS):
        return m.VGroup(self.curve(run["train"], TRAIN_C, hi=hi), self.curve(run["val"], VAL_C, hi=hi))


class Plot5:
    """Episode 5's picture: inputs x in [-1, 1], outputs in [-2, 2], curves cut at the box."""

    def __init__(self, width, height, center, y_max=2.0):
        self.ax = axes([0, 2.1, 0.25], [-y_max, y_max, 1], width, height).move_to(center)
        self.y_max = y_max

    def p(self, x, y):
        return self.ax.c2p(x + 1, y)

    def graph(self, fn, x0, x1, color, width=4.5, n=600):
        xs = np.linspace(x0, x1, n)
        ys = fn(xs)
        pieces, run = [], []
        for x, y in zip(xs, ys):
            if abs(y) <= self.y_max:
                run.append(self.p(x, y))
            elif run:
                pieces.append(run)
                run = []
        if run:
            pieces.append(run)
        return m.VGroup(*[polyline(r, color, width) for r in pieces if len(r) > 1])


# the dropout picture: a schematic network (7 of 256 hidden units drawn), illustrative masks
NET_LAYERS = (5, 6, 6, 1)
NET_XS = (-6.0, -4.45, -2.9, -1.35)
NODE_R = 0.21
NET_P = [[np.array([x, 0.25 + ((n - 1) / 2 - i) * 0.62, 0]) for i in range(n)] for n, x in zip(NET_LAYERS, NET_XS)]
ALL_ON = [np.ones(n, bool) for n in NET_LAYERS]


def net_masks(n, seed=2014):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        inputs = np.ones(5, bool)
        inputs[rng.integers(5)] = False
        hidden = []
        for _ in range(2):
            h = np.zeros(6, bool)
            h[rng.choice(6, 3, replace=False)] = True
            hidden.append(h)
        mask = [inputs, *hidden, np.ones(1, bool)]
        if not out or any((a != b).any() for a, b in zip(mask[1:3], out[-1][1:3])):
            out.append(mask)
    return out


MASKS = net_masks(4)


def network(on, boost=False):
    """The schematic network; dropped units dim; boost draws the hidden survivors larger (scaled up)."""
    edges, nodes = m.VGroup(), m.VGroup()
    for layer in range(3):
        for i, a in enumerate(NET_P[layer]):
            for j, b in enumerate(NET_P[layer + 1]):
                live = on[layer][i] and on[layer + 1][j]
                edges.add(m.Line(a, b, buff=NODE_R).set_stroke(DROP_C if live else FAINT, width=2.2 if live else 1.5, opacity=0.5 if live else 0.16))
    for layer, points in enumerate(NET_P):
        for i, p in enumerate(points):
            alive = on[layer][i]
            big = boost and alive and layer in (1, 2)
            dot = m.Circle(radius=0.28 if big else NODE_R).move_to(p)
            fill = mix(TEAL_FILL, DROP_C, 0.55) if big else (TEAL_FILL if alive else BG)
            nodes.add(dot.set_fill(fill, opacity=1).set_stroke(DROP_C if alive else FAINT, width=3.5 if big else 3))
    return m.VGroup(edges, nodes)


# ---------------------------------------------------------------------------- the episode


class Ep26Regularization(Episode):
    number = 26

    def lesson(self):
        self.hook()
        self.experiment()
        self.problem()
        self.weight_decay()
        self.dropout()
        self.augmentation()
        self.early_stopping()
        self.big_picture()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes="Callback to Episode 5: the same twenty noisy points and degree-15 polynomial fit (recomputed here from that "
            "episode's seed). Four practical fixes, by where they act: the loss, the network, the data, the stopping time.",
        )
        plot = Plot5(7.0, 3.6, [0, -0.15, 0])
        dots = m.VGroup(*[m.Dot(plot.p(x, y), radius=0.085, color=SKY) for x, y in zip(X5, Y5)]).set_z_index(3)
        wiggle = plot.graph(lambda xs: np.polyval(FIT15, xs), X5[0], X5[-1], BAD)
        tag = pill("EPISODE 5 · OVERFITTING", BAD, BAD_FILL, 22).move_to([0, 2.45, 0])
        caption = text("too much freedom: it memorised the noise", 24, MUTED).move_to([0, -2.65, 0])

        title = text("Four ways to stop memorising", 40, INK, weight=m.BOLD).move_to([0, 2.4, 0])
        specs = (
            (variant("ep05-weight", DECAY_C, AMBER_FILL, "weight"), "Weight decay", "changes the loss", DECAY_C),
            ("ep26-dropout", "Dropout", "changes the network", DROP_C),
            ("ep26-augment", "Data augmentation", "changes the data", AUG_C),
            ("ep26-stopwatch", "Early stopping", "changes when to stop", STOP_C),
        )
        cards, icons, subs = m.VGroup(), [], []
        for k, (name, label, sub, color) in enumerate(specs):
            box = card(3.2, 3.3, color, CARD, stroke=2).move_to([-5.175 + 3.45 * k, -0.25, 0])
            pic = icon(name, 1.15).move_to(box.get_top() + 1.0 * m.DOWN)
            words = text(label, 25, INK, weight=m.BOLD).move_to(box.get_top() + 2.05 * m.DOWN)
            note = text(sub, 22, color, weight=m.SEMIBOLD).move_to(box.get_top() + 2.65 * m.DOWN)
            cards.add(m.VGroup(box, pic, words))
            icons.append(pic)
            subs.append(note)

        self.say(
            "So... [remember episode five]? [A curve with too much freedom memorised the noise in its training points.]",
            m.AnimationGroup(m.FadeIn(tag, shift=0.2 * m.DOWN), m.Create(plot.ax), m.FadeIn(dots)),
            m.AnimationGroup(m.Create(wiggle, run_time=2.4), m.FadeIn(caption)),
        )
        self.say(
            "[Today, a neural network does the same thing], [and we'll fix it in four practical ways.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in (tag, plot.ax, dots, wiggle, caption)], m.FadeIn(title, shift=0.2 * m.UP)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.3 * m.UP) for c in cards], lag_ratio=0.2),
        )
        self.say(
            "[Change the loss], [change the network], [change the data], [or change when you stop.]",
            *[
                m.AnimationGroup(m.FadeIn(s, shift=0.15 * m.UP), m.Indicate(i, color=c, scale_factor=1.12))
                for s, i, c in zip(subs, icons, (DECAY_C, DROP_C, AUG_C, STOP_C))
            ],
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the experiment

    def experiment(self):
        self.next_section(
            "The experiment",
            notes="Illustrative data, computed: 10×10 faces drawn in code (smile or frown at a random spot, noise sd 0.35). "
            "200 training pictures, 20 with a flipped label; 400 validation pictures. Network 100 → 256 → 256 → 1 "
            "(91,392 weights), mini-batch gradient descent, batch 20, step 0.05, 300 epochs. The eight pictures shown are "
            "real training pictures (the clearest ones, two of them mislabelled).",
        )
        xs, ys = (-5.85, -4.35, -2.85, -1.35), (1.4, -0.55)
        pics, names = m.VGroup(), m.VGroup()
        for k, i in enumerate(SHOW):
            x, y = xs[k % 4], ys[k // 4]
            pics.add(face(XTR[i], 1.1, EDGE).move_to([x, y, 0]))
            names.add(text(LABELS[int(YTR[i])], 22, MUTED, weight=m.SEMIBOLD).move_to([x, y - 0.9, 0]))
        wrong = [k for k, i in enumerate(SHOW) if i in WRONG]
        self.pics, self.names, self.wrong = pics, names, wrong

        box = card(6.0, 5.2, EDGE, CARD).move_to([3.55, -0.3, 0])
        left = box.get_left()[0] + 0.45
        count = text("200 training pictures", 28, INK, weight=m.BOLD)
        count.move_to([0, 1.8, 0]).align_to([left, 0, 0], m.LEFT)
        bad = text("20 of them with the wrong label", 22, BAD, weight=m.SEMIBOLD)
        bad.move_to([0, 1.3, 0]).align_to([left, 0, 0], m.LEFT)
        chips = m.VGroup(*[chip(s, VIOLET, VIOLET_FILL, width=0.9, height=0.6, size=26).move_to([x, 0.35, 0]) for s, x in zip(("100", "256", "256", "1"), (1.45, 2.85, 4.25, 5.65))])
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(chips, chips[1:])])
        layer_names = m.VGroup(*[text(s, 20, MUTED).move_to([c.get_x(), -0.2, 0]) for s, c in zip(("pixels", "hidden", "hidden", "output"), chips)])
        weights = pill(f"{N_WEIGHTS:,} weights", VIOLET, VIOLET_FILL, 24).move_to([box.get_x(), -0.9, 0])
        epochs = text(f"{EPOCHS} epochs of gradient descent", 22, INK).move_to([0, -1.7, 0]).align_to([left, 0, 0], m.LEFT)
        held = text(f"{N_VAL} validation pictures, never trained on", 22, VAL_C).move_to([0, -2.3, 0]).align_to([left, 0, 0], m.LEFT)
        self.model_card = m.VGroup(box, count, bad, chips, links, layer_names, weights, epochs, held)

        self.say(
            "[Here's our running example]: [tiny pictures of faces, ten by ten pixels], [each one smiling or frowning.]",
            self.new_header("THE EXPERIMENT", "Smiles, frowns and a few wrong labels"),
            m.LaggedStart(*[m.FadeIn(p, scale=0.85) for p in pics], lag_ratio=0.12),
            m.LaggedStart(*[m.FadeIn(n, shift=0.1 * m.UP) for n in names], lag_ratio=0.08),
        )
        self.say(
            "[We have two hundred pictures to train on], [and twenty of them carry the wrong label], [as real data often does.]",
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(count, shift=0.15 * m.UP)),
            m.AnimationGroup(
                m.FadeIn(bad, shift=0.15 * m.UP),
                *[pics[k][0].animate.set_stroke(BAD, width=4) for k in wrong],
                *[names[k].animate.set_color(BAD) for k in wrong],
            ),
            m.AnimationGroup(*[m.Wiggle(m.VGroup(pics[k], names[k]), scale_value=1.06) for k in wrong]),
        )
        self.say(
            "[The model is a network with two hidden layers], [and over ninety thousand weights.]",
            m.LaggedStart(*[m.FadeIn(m.VGroup(c, n), shift=0.15 * m.RIGHT) for c, n in zip(chips, layer_names)], *[m.GrowArrow(a) for a in links], lag_ratio=0.15),
            m.FadeIn(weights, scale=1.2),
        )
        self.say(
            "[We train it for three hundred epochs], [and check it on four hundred other pictures]: [the validation set.]",
            m.FadeIn(epochs, shift=0.15 * m.UP),
            m.FadeIn(held, shift=0.15 * m.UP),
            m.Circumscribe(held, color=VAL_C, buff=0.08),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the problem

    def problem(self):
        self.next_section(
            "The problem",
            notes=f"Computed: the plain network. Training loss → {PLAIN['train'][-1]:.3f}, training accuracy 100% including all "
            f"20 wrong labels. Validation loss lowest {ES_KEPT:.3f} at epoch {BEST}, {PLAIN_END:.3f} at epoch 300 "
            f"(validation accuracy {PLAIN['val_acc'][-1]:.1%}).",
        )
        chart = Chart()
        self.chart = chart
        key = legend([3.65, 2.3, 0])
        train_c = chart.curve(PLAIN["train"], TRAIN_C)
        val_a, val_b = chart.curve(PLAIN["val"], VAL_C, hi=BEST), chart.curve(PLAIN["val"], VAL_C, lo=BEST)
        self.plain = m.VGroup(train_c, val_a, val_b)
        low = m.Dot(chart.p(BEST, ES_KEPT), radius=0.09, color=VAL_C).set_z_index(3)
        low_label = text(f"lowest: epoch {BEST}", 22, VAL_C, weight=m.SEMIBOLD)
        low_label.move_to(chart.p(BEST, 0.3)).align_to(chart.p(BEST + 12, 0), m.LEFT)
        top = [chart.p(e, PLAIN["val"][e]) for e in range(BEST, EPOCHS + 1)]
        bottom = [chart.p(e, PLAIN["train"][e]) for e in range(EPOCHS, BEST - 1, -1)]
        gap = m.Polygon(*top, *bottom).set_stroke(width=0).set_fill(BAD, opacity=0.14).set_z_index(-1)
        gap_label = text("the gap: memorisation", 24, BAD, weight=m.BOLD).move_to(chart.p(232, 0.13))

        acc = two_part(("training accuracy", 24, MUTED, "sans"), ("100%", 28, TRAIN_C, "mono")).move_to([-3.6, -2.2, 0])
        learned = two_part(("wrong labels learned", 24, MUTED, "sans"), (f"{PLAIN['memorised_wrong']} of {N_WRONG}", 28, BAD, "mono")).move_to([-3.6, -2.85, 0])
        self.problem_left = m.VGroup(acc, learned, self.pics, self.names)
        self.problem_marks = m.VGroup(low, low_label, gap, gap_label)
        self.legend = key

        self.say(
            "[Watch the training loss as the network trains.] [It falls, and keeps falling, almost to zero.]",
            m.AnimationGroup(self.new_header("THE PROBLEM", "It memorised its training set"), m.FadeOut(self.model_card), m.Create(chart.frame), m.FadeIn(key)),
            m.Create(train_c, run_time=2.6),
        )
        self.say(
            "[By the end, the model gets every training picture right], [even the twenty with wrong labels.]",
            m.FadeIn(acc, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(learned, shift=0.15 * m.UP), *[m.Indicate(m.VGroup(self.pics[k], self.names[k]), color=BAD, scale_factor=1.08) for k in self.wrong]),
        )
        self.say(
            "[But the validation loss tells another story.] [It bottoms out early], [then climbs for the rest of training.]",
            m.Create(val_a, run_time=1.2),
            m.AnimationGroup(m.FadeIn(low, scale=0.4), m.FadeIn(low_label, shift=0.15 * m.UP)),
            m.Create(val_b, run_time=2.4),
        )
        self.say(
            "[That widening gap is memorisation.] [The network is learning its training set by heart, noise and mistakes included.]",
            m.AnimationGroup(m.FadeIn(gap), m.FadeIn(gap_label, shift=0.15 * m.UP)),
            m.LaggedStart(*[m.Indicate(p, color=INK, scale_factor=1.08) for p in self.pics], lag_ratio=0.1),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ fix 1: weight decay

    def weight_decay(self):
        self.next_section(
            "Fix 1: weight decay",
            notes=f"Computed: the same network with loss + λΣw², λ = 0.01. Sum of squared weights at the end: "
            f"{PLAIN['sumsq'][-1]:,.0f} (no fix) vs {DECAY['sumsq'][-1]:.1f}; validation loss at the end {DECAY_END:.3f} "
            f"vs {PLAIN_END:.3f}. With plain gradient descent the L2 penalty and weight decay are the same; with Adam they "
            "differ, and AdamW applies the decay to the weights directly (Loshchilov & Hutter, 2019, 'Decoupled Weight Decay "
            "Regularization', ICLR).",
        )
        chart = self.chart
        formula = m.MathTex(r"\text{loss}", "=", r"\text{error}", "+", r"\lambda", r"\sum w^2", font_size=46, color=INK)
        for part, c in zip(formula, (INK, INK, TRAIN_C, INK, DECAY_C, DECAY_C)):
            part.set_color(c)
        formula.move_to([-3.6, 1.75, 0])
        rows = m.VGroup()
        for sym, desc, c in (
            (r"\text{error}", "how wrong it is on the training data", TRAIN_C),
            (r"\sum w^2", "every weight, squared, all added up", DECAY_C),
            (r"\lambda", "the price: how strong the penalty is", DECAY_C),
        ):
            rows.add(m.VGroup(m.MathTex(sym, font_size=34, color=c), text(desc, 22, MUTED)))
        for k, row in enumerate(rows):
            row[0].move_to([-5.75, 0.75 - 0.6 * k, 0])
            row[1].move_to([0, 0.75 - 0.6 * k, 0]).align_to([-4.95, 0, 0], m.LEFT)

        scale = 3.3 / 1100
        x0 = -4.65
        bar_title = text("sum of squared weights", 22, MUTED, weight=m.SEMIBOLD).move_to([0, -1.4, 0]).align_to([-6.45, 0, 0], m.LEFT)
        plain_name = text("no fix", 22, MUTED, weight=m.SEMIBOLD).move_to([0, -1.95, 0]).align_to([x0 - 0.3, 0, 0], m.RIGHT)
        decay_name = m.MathTex(r"\lambda = 0.01", font_size=30, color=DECAY_C).move_to([0, -2.55, 0]).align_to([x0 - 0.3, 0, 0], m.RIGHT)
        w_plain = PLAIN["sumsq"][-1]
        plain_bar = m.Rectangle(width=w_plain * scale, height=0.32).set_stroke(width=0).set_fill(MUTED, opacity=0.7)
        plain_bar.move_to([x0, -1.95, 0], aligned_edge=m.LEFT)
        plain_num = code(f"{w_plain:,.0f}", 24, MUTED).next_to(plain_bar, m.RIGHT, buff=0.2)
        epoch = m.ValueTracker(0.0)

        def sumsq():
            return float(np.interp(epoch.get_value(), EP, DECAY["sumsq"]))

        decay_bar = m.always_redraw(
            lambda: m.Rectangle(width=max(sumsq() * scale, 0.01), height=0.32).set_stroke(width=0).set_fill(DECAY_C, opacity=0.85).move_to([x0, -2.55, 0], aligned_edge=m.LEFT)
        )
        decay_num = m.DecimalNumber(sumsq(), num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=24, color=DECAY_C)
        decay_num.add_updater(lambda d: d.set_value(sumsq()).move_to([x0 + max(sumsq() * scale, 0.01) + 0.2 + d.width / 2, -2.55, 0]))
        bars = m.VGroup(bar_title, plain_name, decay_name, plain_bar, plain_num)

        adam = text("with Adam: a penalty in the loss is not decay", 22, INK)
        adamw = m.VGroup(text("AdamW", 26, DECAY_C, weight=m.BOLD), text("shrinks the weights directly", 22, INK)).arrange(m.RIGHT, buff=0.25)
        on_baseline(adamw)
        cite = text("Loshchilov & Hutter, 2019", 20, MUTED)
        note = m.VGroup(adam, adamw, cite).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([0, -2.25, 0]).align_to([-6.45, 0, 0], m.LEFT)

        curves = chart.pair(DECAY)
        readout = end_readout(DECAY_END, DECAY_C).move_to([3.65, -3.45, 0])
        self.decay_stuff = m.VGroup(formula, rows, note, curves, readout)

        self.say(
            "[Fix one changes the loss.] [We add a penalty for big weights.]",
            m.AnimationGroup(
                self.new_header("FIX 1 · THE LOSS", "Weight decay", DECAY_C, AMBER_FILL),
                m.FadeOut(self.problem_left),
                m.FadeOut(self.problem_marks),
                self.plain.animate.set_stroke(opacity=0.22),
            ),
            m.Write(formula, run_time=1.4),
        )
        self.say(
            "[The first term is the usual error on the training data.] [The second adds up every weight, squared], [and lambda sets its price.]",
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.15 * m.UP), m.Indicate(formula[2], color=TRAIN_C, scale_factor=1.15)),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.15 * m.UP), m.Indicate(formula[5], color=DECAY_C, scale_factor=1.15)),
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.15 * m.UP), m.Indicate(formula[4], color=DECAY_C, scale_factor=1.3)),
        )
        self.say(
            "[Its gradient pulls every weight toward zero at each step]: [that's why it's called weight decay.]",
            m.AnimationGroup(m.FadeIn(bars, shift=0.15 * m.UP), m.FadeIn(decay_bar), m.FadeIn(decay_num)),
            m.Indicate(self.head[1], color=DECAY_C, scale_factor=1.08),
        )
        self.say(
            "[With lambda at one hundredth], [the sum of squared weights shrinks from about a thousand to about twenty.]",
            m.Indicate(decay_name, color=DECAY_C, scale_factor=1.2),
            epoch.animate(run_time=3.2, rate_func=m.linear).set_value(EPOCHS),
        )
        self.stop(decay_bar, decay_num)
        self.say(
            "[And the validation loss stops climbing]: [it ends near zero point four, instead of about zero point six.]",
            m.AnimationGroup(m.Create(curves[0], run_time=2.4), m.Create(curves[1], run_time=2.4)),
            m.FadeIn(readout, shift=0.15 * m.UP),
        )
        self.say(
            "[One catch]: [with Adam, a penalty in the loss is not the same as decay.] [Adam W decouples them], [and shrinks the weights directly.]",
            m.FadeOut(m.VGroup(bars, decay_bar, decay_num)),
            m.FadeIn(adam, shift=0.15 * m.UP),
            m.FadeIn(adamw[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(adamw[1], shift=0.15 * m.UP), m.FadeIn(cite)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ fix 2: dropout

    def dropout(self):
        self.next_section(
            "Fix 2: dropout",
            notes=f"Dropout (Srivastava et al., 2014, JMLR). Computed: inputs dropped with p = 0.2, hidden units with p = 0.5, "
            f"survivors scaled by 1/(1−p) while training, nothing dropped at test time; validation loss at the end "
            f"{DROP_END:.3f} (lowest {DROP['val'].min():.3f}). The network drawing is a schematic (6 of 256 hidden units per "
            "layer) and its masks are illustrative random draws.",
        )
        chart = self.chart
        net = network(ALL_ON)
        cite = text("Srivastava et al., 2014", 20, MUTED).move_to([-3.65, 2.35, 0])
        low = NET_P[1][-1][1] - NODE_R
        names = m.VGroup(*[text(s, 22, MUTED, weight=m.SEMIBOLD).move_to([x, low - 0.4, 0]) for s, x in zip(("pixels", "hidden", "hidden", "output"), NET_XS)])
        rates = m.VGroup(*[text(s, 20, DROP_C).move_to([x, low - 0.75, 0]) for s, x in zip(("drop 20%", "drop 50%", "drop 50%"), NET_XS)])
        mode_train = pill("TRAINING · random units off", DROP_C, TEAL_FILL, 22).move_to([-3.65, -2.95, 0])
        mode_test = pill("TEST · every unit on", DROP_C, TEAL_FILL, 22).move_to(mode_train)
        scaled = text("survivors × 2: the same signal on average", 22, MUTED).move_to([-3.65, -3.55, 0])
        original = text("(the original paper scales the weights at test time instead)", 20, MUTED).move_to(scaled)
        curves = chart.pair(DROP)
        readout = end_readout(DROP_END, DROP_C).move_to([3.65, -3.45, 0])

        self.say(
            "[Fix two changes the network itself.] [It's called dropout.]",
            m.AnimationGroup(
                self.new_header("FIX 2 · THE NETWORK", "Dropout", DROP_C, TEAL_FILL),
                m.FadeOut(self.decay_stuff),
                m.FadeIn(net),
                m.FadeIn(names),
            ),
            m.AnimationGroup(m.FadeIn(cite), m.Indicate(self.head[1], color=DROP_C, scale_factor=1.08)),
        )
        self.say(
            "[At every training step, each hidden unit is switched off at random], [with a probability of one half.]",
            m.Transform(net, network(MASKS[0])),
            m.AnimationGroup(m.FadeIn(rates, shift=0.1 * m.UP), m.FadeIn(mode_train, shift=0.15 * m.UP)),
        )
        self.say(
            "[Each step drops a different half], [so no unit can count on any other being there.]",
            m.Succession(m.Transform(net, network(MASKS[1]), run_time=0.6), m.Transform(net, network(MASKS[2]), run_time=0.6)),
            m.Transform(net, network(MASKS[3]), run_time=0.8),
        )
        self.say(
            "[The units that survive are scaled up by two], [so the signal keeps the same size on average.]",
            m.Transform(net, network(MASKS[3], boost=True)),
            m.FadeIn(scaled, shift=0.15 * m.UP),
        )
        self.say(
            "[At test time, nothing is dropped]: [every unit is on.]",
            m.AnimationGroup(m.FadeTransform(scaled, original), m.FadeOut(rates), m.Transform(net, network(ALL_ON)), m.FadeTransform(mode_train, mode_test)),
            m.Indicate(net[1], color=DROP_C, scale_factor=1.04),
        )
        self.say(
            "[On our faces], [the validation loss now ends near zero point three.]",
            m.AnimationGroup(m.Create(curves[0], run_time=2.0), m.Create(curves[1], run_time=2.0)),
            m.FadeIn(readout, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.drop_stuff = m.VGroup(net, cite, names, mode_test, original, curves, readout)

    # ------------------------------------------------------------------ fix 3: data augmentation

    def augmentation(self):
        self.next_section(
            "Fix 3: data augmentation",
            notes=f"Computed: every epoch each training picture is flipped left-right (half the time), cropped up to one pixel "
            f"off-centre and given fresh noise (sd 0.2). Validation loss at the end {AUG_END:.3f}; training loss {AUG['train'][-1]:.3f}. "
            "The flip, crop and noise examples and the four 'epochs' are real outputs of the same augment() on one training "
            "picture. Upside-down flips are not used: for faces they change what the mouth means.",
        )
        chart = self.chart
        home = np.array([-3.6, 1.15, 0])
        original = face(DEMO_IMG, 1.8, AUG_C).move_to(home)
        caption = text("a training picture: smile", 22, MUTED).move_to([-3.6, -0.1, 0])
        every = text("a new random version every epoch", 22, MUTED).move_to(caption)
        slots = (-5.95, -4.4, -2.85, -1.3)
        flipped = face(DEMO_IMG, 1.1, AUG_C, where=lambda r, c: (r, S - 1 - c)).move_to([slots[0], -1.25, 0])
        cropped = face(CROPPED, 1.1, AUG_C).move_to([slots[1], -1.25, 0])
        noisier = face(NOISIER, 1.1, AUG_C).move_to([slots[2], -1.25, 0])
        upside = face(DEMO_IMG, 1.1, BAD, where=lambda r, c: (S - 1 - r, c)).move_to([slots[3], -1.25, 0])

        def tag(name, verdict, color, x):
            return m.VGroup(text(name, 22, INK, weight=m.SEMIBOLD).move_to([x, -2.25, 0]), text(verdict, 22, color, weight=m.SEMIBOLD).move_to([x, -2.6, 0]))

        tags = [tag("flip", "✓ smile", AUG_C, slots[0]), tag("crop", "✓ smile", AUG_C, slots[1]), tag("noise", "✓ smile", AUG_C, slots[2]), tag("upside down", "✗ unsafe", BAD, slots[3])]
        rule = pill("only changes that keep the label true", AUG_C, BLUE_FILL, 22).move_to([-3.6, -3.3, 0])
        cycle = m.ValueTracker(0.0)
        live = m.always_redraw(lambda: face(blended(cycle.get_value()), 1.8, AUG_C).move_to(home))
        curves = chart.pair(AUG)
        readout = end_readout(AUG_END, AUG_C).move_to([3.65, -3.45, 0])

        self.say(
            "[Fix three changes the data.] [We can't easily collect more faces], [but we can make new ones from the old.]",
            m.AnimationGroup(self.new_header("FIX 3 · THE DATA", "Data augmentation", AUG_C, BLUE_FILL), m.FadeOut(self.drop_stuff)),
            m.AnimationGroup(m.FadeIn(original, scale=0.9), m.FadeIn(caption)),
            m.Indicate(original, color=AUG_C, scale_factor=1.06),
        )
        self.say(
            "[Flip a face sideways]: [still a smile.] [Crop it a pixel off-centre]: [still a smile.] [Add some noise]: [still a smile.]",
            m.ReplacementTransform(original.copy(), flipped, run_time=1.0),
            m.FadeIn(tags[0], shift=0.1 * m.UP),
            m.ReplacementTransform(original.copy(), cropped, run_time=1.0),
            m.FadeIn(tags[1], shift=0.1 * m.UP),
            m.ReplacementTransform(original.copy(), noisier, run_time=1.0),
            m.FadeIn(tags[2], shift=0.1 * m.UP),
        )
        self.remove(original)
        self.add(live)
        self.say(
            "[On every pass through the data, each picture arrives in a fresh random version], [so there's no fixed set to memorise.]",
            m.AnimationGroup(m.FadeTransform(caption, every), cycle.animate(run_time=3.0, rate_func=m.linear).set_value(3.0)),
            cycle.animate(run_time=1.8, rate_func=m.linear).set_value(5.0),
        )
        self.stop(live)
        self.say(
            "[The validation loss falls to about zero point two], [the best so far], [and the training loss stays well above zero.]",
            m.AnimationGroup(m.Create(curves[0], run_time=2.4), m.Create(curves[1], run_time=2.4)),
            m.FadeIn(readout, shift=0.15 * m.UP),
            m.ShowPassingFlash(curves[0].copy().set_stroke(INK, width=9), time_width=0.5, run_time=1.4),
        )
        source = face(DEMO_IMG, 1.8, AUG_C).move_to(home)
        self.say(
            "[But only use changes that keep the label true.] [Upside down, the mouth curves the other way.]",
            m.FadeIn(rule, shift=0.15 * m.UP),
            m.AnimationGroup(m.ReplacementTransform(source, upside, run_time=1.2), m.FadeIn(tags[3], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.aug_stuff = m.VGroup(live, every, flipped, cropped, noisier, upside, *tags, rule, curves, readout)

    # ------------------------------------------------------------------ fix 4: early stopping

    def early_stopping(self):
        self.next_section(
            "Fix 4: early stopping",
            notes=f"Computed from the plain run: lowest validation loss {ES_KEPT:.3f} at epoch {BEST}; with a patience of "
            f"{PATIENCE} epochs, training stops at epoch {STOP} and the weights from epoch {BEST} are kept "
            f"(vs {PLAIN_END:.3f} after 300 epochs). The chart zooms to epochs 0–100.",
        )
        old = self.chart
        chart = Chart(x_max=112, x_step=25, ticks=(0, 25, 50, 75, 100))
        hi = 100
        curves = m.VGroup(chart.curve(PLAIN["train"], TRAIN_C, hi=hi), chart.curve(PLAIN["val"], VAL_C, hi=hi))
        watch = icon("ep26-stopwatch", 1.15).move_to([-5.85, 1.5, 0])
        title = text(f"patience: {PATIENCE} epochs", 28, STOP_C, weight=m.BOLD)
        sub = text("stop when validation loss stops improving", 20, MUTED)
        head = m.VGroup(title, sub).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([0, 1.5, 0]).align_to([-5.05, 0, 0], m.LEFT)

        t = m.ValueTracker(0.0)

        def now():
            return float(t.get_value())

        def best():
            return int(np.argmin(PLAIN["val"][: int(np.floor(now() + 1e-6)) + 1]))

        def waited():
            return int(np.floor(now() + 1e-6)) - best()

        right = -0.9
        labels = ("epoch", "best epoch so far", "its validation loss", "epochs without improvement")
        ys = (0.35, -0.3, -0.95, -1.6)
        names = m.VGroup(*[text(s, 24, MUTED).move_to([0, y, 0]).align_to([-6.45, 0, 0], m.LEFT) for s, y in zip(labels, ys)])

        def number(fn, color, places=0, y=0.0):
            d = m.DecimalNumber(fn(), num_decimal_places=places, mob_class=Mono, font_size=28, color=color)
            d.add_updater(lambda d: d.set_value(fn()).move_to([right - d.width / 2, y, 0]))
            return d.move_to([right - d.width / 2, y, 0])

        values = m.VGroup(
            number(lambda: int(np.floor(now() + 1e-6)), INK, y=ys[0]),
            number(best, STOP_C, y=ys[1]),
            number(lambda: float(PLAIN["val"][best()]), VAL_C, places=2, y=ys[2]),
            number(waited, INK, y=ys[3]),
        )
        track = m.RoundedRectangle(corner_radius=0.08, width=5.55, height=0.18).set_stroke(width=0).set_fill(EDGE, opacity=1)
        track.move_to([0, -2.15, 0]).align_to([-6.45, 0, 0], m.LEFT)
        fill = m.always_redraw(
            lambda: m.RoundedRectangle(corner_radius=0.08, width=max(5.55 * waited() / PATIENCE, 0.18), height=0.18)
            .set_stroke(width=0)
            .set_fill(STOP_C, opacity=1 if waited() else 0)
            .move_to(track, aligned_edge=m.LEFT)
        )
        result = text(f"stop at epoch {STOP} · keep the weights from epoch {BEST}", 22, INK, t2c={f"epoch {BEST}": STOP_C, f"epoch {STOP}": STOP_C})
        result.move_to([0, -2.85, 0]).align_to([-6.45, 0, 0], m.LEFT)

        def overlay():
            x, b = now(), best()
            band = m.Polygon(chart.p(b, 0), chart.p(max(x, b + 0.05), 0), chart.p(max(x, b + 0.05), 0.72), chart.p(b, 0.72))
            band.set_stroke(width=0).set_fill(STOP_C, opacity=0.1 if x > b + 0.5 else 0.0)
            line = m.Line(chart.p(x, 0), chart.p(x, 0.72)).set_stroke(MUTED, width=2, opacity=0.6)
            here = m.Dot(chart.p(x, float(np.interp(x, EP, PLAIN["val"]))), radius=0.07, color=VAL_C)
            low = m.Dot(chart.p(b, float(PLAIN["val"][b])), radius=0.1, color=STOP_C)
            return m.VGroup(band, line, here, low).set_z_index(2)

        marks = m.always_redraw(overlay)
        stop_line = m.DashedLine(chart.p(STOP, 0), chart.p(STOP, 0.72), dash_length=0.09).set_stroke(STOP_C, width=3.5)
        stop_label = text("stop", 22, STOP_C, weight=m.BOLD).next_to(stop_line, m.UP, buff=0.1)
        back = m.CurvedArrow(chart.p(STOP - 1.5, 0.62), chart.p(BEST + 1.2, ES_KEPT + 0.06), angle=0.9, color=STOP_C, stroke_width=4)
        readout = end_readout(ES_KEPT, STOP_C, label="validation loss kept:", ref_label="at epoch 300:").move_to([3.4, -3.45, 0])

        self.say(
            "[Fix four changes when we stop.] [Back to the plain network, with no other fix.]",
            m.AnimationGroup(
                self.new_header("FIX 4 · WHEN TO STOP", "Early stopping", STOP_C, GOOD_FILL),
                m.FadeOut(self.aug_stuff),
                m.FadeOut(self.plain),
                m.FadeOut(old.frame),
                m.FadeIn(watch, scale=0.8),
                m.FadeIn(head, shift=0.15 * m.UP),
            ),
            m.AnimationGroup(m.FadeIn(chart.frame), m.Create(curves[0], run_time=1.4), m.Create(curves[1], run_time=1.4)),
        )
        self.say(
            "[As it trains, we watch the validation loss], [and remember its lowest point so far.]",
            m.AnimationGroup(m.FadeIn(names), m.FadeIn(values), m.FadeIn(track), m.FadeIn(fill), m.FadeIn(marks)),
            t.animate(run_time=2.6, rate_func=m.linear).set_value(BEST),
        )
        self.say(
            "[Twenty passes later, it still hasn't improved], [so we stop], [and roll back to the best weights, saved at the lowest point.]",
            t.animate(run_time=2.6, rate_func=m.linear).set_value(STOP),
            m.AnimationGroup(m.Create(stop_line), m.FadeIn(stop_label, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.Create(back), m.FadeIn(result, shift=0.15 * m.UP)),
        )
        self.stop(*values, fill, marks)
        self.say(
            "[That keeps a validation loss of about zero point four], [and it costs almost nothing.]",
            m.FadeIn(readout, shift=0.15 * m.UP),
            m.Indicate(watch, color=STOP_C, scale_factor=1.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section(
            "The big picture",
            notes=f"Computed: validation loss of the model you keep — no fix {PLAIN_END:.3f} (epoch 300), weight decay "
            f"{DECAY_END:.3f}, dropout {DROP_END:.3f}, augmentation {AUG_END:.3f}, early stopping {ES_KEPT:.3f} (epoch {BEST}). "
            f"Too much weight decay (λ = 0.1): training loss {HEAVY['train'][-1]:.3f}, validation loss {HEAVY_END:.3f}, "
            f"validation accuracy {HEAVY['val_acc'][-1]:.1%} (it predicts one class). Other fixes and strengths were not tuned; "
            "one dataset and seed, so the ranking is for this experiment only.",
        )
        rows = (
            ("no fix", "300 epochs", PLAIN_END, MUTED),
            ("weight decay", "λ = 0.01", DECAY_END, DECAY_C),
            ("dropout", "p = 0.5 (pixels 0.2)", DROP_END, DROP_C),
            ("data augmentation", "flip, crop, noise", AUG_END, AUG_C),
            ("early stopping", f"patience {PATIENCE}", ES_KEPT, STOP_C),
            ("weight decay", "λ = 0.1", HEAVY_END, BAD),
        )
        x0, unit, ys = -2.0, 6.3 / 0.72, (1.65, 0.9, 0.15, -0.6, -1.35, -2.1)
        title = text("validation loss of the model we keep  ·  lower is better", 24, MUTED).move_to([0, 2.35, 0])
        names, settings, bars, values = [], [], [], []
        for (name, setting, v, c), y in zip(rows, ys):
            label = m.VGroup(text(name, 24, INK, weight=m.SEMIBOLD), text(setting, 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.RIGHT)
            label.move_to([0, y, 0]).align_to([x0 - 0.3, 0, 0], m.RIGHT)
            names.append(label[0])
            settings.append(label[1])
            bar = m.Rectangle(width=v * unit, height=0.42).move_to([x0, y, 0], aligned_edge=m.LEFT)
            if c == BAD:
                bar.set_fill(BAD_FILL, opacity=1).set_stroke(BAD, width=2.5)
            else:
                bar.set_fill(c, opacity=0.85).set_stroke(width=0)
            bars.append(bar)
            values.append(code(f"{v:.2f}", 24, c).next_to(bar, m.RIGHT, buff=0.2))
        nothing = text(f"learned nothing: right {HEAVY['val_acc'][-1]:.0%} of the time", 22, INK).move_to(bars[5])
        line = m.DashedLine([bars[0].get_right()[0], ys[0] - 0.25, 0], [bars[0].get_right()[0], ys[4] - 0.3, 0], dash_length=0.08).set_stroke(MUTED, width=2.5)
        tips = m.VGroup(
            pill("tune each strength on the validation set", VIOLET, VIOLET_FILL, 22),
            pill("save the test set for one final check", PINK, PINK_FILL, 22),
        ).arrange(m.RIGHT, buff=0.4).move_to([0, -3.25, 0])

        def grow(k):
            return m.AnimationGroup(m.FadeIn(names[k], shift=0.1 * m.RIGHT), m.GrowFromEdge(bars[k], m.LEFT), m.FadeIn(values[k]))

        self.say(
            "[Here are all four, side by side]: [the validation loss of the model we keep.]",
            m.AnimationGroup(self.new_header("THE BIG PICTURE", "Four fixes, one experiment"), m.FadeIn(title)),
            m.LaggedStart(*[grow(k) for k in range(5)], lag_ratio=0.25),
        )
        self.say(
            "[Every fix beats doing nothing.] [Dropout and augmentation both shake up training with randomness.]",
            m.Create(line),
            m.AnimationGroup(*[m.Indicate(m.VGroup(names[k], bars[k]), color=c, scale_factor=1.06) for k, c in ((2, DROP_C), (3, AUG_C))]),
        )
        self.say(
            "[But each fix has a strength to tune], [and too much makes the model underfit.]",
            m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.UP) for s in settings[:5]], lag_ratio=0.15),
            m.AnimationGroup(m.FadeIn(names[5], shift=0.1 * m.RIGHT), m.FadeIn(settings[5], shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[With lambda ten times larger], [our network learned nothing at all]: [it's right only half the time.]",
            m.Indicate(settings[5], color=BAD, scale_factor=1.25),
            m.AnimationGroup(m.GrowFromEdge(bars[5], m.LEFT), m.FadeIn(values[5])),
            m.FadeIn(nothing, shift=0.1 * m.UP),
        )
        self.say(
            "[So choose each strength on the validation set], [and save the test set for one final check.]",
            m.FadeIn(tips[0], shift=0.2 * m.UP),
            m.FadeIn(tips[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Shrink (weight decay), shake (dropout and augmentation add randomness), stop (early stopping).")
        halo = glow(VIOLET, radius=4.0)
        words = on_baseline(
            m.VGroup(
                text("Shrink", 58, DECAY_C, weight=m.BOLD),
                text("Shake", 58, DROP_C, weight=m.BOLD, gradient=(DROP_C, AUG_C)),
                text("Stop", 58, STOP_C, weight=m.BOLD),
            ).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        notes = m.VGroup(*[text(s, 22, MUTED).next_to(w, m.DOWN, buff=0.3) for s, w in zip(("weight decay", "dropout · augmentation", "early stopping"), words)])
        for n in notes:
            n.set_y(notes[0].get_y())
        row = m.VGroup(words, links, notes).move_to(0.45 * m.UP)
        tagline = text("Make memorising hard, and learning easy.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [shrink the weights], [shake up the training], [and stop in time.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(notes[0])),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(notes[1])),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(notes[2])),
        )
        self.say("[Make memorising hard, and learning easy.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
