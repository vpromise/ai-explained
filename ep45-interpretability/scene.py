"""AI, Explained · Episode 45 · Mechanistic Interpretability

Running example: is there a "Golden Gate Bridge" inside a language model? Neurons are
polysemantic; features are directions; superposition packs more features than dimensions
(a toy model trained here, after Elhage et al., 2022); a sparse dictionary unpacks them
(sparse coding / sparse autoencoders, Bricken et al., 2023); the real Golden Gate Bridge feature
in Claude 3 Sonnet and Golden Gate Claude (Templeton et al., 2024); circuits, with an induction
head completing "Golden Gate … Golden Gate → Bridge" (Elhage et al., 2021; Olsson et al., 2022).

Everything plotted is computed below with seeded numpy:
- the toy model x̂ = ReLU(WᵀWx + b), 5 features, 2 dimensions, trained with Adam on dense
  (every feature on) and sparse (each feature off 95% of the time) inputs, as in the paper;
- a dictionary learned on the sparse toy's 2-D hidden activations by minimising the sparse
  coding loss ‖h − Σ fᵢdᵢ‖² + λΣ|fᵢ| (codes by projected ISTA, dictionary by least squares);
- the asserts tie every number the narration quotes to these computations.
The feature names (Golden Gate, DNA, legal, Hebrew, recipes) are illustrative labels.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

MENLO = "Menlo"  # exact characters (no ligatures) for code and IDs

# ---------------------------------------------------------------------------- the toy model

SPARSITY = 0.95  # each feature is zero with this probability (Elhage et al.'s S)
N_FEATURES, N_DIMS = 5, 2


def adam(params, grads, state, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    for k in grads:
        mean, var = state.get(k, (np.zeros_like(params[k]), np.zeros_like(params[k])))
        mean = b1 * mean + (1 - b1) * grads[k]
        var = b2 * var + (1 - b2) * grads[k] ** 2
        state[k] = (mean, var)
        params[k] -= lr * (mean / (1 - b1**t)) / (np.sqrt(var / (1 - b2**t)) + eps)


def sample(rng, batch, sparsity):
    """Toy inputs: each feature uniform on [0, 1], or zero with probability `sparsity`."""
    x = rng.uniform(0, 1, (batch, N_FEATURES))
    return x * (rng.uniform(0, 1, (batch, N_FEATURES)) >= sparsity)


def train_toy(sparsity, seed=0, steps=4000, batch=1024, lr=1e-2, decay=0.9):
    """x̂ = ReLU(WᵀWx + b), loss Σ importanceᵢ (xᵢ − x̂ᵢ)², importance 0.9^i (feature 0 matters most)."""
    rng = np.random.default_rng(seed)
    importance = decay ** np.arange(N_FEATURES)
    p = {"W": rng.normal(0, 0.3, (N_DIMS, N_FEATURES)), "b": np.zeros(N_FEATURES)}
    state = {}
    for t in range(1, steps + 1):
        x = sample(rng, batch, sparsity)
        h = x @ p["W"].T
        pre = h @ p["W"] + p["b"]
        g_pre = 2 * importance * (np.maximum(pre, 0) - x) / batch * (pre > 0)
        g_w = h.T @ g_pre + (g_pre @ p["W"].T).T @ x
        adam(p, {"W": g_w, "b": g_pre.sum(0)}, state, lr, t)
    return p["W"], p["b"]


def rotate_to(w, index, degrees):
    """Rotate the hidden space so feature `index` points at `degrees` (the loss only sees WᵀW)."""
    a = np.radians(degrees) - np.arctan2(w[1, index], w[0, index])
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ w


def sparse_codes(h, d, lam, iters=300):
    """argmin over f ≥ 0 of ½‖h − D f‖² + λ Σ f, row by row (projected ISTA)."""
    step = np.linalg.norm(d, 2) ** 2
    f = np.zeros((len(h), d.shape[1]))
    for _ in range(iters):
        f = np.maximum(f - ((f @ d.T - h) @ d + lam) / step, 0)
    return f


def learn_dictionary(w, lam, seed=1, atoms=5, rounds=30, batch=4096):
    """Sparse coding on the toy's hidden activations h = Wx: alternate codes and a least-squares dictionary."""
    rng = np.random.default_rng(seed)
    start = sample(rng, batch, SPARSITY) @ w.T
    start = start[np.linalg.norm(start, axis=1) > 0.2]
    d = start[rng.choice(len(start), atoms, replace=False)].T
    d = d / np.linalg.norm(d, axis=0)
    for _ in range(rounds):
        h = sample(rng, batch, SPARSITY) @ w.T
        f = sparse_codes(h, d, lam)
        d = h.T @ f @ np.linalg.pinv(f.T @ f + 1e-6 * np.eye(atoms))
        d = d / np.linalg.norm(d, axis=0)
    return d


def degrees_between(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


DENSE_W, DENSE_B = train_toy(0.0)
SPARSE_W, SPARSE_B = train_toy(SPARSITY, seed=3)
DENSE_W, SPARSE_W = rotate_to(DENSE_W, 0, 40.0), rotate_to(SPARSE_W, 0, 40.0)
DENSE_NORMS = np.linalg.norm(DENSE_W, axis=0)
SPARSE_NORMS = np.linalg.norm(SPARSE_W, axis=0)

KEPT = [int(i) for i in np.flatnonzero(DENSE_NORMS > 0.5)]
assert KEPT == [0, 1] and (DENSE_NORMS[2:] < 0.1).all()  # "keeps just the two most important, drops the rest"
DENSE_ANGLE = degrees_between(DENSE_W[:, 0], DENSE_W[:, 1])
assert abs(DENSE_ANGLE - 90) < 2  # "at right angles"
assert (SPARSE_NORMS > 1.0).all()  # "packs all five"

ANGLES = np.degrees(np.arctan2(SPARSE_W[1], SPARSE_W[0])) % 360
RING = [int(i) for i in sorted(range(N_FEATURES), key=lambda i: (ANGLES[i] - ANGLES[0]) % 360)]  # counter-clockwise from feature 0
GAPS = [float((ANGLES[b] - ANGLES[a]) % 360) for a, b in zip(RING, RING[1:] + RING[:1])]
assert all(abs(g - 72) < 2.5 for g in GAPS)  # "about seventy-two degrees apart": a pentagon
assert (SPARSE_B < 0).all()  # "the negative bias"


def overlap_left():
    """Single feature j active at strength s ~ U(0, 1): how much of the raw overlap (Wᵢ·Wⱼ)s reaches a
    neighbour's output after ReLU(… + bᵢ)?"""
    s = np.linspace(0, 1, 20001)
    raw = kept = 0.0
    for a, b in zip(RING, RING[1:] + RING[:1]):
        for i, j in ((a, b), (b, a)):
            dot = SPARSE_W[:, i] @ SPARSE_W[:, j]
            raw += np.mean(dot * s)
            kept += np.mean(np.maximum(dot * s + SPARSE_B[i], 0))
    return kept / raw


OVERLAP_LEFT = overlap_left()
assert OVERLAP_LEFT < 0.5  # "filters out most of their overlap"

NAMES = ["Golden Gate", "DNA", "legal", "Hebrew", "recipes"]  # illustrative, in order around the pentagon
HUES = [AMBER, TEAL, BLUE, VIOLET, LIME]
NAME = {f: NAMES[k] for k, f in enumerate(RING)}
HUE = {f: HUES[k] for k, f in enumerate(RING)}
DNA = RING[1]  # the counter-clockwise neighbour of Golden Gate
assert NAME[0] == "Golden Gate" and NAME[DNA] == "DNA"

DIRS = SPARSE_W / SPARSE_NORMS
EX_GG, EX_DNA = 0.8, 0.5  # illustrative strengths
EX_X = EX_GG * DIRS[:, 0] + EX_DNA * DIRS[:, DNA]
assert (EX_X > 0.3).all()  # "in both neurons at once"

LAM = 0.1
DICT = learn_dictionary(SPARSE_W, LAM)
COS = DICT.T @ DIRS
MATCH = [int(a) for a in COS.argmax(0)]  # the dictionary atom of each feature
DICT_ERR = np.degrees(np.arccos(np.clip(COS.max(0), -1, 1)))
assert sorted(MATCH) == list(range(N_FEATURES))  # one atom per feature
assert DICT_ERR.max() < 0.5  # "all five feature directions, each within half a degree"

GG_IN = 0.8
GG_H = GG_IN * SPARSE_W[:, 0]
GG_CODE = sparse_codes(GG_H[None], DICT, LAM)[0]
GG_BY_FEATURE = GG_CODE[MATCH]
assert (np.abs(GG_H) > 0.3).all()  # "lights up both dimensions"
assert GG_BY_FEATURE.argmax() == 0 and (GG_BY_FEATURE[1:] < 1e-9).all()  # "but only one dictionary feature"

# ---------------------------------------------------------------------------- helpers


def cite(s, size=20):
    return text(s, size, MUTED)


def menlo(s, size=22, color=INK):
    return m.Text(s, font=MENLO, font_size=size, color=color)


def formula(*parts, size=40):
    """A MathTex from (latex, colour) pairs, one part each."""
    t = m.MathTex(*[p for p, _ in parts], font_size=size, color=INK)
    for part, (_, c) in zip(t, parts):
        part.set_color(c)
    return t


def ep_recolored(name, color, fill):
    """kit.recolored(), written as assets/ep45-<name>-<color>.svg (this episode only adds ep45- files)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep45-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


def point(center, v, scale):
    return np.array(center, dtype=float) + scale * np.array([v[0], v[1], 0.0])


def vec_arrow(center, v, scale, color, width=5, tip=0.2):
    start = np.array(center, dtype=float)
    return m.Arrow(start, point(center, v, scale), buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=tip)


def cross(center, half, labels, color=FAINT, y_side=m.RIGHT):
    c = np.array(center, dtype=float)
    xa = m.Arrow(c + [-half, 0, 0], c + [half, 0, 0], buff=0, color=color, stroke_width=2.5, max_tip_length_to_length_ratio=0.05)
    ya = m.Arrow(c + [0, -half, 0], c + [0, half, 0], buff=0, color=color, stroke_width=2.5, max_tip_length_to_length_ratio=0.05)
    xl = text(labels[0], 20, MUTED).next_to(xa.get_end(), m.RIGHT, buff=0.08)
    yl = text(labels[1], 20, MUTED).next_to(ya.get_end(), y_side, buff=0.2).shift(0.22 * m.DOWN)
    return m.VGroup(xa, ya), m.VGroup(xl, yl)


def word_chip(word, color=SKY, fill=SKY_FILL, size=24, height=0.62, pad=0.3):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def cell(color=EDGE, fill=CARD, side=0.36):
    return m.Square(side_length=side).set_stroke(color, width=2).set_fill(fill, opacity=1)


def legend(features, size=20):
    items = m.VGroup()
    for f in features:
        items.add(m.VGroup(m.Dot(radius=0.08, color=HUE[f]), text(NAME[f], size, INK)).arrange(m.RIGHT, buff=0.12))
    return items.arrange(m.RIGHT, buff=0.38)


def bar_row(label, value, color, width=2.4, label_w=1.75):
    name = text(label, 20, INK if value > 0 else MUTED)
    bar = m.Rectangle(width=max(0.03, width * value), height=0.24, stroke_width=0).set_fill(color if value > 0 else FAINT, opacity=0.95)
    num = text(f"{value:.2f}", 20, color if value > 0 else MUTED, weight=m.SEMIBOLD)
    name.move_to([0, 0, 0], aligned_edge=m.RIGHT)
    bar.move_to([0.2, 0, 0], aligned_edge=m.LEFT)
    num.next_to(bar, m.RIGHT, buff=0.15)
    return m.VGroup(name, bar, num)


def point_tile(icon_name, title, lines, color, width=10.6):
    words = m.VGroup(text(title, 28, INK, weight=m.BOLD), *[text(s, 22, MUTED) for s in lines])
    words.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
    box = card(width, words.height + 0.6, color, CARD, stroke=2)
    glyph = icon(icon_name, 0.85).move_to(box.get_left() + 0.85 * m.RIGHT)
    words.move_to(box).align_to(box.get_left() + 1.65 * m.RIGHT, m.LEFT)
    return m.VGroup(box, glyph, words)


# ---------------------------------------------------------------------------- the episode


class Ep45Interpretability(Episode):
    number = 45

    def lesson(self):
        self.hook()
        self.neurons()
        self.features()
        self.superposition()
        self.autoencoders()
        self.toy_dictionary()
        self.golden_gate()
        self.circuits()
        self.why()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Is there a place inside a language model that means the Golden Gate Bridge? Facts checked 2026-10-21.")
        slabs = m.VGroup(*[card(0.4, 2.6, VIOLET, VIOLET_FILL, radius=0.12, stroke=2.5) for _ in range(7)])
        slabs.arrange(m.RIGHT, buff=0.24).move_to([0, 0.45, 0])
        model_label = text("a language model", 24, MUTED, weight=m.SEMIBOLD).next_to(slabs, m.UP, buff=0.3)
        prompt = word_chip("Golden Gate Bridge", AMBER, AMBER_FILL, size=26, height=0.72, pad=0.5).move_to([-4.9, 0.45, 0])
        feed = arrow(prompt.get_right(), slabs.get_left(), MUTED, buff=0.18)
        bridge = icon("ep45-bridge", 1.35).move_to([4.95, 0.45, 0])
        link = m.DashedLine(slabs.get_right() + 0.25 * m.RIGHT, bridge.get_left() + 0.25 * m.LEFT, dash_length=0.12).set_stroke(MUTED, width=3)
        ask = text("?", 60, PINK, weight=m.BOLD).next_to(link, m.UP, buff=0.25)
        lens = icon(ep_recolored("search", PINK, PINK_FILL), 1.35).move_to(slabs[3].get_center() + np.array([0.3, -0.25, 0]))
        define = text("reverse-engineer a trained network into parts we understand", 26, INK).move_to([0, -1.65, 0])
        topics = m.VGroup(
            pill("features", SKY, SKY_FILL, 24), pill("superposition", VIOLET, VIOLET_FILL, 24),
            pill("sparse autoencoders", TEAL, TEAL_FILL, 24), pill("circuits", GOOD, GOOD_FILL, 24),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -2.85, 0])

        self.say(
            "[Somewhere inside a language model], [is there a place that means the Golden Gate Bridge?]",
            m.AnimationGroup(
                self.new_header("THE QUESTION", "Where does a model keep an idea?"),
                m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.UP) for s in slabs], lag_ratio=0.12), m.FadeIn(model_label),
            ),
            m.AnimationGroup(
                m.FadeIn(prompt, shift=0.3 * m.RIGHT), m.GrowArrow(feed), m.Create(link),
                m.FadeIn(bridge, scale=0.8), m.FadeIn(ask, scale=0.6),
            ),
        )
        self.say(
            "[Mechanistic interpretability tries to answer that], [by reverse engineering a trained network into parts we can understand.]",
            m.AnimationGroup(m.FadeIn(lens, shift=0.5 * (m.UP + m.LEFT)), slabs[3].animate.set_stroke(PINK, width=3.5)),
            m.FadeIn(define, shift=0.2 * m.UP),
        )
        self.say(
            "We'll meet four ideas: [features], [superposition], [sparse autoencoders], [and circuits.]",
            *[m.FadeIn(t, shift=0.2 * m.UP) for t in topics],
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ neurons are polysemantic

    def neurons(self):
        self.next_section(
            "Neurons",
            notes="Bricken et al., 2023 (Towards Monosemanticity): in their one-layer transformer (512-neuron MLP), one neuron "
            "responds to a mixture of academic citations, English dialogue, HTTP requests and Korean text. The snippets on the "
            "cards are illustrative examples of each category, not the paper's dataset examples.",
        )
        neurons = m.VGroup(*[m.Circle(radius=0.22).set_stroke(VIOLET, width=3).set_fill(VIOLET_FILL, opacity=1) for _ in range(7)])
        neurons.arrange(m.DOWN, buff=0.22).move_to([-5.5, -0.3, 0])
        star = neurons[3]
        layer_label = text("one layer's neurons", 22, MUTED).next_to(neurons, m.UP, buff=0.25)
        rows = [
            ("academic citations", text("…as shown in [14].", 22, MUTED)),
            ("English dialogue", text("“Where were you?”", 22, MUTED)),
            ("HTTP requests", menlo("GET /index.html", 22, MUTED)),
            ("Korean text", text("안녕하세요", 24, MUTED)),
        ]
        cards = m.VGroup()
        for k, (title, snippet) in enumerate(rows):
            box = card(6.9, 0.8, PINK, CARD, radius=0.18, stroke=2).move_to([2.75, 1.13 - 0.95 * k, 0])
            name = text(title, 24, INK, weight=m.BOLD).move_to(box).align_to(box.get_left() + 0.3 * m.RIGHT, m.LEFT)
            snippet.move_to(box).align_to(box.get_left() + 3.75 * m.RIGHT, m.LEFT)
            cards.add(m.VGroup(box, name, snippet))
        wires = m.VGroup(*[m.Line(star.get_right(), c[0].get_left()).set_stroke(PINK, width=3) for c in cards])
        source = cite("one neuron in a one-layer transformer · Bricken et al., 2023").move_to([2.75, -2.5, 0])
        poly = pill("polysemantic: one neuron, unrelated meanings", BAD, BAD_FILL, 22).move_to([2.75, -3.2, 0])
        wrong = text("✗", 40, BAD, weight=m.BOLD).next_to(star, m.LEFT, buff=0.25)

        self.say(
            "[The obvious place to look is single neurons]: [what makes each one fire?]",
            m.AnimationGroup(
                self.new_header("NEURONS", "One neuron, many meanings"),
                m.LaggedStart(*[m.FadeIn(n, scale=0.5) for n in neurons], lag_ratio=0.1), m.FadeIn(layer_label),
            ),
            m.AnimationGroup(star.animate.set_stroke(PINK, width=4).set_fill(PINK_FILL, opacity=1), m.Flash(star, color=PINK, flash_radius=0.45)),
        )
        self.say(
            "[In one small transformer], [a single neuron fired on academic citations], [English dialogue], [web requests], [and Korean text.]",
            m.FadeIn(source, shift=0.15 * m.UP),
            *[m.AnimationGroup(m.Create(w), m.FadeIn(c, shift=0.2 * m.RIGHT)) for w, c in zip(wires, cards)],
        )
        self.say(
            "[Neurons like this are polysemantic]: [they mix unrelated meanings], [so they're the wrong unit to read.]",
            m.FadeIn(poly, shift=0.2 * m.UP),
            m.LaggedStart(*[m.Indicate(c[1], color=PINK, scale_factor=1.06) for c in cards], lag_ratio=0.2),
            m.FadeIn(wrong, scale=1.4),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ features as directions

    def features(self):
        self.next_section(
            "Features are directions",
            notes=f"Directions from the trained toy (below): Golden Gate at 40°, DNA at {ANGLES[DNA]:.0f}°. Example activation "
            f"0.8·d_GG + 0.5·d_DNA = ({EX_X[0]:.2f}, {EX_X[1]:.2f}); the strengths are illustrative.",
        )
        c = np.array([-3.4, -0.55, 0])
        sc = 2.0
        ax, ax_labels = cross(c, 2.25, ("neuron 1", "neuron 2"), y_side=m.LEFT)
        ax_labels[1].shift(0.25 * m.UP)
        gg = vec_arrow(c, DIRS[:, 0], sc, AMBER)
        gg_label = text("Golden Gate", 22, AMBER, weight=m.SEMIBOLD).next_to(gg.get_end(), m.UR, buff=0.08)
        recall = pill("Episode 9: meaning as a direction", SKY, SKY_FILL, 20).move_to([3.4, 1.95, 0])
        eq = formula(("x", SKY), ("=", INK), (r"\sum_i", INK), ("f_i", PINK), (r"\,d_i", INK), size=54).move_to([3.4, 0.75, 0])
        f_note = m.VGroup(tex("f_i", 32, PINK), text("how strongly feature i is active", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        d_note = m.VGroup(tex("d_i", 32, INK), text("its direction in activation space", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        notes = m.VGroup(f_note, d_note).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([3.4, -0.65, 0])

        tip_gg = point(c, EX_GG * DIRS[:, 0], sc)
        part_gg = m.Arrow(c, tip_gg, buff=0, color=AMBER, stroke_width=5, max_tip_length_to_length_ratio=0.2)
        part_gg_label = text("0.8 × Golden Gate", 22, AMBER, weight=m.SEMIBOLD).next_to(tip_gg, m.DR, buff=0.08)
        tip_x = point(c, EX_X, sc)
        part_dna = m.Arrow(tip_gg, tip_x, buff=0, color=TEAL, stroke_width=5, max_tip_length_to_length_ratio=0.25)
        dna_label = text("0.5 × DNA", 22, TEAL, weight=m.SEMIBOLD).next_to(part_dna.get_center(), m.RIGHT, buff=0.18)
        x_arrow = m.Arrow(c, tip_x, buff=0, color=SKY, stroke_width=6, max_tip_length_to_length_ratio=0.15)
        x_label = tex("x", 34, SKY).next_to(tip_x, m.UP, buff=0.15)
        drop_x = m.DashedLine(tip_x, [tip_x[0], c[1], 0], dash_length=0.08).set_stroke(SKY, width=2.5)
        drop_y = m.DashedLine(tip_x, [c[0], tip_x[1], 0], dash_length=0.08).set_stroke(SKY, width=2.5)
        val_x = text(f"{EX_X[0]:.2f}", 20, SKY, weight=m.SEMIBOLD).next_to([tip_x[0], c[1], 0], m.DOWN, buff=0.15)
        val_y = text(f"{EX_X[1]:.2f}", 20, SKY, weight=m.SEMIBOLD).next_to([c[0], tip_x[1], 0], m.LEFT, buff=0.15)
        example = text("an activation with two features on", 22, MUTED).move_to([3.4, -2.3, 0])

        self.say(
            "[Instead, think of a feature as a direction in activation space], [like the meaning directions of episode nine.]",
            m.AnimationGroup(
                self.new_header("FEATURES", "Meanings are directions"),
                m.Create(ax), m.FadeIn(ax_labels), m.GrowArrow(gg), m.FadeIn(gg_label),
            ),
            m.FadeIn(recall, shift=0.2 * m.UP),
        )
        self.say(
            "[An activation is a sum of features]: [each one's strength], [times its direction.]",
            m.Write(eq),
            m.AnimationGroup(m.Circumscribe(eq[3], color=PINK, buff=0.08), m.FadeIn(f_note, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Circumscribe(eq[4], color=INK, buff=0.08), m.FadeIn(d_note, shift=0.15 * m.UP)),
        )
        self.say(
            "[With Golden Gate at zero point eight, and DNA at zero point five], [the activation lands here], [in both neurons at once.]",
            m.AnimationGroup(
                m.ReplacementTransform(gg, part_gg), m.ReplacementTransform(gg_label, part_gg_label),
                m.GrowArrow(part_dna), m.FadeIn(dna_label), m.FadeIn(example),
            ),
            m.AnimationGroup(m.GrowArrow(x_arrow), m.FadeIn(x_label, scale=0.6)),
            m.AnimationGroup(m.Create(drop_x), m.Create(drop_y), m.FadeIn(val_x), m.FadeIn(val_y)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ superposition

    def superposition(self):
        gaps = ", ".join(f"{g:.1f}°" for g in GAPS)
        self.next_section(
            "Superposition",
            notes=f"Toy model of Elhage et al., 2022, trained here (Adam, 4000 steps, importance 0.9^i). Dense (S = 0): features "
            f"{KEPT} kept at {DENSE_ANGLE:.1f}°, the rest norm < 0.1. Sparse (S = {SPARSITY}): all five, norms "
            f"{', '.join(f'{v:.2f}' for v in SPARSE_NORMS)}; neighbour gaps {gaps}; biases {', '.join(f'{v:.2f}' for v in SPARSE_B)}; "
            f"only {100 * OVERLAP_LEFT:.0f}% of neighbours' raw overlap survives the ReLU. Hidden space rotated for display "
            "(the loss depends only on WᵀW).",
        )
        xs = (-6.1, -4.45, -2.8)
        ys = [1.75 - 0.42 * k for k in range(N_FEATURES)]
        ins = m.VGroup(*[m.Circle(radius=0.15).set_stroke(HUE[f], width=3).set_fill(HUE[f], opacity=0.9).move_to([xs[0], y, 0]) for f, y in zip(range(N_FEATURES), ys)])
        hid = m.VGroup(*[m.Circle(radius=0.17).set_stroke(PINK, width=3).set_fill(PINK_FILL, opacity=1).move_to([xs[1], y, 0]) for y in (1.21, 0.49)])
        outs = m.VGroup(*[m.Circle(radius=0.15).set_stroke(HUE[f], width=3).set_fill(BG, opacity=1).move_to([xs[2], y, 0]) for f, y in zip(range(N_FEATURES), ys)])
        e_in = m.VGroup(*[m.Line(a.get_center(), b.get_center(), buff=0.17).set_stroke(MUTED, width=1.5, opacity=0.5) for a in ins for b in hid])
        e_out = m.VGroup(*[m.Line(a.get_center(), b.get_center(), buff=0.17).set_stroke(MUTED, width=1.5, opacity=0.5) for a in hid for b in outs])
        e_in.set_z_index(-1)
        e_out.set_z_index(-1)
        tags = m.VGroup(*[text(s, 20, MUTED).move_to([x, -0.35, 0]) for s, x in zip(("5 features", "2 dims", "5 outputs"), xs)])
        eq = formula(
            (r"\hat{x}", INK), ("=", INK), (r"\mathrm{ReLU}\big(", INK), (r"W^{\top}", VIOLET), ("W", VIOLET), ("x", SKY),
            ("+", INK), ("b", PINK), (r"\big)", INK), size=38,
        ).move_to([-4.45, -1.25, 0])
        source = cite("Elhage et al., 2022").move_to([-4.45, -1.95, 0])
        sp_pill = pill("superposition", VIOLET, VIOLET_FILL, 26).move_to([-4.45, -2.65, 0])
        sp_note = text("features share dimensions", 22, MUTED).move_to([-4.45, -3.3, 0])

        def panel(x, title, w, features, verdict, color, fill):
            box = card(4.2, 4.6, EDGE, CARD, stroke=2).move_to([x, -0.35, 0])
            head = text(title, 24, INK, weight=m.BOLD).move_to([x, 1.58, 0])
            centre = np.array([x, -0.3, 0])
            ax, labels = cross(centre, 1.28, ("dim 1", "dim 2"))
            arrows = m.VGroup(*[vec_arrow(centre, w[:, f], 1.0, HUE[f], width=5, tip=0.22) for f in features])
            tag = pill(verdict, color, fill, 22).move_to([x, -2.22, 0])
            return box, head, centre, ax, labels, arrows, tag

        p1 = panel(0.2, "often active (dense)", DENSE_W, KEPT, "keeps 2 of 5", BAD, BAD_FILL)
        p2 = panel(4.5, "rarely active (sparse)", SPARSE_W, RING, "keeps all 5", GOOD, GOOD_FILL)
        box1, head1, c1, ax1, lab1, arr1, tag1 = p1
        box2, head2, c2, ax2, lab2, arr2, tag2 = p2
        u, v = DENSE_W[:, 0] / DENSE_NORMS[0], DENSE_W[:, 1] / DENSE_NORMS[1]
        right = m.Polygon(point(c1, u, 0.24), point(c1, u + v, 0.24), point(c1, v, 0.24), c1).set_stroke(INK, width=2).set_fill(opacity=0)
        tips = [point(c2, SPARSE_W[:, f], 1.0) for f in RING]
        pentagon = m.DashedVMobject(m.Polygon(*tips).set_stroke(FAINT, width=2.5), num_dashes=40).set_stroke(MUTED, width=2.5)
        # the angle mark between the neighbouring pair whose bisector sits farthest from both axes
        pair = max((k for k in range(N_FEATURES) if round(GAPS[k]) == 72), key=lambda k: -abs(((ANGLES[RING[k]] + GAPS[k] / 2) % 90) - 45))
        a0, gap = ANGLES[RING[pair]], GAPS[pair]
        arc = m.Arc(radius=0.32, start_angle=np.radians(a0), angle=np.radians(gap), arc_center=c2).set_stroke(PINK, width=3)
        mid = np.radians(a0 + gap / 2)
        arc_label = text(f"{gap:.0f}°", 22, PINK, weight=m.SEMIBOLD).move_to(c2 + 0.66 * np.array([np.cos(mid), np.sin(mid), 0]))
        keys = legend(RING).move_to([2.4, -3.15, 0])
        drops = m.VGroup(
            *[m.DashedLine(t, [t[0], c2[1], 0], dash_length=0.07).set_stroke(HUE[f], width=2.5) for t, f in zip(tips, RING) if abs(t[1] - c2[1]) > 0.15]
        )
        mixed = pill("dim 1 reads a mix", PINK, PINK_FILL, 22).move_to(tag2)

        self.say(
            "[But a model needs far more features than it has neurons.] [So how do they all fit?]",
            m.AnimationGroup(
                self.new_header("SUPERPOSITION", "More features than dimensions"),
                m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in ins], lag_ratio=0.12), m.FadeIn(hid, scale=0.5),
            ),
            m.Circumscribe(hid, color=PINK, buff=0.12),
        )
        self.say(
            "[A twenty twenty-two study built a toy to find out]: [five features squeezed through two dimensions], [then decoded back out.]",
            m.FadeIn(source, shift=0.15 * m.UP),
            m.AnimationGroup(m.Create(e_in), m.FadeIn(tags[0]), m.FadeIn(tags[1])),
            m.AnimationGroup(m.Create(e_out), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in outs], lag_ratio=0.1), m.FadeIn(tags[2])),
        )
        self.say(
            "[One matrix compresses five numbers into two], [its transpose spreads them back out], [and a ReLU with a bias cleans up.]",
            m.AnimationGroup(m.Write(eq), m.ShowPassingFlash(e_in.copy().set_stroke(VIOLET, width=2.5, opacity=1), time_width=0.6)),
            m.AnimationGroup(m.Circumscribe(eq[3], color=VIOLET, buff=0.06), m.ShowPassingFlash(e_out.copy().set_stroke(VIOLET, width=2.5, opacity=1), time_width=0.6)),
            m.AnimationGroup(m.Circumscribe(eq[2], color=PINK, buff=0.06), m.Circumscribe(eq[7], color=PINK, buff=0.06)),
        )
        self.say(
            "[We trained this toy ourselves.] [When features are often active], [it keeps just the two most important, at right angles], [and drops the rest.]",
            m.AnimationGroup(m.FadeIn(box1), m.FadeIn(box2), m.FadeIn(keys, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(head1), m.Create(ax1), m.FadeIn(lab1)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in arr1], lag_ratio=0.3), m.Create(right)),
            m.FadeIn(tag1, shift=0.2 * m.UP),
        )
        self.say(
            "[But when each feature is rarely active], [it packs all five into a pentagon.]",
            m.AnimationGroup(m.FadeIn(head2), m.Create(ax2), m.FadeIn(lab2)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in arr2], lag_ratio=0.2), m.Create(pentagon), m.FadeIn(tag2, shift=0.2 * m.UP)),
        )
        self.say(
            "[Neighbours sit about seventy-two degrees apart], [and the negative bias filters out most of their overlap.]",
            m.AnimationGroup(m.Create(arc), m.FadeIn(arc_label)),
            m.Circumscribe(eq[7], color=PINK, buff=0.08),
        )
        self.say(
            "[That's superposition]: [more features than dimensions], [sharing space because they rarely fire together.]",
            m.FadeIn(sp_pill, shift=0.2 * m.UP),
            m.FadeIn(sp_note, shift=0.15 * m.UP),
            m.Circumscribe(m.VGroup(box2), color=VIOLET, buff=0.05),
        )
        self.say(
            "[Read along any one axis], [and several features mix together]: [that's why neurons look polysemantic.]",
            m.AnimationGroup(m.FadeOut(arc), m.FadeOut(arc_label), m.ShowPassingFlash(m.Line(c2 + [-1.28, 0, 0], c2 + [1.28, 0, 0]).set_stroke(PINK, width=8), time_width=0.6)),
            m.LaggedStart(*[m.Create(d) for d in drops], lag_ratio=0.15),
            m.AnimationGroup(m.FadeOut(tag2), m.FadeIn(mixed, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ sparse autoencoders

    def autoencoders(self):
        self.next_section(
            "Sparse autoencoders",
            notes="SAE (Bricken et al., 2023): f = ReLU(W_enc(x − b_dec) + b_enc), x̂ = W_dec f + b_dec, loss ‖x − x̂‖² + λ‖f‖₁. "
            "The sparse coding objective: Olshausen & Field, 1996. Towards Monosemanticity decomposed a 512-neuron MLP layer of a "
            "one-layer transformer into more than 4,000 features (the 4,096-feature run; dictionaries up to 131,072). "
            "The lit cells in the strip are illustrative.",
        )
        y = 1.35
        x_col = m.VGroup(*[cell(SKY, SKY_FILL) for _ in range(4)]).arrange(m.DOWN, buff=0.06).move_to([-5.7, y, 0])
        x_tag = tex("x", 32, SKY).next_to(x_col, m.LEFT, buff=0.2)
        strip = m.VGroup(*[cell() for _ in range(16)]).arrange(m.RIGHT, buff=0.07).move_to([0, y, 0])
        hat_col = m.VGroup(*[cell(SKY, SKY_FILL) for _ in range(4)]).arrange(m.DOWN, buff=0.06).move_to([5.7, y, 0])
        hat_tag = tex(r"\hat{x}", 32, SKY).next_to(hat_col, m.RIGHT, buff=0.2)
        enc = arrow(x_col.get_right(), strip.get_left(), TEAL, buff=0.2)
        dec = arrow(strip.get_right(), hat_col.get_left(), TEAL, buff=0.2)
        enc_tag = text("encoder", 22, TEAL, weight=m.SEMIBOLD).next_to(enc, m.UP, buff=0.12)
        dec_tag = text("decoder", 22, TEAL, weight=m.SEMIBOLD).next_to(dec, m.UP, buff=0.12)
        strip_tag = text("f: many features, almost all zero", 22, MUTED).next_to(strip, m.DOWN, buff=0.25)
        lit = [(4, AMBER, 0.95), (11, TEAL, 0.55)]

        eq = formula(
            (r"\mathcal{L}", INK), ("=", INK), (r"\Big\|", INK), ("x", SKY), ("-", INK), (r"\sum_i", INK), ("f_i", PINK),
            (r"\,d_i", INK), (r"\Big\|^2", INK), ("+", INK), (r"\lambda", AMBER), (r"\sum_i", INK), (r"|f_i|", PINK), size=48,
        ).move_to([0, -0.75, 0])
        brace1 = m.Brace(eq[2:9], m.DOWN, buff=0.1).set_color(SKY)
        brace2 = m.Brace(eq[10:13], m.DOWN, buff=0.1).set_color(AMBER)
        lab1 = text("reconstruction error", 22, SKY, weight=m.SEMIBOLD).next_to(brace1, m.DOWN, buff=0.1)
        lab2 = text("sparsity penalty", 22, AMBER, weight=m.SEMIBOLD).next_to(brace2, m.DOWN, buff=0.1)
        sc_cite = cite("sparse coding: Olshausen & Field, 1996").move_to([0, -2.55, 0])
        result = pill("512 neurons  →  over 4,000 features", TEAL, TEAL_FILL, 24).move_to([-1.2, -3.25, 0])
        b_cite = cite("Bricken et al., 2023").next_to(result, m.RIGHT, buff=0.4)

        self.say(
            "[To pull the features back out], [we train a sparse autoencoder on the model's activations.]",
            self.new_header("SPARSE AUTOENCODERS", "A dictionary of features"),
            m.AnimationGroup(m.FadeIn(x_col, shift=0.2 * m.RIGHT), m.FadeIn(x_tag)),
        )
        self.say(
            "[Its encoder spreads each activation over many more features], [almost all of them zero.]",
            m.AnimationGroup(m.GrowArrow(enc), m.FadeIn(enc_tag), m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in strip], lag_ratio=0.05)),
            m.AnimationGroup(*[strip[i].animate.set_fill(c, opacity=o).set_stroke(c) for i, c, o in lit], m.FadeIn(strip_tag)),
        )
        self.say(
            "[Its decoder adds up their directions] [to rebuild the activation.]",
            m.AnimationGroup(m.GrowArrow(dec), m.FadeIn(dec_tag)),
            m.AnimationGroup(m.FadeIn(hat_col, shift=0.2 * m.RIGHT), m.FadeIn(hat_tag)),
        )
        self.say(
            "[The loss has two parts]: [the reconstruction error], [plus lambda times the total feature activity], [which keeps the code sparse.]",
            m.Write(eq),
            m.AnimationGroup(m.GrowFromCenter(brace1), m.FadeIn(lab1)),
            m.AnimationGroup(m.GrowFromCenter(brace2), m.FadeIn(lab2)),
            m.Circumscribe(strip, color=AMBER, buff=0.1),
        )
        self.say(
            "[That's the classic sparse coding objective], [with an encoder that makes it fast enough for millions of activations.]",
            m.FadeIn(sc_cite, shift=0.15 * m.UP),
            m.ShowPassingFlash(m.Line(enc.get_start(), enc.get_end()).set_stroke(TEAL, width=10), time_width=0.7),
        )
        self.say(
            "[On a small transformer, five hundred and twelve neurons became over four thousand features], [most of them far easier to interpret.]",
            m.AnimationGroup(m.FadeIn(result, shift=0.2 * m.UP), m.FadeIn(b_cite)),
            m.Circumscribe(result, color=TEAL, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the dictionary on the toy

    def toy_dictionary(self):
        errs = ", ".join(f"{NAME[f]} {DICT_ERR[f]:.3f}°" for f in range(N_FEATURES))
        self.next_section(
            "Dictionary on the toy",
            notes=f"Sparse coding (λ = {LAM}) on the sparse toy's hidden activations, 30 rounds of ISTA codes + least-squares "
            f"dictionary, seeded. Angle to the true feature: {errs}. Golden Gate input 0.8 → h = ({GG_H[0]:.2f}, {GG_H[1]:.2f}); "
            f"code: Golden Gate {GG_BY_FEATURE[0]:.2f}, all others exactly 0.",
        )
        box = card(6.2, 5.0, EDGE, CARD, stroke=2).move_to([-3.25, -0.45, 0])
        title = text("true features (faint) · learned dictionary (white)", 20, MUTED).move_to([-3.25, 1.75, 0])
        c = np.array([-3.25, -0.55, 0])
        sc = 1.45
        ax, ax_labels = cross(c, 2.0, ("dim 1", "dim 2"))
        truth = m.VGroup(*[vec_arrow(c, SPARSE_W[:, f], sc, HUE[f], width=11, tip=0.2).set_opacity(0.4) for f in RING])
        learned = m.VGroup(
            *[vec_arrow(c, DICT[:, MATCH[f]] * SPARSE_NORMS[f], sc, INK, width=3, tip=0.16) for f in RING]
        )
        labels = m.VGroup()
        for f in RING:
            tip = point(c, SPARSE_W[:, f], sc)
            out = np.array([DIRS[0, f], DIRS[1, f], 0])
            labels.add(text(NAME[f], 20, HUE[f], weight=m.SEMIBOLD).move_to(tip + 0.42 * out + np.array([0.35 * np.sign(out[0]) * abs(out[0]), 0, 0])))
        err = pill(f"largest error: {DICT_ERR.max():.2f}°", GOOD, GOOD_FILL, 22).move_to([3.6, 1.75, 0])

        dot = m.VGroup(
            m.Circle(radius=0.16).set_stroke(AMBER, width=3).set_fill(BG, opacity=0), m.Dot(radius=0.07, color=AMBER)
        ).move_to(point(c, GG_H, sc)).set_z_index(3)
        heading = text("a Golden Gate input, strength 0.8", 24, INK, weight=m.BOLD).move_to([3.6, 0.85, 0])
        dims_title = text("the two dimensions", 20, MUTED).move_to([3.6, 0.25, 0])
        dims = m.VGroup(*[bar_row(f"dim {k + 1}", abs(GG_H[k]), PINK) for k in range(2)]).arrange(m.DOWN, buff=0.14)
        feats_title = text("the dictionary's features", 20, MUTED)
        feats = m.VGroup(*[bar_row(NAME[f], GG_BY_FEATURE[f], HUE[f]) for f in RING]).arrange(m.DOWN, buff=0.14)
        for rows in (dims, feats):
            for r in rows:
                r.shift((rows[0][1].get_left()[0] - r[1].get_left()[0]) * m.RIGHT)
        dims.move_to([3.6, -0.35, 0]).shift((3.0 - dims[0][1].get_left()[0]) * m.RIGHT)
        feats_title.move_to([3.6, -1.15, 0])
        feats.next_to(feats_title, m.DOWN, buff=0.2).shift((3.0 - feats[0][1].get_left()[0]) * m.RIGHT)

        self.say(
            "[Back to our toy]: [minimizing this loss on its two dimensions] [recovers all five feature directions, each within half a degree.]",
            m.AnimationGroup(
                self.new_header("DICTIONARY LEARNING", "Unpacking the toy"),
                m.FadeIn(box), m.FadeIn(title), m.Create(ax), m.FadeIn(ax_labels), m.FadeIn(truth), m.FadeIn(labels),
            ),
            m.LaggedStart(*[m.GrowArrow(a) for a in learned], lag_ratio=0.2),
            m.FadeIn(err, shift=0.2 * m.UP),
        )
        self.say(
            "[A Golden Gate input lights up both dimensions], [but only one dictionary feature.]",
            m.AnimationGroup(
                m.FadeIn(dot, scale=1.6), m.FadeIn(heading), m.FadeIn(dims_title),
                *[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2])) for r in dims],
            ),
            m.AnimationGroup(
                m.FadeIn(feats_title),
                *[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2])) for r in feats],
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the Golden Gate feature

    def golden_gate(self):
        self.next_section(
            "The Golden Gate feature",
            notes="Templeton et al., 2024 (Scaling Monosemanticity): SAEs with ~1M, ~4M and ~34M features on the middle layer's "
            "residual stream of Claude 3 Sonnet. Feature 34M/31164353 responds to the Golden Gate Bridge in English, in other "
            "languages (e.g. Japanese) and in images. Clamped to 10× its maximum activation, asked 'What is your physical form?', "
            "the model answered 'I am the Golden Gate Bridge, a famous suspension bridge that spans the San Francisco Bay…' "
            "(default: 'I don't actually have a physical form. I'm an artificial intelligence…'). Golden Gate Claude: public "
            "research demo for 24 hours from 23 May 2024 (anthropic.com/news/golden-gate-claude). The example text rows are "
            "illustrative. Facts checked 2026-10-21.",
        )
        source = cite("Templeton et al., 2024").move_to([-5.35, -3.35, 0])
        model = card(3.6, 4.7, VIOLET, CARD, stroke=2).move_to([-4.6, -0.45, 0])
        model_name = text("Claude 3 Sonnet", 26, INK, weight=m.BOLD).move_to([-4.6, 1.5, 0])
        layers = m.VGroup(*[m.RoundedRectangle(corner_radius=0.08, width=2.6, height=0.2).set_stroke(VIOLET, width=2).set_fill(VIOLET_FILL, opacity=1) for _ in range(11)])
        layers.arrange(m.DOWN, buff=0.12).move_to([-4.6, -0.75, 0])
        middle = layers[5]
        sae = card(2.3, 1.05, TEAL, TEAL_FILL, radius=0.2).move_to([-1.25, -0.75, 0])
        sae_words = m.VGroup(text("sparse", 22, INK, weight=m.SEMIBOLD), text("autoencoder", 22, INK, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.06).move_to(sae)
        tap = arrow(middle.get_right(), sae.get_left(), PINK, buff=0.12)
        sizes = m.VGroup(*[pill(s, TEAL, TEAL_FILL, 20) for s in ("~1M features", "~4M features", "~34M features")]).arrange(m.DOWN, buff=0.15)
        sizes.next_to(sae, m.DOWN, buff=0.3)

        fcard = card(5.4, 4.7, AMBER, CARD, stroke=2).move_to([3.85, -0.45, 0])
        fid = menlo("feature 34M/31164353", 20, MUTED).move_to([3.85, 1.45, 0])
        fname = text("Golden Gate Bridge", 30, AMBER, weight=m.BOLD).move_to([3.85, 0.95, 0])

        def example(kind, body, y):
            tag = text(kind, 20, MUTED).move_to([2.45, y, 0], aligned_edge=m.RIGHT)
            body.move_to([2.65, y, 0], aligned_edge=m.LEFT)
            return m.VGroup(tag, body)

        rows = m.VGroup(
            example("English", text("…the Golden Gate Bridge", 22, INK, t2c={"Golden Gate Bridge": AMBER}), 0.15),
            example("Japanese", text("ゴールデンゲートブリッジ", 22, AMBER), -0.75),
            example("images", icon("ep45-bridge", 0.75), -1.75),
        )

        self.say(
            "[In twenty twenty-four, the same research team scaled this up], [to a production Claude model.]",
            m.AnimationGroup(self.new_header("SCALING UP", "A Golden Gate feature"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(model), m.FadeIn(model_name), m.LaggedStart(*[m.FadeIn(l, shift=0.1 * m.UP) for l in layers], lag_ratio=0.06)),
        )
        self.say(
            "[They trained sparse autoencoders on a middle layer], [with up to about thirty-four million features.]",
            m.AnimationGroup(middle.animate.set_fill(PINK, opacity=1).set_stroke(PINK), m.GrowArrow(tap), m.FadeIn(sae), m.FadeIn(sae_words)),
            m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.UP) for s in sizes], lag_ratio=0.3),
        )
        self.say(
            "[One feature responds to the Golden Gate Bridge]: [in English], [in other languages], [and even in images of the bridge.]",
            m.AnimationGroup(m.FadeIn(fcard), m.FadeIn(fid), m.FadeIn(fname, shift=0.15 * m.UP)),
            *[m.FadeIn(r, shift=0.2 * m.LEFT) for r in rows],
        )

        track = m.Line([-2.4, 1.75, 0], [3.6, 1.75, 0]).set_stroke(EDGE, width=6)
        at = lambda v: np.array([-2.4 + 0.6 * v, 1.75, 0])  # noqa: E731  (0 … 10× max)
        knob = m.Circle(radius=0.17).set_stroke(AMBER, width=3).set_fill(AMBER_FILL, opacity=1).move_to(at(1)).set_z_index(2)
        fill = m.Line(at(0), at(1)).set_stroke(AMBER, width=6).set_z_index(1)
        fill_on = m.Line(at(0), at(10)).set_stroke(AMBER, width=6).set_z_index(1)
        slider_name = text("Golden Gate feature", 24, AMBER, weight=m.BOLD).next_to(track, m.LEFT, buff=0.45)
        ticks = m.VGroup(
            text("1× max", 20, MUTED).next_to(at(1), m.DOWN, buff=0.28), text("10× max", 20, AMBER, weight=m.SEMIBOLD).next_to(at(10), m.DOWN, buff=0.28)
        )
        user = bubble(text("What is your physical form?", 24, INK), SKY, SKY_FILL, tail="left").move_to([0, 0.3, 0])
        plain_tag = pill("normal", MUTED, CARD, 20).move_to([-3.4, -0.75, 0])
        plain_words = text("I don't actually have a\nphysical form. I'm an\nartificial intelligence.", 22, INK)
        plain = m.VGroup(card(plain_words.width + 0.7, plain_words.height + 0.55, EDGE, CARD, radius=0.3), plain_words)
        plain_words.move_to(plain[0])
        plain.next_to(plain_tag, m.DOWN, buff=0.2)
        bridge_tag = pill("clamped at 10×", AMBER, AMBER_FILL, 20).move_to([3.15, -0.75, 0])
        bridge_words = text("I am the Golden Gate Bridge,\na famous suspension bridge that\nspans the San Francisco Bay.", 22, INK)
        bridge_box = m.VGroup(card(bridge_words.width + 0.7, bridge_words.height + 0.55, AMBER, AMBER_FILL, radius=0.3), bridge_words)
        bridge_words.move_to(bridge_box[0])
        bridge_box.next_to(bridge_tag, m.DOWN, buff=0.2)
        demo = pill("Golden Gate Claude · public research demo · 24 hours, May 2024", AMBER, AMBER_FILL, 20).move_to([1.0, -3.35, 0])
        old = [model, model_name, layers, sae, sae_words, tap, sizes, fcard, fid, fname, rows]

        self.say(
            "[Then they turned it up]: [clamping it at ten times its maximum activation.]",
            m.AnimationGroup(
                *[m.FadeOut(o) for o in old],
                m.AnimationGroup(m.FadeIn(track), m.FadeIn(fill), m.FadeIn(knob), m.FadeIn(slider_name), m.FadeIn(ticks[0])),
            ),
            m.AnimationGroup(knob.animate.move_to(at(10)), m.ReplacementTransform(fill, fill_on), m.FadeIn(ticks[1])),
        )
        self.say(
            "[Asked about its physical form], [the model normally says it has none.]",
            m.FadeIn(user, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(plain_tag), m.FadeIn(plain, shift=0.2 * m.UP)),
        )
        self.say(
            "[With the feature clamped, it answered]: [I am the Golden Gate Bridge.]",
            m.FadeIn(bridge_tag, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(bridge_box[0]), m.AddTextLetterByLetter(bridge_words, time_per_char=0.02)),
        )
        self.say(
            "[For twenty-four hours, the public could chat with this Golden Gate Claude], [and it brought up the bridge in reply to almost anything.]",
            m.FadeIn(demo, shift=0.2 * m.UP),
            m.Circumscribe(bridge_box, color=AMBER, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ circuits

    def circuits(self):
        self.next_section(
            "Circuits",
            notes="Induction heads: described in Elhage et al., 2021 (A Mathematical Framework for Transformer Circuits); "
            "Olsson et al., 2022 (In-context Learning and Induction Heads) show they form in an abrupt phase change early in "
            "training, coinciding with a sharp improvement in in-context learning. The sentence is illustrative.",
        )
        top = m.VGroup(pill("features ≈ variables", SKY, SKY_FILL, 22), pill("circuits ≈ code", GOOD, GOOD_FILL, 22)).arrange(m.RIGHT, buff=0.6).move_to([0, 2.05, 0])
        words = ["the", "Golden", "Gate", "Bridge", "(…)", "drove", "over", "the", "Golden", "Gate", "?"]
        wide = text("Bridge", 24, weight=m.SEMIBOLD).width + 0.3
        chips = m.VGroup(*[word_chip(w) if w != "?" else chip(w, SKY, SKY_FILL, width=wide, height=0.62, size=24) for w in words]).arrange(m.RIGHT, buff=0.12).move_to([0, 0.3, 0])
        prev = m.VGroup(
            *[
                m.ArcBetweenPoints(chips[k - 1].get_bottom() + 0.06 * m.DOWN, chips[k].get_bottom() + 0.06 * m.DOWN, angle=1.4).set_stroke(TEAL, width=3)
                for k in range(1, len(chips))
                if "…" not in words[k] + words[k - 1]
            ]
        )
        note = pill("after: Gate", TEAL, TEAL_FILL, 20).next_to(chips[3], m.DOWN, buff=0.55)
        jump = m.CurvedArrow(chips[9].get_top() + 0.08 * m.UP, chips[3].get_top() + 0.08 * m.UP, angle=0.6, color=PINK)
        jump.set_stroke(PINK, width=4)
        guess = word_chip("Bridge", GOOD, GOOD_FILL).move_to(chips[10])
        steps = m.VGroup(
            text("1   previous-token head: each position stores the word before it", 22, TEAL),
            text("2   induction head: find where this word came before, copy what followed", 22, PINK),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([0, -1.75, 0])
        phase = pill("forms abruptly in training, as in-context learning jumps", GOOD, GOOD_FILL, 22).move_to([0, -2.8, 0])
        source = cite("Elhage et al., 2021 · Olsson et al., 2022").move_to([0, -3.45, 0])

        self.say(
            "[If features are the variables], [circuits are the code that wires them into algorithms.]",
            m.AnimationGroup(self.new_header("CIRCUITS", "How features connect"), m.FadeIn(top[0], shift=0.2 * m.UP)),
            m.FadeIn(top[1], shift=0.2 * m.UP),
        )
        self.say(
            "[A classic example is the induction head], [found in many transformers.]",
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.08),
            m.FadeIn(source),
        )
        self.say(
            "[Say our text mentioned the Golden Gate Bridge], [and now it says Golden Gate again.]",
            m.LaggedStart(*[m.Indicate(c, color=AMBER, scale_factor=1.1) for c in chips[1:4]], lag_ratio=0.2),
            m.LaggedStart(*[m.Indicate(c, color=AMBER, scale_factor=1.1) for c in chips[8:11]], lag_ratio=0.2),
        )
        self.say(
            "[First, a previous-token head copies each word into the next position]: [so Bridge carries a note that it followed Gate.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(a) for a in prev], lag_ratio=0.08), m.FadeIn(steps[0], shift=0.15 * m.UP)),
            m.FadeIn(note, shift=0.2 * m.UP),
        )
        self.say(
            "[Then the induction head, at the latest Gate], [looks for a word whose note says Gate], [finds Bridge], [and copies it as the next word.]",
            m.AnimationGroup(m.FadeIn(steps[1], shift=0.15 * m.UP), m.Indicate(chips[9], color=PINK, scale_factor=1.12)),
            m.Create(jump),
            m.Circumscribe(m.VGroup(chips[3], note), color=PINK, buff=0.08),
            m.AnimationGroup(m.FadeOut(chips[10]), m.FadeIn(guess, target_position=chips[3])),
        )
        self.say(
            "[These heads form abruptly during training], [right as in-context learning improves.]",
            m.FadeIn(phase, shift=0.2 * m.UP),
            m.Circumscribe(phase, color=GOOD, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ why it matters, and the catch

    def why(self):
        self.next_section(
            "Why it matters",
            notes="Audit and steering (Templeton et al., 2024). Catches: SAE reconstruction error (unexplained 'dark matter'), "
            "feature splitting as dictionaries grow (Bricken et al., 2023), and circuit analysis is labour-intensive.",
        )
        tiles = m.VGroup(
            point_tile(ep_recolored("search", GOOD, GOOD_FILL), "Audit", ["see which features drive an answer"], GOOD),
            point_tile(ep_recolored("gear", AMBER, AMBER_FILL), "Steer", ["turn a feature up or down, like Golden Gate Claude"], AMBER),
            point_tile(
                "alert", "The catch",
                [
                    "autoencoders leave part of each activation unexplained",
                    "features split into finer ones as the dictionary grows",
                    "tracing full circuits is slow, careful work",
                ],
                BAD,
            ),
        ).arrange(m.DOWN, buff=0.25).move_to([0, -0.5, 0])
        catch = tiles[2]
        lines = catch[2][1:]

        self.say(
            "[Why does this matter?] [Features let us audit what a model relies on], [and steer it, as Golden Gate Claude showed.]",
            self.new_header("WHY IT MATTERS", "Audit, steer, and the catch"),
            m.FadeIn(tiles[0], shift=0.2 * m.UP),
            m.FadeIn(tiles[1], shift=0.2 * m.UP),
        )
        self.say(
            "[The catch]: [autoencoders leave part of each activation unexplained], [and features split into finer ones as dictionaries grow.]",
            m.FadeIn(m.VGroup(catch[0], catch[1], catch[2][0]), shift=0.2 * m.UP),
            m.FadeIn(lines[0], shift=0.15 * m.UP),
            m.FadeIn(lines[1], shift=0.15 * m.UP),
        )
        self.say("[And tracing full circuits is still slow, careful work.]", m.FadeIn(lines[2], shift=0.15 * m.UP))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Directions, dictionaries, circuits.")
        halo = glow(PINK, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 52, c, weight=m.BOLD) for w, c in zip(("Directions", "Dictionaries", "Circuits"), (SKY, TEAL, GOOD))]).arrange(m.RIGHT, buff=1.0)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Find the features, then trace how they connect.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [directions], [dictionaries], [circuits.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Find the features, then trace how they connect.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
