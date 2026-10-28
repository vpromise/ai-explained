"""AI, Explained · Episode 39 · Mixture of Experts

Running example: routing tokens to specialists. The sentence "The capital of France is Paris"
goes through one mixture-of-experts layer with eight experts and top-2 routing; we follow its
last token, "Paris": router scores (a linear layer), top-k, a softmax over the survivors, the
two chosen experts, the weighted sum. Then Mixtral 8x7B's parameter count (computed from its
published architecture), a toy router trained without and with the Switch Transformer's
load-balancing loss, and the costs (memory, communication, experts that are not topic experts).

Everything plotted is computed below with seeded numpy, and every number the narration quotes
is asserted, so the scene refuses to render if the data and the words disagree. The token
vectors, router weights and the toy router are illustrative; Mixtral's counts are real.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

TOKEN, TOKEN_FILL = SKY, SKY_FILL  # tokens and their vectors
ROUTER, ROUTER_FILL = AMBER, AMBER_FILL  # the router, its scores and gate weights
EXPERT, EXPERT_FILL = VIOLET, VIOLET_FILL  # experts (and the parameters they hold)
PICK, PICK_FILL = PINK, PINK_FILL  # the top-k choice
OUT, OUT_FILL = TEAL, TEAL_FILL  # the layer's output
SHARED = BLUE  # parameters every token uses

N_EXPERTS, TOP_K = 8, 2


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------- Mixtral 8x7B

# Table 1 of Jiang et al., 2024 ("Mixtral of Experts", arXiv:2401.04088)
MX = dict(dim=4096, layers=32, head_dim=128, hidden=14336, heads=32, kv_heads=8, vocab=32000, experts=8, top_k=2)
_ffn = 3 * MX["dim"] * MX["hidden"]  # one SwiGLU expert in one layer: three matrices
_attn = MX["dim"] * MX["heads"] * MX["head_dim"] * 2 + 2 * MX["dim"] * MX["kv_heads"] * MX["head_dim"]  # Wq, Wo, Wk, Wv
_per_layer_shared = _attn + 2 * MX["dim"] + MX["dim"] * MX["experts"]  # attention, two norms, the router
EXPERT_B = MX["layers"] * _ffn / 1e9  # one expert's weights across all layers
SHARED_B = (MX["layers"] * _per_layer_shared + 2 * MX["vocab"] * MX["dim"] + MX["dim"]) / 1e9  # + embeddings, output, final norm
TOTAL_B = MX["experts"] * EXPERT_B + SHARED_B
ACTIVE_B = MX["top_k"] * EXPERT_B + SHARED_B
GB_16BIT = TOTAL_B * 2  # two bytes per parameter
assert round(TOTAL_B) == 47 and round(ACTIVE_B) == 13  # "about forty-seven billion", "about thirteen billion"
assert round(GB_16BIT) == 93  # "about ninety-three gigabytes in sixteen-bit"
assert 8 * 7 == 56 and TOTAL_B < 56  # "eight times seven, fifty-six billion"

# ---------------------------------------------------------------------------- the running example

WORDS = ["The", "capital", "of", "France", "is", "Paris"]
_rng = np.random.default_rng(1593)
TOKEN_VECS = _rng.normal(0, 1, (6, 6))  # illustrative 6-number token vectors (real models: 4096)
ROUTER_W = _rng.normal(0, 0.5, (6, N_EXPERTS))  # the router: one linear layer, 6 → 8
SCORES = np.round(TOKEN_VECS @ ROUTER_W, 1)  # one score per expert, rounded as shown
TOPS = np.argsort(-SCORES, axis=1)[:, :TOP_K]  # each token's two experts (0-based)
GATES = np.array([softmax(SCORES[t][TOPS[t]]) for t in range(6)])
FOLLOW = 5  # "Paris"
PARIS = SCORES[FOLLOW]
PICKED = TOPS[FOLLOW]
PCT = np.round(100 * GATES[FOLLOW]).astype(int)
assert list(PICKED + 1) == [5, 8]  # "experts five and eight"
assert list(PCT) == [62, 38]  # "sixty-two percent, and thirty-eight percent"
assert len(set(TOPS.ravel())) == 8  # "the six tokens use all eight experts"
assert not np.any(SCORES == 0)

# ---------------------------------------------------------------------------- the toy router (load balancing)

T_TOY, D_TOY, ALPHA_TOY, CAPACITY = 4096, 16, 6.0, 1.25


def toy_router(alpha, steps=200, record=2, seed=1):
    """A linear router (16 → 8, top-2) trained by gradient descent on a skewed routing task
    (cross-entropy toward each token's preferred expert; preferences favour the first experts),
    plus alpha × the Switch Transformer's balancing loss N·Σ f_i·P_i (gradient through P only).
    Returns the share of top-2 assignments per expert every `record` steps, and N·Σ f·P."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (T_TOY, D_TOY))
    teacher = rng.normal(0, 0.4, (D_TOY, N_EXPERTS))
    lean = np.array([1.8, 1.0, 0.5, 0.1, -0.3, -0.7, -1.1, -1.6])
    want = np.argmax(x @ teacher + lean + rng.gumbel(size=(T_TOY, N_EXPERTS)), axis=1)
    target = np.eye(N_EXPERTS)[want]
    w, b = rng.normal(0, 0.05, (D_TOY, N_EXPERTS)), np.zeros(N_EXPERTS)
    lr = 1.0 if alpha == 0 else 0.3
    shares, losses = [], []
    for step in range(steps + 1):
        p = softmax(x @ w + b)
        top = np.argsort(-p, axis=1)[:, :TOP_K]
        f = np.bincount(top.ravel(), minlength=N_EXPERTS) / (T_TOY * TOP_K)
        if step % record == 0:
            shares.append(f)
            losses.append(N_EXPERTS * float(np.sum(f * p.mean(0))))
        grad = (p - target) / T_TOY
        if alpha:
            grad += alpha * N_EXPERTS / T_TOY * p * (f - (p * f).sum(1, keepdims=True))
        w -= lr * x.T @ grad
        b -= lr * grad.sum(0)
    return np.array(shares), np.array(losses)


PLAIN, PLAIN_L = toy_router(0.0)
BAL, BAL_L = toy_router(ALPHA_TOY)
F_PLAIN, F_BAL = PLAIN[-1], BAL[-1]
CAP = CAPACITY / N_EXPERTS
DROP_PLAIN = float(np.clip(F_PLAIN - CAP, 0, None).sum())
DROP_BAL = float(np.clip(F_BAL - CAP, 0, None).sum())
assert F_PLAIN.max() > 1 / 3 and F_PLAIN.min() < 0.01  # "over a third … almost none"
assert round(100 * F_BAL.min()) == 11 and round(100 * F_BAL.max()) == 16  # "between eleven and sixteen percent"
assert DROP_PLAIN > 0.25 and DROP_BAL < 0.01  # "over a quarter … almost none"
assert PLAIN_L[-1] > 1.6 and abs(BAL_L[-1] - 1) < 0.02

# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """kit.recolored(), but written as assets/ep39-<name>-<hex>.svg (this episode adds only ep39- files)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    stem = name if name.startswith("ep39-") else f"ep39-{name}"
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


def formula(*parts, size=40):
    """A MathTex from (latex, colour) pairs, one part each."""
    t = m.MathTex(*[p for p, _ in parts], font_size=size, color=INK)
    for part, (_, c) in zip(t, parts):
        part.set_color(c)
    return t


def cite(s, size=20):
    return text(s, size, MUTED)


def num(v):
    return f"{v:.1f}".replace("-", "−")


def word_chip(word, color=TOKEN, fill=TOKEN_FILL, size=26, height=0.66, pad=0.5):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def plus_node(radius=0.3, color=OUT):
    ring = m.Circle(radius=radius).set_stroke(color, width=3.5).set_fill(OUT_FILL, opacity=1)
    bar = m.VGroup(
        m.Line([-radius * 0.55, 0, 0], [radius * 0.55, 0, 0]), m.Line([0, -radius * 0.55, 0], [0, radius * 0.55, 0])
    ).set_stroke(color, width=4)
    return m.VGroup(ring, bar)


def strike(mob, color=BAD, width=5):
    return m.Line(mob.get_corner(m.DL) + np.array([-0.08, 0.05, 0]), mob.get_corner(m.UR) + np.array([0.08, -0.05, 0])).set_stroke(color, width=width)


def tile_box(label, w, h, size=22, color=EDGE, fill=CARD, ink=MUTED):
    box = card(w, h, color, fill, radius=0.14, stroke=2.5)
    return m.VGroup(box, text(label, size, ink, weight=m.SEMIBOLD).move_to(box))


def light(t, on=True):
    """Restyle an expert tile: lit (violet) or idle."""
    box, label = t
    if on:
        return m.AnimationGroup(box.animate.set_stroke(EXPERT, width=3.5).set_fill(EXPERT_FILL, opacity=1), label.animate.set_color(INK))
    return m.AnimationGroup(box.animate.set_stroke(EDGE, width=2.5).set_fill(CARD, opacity=1), label.animate.set_color(MUTED))


# ---------------------------------------------------------------------------- the episode

EX_X, EX_W, EX_H, EX_STEP, EX_TOP = 0.2, 1.8, 0.48, 0.6, 1.3  # the expert column (router sections)
MID_Y = EX_TOP - EX_STEP * 3.5  # the column's middle: the token and the router sit here
SCORE_X = EX_X - EX_W / 2 - 0.62
GATE_X = EX_X + EX_W / 2 + 0.6
RIGHT_X = 4.6  # centre of the right-hand area (formulas, notes)


def expert_y(i):
    return EX_TOP - EX_STEP * i


class Ep39MixtureOfExperts(Episode):
    number = 39

    def lesson(self):
        self.hook()
        self.idea()
        self.router()
        self.routing()
        self.routing_map()
        self.sparsity()
        self.balancing()
        self.costs()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Mixtral 8x7B (Jiang et al., 2024): {TOTAL_B:.1f}B parameters stored, {ACTIVE_B:.1f}B used per token (computed from its Table 1). Facts checked 2026-10-21.")
        panel = card(6.3, 3.9, EDGE, CARD, stroke=2).move_to([-3.25, -0.45, 0])
        caption = text("one layer, eight experts", 22, MUTED).next_to(panel.get_bottom(), m.UP, buff=0.3)
        tiles = m.VGroup(*[tile_box(f"E{i + 1}", 0.62, 1.0, 22) for i in range(8)]).arrange(m.RIGHT, buff=0.1)
        tiles.move_to(panel.get_center() + 0.35 * m.DOWN)
        token = word_chip("Paris").move_to([panel.get_x(), panel.get_top()[1] - 0.65, 0])
        lit = [tile_box(f"E{i + 1}", 0.62, 1.0, 22, EXPERT, EXPERT_FILL, INK).move_to(tiles[i]) for i in PICKED]
        for t in lit:
            t[0].set_stroke(EXPERT, width=3.5)
        links = m.VGroup(*[arrow(token.get_bottom(), t.get_top(), PICK, buff=0.08) for t in lit])
        links.set_z_index(3)

        big = m.VGroup(text("≈ 47B", 66, EXPERT, weight=m.BOLD), text("parameters stored", 26, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        small = m.VGroup(text("≈ 13B", 66, PICK, weight=m.BOLD), text("used for each token", 26, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        stats = m.VGroup(big, small).arrange(m.DOWN, buff=0.5, aligned_edge=m.LEFT).move_to([3.7, 0.1, 0])
        source = cite("Mixtral 8x7B · Jiang et al., 2024").next_to(stats, m.DOWN, buff=0.4).align_to(stats, m.LEFT)
        name = pill("mixture of experts (MoE): only a few experts wake up per token", PICK, PICK_FILL, 24).move_to([0, -3.25, 0])

        self.say(
            "So... [what if a model could be huge], [yet use only a small part of itself for each word?]",
            m.AnimationGroup(m.FadeIn(panel), m.FadeIn(caption), m.LaggedStart(*[m.FadeIn(t, scale=0.8) for t in tiles], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(token, shift=0.3 * m.DOWN), *[m.GrowArrow(a) for a in links], *[m.FadeIn(t) for t in lit], *[m.FadeOut(tiles[i]) for i in PICKED]),
        )
        self.say(
            "[The open model Mixtral stores about forty-seven billion parameters.] [Yet each token uses only about thirteen billion of them.]",
            m.FadeIn(big, shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(small, shift=0.3 * m.UP), m.FadeIn(source)),
        )
        self.say(
            "[The trick is called a mixture of experts]: [for each token, only a few experts wake up.]",
            m.FadeIn(name, shift=0.2 * m.UP),
            m.AnimationGroup(*[m.Circumscribe(t, color=PICK, buff=0.06) for t in lit]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the idea: dense → sparse

    def idea(self):
        self.next_section("The idea", notes="A Transformer block (Episode 24): attention, then a feed-forward network. An MoE layer replaces the feed-forward network with N separate ones (experts) and a router that picks k per token.")
        y = 0.35
        token = word_chip("Paris").move_to([-6.0, y, 0])
        attn = card(1.85, 1.35, SHARED, CARD, stroke=2.5).move_to([-4.0, y, 0])
        attn_words = m.VGroup(text("Attention", 26, INK, weight=m.BOLD), text("mixes tokens", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to(attn)
        ffn = card(4.25, 2.0, EXPERT, EXPERT_FILL, stroke=2.5).move_to([1.5, y, 0])
        ffn_words = m.VGroup(text("Feed-forward", 30, INK, weight=m.BOLD), text("one big network", 22, MUTED)).arrange(m.DOWN, buff=0.1).move_to(ffn)
        out = word_chip("out", OUT, OUT_FILL).move_to([5.85, y, 0])
        a1 = arrow(token.get_right(), attn.get_left(), MUTED)
        a2 = arrow(attn.get_right(), ffn.get_left(), MUTED)
        a3 = arrow(ffn.get_right(), out.get_left(), MUTED)
        dense = pill("dense: every token uses every weight", SHARED, SKY_FILL, 22).move_to([0, -2.2, 0])

        tiles = m.VGroup(*[tile_box(f"E{i + 1}", 0.95, 0.8, 22) for i in range(8)]).arrange_in_grid(rows=2, buff=(0.15, 0.2))
        tiles.move_to(ffn)
        router = m.VGroup(card(1.1, 1.05, ROUTER, ROUTER_FILL, stroke=2.5), icon("ep39-router", 0.6))
        router[1].move_to(router[0])
        router.move_to([-1.95, y, 0])
        r_label = text("router", 22, ROUTER, weight=m.SEMIBOLD).next_to(router, m.DOWN, buff=0.15)
        b1 = arrow(attn.get_right(), router.get_left(), MUTED)
        b2 = arrow(router.get_right(), tiles.get_left(), ROUTER)
        node_plus = plus_node().move_to([4.4, y, 0])
        b3 = arrow(tiles.get_right(), node_plus.get_left(), MUTED, buff=0.08)
        b4 = arrow(node_plus.get_right(), out.get_left(), MUTED, buff=0.08)
        lit = [tiles[i] for i in PICKED]
        sparse = pill("sparse: each token uses 2 of the 8 experts", PICK, PICK_FILL, 22).move_to([0, -2.2, 0])
        desk = text("like a front desk sending each patient to two specialists, not to every doctor", 22, MUTED).move_to([0, -3.15, 0])

        self.say(
            "[Recall the Transformer block from episode twenty-four]: [attention, then a feed-forward network.]",
            m.AnimationGroup(self.new_header("THE IDEA", "One big block, or many experts?"), m.FadeIn(token, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(attn), m.FadeIn(attn_words), m.GrowArrow(a2), m.FadeIn(ffn), m.FadeIn(ffn_words), m.GrowArrow(a3), m.FadeIn(out)),
        )
        self.say(
            "[In a dense model, every token runs through all of the feed-forward weights.]",
            m.AnimationGroup(m.Succession(pulse(a1, TOKEN), pulse(a2, TOKEN), m.Indicate(ffn, color=EXPERT, scale_factor=1.04), pulse(a3, TOKEN)), m.FadeIn(dense, shift=0.2 * m.UP)),
        )
        self.say(
            "[A mixture of experts layer replaces that block with several separate networks, called experts.]",
            m.AnimationGroup(m.FadeOut(ffn), m.FadeOut(ffn_words), m.LaggedStart(*[m.FadeIn(t, scale=0.6) for t in tiles], lag_ratio=0.08), m.FadeOut(a3), m.GrowArrow(b3), m.FadeIn(node_plus), m.GrowArrow(b4), run_time=1.6),
        )
        self.say(
            "[A small router in front decides which experts each token visits]: [here, two out of eight.]",
            m.AnimationGroup(m.FadeOut(a2), m.GrowArrow(b1), m.FadeIn(router, scale=0.8), m.FadeIn(r_label), m.GrowArrow(b2)),
            m.AnimationGroup(*[light(t) for t in lit], m.FadeOut(dense), m.FadeIn(sparse, shift=0.2 * m.UP)),
        )
        self.say(
            "[It's like a front desk], [sending each patient to a couple of specialists, not to every doctor.]",
            m.FadeIn(desk, shift=0.15 * m.UP),
            m.AnimationGroup(*[m.Circumscribe(t, color=PICK, buff=0.05) for t in lit]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the router

    def router(self):
        self.next_section(
            "The router",
            notes=f"Illustrative router (seed 1593): 6-number token vectors, a 6×8 weight matrix. Scores for 'Paris': {', '.join(num(v) for v in PARIS)}; "
            f"top-2 = experts {PICKED[0] + 1} and {PICKED[1] + 1}; softmax over those two = {PCT[0]}% / {PCT[1]}%. "
            "Gating G(x) = Softmax(TopK(x·W_g)) as in Mixtral (Jiang et al., 2024); noisy top-k gating: Shazeer et al., 2017.",
        )
        sentence = m.VGroup(*[word_chip(w, size=24, height=0.6, pad=0.42) for w in WORDS]).arrange(m.RIGHT, buff=0.14)
        sentence.move_to([-2.6, 2.3, 0])
        token = word_chip("Paris").move_to([-5.75, MID_Y + 0.75, 0])
        vec = vector(TOKEN, [num(v) for v in TOKEN_VECS[FOLLOW][[0, 1, 2, 5]]], 20).next_to(token, m.DOWN, buff=0.3)
        vec_label = text("x", 24, TOKEN, weight=m.BOLD, slant=m.ITALIC).next_to(vec, m.LEFT, buff=0.15)
        box = card(1.6, 1.6, ROUTER, ROUTER_FILL, stroke=2.5).move_to([-3.45, MID_Y, 0])
        box_words = m.VGroup(text("router", 24, INK, weight=m.BOLD), text("linear layer", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to(box)
        router = m.VGroup(box, box_words)
        feed = arrow(token.get_right() + 0.05 * m.DOWN, [box.get_left()[0], token.get_y(), 0], TOKEN)
        experts = m.VGroup(*[tile_box(f"Expert {i + 1}", EX_W, EX_H, 22).move_to([EX_X, expert_y(i), 0]) for i in range(8)])
        cells = m.VGroup()
        for i in range(8):
            c = card(0.95, 0.42, EDGE, CARD, radius=0.1, stroke=2).move_to([SCORE_X, expert_y(i), 0])
            cells.add(m.VGroup(c, code(num(PARIS[i]), 22, ROUTER).move_to(c)))
        fan = m.VGroup(*[m.Line(box.get_right(), c.get_left()).set_stroke(ROUTER, width=2, opacity=0.5) for c in cells])
        fan.set_z_index(-1)
        rest = [i for i in range(8) if i not in PICKED]
        gates = m.VGroup(*[pill(f"{p}%", ROUTER, ROUTER_FILL, 22).move_to([GATE_X, expert_y(i), 0]) for i, p in zip(PICKED, PCT)])

        g = formula(("g(x)", ROUTER), ("=", INK), (r"\mathrm{softmax}\big(", INK), (r"\mathrm{TopK}\big(", PICK), ("x", TOKEN), (r"\,W_r", ROUTER), (r"\big)", PICK), (r"\big)", INK), size=36)
        g.move_to([RIGHT_X, 1.75, 0]).align_to([6.6, 0, 0], m.RIGHT)
        notes = m.VGroup(
            text("x · Wr: one score per expert", 22, ROUTER),
            text("TopK: keep the k largest", 22, PICK),
            text("softmax: weights that sum to 1", 22, INK),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([RIGHT_X, 0.55, 0])
        source = m.VGroup(cite("top-k gating: Shazeer et al., 2017"), cite("k = 2 in Mixtral: Jiang et al., 2024")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        source.move_to([RIGHT_X, -2.75, 0])
        self.col = dict(token=token, vec=vec, vec_label=vec_label, router=router, feed=feed, experts=experts, cells=cells, fan=fan, gates=gates)

        self.say(
            "[Take the sentence: the capital of France is Paris.] [Let's follow its last token through a layer with eight experts.]",
            m.AnimationGroup(self.new_header("THE ROUTER", "One score per expert"), m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in sentence], lag_ratio=0.12)),
            m.AnimationGroup(
                m.FadeIn(token, target_position=sentence[FOLLOW]), m.FadeOut(sentence, shift=0.2 * m.UP),
                m.FadeIn(vec), m.FadeIn(vec_label), m.LaggedStart(*[m.FadeIn(e, shift=0.15 * m.LEFT) for e in experts], lag_ratio=0.06),
            ),
        )
        self.say(
            "[The router is just a linear layer]: [it multiplies the token's vector by a weight matrix], [giving one score per expert.]",
            m.AnimationGroup(m.FadeIn(router, scale=0.85), m.GrowArrow(feed)),
            m.AnimationGroup(pulse(feed, TOKEN), m.FadeIn(g[4]), m.FadeIn(g[5]), m.FadeIn(notes[0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(fan), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in cells], lag_ratio=0.08)),
        )
        self.say(
            "[Top k keeps the two highest scores], [experts five and eight], [and drops the rest.]",
            m.AnimationGroup(m.FadeIn(g[3]), m.FadeIn(g[6]), m.FadeIn(notes[1], shift=0.15 * m.UP)),
            m.AnimationGroup(*[cells[i][0].animate.set_stroke(PICK, width=3.5).set_fill(PICK_FILL, opacity=1) for i in PICKED], *[fan[i].animate.set_stroke(PICK, width=4, opacity=1) for i in PICKED]),
            m.AnimationGroup(*[cells[i].animate.set_opacity(0.3) for i in rest], *[fan[i].animate.set_stroke(opacity=0.12) for i in rest]),
        )
        self.say(
            "[Then a softmax over the two survivors turns them into weights]: [sixty-two percent, and thirty-eight percent.]",
            m.AnimationGroup(m.FadeIn(g[0:3]), m.FadeIn(g[7]), m.FadeIn(notes[2], shift=0.15 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p, target_position=cells[i]) for i, p in zip(PICKED, gates)], lag_ratio=0.4),
        )
        self.say(
            "[So the gate is a softmax over the top k scores.] [This top k gating dates from twenty seventeen], [and Mixtral uses it with k equal to two.]",
            m.Circumscribe(g, color=ROUTER, buff=0.12),
            m.FadeIn(source[0], shift=0.15 * m.UP),
            m.FadeIn(source[1], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.play(*[m.FadeOut(x) for x in (g, notes, source)], run_time=0.6)

    # ------------------------------------------------------------------ top-k routing: only the chosen experts run

    def routing(self):
        self.next_section("Top-k routing", notes="Only the k chosen experts compute; the output is y = Σ over the top k of g_i(x)·E_i(x). Each expert is an ordinary feed-forward network (in Mixtral, a SwiGLU block).")
        c = self.col
        experts, gates, fan = c["experts"], c["gates"], c["fan"]
        rest = [i for i in range(8) if i not in PICKED]
        node_plus = plus_node().move_to([3.15, expert_y(6) + 0.3, 0])
        joins = m.VGroup(*[arrow(p.get_right(), node_plus.get_center(), ROUTER, buff=0.32) for p in gates])
        out = word_chip("y", OUT, OUT_FILL, size=28).next_to(node_plus, m.RIGHT, buff=0.75)
        out_arrow = arrow(node_plus.get_right(), out.get_left(), OUT, buff=0.08)
        out_label = text("output", 22, OUT, weight=m.SEMIBOLD).next_to(out, m.DOWN, buff=0.15)
        idle = text("6 experts idle", 22, MUTED).next_to(experts, m.LEFT, buff=0.15).align_to(experts[1], m.UP)
        idle.move_to([SCORE_X, expert_y(0) + 0.62, 0])

        y = formula(("y", OUT), ("=", INK), (r"\sum_{i \in \mathrm{TopK}}", PICK), (r"g_i(x)", ROUTER), (r"\,E_i(x)", EXPERT), size=44)
        y.move_to([RIGHT_X, 1.55, 0])
        y.align_to([6.6, 0, 0], m.RIGHT)
        terms = m.VGroup(
            text("Σ over TopK: the chosen experts", 22, PICK),
            text("gᵢ(x): its gate value", 22, ROUTER),
            text("Eᵢ(x): its output", 22, EXPERT),
        ).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).next_to(y, m.DOWN, buff=0.35).align_to(y, m.LEFT)
        n_sum, n_gate, n_exp = terms
        self.say(
            "[Only those two experts run.] [The other six do no work at all for this token.]",
            m.AnimationGroup(self.new_header("TOP-K ROUTING", "Only the chosen experts run"), *[light(experts[i]) for i in PICKED], *[m.ShowPassingFlash(m.Line(fan[i].get_start(), fan[i].get_end()).set_stroke(PICK, width=9), time_width=0.7) for i in PICKED]),
            m.AnimationGroup(*[experts[i].animate.set_opacity(0.35) for i in rest], m.FadeIn(idle)),
        )
        self.say(
            "[Each expert is an ordinary feed-forward network], [and their outputs are mixed, sixty-two to thirty-eight.]",
            m.AnimationGroup(*[m.Indicate(experts[i], color=EXPERT, scale_factor=1.06) for i in PICKED]),
            m.AnimationGroup(*[m.GrowArrow(j) for j in joins], m.FadeIn(node_plus, scale=0.6), m.GrowArrow(out_arrow), m.FadeIn(out), m.FadeIn(out_label)),
        )
        self.say(
            "[In symbols], [y is the sum over the chosen experts], [of each one's gate value times its output.]",
            m.Write(y, run_time=1.4),
            m.AnimationGroup(m.Indicate(y[2], color=PICK, scale_factor=1.15), m.FadeIn(n_sum, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Indicate(y[3], color=ROUTER, scale_factor=1.15), m.FadeIn(n_gate, shift=0.1 * m.UP), m.Indicate(y[4], color=EXPERT, scale_factor=1.15), m.FadeIn(n_exp, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the whole sentence

    def routing_map(self):
        pairs = "; ".join(f"{w}: E{TOPS[t][0] + 1} {round(100 * GATES[t][0])}%, E{TOPS[t][1] + 1} {round(100 * GATES[t][1])}%" for t, w in enumerate(WORDS))
        self.next_section("Every token, its own pair", notes=f"Illustrative routing of the whole sentence (same router): {pairs}. Switch Transformer (Fedus et al., 2021) routes each token to one expert (top-1).")
        x0, step, cw, ch = -3.75, 0.84, 0.74, 0.5

        def at(r, i):
            return np.array([x0 + step * i, 1.45 - 0.62 * r, 0])

        heads = m.VGroup(*[text(f"E{i + 1}", 22, EXPERT, weight=m.SEMIBOLD).move_to(at(0, i) + 0.6 * m.UP) for i in range(8)])
        rows = m.VGroup(*[word_chip(w, size=22, height=0.5, pad=0.36).move_to(at(r, 0)).align_to([x0 - 0.72, 0, 0], m.RIGHT) for r, w in enumerate(WORDS)])
        empty = m.VGroup(*[card(cw, ch, EDGE, CARD, radius=0.08, stroke=1.5).move_to(at(r, i)) for r in range(6) for i in range(8)])
        firsts, seconds, labels = m.VGroup(), m.VGroup(), m.VGroup()
        for r in range(6):
            for j, i in enumerate(TOPS[r]):
                g = GATES[r][j]
                cell = card(cw, ch, EXPERT, EXPERT, radius=0.08, stroke=2).set_fill(EXPERT, opacity=0.25 + 0.6 * g).move_to(at(r, i))
                (firsts if j == 0 else seconds).add(cell)
                labels.add(text(f"{round(100 * g)}%", 20, INK, weight=m.BOLD).move_to(cell))
        firsts.set_z_index(1)
        seconds.set_z_index(1)
        labels.set_z_index(2)
        used = len(set(TOPS.ravel()))
        facts = m.VGroup(
            text(f"6 tokens × top-{TOP_K} = 12 visits", 24, INK, weight=m.SEMIBOLD),
            text(f"all {used} experts get work", 24, EXPERT, weight=m.SEMIBOLD),
            text("each token skips 6 of 8", 24, PICK, weight=m.SEMIBOLD),
        ).arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT).move_to([4.75, 0.35, 0])
        switch = m.VGroup(pill("Switch Transformer: top-1, one expert per token", PICK, PICK_FILL, 22), cite("Fedus et al., 2021")).arrange(m.RIGHT, buff=0.3)
        switch.move_to([0, -2.75, 0])
        top1 = text("top-1", 24, PICK, weight=m.BOLD).move_to([4.75, -1.45, 0]).align_to(facts, m.LEFT)

        self.say(
            "[Every token gets its own pair.] [Across our sentence, the six tokens use all eight experts], [while each one skips six of them.]",
            m.AnimationGroup(
                self.new_header("TOP-K ROUTING", "Every token, its own pair"),
                m.FadeIn(heads), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows], lag_ratio=0.1), m.FadeIn(empty),
            ),
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(firsts[r], scale=0.7), m.FadeIn(seconds[r], scale=0.7), m.FadeIn(labels[2 * r]), m.FadeIn(labels[2 * r + 1])) for r in range(6)], lag_ratio=0.25),
                m.FadeIn(facts[0], shift=0.15 * m.UP), m.FadeIn(facts[1], shift=0.15 * m.UP),
            ),
            m.FadeIn(facts[2], shift=0.15 * m.UP),
        )
        self.say(
            "[The Switch Transformer, in twenty twenty-one, went even sparser]: [just one expert per token.]",
            m.FadeIn(switch, shift=0.2 * m.UP),
            m.AnimationGroup(seconds.animate.set_opacity(0.08), m.FadeOut(labels), m.FadeIn(top1, shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ sparsity: total vs active

    def sparsity(self):
        self.next_section(
            "Sparsity",
            notes=f"Mixtral 8x7B from Table 1 (Jiang et al., 2024): one expert = 32 layers × 3 × 4096 × 14336 = {EXPERT_B:.2f}B; shared (attention with 8 KV heads, "
            f"norms, router, embeddings, output layer) = {SHARED_B:.2f}B. Total {TOTAL_B:.2f}B, active (2 experts) {ACTIVE_B:.2f}B. The paper: 47B / 13B, and it "
            "'outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks'. Facts checked 2026-10-21.",
        )
        s, x0, h = 0.24, -5.85, 0.62
        y_total, y_active = 1.0, -0.85
        shared = m.Rectangle(width=SHARED_B * s, height=h).set_stroke(SHARED, width=2).set_fill(SKY_FILL, opacity=1)
        shared.move_to([x0 + SHARED_B * s / 2, y_total, 0])
        segs = m.VGroup()
        for i in range(8):
            r = m.Rectangle(width=EXPERT_B * s, height=h).set_stroke(EXPERT, width=2).set_fill(EXPERT_FILL, opacity=1)
            r.move_to([x0 + (SHARED_B + EXPERT_B * (i + 0.5)) * s, y_total, 0])
            segs.add(m.VGroup(r, text(f"E{i + 1}", 20, INK, weight=m.SEMIBOLD).move_to(r)))
        a_shared = shared.copy().set_y(y_active)
        a_segs = m.VGroup()
        for k, i in enumerate(PICKED):
            r = m.Rectangle(width=EXPERT_B * s, height=h).set_stroke(PICK, width=2.5).set_fill(PICK_FILL, opacity=1)
            r.move_to([x0 + (SHARED_B + EXPERT_B * (k + 0.5)) * s, y_active, 0])
            a_segs.add(m.VGroup(r, text(f"E{i + 1}", 20, INK, weight=m.SEMIBOLD).move_to(r)))
        end_total = x0 + TOTAL_B * s
        end_active = x0 + ACTIVE_B * s
        stored = text("stored in memory", 24, MUTED, weight=m.SEMIBOLD).move_to([0, y_total + 0.7, 0]).align_to([x0, 0, 0], m.LEFT)
        total_v = text(f"{TOTAL_B:.1f}B ≈ 47B", 30, EXPERT, weight=m.BOLD).move_to([0, y_total + 0.72, 0]).align_to([end_total, 0, 0], m.RIGHT)
        used = text("used for each token", 24, MUTED, weight=m.SEMIBOLD).move_to([0, y_active + 0.7, 0]).align_to([x0, 0, 0], m.LEFT)
        active_v = text(f"{ACTIVE_B:.1f}B ≈ 13B", 30, PICK, weight=m.BOLD).next_to([end_active, y_active, 0], m.RIGHT, buff=0.35)
        naive = text("8 × 7B = 56B ?", 28, INK, weight=m.BOLD).move_to([0.9, y_total + 0.72, 0])
        cross = text("✗", 34, BAD, weight=m.BOLD).next_to(naive, m.RIGHT, buff=0.25)
        spec = code("8 experts per layer · top-2 · 32 layers", 22, MUTED).move_to([2.6, 2.3, 0])
        legend = m.VGroup(
            m.VGroup(m.Square(0.26).set_stroke(SHARED, width=2).set_fill(SKY_FILL, 1), text(f"shared: attention, embeddings, norms, router · {SHARED_B:.1f}B", 22, MUTED)).arrange(m.RIGHT, buff=0.2),
            m.VGroup(m.Square(0.26).set_stroke(EXPERT, width=2).set_fill(EXPERT_FILL, 1), text(f"one expert: 32 layers × 3 × 4096 × 14336 ≈ {EXPERT_B:.1f}B", 22, MUTED)).arrange(m.RIGHT, buff=0.2),
        ).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([0, -1.85, 0]).align_to([x0, 0, 0], m.LEFT)
        source = cite("Mixtral 8x7B · Jiang et al., 2024").move_to([0, -2.05, 0]).align_to([6.6, 0, 0], m.RIGHT)
        compute = pill("compute per token ≈ a 13B dense model", GOOD, GOOD_FILL, 22).move_to([0, -2.85, 0]).align_to([x0, 0, 0], m.LEFT)
        bench = text("quality: matches or beats Llama 2 70B on all evaluated benchmarks", 22, MUTED).next_to(compute, m.DOWN, buff=0.2).align_to(compute, m.LEFT)

        self.say(
            "[Now the payoff: sparsity.] [Mixtral has eight experts in every layer, and picks two for each token.]",
            self.new_header("SPARSITY", "Total versus active parameters"),
            m.FadeIn(spec, shift=0.15 * m.DOWN),
        )
        self.say(
            "[You might expect eight times seven: fifty-six billion parameters.] [But only the feed-forward blocks come in eights]; [attention and embeddings are shared.]",
            m.FadeIn(naive, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(cross, scale=1.4), naive.animate.set_color(MUTED), m.LaggedStart(*[m.FadeIn(g, shift=0.15 * m.RIGHT) for g in segs], lag_ratio=0.1), m.FadeIn(legend[1])),
            m.AnimationGroup(m.FadeIn(shared, scale=0.6), m.FadeIn(legend[0]), m.FadeIn(source)),
        )
        self.say(
            "[So it stores about forty-seven billion parameters], [yet each token runs through only about thirteen billion.]",
            m.AnimationGroup(m.FadeOut(naive), m.FadeOut(cross), m.FadeIn(stored), m.FadeIn(total_v, shift=0.15 * m.UP)),
            m.AnimationGroup(
                m.FadeIn(used), m.FadeIn(a_shared, target_position=shared),
                *[m.FadeIn(a, target_position=segs[i]) for a, i in zip(a_segs, PICKED)], m.FadeIn(active_v, shift=0.15 * m.UP),
            ),
        )
        self.say(
            "[Its compute per token is roughly that of a thirteen billion dense model.]",
            m.FadeIn(compute, shift=0.2 * m.UP),
        )
        self.say(
            "[Yet its authors report it matching or beating Llama 2, a seventy billion dense model, on every benchmark they tested.]",
            m.FadeIn(bench, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ load balancing

    def histogram(self, x, title, color, shares, losses):
        """A panel with a bar chart of each expert's share of the router's top-2 assignments,
        redrawn from a training-step tracker, and a live N·Σ f·P readout."""
        box = card(6.1, 3.9, color, CARD, stroke=2).move_to([x, -1.25, 0])
        left, right, top, bottom = box.get_left()[0], box.get_right()[0], box.get_top()[1], box.get_bottom()[1]
        base, scale = bottom + 0.62, 2.4 / 0.4
        x_axis = left + 1.12
        slots = [x_axis + 0.36 + 0.57 * i for i in range(8)]
        name = text(title, 24, color, weight=m.BOLD).move_to([0, top - 0.38, 0]).align_to([left + 0.3, 0, 0], m.LEFT)
        frame = m.VGroup(
            m.Line([x_axis, base, 0], [right - 0.3, base, 0]).set_stroke(FAINT, width=2.5),
            m.Line([x_axis, base, 0], [x_axis, base + 0.4 * scale, 0]).set_stroke(FAINT, width=2.5),
        )
        ticks = m.VGroup(*[text(s, 20, MUTED).move_to([0, base + v * scale, 0]).align_to([x_axis - 0.14, 0, 0], m.RIGHT) for s, v in (("0", 0), ("25%", 0.25))])
        even_tick = text("12.5%", 20, INK).move_to([0, base + 0.125 * scale, 0]).align_to([x_axis - 0.14, 0, 0], m.RIGHT)
        even = m.DashedLine([x_axis, base + 0.125 * scale, 0], [right - 0.3, base + 0.125 * scale, 0], dash_length=0.1).set_stroke(INK, width=2, opacity=0.7)
        even.set_z_index(3)
        names = m.VGroup(*[text(str(i + 1), 20, MUTED).move_to([sx, base - 0.24, 0]) for i, sx in enumerate(slots)])
        step = m.ValueTracker(0.0)

        def share_at(t):
            lo = int(np.clip(np.floor(t), 0, len(shares) - 1))
            hi = min(lo + 1, len(shares) - 1)
            a = float(np.clip(t - lo, 0, 1))
            return (1 - a) * shares[lo] + a * shares[hi]

        def draw():
            f = share_at(step.get_value())
            return m.VGroup(*[
                m.Rectangle(width=0.38, height=max(v * scale, 0.02)).set_stroke(color, width=2).set_fill(color, opacity=0.55).move_to([sx, base + max(v * scale, 0.02) / 2, 0])
                for sx, v in zip(slots, f)
            ]).set_z_index(1)

        bars = m.always_redraw(draw)
        label = text("N·Σ f·P =", 22, MUTED).move_to([0, top - 0.95, 0])
        value = m.DecimalNumber(float(losses[0]), num_decimal_places=2, mob_class=Mono, font_size=24, color=INK)
        value.add_updater(lambda d: d.set_value(float(np.interp(step.get_value(), np.arange(len(losses)), losses))))
        value.add_updater(lambda d: d.next_to(label, m.RIGHT, buff=0.12))
        label.align_to([right - 1.15, 0, 0], m.RIGHT)
        static = m.VGroup(box, name, frame, ticks, even_tick, even, names, label)

        def bar_at(i, f):
            return m.Rectangle(width=0.38, height=max(f[i] * scale, 0.02)).move_to([slots[i], base + max(f[i] * scale, 0.02) / 2, 0])

        return dict(static=static, bars=bars, value=value, step=step, end=len(shares) - 1, bar_at=bar_at, base=base, scale=scale, slots=slots,
                    x_axis=x_axis, right=right, top=top, even=even)

    def balancing(self):
        self.next_section(
            "Load balancing",
            notes=f"Toy router, illustrative: {T_TOY} random 16-number tokens, 8 experts, top-2, trained 200 steps on a lopsided routing task (each token's preferred expert "
            f"leans toward the first experts). Without the balancing loss: shares {', '.join(f'{100 * v:.1f}' for v in F_PLAIN)}%, N·Σf·P = {PLAIN_L[-1]:.2f}. "
            f"With α = {ALPHA_TOY:g} (toy scale; Switch used 0.01 next to the language-model loss): {', '.join(f'{100 * v:.1f}' for v in F_BAL)}%, {BAL_L[-1]:.2f}. "
            f"Capacity factor {CAPACITY} (cap {100 * CAP:.1f}% of assignments per expert): {100 * DROP_PLAIN:.1f}% vs {100 * DROP_BAL:.1f}% dropped. "
            "Loss: Fedus et al., 2021 (Switch Transformer), f_i = fraction of tokens routed to expert i, P_i = mean router probability; f is not differentiable, so the gradient flows through P. "
            "Self-reinforcing imbalance: Shazeer et al., 2017.",
        )
        loss = formula((r"\mathcal{L}_{\text{aux}}", INK), ("=", INK), (r"\alpha", PICK), (r"\cdot N \cdot", INK), (r"\sum_{i=1}^{N}", INK), ("f_i", TOKEN), (r"\cdot", INK), ("P_i", ROUTER), size=42)
        loss.move_to([-3.0, 1.95, 0])
        source = cite("Switch Transformer · Fedus et al., 2021").move_to([-3.0, 0.98, 0])
        n_f = text("fᵢ: share of tokens sent to expert i", 22, TOKEN)
        n_p = text("Pᵢ: its mean router probability (carries the gradient)", 22, ROUTER)
        n_a = text("α: a small weight", 22, PICK)
        m.VGroup(n_f, n_p, n_a).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).move_to([0, 1.8, 0]).align_to([0.2, 0, 0], m.LEFT)
        lhs = self.histogram(-3.35, "without balancing loss", BAD, PLAIN, PLAIN_L)
        rhs = self.histogram(3.35, "with balancing loss", GOOD, BAL, BAL_L)
        hi, lo = int(np.argmax(F_PLAIN)), int(np.argmin(F_PLAIN))
        cap_lines, drops, notes = m.VGroup(), m.VGroup(), m.VGroup()
        for side, f, dropped in ((lhs, F_PLAIN, DROP_PLAIN), (rhs, F_BAL, DROP_BAL)):
            yc = side["base"] + CAP * side["scale"]
            cap_lines.add(m.Line([side["x_axis"], yc, 0], [side["right"] - 0.3, yc, 0]).set_stroke(AMBER, width=3).set_z_index(4))
            parts = m.VGroup()
            for i, v in enumerate(f):
                if v > CAP:
                    hgt = (v - CAP) * side["scale"]
                    parts.add(m.Rectangle(width=0.38, height=hgt).set_stroke(AMBER, width=2).set_fill(AMBER, opacity=0.85).move_to([side["slots"][i], yc + hgt / 2, 0]))
            drops.add(parts.set_z_index(5))
            notes.add(text(f"capacity · {100 * dropped:.1f}% dropped", 22, AMBER, weight=m.SEMIBOLD).move_to([0, side["top"] - 1.45, 0]).align_to([side["right"] - 0.3, 0, 0], m.RIGHT))

        self.say(
            "[One catch: the router can fall into a rut]: favoured experts get more tokens, learn faster, and get favoured even more.",
            m.AnimationGroup(self.new_header("LOAD BALANCING", "Don't crowd one expert"), m.FadeIn(lhs["static"]), m.FadeIn(lhs["bars"]), m.FadeIn(lhs["value"])),
        )
        self.say(
            "[For example, train a toy router on a lopsided task, without help]: [over a third of the traffic piles onto one expert], [and another gets almost none.]",
            lhs["step"].animate(run_time=3.0, rate_func=m.smooth).set_value(lhs["end"]),
            m.Circumscribe(lhs["bar_at"](hi, F_PLAIN), color=BAD, buff=0.08),
            m.Circumscribe(lhs["bar_at"](lo, F_PLAIN), color=BAD, buff=0.08),
        )
        self.say(
            "[The fix is an auxiliary loss, from the Switch Transformer.] [f is the share of tokens each expert receives], [and P is its average router probability, which carries the gradient.]",
            m.AnimationGroup(m.Write(loss, run_time=1.5), m.FadeIn(source)),
            m.AnimationGroup(m.Indicate(loss[5], color=TOKEN, scale_factor=1.3), m.FadeIn(n_f, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Indicate(loss[7], color=ROUTER, scale_factor=1.3), m.FadeIn(n_p, shift=0.1 * m.UP)),
        )
        self.say(
            "[Summed over the experts, their product is smallest when the load is even], [and alpha keeps it a small part of the total loss.]",
            m.AnimationGroup(m.Indicate(loss[4], color=INK, scale_factor=1.15), m.Indicate(lhs["even"], color=INK)),
            m.AnimationGroup(m.Indicate(loss[2], color=PICK, scale_factor=1.4), m.FadeIn(n_a, shift=0.1 * m.UP)),
        )
        self.say(
            "[Train the same router with it], [and the load stays close to even]: [every expert gets between eleven and sixteen percent.]",
            m.AnimationGroup(m.FadeIn(rhs["static"]), m.FadeIn(rhs["bars"]), m.FadeIn(rhs["value"])),
            rhs["step"].animate(run_time=2.5, rate_func=m.smooth).set_value(rhs["end"]),
            m.Circumscribe(m.VGroup(*[rhs["bar_at"](i, F_BAL) for i in range(8)]), color=GOOD, buff=0.1),
        )
        self.stop(lhs["bars"], lhs["value"], rhs["bars"], rhs["value"])
        self.say(
            "[Many systems also cap each expert's capacity.] [Then the lopsided router would drop over a quarter of its assignments], [the balanced one almost none.]",
            m.Create(cap_lines),
            m.AnimationGroup(m.FadeIn(drops[0]), m.FadeIn(notes[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(drops[1]), m.FadeIn(notes[1], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the costs

    def costs(self):
        self.next_section(
            "The costs",
            notes=f"Memory: every expert must be resident; Mixtral {TOTAL_B:.1f}B × 2 bytes ≈ {GB_16BIT:.0f} GB in 16-bit. Communication: expert parallelism "
            "puts experts on different devices, so tokens are exchanged all-to-all before and after each MoE layer (GShard, Lepikhin et al., 2020; Switch). "
            "Specialisation: Mixtral's routing analysis found no obvious topic pattern (ArXiv, PubMed, PhilPapers looked alike), but syntactic regularities: "
            "indentation tokens in code always went to the same experts, and 'self' in Python often shared an expert (Jiang et al., 2024, Section 5).",
        )
        xs = (-4.45, 0.0, 4.45)

        def frame(x, glyph, title, color):
            box = card(4.15, 4.6, color, CARD, stroke=2).move_to([x, -0.2, 0])
            head = m.VGroup(icon(glyph, 0.5), text(title, 28, color, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25)
            head.move_to(box.get_top() + 0.5 * m.DOWN)
            return box, head

        def lines(x, y, *rows):
            g = m.VGroup(*[text(s, 22, c, weight=w) for s, c, w in rows]).arrange(m.DOWN, buff=0.14)
            return g.move_to([x, y, 0])

        # memory
        m_box, m_head = frame(xs[0], "ep39-memory", "Memory", EXPERT)
        blocks = m.VGroup(*[tile_box(f"E{i + 1}", 0.72, 0.5, 20, EXPERT, EXPERT_FILL, INK) for i in range(8)]).arrange_in_grid(rows=2, buff=(0.12, 0.16))
        blocks.move_to([xs[0], 0.55, 0])
        m_text = lines(xs[0], -1.4, ("every expert is loaded", INK, m.SEMIBOLD), (f"Mixtral: ≈ {GB_16BIT:.0f} GB in 16-bit", MUTED, m.NORMAL), ("a token uses 2 of 8", MUTED, m.NORMAL))
        # communication
        c_box, c_head = frame(xs[1], "ep39-gpu", "Communication", TOKEN)
        gpus = m.VGroup(*[
            m.VGroup(card(1.35, 0.82, TOKEN, TOKEN_FILL, radius=0.14, stroke=2.5), m.VGroup(text(f"GPU {g + 1}", 20, INK, weight=m.SEMIBOLD), text(f"E{2 * g + 1} · E{2 * g + 2}", 20, MUTED)).arrange(m.DOWN, buff=0.07))
            for g in range(4)
        ])
        for g in gpus:
            g[1].move_to(g[0])
        gpus.arrange_in_grid(rows=2, buff=(0.6, 0.4)).move_to([xs[1], 0.3, 0]).set_z_index(2)
        ends = [
            (gpus[0].get_right(), gpus[1].get_left()), (gpus[2].get_right(), gpus[3].get_left()),
            (gpus[0].get_bottom(), gpus[2].get_top()), (gpus[1].get_bottom(), gpus[3].get_top()),
            (gpus[0].get_corner(m.DR), gpus[3].get_corner(m.UL)), (gpus[1].get_corner(m.DL), gpus[2].get_corner(m.UR)),
        ]
        links = m.VGroup(*[m.DoubleArrow(a, b, buff=0.04, stroke_width=3, tip_length=0.13, color=TOKEN) for a, b in ends]).set_z_index(1)
        c_text = lines(xs[1], -1.4, ("tokens travel to their experts", INK, m.SEMIBOLD), ("and back, at every", MUTED, m.NORMAL), ("expert layer", MUTED, m.NORMAL))
        # not topic experts
        t_box, t_head = frame(xs[2], "ep39-router", "Specialists?", ROUTER)
        topic = pill("biology expert", EXPERT, EXPERT_FILL, 22).move_to([xs[2], 1.05, 0])
        no = text("✗", 34, BAD, weight=m.BOLD).next_to(topic, m.RIGHT, buff=0.2)
        snippet = m.VGroup(m.Text("····return x", font="Menlo", font_size=24, color=INK))
        snippet[0][0:4].set_color(ROUTER)
        snippet.move_to([xs[2], 0.3, 0])
        same = text("indentation → same experts", 22, ROUTER, weight=m.SEMIBOLD).next_to(snippet, m.DOWN, buff=0.18)
        t_text = lines(xs[2], -1.3, ("routing looks syntactic,", INK, m.SEMIBOLD), ("not topical", INK, m.SEMIBOLD))
        t_cite = cite("Jiang et al., 2024").next_to(t_text, m.DOWN, buff=0.2)
        deal = pill("the deal: big-model capacity, small-model compute per token", GOOD, GOOD_FILL, 22).move_to([0, -3.3, 0])
        lit = [blocks[i] for i in PICKED]
        idle = [b for i, b in enumerate(blocks) if i not in PICKED]

        self.say(
            "[Now the costs.] [First, memory]: [every expert must be loaded.] [Mixtral needs about ninety-three gigabytes in sixteen-bit.]",
            self.new_header("THE COSTS", "Memory, communication, murky experts"),
            m.AnimationGroup(m.FadeIn(m_box), m.FadeIn(m_head, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(b, scale=0.7) for b in blocks], lag_ratio=0.08), m.FadeIn(m_text[0])),
            m.AnimationGroup(m.FadeIn(m_text[1:]), *[b.animate.set_opacity(0.45) for b in idle], *[b[0].animate.set_stroke(PICK, width=3.5) for b in lit]),
        )
        self.say(
            "[Second, communication.] [Big models spread their experts across many GPUs], [so tokens travel to their experts and back], [at every expert layer.]",
            m.AnimationGroup(m.FadeIn(c_box), m.FadeIn(c_head, shift=0.15 * m.DOWN)),
            m.LaggedStart(*[m.FadeIn(g, scale=0.8) for g in gpus], lag_ratio=0.15),
            m.AnimationGroup(m.Create(links), m.FadeIn(c_text[0])),
            m.AnimationGroup(m.FadeIn(c_text[1:]), *[m.ShowPassingFlash(m.Line(l.get_start(), l.get_end()).set_stroke(INK, width=6), time_width=0.6) for l in links]),
        )
        self.say(
            "[And third, a caution]: [experts are rarely neat specialists in topics like biology or law.]",
            m.AnimationGroup(m.FadeIn(t_box), m.FadeIn(t_head, shift=0.15 * m.DOWN)),
            m.Succession(m.FadeIn(topic, scale=0.8), m.AnimationGroup(m.FadeIn(no, scale=1.4), topic.animate.set_opacity(0.5))),
        )
        self.say(
            "[In Mixtral, the authors found no obvious topic pattern.] [Routing looked more syntactic.] [For example, indentation in code kept going to the same experts.]",
            m.FadeIn(t_cite),
            m.FadeIn(t_text, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(snippet), m.FadeIn(same, shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[Still, the deal is often worth it]: [the capacity of a big model, at the compute per token of a much smaller one.]",
            m.FadeIn(deal, shift=0.2 * m.UP),
            m.Circumscribe(deal, color=GOOD, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Score, pick, balance.")
        halo = glow(PICK, radius=4.0)
        words = on_baseline(m.VGroup(*[text(w, 62, c, weight=m.BOLD) for w, c in zip(("Score", "Pick", "Balance"), (ROUTER, PICK, GOOD))]).arrange(m.RIGHT, buff=1.3))
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Store a huge model; wake only the experts each token's router picks.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [score], [pick], [balance.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Store a huge model, and wake only the few experts each token's router picks.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
