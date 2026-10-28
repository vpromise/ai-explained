"""AI, Explained · Episode 37 · Efficient Attention

Running example: serving one long chat (32,000 tokens) with an illustrative 7B-class model:
32 layers, 32 attention heads of 128 dimensions, 16-bit numbers (2 bytes), 14 GB of weights,
on an 80 GB GPU. Every number the narration quotes is computed below and asserted:

- the n x n score matrix at n = 32,000: 1.02 billion scores, 2.05 GB per head per layer;
- the KV cache, 2 x layers x kv_heads x head_dim x tokens x bytes: 524,288 bytes per token,
  16.8 GB for the chat; 4.2 GB with 8 KV heads (grouped-query), 0.52 GB with 1 (multi-query);
- how many such chats fit beside the weights: 3, 15 and 125;
- FlashAttention's online softmax, run on one row of a seeded 12 x 12 attention matrix in
  three tiles of four: the running max rises every tile, the result equals exact softmax
  attention to 2.2e-16.

Facts (checked 2026-10-21 against the papers): Vaswani et al., 2017 (attention); Dao et al.,
2022, arXiv:2205.14135 (FlashAttention: tiling, online softmax, recomputation in the backward
pass, O(N) extra memory, A100 HBM 40-80 GB at 1.5-2.0 TB/s and SRAM ~19 TB/s, 192 KB x 108
SMs ~ 20 MB, GPT-2 training 3x faster); Milakov & Gimelshein, 2018 (online softmax); Shazeer,
2019, arXiv:1911.02150 (multi-query attention: decoding is memory-bandwidth bound, minor
quality loss); Ainslie et al., 2023, arXiv:2305.13245 (grouped-query attention: quality close
to multi-head at speed close to multi-query, uptrained with 5% of pre-training compute).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

QUERY, QUERY_FILL = TEAL, TEAL_FILL  # as in Episode 24
KEY, KEY_FILL = AMBER, AMBER_FILL
VALUE, VALUE_FILL = PINK, PINK_FILL
HBM_C = BLUE  # main memory
SRAM_C = LIME  # on-chip memory
MHA_C, MQA_C, GQA_C = BLUE, VIOLET, GOOD

# ---------------------------------------------------------------------------- the numbers

N_CHAT = 32_000
LAYERS, HEADS, HEAD_DIM, BYTES = 32, 32, 128, 2
GQA_HEADS, MQA_HEADS = 8, 1
GPU_GB, PARAMS = 80, 7e9


def cache_bytes(kv_heads, tokens=N_CHAT):
    """2 (keys and values) x layers x KV heads x head size x tokens x bytes per number."""
    return 2 * LAYERS * kv_heads * HEAD_DIM * tokens * BYTES


PER_TOKEN = cache_bytes(HEADS, 1)
CACHE_GB = {h: cache_bytes(h) / 1e9 for h in (HEADS, GQA_HEADS, MQA_HEADS)}
SCORES = N_CHAT**2
SCORE_GB = SCORES * BYTES / 1e9
WEIGHTS_GB = PARAMS * BYTES / 1e9
FIT = {h: int((GPU_GB - WEIGHTS_GB) // CACHE_GB[h]) for h in CACHE_GB}

assert PER_TOKEN == 524_288 and round(PER_TOKEN / 1e6, 1) == 0.5  # "about half a megabyte per token"
assert f"{CACHE_GB[HEADS]:.1f}" == "16.8" and 16.5 < CACHE_GB[HEADS] < 17  # "almost seventeen gigabytes"
assert round(CACHE_GB[GQA_HEADS]) == 4 and HEADS // GQA_HEADS == 4  # "about four", "groups of four"
assert cache_bytes(HEADS) == 32 * cache_bytes(MQA_HEADS)  # "thirty-two times"
assert 1.0e9 < SCORES < 1.05e9 and round(SCORE_GB) == 2  # "about a billion scores", "two gigabytes"
assert WEIGHTS_GB == 14 and FIT == {32: 3, 8: 15, 1: 125}  # "three", "fifteen", "a hundred and twenty-five"

# a small attention head for the pictures: 12 tokens, head size 8, seeded
N, D, TILE = 12, 8, 4
_rng = np.random.default_rng(41)
QD = _rng.normal(0, 1, (N, D))
KD = _rng.normal(0, 1, (N, D))
VD = _rng.normal(0, 1, (N, 4))
SD = QD @ KD.T / np.sqrt(D)
CAUSAL = np.tril(np.ones((N, N), dtype=bool))
_e = np.where(CAUSAL, np.exp(SD - SD.max(axis=1, keepdims=True)), 0.0)
WD = _e / _e.sum(axis=1, keepdims=True)
assert np.allclose(WD.sum(axis=1), 1)


def online_row(s, v, tile=TILE):
    """FlashAttention's online softmax for one query row: a running max m, a running sum l of
    exp(s - m) and a running output o, all rescaled by exp(m_old - m_new) when the max rises."""
    m_run, l_run, o_run, steps = -np.inf, 0.0, np.zeros(v.shape[1]), []
    for j in range(0, len(s), tile):
        sj, vj = s[j : j + tile], v[j : j + tile]
        m_new = max(m_run, float(sj.max()))
        scale = np.exp(m_run - m_new)
        l_run = scale * l_run + float(np.exp(sj - m_new).sum())
        o_run = scale * o_run + np.exp(sj - m_new) @ vj
        m_run = m_new
        steps.append((m_run, l_run))
    return o_run / l_run, steps


ROW = N - 1  # the last query sees all twelve keys
O_EXACT = WD[ROW] @ VD
O_FLASH, STEPS = online_row(SD[ROW], VD)
ERR = float(np.abs(O_FLASH - O_EXACT).max())
assert ERR < 1e-12  # "matches standard attention to within rounding"
assert STEPS[0][0] < STEPS[1][0] < STEPS[2][0]  # the max rises on every tile: the rescaling is used


# ---------------------------------------------------------------------------- helpers


def mix(a, b, t):
    """A colour between two hex colours."""
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


EMPTY = "#25314F"  # a score cell not computed yet
MASKED = "#0E1426"  # a masked (future) cell
LIME_FILL = mix(BG, LIME, 0.14)


def heat(t):
    return mix(EMPTY, VIOLET, 0.12 + 0.88 * float(t))


RAW_T = (SD - SD.min()) / (SD.max() - SD.min())
W_T = (WD / WD.max()) ** 0.6


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def grid_points(center, cell, gap, tile_gap=0.0, n=N):
    step = cell + gap
    offs = [j * step + (j // TILE) * tile_gap for j in range(n)]
    width = offs[-1] + cell
    x0 = center[0] - width / 2 + cell / 2
    y0 = center[1] + width / 2 - cell / 2
    return {(i, j): np.array([x0 + offs[j], y0 - offs[i], 0.0]) for i in range(n) for j in range(n)}


def cells(points, cell, color=EMPTY):
    return {k: m.Square(side_length=cell).set_fill(color, 1).set_stroke(width=0).move_to(p) for k, p in points.items()}


def tex_row(sym, desc, color, sym_x, desc_x, y, sym_size=34):
    s = tex(sym, sym_size, color).move_to([sym_x, y, 0])
    d = text(desc, 22, MUTED)
    d.move_to([desc_x + d.width / 2, y, 0])
    return m.VGroup(s, d)


def cite(s):
    return text(s, 20, MUTED)


def kv_head(x, y=0.5, size=0.24):
    k = m.Square(side_length=size).set_fill(KEY_FILL, 1).set_stroke(KEY, width=2).move_to([x, y + size / 2 + 0.01, 0])
    v = m.Square(side_length=size).set_fill(VALUE_FILL, 1).set_stroke(VALUE, width=2).move_to([x, y - size / 2 - 0.01, 0])
    return m.VGroup(k, v)


def bar(width, height, color, fill_opacity=0.9):
    return m.Rectangle(width=max(width, 0.02), height=height, stroke_width=0).set_fill(color, opacity=fill_opacity)


# ---------------------------------------------------------------------------- the episode


class Ep37EfficientAttention(Episode):
    number = 37

    def lesson(self):
        self.hook()
        self.quadratic()
        self.bottleneck()
        self.kv_cache()
        self.flash()
        self.sharing()
        self.big_picture()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Serving one long chat: 32,000 tokens, each new token looks back at all of them; the cost is memory traffic.")
        win = card(6.4, 5.6, EDGE, CARD).move_to([-3.6, -0.55, 0])
        lights = m.VGroup(*[m.Circle(radius=0.075).set_fill(c, 1).set_stroke(width=0) for c in (BAD, AMBER, GOOD)]).arrange(m.RIGHT, buff=0.13)
        lights.move_to(win.get_corner(m.UL) + np.array([0.5, -0.35, 0]))
        title = text("Assistant", 22, MUTED, weight=m.SEMIBOLD).next_to(lights, m.RIGHT, buff=0.3)
        rule = m.Line(win.get_left() + 0.0 * m.RIGHT, win.get_right()).set_stroke(EDGE, width=2).set_y(win.get_top()[1] - 0.68)
        frame_parts = m.VGroup(win, lights, title, rule)

        msgs = (
            ("Here are all my meeting notes.", "right"),
            ("Got it. I've read them all.", "left"),
            ("What did we decide on the launch?", "right"),
        )
        bubbles = []
        for k, (words, side) in enumerate(msgs):
            color, fill = (SKY, SKY_FILL) if side == "right" else (VIOLET, VIOLET_FILL)
            b = bubble(text(words, 22, INK), color, fill, tail=side)
            b.move_to([0, 0.95 - 1.12 * k, 0])
            if side == "right":
                b.align_to(win, m.RIGHT).shift(0.3 * m.LEFT)
            else:
                b.align_to(win, m.LEFT).shift(0.7 * m.RIGHT)
            bubbles.append(b)
        reply = bubble(text("You agreed to delay it to June.", 22, INK), VIOLET, VIOLET_FILL, tail="left")
        reply.move_to([0, 0.95 - 1.12 * 3, 0]).align_to(win, m.LEFT).shift(0.7 * m.RIGHT)
        gutter = win.get_left()[0] + 0.36
        look = arrow([gutter, reply.get_y(), 0], [gutter, bubbles[0].get_top()[1], 0], VIOLET, buff=0.0)

        tokens = m.ValueTracker(0)
        count = m.DecimalNumber(0, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=72, color=INK)
        count_at = np.array([3.5, 1.3, 0])
        count.move_to(count_at)
        count.add_updater(lambda d: d.set_value(int(round(tokens.get_value()))).move_to(count_at))
        count_label = text("tokens in the conversation", 24, MUTED).move_to([3.5, 2.15, 0])

        gpu = icon("ep37-gpu", 1.3).move_to([1.0, -0.5, 0])
        row1 = pill("arithmetic: plenty to spare", GOOD, GOOD_FILL, 20)
        row2 = pill("memory traffic: the bottleneck", BAD, BAD_FILL, 20)
        for r, y in ((row1, -0.15), (row2, -0.85)):
            r.move_to([2.0 + r.width / 2, y, 0])
        ideas = m.VGroup(
            pill("1 · KV cache", KEY, KEY_FILL, 22),
            pill("2 · FlashAttention", SRAM_C, LIME_FILL, 22),
            pill("3 · shared K/V heads", QUERY, QUERY_FILL, 22),
        )
        for r, y in zip(ideas, (-1.95, -2.6, -3.25)):
            r.move_to([2.0 + r.width / 2, y, 0])
        ideas_label = text("three fixes", 22, MUTED, weight=m.SEMIBOLD).move_to([1.0, -1.95, 0])

        self.say(
            "So... [picture a chatbot in one very long conversation.]",
            m.AnimationGroup(m.FadeIn(frame_parts), m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.UP) for b in bubbles], lag_ratio=0.3), run_time=1.6),
        )
        self.say(
            "[The chat now holds thirty-two thousand tokens], [and the model is writing its reply.]",
            m.AnimationGroup(m.FadeIn(count_label), m.FadeIn(count), tokens.animate(run_time=2.2, rate_func=m.smooth).set_value(N_CHAT)),
            m.AnimationGroup(m.FadeIn(reply[:2]), m.AddTextLetterByLetter(reply[2], time_per_char=0.04)),
        )
        self.say(
            "[Every new token has to look back at all of them.]",
            m.AnimationGroup(m.GrowArrow(look), m.LaggedStart(*[m.Circumscribe(b[1], color=VIOLET, buff=0.06) for b in reversed(bubbles)], lag_ratio=0.3)),
        )
        self.say(
            "[On a GPU, the hard part isn't the arithmetic.] [It's moving memory.]",
            m.AnimationGroup(m.FadeIn(gpu, scale=0.8), m.FadeIn(row1, shift=0.2 * m.LEFT)),
            m.Succession(m.FadeIn(row2, shift=0.2 * m.LEFT), m.Wiggle(row2, scale_value=1.06)),
        )
        self.say(
            "Let's see why, [and three ideas that fix it.]",
            m.AnimationGroup(m.FadeIn(ideas_label), m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.LEFT) for p in ideas], lag_ratio=0.3)),
        )
        self.wait(0.3)
        self.stop(count)
        self.clear()

    # ------------------------------------------------------------------ the n x n matrix

    def quadratic(self):
        self.next_section(
            "The n × n matrix",
            notes="S = QKᵀ/√d (Vaswani et al., 2017): one score per query-key pair. Scores computed from seeded Q, K (12 tokens, d = 8); causal mask, row softmax. At n = 32,000: 1.02e9 scores, 2.05 GB per head per layer in 16-bit.",
        )
        cell, gap = 0.32, 0.04
        pts = grid_points([-3.3, -0.5, 0], cell, gap)
        sq = cells(pts, cell)
        grid = m.VGroup(*[sq[i, j] for i in range(N) for j in range(N)])
        qmarks = m.VGroup(*[m.Square(side_length=0.2).set_fill(QUERY_FILL, 1).set_stroke(QUERY, width=2).move_to(pts[i, 0] + 0.48 * m.LEFT) for i in range(N)])
        kmarks = m.VGroup(*[m.Square(side_length=0.2).set_fill(KEY_FILL, 1).set_stroke(KEY, width=2).move_to(pts[0, j] + 0.48 * m.UP) for j in range(N)])
        klabel = text("n keys", 22, KEY, weight=m.SEMIBOLD).next_to(kmarks, m.RIGHT, buff=0.25)
        qlabel = text("n queries", 22, QUERY, weight=m.SEMIBOLD).next_to(qmarks, m.DOWN, buff=0.2).align_to(qmarks, m.LEFT)

        formula = m.MathTex("S", "=", "Q", r"K^{\top}", "/", r"\sqrt{d}", font_size=56, color=INK).move_to([3.4, 1.75, 0])
        formula[0].set_color(VIOLET)
        formula[2].set_color(QUERY)
        formula[3].set_color(KEY)
        source = cite("Vaswani et al., 2017").next_to(formula, m.DOWN, buff=0.2).align_to(formula, m.RIGHT)
        sx, dx = 1.15, 1.85
        legend = [
            tex_row("Q", "queries, one per token  (n × d)", QUERY, sx, dx, 0.55),
            tex_row(r"K^{\top}", "keys, transposed  (d × n)", KEY, sx, dx, -0.05),
            tex_row(r"\sqrt{d}", "d is the head size", INK, sx, dx, -0.65),
            tex_row("S", "the scores  (n × n)", VIOLET, sx, dx, -1.25),
        ]
        soft = text("then: mask the future, softmax each row", 22, INK).move_to([0, -1.95, 0])
        soft.move_to([dx - 0.7 + soft.width / 2, -1.95, 0])

        row_box = m.SurroundingRectangle(m.VGroup(*[sq[7, j] for j in range(N)]), buff=0.05).set_stroke(QUERY, width=3)
        col_box = m.SurroundingRectangle(m.VGroup(*[sq[i, 3] for i in range(N)]), buff=0.05).set_stroke(KEY, width=3)

        grow = m.ValueTracker(1.0)
        shown = m.ValueTracker(0.0)
        corner = np.array([1.15, -1.35, 0])
        ghost = m.DashedVMobject(m.Square(side_length=0.95).set_stroke(MUTED, width=2.5), num_dashes=24).move_to(corner, aligned_edge=m.DL)
        square = m.always_redraw(
            lambda: m.Square(side_length=0.95 * grow.get_value())
            .set_fill(VIOLET, 0.3 * shown.get_value())
            .set_stroke(VIOLET, width=3, opacity=shown.get_value())
            .move_to(corner, aligned_edge=m.DL)
        )
        tok_x = m.DecimalNumber(1.0, num_decimal_places=1, mob_class=Mono, font_size=30, color=INK)
        sc_x = m.DecimalNumber(1.0, num_decimal_places=1, mob_class=Mono, font_size=30, color=VIOLET)
        tok_row = m.VGroup(text("tokens ×", 26, MUTED), tok_x).arrange(m.RIGHT, buff=0.2).move_to([4.6, 0.05, 0])
        sc_row = m.VGroup(text("scores ×", 26, MUTED), sc_x).arrange(m.RIGHT, buff=0.2).move_to([4.6, -0.7, 0])
        sc_row.align_to(tok_row, m.LEFT)
        tok_x.add_updater(lambda d: d.set_value(grow.get_value()))
        sc_x.add_updater(lambda d: d.set_value(grow.get_value() ** 2))

        n_line = text(
            f"n = {N_CHAT:,}  →  n² ≈ {SCORES / 1e9:.2f} billion scores", 26, INK, t2c={f"{SCORES / 1e9:.2f} billion": VIOLET}
        ).move_to([3.4, -2.3, 0])
        gb_line = text(
            f"× 2 bytes = {SCORE_GB:.2f} GB, for one head in one layer", 26, INK, t2c={f"{SCORE_GB:.2f} GB": BAD}
        ).move_to([3.4, -2.95, 0])

        self.say(
            "[Self-attention compares every token with every other token.]",
            m.AnimationGroup(
                self.new_header("THE PROBLEM", "Attention builds an n × n matrix"),
                m.FadeIn(grid),
                m.LaggedStart(*[m.FadeIn(q, scale=0.5) for q in qmarks], lag_ratio=0.05),
                m.LaggedStart(*[m.FadeIn(k, scale=0.5) for k in kmarks], lag_ratio=0.05),
                m.FadeIn(klabel),
                m.FadeIn(qlabel),
            ),
        )
        self.say(
            "[Each query takes a dot product with every key], [divided by the square root of the head size.]",
            m.AnimationGroup(m.Write(formula[:4]), m.FadeIn(source), m.FadeIn(legend[0], shift=0.15 * m.UP), m.FadeIn(legend[1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Write(formula[4:]), m.FadeIn(legend[2], shift=0.15 * m.UP)),
        )
        self.say(
            "[The result is an n by n matrix of scores]: [one row for each query, one column for each key.]",
            m.AnimationGroup(
                m.LaggedStart(*[sq[i, j].animate.set_fill(heat(RAW_T[i, j])) for i in range(N) for j in range(N)], lag_ratio=0.01, run_time=1.8),
                m.FadeIn(legend[3], shift=0.15 * m.UP),
            ),
            m.AnimationGroup(m.Create(row_box), m.Create(col_box)),
        )
        self.say(
            "[A chat model masks out future tokens], [and a softmax turns each row into weights for the values.]",
            m.AnimationGroup(
                m.FadeOut(row_box),
                m.FadeOut(col_box),
                *[sq[i, j].animate.set_fill(MASKED) for i in range(N) for j in range(N) if j > i],
            ),
            m.AnimationGroup(m.FadeIn(soft, shift=0.15 * m.UP), *[sq[i, j].animate.set_fill(heat(W_T[i, j])) for i in range(N) for j in range(N) if j <= i]),
        )
        self.add(square)
        self.say(
            "[Double the chat, and this matrix grows four times.]",
            m.Succession(
                m.AnimationGroup(*[m.FadeOut(r) for r in legend], m.FadeOut(soft), run_time=0.5),
                m.AnimationGroup(
                m.FadeIn(ghost),
                m.FadeIn(tok_row),
                m.FadeIn(sc_row),
                shown.animate(run_time=0.5).set_value(1.0),
                grow.animate(run_time=2.4, rate_func=m.smooth).set_value(2.0),
                ),
            ),
        )
        self.say(
            "[At thirty-two thousand tokens, that's about a billion scores]: [two gigabytes in sixteen-bit numbers, for just one head in one layer.]",
            m.FadeIn(n_line, shift=0.2 * m.UP),
            m.FadeIn(gb_line, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.stop(square, tok_x, sc_x)
        self.clear()

    # ------------------------------------------------------------------ memory-bound

    def bottleneck(self):
        self.next_section(
            "Memory-bound",
            notes="A100 figures from Dao et al., 2022: HBM 40-80 GB at 1.5-2.0 TB/s; SRAM 192 KB on each of 108 SMs (~20 MB) at ~19 TB/s. Standard attention (their Algorithm 0) writes S, reads S, writes P, reads P in HBM.",
        )
        chip_box = card(5.4, 3.8, SKY, CARD, stroke=2).move_to([-3.6, 0.3, 0])
        chip_title = text("GPU chip", 26, INK, weight=m.BOLD).move_to(chip_box.get_corner(m.UL) + np.array([0.95, -0.42, 0]))
        units = m.VGroup(*[m.Square(side_length=0.34).set_fill(SKY, 0.85).set_stroke(SKY, width=1.5) for _ in range(15)])
        units.arrange_in_grid(rows=3, cols=5, buff=0.12).move_to([-5.0, 0.15, 0])
        units_label = text("arithmetic units", 22, MUTED).next_to(units, m.DOWN, buff=0.25)
        idle_label = text("mostly waiting", 22, BAD, weight=m.SEMIBOLD).move_to(units_label)
        sram = card(2.2, 2.0, SRAM_C, LIME_FILL, stroke=2.5).move_to([-2.25, 0.05, 0])
        sram_title = text("SRAM", 26, SRAM_C, weight=m.BOLD).move_to(sram.get_center() + 0.55 * m.UP)
        sram_speed = code("≈ 19 TB/s", 22, INK).move_to(sram.get_center() + 0.0 * m.UP)
        sram_size = code("≈ 20 MB", 22, INK).move_to(sram.get_center() + 0.5 * m.DOWN)
        chip = m.VGroup(chip_box, chip_title, units, units_label, sram, sram_title, sram_speed)

        hbm = card(4.8, 3.8, HBM_C, CARD, stroke=2).move_to([3.75, 0.3, 0])
        hbm_title = text("main memory (HBM)", 26, HBM_C, weight=m.BOLD).move_to(hbm.get_top() + 0.45 * m.DOWN)
        hbm_specs = code("40–80 GB · 1.5–2.0 TB/s", 22, INK).move_to(hbm.get_top() + 1.05 * m.DOWN)
        source = cite("A100 figures: Dao et al., 2022").next_to(hbm, m.DOWN, buff=0.15).align_to(hbm, m.RIGHT)

        def block(x, name):
            g = m.VGroup(*[m.Square(side_length=0.22).set_fill(heat(0.25 + 0.7 * ((i * 7 + 3) % 9) / 8), 1).set_stroke(width=0) for i in range(16)])
            g.arrange_in_grid(rows=4, cols=4, buff=0.04).move_to([x, -0.3, 0])
            label = tex(name, 32, VIOLET).next_to(g, m.DOWN, buff=0.15)
            return m.VGroup(g, label)

        s_block, p_block = block(2.95, r"S\ (n \times n)"), block(4.55, r"P\ (n \times n)")
        write = arrow([-0.9, 0.75, 0], [1.35, 0.75, 0], VIOLET, buff=0.0)
        read = arrow([1.35, -0.15, 0], [-0.9, -0.15, 0], VIOLET, buff=0.0)
        write_label = text("write", 22, MUTED).next_to(write, m.UP, buff=0.12)
        read_label = text("read", 22, MUTED).next_to(read, m.DOWN, buff=0.12)
        steps = m.VGroup(
            *[pill(s, VIOLET, VIOLET_FILL, 22) for s in ("1 · write S", "2 · read S", "3 · write P", "4 · read P")]
        ).arrange(m.RIGHT, buff=0.3).move_to([0, -2.4, 0])
        bound = pill("memory-bound: the time is set by bytes moved, not by arithmetic", BAD, BAD_FILL, 24).move_to([0, -3.25, 0])

        self.say(
            "[A data center GPU's main memory is large]: [tens of gigabytes, read at one and a half to two terabytes per second.]",
            m.AnimationGroup(self.new_header("THE BOTTLENECK", "Moving bytes, not doing math"), m.FadeIn(hbm), m.FadeIn(hbm_title, shift=0.15 * m.UP)),
            m.FadeIn(hbm_specs, shift=0.15 * m.UP),
        )
        self.say(
            "[The memory on the chip itself is about ten times faster], [but only around twenty megabytes in total.]",
            m.FadeIn(chip, shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(sram_size, shift=0.15 * m.UP), m.FadeIn(source)),
        )
        self.say(
            "[Standard attention writes the score matrix to main memory], [reads it back for the softmax], [writes the weights], [and reads them again.]",
            m.AnimationGroup(m.GrowArrow(write), m.FadeIn(write_label), m.FadeIn(s_block, target_position=sram), m.FadeIn(steps[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(read), m.FadeIn(read_label), pulse(read, INK), m.FadeIn(steps[1], shift=0.2 * m.UP)),
            m.AnimationGroup(pulse(write, INK), m.FadeIn(p_block, target_position=sram), m.FadeIn(steps[2], shift=0.2 * m.UP)),
            m.AnimationGroup(pulse(read, INK), m.FadeIn(steps[3], shift=0.2 * m.UP)),
        )
        self.say(
            "[Softmax does only a few operations per number, so the arithmetic units mostly wait.] [That's what memory-bound means.]",
            m.AnimationGroup(units.animate.set_fill(FAINT, 0.6).set_stroke(FAINT), m.FadeTransform(units_label, idle_label)),
            m.FadeIn(bound, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the KV cache

    def kv_cache(self):
        self.next_section(
            "KV cache",
            notes="Decoding one token at a time: keep every past key and value instead of recomputing them. Illustrative 7B-class config: 32 layers, 32 KV heads, head size 128, 2 bytes: 524,288 B per token, 16.8 GB at 32,000 tokens. Each new token reads the whole cache: about one multiply-add per number loaded (q·k per key element, p·v per value element).",
        )
        words = ["about", "the", "launch", "?", "You", "agreed", "to", "delay", "it", "to"]
        xs = np.linspace(-5.75, 5.75, len(words))
        chips = [chip(w, SKY, SKY_FILL, width=1.12, height=0.62, size=22) for w in words]
        for c, x in zip(chips, xs):
            c.move_to([x, 1.75, 0])
        new_chip = chips[-1]
        new_chip[0].set_stroke(QUERY, width=3)
        old_chips = m.VGroup(*chips[:-1])
        new_tag = text("newest", 22, QUERY, weight=m.BOLD).next_to(new_chip, m.UP, buff=0.12)

        def cache_cell(x, y, color, fill):
            return m.RoundedRectangle(corner_radius=0.08, width=0.7, height=0.4, stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1).move_to([x, y, 0])

        VY, KY, QY = 0.85, 0.3, -0.6
        v_old = m.VGroup(*[cache_cell(x, VY, BAD, BAD_FILL) for x in xs[:-1]])
        k_old = m.VGroup(*[cache_cell(x, KY, BAD, BAD_FILL) for x in xs[:-1]])
        v_new = cache_cell(xs[-1], VY, VALUE, VALUE_FILL)
        k_new = cache_cell(xs[-1], KY, KEY, KEY_FILL)
        q_new = cache_cell(xs[-1], QY, QUERY, QUERY_FILL)
        v_tag = text("V", 24, VALUE, weight=m.BOLD).move_to([-6.6, VY, 0])
        k_tag = text("K", 24, KEY, weight=m.BOLD).move_to([-6.6, KY, 0])
        q_tag = text("q", 26, QUERY, weight=m.BOLD).next_to(q_new, m.RIGHT, buff=0.18)
        recompute = pill("no cache: recompute every key and value, every step", BAD, BAD_FILL, 22).move_to([0, -1.45, 0])
        box = m.SurroundingRectangle(m.VGroup(v_old, k_old), buff=0.12, corner_radius=0.12).set_stroke(KEY, width=2.5)
        box_big = m.SurroundingRectangle(m.VGroup(v_old, k_old, v_new, k_new), buff=0.12, corner_radius=0.12).set_stroke(KEY, width=2.5)
        cache_tag = pill("KV cache", KEY, KEY_FILL, 22).move_to([-4.6, -0.45, 0])
        links = m.VGroup(*[m.Line(q_new.get_left(), k.get_bottom()).set_stroke(QUERY, width=2, opacity=0.7) for k in k_old])
        own = m.Line(q_new.get_top(), k_new.get_bottom()).set_stroke(QUERY, width=2, opacity=0.7)
        linear = pill("per new token: O(n) work, not O(n²)", GOOD, GOOD_FILL, 22).move_to([-3.1, -1.6, 0])
        grows = pill("the cache: O(n) memory, and growing", KEY, KEY_FILL, 22).move_to([3.3, -1.6, 0])
        diagram = m.VGroup(old_chips, new_chip, new_tag, v_old, k_old, v_new, k_new, q_new, v_tag, k_tag, q_tag, cache_tag, links, own, linear, grows)

        # the size formula, one part per term, spread out so each term has a label
        parts = [r"\text{KV cache}", "=", "2", r"\times", "L", r"\times", r"H_{kv}", r"\times", "d", r"\times", "n", r"\times", "b"]
        f = m.MathTex(*parts, font_size=54, color=INK)
        term_x = [-3.0, -1.35, 0.3, 1.95, 3.6, 5.25]
        y0 = 1.45
        f.shift((y0 - f.get_y()) * m.UP)  # the whole formula first, so the parts keep one baseline
        f[0].set_x(-5.3)
        f[1].set_x(f[0].get_right()[0] + 0.3 + f[1].width / 2)
        for k, x in enumerate(term_x):
            f[2 + 2 * k].set_x(x)
        for k in range(5):  # each × midway between the edges of its neighbours
            f[3 + 2 * k].set_x((f[2 + 2 * k].get_right()[0] + f[4 + 2 * k].get_left()[0]) / 2)
        f[6].set_color(GQA_C)
        f[10].set_color(SKY)
        labels = m.VGroup(*[text(s, 22, MUTED).move_to([x, 0.7, 0]) for s, x in zip(("K and V", "layers", "KV heads", "head size", "tokens", "bytes"), term_x)])
        ntok = m.ValueTracker(1)
        nums = [code(s, 30, INK).move_to([x, 0.0, 0]) for s, x in zip(("2", str(LAYERS), str(HEADS), str(HEAD_DIM)), term_x)]
        n_num = m.DecimalNumber(1, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=30, color=SKY)
        n_at = np.array([term_x[4], 0.0, 0])
        n_num.move_to(n_at)
        n_num.add_updater(lambda d: d.set_value(int(round(ntok.get_value()))).move_to(n_at))
        b_num = code(str(BYTES), 30, INK).move_to([term_x[5], 0.0, 0])
        b_note = text("16-bit", 20, MUTED).next_to(b_num, m.DOWN, buff=0.15)
        per_token = text(
            f"per token: {PER_TOKEN:,} bytes ≈ {PER_TOKEN / 1e6:.2f} MB", 28, INK, t2c={f"≈ {PER_TOKEN / 1e6:.2f} MB": KEY}
        ).move_to([0, -1.1, 0])
        gb = m.DecimalNumber(PER_TOKEN / 1e9, num_decimal_places=1, mob_class=Mono, font_size=48, color=KEY)
        gb_at = np.array([1.35, -2.05, 0])
        gb.move_to(gb_at, aligned_edge=m.RIGHT)
        gb.add_updater(lambda d: d.set_value(cache_bytes(HEADS, ntok.get_value()) / 1e9).move_to(gb_at, aligned_edge=m.RIGHT))
        gb_head = text("one chat:", 30, INK, weight=m.SEMIBOLD).next_to(gb_at, m.LEFT, buff=1.55)
        gb_unit = text("GB", 30, KEY, weight=m.BOLD).next_to(gb_at, m.RIGHT, buff=0.2)
        reads = m.VGroup(
            pill(f"each new token reads all {CACHE_GB[HEADS]:.1f} GB", BAD, BAD_FILL, 22),
            pill("≈ 1 multiply-add per number read", BAD, BAD_FILL, 22),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -3.1, 0])

        self.say(
            "[Back to our chat.] [The model writes its reply one token at a time.]",
            m.AnimationGroup(self.new_header("IDEA 1 · KV CACHE", "Keep the past keys and values"), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in old_chips], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(new_chip, shift=0.3 * m.LEFT), m.FadeIn(new_tag)),
        )
        self.say(
            "[Without a cache, every new token would recompute the keys and values of the whole chat.]",
            m.AnimationGroup(
                m.FadeIn(v_tag),
                m.FadeIn(k_tag),
                m.LaggedStart(*[m.FadeIn(m.VGroup(v, k), shift=0.15 * m.DOWN) for v, k in zip(v_old, k_old)], lag_ratio=0.1),
                m.FadeIn(recompute, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[Instead, we keep them.] [The KV cache stores every past key and value], [so a new token computes only its own], [and attends over the cache.]",
            m.AnimationGroup(
                m.FadeOut(recompute),
                v_old.animate.set_fill(VALUE_FILL).set_stroke(VALUE),
                k_old.animate.set_fill(KEY_FILL).set_stroke(KEY),
            ),
            m.AnimationGroup(m.Create(box), m.FadeIn(cache_tag, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(v_new, shift=0.25 * m.DOWN), m.FadeIn(k_new, shift=0.25 * m.DOWN), m.FadeIn(q_new, shift=0.25 * m.DOWN), m.FadeIn(q_tag)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(l) for l in reversed(links)], lag_ratio=0.08), m.Create(own)),
        )
        self.say(
            "[Each step now costs work in proportion to n, not n squared.] [But the cache grows with every token.]",
            m.FadeIn(linear, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(grows, shift=0.2 * m.UP), m.Transform(box, box_big)),
        )
        self.say(
            "[How big does it get?] [Two, for keys and values], [times layers], [times key-value heads], [times head size], [times tokens], [times bytes per number.]",
            m.Succession(m.AnimationGroup(m.FadeOut(diagram), m.FadeOut(box), run_time=0.5), m.FadeIn(f[:2], shift=0.2 * m.UP, run_time=0.6)),
            *[
                m.AnimationGroup(
                    m.FadeIn(f[2 + 2 * k], shift=0.2 * m.UP),
                    *([m.FadeIn(f[1 + 2 * k])] if k > 0 else []),
                    m.FadeIn(labels[k]),
                )
                for k in range(6)
            ],
        )
        self.say(
            "[For example, take thirty-two layers], [thirty-two heads of a hundred and twenty-eight dimensions], [and sixteen-bit numbers.]",
            m.AnimationGroup(m.FadeIn(nums[0], shift=0.15 * m.DOWN), m.FadeIn(nums[1], shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(nums[2], shift=0.15 * m.DOWN), m.FadeIn(nums[3], shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(b_num, shift=0.15 * m.DOWN), m.FadeIn(b_note)),
        )
        self.say(
            "[That's about half a megabyte per token.] [At thirty-two thousand tokens], [one chat's cache is almost seventeen gigabytes.]",
            m.AnimationGroup(m.FadeIn(n_num), m.FadeIn(per_token, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(gb_head), m.FadeIn(gb), m.FadeIn(gb_unit), ntok.animate(run_time=2.2, rate_func=m.smooth).set_value(N_CHAT)),
            m.Circumscribe(m.VGroup(gb_head, gb, gb_unit), color=KEY, buff=0.15),
        )
        self.say(
            "[And every new token must read all of it from main memory], [doing about one multiply-add per number it reads.]",
            m.FadeIn(reads[0], shift=0.2 * m.UP),
            m.FadeIn(reads[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.stop(n_num, gb)
        self.clear()

    # ------------------------------------------------------------------ FlashAttention

    def flash(self):
        self.next_section(
            "FlashAttention",
            notes="Dao et al., 2022: exact attention, tiled so blocks of Q, K, V fit in SRAM; the n × n S and P are never written to HBM; online softmax (Milakov & Gimelshein, 2018) keeps a running max and sum per row; the backward pass recomputes scores from stored statistics; extra memory O(n); GPT-2 (1K tokens) trained 3× faster than the Hugging Face baseline. Tile order follows the 2022 paper's loops (outer over K/V blocks). The online-softmax numbers are computed for row 12 of the seeded matrix.",
        )
        cell, gap, tg = 0.3, 0.03, 0.17
        center = np.array([-3.55, -0.55, 0])
        flat = grid_points(center, cell, gap)
        tiled = grid_points(center, cell, gap, tile_gap=tg)
        sq = cells(flat, cell)
        for (i, j), c in sq.items():
            if j > i:
                c.set_fill(MASKED)
        grid = m.VGroup(*[sq[i, j] for i in range(N) for j in range(N)])
        top = tiled[0, 0][1] + cell / 2
        left = tiled[0, 0][0] - cell / 2
        bottom = tiled[N - 1, 0][1] - cell / 2
        right = tiled[0, N - 1][0] + cell / 2

        def tile_center(points, a):
            return np.mean([points[TILE * a + r, TILE * a + r] for r in range(TILE)], axis=0)

        def qbar(points, a):
            ys = [points[TILE * a + r, 0][1] for r in range(TILE)]
            return m.Rectangle(width=0.16, height=max(ys) - min(ys) + cell, stroke_width=0).set_fill(QUERY, 0.9).move_to([points[0, 0][0] - cell / 2 - 0.25, np.mean(ys), 0])

        def kvbar(points, a, dy, color):
            xs_ = [points[0, TILE * a + r][0] for r in range(TILE)]
            return m.Rectangle(width=max(xs_) - min(xs_) + cell, height=0.14, stroke_width=0).set_fill(color, 0.9).move_to([np.mean(xs_), points[0, 0][1] + cell / 2 + dy, 0])

        qbars_flat = [qbar(flat, a) for a in range(3)]
        qbars_tiled = [qbar(tiled, a) for a in range(3)]
        kbars_flat = [kvbar(flat, a, 0.22, KEY) for a in range(3)]
        kbars_tiled = [kvbar(tiled, a, 0.22, KEY) for a in range(3)]
        vbars_flat = [kvbar(flat, a, 0.42, VALUE) for a in range(3)]
        vbars_tiled = [kvbar(tiled, a, 0.42, VALUE) for a in range(3)]
        bars = m.VGroup(*qbars_flat, *kbars_flat, *vbars_flat)
        q_label = text("Q", 24, QUERY, weight=m.BOLD).move_to([left - 0.25, bottom - 0.32, 0])
        kv_label = text("K, V", 24, KEY, weight=m.BOLD, t2c={"V": VALUE}).move_to([right + 0.5, top + 0.32, 0])

        hbm = card(5.6, 2.25, HBM_C, CARD, stroke=2).move_to([3.55, 1.25, 0])
        hbm_title = text("main memory (HBM)", 24, HBM_C, weight=m.BOLD)
        hbm_title.move_to(hbm.get_corner(m.UL) + np.array([0.4 + hbm_title.width / 2, -0.4, 0]))
        stored = m.VGroup(
            chip("Q", QUERY, QUERY_FILL, width=0.75, height=0.55, size=24),
            chip("K", KEY, KEY_FILL, width=0.75, height=0.55, size=24),
            chip("V", VALUE, VALUE_FILL, width=0.75, height=0.55, size=24),
            chip("O", INK, CARD, width=0.75, height=0.55, size=24),
        ).arrange(m.RIGHT, buff=0.25)
        stored.move_to(hbm.get_center() + 0.05 * m.DOWN).align_to(hbm, m.LEFT).shift(0.4 * m.RIGHT)
        never = text("S, P (n × n): never written", 22, GOOD, weight=m.SEMIBOLD).move_to(hbm.get_bottom() + 0.38 * m.UP)
        source = cite("FlashAttention: Dao et al., 2022").move_to([0, -3.25, 0]).align_to(hbm, m.RIGHT)

        sram = card(5.6, 2.25, SRAM_C, CARD, stroke=2).move_to([3.55, -1.4, 0])
        sram_title = text("on-chip SRAM: one tile at a time", 24, SRAM_C, weight=m.BOLD).move_to(sram.get_top() + 0.38 * m.DOWN)
        mini_pts = {(i, j): np.array([1.7 + 0.29 * j - 0.435, -1.5 - 0.29 * i + 0.435, 0]) for i in range(TILE) for j in range(TILE)}
        mini = m.VGroup(*[m.Square(side_length=0.25).set_fill(heat(RAW_T[i, j]), 1).set_stroke(width=0).move_to(p) for (i, j), p in mini_pts.items()])
        steps_txt = m.VGroup(text("scores → softmax → × V", 22, INK), text("then the tile is discarded", 22, MUTED)).arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT)
        steps_txt.next_to(mini, m.RIGHT, buff=0.45)

        order = [(i, j) for j in range(3) for i in range(3) if i >= j]  # outer loop over K/V blocks, as in the 2022 paper
        def tile_cells(i, j):
            return [sq[TILE * i + r, TILE * j + c] for r in range(TILE) for c in range(TILE)]

        def tile_lit(i, j):
            return [sq[TILE * i + r, TILE * j + c].animate.set_fill(heat(W_T[TILE * i + r, TILE * j + c])) for r in range(TILE) for c in range(TILE) if TILE * j + c <= TILE * i + r]

        def tile_dim(i, j):
            return [sq[TILE * i + r, TILE * j + c].animate.set_fill(EMPTY) for r in range(TILE) for c in range(TILE) if TILE * j + c <= TILE * i + r]

        def tile_box(i, j):
            g = m.VGroup(*[m.Square(side_length=cell).move_to(tiled[TILE * i + r, TILE * j + c]) for r in range(TILE) for c in range(TILE)])
            return m.SurroundingRectangle(g, buff=0.06).set_stroke(SRAM_C, width=3.5)

        hl = tile_box(*order[0])
        sweep = [m.AnimationGroup(m.FadeIn(hl), *tile_lit(*order[0]), run_time=0.5)]
        for prev, cur in zip(order, order[1:]):
            sweep.append(m.AnimationGroup(hl.animate.move_to(tile_box(*cur).get_center()), *tile_dim(*prev), *tile_lit(*cur), run_time=0.5))
        sweep.append(m.AnimationGroup(*tile_dim(*order[-1]), run_time=0.4))

        row_box = m.SurroundingRectangle(m.VGroup(*[m.Square(side_length=cell).move_to(tiled[ROW, j]) for j in range(N)]), buff=0.06).set_stroke(QUERY, width=3.5)
        row_tag = text("one query's row", 22, QUERY, weight=m.SEMIBOLD).next_to(row_box, m.DOWN, buff=0.15)
        row_tiles = [m.SurroundingRectangle(m.VGroup(*[m.Square(side_length=cell).move_to(tiled[ROW, TILE * a + c]) for c in range(TILE)]), buff=0.06).set_stroke(SRAM_C, width=3.5) for a in range(3)]

        formula = m.MathTex(r"\ell'", "=", r"e^{\,m - m'}", r"\ell", "+", r"\sum_{j} e^{\,s_j - m'}", font_size=48, color=INK).move_to([3.45, 1.75, 0])
        formula[0].set_color(SRAM_C)
        formula[2].set_color(SKY)
        formula[3].set_color(SRAM_C)
        osource = cite("online softmax: Milakov & Gimelshein, 2018").next_to(formula, m.DOWN, buff=0.25).align_to(formula, m.RIGHT)
        sx, dx = 1.25, 2.05
        legend = [
            tex_row("m", "running maximum of the scores", INK, sx, dx, 0.35),
            tex_row(r"\ell", "running sum of exponentials", SRAM_C, sx, dx, -0.3),
            tex_row("m'", "the maximum after this tile", INK, sx, dx, -0.95),
            tex_row(r"e^{\,m-m'}", "rescales the old sum", SKY, sx, dx, -1.6, sym_size=30),
        ]
        cols = [1.55, 3.4, 5.3]
        t_head = m.VGroup(*[text(s, 22, MUTED, weight=m.SEMIBOLD).move_to([x, 0.35, 0]) for s, x in zip(("tile", "running max", "running sum"), cols)])
        t_rows = m.VGroup(
            *[
                m.VGroup(code(str(k + 1), 26, INK).move_to([cols[0], -0.25 - 0.55 * k, 0]), code(f"{mx:.2f}", 26, INK).move_to([cols[1], -0.25 - 0.55 * k, 0]), code(f"{sm:.2f}", 26, SRAM_C).move_to([cols[2], -0.25 - 0.55 * k, 0]))
                for k, (mx, sm) in enumerate(STEPS)
            ]
        )
        exact = text(f"output vs exact attention: max difference {ERR:.1e}", 22, GOOD, weight=m.SEMIBOLD).move_to([3.45, -2.1, 0])

        facts = m.VGroup(
            pill("backward pass: recompute, don't store", SRAM_C, LIME_FILL, 22),
            pill("extra memory: O(n), not O(n²)", SRAM_C, LIME_FILL, 22),
            pill("exact, with far fewer trips to HBM", GOOD, GOOD_FILL, 22),
            pill("GPT-2 training: up to 3× faster", GOOD, GOOD_FILL, 22),
        )
        for p, y in zip(facts, (0.9, 0.0, -0.9, -1.8)):
            p.move_to([3.25, y, 0])

        self.say(
            "[Now, the score matrix itself.] [FlashAttention, published in twenty twenty-two, never writes it to main memory.]",
            m.AnimationGroup(
                self.new_header("IDEA 2 · FLASHATTENTION", "Exact attention that minds the memory"),
                m.FadeIn(grid),
                m.FadeIn(bars),
                m.FadeIn(q_label),
                m.FadeIn(kv_label),
                m.FadeIn(hbm),
                m.FadeIn(hbm_title),
                m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.UP) for s in stored], lag_ratio=0.2),
            ),
            m.AnimationGroup(m.FadeIn(never, shift=0.15 * m.UP), m.FadeIn(source)),
        )
        self.say(
            "[It cuts queries, keys and values into tiles], [small enough to fit in the fast on-chip memory.]",
            m.AnimationGroup(
                *[sq[k].animate.move_to(tiled[k]) for k in sq],
                *[a.animate.move_to(b) for a, b in zip(qbars_flat + kbars_flat + vbars_flat, qbars_tiled + kbars_tiled + vbars_tiled)],
                run_time=1.2,
            ),
            m.AnimationGroup(m.FadeIn(sram), m.FadeIn(sram_title), m.FadeIn(mini, target_position=tile_center(tiled, 0)), m.FadeIn(steps_txt, shift=0.15 * m.UP)),
        )
        self.say(
            "[Tile by tile, it computes the scores and multiplies them by the values], [without leaving the chip.]",
            m.Succession(*sweep),
            m.Circumscribe(sram, color=SRAM_C, buff=0.05),
        )
        self.say(
            "[The catch: softmax needs a whole row, its maximum and its sum, before any weight is known.]",
            m.AnimationGroup(m.FadeOut(hl), m.Create(row_box), m.FadeIn(row_tag, shift=0.15 * m.UP)),
        )
        self.say(
            "[The fix is an online softmax.] [It keeps a running maximum and a running sum], [rescaling the sum whenever a larger score arrives.]",
            m.Succession(m.AnimationGroup(*[m.FadeOut(x) for x in (hbm, hbm_title, stored, never, sram, sram_title, mini, steps_txt)], run_time=0.5), m.Write(formula, run_time=1.2)),
            m.AnimationGroup(m.FadeIn(legend[0], shift=0.15 * m.UP), m.FadeIn(legend[1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(legend[2], shift=0.15 * m.UP), m.FadeIn(legend[3], shift=0.15 * m.UP), m.FadeIn(osource)),
        )
        row_hl = row_tiles[0]
        tile_walk = [m.FadeIn(row_hl, run_time=0.4)] + [row_hl.animate(run_time=0.5).move_to(row_tiles[a + 1].get_center()) for a in range(2)]
        self.say(
            "[The output is rescaled the same way], [so after the last tile, it matches standard attention to within rounding.]",
            m.Succession(
                m.AnimationGroup(*[m.FadeOut(r) for r in legend], run_time=0.4),
                m.FadeIn(t_head, run_time=0.4),
                *[m.AnimationGroup(t, m.FadeIn(r, shift=0.15 * m.UP)) for t, r in zip(tile_walk, t_rows)],
            ),
            m.FadeIn(exact, shift=0.15 * m.UP),
        )
        self.say(
            "[In training, it recomputes the scores in the backward pass instead of storing them], [so the extra memory grows only linearly.]",
            m.Succession(m.AnimationGroup(*[m.FadeOut(x) for x in (formula, osource, t_head, t_rows, exact)], run_time=0.5), m.FadeIn(facts[0], shift=0.15 * m.UP, run_time=0.6)),
            m.FadeIn(facts[1], shift=0.15 * m.UP),
        )
        self.say(
            "[Same exact answer, far fewer trips to memory]: [the paper reports training up to three times faster.]",
            m.FadeIn(facts[2], shift=0.15 * m.UP),
            m.FadeIn(facts[3], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ shared K/V heads

    def sharing(self):
        self.next_section(
            "Shared K/V heads",
            notes="Multi-query attention (Shazeer, 2019): one K/V head for all query heads; grouped-query attention (Ainslie et al., 2023): groups of query heads share a K/V head; here 32 query heads and 8 groups of 4. Cache sizes from the same formula: 16.8, 0.52, 4.2 GB. GQA paper: quality close to multi-head, speed comparable to multi-query.",
        )
        qx = [-5.115 + 0.33 * k for k in range(HEADS)]
        QYR, KVY = 1.75, 0.5
        queries = m.VGroup(*[m.Square(side_length=0.24).set_fill(QUERY_FILL, 1).set_stroke(QUERY, width=2).move_to([x, QYR, 0]) for x in qx])
        q_label = text("32 query heads", 22, QUERY, weight=m.SEMIBOLD).next_to(queries, m.UP, buff=0.18).align_to(queries, m.LEFT)

        def heads_at(xs_):
            return m.VGroup(*[kv_head(x, KVY) for x in xs_])

        def links_to(xs_):
            return m.VGroup(*[m.Line([qx[k], QYR - 0.12, 0], [xs_[k], KVY + 0.26, 0]).set_stroke(MUTED, width=2) for k in range(HEADS)])

        mha_x = qx
        mqa_x = [0.0] * HEADS
        gqa_x = [float(np.mean(qx[4 * (k // 4) : 4 * (k // 4) + 4])) for k in range(HEADS)]
        kv = heads_at(mha_x)
        lines = links_to(mha_x)
        kv_label = text("key-value heads", 22, KEY, weight=m.SEMIBOLD).move_to([0, -0.15, 0]).align_to(queries, m.LEFT)

        def mode(label, color, fill, source):
            p = pill(label, color, fill, 22).move_to([0, -0.15, 0]).align_to(queries, m.RIGHT)
            if source:
                return m.VGroup(p, cite(source).next_to(p, m.DOWN, buff=0.12).align_to(p, m.RIGHT))
            return m.VGroup(p)

        mha_mode = mode("multi-head: 32 key-value heads", MHA_C, SKY_FILL, "")
        mqa_mode = mode("multi-query: 1 key-value head", MQA_C, VIOLET_FILL, "Shazeer, 2019")
        gqa_mode = mode("grouped-query: 8 heads, groups of 4", GQA_C, GOOD_FILL, "Ainslie et al., 2023")

        scale = 0.42
        x0 = -3.4

        def size_row(y, name, gb, color, note):
            label = text(name, 24, INK, weight=m.SEMIBOLD).move_to([0, y, 0])
            label.move_to([-6.3 + label.width / 2, y, 0])
            b = bar(gb * scale, 0.36, color).move_to([x0, y, 0], aligned_edge=m.LEFT)
            v = text(note, 22, INK, t2c={note.split(" · ")[0]: color}).next_to(b, m.RIGHT, buff=0.2)
            return m.VGroup(label, b, v)

        mha_row = size_row(-1.35, "multi-head", CACHE_GB[HEADS], MHA_C, f"{CACHE_GB[HEADS]:.1f} GB")
        mqa_row = size_row(-2.05, "multi-query", CACHE_GB[MQA_HEADS], MQA_C, f"{CACHE_GB[MQA_HEADS]:.2f} GB · 32× smaller")
        gqa_row = size_row(-2.75, "grouped-query", CACHE_GB[GQA_HEADS], GQA_C, f"{CACHE_GB[GQA_HEADS]:.1f} GB · 4× smaller")
        loss = pill("some quality loss", BAD, BAD_FILL, 20).move_to([0, -2.05, 0]).align_to(queries, m.RIGHT)
        good = m.VGroup(
            pill("quality near multi-head", GOOD, GOOD_FILL, 20).move_to([0, -2.75, 0]).align_to(queries, m.RIGHT),
            pill("speed near multi-query", GOOD, GOOD_FILL, 20).move_to([0, -3.45, 0]).align_to(queries, m.RIGHT),
        )
        cap = text("cache for our 32,000-token chat", 20, MUTED).move_to([0, -0.85, 0])
        cap.move_to([-6.3 + cap.width / 2, -0.85, 0])

        f = m.MathTex(r"2 \times L \times", r"H_{kv}", r"\times d \times n \times b", font_size=36, color=INK)
        f[1].set_color(GQA_C)
        swap = m.MathTex(r"H_{kv}:\ 32 \to 8", font_size=36, color=GQA_C)
        formula = m.VGroup(f, swap).arrange(m.RIGHT, buff=0.8).move_to([0, -3.45, 0])
        formula.shift((-6.3 - formula.get_left()[0]) * m.RIGHT)
        group_box = m.SurroundingRectangle(m.VGroup(*queries[4:8], kv[4:8]), buff=0.1, corner_radius=0.1).set_stroke(GQA_C, width=3)

        self.say(
            "[The third idea shrinks the KV cache itself.]",
            m.AnimationGroup(
                self.new_header("IDEA 3 · SHARED K/V HEADS", "Multi-query and grouped-query attention"),
                m.LaggedStart(*[m.FadeIn(q, scale=0.5) for q in queries], lag_ratio=0.02),
                m.FadeIn(q_label),
            ),
        )
        self.say(
            "[In multi-head attention, each of the thirty-two query heads has its own key and value head.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.Create(l), m.FadeIn(h, shift=0.15 * m.DOWN)) for l, h in zip(lines, kv)], lag_ratio=0.03),
                m.FadeIn(kv_label),
                m.FadeIn(mha_mode),
                m.FadeIn(cap),
                m.FadeIn(mha_row[0]),
                m.GrowFromEdge(mha_row[1], m.LEFT),
                m.FadeIn(mha_row[2]),
            ),
        )
        self.say(
            "[Multi-query attention, proposed in twenty nineteen, shares a single key and value head across all of them.]",
            m.AnimationGroup(
                m.Transform(kv, heads_at(mqa_x)),
                m.Transform(lines, links_to(mqa_x)),
                m.FadeOut(mha_mode),
                m.FadeIn(mqa_mode, shift=0.15 * m.UP),
                run_time=1.6,
            ),
        )
        self.say(
            "[The cache shrinks thirty-two times], [but quality can suffer a little.]",
            m.AnimationGroup(m.FadeIn(mqa_row[0]), m.GrowFromEdge(mqa_row[1], m.LEFT), m.FadeIn(mqa_row[2])),
            m.FadeIn(loss, shift=0.15 * m.LEFT),
        )
        self.say(
            "[Grouped-query attention, from twenty twenty-three, sits in between]: [eight key-value heads, each shared by a group of four query heads.]",
            m.AnimationGroup(m.FadeOut(mqa_mode), m.FadeIn(gqa_mode, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Transform(kv, heads_at(gqa_x)), m.Transform(lines, links_to(gqa_x)), run_time=1.6),
        )
        self.say(
            "[In the cache formula, only one term changes]: [the number of key-value heads], [so our chat's cache drops from almost seventeen gigabytes to about four.]",
            m.FadeIn(f, shift=0.15 * m.UP),
            m.AnimationGroup(m.Circumscribe(f[1], color=GQA_C, buff=0.08), m.FadeIn(swap, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(gqa_row[0]), m.GrowFromEdge(gqa_row[1], m.LEFT), m.FadeIn(gqa_row[2])),
        )
        self.say(
            "[Each number read from memory now serves four query heads], [and the paper finds quality close to multi-head attention, at nearly multi-query speed.]",
            m.Create(group_box),
            m.LaggedStart(*[m.FadeIn(g, shift=0.15 * m.LEFT) for g in good], lag_ratio=0.5),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section(
            "Serving the chat",
            notes="Illustrative: an 80 GB GPU, 14 GB of weights (7B parameters × 2 bytes), 32,000-token chats; activations and other overheads ignored. Chats that fit: 3 (32 KV heads), 15 (8), 125 (1).",
        )
        x0, width = -3.0, 7.6
        per_gb = width / GPU_GB
        ys = (1.35, 0.25, -0.85)
        rows = []
        for (h, name, sub, color), y in zip(
            ((HEADS, "multi-head", "32 KV heads", MHA_C), (GQA_HEADS, "grouped-query", "8 KV heads", GQA_C), (MQA_HEADS, "multi-query", "1 KV head", MQA_C)), ys
        ):
            label = m.VGroup(text(name, 26, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            label.move_to([-6.4 + label.width / 2, y, 0])
            frame = m.Rectangle(width=width, height=0.62).set_stroke(EDGE, width=2.5).set_fill(CARD, 1).move_to([x0 + width / 2, y, 0])
            weights = bar(WEIGHTS_GB * per_gb, 0.62, FAINT, 1.0).move_to([x0, y, 0], aligned_edge=m.LEFT)
            w_text = text("weights", 20, INK).move_to(weights)
            chats = m.VGroup(
                *[
                    m.Rectangle(width=CACHE_GB[h] * per_gb, height=0.62).set_fill(color, 0.85).set_stroke(BG, width=1.2 if h > 1 else 0.6)
                    .move_to([x0 + (WEIGHTS_GB + k * CACHE_GB[h]) * per_gb, y, 0], aligned_edge=m.LEFT)
                    for k in range(FIT[h])
                ]
            )
            count = text(f"{FIT[h]} chats", 28, color, weight=m.BOLD).next_to(frame, m.RIGHT, buff=0.3)
            rows.append((label, frame, weights, w_text, chats, count))
        ruler = m.VGroup(
            text("0", 20, MUTED).next_to(rows[0][1].get_corner(m.UL), m.UP, buff=0.12),
            text(f"{GPU_GB} GB", 20, MUTED).next_to(rows[0][1].get_corner(m.UR), m.UP, buff=0.12),
        )
        ruler[1].shift(0.3 * m.LEFT)
        note = text("one 32,000-token chat per block · activations and overheads ignored", 20, MUTED).move_to([0, -1.5, 0])
        note.align_to(rows[2][1], m.RIGHT)

        def fix_card(x, title, desc, color):
            box = card(4.15, 1.35, color, CARD, stroke=2).move_to([x, -2.75, 0])
            words = m.VGroup(text(title, 26, color, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, buff=0.12)
            return m.VGroup(box, words.move_to(box))

        fixes = [
            fix_card(-4.4, "KV cache", "no recomputing the past", KEY),
            fix_card(0.0, "FlashAttention", "no n × n matrix in HBM", SRAM_C),
            fix_card(4.4, "Shared K/V heads", "less to store and to read", QUERY),
        ]

        self.say(
            "[Picture, for example, an eighty-gigabyte GPU], [holding a seven-billion-parameter model: fourteen gigabytes of weights.]",
            m.AnimationGroup(self.new_header("THE BIG PICTURE", "How many long chats fit on one GPU?"), *[m.AnimationGroup(m.FadeIn(r[0]), m.FadeIn(r[1])) for r in rows], m.FadeIn(ruler)),
            m.AnimationGroup(*[m.AnimationGroup(m.GrowFromEdge(r[2], m.LEFT), m.FadeIn(r[3])) for r in rows], m.FadeIn(note)),
        )
        self.say(
            "[With multi-head caches, only three of our long chats fit.] [With grouped-query attention, fifteen.] [With multi-query, a hundred and twenty-five.]",
            *[
                m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.RIGHT) for c in r[4]], lag_ratio=0.3 if len(r[4]) < 20 else 0.02, run_time=1.4), m.FadeIn(r[5], shift=0.15 * m.LEFT))
                for r in rows
            ],
        )
        self.say(
            "[Each idea removes a different memory cost]: [the cache avoids recomputing], [FlashAttention avoids the giant matrix], [and shared heads shrink what every token reads.]",
            m.Indicate(note, color=INK, scale_factor=1.0),
            *[m.FadeIn(c, shift=0.2 * m.UP) for c in fixes],
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Cache, tile, share.")
        halo = glow(PINK, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Cache", "Tile", "Share"), (KEY, SRAM_C, QUERY))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Efficient attention is about moving fewer bytes.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [cache], [tile], [share.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Efficient attention is about moving fewer bytes.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
