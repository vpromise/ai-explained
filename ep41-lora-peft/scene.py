"""AI, Explained · Episode 41 · LoRA & Parameter-Efficient Fine-Tuning

Running example: personalising an open language model on a laptop. The model is an illustrative
Llama-style 7B model (32 layers, width 4096); the laptop has 16 GB of memory (illustrative).

- Full fine-tuning memory: mixed-precision Adam keeps 16 bytes per parameter (2 weights + 2
  gradients + 12 for the fp32 copy and the two moments; Rajbhandari et al., 2020, ZeRO), so 7B
  parameters need 112 GB before activations: seven times the laptop's 16 GB.
- Adapters (Houlsby et al., 2019): frozen network, bottleneck layers inserted in every block; on
  GLUE within 0.4% of full fine-tuning with 3.6% extra parameters per task; extra inference latency.
- LoRA (Hu et al., 2021): freeze W (d × k), learn ΔW = B·A, B is d × r, A is r × k, r ≪ d; B starts
  at zero; merge W' = W + BA after training (no inference cost). d = k = 4096, r = 8: 65,536 vs
  16,777,216 parameters (0.39%). On W_q and W_v of all 32 layers: 4,194,304 (0.06% of 7B),
  8.4 MB in 16-bit.
- The low-rank picture is computed: B (12 × 2) and A (2 × 12) from a seeded generator, their
  product, its two rank-one layers and its singular values (exactly two non-zero).
- QLoRA (Dettmers et al., 2023): frozen base in 4-bit NormalFloat + LoRA; a 65B model fine-tuned on
  one 48 GB GPU. Our 7B base: 3.5 GB in 4 bits (+ 0.11 GB quantisation constants with double
  quantisation, 0.127 bits per parameter) + 0.07 GB adapter training state: under 4 GB.
- The catch (Biderman et al., 2024, "LoRA Learns Less and Forgets Less").

Every number the narration quotes is asserted below. Facts as of October 2026.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

BLUE_FILL = "#14213D"
FROZEN, FROZEN_FILL = BLUE, BLUE_FILL  # frozen, pre-trained weights
TRAIN, TRAIN_FILL = AMBER, AMBER_FILL  # what we train: adapters, B, A, ΔW
RANK, RANK_FILL = PINK, PINK_FILL  # the rank r (and the level accent)
NEG = TEAL  # negative entries in the matrix pictures

# ---------------------------------------------------------------------------- the numbers

GB = 1e9
N_PARAMS = 7e9  # an illustrative 7B model: 32 layers, width 4096 (Llama-style)
LAYERS, WIDTH = 32, 4096
LAPTOP_GB = 16

# full fine-tuning with mixed-precision Adam (Rajbhandari et al., 2020): bytes per parameter
BYTES_W, BYTES_G, BYTES_ADAM = 2, 2, 12  # fp16 weights, fp16 gradients, fp32 copy + m + v
BYTES_FULL = BYTES_W + BYTES_G + BYTES_ADAM
MEM_W = N_PARAMS * BYTES_W / GB
MEM_G = N_PARAMS * BYTES_G / GB
MEM_ADAM = N_PARAMS * BYTES_ADAM / GB
MEM_FULL = N_PARAMS * BYTES_FULL / GB
assert BYTES_FULL == 16 and MEM_W == 14 and MEM_FULL == 112  # "two bytes", "fourteen", "sixteen bytes", "a hundred and twelve"
assert MEM_FULL / LAPTOP_GB == 7  # "seven times too little"

# LoRA on one d × k matrix
D = K = WIDTH
R = 8
FULL_COUNT = D * K
LORA_COUNT = R * (D + K)
LORA_SHARE = 100 * LORA_COUNT / FULL_COUNT
assert FULL_COUNT == 16_777_216 and round(FULL_COUNT / 1e6, 1) == 16.8  # "about sixteen point eight million"
assert LORA_COUNT == 65_536  # "sixty-five thousand, five hundred and thirty-six"
assert f"{LORA_SHARE:.2f}" == "0.39"  # "zero point three nine percent"

# the whole model: W_q and W_v in each of the 32 layers (as in Hu et al., 2021)
MODEL_LORA = LAYERS * 2 * LORA_COUNT
MODEL_SHARE = 100 * MODEL_LORA / N_PARAMS
ADAPTER_MB = MODEL_LORA * 2 / 1e6  # 16-bit
assert MODEL_LORA == 4_194_304 and round(MODEL_LORA / 1e6, 1) == 4.2  # "about four point two million"
assert f"{MODEL_SHARE:.2f}" == "0.06" and MODEL_SHARE < 0.1  # on screen "0.06%", spoken "under a tenth of a percent"
assert round(ADAPTER_MB) == 8  # "about eight megabytes"

# QLoRA for our model: 4-bit base + quantisation constants + the adapter's training state
MEM_4BIT = N_PARAMS * 0.5 / GB
MEM_CONST = N_PARAMS * 0.127 / 8 / GB  # double quantisation: 0.127 bits per parameter (Dettmers et al., 2023)
MEM_ADAPTER = MODEL_LORA * BYTES_FULL / GB  # the adapter, trained with mixed-precision Adam
MEM_LORA16 = MEM_W + MEM_ADAPTER
MEM_QLORA = MEM_4BIT + MEM_CONST + MEM_ADAPTER
assert MEM_4BIT == 3.5 and MEM_QLORA < 4 < LAPTOP_GB  # "under four gigabytes"
assert MEM_LORA16 < LAPTOP_GB < MEM_FULL

# the small low-rank example: B is 12 × 2, A is 2 × 12
_rng = np.random.default_rng(41)
SB = _rng.normal(size=(12, 2))
SA = _rng.normal(size=(2, 12))
SDW = SB @ SA
SV = np.linalg.svd(SDW, compute_uv=False)
assert SDW.shape == (12, 12) and np.linalg.matrix_rank(SDW) == 2  # "its rank is two"
assert np.sum(SV > 1e-9 * SV[0]) == 2  # "only two singular values are not zero"
LAYER1, LAYER2 = np.outer(SB[:, 0], SA[0]), np.outer(SB[:, 1], SA[1])
assert np.allclose(LAYER1 + LAYER2, SDW)  # "column one times row one, plus column two times row two"

# the 10 × 10 pictures of W, B, A (LoRA and merging sections)
_rng2 = np.random.default_rng(2021)
W10 = _rng2.normal(size=(10, 10))
B10 = _rng2.normal(size=(10, 2))
A10 = _rng2.normal(size=(2, 10))
DW10 = B10 @ A10
assert np.linalg.matrix_rank(DW10) == 2


# ---------------------------------------------------------------------------- helpers


def mix(a, b, t):
    ca = np.array([int(a[i : i + 2], 16) for i in (1, 3, 5)], float)
    cb = np.array([int(b[i : i + 2], 16) for i in (1, 3, 5)], float)
    c = np.round(ca + (cb - ca) * float(np.clip(t, 0, 1))).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*c)


def heat(values, cell, color, neg=None, scale=None, base=CARD):
    """A matrix as a grid of squares: |value| → from the card colour toward `color` (or `neg` when < 0)."""
    v = np.asarray(values, float)
    s = scale or (np.abs(v).max() or 1.0)
    rows, cols = v.shape
    g = m.VGroup()
    for i in range(rows):
        for j in range(cols):
            t = min(abs(v[i, j]) / s, 1.0)
            c = neg if (neg and v[i, j] < 0) else color
            sq = m.Square(side_length=cell * 0.86).set_stroke(width=0).set_fill(mix(base, c, 0.12 + 0.88 * t**0.75), 1)
            sq.move_to([(j - (cols - 1) / 2) * cell, ((rows - 1) / 2 - i) * cell, 0])
            g.add(sq)
    return g


def framed(grid, color, buff=0.06, stroke=2.5):
    frame = m.Rectangle(width=grid.width + 2 * buff, height=grid.height + 2 * buff).set_stroke(color, width=stroke).move_to(grid)
    return m.VGroup(frame, grid)


def formula(*parts, size=44):
    """A MathTex from (latex, colour) pairs, one part each."""
    t = m.MathTex(*[p for p, _ in parts], font_size=size, color=INK)
    for part, (_, c) in zip(t, parts):
        part.set_color(c)
    return t


def cite(s):
    """A paper reference, small and muted, top right (level with the header)."""
    words = text(s, 20, MUTED)
    return words.move_to([6.65 - words.width / 2, 3.22, 0])


def ep_recolored(name, color, fill):
    """kit.recolored(), but written as assets/ep41-<name>-<color>.svg (this episode's files only)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep41-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


def lock(height=0.5):
    return icon(ep_recolored("lock", FROZEN, FROZEN_FILL), height)


def box(label, width, height, color, fill, size=22, weight=m.SEMIBOLD, radius=0.15):
    b = card(width, height, color, fill, radius=radius, stroke=2.5)
    return m.VGroup(b, text(label, size, INK, weight=weight).move_to(b))


def two_lines(top, bottom, color, top_size=24, bottom_size=20, buff=0.08):
    return m.VGroup(text(top, top_size, color, weight=m.BOLD), text(bottom, bottom_size, MUTED)).arrange(m.DOWN, buff=buff, aligned_edge=m.LEFT)


def hbar(x0, x1, y, color, fill, height=0.62):
    w = max(x1 - x0, 0.05)
    return m.Rectangle(width=w, height=height).set_stroke(color, width=2).set_fill(fill, 1).move_to([x0 + w / 2, y, 0])


def trapezoid(cx, cy, width, left_h, right_h, color, fill):
    """A horizontal trapezoid: data flows left to right, from left_h wide to right_h wide."""
    x0, x1 = cx - width / 2, cx + width / 2
    pts = [[x0, cy + left_h / 2, 0], [x1, cy + right_h / 2, 0], [x1, cy - right_h / 2, 0], [x0, cy - left_h / 2, 0]]
    return m.Polygon(*pts).set_stroke(color, width=2.5).set_fill(fill, 1)


def poly(*points, color=MUTED, width=3):
    return m.VMobject().set_points_as_corners([[x, y, 0] for x, y in points]).set_stroke(color, width=width)


def oplus(center, color=INK, r=0.2):
    c = m.Circle(radius=r).set_stroke(color, width=2.5).set_fill(BG, 1).move_to(center)
    h = m.Line(c.get_left() + 0.07 * m.RIGHT, c.get_right() + 0.07 * m.LEFT).set_stroke(color, width=2.5)
    v = m.Line(c.get_bottom() + 0.07 * m.UP, c.get_top() + 0.07 * m.DOWN).set_stroke(color, width=2.5)
    return m.VGroup(c, h, v)


# ---------------------------------------------------------------------------- the episode


class Ep41LoRAPEFT(Episode):
    number = 41

    def series_intro(self):
        """The kit's series card, with its heading scaled to fit the frame (this title is wider than
        the frame at the kit's 60 pt). Same narration, so the cached intro line is unchanged."""
        title, level = EPISODES[self.number]
        name, accent, tint = LEVELS[level]
        self.next_section(f"Episode {self.number} · {title}", notes="Series card.")
        halo = glow(accent, radius=4.2).move_to(0.2 * m.UP)
        brand = text("AI, Explained", 34, INK, weight=m.BOLD, gradient=(SKY, VIOLET))
        badge = pill(f"LEVEL {level} · {name}", accent, tint, 20)
        number = text(f"Episode {self.number:02d}", 30, MUTED, weight=m.SEMIBOLD)
        heading = text(title, 76 if len(title) < 22 else 60, INK, weight=m.BOLD)
        if heading.width > 12.4:
            heading.scale(12.4 / heading.width)
        half = min(heading.width, 6) / 2
        rule = m.Line([-half, 0, 0], [half, 0, 0]).set_stroke(accent, width=4)
        card_group = m.VGroup(brand, badge, number, heading, rule).arrange(m.DOWN, buff=0.38).move_to(0.15 * m.UP)
        rule.next_to(heading, m.DOWN, buff=0.35)

        self.wait(0.3)
        self.say(
            f"[Welcome to AI, Explained.] [Episode {say_number(self.number)}]: [{spoken_title(title)}.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(brand, shift=0.2 * m.DOWN), run_time=1.0),
            m.AnimationGroup(m.FadeIn(badge, shift=0.2 * m.UP), m.FadeIn(number, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Write(heading), m.GrowFromCenter(rule)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, card_group)], run_time=0.8)

    def lesson(self):
        self.hook()
        self.problem()
        self.adapters()
        self.lora()
        self.low_rank()
        self.savings()
        self.training()
        self.merging()
        self.qlora()
        self.catch()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Running example: a model that writes like you, fine-tuned on a laptop. 7B parameters and 16 GB are illustrative.")
        you = icon("user", 1.3)
        docs = m.VGroup(icon(ep_recolored("doc", SKY, SKY_FILL), 0.9), icon("mail", 0.62)).arrange(m.RIGHT, buff=0.25)
        left = m.VGroup(you, docs).arrange(m.DOWN, buff=0.3).move_to([-4.6, 0.85, 0])
        left_cap = text("your notes and emails", 22, MUTED).move_to([-4.6, -0.95, 0])
        model = icon("llm", 1.8).move_to([0, 0.85, 0])
        model_cap = text("a language model", 22, MUTED).move_to([0, -0.95, 0])
        laptop = icon("ep41-laptop", 1.45).move_to([4.6, 0.85, 0])
        laptop_cap = text("your laptop · 16 GB", 22, MUTED).move_to([4.6, -0.95, 0])
        link = arrow([-3.35, 0.85, 0], [-1.25, 0.85, 0], SKY)
        link_label = text("fine-tune", 22, SKY, weight=m.SEMIBOLD).next_to(link, m.UP, buff=0.12)
        big = m.VGroup(text("7,000,000,000", 60, VIOLET, weight=m.BOLD), text("parameters", 30, MUTED))
        big.arrange(m.RIGHT, buff=0.35, aligned_edge=m.DOWN).move_to([0, -2.1, 0])
        peft = pill("parameter-efficient fine-tuning", RANK, RANK_FILL, 24).move_to([0, -3.3, 0])

        self.say(
            "[So... you want a language model that writes like you], [trained on your own notes and emails.]",
            m.AnimationGroup(m.FadeIn(model, scale=0.8), m.FadeIn(model_cap)),
            m.AnimationGroup(m.FadeIn(left, shift=0.3 * m.RIGHT), m.FadeIn(left_cap), m.GrowArrow(link), m.FadeIn(link_label)),
        )
        self.say(
            "[And you want to do it on your own laptop.] [But even a small open model has seven billion parameters.]",
            m.AnimationGroup(m.FadeIn(laptop, shift=0.3 * m.LEFT), m.FadeIn(laptop_cap)),
            m.FadeIn(big, shift=0.2 * m.UP),
        )
        self.say(
            "[Can a laptop really fine-tune that?] [It can, with parameter-efficient fine-tuning.]",
            m.Circumscribe(m.VGroup(laptop, laptop_cap), color=SKY, buff=0.15),
            m.FadeIn(peft, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the problem: memory

    def problem(self):
        self.next_section(
            "The problem: memory",
            notes="Full fine-tuning with mixed-precision Adam: 2 bytes (fp16 weights) + 2 (fp16 gradients) + 12 (fp32 master copy, "
            "first and second moments) = 16 bytes per parameter (Rajbhandari et al., 2020, ZeRO, sec. 3.1). 7B × 16 B = 112 GB, "
            "activations not counted. Laptop: 16 GB (illustrative).",
        )
        x0, x1 = -6.2, 6.2
        s = (x1 - x0) / MEM_FULL
        y = 0.75
        segs = [
            (0, MEM_W, FROZEN, FROZEN_FILL, "weights", f"{MEM_W:.0f} GB"),
            (MEM_W, MEM_W + MEM_G, PINK, PINK_FILL, "gradients", f"{MEM_G:.0f} GB"),
            (MEM_W + MEM_G, MEM_FULL, VIOLET, VIOLET_FILL, "Adam states", f"{MEM_ADAM:.0f} GB"),
        ]
        bars, labels = [], []
        for a, b, c, f, name, size in segs:
            bar = hbar(x0 + a * s, x0 + b * s, y, c, f, height=0.9)
            nm, sz = text(name, 22, c, weight=m.SEMIBOLD), text(size, 20, MUTED)
            nm.move_to([bar.get_x(), 0, 0]).align_to([0, y + 1.12, 0], m.UP)
            sz.move_to([bar.get_x(), 0, 0]).align_to([0, y + 0.8, 0], m.UP)
            lab = m.VGroup(nm, sz)
            bars.append(bar)
            labels.append(lab)
        inside = text("fp32 copy + two moments", 20, INK).move_to(bars[2])
        total = text(f"{MEM_FULL:.0f} GB", 30, BAD, weight=m.BOLD).next_to(bars[2], m.DOWN, buff=0.2).align_to(bars[2], m.RIGHT)
        lx = x0 + LAPTOP_GB * s
        line = m.DashedLine([lx, y + 0.57, 0], [lx, y - 0.95, 0], dash_length=0.1).set_stroke(SKY, width=3).set_z_index(2)
        lap = m.VGroup(icon("ep41-laptop", 0.42), text("laptop: 16 GB", 22, SKY, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.18)
        lap.next_to(line, m.DOWN, buff=0.15)
        lap.shift((lx - lap[0].get_center()[0]) * m.RIGHT)
        too_big = pill("7× more than the laptop has", BAD, BAD_FILL, 22).move_to([0.6, lap.get_y(), 0])
        f1 = formula(("2", FROZEN), ("+", INK), ("2", PINK), ("+", INK), ("12", VIOLET), ("=", INK), (r"16\ \text{bytes per weight}", INK), size=42)
        f1.move_to([0, -1.75, 0])
        f2 = formula((r"16\ \text{bytes} \times 7\ \text{billion}", INK), ("=", INK), (r"112\ \text{GB}", BAD), size=42).move_to([0, -2.7, 0])
        who = cite("ZeRO · Rajbhandari et al., 2020")

        self.say(
            "[Let's count the memory that ordinary, full fine-tuning needs.]",
            self.new_header("THE PROBLEM", "Full fine-tuning won't fit"),
        )
        self.say(
            "[In sixteen-bit precision, each weight takes two bytes]: [fourteen gigabytes for the model.]",
            m.FadeIn(f1[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(labels[0], shift=0.15 * m.UP)),
        )
        self.say(
            "[Training adds a gradient for every weight], [and mixed-precision Adam keeps a thirty-two-bit copy, plus two running averages.]",
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(labels[1], shift=0.15 * m.UP), m.FadeIn(f1[1:3])),
            m.AnimationGroup(m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(labels[2], shift=0.15 * m.UP), m.FadeIn(inside), m.FadeIn(f1[3:5])),
        )
        self.say(
            "[That's sixteen bytes per weight]: [about a hundred and twelve gigabytes], before we even count activations.",
            m.FadeIn(f1[5:], shift=0.15 * m.LEFT),
            m.AnimationGroup(m.Write(f2), m.FadeIn(total, shift=0.15 * m.DOWN), m.FadeIn(who)),
        )
        self.say(
            "[Our laptop has sixteen.] [That's seven times too little.]",
            m.AnimationGroup(m.Create(line), m.FadeIn(lap, shift=0.15 * m.UP)),
            m.FadeIn(too_big, scale=1.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ adapters

    def adapters(self):
        self.next_section(
            "Adapters",
            notes="Houlsby et al., 2019 (Parameter-Efficient Transfer Learning for NLP): bottleneck adapters (down-projection, "
            "nonlinearity, up-projection, skip connection) inserted twice per Transformer layer; on GLUE within 0.4% of full "
            "fine-tuning with 3.6% parameters per task. Extra sequential layers add inference latency, most visible at small "
            "batch sizes (Hu et al., 2021, sec. 3).",
        )
        cx, cy = -3.9, -0.2
        block = card(3.4, 4.0, FROZEN, FROZEN_FILL).move_to([cx, cy, 0])
        title = text("Transformer block", 22, MUTED).move_to([cx, cy + 1.62, 0])
        attn = box("attention", 2.6, 0.62, FROZEN, CARD).move_to([cx, cy - 1.35, 0])
        ffn = box("feed-forward", 2.6, 0.62, FROZEN, CARD).move_to([cx, cy + 0.05, 0])
        ad1 = box("adapter", 1.6, 0.46, TRAIN, TRAIN_FILL, size=20).move_to([cx, cy - 0.65, 0])
        ad2 = box("adapter", 1.6, 0.46, TRAIN, TRAIN_FILL, size=20).move_to([cx, cy + 0.75, 0])
        top_y = cy + 1.3

        def link(a, b):
            return m.Line([cx, a, 0], [cx, b, 0]).set_stroke(MUTED, width=2.5)

        old_link = link(attn.get_top()[1], ffn.get_bottom()[1])
        old_top = link(ffn.get_top()[1], top_y)
        new_links = m.VGroup(
            link(attn.get_top()[1], ad1.get_bottom()[1]),
            link(ad1.get_top()[1], ffn.get_bottom()[1]),
            link(ffn.get_top()[1], ad2.get_bottom()[1]),
            link(ad2.get_top()[1], top_y),
        )
        bottom_link = link(cy - 1.9, attn.get_bottom()[1])
        frozen = lock(0.55).move_to(block.get_corner(m.UR) + np.array([-0.05, -0.05, 0]))
        net = m.VGroup(block, title, attn, ffn, old_link, old_top, bottom_link)

        # inside one adapter: a bottleneck with a skip connection
        px, py = 2.85, -0.2
        panel = card(5.8, 4.0, TRAIN, CARD, stroke=2).move_to([px, py, 0])
        p_title = text("inside an adapter", 22, MUTED).move_to([px, py + 1.62, 0])
        bx = 1.55
        wide_lo = m.Rectangle(width=2.2, height=0.24).set_stroke(TRAIN, width=2).set_fill(TRAIN, 0.85).move_to([bx, py - 1.45, 0])
        down = m.Polygon([bx - 1.1, py - 1.25, 0], [bx + 1.1, py - 1.25, 0], [bx + 0.35, py - 0.6, 0], [bx - 0.35, py - 0.6, 0])
        down.set_stroke(TRAIN, width=2.5).set_fill(TRAIN_FILL, 1)
        narrow = m.Rectangle(width=0.7, height=0.24).set_stroke(TRAIN, width=2).set_fill(TRAIN, 0.85).move_to([bx, py - 0.43, 0])
        up = m.Polygon([bx - 0.35, py - 0.26, 0], [bx + 0.35, py - 0.26, 0], [bx + 1.1, py + 0.39, 0], [bx - 1.1, py + 0.39, 0])
        up.set_stroke(TRAIN, width=2.5).set_fill(TRAIN_FILL, 1)
        wide_hi = m.Rectangle(width=2.2, height=0.24).set_stroke(TRAIN, width=2).set_fill(TRAIN, 0.85).move_to([bx, py + 0.59, 0])
        plus = oplus([bx, py + 1.12, 0])
        to_plus = m.Line(wide_hi.get_top(), plus.get_bottom()).set_stroke(MUTED, width=2.5)
        skip = poly((bx - 1.1, py - 1.45), (bx - 1.35, py - 1.45), (bx - 1.35, py + 1.12), (bx - 0.2, py + 1.12), color=MUTED, width=2.5)
        lx = 3.05
        notes = m.VGroup(
            text("down: d → m", 22, INK).move_to([0, py - 0.95, 0]),
            text("nonlinearity", 22, MUTED).move_to([0, py - 0.43, 0]),
            text("up: m → d", 22, INK).move_to([0, py + 0.08, 0]),
            text("+ the input", 22, INK).move_to([0, py + 1.12, 0]),
        )
        for n in notes:
            n.align_to([lx, 0, 0], m.LEFT)
        inner = m.VGroup(wide_lo, down, narrow, up, wide_hi, to_plus, plus, skip)

        facts = pill("GLUE: +3.6% parameters per task, within 0.4% of full fine-tuning", GOOD, GOOD_FILL, 20).move_to([0, -2.72, 0])
        catch = pill("extra layers at inference → added latency", BAD, BAD_FILL, 20).move_to([0, -3.38, 0])
        who = cite("Houlsby et al., 2019")

        self.say(
            "[An earlier idea, from twenty nineteen, was the adapter.] [Freeze the whole pre-trained network.]",
            m.AnimationGroup(self.new_header("AN EARLIER IDEA", "Adapters"), m.FadeIn(who)),
            m.AnimationGroup(m.FadeIn(net, shift=0.2 * m.UP), m.FadeIn(frozen, scale=1.4)),
        )
        self.say(
            "[Then insert small bottleneck layers into every block]: "
            "[project down, apply a nonlinearity, project back up, and add the input back.]",
            m.AnimationGroup(m.FadeOut(old_link), m.FadeOut(old_top), m.FadeIn(ad1, scale=0.7), m.FadeIn(ad2, scale=0.7), m.Create(new_links)),
            m.AnimationGroup(
                m.FadeIn(panel, target_position=ad1),
                m.FadeIn(p_title),
                m.LaggedStart(*[m.FadeIn(x, shift=0.1 * m.UP) for x in inner], lag_ratio=0.15),
                m.LaggedStart(*[m.FadeIn(n) for n in notes], lag_ratio=0.3),
            ),
        )
        self.say(
            "[Train only those.] [On a standard benchmark, three point six percent more parameters came within zero point four percent of full fine-tuning.]",
            m.AnimationGroup(m.Indicate(ad1, color=TRAIN, scale_factor=1.12), m.Indicate(ad2, color=TRAIN, scale_factor=1.12)),
            m.FadeIn(facts, shift=0.2 * m.UP),
        )
        self.say(
            "[The catch]: [those extra layers run in sequence at inference, which adds latency, especially when serving one request at a time.]",
            m.AnimationGroup(m.Circumscribe(ad1, color=BAD, buff=0.08), m.Circumscribe(ad2, color=BAD, buff=0.08)),
            m.FadeIn(catch, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ LoRA: the idea

    def lora(self):
        self.next_section(
            "LoRA: the idea",
            notes="Hu et al., 2021 (LoRA: Low-Rank Adaptation of Large Language Models): W0 frozen, ΔW = BA with B ∈ R^{d×r}, "
            "A ∈ R^{r×k}, r ≪ min(d, k). The 10 × 10 pictures use a seeded rank-2 ΔW = B·A (illustrative).",
        )
        cell = 0.3
        w = framed(heat(W10, cell, FROZEN), FROZEN).move_to([-4.2, -0.75, 0])
        w_lab = text("W · d × k", 24, FROZEN, weight=m.SEMIBOLD).next_to(w, m.DOWN, buff=0.25)
        frozen = m.VGroup(lock(0.42), text("frozen", 22, FROZEN, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15).next_to(w, m.UP, buff=0.2)
        plus = text("+", 48, INK, weight=m.BOLD).move_to([-1.95, -0.75, 0])
        dw = framed(heat(DW10, cell, TRAIN, NEG), TRAIN).move_to([0.3, -0.75, 0])
        dw_lab = text("ΔW · d × k", 24, TRAIN, weight=m.SEMIBOLD).next_to(dw, m.DOWN, buff=0.25)
        learned = text("learned", 22, TRAIN, weight=m.SEMIBOLD).next_to(dw, m.UP, buff=0.25)
        b = framed(heat(B10, cell, TRAIN, NEG), TRAIN).move_to([-0.85, -0.75, 0])
        a = framed(heat(A10, cell, TRAIN, NEG), TRAIN).move_to([1.45, -0.75, 0])
        b_lab = text("B · d × r", 24, TRAIN, weight=m.SEMIBOLD, t2c={"r": RANK}).next_to(b, m.DOWN, buff=0.25)
        a_lab = text("A · r × k", 24, TRAIN, weight=m.SEMIBOLD, t2c={"r": RANK}).next_to(a, m.DOWN, buff=0.25)
        tall = text("tall and thin", 20, MUTED).next_to(b_lab, m.DOWN, buff=0.12)
        wide = text("short and wide", 20, MUTED).next_to(a_lab, m.DOWN, buff=0.12)
        rank = pill("r = 8  ≪  d = 4096", RANK, RANK_FILL, 22).move_to([5.05, -0.75, 0])
        f = formula(("W", FROZEN), ("+", INK), (r"\Delta W", TRAIN), size=54).move_to([0, 1.95, 0])
        f2 = formula(("W", FROZEN), ("+", INK), ("B", TRAIN), ("A", TRAIN), size=54).move_to([0, 1.95, 0])
        who = cite("Hu et al., 2021")

        self.say(
            "[Now, LoRA: low-rank adaptation, from twenty twenty-one.]",
            m.AnimationGroup(self.new_header("LoRA", "Learn the change, not the weights"), m.FadeIn(who)),
        )
        self.say(
            "[Take one weight matrix, W, with d rows and k columns], [and freeze it.]",
            m.AnimationGroup(m.FadeIn(w, scale=0.9), m.FadeIn(w_lab)),
            m.FadeIn(frozen, shift=0.15 * m.DOWN),
        )
        self.say(
            "[Fine-tuning would learn an update, delta W, the same size as W.]",
            m.AnimationGroup(m.Write(f), m.FadeIn(plus), m.FadeIn(dw, scale=0.9), m.FadeIn(dw_lab), m.FadeIn(learned)),
        )
        self.say(
            "[The key bet: this update has low rank], [so we can write it as B times A.]",
            m.Circumscribe(dw, color=TRAIN, buff=0.1),
            m.AnimationGroup(
                m.ReplacementTransform(f[0], f2[0]), m.ReplacementTransform(f[1], f2[1]), m.ReplacementTransform(f[2], f2[2:4]),
                m.ReplacementTransform(dw, m.VGroup(b, a)), m.FadeOut(dw_lab), m.FadeOut(learned),
            ),
        )
        self.say(
            "[B is tall and thin: d by r.] [A is short and wide: r by k.] [And the rank, r, is tiny, say eight.]",
            m.FadeIn(m.VGroup(b_lab, tall), shift=0.15 * m.UP),
            m.FadeIn(m.VGroup(a_lab, wide), shift=0.15 * m.UP),
            m.FadeIn(rank, scale=1.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ low rank, visually

    def low_rank(self):
        self.next_section(
            "Low rank, visually",
            notes=f"Seeded B (12 × 2) and A (2 × 12), numpy seed 41 (illustrative). BA is 12 × 12; singular values "
            f"{SV[0]:.2f}, {SV[1]:.2f}, then {SV[2]:.0e} (numerically zero). Amber = positive entries, teal = negative.",
        )
        cell = 0.24
        dw = framed(heat(SDW, cell, TRAIN, NEG), TRAIN, buff=0.05).move_to([-2.6, -0.95, 0])
        b = framed(heat(SB, cell, TRAIN, NEG), TRAIN, buff=0.05)
        b.next_to(dw, m.LEFT, buff=0.18)
        a = framed(heat(SA, cell, TRAIN, NEG), TRAIN, buff=0.05)
        a.next_to(dw, m.UP, buff=0.18)
        b_lab = m.VGroup(text("B", 32, TRAIN, weight=m.BOLD), text("12 × 2", 20, MUTED)).arrange(m.DOWN, buff=0.08).next_to(b, m.LEFT, buff=0.25)
        a_lab = m.VGroup(text("A", 32, TRAIN, weight=m.BOLD), text("2 × 12", 20, MUTED)).arrange(m.RIGHT, buff=0.2).next_to(a, m.UP, buff=0.18)
        dw_lab = text("BA · 12 × 12", 22, TRAIN, weight=m.SEMIBOLD).next_to(dw, m.DOWN, buff=0.22)
        legend = m.VGroup(
            m.Square(side_length=0.2).set_stroke(width=0).set_fill(TRAIN, 1), text("positive", 20, MUTED),
            m.Square(side_length=0.2).set_stroke(width=0).set_fill(NEG, 1), text("negative", 20, MUTED),
        ).arrange(m.RIGHT, buff=0.12)
        legend[2].shift(0.25 * m.RIGHT)
        legend[3].shift(0.25 * m.RIGHT)
        legend.next_to(dw_lab, m.DOWN, buff=0.2)

        scale = np.abs(SDW).max()
        small = 0.16
        l1 = framed(heat(LAYER1, small, TRAIN, NEG, scale=scale), TRAIN, buff=0.04).move_to([1.75, 0.55, 0])
        l2 = framed(heat(LAYER2, small, TRAIN, NEG, scale=scale), TRAIN, buff=0.04).move_to([4.85, 0.55, 0])
        eq = text("=", 44, INK, weight=m.BOLD).move_to([0.2, 0.55, 0])
        plus = text("+", 44, INK, weight=m.BOLD).move_to([3.3, 0.55, 0])
        cap1 = text("column 1 × row 1", 20, MUTED).next_to(l1, m.DOWN, buff=0.18)
        cap2 = text("column 2 × row 2", 20, MUTED).next_to(l2, m.DOWN, buff=0.18)
        grid_b, grid_a = b[1], a[1]

        def ring(cells, color=RANK):
            return m.SurroundingRectangle(m.VGroup(*cells), buff=0.03).set_stroke(color, width=3)

        col1 = ring([grid_b[2 * i] for i in range(12)])
        col2 = ring([grid_b[2 * i + 1] for i in range(12)])
        row1 = ring(list(grid_a[:12]))
        row2 = ring(list(grid_a[12:]))

        base_y, top = -3.2, 1.45
        xs = [0.75 + 0.5 * i for i in range(12)]
        bars = m.VGroup()
        for x, sv in zip(xs, SV):
            h = top * sv / SV[0]
            if h > 0.02:
                bars.add(m.Rectangle(width=0.3, height=h).set_stroke(width=0).set_fill(RANK, 1).move_to([x, base_y + h / 2, 0]))
            else:
                bars.add(m.Line([x - 0.15, base_y, 0], [x + 0.15, base_y, 0]).set_stroke(FAINT, width=4))
        axis = m.Line([0.45, base_y, 0], [6.45, base_y, 0]).set_stroke(FAINT, width=2)
        sv_lab = text("singular values of BA: only 2 are not zero", 20, MUTED).move_to([3.45, -1.45, 0])

        self.say(
            "[Here's a small example. B is twelve by two], [and A is two by twelve.]",
            m.AnimationGroup(self.new_header("LOW RANK", "A thin matrix times a wide one"), m.FadeIn(b, shift=0.2 * m.RIGHT), m.FadeIn(b_lab)),
            m.AnimationGroup(m.FadeIn(a, shift=0.2 * m.DOWN), m.FadeIn(a_lab)),
        )
        rows = [m.VGroup(*dw[1][12 * i : 12 * i + 12]) for i in range(12)]
        self.say(
            "[Multiply them, and you get a full twelve by twelve grid of changes.]",
            m.AnimationGroup(
                m.FadeIn(dw[0]),
                m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in rows], lag_ratio=0.12),
                m.FadeIn(dw_lab),
                m.FadeIn(legend),
            ),
        )
        self.say(
            "[But it holds just two patterns]: [column one of B times row one of A], [plus column two times row two.]",
            m.FadeIn(eq, scale=1.3),
            m.AnimationGroup(m.Create(col1), m.Create(row1), m.FadeIn(l1, target_position=dw), m.FadeIn(cap1)),
            m.AnimationGroup(
                m.ReplacementTransform(col1, col2), m.ReplacementTransform(row1, row2),
                m.FadeIn(plus), m.FadeIn(l2, target_position=dw), m.FadeIn(cap2),
            ),
        )
        self.say(
            "[As in episode eighteen, that makes its rank two]: [only two singular values are not zero.]",
            m.AnimationGroup(m.FadeOut(col2), m.FadeOut(row2), m.FadeIn(sv_lab), m.Create(axis)),
            m.LaggedStart(*[m.GrowFromEdge(x, m.DOWN) if isinstance(x, m.Rectangle) else m.FadeIn(x) for x in bars], lag_ratio=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the savings

    def savings(self):
        self.next_section(
            "Counting parameters",
            notes=f"d = k = 4096, r = 8: d·k = {FULL_COUNT:,} vs r(d + k) = {LORA_COUNT:,} ({LORA_SHARE:.3f}%). "
            f"LoRA on W_q and W_v in all 32 layers (Hu et al.'s main setting): {MODEL_LORA:,} parameters = {MODEL_SHARE:.3f}% "
            f"of 7B; {ADAPTER_MB:.1f} MB in 16-bit. B and A are drawn to scale (8/4096 of the square's side).",
        )
        side = 3.4
        sq = m.Rectangle(width=side, height=side).set_stroke(TRAIN, width=2, opacity=0.6).set_fill(TRAIN_FILL, 1).move_to([-4.0, -0.75, 0])
        sq_lab = m.VGroup(text("full update", 24, INK, weight=m.SEMIBOLD), text("4096 × 4096", 22, MUTED)).arrange(m.DOWN, buff=0.1).move_to(sq)
        thin = side * R / D
        b = m.Rectangle(width=thin, height=side).set_stroke(width=0).set_fill(TRAIN, 1).next_to(sq, m.LEFT, buff=0.16)
        a = m.Rectangle(width=side, height=thin).set_stroke(width=0).set_fill(TRAIN, 1).next_to(sq, m.UP, buff=0.16)
        b_tag = text("B", 26, TRAIN, weight=m.BOLD).next_to(b, m.LEFT, buff=0.2)
        a_tag = text("A", 26, TRAIN, weight=m.BOLD).next_to(a, m.UP, buff=0.15)
        to_scale = text("B and A drawn to scale: 8 of 4096", 20, MUTED).next_to(sq, m.DOWN, buff=0.25)

        cx = 3.0
        full_name = text("full", 24, MUTED, weight=m.SEMIBOLD)
        full = formula((r"d \times k", INK), ("=", INK), size=44)
        full = m.VGroup(full, text("16,777,216", 40, INK, weight=m.SEMIBOLD).next_to(full, m.RIGHT, buff=0.25).align_to(full[0], m.DOWN))
        lora_name = text("LoRA", 24, TRAIN, weight=m.SEMIBOLD)
        lora_f = formula(("r", RANK), (r"\,(d + k)", INK), ("=", INK), size=44)
        lora_f = m.VGroup(lora_f, text("65,536", 40, TRAIN, weight=m.SEMIBOLD).next_to(lora_f, m.RIGHT, buff=0.25).align_to(lora_f[0], m.DOWN))
        full.move_to([cx + 0.55, 1.35, 0])
        lora_f.move_to([cx + 0.55, 0.3, 0])
        lora_f.align_to(full, m.LEFT)
        full_name.next_to(full, m.LEFT, buff=0.35)
        lora_name.next_to(lora_f, m.LEFT, buff=0.35)
        lora_name.align_to(full_name, m.RIGHT)
        share = pill("0.39% of the full update", GOOD, GOOD_FILL, 24).move_to([cx, -0.75, 0])
        model = m.VGroup(
            text("query and value matrices × 32 layers", 22, INK),
            text("4,194,304 trainable  ·  0.06% of 7B", 22, TRAIN, weight=m.SEMIBOLD),
        ).arrange(m.DOWN, buff=0.12).move_to([cx, -1.95, 0])
        file = pill("your adapter: ≈ 8 MB in 16-bit", SKY, SKY_FILL, 22).move_to([cx, -3.1, 0])

        self.say(
            "[How much does this save? Take a layer four thousand and ninety-six wide, with rank eight.]",
            m.AnimationGroup(self.new_header("THE SAVINGS", "Count the parameters"), m.FadeIn(sq), m.FadeIn(sq_lab)),
        )
        self.say(
            "[A full update has d times k numbers]: [about sixteen point eight million.]",
            m.AnimationGroup(m.FadeIn(full_name), m.Write(full[0])),
            m.FadeIn(full[1], shift=0.15 * m.LEFT),
        )
        self.say(
            "[The low-rank version trains r times, d plus k]: [just sixty-five thousand, five hundred and thirty-six.]",
            m.AnimationGroup(m.FadeIn(lora_name), m.Write(lora_f[0]), m.FadeIn(b), m.FadeIn(a)),
            m.FadeIn(lora_f[1], shift=0.15 * m.LEFT),
        )
        self.say(
            "[That's zero point three nine percent.] [Drawn to scale, B and A are hairlines.]",
            m.FadeIn(share, scale=1.2),
            m.AnimationGroup(m.FadeIn(b_tag, shift=0.15 * m.RIGHT), m.FadeIn(a_tag, shift=0.15 * m.DOWN), m.FadeIn(to_scale)),
        )
        self.say(
            "[Put LoRA on two attention matrices in each of thirty-two layers], "
            "[and our seven-billion model trains about four point two million weights], [under a tenth of a percent.]",
            m.FadeIn(model[0], shift=0.15 * m.UP),
            m.FadeIn(model[1], shift=0.15 * m.UP),
            m.Indicate(model[1], color=TRAIN, scale_factor=1.06),
        )
        self.say("[Saved in sixteen-bit, your personal adapter is about eight megabytes.]", m.FadeIn(file, shift=0.2 * m.UP))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ training

    def training(self):
        self.next_section(
            "Training",
            notes="Forward pass h = Wx + BAx (Hu et al. also scale BAx by α/r). A: random Gaussian init, B: zero, so BA = 0 and "
            "training starts from the pre-trained model. Gradients still pass through W to reach the layers below, but W gets "
            "no weight gradient and no optimizer state.",
        )
        f = formula(("h", INK), ("=", INK), ("W", FROZEN), ("x", SKY), ("+", INK), ("B", TRAIN), ("A", TRAIN), ("x", SKY), size=50)
        f.move_to([-1.2, 1.95, 0])
        ym, yw, yl = -0.9, 0.15, -2.4
        x_chip = box("x", 0.75, 1.3, SKY, SKY_FILL, size=30, weight=m.BOLD).move_to([-5.9, ym, 0])
        wbox = card(2.2, 1.6, FROZEN, FROZEN_FILL).move_to([-1.8, yw, 0])
        wtext = m.VGroup(text("W", 34, FROZEN, weight=m.BOLD), text("d × k", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to(wbox)
        wlock = lock(0.45).move_to(wbox.get_corner(m.UR) + np.array([-0.05, -0.05, 0]))
        wgroup = m.VGroup(wbox, wtext, wlock)
        a_shape = trapezoid(-2.65, yl, 1.3, 1.2, 0.34, TRAIN, TRAIN_FILL)
        b_shape = trapezoid(-0.95, yl, 1.3, 0.34, 1.2, TRAIN, TRAIN_FILL)
        a_t = text("A", 28, TRAIN, weight=m.BOLD).move_to(a_shape.get_center() + 0.2 * m.LEFT)
        b_t = text("B", 28, TRAIN, weight=m.BOLD).move_to(b_shape.get_center() + 0.2 * m.RIGHT)
        r_t = text("r", 26, RANK, weight=m.BOLD).move_to([-1.8, yl + 0.5, 0])
        detour = m.VGroup(a_shape, b_shape, a_t, b_t, r_t)
        split = -4.4
        sx = 1.3
        path_w = poly((-5.52, ym), (split, ym), (split, yw), (-2.9, yw), color=SKY)
        path_a = poly((split, ym), (split, yl), (-3.3, yl), color=SKY)
        join = oplus([sx, ym, 0])
        out_w = poly((-0.7, yw), (sx, yw), (sx, ym + 0.2), color=FROZEN)
        out_b = poly((-0.3, yl), (sx, yl), (sx, ym - 0.2), color=TRAIN)
        h_chip = box("h", 0.75, 1.3, INK, CARD, size=30, weight=m.BOLD).move_to([2.6, ym, 0])
        out_h = m.Line([sx + 0.2, ym, 0], h_chip.get_left()).set_stroke(MUTED, width=3)
        nx = 3.75
        n1 = two_lines("W: frozen", "no weight gradients", FROZEN).move_to([0, 0.75, 0]).align_to([nx, 0, 0], m.LEFT)
        n2 = two_lines("A and B: trained", "gradients + Adam state", TRAIN).move_to([0, -0.45, 0]).align_to([nx, 0, 0], m.LEFT)
        n3 = two_lines("B starts at zero", "so BAx = 0 at step 0", TRAIN).move_to([0, -1.65, 0]).align_to([nx, 0, 0], m.LEFT)
        zero_b = text("B = 0", 22, TRAIN, weight=m.SEMIBOLD).next_to(b_shape, m.DOWN, buff=0.18)

        self.say(
            "[During training, the input x takes two paths]: [through the frozen W], [and through the thin detour, A, then B.]",
            m.AnimationGroup(self.new_header("TRAINING", "Two paths, one of them frozen"), m.Write(f), m.FadeIn(x_chip, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.Create(path_w), m.FadeIn(wgroup)),
            m.AnimationGroup(m.Create(path_a), m.FadeIn(detour)),
        )
        self.say(
            "[The two outputs are added.] [Only A and B are updated, so only they need gradients and Adam state.]",
            m.AnimationGroup(m.Create(out_w), m.Create(out_b), m.FadeIn(join), m.Create(out_h), m.FadeIn(h_chip)),
            m.AnimationGroup(m.FadeIn(n1, shift=0.15 * m.LEFT), m.FadeIn(n2, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[A starts random, but B starts at zero.] [So at the first step, the detour adds nothing: training begins exactly at the original model.]",
            m.AnimationGroup(m.FadeIn(n3, shift=0.15 * m.LEFT), m.FadeIn(zero_b, shift=0.15 * m.UP)),
            m.Circumscribe(f[5:], color=TRAIN, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ merging

    def merging(self):
        self.next_section(
            "Merge or swap",
            notes="After training, W' = W + BA has W's shape: no extra layers and no added inference latency (Hu et al., 2021). "
            "Or keep one frozen base and swap small adapters per user or task. Adapter names are illustrative.",
        )
        cell = 0.26
        w = framed(heat(W10, cell, FROZEN), FROZEN).move_to([-5.15, -0.7, 0])
        plus = text("+", 44, INK, weight=m.BOLD).move_to([-3.5, -0.7, 0])
        ba = framed(heat(DW10, cell, TRAIN, NEG), TRAIN).move_to([-1.85, -0.7, 0])
        merged_values = W10 + 0.35 * DW10
        merged = framed(heat(merged_values, cell, GOOD), GOOD).move_to([-3.5, -0.7, 0])
        w_lab = text("W", 26, FROZEN, weight=m.BOLD).next_to(w, m.DOWN, buff=0.2)
        ba_lab = text("BA", 26, TRAIN, weight=m.BOLD).next_to(ba, m.DOWN, buff=0.2)
        m_lab = text("W′ · d × k, the same shape", 22, GOOD, weight=m.SEMIBOLD).next_to(merged, m.DOWN, buff=0.2)
        f = formula(("W'", GOOD), ("=", INK), ("W", FROZEN), ("+", INK), ("B", TRAIN), ("A", TRAIN), size=54).move_to([-3.5, 1.65, 0])
        fast = pill("no extra layers · no added latency", GOOD, GOOD_FILL, 22).move_to([-3.5, -3.05, 0])

        base = card(4.6, 1.5, FROZEN, FROZEN_FILL).move_to([3.95, 1.1, 0])
        base_body = m.VGroup(icon("llm", 0.9), two_lines("one frozen base", "7B parameters, one copy", INK)).arrange(m.RIGHT, buff=0.35).move_to(base)
        chips = m.VGroup(
            *[
                m.VGroup(card(1.7, 0.95, c, f_, radius=0.18, stroke=2.5), m.VGroup(text(n, 22, INK, weight=m.SEMIBOLD), text("≈ 8 MB", 20, MUTED)).arrange(m.DOWN, buff=0.06))
                for n, c, f_ in (("your style", SKY, SKY_FILL), ("legal", TRAIN, TRAIN_FILL), ("support", TRAIN, TRAIN_FILL))
            ]
        )
        for ch in chips:
            ch[1].move_to(ch[0])
        chips.arrange(m.RIGHT, buff=0.25).move_to([3.95, -1.2, 0])
        links = m.VGroup(*[m.DashedLine(ch.get_top(), [ch.get_x(), base.get_bottom()[1], 0], dash_length=0.08).set_stroke(MUTED, width=2) for ch in chips])
        swap = text("swap in one adapter per user or task", 20, MUTED).next_to(chips, m.DOWN, buff=0.3)

        self.say(
            "[After training, fold the update in]: [W prime equals W plus B A], [one matrix, the very same shape.]",
            m.AnimationGroup(
                self.new_header("AFTER TRAINING", "Merge, or swap"),
                m.FadeIn(w), m.FadeIn(plus), m.FadeIn(ba), m.FadeIn(w_lab), m.FadeIn(ba_lab),
            ),
            m.Write(f),
            m.AnimationGroup(m.ReplacementTransform(m.VGroup(w, plus, ba), merged), m.FadeOut(w_lab), m.FadeOut(ba_lab), m.FadeIn(m_lab)),
        )
        self.say(
            "[So the merged model runs exactly as fast as the original]: [no extra layers, no added latency.]",
            m.Circumscribe(merged, color=GOOD, buff=0.1),
            m.FadeIn(fast, shift=0.2 * m.UP),
        )
        self.say(
            "[Or keep them separate]: [one frozen base, and a small adapter for each user or task, swapped in when needed.]",
            m.FadeIn(m.VGroup(base, base_body), shift=0.2 * m.DOWN),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in chips], lag_ratio=0.3), m.Create(links), m.FadeIn(swap)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ QLoRA

    def qlora(self):
        self.next_section(
            "QLoRA",
            notes=f"Dettmers et al., 2023 (QLoRA): frozen base in 4-bit NormalFloat (NF4), double quantisation, paged optimizers; "
            f"LoRA adapters (on every linear layer) in 16-bit, trained by backpropagating through the 4-bit weights; a 65B model "
            f"fine-tuned on one 48 GB GPU. Our 7B: LoRA with a 16-bit base {MEM_LORA16:.2f} GB; QLoRA {MEM_4BIT:.1f} + "
            f"{MEM_CONST:.2f} (constants) + {MEM_ADAPTER:.3f} (adapter + Adam) = {MEM_QLORA:.2f} GB, activations not counted.",
        )
        x0, s = -3.4, 0.46
        rows = {"full": 1.55, "lora": 0.45, "qlora": -0.65}
        lx = x0 + LAPTOP_GB * s
        full_bar = m.Rectangle(width=6.55 - x0, height=0.62).set_stroke(BAD, width=2).set_fill(BAD_FILL, 1).move_to([(x0 + 6.55) / 2, rows["full"], 0])
        full_in = text("112 GB, far off the chart →", 22, INK).move_to([0.35, rows["full"], 0])
        lora_base = hbar(x0, x0 + MEM_W * s, rows["lora"], FROZEN, FROZEN_FILL)
        lora_ad = hbar(x0 + MEM_W * s, x0 + MEM_W * s + 0.08, rows["lora"], TRAIN, TRAIN)
        q_base = hbar(x0, x0 + MEM_4BIT * s, rows["qlora"], FROZEN, FROZEN_FILL)
        q_ad = hbar(x0 + (MEM_4BIT + MEM_CONST) * s, x0 + (MEM_4BIT + MEM_CONST) * s + 0.08, rows["qlora"], TRAIN, TRAIN)
        names = m.VGroup(
            text("full fine-tuning", 22, INK, weight=m.SEMIBOLD).move_to([0, rows["full"], 0]),
            text("LoRA · 16-bit base", 22, INK, weight=m.SEMIBOLD).move_to([0, rows["lora"], 0]),
            text("QLoRA · 4-bit base", 22, INK, weight=m.SEMIBOLD).move_to([0, rows["qlora"], 0]),
        )
        for n in names:
            n.align_to([x0 - 0.25, 0, 0], m.RIGHT)
        lora_val = text(f"≈ {MEM_LORA16:.0f} GB", 22, INK)
        lora_val.move_to([lora_base.get_right()[0] - 0.2 - lora_val.width / 2, rows["lora"], 0])
        q_val = text("< 4 GB", 22, GOOD, weight=m.BOLD).next_to(q_ad, m.RIGHT, buff=0.2)
        nf4 = pill("NF4: 4-bit NormalFloat", VIOLET, VIOLET_FILL, 20).next_to(q_val, m.RIGHT, buff=0.35)
        line = m.DashedLine([lx, rows["full"] - 0.4, 0], [lx, rows["qlora"] - 0.5, 0], dash_length=0.1).set_stroke(SKY, width=3).set_z_index(2)
        lap = m.VGroup(icon("ep41-laptop", 0.4), text("laptop: 16 GB", 22, SKY, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.18)
        lap.next_to(line, m.DOWN, buff=0.12)
        lap.shift((lx - lap[0].get_center()[0]) * m.RIGHT)
        flow = m.ShowPassingFlash(
            m.Line(q_ad.get_right() + 0.05 * m.RIGHT, q_base.get_left()).set_stroke(TRAIN, width=8), time_width=0.6, run_time=1.6
        )
        gpu = m.VGroup(
            icon("ep41-gpu", 0.85),
            two_lines("65B model, one 48 GB GPU", "NF4 · double quantization · paged optimizers", GOOD, top_size=26),
        ).arrange(m.RIGHT, buff=0.35).move_to([-2.6, -2.75, 0])
        ours = pill("our 7B: fits on a laptop", SKY, SKY_FILL, 22).move_to([4.0, -2.75, 0])
        who = cite("Dettmers et al., 2023")

        self.say(
            "[So, can we shrink the frozen base too?] [With LoRA alone, it's still fourteen gigabytes of sixteen-bit weights.]",
            m.AnimationGroup(
                self.new_header("QLoRA", "A 4-bit base, plus LoRA"),
                m.FadeIn(names[0]), m.FadeIn(full_bar), m.FadeIn(full_in), m.Create(line), m.FadeIn(lap),
            ),
            m.AnimationGroup(m.FadeIn(names[1]), m.GrowFromEdge(lora_base, m.LEFT), m.FadeIn(lora_ad), m.FadeIn(lora_val)),
        )
        self.say(
            "[Quantized LoRA, from twenty twenty-three, stores those frozen weights in just four bits each], [using a data type called four-bit NormalFloat.]",
            m.AnimationGroup(m.FadeIn(who), m.FadeIn(names[2]), m.GrowFromEdge(q_base, m.LEFT), m.FadeIn(q_ad), m.FadeIn(q_val)),
            m.FadeIn(nf4, shift=0.15 * m.LEFT),
        )
        self.say(
            "[The LoRA matrices stay in sixteen-bit, and gradients flow back through the four-bit weights into them.]",
            flow,
        )
        self.say(
            "[That let the authors fine-tune a sixty-five-billion-parameter model on a single forty-eight-gigabyte GPU.]",
            m.FadeIn(gpu, shift=0.2 * m.UP),
        )
        self.say(
            "[Our seven-billion model needs under four gigabytes, before activations]: [personalising it on a laptop is within reach.]",
            m.Circumscribe(m.VGroup(q_base, q_val), color=GOOD, buff=0.1),
            m.FadeIn(ours, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section(
            "The catch",
            notes="Biderman et al., 2024 (LoRA Learns Less and Forgets Less): on code and math, LoRA underperforms full fine-tuning "
            "on the target domain but better preserves performance outside it; full fine-tuning's weight perturbations have "
            "10–100× higher rank than typical LoRA configurations. Rank, target matrices and scale α are hyperparameters.",
        )
        cols = {"full": 0.6, "lora": 4.1}
        heads = on_baseline(m.VGroup(
            text("full fine-tuning", 26, INK, weight=m.BOLD).move_to([cols["full"], 1.6, 0]),
            text("LoRA", 26, TRAIN, weight=m.BOLD).move_to([cols["lora"], 1.6, 0]),
        ))
        rule = m.Line([-6.4, 1.15, 0], [5.6, 1.15, 0]).set_stroke(EDGE, width=2)
        q1 = two_lines("learning a new domain", "e.g. code or maths", INK, top_size=26).move_to([0, 0.4, 0]).align_to([-6.4, 0, 0], m.LEFT)
        q2 = two_lines("keeping what it knew", "skills outside that domain", INK, top_size=26).move_to([0, -1.0, 0]).align_to([-6.4, 0, 0], m.LEFT)
        r1 = m.VGroup(pill("learns more", GOOD, GOOD_FILL, 22).move_to([cols["full"], 0.4, 0]), pill("learns less", BAD, BAD_FILL, 22).move_to([cols["lora"], 0.4, 0]))
        r2 = m.VGroup(pill("forgets more", BAD, BAD_FILL, 22).move_to([cols["full"], -1.0, 0]), pill("forgets less", GOOD, GOOD_FILL, 22).move_to([cols["lora"], -1.0, 0]))
        note = text("For style and format, a small rank is often enough.", 28, INK, weight=m.SEMIBOLD).move_to([0, -2.45, 0])
        knobs = text("rank, which matrices, and the scale are settings to tune", 20, MUTED).next_to(note, m.DOWN, buff=0.22)
        who = cite("Biderman et al., 2024")

        self.say(
            "[A low-rank update has limits.] [On big shifts, like learning to code or do maths, full fine-tuning learns more.]",
            m.AnimationGroup(self.new_header("THE CATCH", "Learns less, forgets less"), m.FadeIn(who), m.FadeIn(heads), m.Create(rule)),
            m.AnimationGroup(m.FadeIn(q1, shift=0.15 * m.RIGHT), m.FadeIn(r1, shift=0.15 * m.UP)),
        )
        self.say(
            "[But low-rank updates forget less of what the model already knew.]",
            m.AnimationGroup(m.FadeIn(q2, shift=0.15 * m.RIGHT), m.FadeIn(r2, shift=0.15 * m.UP)),
        )
        self.say(
            "[For style and format, which is what personalising needs, a small rank is often enough.]",
            m.AnimationGroup(m.FadeIn(note, shift=0.2 * m.UP), m.FadeIn(knobs)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Freeze, factor, merge.")
        halo = glow(RANK, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 60, c, weight=m.BOLD) for w, c in zip(("Freeze", "Factor", "Merge"), (FROZEN, TRAIN, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Big models, small updates: fine-tuning that fits on a laptop.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [freeze], [factor], [merge.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Big models, small updates: fine-tuning that fits on a laptop.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
