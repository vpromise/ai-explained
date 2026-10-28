"""AI, Explained · Episode 42 · Quantization & Distillation

Running example: a language model that fits on a phone. Memory = parameters × bytes per parameter:
7B at 16-bit is 14 GB, at int8 7 GB, at int4 3.5 GB (3.6 GB with one 16-bit scale per 128 weights);
the phone's 8 GB is illustrative.

- Quantization: symmetric absmax, q = clamp(round(w / s), -127, 127), s = max|w| / 127, on a seeded
  weight row (4,096 weights, N(0, 0.02), seed 42). Relative rounding error ||w - s·q|| / ||w|| for
  8 down to 4 bits, and for int4 with a scale per group of 128.
- Outliers: one token's 4,096 activations (N(0, 1), seed 7) with four outlier dimensions set to
  60, -45, 52, 38 (illustrative; Dettmers' LLM.int8() write-up reports outliers around 60 in 13B
  models). One int8 scale vs. LLM.int8()'s mixed-precision decomposition (Dettmers et al., 2022).
- GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2023) described, not computed.
- Distillation (Hinton et al., 2015): illustrative teacher logits for "My phone's battery is ...",
  softmax at T = 1 and T = 4 (Episode 16), and a student whose logits follow gradient descent on
  T² · KL(teacher_T || student_T). DistilBERT figures: Sanh et al., 2019.

Every number the narration quotes is computed below and asserted.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# colours: one meaning each for the whole film
FP16, FP16_FILL = SKY, SKY_FILL  # 16-bit originals
INT8, INT8_FILL = TEAL, TEAL_FILL
INT4, INT4_FILL = AMBER, AMBER_FILL
SCALE, SCALE_FILL = LIME, "#1C2A10"
OUTLIER, OUTLIER_FILL = BAD, BAD_FILL
TEACHER, TEACHER_FILL = VIOLET, VIOLET_FILL
STUDENT, STUDENT_FILL = BLUE, "#14223D"

# ---------------------------------------------------------------------------- the data


def gigabytes(params, bits):
    return params * bits / 8 / 1e9


PARAMS, STUDENT_PARAMS, PHONE_GB = 7e9, 1e9, 8  # the phone's 8 GB is illustrative
GROUP = 128
BITS_GROUPED = 4 + 16 / GROUP  # int4 plus one 16-bit scale per group of 128
MEM = {b: gigabytes(PARAMS, b) for b in (16, 8, 4)}
MEM_GROUPED = gigabytes(PARAMS, BITS_GROUPED)
STUDENT_16 = gigabytes(STUDENT_PARAMS, 16)
STUDENT_GB = gigabytes(STUDENT_PARAMS, BITS_GROUPED)
assert (MEM[16], MEM[8], MEM[4]) == (14.0, 7.0, 3.5)  # "fourteen", "seven", "three and a half"
assert round(MEM_GROUPED, 1) == 3.6 and MEM_GROUPED < PHONE_GB < MEM[16]  # "it fits" / "can't load"
assert abs(STUDENT_GB - 0.5) < 0.05 and STUDENT_16 == 2.0  # "about half a gigabyte"


def quantize(w, bits, group=None):
    """Symmetric absmax: one scale per row (or per group); returns read-back weights, q, scales."""
    qmax = 2 ** (bits - 1) - 1
    rows = w.reshape(-1, group or w.size)
    s = np.abs(rows).max(axis=1, keepdims=True) / qmax
    q = np.clip(np.round(rows / s), -qmax, qmax)
    return (q * s).ravel(), q.ravel().astype(int), s.ravel()


def rel_error(w, w_hat):
    return float(np.linalg.norm(w - w_hat) / np.linalg.norm(w))


W = np.random.default_rng(42).normal(0.0, 0.02, 4096)  # one row of weights (illustrative)
W_MAX = float(np.abs(W).max())
S8 = W_MAX / 127
SHOW = [0, 1, 2, 3, int(np.argmax(np.abs(W)))]  # four weights and the largest one
W_HAT8, Q8, _ = quantize(W, 8)
ERR = {b: rel_error(W, quantize(W, b)[0]) for b in range(4, 9)}
ERR_GROUPED = rel_error(W, quantize(W, 4, GROUP)[0])
assert Q8[SHOW[-1]] in (-127, 127)  # the largest weight lands on the end of the range
assert np.all(np.abs(W - W_HAT8) <= S8 / 2 + 1e-12)  # "at most half a step"
assert ERR[8] < 0.01 and round(100 * ERR[4]) == 15  # "under one percent", "about fifteen"
assert all(1.9 < ERR[b - 1] / ERR[b] < 2.2 for b in range(5, 9))  # "every bit doubles the error"
assert round(100 * ERR_GROUPED) == 11  # "about eleven percent"

_ar = np.random.default_rng(7)
ACT = _ar.normal(0.0, 1.0, 4096)  # one token's activations (illustrative)
OUT_IDX = _ar.choice(4096, 4, replace=False)
OUT_VALUES = (60.0, -45.0, 52.0, 38.0)
ACT[OUT_IDX] = OUT_VALUES
ORD = np.delete(ACT, OUT_IDX)  # the ordinary values
S_OUT = float(np.abs(ACT).max() / 127)  # one int8 scale for everything
S_ORD = float(np.abs(ORD).max() / 127)  # the scale once the outliers are kept aside
Q_OUT = np.round(ORD / S_OUT)
LEVELS = sorted(set(Q_OUT.astype(int)))
ERR_SHARED = rel_error(ORD, Q_OUT * S_OUT)
ERR_SPLIT = rel_error(ORD, quantize(ORD, 8)[0])
KEPT_INT8 = ORD.size / ACT.size
STRIP = ORD[:64]  # the dots on screen
assert len(LEVELS) == 16  # "just sixteen levels"
assert ERR_SPLIT < 0.01 and round(100 * ERR_SHARED) == 14  # "under one percent to about fourteen"
assert KEPT_INT8 > 0.999  # "over ninety-nine point nine percent"
assert np.abs(STRIP).max() < 4.0

WORDS = ("dead", "low", "flat", "charging", "purple")
Z_TEACHER = np.array([8.0, 6.2, 5.0, 2.5, -4.0])  # illustrative teacher logits
T_HOT = 4.0


def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T)
    return e / e.sum()


P_COLD, P_HOT = softmax(Z_TEACHER, 1.0), softmax(Z_TEACHER, T_HOT)
assert round(100 * P_COLD[0]) == 82  # "eighty-two percent"
assert P_COLD[3] < 0.005 and P_COLD[4] < 0.001 and P_HOT[3] > 0.1 and P_HOT[4] > 0.01


def distill_loss(z):
    p_s = softmax(z, T_HOT)
    return float(T_HOT**2 * np.sum(P_HOT * np.log(P_HOT / p_s)))


def distill(steps=40, rate=2.0):
    """Gradient descent on T² · KL(teacher_T || student_T); d/dz = T · (p_student - p_teacher)."""
    z, path = np.zeros(len(Z_TEACHER)), []
    for _ in range(steps + 1):
        path.append(z.copy())
        z = z - rate * T_HOT * (softmax(z, T_HOT) - P_HOT)
    return np.array(path)


STUDENT_PATH = distill()
LOSSES = [distill_loss(z) for z in STUDENT_PATH]
assert np.allclose(softmax(STUDENT_PATH[0], T_HOT), 0.2)  # "spreads its bets evenly"
assert LOSSES[-1] < 0.01 < LOSSES[0]  # "falls to almost zero"

# ---------------------------------------------------------------------------- helpers


def ep_recolored(name, color, fill):
    """kit.recolored(), written as assets/ep42-<name>-<color>.svg (this episode's files only)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep42-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


def cite(s):
    return text(s, 20, MUTED)


def num(v, digits=4):
    return f"{v:.{digits}f}".replace("-", "−")


def polyline(points, color, width=4):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width)


def bar_up(x, base, height, color, width=0.78):
    h = max(height, 0.012)
    return m.Rectangle(width=width, height=h).set_fill(color, opacity=0.85).set_stroke(color, width=2).move_to([x, base + h / 2, 0])


def bar_right(x0, y, length, color, height=0.5):
    w = max(length, 0.012)
    return m.Rectangle(width=w, height=height).set_fill(color, opacity=0.85).set_stroke(color, width=2).move_to([x0 + w / 2, y, 0])


def cell(s, color, fill, width=1.55, height=0.72, size=24):
    box = card(width, height, color, fill, radius=0.15, stroke=2)
    return m.VGroup(box, code(s, size, INK).move_to(box))


def checkmark(height=0.6):
    return icon("check", height)


# ---------------------------------------------------------------------------- the episode


class Ep42QuantizationDistillation(Episode):
    number = 42

    def lesson(self):
        self.teacher_icon = ep_recolored("llm", TEACHER, TEACHER_FILL)
        self.student_icon = ep_recolored("llm", STUDENT, STUDENT_FILL)
        self.hook()
        self.budget()
        self.quantization()
        self.rounding_error()
        self.outliers()
        self.post_training()
        self.soft_labels()
        self.student()
        self.pocket()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A language model that runs on a phone. 7B is a common open-model size; 7e9 × 2 bytes = 14 GB.")
        phone = icon("ep42-phone", 3.6).move_to([3.7, 0.4, 0])
        chip = icon(self.teacher_icon, 0.95).move_to(phone[1])
        model = card(4.4, 3.0, TEACHER, TEACHER_FILL).move_to([-3.4, 0.4, 0])
        body = m.VGroup(
            icon(self.teacher_icon, 1.05),
            text("7 billion parameters", 28, INK, weight=m.BOLD),
            text("16-bit weights", 22, MUTED),
        ).arrange(m.DOWN, buff=0.2).move_to(model)
        size = pill("≈ 14 GB", BAD, BAD_FILL, 24).next_to(model, m.DOWN, buff=0.35)
        stamp = pill("✗ doesn't fit", BAD, BAD_FILL, 26).move_to([0.75, 0.4, 0])
        tools = m.VGroup(
            pill("quantization · fewer bits per weight", INT8, INT8_FILL, 24),
            pill("distillation · fewer weights", STUDENT, STUDENT_FILL, 24),
        ).arrange(m.RIGHT, buff=0.5).move_to([0, -2.95, 0])

        self.say(
            "[Imagine a language model that runs right on your phone]: [no server, no connection, just the chip in your pocket.]",
            m.FadeIn(phone, shift=0.3 * m.UP),
            m.FadeIn(chip, scale=0.6),
        )
        self.say(
            "[But a typical open model has seven billion parameters]: [about fourteen gigabytes.] [It doesn't fit.]",
            m.AnimationGroup(m.FadeIn(model, shift=0.3 * m.RIGHT), m.FadeIn(body, shift=0.3 * m.RIGHT)),
            m.FadeIn(size, shift=0.2 * m.UP),
            m.Succession(m.FadeIn(stamp, scale=1.2), m.Wiggle(stamp)),
        )
        self.say(
            "[Today, two ways to shrink it]: [quantization stores every weight in fewer bits], [and distillation trains a smaller model.]",
            m.FadeOut(stamp),
            m.FadeIn(tools[0], shift=0.2 * m.UP),
            m.FadeIn(tools[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the budget

    def budget(self):
        self.next_section(
            "The budget",
            notes=f"Memory = parameters × bytes. 7B: 16-bit {MEM[16]:.1f} GB, int8 {MEM[8]:.1f} GB, int4 {MEM[4]:.2f} GB, "
            f"int4 + one fp16 scale per {GROUP} weights ({BITS_GROUPED} bits/weight) {MEM_GROUPED:.2f} GB. "
            "Decimal gigabytes (1e9 bytes); weights only (no KV cache or activations). The phone's 8 GB is illustrative.",
        )
        formula = m.MathTex(r"\text{memory}", "=", r"\text{parameters}", r"\times", r"\text{bytes per parameter}", font_size=42, color=INK)
        formula.move_to([0, 1.95, 0])
        x0, per_gb, ys = -3.4, 0.48, (0.65, -0.5, -1.65)
        specs = [(16, "16-bit", "2 bytes", FP16), (8, "int8", "1 byte", INT8), (4, "int4", "½ byte", INT4)]
        labels, bars, values = [], [], []
        for (bits, name, sub, color), y in zip(specs, ys):
            lab = m.VGroup(text(name, 26, color, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
            labels.append(lab.move_to([x0 - 0.3, y, 0]).align_to([x0 - 0.3, 0, 0], m.RIGHT))
            bars.append(bar_right(x0, y, MEM[bits] * per_gb, color))
            values.append(text(f"{MEM[bits]:g} GB", 26, INK, weight=m.BOLD).move_to([3.9, y, 0]).align_to([3.9, 0, 0], m.LEFT))
        limit_x = x0 + PHONE_GB * per_gb
        limit = m.DashedLine([limit_x, 1.15, 0], [limit_x, -2.15, 0], dash_length=0.12).set_stroke(SKY, width=3)
        limit_label = m.VGroup(icon("ep42-phone", 0.55), text("phone: 8 GB, for example", 22, SKY, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15)
        limit_label.next_to(limit.get_end(), m.DOWN, buff=0.15)
        over = m.Rectangle(width=(MEM[16] - PHONE_GB) * per_gb, height=0.5).set_fill(BAD, opacity=0.55).set_stroke(BAD, width=2.5)
        over.move_to([limit_x + (MEM[16] - PHONE_GB) * per_gb / 2, ys[0], 0])
        grouped = m.VGroup(text(f"{MEM_GROUPED:.1f} GB", 26, GOOD, weight=m.BOLD), text("with scales", 20, MUTED))
        grouped.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT).move_to([3.9, ys[2], 0]).align_to([3.9, 0, 0], m.LEFT)
        tick = checkmark(0.55).move_to([x0 + MEM[4] * per_gb + 0.5, ys[2], 0])

        self.say(
            "[A model's memory is simple]: [its parameter count, times the bytes per parameter.]",
            self.new_header("THE BUDGET", "Parameters × bytes"),
            m.Write(formula),
        )
        self.say(
            "[Seven billion parameters at sixteen bits, two bytes each], [take fourteen gigabytes.]",
            m.FadeIn(labels[0], shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(values[0])),
        )
        self.say(
            "[A phone with, for example, eight gigabytes, shared with everything else], [can't even load that.]",
            m.AnimationGroup(m.Create(limit), m.FadeIn(limit_label, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(over), values[0].animate.set_color(BAD)),
        )
        self.say(
            "[At eight bits, it's seven gigabytes.] [At four bits, three and a half.]",
            m.AnimationGroup(m.FadeIn(labels[1], shift=0.2 * m.RIGHT), m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(values[1])),
            m.AnimationGroup(m.FadeIn(labels[2], shift=0.2 * m.RIGHT), m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(values[2])),
        )
        self.say(
            "[Add a little for the scales we're about to meet], [and it fits.]",
            m.ReplacementTransform(values[2], grouped),
            m.FadeIn(tick, scale=0.5),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ quantization

    def quantization(self):
        shown = ", ".join(num(W[i]) for i in SHOW)
        self.next_section(
            "Quantization",
            notes=f"Symmetric absmax int8 on a seeded row of 4,096 weights (N(0, 0.02), seed 42, illustrative). "
            f"Shown: {shown}; s = {W_MAX:.4f}/127 = {S8:.6f}; q = {', '.join(str(Q8[i]) for i in SHOW)}.",
        )
        formula = m.MathTex(
            "q", "=", r"\mathrm{clamp}(", r"\mathrm{round}(", "w", "/", "s", ")", r",\; -127,\; 127)", font_size=46, color=INK
        ).move_to([0, 1.95, 0])
        formula[0].set_color(INT8)
        formula[4].set_color(FP16)
        formula[6].set_color(SCALE)
        xs, ys = (-3.0, -1.2, 0.6, 2.4, 4.2), (0.6, -0.5, -1.6)
        row_names = [
            tex("w", 38, FP16),
            tex(r"w / s", 34, MUTED),
            tex("q", 38, INT8),
        ]
        for lab, y in zip(row_names, ys):
            lab.move_to([-5.2, y, 0])
        r1 = [cell(num(W[i]), FP16, FP16_FILL).move_to([x, ys[0], 0]) for i, x in zip(SHOW, xs)]
        r2 = [cell(num(W[i] / S8, 1), EDGE, CARD).move_to([x, ys[1], 0]) for i, x in zip(SHOW, xs)]
        r3 = [cell(str(Q8[i]).replace("-", "−"), INT8, INT8_FILL).move_to([x, ys[2], 0]) for i, x in zip(SHOW, xs)]
        largest = text("largest", 20, SCALE, weight=m.SEMIBOLD).next_to(r1[4], m.UP, buff=0.12)
        scale_pill = pill(f"s = {W_MAX:.4f} / 127 ≈ {S8:.6f}", SCALE, SCALE_FILL, 22).move_to([-2.7, -2.75, 0])
        range_pill = pill("int8: −127 … 127", INT8, INT8_FILL, 22).move_to([3.1, -2.75, 0])
        bytes_pill = pill("2 bytes → 1 byte per weight", GOOD, GOOD_FILL, 22).move_to([3.1, -2.75, 0])

        self.say(
            "[Quantization stores each weight as a small integer, plus one shared scale.]",
            m.AnimationGroup(self.new_header("QUANTIZATION", "Scale, round, clamp"), m.Write(formula)),
        )
        self.say(
            "[Here are five weights from a row of four thousand and ninety-six, drawn at random for this example.]",
            m.AnimationGroup(m.FadeIn(row_names[0]), m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in r1], lag_ratio=0.15)),
        )
        self.say(
            "[The scale is the size of the largest weight, divided by one hundred and twenty-seven]: [the biggest value an eight-bit integer can hold.]",
            m.AnimationGroup(
                m.Circumscribe(r1[4], color=SCALE, buff=0.06), m.FadeIn(largest), m.FadeIn(scale_pill, shift=0.2 * m.UP),
                m.Indicate(formula[6], color=SCALE, scale_factor=1.4),
            ),
            m.FadeIn(range_pill, shift=0.2 * m.UP),
        )
        self.say(
            "[Divide each weight by the scale], [then round to the nearest integer.]",
            m.AnimationGroup(
                m.FadeIn(row_names[1]), *[m.FadeIn(b, target_position=a) for a, b in zip(r1, r2)],
                m.Indicate(m.VGroup(formula[4], formula[5], formula[6]), color=FP16, scale_factor=1.2),
            ),
            m.AnimationGroup(
                m.FadeIn(row_names[2]), *[m.FadeIn(b, target_position=a) for a, b in zip(r2, r3)],
                m.Indicate(formula[3], color=INT8, scale_factor=1.2),
            ),
        )
        self.say(
            "[Clamp keeps every result inside that range], [so each weight now fits in one byte instead of two.]",
            m.AnimationGroup(m.Indicate(m.VGroup(formula[2], formula[8]), color=INT8, scale_factor=1.15), m.Circumscribe(r3[4], color=INT8, buff=0.06)),
            m.AnimationGroup(m.FadeOut(range_pill, shift=0.2 * m.UP), m.FadeIn(bytes_pill, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ rounding error

    def rounding_error(self):
        self.next_section(
            "Rounding error",
            notes="Relative error ||w − s·q|| / ||w|| on the 4,096 seeded weights: "
            + ", ".join(f"{b}-bit {100 * ERR[b]:.2f}%" for b in range(8, 3, -1))
            + f"; int4 with a scale per {GROUP}: {100 * ERR_GROUPED:.2f}%. Each bit removed doubles the step and the error.",
        )
        formula = m.MathTex(r"|\, w - s\,q \,|", r"\le", r"\frac{s}{2}", font_size=50, color=INK).move_to([3.6, 1.85, 0])
        formula[2].set_color(SCALE)
        ax = axes([-1, 1, 0.5], [-1, 1, 0.5], 5.0, 4.4).move_to([-3.6, -0.6, 0])
        x_name = text("w", 24, FP16, weight=m.SEMIBOLD).next_to(ax.x_axis.get_end(), m.DOWN, buff=0.15)
        y_name = text("read back: s · q", 22, INT8, weight=m.SEMIBOLD).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.15)
        ends = m.VGroup(
            text("−max", 24, MUTED).next_to(ax.c2p(-1, -1), m.DOWN, buff=0.12),
            text("max", 24, MUTED).next_to(ax.c2p(1, -1), m.DOWN, buff=0.12),
        )
        diagonal = m.DashedLine(ax.c2p(-1, -1), ax.c2p(1, 1), dash_length=0.1).set_stroke(MUTED, width=2)
        bits = m.ValueTracker(8.0)

        def color_of(b):
            return mix(INT8, INT4, (8 - b) / 4)

        def stairs(b):
            L = 2 ** (b - 1) - 1
            pts = []
            for k in range(-L, L + 1):
                lo, hi = max(-1.0, (k - 0.5) / L), min(1.0, (k + 0.5) / L)
                pts += [ax.c2p(lo, k / L), ax.c2p(hi, k / L)]
            return polyline(pts, color_of(b), width=4).set_z_index(2)

        staircase = m.always_redraw(lambda: stairs(int(round(bits.get_value()))))

        def readout():
            b = int(round(bits.get_value()))
            top = text(f"{b} bits · {2 ** b - 1} levels", 28, color_of(b), weight=m.BOLD)
            low = text(f"error {100 * ERR[b]:.1f}%", 26, INK)
            return m.VGroup(top, low).arrange(m.DOWN, buff=0.15).move_to([3.6, 0.55, 0])

        live = m.always_redraw(readout)
        L4 = 7
        wedges = m.VGroup()
        for k in range(-L4, L4 + 1):
            lo, hi = max(-1.0, (k - 0.5) / L4), min(1.0, (k + 0.5) / L4)
            y = k / L4
            wedges.add(m.Polygon(ax.c2p(lo, y), ax.c2p(lo, lo), ax.c2p(y, y)).set_stroke(width=0).set_fill(BAD, opacity=0.6))
            wedges.add(m.Polygon(ax.c2p(y, y), ax.c2p(hi, hi), ax.c2p(hi, y)).set_stroke(width=0).set_fill(BAD, opacity=0.6))
        lens = card(2.35, 1.9, EDGE, CARD, radius=0.15, stroke=2).move_to([-4.95, 0.55, 0])
        c = lens.get_center() + np.array([-0.4, -0.32, 0])
        hw = 0.55
        rise = hw * 4.4 / 5.0  # the plot's slope on screen
        step_line = m.VGroup(
            m.Line(c + [-hw, -0.32, 0], c + [-hw, 0, 0]), m.Line(c + [-hw, 0, 0], c + [hw, 0, 0]), m.Line(c + [hw, 0, 0], c + [hw, 0.32, 0]),
        ).set_stroke(INT4, width=5)
        lens_diag = m.DashedLine(c + [-hw, -rise, 0], c + [hw, rise, 0], dash_length=0.08).set_stroke(MUTED, width=2)
        lens_err = m.VGroup(
            m.Polygon(c + [-hw, 0, 0], c + [-hw, -rise, 0], c).set_stroke(width=0).set_fill(BAD, opacity=0.6),
            m.Polygon(c, c + [hw, rise, 0], c + [hw, 0, 0]).set_stroke(width=0).set_fill(BAD, opacity=0.6),
        )
        gap = m.Line(c + [hw + 0.12, 0, 0], c + [hw + 0.12, rise, 0]).set_stroke(BAD, width=3)
        gap_name = tex(r"\frac{s}{2}", 40, BAD).next_to(gap, m.RIGHT, buff=0.1)
        lens_group = m.VGroup(lens, lens_err, lens_diag, step_line, gap, gap_name)
        wedge_label = text("error ≤ half a step", 20, BAD, weight=m.SEMIBOLD).move_to(lens.get_top() + 0.3 * m.DOWN)

        title = text("relative error, all 4,096 weights", 22, MUTED).move_to([3.7, -0.75, 0])
        rows = [("int8 · one scale", ERR[8], INT8), ("int4 · one scale", ERR[4], INT4), (f"int4 · scale per {GROUP}", ERR_GROUPED, INT4)]
        chart = []
        for i, (name, e, color) in enumerate(rows):
            y = -1.4 - 0.75 * i
            lab = text(name, 22, color, weight=m.SEMIBOLD).move_to([0.7, y, 0]).align_to([0.7, 0, 0], m.LEFT)
            bar = bar_right(3.45, y, 100 * e * 0.13, color, height=0.42)
            val = code(f"{100 * e:.1f}%", 22, INK).next_to(bar, m.RIGHT, buff=0.15)
            chart.append((lab, bar, val))

        self.say(
            "[To read a weight back, multiply by the scale.] [What's left over is the rounding error.]",
            self.new_header("ROUNDING ERROR", "Half a step, at most"),
            m.Write(formula),
        )
        self.say(
            "[Plotted against the original, the read-back weights form a staircase.] [At eight bits, the steps are tiny.]",
            m.AnimationGroup(m.Create(ax), m.FadeIn(x_name), m.FadeIn(y_name), m.FadeIn(ends), m.Create(diagonal)),
            m.AnimationGroup(m.FadeIn(staircase), m.FadeIn(live, shift=0.2 * m.UP)),
        )
        self.say(
            "[Every bit we remove halves the number of levels], [so the step doubles, and so does the error.]",
            bits.animate(run_time=3.2, rate_func=m.linear).set_value(4.0),
            m.Circumscribe(m.Rectangle(width=4.2, height=1.1).move_to([3.6, 0.55, 0]), color=INT4, buff=0.0),
        )
        self.say(
            "[No weight moves by more than half a step]: [half the scale.]",
            m.AnimationGroup(m.FadeIn(wedges), m.FadeIn(lens_group, scale=0.8), m.FadeIn(wedge_label, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.Indicate(formula[2], color=SCALE, scale_factor=1.35), m.Circumscribe(gap_name, color=BAD, buff=0.06)),
        )
        self.say(
            "[On the whole row], [eight-bit is off by under one percent], [four-bit by about fifteen.]",
            m.FadeIn(title, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(chart[0][0]), m.GrowFromEdge(chart[0][1], m.LEFT), m.FadeIn(chart[0][2])),
            m.AnimationGroup(m.FadeIn(chart[1][0]), m.GrowFromEdge(chart[1][1], m.LEFT), m.FadeIn(chart[1][2])),
        )
        self.say(
            "[Giving every group of a hundred and twenty-eight weights its own scale], [brings four-bit down to about eleven.]",
            m.FadeIn(chart[2][0], shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.GrowFromEdge(chart[2][1], m.LEFT), m.FadeIn(chart[2][2])),
        )
        self.wait(0.3)
        self.stop(staircase, live)
        self.clear()

    # ------------------------------------------------------------------ outliers

    def outliers(self):
        self.next_section(
            "Outliers",
            notes=f"4,096 activations N(0, 1), seed 7, four outlier dimensions set to {OUT_VALUES} (illustrative; outliers near 60 "
            f"are reported for 13B models in Dettmers' LLM.int8() write-up). One scale 60/127: ordinary values use {len(LEVELS)} levels, "
            f"error {100 * ERR_SHARED:.1f}%. Outliers kept in 16-bit, the rest int8 with its own scale: {100 * ERR_SPLIT:.2f}%, "
            f"{100 * KEPT_INT8:.2f}% of values in int8. LLM.int8(): Dettmers et al., 2022 (vector-wise quantization + mixed-precision decomposition).",
        )
        top_y, top_unit = 1.45, 5.8 / 66
        low_y, low_unit = -0.95, 5.8 / 4.15
        top_line = m.Line([-5.8, top_y, 0], [5.8, top_y, 0]).set_stroke(FAINT, width=3)
        top_ticks = m.VGroup(
            *[text(s, 20, MUTED).move_to([v * top_unit, top_y - 0.33, 0]) for s, v in (("−60", -60), ("0", 0), ("60", 60))]
        )
        jitter = np.random.default_rng(3).uniform(0.0, 1.0, len(STRIP))
        top_dots = m.VGroup(*[m.Dot([v * top_unit, top_y + 0.12 + 0.22 * j, 0], radius=0.045, color=FP16) for v, j in zip(STRIP, jitter)])
        out_dots = m.VGroup(*[m.Dot([v * top_unit, top_y + 0.22, 0], radius=0.11, color=OUTLIER) for v in OUT_VALUES]).set_z_index(2)
        out_labels = m.VGroup(*[code(f"{v:g}".replace("-", "−"), 20, OUTLIER).next_to(d, m.UP, buff=0.12) for v, d in zip(OUT_VALUES, out_dots)])
        wedge = m.VGroup(
            m.DashedLine([-4.15 * top_unit, top_y - 0.06, 0], [-5.8, low_y + 0.85, 0], dash_length=0.08).set_stroke(FAINT, width=2),
            m.DashedLine([4.15 * top_unit, top_y - 0.06, 0], [5.8, low_y + 0.85, 0], dash_length=0.08).set_stroke(FAINT, width=2),
        )
        low_line = m.Line([-5.8, low_y, 0], [5.8, low_y, 0]).set_stroke(FAINT, width=3)
        low_ticks = m.VGroup(*[text(s, 20, MUTED).move_to([v * low_unit, low_y - 0.33, 0]) for s, v in (("−4", -4), ("0", 0), ("4", 4))])

        def at(v, j):
            return [v * low_unit, low_y + 0.15 + 0.55 * j, 0]

        low_dots = m.VGroup(*[m.Dot(at(v, j), radius=0.06, color=FP16) for v, j in zip(STRIP, jitter)]).set_z_index(2)
        guides = m.VGroup(
            *[m.Line([k * S_OUT * low_unit, low_y - 0.12, 0], [k * S_OUT * low_unit, low_y + 0.8, 0]).set_stroke(INT8, width=2.5, opacity=0.6) for k in LEVELS]
        )
        fine_levels = [k for k in range(-127, 128) if abs(k * S_ORD) <= 4.1]
        fine = m.VGroup(
            *[m.Line([k * S_ORD * low_unit, low_y - 0.08, 0], [k * S_ORD * low_unit, low_y + 0.8, 0]).set_stroke(INT8, width=1, opacity=0.3) for k in fine_levels]
        )
        snapped = [at(np.round(v / S_OUT) * S_OUT, j) for v, j in zip(STRIP, jitter)]
        unsnapped = [at(np.round(v / S_ORD) * S_ORD, j) for v, j in zip(STRIP, jitter)]
        levels_text = text(f"ordinary values share {len(LEVELS)} of 255 levels", 24, INK).move_to([-3.2, -2.0, 0])
        levels_fixed = text("their own scale: the full int8 range", 24, GOOD).move_to([-3.2, -2.0, 0])
        err_text = text(f"their error: {100 * ERR_SPLIT:.1f}% → {100 * ERR_SHARED:.1f}%", 24, BAD, weight=m.SEMIBOLD).move_to([-3.2, -2.6, 0])
        err_fixed = text(f"their error: back to {100 * ERR_SPLIT:.1f}%", 24, GOOD, weight=m.SEMIBOLD).move_to([-3.2, -2.6, 0])
        for t in (levels_text, levels_fixed, err_text, err_fixed):
            t.align_to([-6.2, 0, 0], m.LEFT)
        kept = pill(f"{100 * KEPT_INT8:.1f}% of values stay int8", GOOD, GOOD_FILL, 22).move_to([0, -3.3, 0]).align_to([-6.2, 0, 0], m.LEFT)
        box = card(4.6, 1.45, FP16, FP16_FILL, radius=0.2, stroke=2).move_to([3.9, -2.3, 0])
        box_title = text("kept in 16-bit", 22, FP16, weight=m.BOLD).move_to(box.get_top() + 0.38 * m.DOWN)
        box_dots = m.VGroup(*[m.Dot(radius=0.1, color=OUTLIER) for _ in OUT_VALUES]).arrange(m.RIGHT, buff=0.75)
        box_dots.move_to(box.get_center() + 0.05 * m.DOWN)
        box_vals = m.VGroup(*[code(f"{v:g}".replace("-", "−"), 20, INK).next_to(d, m.DOWN, buff=0.1) for v, d in zip(OUT_VALUES, box_dots)])
        source = cite("LLM.int8() · Dettmers et al., 2022").next_to(box, m.DOWN, buff=0.15)

        self.say(
            "[Weights are well behaved. Activations, the values passed between layers, are harder.]",
            m.AnimationGroup(self.new_header("OUTLIERS", "One giant ruins the scale"), m.Create(top_line), m.FadeIn(top_ticks), m.FadeIn(top_dots)),
        )
        self.say(
            "[In large models, a few feature dimensions carry huge values]: [here, for example, up to sixty, among thousands near one.]",
            m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in out_dots], lag_ratio=0.2),
            m.FadeIn(out_labels, shift=0.15 * m.DOWN),
        )
        self.say(
            "[One shared scale must stretch to fit the giants], [so the ordinary values get just sixteen levels.]",
            m.AnimationGroup(m.Create(wedge), m.Create(low_line), m.FadeIn(low_ticks), m.FadeIn(low_dots, target_position=top_dots)),
            m.AnimationGroup(m.Create(guides), m.FadeIn(levels_text, shift=0.15 * m.UP)),
        )
        self.say(
            "[Rounding to so few levels], [pushes their error from under one percent to about fourteen.]",
            m.AnimationGroup(*[d.animate.move_to(p) for d, p in zip(low_dots, snapped)], run_time=1.2),
            m.FadeIn(err_text, shift=0.15 * m.UP),
        )
        self.say(
            "[The fix, from twenty twenty-two]: [keep those few dimensions in sixteen-bit], [and quantize the rest to eight bits.]",
            m.FadeIn(source),
            m.AnimationGroup(
                m.FadeIn(box), m.FadeIn(box_title), m.FadeIn(box_vals),
                *[m.FadeIn(b, target_position=a) for a, b in zip(out_dots, box_dots)],
                m.FadeOut(out_dots), m.FadeOut(out_labels),
            ),
            m.AnimationGroup(
                m.FadeOut(guides), m.FadeIn(fine), m.ReplacementTransform(levels_text, levels_fixed),
                *[d.animate.move_to(p) for d, p in zip(low_dots, unsnapped)],
            ),
        )
        self.say(
            "[Over ninety-nine point nine percent of the values stay eight-bit], [and the error is back under one percent.]",
            m.FadeIn(kept, shift=0.15 * m.UP),
            m.ReplacementTransform(err_text, err_fixed),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ post-training methods

    def post_training(self):
        self.next_section(
            "GPTQ and AWQ",
            notes="Post-training weight quantization with a small calibration set, no retraining, 3-4 bits. "
            "GPTQ (Frantar et al., 2022): quantizes one column at a time and updates the not-yet-quantized weights to compensate, "
            "using approximate second-order information. AWQ (Lin et al., 2023): ~1% salient weight channels, found from activation "
            "magnitudes, are protected by scaling them up before quantization (an equivalent transformation). Facts checked 2026-10.",
        )

        def grid(rows=3, cols=6, size=0.42, gap=0.08):
            squares = m.VGroup(
                *[
                    m.VGroup(*[m.Square(side_length=size).set_fill(FP16_FILL, opacity=1).set_stroke(FP16, width=2) for _ in range(rows)]).arrange(m.DOWN, buff=gap)
                    for _ in range(cols)
                ]
            ).arrange(m.RIGHT, buff=gap)
            return squares  # squares[c] is a column

        left = card(6.1, 4.3, INT4, CARD, stroke=2).move_to([-3.35, -0.3, 0])
        right = card(6.1, 4.3, INT4, CARD, stroke=2).move_to([3.35, -0.3, 0])

        def head(name, source, box):
            row = m.VGroup(text(name, 30, INT4, weight=m.BOLD), cite(source)).arrange(m.RIGHT, buff=0.35)
            return row.move_to(box.get_top() + 0.5 * m.DOWN)

        def desc(line, box):
            return text(line, 22, MUTED).move_to(box.get_bottom() + 0.45 * m.UP)

        g_head, a_head = head("GPTQ", "Frantar et al., 2022", left), head("AWQ", "Lin et al., 2023", right)
        g_grid = grid().move_to(left.get_center() + 0.05 * m.DOWN)
        frame = m.SurroundingRectangle(g_grid[0], buff=0.07, color=INT4).set_stroke(INT4, width=4)
        g_desc = desc("round column by column, fix up the rest", left)

        a_grid = grid().move_to(right.get_center() + 0.05 * m.DOWN)
        heights = [0.16, 0.26, 0.5, 0.2, 0.12, 0.22]
        acts = m.VGroup(
            *[
                bar_up(col.get_x(), a_grid.get_top()[1] + 0.12, h, OUTLIER if h > 0.5 else MUTED, width=0.3)
                for col, h in zip(a_grid, heights)
            ]
        )
        salient = 2
        a_desc = desc("scale up weights that meet big activations", right)
        times = text("× s", 24, GOOD, weight=m.BOLD).next_to(a_grid[salient], m.DOWN, buff=0.1)
        need = pill("a small calibration set · no retraining", GOOD, GOOD_FILL, 22).move_to([0, -3.15, 0])

        def round_column(c):
            return m.AnimationGroup(
                frame.animate.move_to(g_grid[c]),
                *[sq.animate.set_fill(INT4_FILL).set_stroke(INT4) for sq in g_grid[c]],
                run_time=0.55,
            )

        self.say(
            "[Rounding itself can be smarter. Two popular methods reach four bits, sometimes three], [with no retraining, just a small set of sample inputs.]",
            m.AnimationGroup(
                self.new_header("POST-TRAINING", "Smarter rounding"),
                m.FadeIn(left), m.FadeIn(right), m.FadeIn(g_head), m.FadeIn(a_head), m.FadeIn(g_grid), m.FadeIn(frame), m.FadeIn(a_grid), m.FadeIn(acts),
            ),
            m.FadeIn(need, shift=0.2 * m.UP),
        )
        self.say(
            "[One rounds the weights a column at a time], [and nudges the columns not yet rounded, to cancel the error.]",
            m.AnimationGroup(m.Succession(round_column(0), round_column(1), round_column(2)), m.FadeIn(g_desc)),
            m.Indicate(m.VGroup(*g_grid[3:]), color=FP16, scale_factor=1.06),
        )
        self.say(
            "[The other finds the weights that meet large activations], [and scales them up before rounding, to protect them.]",
            m.AnimationGroup(m.Indicate(acts[salient], color=OUTLIER, scale_factor=1.3), m.FadeIn(a_desc)),
            m.AnimationGroup(
                *[sq.animate.set_fill(GOOD_FILL).set_stroke(GOOD, width=3) for sq in a_grid[salient]],
                m.FadeIn(times, shift=0.1 * m.DOWN),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ distillation: soft labels

    def soft_labels(self):
        self.next_section(
            "Soft labels",
            notes="Illustrative teacher logits " + ", ".join(f"{w} {z:g}" for w, z in zip(WORDS, Z_TEACHER))
            + ". T = 1: " + ", ".join(f"{100 * p:.1f}%" for p in P_COLD)
            + f"; T = {T_HOT:g}: " + ", ".join(f"{100 * p:.1f}%" for p in P_HOT)
            + ". Distillation with soft targets at a raised temperature: Hinton, Vinyals and Dean, 2015. Temperature: Episode 16.",
        )
        self.T = m.ValueTracker(1.0)
        self.bar_x = (-6.1, -4.9, -3.7, -2.5, -1.3)
        self.base, self.unit = -2.15, 3.4
        teacher = m.VGroup(icon(self.teacher_icon, 0.8), text("teacher · 7B", 24, TEACHER, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        teacher.move_to([0, 1.95, 0]).align_to([-6.4, 0, 0], m.LEFT)
        prompt = text("My phone's battery is …", 26, INK)
        bubble_box = card(prompt.width + 0.6, prompt.height + 0.45, SKY, SKY_FILL, radius=0.25)
        bubble_box.move_to([0, 1.95, 0]).align_to(teacher.get_right() + 0.4 * m.RIGHT, m.LEFT)
        bubble = m.VGroup(bubble_box, prompt.move_to(bubble_box))
        student = m.VGroup(icon(self.student_icon, 0.8), text("student · 1B", 24, STUDENT, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        student.move_to([3.7, 1.95, 0])
        floor = m.Line([self.bar_x[0] - 0.55, self.base, 0], [self.bar_x[-1] + 0.55, self.base, 0]).set_stroke(FAINT, width=2.5)
        names = m.VGroup(*[text(w, 22, MUTED).move_to([x, self.base - 0.33, 0]) for w, x in zip(WORDS, self.bar_x)])

        def teacher_bars():
            p = softmax(Z_TEACHER, self.T.get_value())
            bars = m.VGroup(*[bar_up(x, self.base, v * self.unit, TEACHER) for v, x in zip(p, self.bar_x)])
            vals = m.VGroup(*[code(f"{100 * v:.0f}%", 22, INK).next_to(b, m.UP, buff=0.1) for v, b in zip(p, bars)])
            return m.VGroup(bars, vals)

        live = m.always_redraw(teacher_bars)
        formula = tex(r"p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}", 44, INK).move_to([3.7, 0.6, 0])
        t_value = m.always_redraw(lambda: text(f"T = {self.T.get_value():.1f}", 32, TEACHER, weight=m.BOLD).move_to([3.7, -0.75, 0]))
        hard = pill("hard label: just “dead”", MUTED, CARD, 22).move_to([3.7, -1.75, 0])
        soft = pill("soft labels: the whole picture", TEACHER, TEACHER_FILL, 22).move_to([3.7, -2.55, 0])
        source = cite("Hinton et al., 2015").move_to([3.7, -3.3, 0])

        def region(i, j, p):
            top = max(p[i : j + 1]) * self.unit
            r = m.Rectangle(width=self.bar_x[j] - self.bar_x[i] + 1.0, height=top + 0.75)
            return r.move_to([(self.bar_x[i] + self.bar_x[j]) / 2, self.base + (top + 0.75) / 2 - 0.05, 0])

        self.say(
            "[Distillation shrinks the model itself]: [a small student learns from a big teacher.]",
            m.AnimationGroup(self.new_header("DISTILLATION", "A teacher's soft labels"), m.FadeIn(teacher, shift=0.2 * m.RIGHT)),
            m.FadeIn(student, shift=0.2 * m.LEFT),
        )
        self.say(
            "[Ask the teacher to finish]: [my phone's battery is...]",
            m.FadeIn(bubble, shift=0.2 * m.UP),
            m.AnimationGroup(m.Create(floor), m.FadeIn(names), m.FadeIn(live, shift=0.2 * m.UP)),
        )
        self.say(
            "[It says dead, at eighty-two percent.] [A hard label would keep only that answer, and throw the rest away.]",
            m.Circumscribe(region(0, 0, P_COLD), color=TEACHER, buff=0.0),
            m.FadeIn(hard, shift=0.2 * m.UP),
        )
        self.say(
            "[Raise the softmax temperature, as in episode sixteen], [and the rest comes into view.]",
            m.AnimationGroup(m.Write(formula), m.FadeIn(t_value)),
            self.T.animate(run_time=2.4).set_value(T_HOT),
        )
        self.say(
            "[Low and flat are close], [purple is absurd]: [soft labels carry far more than one right answer.]",
            m.Circumscribe(region(1, 2, P_HOT), color=TEACHER, buff=0.0),
            m.Circumscribe(region(4, 4, P_HOT), color=BAD, buff=0.0),
            m.AnimationGroup(m.FadeIn(soft, shift=0.2 * m.UP), m.FadeIn(source)),
        )
        self.wait(0.3)
        self.stop(t_value)
        self.live_teacher = live
        self.play(*[m.FadeOut(x) for x in (formula, t_value, hard, soft, source)], run_time=0.6)

    # ------------------------------------------------------------------ distillation: the loss

    def student(self):
        self.next_section(
            "The student's loss",
            notes=f"Loss = T² · KL(teacher_T || student_T) at T = {T_HOT:g}; gradient descent on the student's logits from zero "
            f"(uniform), rate 2, {len(LOSSES) - 1} steps: loss {LOSSES[0]:.2f} → {LOSSES[-1]:.3f}. In a real student the logits come from "
            "a smaller network, so the match is never perfect. Hinton et al. multiply by T² so soft-target gradients keep their scale, "
            "and mix in the ordinary loss on the true labels.",
        )
        step = m.ValueTracker(0.0)
        xs = tuple(x + 7.4 for x in self.bar_x)  # 1.3 … 6.1

        def student_logits():
            s = step.get_value()
            k = int(np.floor(min(s, len(STUDENT_PATH) - 1.001)))
            a = s - k
            return (1 - a) * STUDENT_PATH[k] + a * STUDENT_PATH[k + 1]

        def student_bars():
            p = softmax(student_logits(), T_HOT)
            bars = m.VGroup(*[bar_up(x, self.base, v * self.unit, STUDENT) for v, x in zip(p, xs)])
            vals = m.VGroup(*[code(f"{100 * v:.0f}%", 22, INK).next_to(b, m.UP, buff=0.1) for v, b in zip(p, bars)])
            return m.VGroup(bars, vals)

        live = m.always_redraw(student_bars)
        floor = m.Line([xs[0] - 0.55, self.base, 0], [xs[-1] + 0.55, self.base, 0]).set_stroke(FAINT, width=2.5)
        names = m.VGroup(*[text(w, 22, MUTED).move_to([x, self.base - 0.33, 0]) for w, x in zip(WORDS, xs)])
        formula = m.MathTex(
            r"\mathcal{L}", "=", "T^2", r"\cdot", r"\mathrm{KL}(", r"p^{T}_{\text{teacher}}", r"\,\|\,", r"p^{T}_{\text{student}}", ")",
            font_size=40, color=INK,
        ).move_to([0.3, -3.3, 0])
        formula[5].set_color(TEACHER)
        formula[7].set_color(STUDENT)

        def loss_now():
            return distill_loss(student_logits())

        readout = m.always_redraw(lambda: text(f"loss {loss_now():.2f}", 28, INK, weight=m.BOLD).move_to([3.7, 1.0, 0]))
        note = text("+ the usual loss on the true word", 22, MUTED).move_to([3.7, 0.35, 0])

        self.say(
            "[The student reads the same sentence], [and at first, it spreads its bets evenly.]",
            self.new_header("THE STUDENT", "Match the teacher"),
            m.AnimationGroup(m.Create(floor), m.FadeIn(names), m.FadeIn(live, shift=0.2 * m.UP)),
        )
        self.say(
            "[Its loss is the divergence from episode seventeen], [between the teacher's and the student's softened probabilities.]",
            m.AnimationGroup(m.Write(formula), m.FadeIn(readout)),
            m.AnimationGroup(m.Circumscribe(formula[5], color=TEACHER, buff=0.06), m.Circumscribe(formula[7], color=STUDENT, buff=0.06)),
        )
        self.say(
            "[It's multiplied by the temperature squared, so the gradients keep their size], [and mixed with the usual loss on the true next word.]",
            m.Indicate(formula[2], color=TEACHER, scale_factor=1.4),
            m.FadeIn(note, shift=0.15 * m.UP),
        )
        self.say(
            "[Step by step, the student's bars move toward the teacher's], [and the loss falls to almost zero.]",
            step.animate(run_time=3.4, rate_func=m.linear).set_value(len(STUDENT_PATH) - 1),
            m.Circumscribe(m.Rectangle(width=2.6, height=0.7).move_to([3.7, 1.0, 0]), color=GOOD, buff=0.0),
        )
        self.wait(0.3)
        self.stop(live, readout, self.live_teacher)
        self.clear()

    # ------------------------------------------------------------------ both: in your pocket

    def pocket(self):
        self.next_section(
            "In your pocket",
            notes=f"DistilBERT (Sanh et al., 2019): 40% fewer parameters than BERT-base, 97% of its language-understanding performance "
            f"(GLUE), 60% faster. Illustrative pipeline: 7B teacher → 1B student ({STUDENT_16:.0f} GB at 16-bit) → int4 with group "
            f"scales ({STUDENT_GB:.2f} GB). Token-by-token decoding is usually memory-bound, so fewer bytes per weight usually means faster.",
        )
        fact = card(11.6, 1.4, STUDENT, CARD, stroke=2).move_to([0, 1.7, 0])
        fact_words = m.VGroup(
            text("DistilBERT: 40% fewer parameters, 97% of BERT's language understanding", 24, INK),
            cite("Sanh et al., 2019"),
        ).arrange(m.DOWN, buff=0.14).move_to(fact)
        xs, y = (-5.1, -0.9, 3.3), -0.4
        nodes = [
            node(self.teacher_icon, "Teacher", "7B · 16-bit · 14 GB", TEACHER, TEACHER_FILL, width=2.9, height=2.1).move_to([xs[0], y, 0]),
            node(self.student_icon, "Student", f"1B · 16-bit · {STUDENT_16:.0f} GB", STUDENT, STUDENT_FILL, width=2.9, height=2.1).move_to([xs[1], y, 0]),
            node(self.student_icon, "Student, int4", f"1B · 4-bit · {STUDENT_GB:.2f} GB", INT4, INT4_FILL, width=2.9, height=2.1).move_to([xs[2], y, 0]),
        ]
        links = [arrow(nodes[i].get_right(), nodes[i + 1].get_left(), MUTED) for i in range(2)]
        link_names = [text(s, 20, MUTED, weight=m.SEMIBOLD).next_to(a, m.UP, buff=0.12) for s, a in zip(("distill", "quantize"), links)]
        phone = icon("ep42-phone", 1.9).move_to([5.85, y, 0])
        fits = checkmark(0.55).move_to(phone.get_corner(m.UR) + np.array([0.0, 0.1, 0]))
        catch = m.VGroup(
            pill("catch: every step costs some quality", BAD, BAD_FILL, 22),
            pill("measure the small model against the big one", GOOD, GOOD_FILL, 22),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -2.85, 0])

        self.say(
            "[Students can be remarkably good]: [one early student kept ninety-seven percent of its teacher's language understanding, with forty percent fewer parameters.]",
            self.new_header("IN YOUR POCKET", "Distill, then quantize"),
            m.AnimationGroup(m.FadeIn(fact), m.FadeIn(fact_words, shift=0.15 * m.UP)),
        )
        self.say(
            "[Now combine both tools]: [distill seven billion parameters into a one-billion student], [then quantize it to four bits.]",
            m.FadeIn(nodes[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(link_names[0]), m.FadeIn(nodes[1], shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(link_names[1]), m.FadeIn(nodes[2], shift=0.2 * m.LEFT)),
        )
        self.say(
            "[That's about half a gigabyte]: [it fits easily, and it runs faster, with fewer bytes to read for every word.]",
            m.Circumscribe(nodes[2], color=INT4, buff=0.06),
            m.AnimationGroup(m.FadeIn(phone, shift=0.2 * m.LEFT), m.FadeIn(fits, scale=0.5)),
        )
        self.say(
            "[The catch: every step costs some quality], [so always measure the small model against the big one.]",
            m.FadeIn(catch[0], shift=0.2 * m.UP),
            m.FadeIn(catch[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Round (quantize), protect (outliers), distill.")
        halo = glow(PINK, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Round", "Protect", "Distill"), (INT8, INT4, STUDENT))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Fewer bits, fewer weights: a big model's knowledge, in your pocket.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [round], [protect], [distill.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Fewer bits, fewer weights: a big model's knowledge, in your pocket.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
