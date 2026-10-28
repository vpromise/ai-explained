"""AI, Explained · Episode 28 · Scaling Laws

Running example: your training budget just doubled. How should parameters N and tokens D
grow, and how much better will the model get?

Every curve and number on screen is computed below, and the narration's numbers are asserted:

- Power law (Kaplan et al., 2020, eq. 1.1): L(N) = (N_c / N)^alpha with their fitted values
  N_c = 8.8e13 non-embedding parameters, alpha = 0.076. Doubling N multiplies the loss by
  2^-0.076 = 0.949 ("about five percent lower"); from 1e6 to 1e10 parameters it halves (4.0 to 2.0).
- The evidence panels: illustrative "training runs" drawn around Kaplan's three laws
  (alpha_N 0.076, alpha_D 0.095, alpha_C 0.050; N_c 8.8e13, D_c 5.4e13 tokens, C_c 3.1e8
  PF-days) with 1.2 % seeded noise; lines are least-squares fits on log-log axes. The compute
  panel fits only the small runs and extrapolates to a big run, which lands within 1 %.
- Compute: C ~ 6·N·D FLOPs (2 per parameter per token forward, 4 backward).
- Compute-optimal curves: an illustrative loss in Hoffmann et al.'s (2022) form,
  L(N, D) = E + A/N^a + B/D^a with E = 1.69, A = 406.4 (their E and A), a = 0.3 for both terms,
  and B = A·20^a so that the best split is 20 tokens per parameter (their rule of thumb).
  The minimum of each fixed-budget curve is found numerically (a fine grid), not assumed.
- Doubling Chinchilla's budget (70B parameters, 1.4T tokens, 5.9e23 FLOPs): the numerical optimum
  is 99B parameters and 2.0T tokens (x1.41 each, as N = sqrt(C/120) says). Kaplan's 2020
  allocation (N ~ C^0.73, D ~ C^0.27) would give x1.66 and x1.21.
- Serving: with the same illustrative loss, a model half the size matches the loss with 2.4x the
  tokens, for 1.2x the training compute and half the compute per generated token.
- Llama 3 8B: over 15T tokens (Meta, 2024), about 1,900 tokens per parameter.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PARAMS, TOKENS, COMPUTE, LOSS, EXPO = VIOLET, SKY, AMBER, PINK, LIME  # one colour per quantity

# ---------------------------------------------------------------------------- Kaplan et al. (2020)

KAPLAN = {"N": (8.8e13, 0.076), "D": (5.4e13, 0.095), "C": (3.1e8, 0.050)}  # eq. 1.1-1.3
ALPHA = KAPLAN["N"][1]


def kaplan(kind, x):
    xc, a = KAPLAN[kind]
    return (xc / np.asarray(x, dtype=float)) ** a


STEP = 2**-ALPHA
assert ALPHA < 0.1  # "the exponent is small: under a tenth"
assert round(100 * (1 - STEP)) == 5  # "about five percent lower, every time"
assert round(float(kaplan("N", 1e6)), 1) == 4.0 and round(float(kaplan("N", 1e10)), 1) == 2.0  # "the loss halves"

# illustrative training runs around the three laws, with seeded noise
_rng = np.random.default_rng(7)
RUNS = {"N": np.logspace(5.2, 9.0, 9), "D": np.logspace(7.4, 10.2, 8), "C": np.logspace(-7.0, -2.5, 7)}
RUN_LOSS = {k: kaplan(k, x) * np.exp(0.012 * _rng.normal(size=len(x))) for k, x in RUNS.items()}
FIT = {k: np.polyfit(np.log10(RUNS[k]), np.log10(RUN_LOSS[k]), 1) for k in RUNS}
BIG_C = 1.0  # PF-days: a run 300 times bigger than the largest small run
BIG_LOSS = float(kaplan("C", BIG_C) * np.exp(0.012 * _rng.normal()))
PREDICTED = float(10 ** np.polyval(FIT["C"], np.log10(BIG_C)))
for _k in RUNS:
    assert abs(-FIT[_k][0] - KAPLAN[_k][1]) < 0.004  # the fitted slopes recover the exponents
assert abs(PREDICTED / BIG_LOSS - 1) < 0.01  # the extended line predicts the big run


def fitted(kind, x):
    return 10 ** np.polyval(FIT[kind], np.log10(x))


# ---------------------------------------------------------------------------- compute-optimal (illustrative)

E, A, A_EXP, PER_PARAM = 1.69, 406.4, 0.30, 20
B = A * PER_PARAM**A_EXP  # puts the best split at 20 tokens per parameter


def loss(n, d):
    return E + A * np.asarray(n, dtype=float) ** -A_EXP + B * np.asarray(d, dtype=float) ** -A_EXP


def iso(c, n):
    """Loss of a model with n parameters when the budget c (FLOPs) fixes its tokens: d = c / 6n."""
    return loss(n, c / (6 * np.asarray(n, dtype=float)))


_GRID = np.logspace(7, 13, 240001)


def optimum(c):
    """The best model size for a budget, found numerically."""
    return float(_GRID[int(np.argmin(iso(c, _GRID)))])


BUDGETS = (1e21, 1e22, 1e23, 1e24)
OPT = {c: optimum(c) for c in BUDGETS}
for _c, _n in OPT.items():
    assert abs(_c / (6 * _n) / _n / PER_PARAM - 1) < 0.01  # "about twenty training tokens for every parameter"

# Chinchilla and Gopher (Hoffmann et al., 2022)
N0, D0 = 70e9, 1.4e12
C0 = 6 * N0 * D0
GOPHER_N, GOPHER_D = 280e9, 300e9
assert abs(optimum(C0) / N0 - 1) < 0.001  # Chinchilla sits on the optimum of this fit
assert 0.8 < 6 * GOPHER_N * GOPHER_D / C0 < 1.2 and GOPHER_N == 4 * N0  # "about the same compute, four times the parameters"

# the running example: double the budget
N1 = optimum(2 * C0)
D1 = 2 * C0 / (6 * N1)
assert abs(N1 / np.sqrt(2 * C0 / 120) - 1) < 0.001  # N = sqrt(C / 120)
assert round(N1 / N0, 1) == 1.4 and round(D1 / D0, 1) == 1.4  # "about one point four times", "just as much"
assert round(N1 / 1e10) == 10 and round(D1 / 1e12) == 2  # "about a hundred billion", "about two trillion"
KAPLAN_N, KAPLAN_D = 2**0.73, 2**0.27  # Kaplan et al.'s allocation N ~ C^0.73, D ~ C^0.27
assert round(KAPLAN_N, 1) == 1.7 and round(KAPLAN_D, 1) == 1.2  # "one point seven", "one point two"

# serving: a model half the size, trained longer to the same loss
L0 = float(loss(N0, D0))
HALF_N = N0 / 2
HALF_D = float((B / (L0 - E - A * HALF_N**-A_EXP)) ** (1 / A_EXP))
assert abs(float(loss(HALF_N, HALF_D)) - L0) < 1e-9
TOKENS_X = HALF_D / D0  # 2.40
TRAIN_X = 6 * HALF_N * HALF_D / C0  # 1.20
SERVE_X = HALF_N / N0  # about 2N FLOPs per generated token
assert round(TOKENS_X * 2) / 2 == 2.5  # "about two and a half times the tokens"
assert round(100 * (TRAIN_X - 1)) == 20 and SERVE_X == 0.5  # "twenty percent more", "half as much"
HALF_RATIO = HALF_D / HALF_N
assert round(HALF_RATIO) == 96
LLAMA_RATIO = 15e12 / 8e9  # Llama 3 8B: over 15T tokens (Meta, 2024)
assert 1800 < LLAMA_RATIO < 2000  # "nearly two thousand per parameter"


# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """An icon in another accent, like kit.recolored(), but saved as assets/ep28-….svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = f"ep28-{name}-{color.lstrip('#').lower()}"
    svg = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{out}.svg"
    if not path.exists() or path.read_text() != svg:
        path.write_text(svg)
    return out


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners([list(p) for p in points]).set_stroke(color, width=width, opacity=opacity)


def frame_axes(center, width, height):
    """Plain axes on the unit square (no ticks): a Scale maps data onto them."""
    return axes([0, 1.0, 2], [0, 1.0, 2], width, height).move_to(center)


class Scale:
    """Data → screen on a frame_axes(), each axis linear or log10, limits (lo, hi)."""

    def __init__(self, ax, xlim, ylim, xlog=True, ylog=True):
        self.ax, self.xlim, self.ylim, self.xlog, self.ylog = ax, xlim, ylim, xlog, ylog

    @staticmethod
    def _unit(v, lim, log):
        f = np.log10 if log else (lambda t: t)
        return (f(v) - f(lim[0])) / (f(lim[1]) - f(lim[0]))

    def p(self, x, y):
        return self.ax.c2p(self._unit(x, self.xlim, self.xlog), self._unit(y, self.ylim, self.ylog))

    def curve(self, fn, x0, x1, color, width=4.5, n=200, opacity=1.0):
        xs = np.geomspace(x0, x1, n) if self.xlog else np.linspace(x0, x1, n)
        return polyline([self.p(x, fn(x)) for x in xs], color, width, opacity)

    def xticks(self, values, labels, size=26, color=MUTED):
        group = m.VGroup()
        for v, s in zip(values, labels):
            at = self.p(v, self.ylim[0])
            at[1] = self.ax.c2p(0, 0)[1]
            mark = m.Line(at, at + 0.12 * m.DOWN).set_stroke(FAINT, width=2.5)
            label = (tex(s, size, color) if "^" in s else text(s, 22, color)).next_to(mark, m.DOWN, buff=0.1)
            group.add(m.VGroup(mark, label))
        return group

    def yticks(self, values, labels, color=MUTED):
        group = m.VGroup()
        for v, s in zip(values, labels):
            at = self.p(self.xlim[0], v)
            at[0] = self.ax.c2p(0, 0)[0]
            mark = m.Line(at, at + 0.12 * m.LEFT).set_stroke(FAINT, width=2.5)
            group.add(m.VGroup(mark, text(s, 22, color).next_to(mark, m.LEFT, buff=0.1)))
        return group


def cite(s, head):
    """A paper reference under the header title (small, muted)."""
    return text(s, 20, MUTED).next_to(head[1], m.DOWN, buff=0.2).align_to(head[1], m.LEFT)


def legend_row(symbol, color, words, size=40):
    return m.VGroup(tex(symbol, size, color), text(words, 24, MUTED))


def box(width, height, corner, color, fill, opacity=1.0, stroke=3.5, dashed=False):
    """A rectangle with its bottom-left corner at `corner` (the compute picture: area = N·D)."""
    rect = m.Rectangle(width=width, height=height).set_stroke(color, width=stroke).set_fill(fill, opacity=opacity)
    rect.move_to(np.array(corner) + [width / 2, height / 2, 0])
    if dashed:
        rect = m.DashedVMobject(rect, num_dashes=int(14 * (width + height))).set_stroke(color, width=stroke)
    return rect


def sci(x, digits=1):
    exponent = int(np.floor(np.log10(x)))
    return rf"{x / 10**exponent:.{digits}f} \times 10^{{{exponent}}}"


# ---------------------------------------------------------------------------- the episode


class Ep28ScalingLaws(Episode):
    number = 28

    def lesson(self):
        self.hook()
        self.power_law()
        self.evidence()
        self.compute()
        self.optimal()
        self.doubling()
        self.serving()
        self.caveats()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Your training budget just doubled: a bigger model, more data, or both? And how much better will it get?")
        halo = glow(COMPUTE, radius=3.2).move_to([0, 1.1, 0])
        big = text("2×", 170, COMPUTE, weight=m.BOLD)
        bolt = icon("bolt", 1.7)
        top = m.VGroup(bolt, big).arrange(m.RIGHT, buff=0.4).move_to([0, 1.15, 0])
        caption = text("your training budget", 32, MUTED).move_to([0, -0.25, 0])

        def option(icons, label, color, fill):
            frame = card(3.7, 1.25, color, CARD, stroke=2)
            pics = m.VGroup(*[icon(n, 0.7) for n in icons]).arrange(m.RIGHT, buff=0.15)
            words = text(label, 26, INK, weight=m.SEMIBOLD)
            m.VGroup(pics, words).arrange(m.RIGHT, buff=0.3).move_to(frame)
            return m.VGroup(frame, pics, words)

        data_icon = variant("database", TOKENS, SKY_FILL)
        options = m.VGroup(
            option(["llm"], "a bigger model?", PARAMS, VIOLET_FILL),
            option([data_icon], "more data?", TOKENS, SKY_FILL),
            option(["llm", data_icon], "both?", TEAL, TEAL_FILL),
        ).arrange(m.RIGHT, buff=0.45).move_to([0, -1.75, 0])
        question = pill("How much lower will the loss be?", LOSS, PINK_FILL, 26).move_to([0, -3.15, 0])
        answer = pill("Scaling laws: predict it before you train", TEAL, TEAL_FILL, 26).move_to([0, -3.15, 0])

        self.say(
            "So... [imagine your training budget just doubled.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(top, shift=0.3 * m.UP), m.FadeIn(caption), run_time=1.2),
        )
        self.say(
            "[Should you build a bigger model], [train it on more data], [or both?]",
            *[m.FadeIn(o, shift=0.25 * m.UP) for o in options],
        )
        self.say(
            "[And how much better will it get?] [Scaling laws let you predict that, before you spend a thing.]",
            m.FadeIn(question, shift=0.2 * m.UP),
            m.FadeTransform(question, answer),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the power law

    def power_law(self):
        self.next_section(
            "Power laws",
            notes="L(N) = (N_c/N)^alpha with Kaplan et al.'s (2020) fit: N_c = 8.8e13, alpha = 0.076. Linear axes, then log-log: a straight line of slope -alpha; each doubling x0.949; 1e6 -> 1e10 parameters: loss 4.0 -> 2.0.",
        )
        header = self.new_header("POWER LAWS", "The shape of a scaling law")
        ax = frame_axes([-3.25, -0.4, 0], 6.2, 4.3)
        lin = Scale(ax, (0, 1.06e10), (0, 4.6), xlog=False, ylog=False)
        log = Scale(ax, (10**5.75, 10**10.3), (1.8, 4.5))
        ns = np.geomspace(1e6, 1e10, 320)
        at_lin = np.array([lin.p(n, kaplan("N", n)) for n in ns])
        at_log = np.array([log.p(n, kaplan("N", n)) for n in ns])
        straight = m.ValueTracker(0.0)
        drawn = polyline(at_lin, LOSS, width=5)
        curve = m.always_redraw(lambda: polyline(at_lin + straight.get_value() * (at_log - at_lin), LOSS, width=5))

        lin_ticks = m.VGroup(lin.xticks([0, 5e9, 1e10], ["0", "5 billion", "10 billion"]), lin.yticks([1, 2, 3, 4], ["1", "2", "3", "4"]))
        log_ticks = m.VGroup(
            log.xticks([10.0**e for e in range(6, 11)], [f"10^{{{e}}}" for e in range(6, 11)]),
            log.yticks([2, 3, 4], ["2", "3", "4"]),
        )
        x_name = text("parameters N", 22, PARAMS, weight=m.SEMIBOLD).next_to(ax.c2p(1, 0), m.DOWN, buff=0.62).align_to(ax.c2p(1, 0), m.RIGHT)
        y_name = text("loss L", 22, LOSS, weight=m.SEMIBOLD).next_to(ax.c2p(0, 1), m.RIGHT, buff=0.2)
        log_note = text("log scale", 22, MUTED).next_to(ax.c2p(0.5, 0), m.DOWN, buff=0.62)

        formula = m.MathTex(r"L(N) = \left(\frac{N_c}{N}\right)^{\alpha}", font_size=62, color=INK).move_to([3.85, 1.45, 0])
        f = formula[0]
        f[0].set_color(LOSS)
        for i in (2, 8):
            f[i].set_color(PARAMS)
        f[11].set_color(EXPO)
        rows = m.VGroup(
            legend_row("L", LOSS, "test loss: lower is better"),
            legend_row("N", PARAMS, "number of parameters"),
            legend_row("N_c", INK, "a fitted constant: the scale"),
            legend_row(r"\alpha", EXPO, "the exponent: about 0.076"),
        )
        for k, row in enumerate(rows):
            row[0].move_to([1.75, 0.2 - 0.68 * k, 0])
            row[1].move_to([0, row[0].get_y(), 0]).align_to([2.35, 0, 0], m.LEFT)
        source = text("Kaplan et al., 2020", 20, MUTED).move_to([0, -2.45, 0]).align_to([2.35, 0, 0], m.LEFT)

        stairs = m.VGroup()
        for k in range(6):
            n = 1e7 * 2**k
            a, b, c = log.p(n, kaplan("N", n)), log.p(2 * n, kaplan("N", n)), log.p(2 * n, kaplan("N", 2 * n))
            stairs.add(m.VGroup(m.Line(a, b).set_stroke(PARAMS, width=4), m.Line(b, c).set_stroke(LOSS, width=4)))
        stairs.set_z_index(2)
        times_two = text("×2", 22, PARAMS, weight=m.BOLD).next_to(stairs[0][0], m.UP, buff=0.12)
        slope = tex(r"\text{slope} = -\alpha", 34, EXPO).move_to(log.p(10**6.9, 2.35))
        doubling = pill("every ×2 in parameters: loss ×0.95", PARAMS, VIOLET_FILL, 24).move_to([3.85, -3.15, 0])
        ends = m.VGroup(
            m.Dot(log.p(1e6, kaplan("N", 1e6)), radius=0.1, color=INK),
            m.Dot(log.p(1e10, kaplan("N", 1e10)), radius=0.1, color=INK),
        ).set_z_index(3)
        end_labels = m.VGroup(
            code(f"{float(kaplan('N', 1e6)):.1f}", 26, LOSS).move_to(ends[0].get_center() + [0.32, -0.62, 0]),
            code(f"{float(kaplan('N', 1e10)):.1f}", 26, LOSS).next_to(ends[1], m.UP, buff=0.2),
        )

        self.say(
            "[Here's the shape behind it all]: [a power law.]",
            header,
            m.Write(formula, run_time=1.3),
        )
        self.say(
            "[L is the model's test loss], [and N is its number of parameters.]",
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.15 * m.UP), m.Indicate(f[0], color=LOSS, scale_factor=1.3)),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.15 * m.UP), m.Indicate(m.VGroup(f[2], f[8]), color=PARAMS, scale_factor=1.3)),
        )
        self.say(
            "[The constant on top just sets the scale.] [The exponent, alpha, is small: under a tenth.]",
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.15 * m.UP), m.Indicate(f[6:8], color=INK, scale_factor=1.3)),
            m.AnimationGroup(m.FadeIn(rows[3], shift=0.15 * m.UP), m.Indicate(f[11], color=EXPO, scale_factor=1.6), m.FadeIn(source)),
        )
        self.say(
            "[On ordinary axes], [the loss plunges at first, then crawls along, almost flat.]",
            m.AnimationGroup(m.Create(ax), m.FadeIn(lin_ticks), m.FadeIn(x_name), m.FadeIn(y_name)),
            m.Create(drawn, run_time=2.4),
        )
        self.remove(drawn)
        self.add(curve)
        self.say(
            "[Now put both axes on a log scale], [and the curve becomes a straight line.]",
            m.AnimationGroup(m.FadeOut(lin_ticks), m.FadeIn(log_ticks), m.FadeIn(log_note)),
            straight.animate(run_time=2.2, rate_func=m.smooth).set_value(1.0),
        )
        self.stop(curve)
        self.say("[Its slope is minus alpha.]", m.AnimationGroup(m.Indicate(curve, color=EXPO, scale_factor=1.0), m.FadeIn(slope, shift=0.15 * m.UP)))
        self.say(
            "[Every doubling of the parameters] [multiplies the loss by the same factor]: [about five percent lower, every time.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(s) for s in stairs], lag_ratio=0.3, run_time=2.0), m.FadeIn(times_two)),
            m.LaggedStart(*[m.Indicate(s[1], color=LOSS, scale_factor=1.4) for s in stairs], lag_ratio=0.15),
            m.FadeIn(doubling, shift=0.2 * m.UP),
        )
        self.say(
            "[From a million parameters to ten billion], [the loss halves.]",
            m.AnimationGroup(m.FadeIn(ends, scale=0.4), m.FadeIn(end_labels)),
            m.AnimationGroup(*[m.Indicate(lab, color=LOSS, scale_factor=1.3) for lab in end_labels]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the evidence

    def evidence(self):
        self.next_section(
            "The evidence",
            notes="Kaplan et al. (2020): loss is a power law in N, D and C when the other two are not the bottleneck; trends over more than seven orders of magnitude. Points are illustrative runs (1.2% seeded noise) around their fitted laws; lines are least-squares fits. Compute panel: fit on small runs, extrapolate to a run 300x bigger.",
        )
        header = self.new_header("THE EVIDENCE", "Straight lines, again and again")
        source = cite("Kaplan et al., 2020 · points illustrative", self.head)
        spec = (
            ("N", "Parameters N", "", PARAMS, (10**4.9, 10**9.3), (5, 9), r"\alpha \approx 0.076"),
            ("D", "Data D (tokens)", "", TOKENS, (10**7.1, 10**10.5), (8, 10), r"\alpha \approx 0.095"),
            ("C", "Compute C (PF-days)", "", COMPUTE, (10**-7.4, 10**1.2), (-7, 0), r"\alpha \approx 0.050"),
        )
        panels = []
        for k, (key, title, unit, color, xlim, ticks, expo) in enumerate(spec):
            x = -4.45 + 4.45 * k
            frame = card(4.15, 4.3, EDGE, CARD, stroke=2).move_to([x, -0.4, 0])
            head = text(title, 26, color, weight=m.BOLD).move_to(frame.get_top() + 0.4 * m.DOWN)
            ax = frame_axes([x + 0.15, -0.75, 0], 3.3, 2.3)
            losses = np.r_[RUN_LOSS[key], [BIG_LOSS] if key == "C" else []]
            sc = Scale(ax, xlim, (losses.min() / (1.25 if key == "C" else 1.1), losses.max() * 1.08))
            tick_group = sc.xticks([10.0 ** ticks[0], 10.0 ** ticks[1]], [f"10^{{{ticks[0]}}}", f"10^{{{ticks[1]}}}"])
            y_name = text("loss", 22, LOSS, weight=m.SEMIBOLD).next_to(ax.c2p(0, 1), m.UP, buff=0.2)
            dots = m.VGroup(*[m.Dot(sc.p(a, b), radius=0.075, color=color) for a, b in zip(RUNS[key], RUN_LOSS[key])]).set_z_index(3)
            line = sc.curve(lambda v: fitted(key, v), RUNS[key][0] / 1.6, RUNS[key][-1] * 1.6, INK, width=3.5)
            label = tex(expo, 30, EXPO).move_to(ax.c2p(0.27, 0.34) if key == "C" else ax.c2p(0.3, 0.16))
            panels.append(dict(frame=m.VGroup(frame, head, ax, tick_group, y_name), dots=dots, line=line, label=label, sc=sc))

        sc = panels[2]["sc"]
        extend = m.DashedVMobject(sc.curve(lambda v: fitted("C", v), RUNS["C"][-1] * 1.6, BIG_C * 1.8, INK, width=3.5), num_dashes=16)
        big = m.Dot(sc.p(BIG_C, BIG_LOSS), radius=0.1, color=GOOD).set_z_index(4)
        ring = m.Circle(radius=0.2).set_stroke(GOOD, width=3.5).move_to(big).set_z_index(4)
        big_label = text("big run", 22, GOOD, weight=m.BOLD).next_to(ring, m.LEFT, buff=0.12).shift(0.14 * m.DOWN)
        notes = m.VGroup(
            pill("each holds when the other two aren't the bottleneck", TEAL, TEAL_FILL, 22),
            pill("trends over 7+ orders of magnitude", MUTED, CARD, 22),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -3.2, 0])

        self.say(
            "[In twenty twenty, researchers at OpenAI trained language models across a huge range of sizes.]",
            m.AnimationGroup(header, m.FadeIn(source), *[m.FadeIn(p["frame"], shift=0.2 * m.UP) for p in panels]),
        )
        self.say(
            "[Against parameters], [against data], [and against compute], [the loss fell along a straight line.]",
            *[m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in p["dots"]], lag_ratio=0.1), m.Create(p["line"])) for p in panels],
            m.LaggedStart(*[m.FadeIn(p["label"], shift=0.15 * m.UP) for p in panels], lag_ratio=0.25),
        )
        self.say(
            "[Each law holds as long as the other two aren't holding it back], [and some trends spanned more than seven orders of magnitude.]",
            m.FadeIn(notes[0], shift=0.2 * m.UP),
            m.FadeIn(notes[1], shift=0.2 * m.UP),
        )
        self.say(
            "[Better still]: [fit the line on small, cheap runs], [extend it], [and it predicts a big run's loss before you train it.]",
            m.Indicate(panels[2]["frame"][1], color=COMPUTE, scale_factor=1.15),
            m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.6) for d in panels[2]["dots"]], lag_ratio=0.1),
            m.Create(extend, run_time=1.2),
            m.AnimationGroup(m.FadeIn(big, scale=0.3), m.Create(ring), m.FadeIn(big_label, shift=0.1 * m.DOWN)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ compute

    def compute(self):
        self.next_section(
            "Compute",
            notes="Training compute C ~ 6·N·D FLOPs: about 2 operations per parameter per token forward, 4 backward (Kaplan et al., 2020). As a rectangle: width N, height D, area C. Doubling C: twice as wide, twice as tall, or both sides x sqrt(2).",
        )
        header = self.new_header("COMPUTE", "Where the budget goes")
        formula = m.MathTex("C", r"\approx", "6", r"\cdot", "N", r"\cdot", "D", font_size=76, color=INK).move_to([-3.4, 1.45, 0])
        for i, c in ((0, COMPUTE), (4, PARAMS), (6, TOKENS)):
            formula[i].set_color(c)
        rows = m.VGroup(
            legend_row("C", COMPUTE, "training compute, in FLOPs"),
            legend_row("N", PARAMS, "parameters"),
            legend_row("D", TOKENS, "training tokens"),
            legend_row("6", INK, "operations per parameter, per token"),
        )
        for k, row in enumerate(rows):
            row[0].move_to([-5.9, 0.15 - 0.72 * k, 0])
            row[1].move_to([0, row[0].get_y(), 0]).align_to([-5.35, 0, 0], m.LEFT)
        split = text("2 forward + 4 backward", 22, MUTED).next_to(rows[3][1], m.DOWN, buff=0.18).align_to(rows[3][1], m.LEFT)

        corner = np.array([0.95, -2.85, 0])
        bw, bh = 2.35, 1.85
        base = box(bw, bh, corner, COMPUTE, AMBER_FILL).set_z_index(1)
        base_c = tex("C", 44, COMPUTE).move_to(base).set_z_index(2)
        n_label = tex("N", 36, PARAMS).next_to(base, m.DOWN, buff=0.15)
        d_label = tex("D", 36, TOKENS).next_to(base, m.LEFT, buff=0.18)
        wide = box(2 * bw, bh, corner, PARAMS, BG, opacity=0, dashed=True)
        tall = box(bw, 2 * bh, corner, TOKENS, BG, opacity=0, dashed=True)
        both = box(bw * np.sqrt(2), bh * np.sqrt(2), corner, GOOD, GOOD_FILL, opacity=0.35, stroke=4)
        wide_label = text("2× wider", 22, PARAMS, weight=m.BOLD).move_to([corner[0] + 1.6 * bw, corner[1] - 0.32, 0])
        tall_label = text("2× taller", 22, TOKENS, weight=m.BOLD).next_to(tall, m.UP, buff=0.15)
        both_label = text("both sides ×1.4", 22, GOOD, weight=m.BOLD).next_to(both.get_corner(m.UR), m.UR, buff=0.12)
        doubled = text("budget ×2 = area ×2", 24, COMPUTE, weight=m.BOLD).move_to([3.6, 1.75, 0])

        self.say(
            "[So where does the compute go?] [A handy rule: training takes about six times N times D operations.]",
            header,
            m.AnimationGroup(m.Write(formula, run_time=1.3), m.FadeIn(rows[0], shift=0.15 * m.UP)),
        )
        self.say(
            "[N is the parameters, and D the training tokens.] [Six is the work per parameter, per token]: [two for the forward pass, four for the backward pass.]",
            m.AnimationGroup(m.FadeIn(rows[1:3], shift=0.15 * m.UP), m.Indicate(formula[4], color=PARAMS, scale_factor=1.3), m.Indicate(formula[6], color=TOKENS, scale_factor=1.3)),
            m.AnimationGroup(m.FadeIn(rows[3], shift=0.15 * m.UP), m.Indicate(formula[2], color=INK, scale_factor=1.3)),
            m.FadeIn(split, shift=0.15 * m.UP),
        )
        self.say(
            "[Picture it as a rectangle]: [parameters across, tokens up], [and the area is the compute.]",
            m.Create(base.copy().set_fill(opacity=0)),
            m.AnimationGroup(m.FadeIn(n_label, shift=0.15 * m.UP), m.FadeIn(d_label, shift=0.15 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(base), m.FadeIn(base_c, scale=0.6)),
        )
        self.say(
            "[Double the budget, and the area doubles.] [Make it twice as wide], [twice as tall], [or grow both sides about one point four times.]",
            m.AnimationGroup(m.FadeIn(doubled, shift=0.15 * m.DOWN), m.Indicate(base_c, color=COMPUTE, scale_factor=1.4)),
            m.AnimationGroup(m.Create(wide), m.FadeIn(wide_label, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.Create(tall), m.FadeIn(tall_label, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(both), m.FadeIn(both_label, shift=0.15 * m.LEFT), wide.animate.set_stroke(opacity=0.35), tall.animate.set_stroke(opacity=0.35)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ compute-optimal

    def optimal(self):
        self.next_section(
            "Compute-optimal",
            notes="Fixed-budget (isoFLOP) curves from an illustrative fit in Hoffmann et al.'s form (E=1.69, A=406.4, a=b=0.3, B set for 20 tokens/param); minima found numerically. Hoffmann et al. (2022): scale N and D equally, ~20 tokens per parameter. Chinchilla 70B/1.4T vs Gopher 280B/300B, about the same compute; Chinchilla outperformed Gopher on a large range of downstream tasks.",
        )
        header = self.new_header("COMPUTE-OPTIMAL", "The best split for a budget")
        ax = frame_axes([-3.25, -0.1, 0], 6.1, 3.9)
        sc = Scale(ax, (10**8.1, 10**12.3), (1.95, 3.6), xlog=True, ylog=False)
        ticks = sc.xticks([10.0**e for e in range(9, 13)], [f"10^{{{e}}}" for e in range(9, 13)])
        x_name = text("parameters N (log scale)", 22, PARAMS, weight=m.SEMIBOLD).next_to(ax.c2p(0.5, 0), m.DOWN, buff=0.62)
        y_name = text("loss", 22, LOSS, weight=m.SEMIBOLD).next_to(ax.c2p(0, 1), m.UP, buff=0.22)
        tag = text("illustrative fit", 20, MUTED).move_to(ax.c2p(0.17, 0.1))
        curves, mins, budget_labels = m.VGroup(), m.VGroup(), m.VGroup()
        for c in BUDGETS:
            n = OPT[c]
            curves.add(sc.curve(lambda v, c=c: float(iso(c, v)), n / 14, n * 14, COMPUTE, width=4))
            mins.add(m.Dot(sc.p(n, float(iso(c, n))), radius=0.09, color=GOOD).set_z_index(3))
            budget_labels.add(tex(f"10^{{{int(np.log10(c))}}}", 26, COMPUTE).next_to(curves[-1].get_end(), m.RIGHT, buff=0.12))
        budget_name = text("budget (FLOPs)", 22, COMPUTE, weight=m.SEMIBOLD).move_to(ax.c2p(0.84, 0.93))
        frontier = m.DashedVMobject(
            polyline([sc.p(np.sqrt(c / 120), float(iso(c, np.sqrt(c / 120)))) for c in np.logspace(20.6, 24.4, 60)], GOOD, width=3.5),
            num_dashes=30,
        ).set_z_index(2)
        first = OPT[BUDGETS[0]]
        too_small = text("too small", 22, BAD, weight=m.BOLD).move_to(sc.p(first / 4.5, 3.45))
        fixed = text("one fixed budget", 22, COMPUTE, weight=m.SEMIBOLD).next_to(curves[0].get_end(), m.RIGHT, buff=0.12)
        too_big = text("too big", 22, BAD, weight=m.BOLD).move_to(sc.p(first * 9, 3.5))
        best_label = text("compute-optimal", 22, GOOD, weight=m.BOLD).next_to(mins[0], m.DOWN, buff=0.22)
        rule = pill("≈ 20 tokens per parameter", GOOD, GOOD_FILL, 24).move_to([-3.25, -3.3, 0])

        # Chinchilla and Gopher as compute rectangles (width ∝ parameters, height ∝ tokens)
        per_n, per_d = 1 / 100e9, 1 / 0.5e12
        ground = -2.1
        chin = box(N0 * per_n, D0 * per_d, [1.55, ground, 0], GOOD, GOOD_FILL)
        goph = box(GOPHER_N * per_n, GOPHER_D * per_d, [3.55, ground, 0], BAD, BAD_FILL)

        def caption(rect, name, color, params, tokens):
            lines = m.VGroup(text(name, 26, color, weight=m.BOLD), text(params, 22, PARAMS), text(tokens, 22, TOKENS))
            return lines.arrange(m.DOWN, buff=0.08).next_to(rect, m.UP, buff=0.2)

        chin_text = caption(chin, "Chinchilla", GOOD, "70B params", "1.4T tokens")
        goph_text = caption(goph, "Gopher", BAD, "280B params", "300B tokens")
        chin_ratio = text("20 tokens / param", 22, GOOD, weight=m.SEMIBOLD).next_to(chin, m.DOWN, buff=0.2)
        goph_ratio = text("≈ 1 token / param", 22, BAD, weight=m.SEMIBOLD).next_to(goph, m.DOWN, buff=0.2)
        chin_ratio.align_to(goph_ratio, m.DOWN)
        same = text("same area = same compute", 20, MUTED).move_to([4.95, 0.3, 0])
        verdict = pill("Chinchilla won across many tasks", GOOD, GOOD_FILL, 22).move_to([3.9, -3.3, 0])
        source = cite("Hoffmann et al., 2022", self.head)

        self.say(
            "[So which split is best?] [Fix the budget, and try different model sizes.]",
            m.AnimationGroup(header, m.Create(ax), m.FadeIn(ticks), m.FadeIn(x_name), m.FadeIn(y_name), m.FadeIn(tag)),
            m.AnimationGroup(m.Create(curves[0], run_time=1.6), m.FadeIn(fixed)),
        )
        self.say(
            "[Too small, and the model can't make use of all that data.] [Too big, and it sees too few tokens to learn well.]",
            m.FadeIn(too_small, shift=0.15 * m.DOWN),
            m.FadeIn(too_big, shift=0.15 * m.DOWN),
        )
        self.say(
            "[In between sits the sweet spot]: [the compute-optimal model.]",
            m.FadeIn(mins[0], scale=0.3),
            m.AnimationGroup(m.FadeIn(best_label, shift=0.15 * m.UP), m.Flash(mins[0].get_center(), color=GOOD, flash_radius=0.35)),
        )
        self.say(
            "[Repeat with bigger budgets], [and the sweet spots line up.]",
            m.AnimationGroup(
                m.FadeOut(best_label),
                m.FadeOut(too_small),
                m.FadeOut(too_big),
                m.FadeTransform(fixed, budget_labels[0]),
                m.FadeIn(budget_name),
                m.LaggedStart(*[m.AnimationGroup(m.Create(cv), m.FadeIn(d, scale=0.3), m.FadeIn(lab)) for cv, d, lab in zip(curves[1:], mins[1:], budget_labels[1:])], lag_ratio=0.35, run_time=2.2),
            ),
            m.Create(frontier, run_time=1.2),
        )
        self.say(
            "[In twenty twenty-two, researchers at DeepMind found] [that the best models see about twenty training tokens for every parameter.]",
            m.FadeIn(source),
            m.AnimationGroup(m.FadeIn(rule, shift=0.2 * m.UP), m.Indicate(frontier, color=GOOD, scale_factor=1.0)),
        )
        self.say(
            "[Their model, Chinchilla, had seventy billion parameters], [trained on one point four trillion tokens.]",
            m.AnimationGroup(m.FadeIn(chin_text[0], shift=0.15 * m.UP), m.FadeIn(chin_text[1]), m.GrowFromEdge(chin, m.DOWN)),
            m.FadeIn(chin_text[2], shift=0.15 * m.UP),
        )
        self.say(
            "[The earlier Gopher used about the same compute], [with four times the parameters, but far fewer tokens.]",
            m.AnimationGroup(m.GrowFromEdge(goph, m.LEFT), m.FadeIn(goph_text, shift=0.15 * m.UP), m.FadeIn(same)),
            m.AnimationGroup(m.FadeIn(chin_ratio), m.FadeIn(goph_ratio)),
        )
        self.say(
            "[Chinchilla beat it on a wide range of tasks.]",
            m.AnimationGroup(m.FadeIn(verdict, shift=0.2 * m.UP), m.Indicate(chin, color=GOOD, scale_factor=1.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the running example

    def doubling(self):
        self.next_section(
            "Doubling the budget",
            notes="C = 6ND and D = 20N give C = 120 N^2, N = sqrt(C/120): doubling C multiplies N and D by sqrt(2) ~ 1.41. From Chinchilla's 5.9e23 FLOPs (70B, 1.4T) to 1.2e24: 99B parameters, 2.0T tokens (numerical optimum of the illustrative fit agrees). Kaplan et al. (2020) allocation N ~ C^0.73, D ~ C^0.27: x1.66 and x1.21.",
        )
        header = self.new_header("THE EXAMPLE", "Doubling the budget")

        def line(parts, colors, size=48):
            f = m.MathTex(*parts, font_size=size, color=INK)
            for i, c in colors.items():
                f[i].set_color(c)
            return f

        eq_x = -3.6
        rows = [
            line(["C", r"\approx", "6", r"\cdot", "N", r"\cdot", "D"], {0: COMPUTE, 4: PARAMS, 6: TOKENS}),
            line(["D", r"\approx", "20", r"\cdot", "N"], {0: TOKENS, 4: PARAMS}),
            line(["C", r"\approx", "120", r"\cdot", "N^2"], {0: COMPUTE, 4: PARAMS}),
            line(["N", r"\approx", r"\sqrt{C / 120}"], {0: PARAMS}, size=56),
        ]
        for f, y in zip(rows, (1.6, 0.65, -0.3, -1.4)):
            f.shift([eq_x - f[1].get_x(), y - f[1].get_y(), 0])
        rows[3][2][2].set_color(COMPUTE)
        notes = m.VGroup(
            text("the rule of thumb", 22, MUTED).next_to(rows[1], m.RIGHT, buff=0.4),
            text("substitute", 22, MUTED).next_to(rows[2], m.RIGHT, buff=0.4),
        )
        frame = m.RoundedRectangle(corner_radius=0.15, width=rows[3].width + 0.45, height=rows[3].height + 0.3)
        frame.move_to(rows[3]).set_stroke(GOOD, width=3).set_fill(BG, opacity=0)
        result = tex(r"C \times 2 \;\Rightarrow\; N,\, D \times \sqrt{2} \approx 1.41", 38, GOOD).move_to([eq_x, -2.45, 0])
        old = pill(f"Kaplan et al., 2020 fits: N ×{KAPLAN_N:.1f}, D ×{KAPLAN_D:.1f}", MUTED, CARD, 22).move_to([-3.3, -3.35, 0])

        corner = np.array([0.95, -2.75, 0])
        side = 2.4
        base = box(side, side, corner, COMPUTE, AMBER_FILL).set_z_index(1)
        base_text = m.VGroup(tex(sci(C0), 32, COMPUTE), text("FLOPs", 22, MUTED)).arrange(m.DOWN, buff=0.12).move_to(base).set_z_index(2)
        grow = m.ValueTracker(1.0)

        def grown():
            k = grow.get_value()
            seen = float(np.clip((k - 1) / 0.08, 0, 1))
            return box(side * k, side * k, corner, GOOD, GOOD_FILL, opacity=0.55 * seen, stroke=4).set_stroke(opacity=seen)

        bigger = m.always_redraw(grown)
        new_c = tex(sci(2 * C0), 30, GOOD).move_to(corner + [side * np.sqrt(2) / 2, side + (np.sqrt(2) - 1) * side / 2, 0]).set_z_index(2)
        params = text(f"70B → {N1 / 1e9:.0f}B", 28, PARAMS, weight=m.SEMIBOLD).move_to([corner[0] + side * np.sqrt(2) / 2 - 0.45, -3.15, 0])
        tokens = m.VGroup(text(f"{D0 / 1e12:.1f}T → {D1 / 1e12:.1f}T", 26, TOKENS, weight=m.SEMIBOLD), text("tokens", 22, TOKENS)).arrange(m.DOWN, buff=0.12)
        tokens.move_to([corner[0] + side * np.sqrt(2) + 1.2, -1.0, 0])
        params_word = text("params", 22, PARAMS).next_to(params, m.RIGHT, buff=0.2)

        self.say(
            "[Now, back to our doubled budget.]",
            m.AnimationGroup(header, m.FadeIn(rows[0], shift=0.15 * m.UP), m.FadeIn(base), m.FadeIn(base_text)),
        )
        self.add(bigger)
        self.say(
            "[With twenty tokens per parameter], [compute grows with N squared], [so N is the square root of C over a hundred and twenty.]",
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.15 * m.UP), m.FadeIn(notes[0])),
            m.AnimationGroup(m.TransformFromCopy(rows[0], rows[2]), m.FadeIn(notes[1])),
            m.AnimationGroup(m.TransformFromCopy(rows[2], rows[3]), m.Create(frame)),
        )
        self.say(
            "[Double C, and N grows by the square root of two]: [about one point four times.] [The tokens grow just as much.]",
            m.AnimationGroup(grow.animate(run_time=1.8).set_value(np.sqrt(2)), m.FadeIn(new_c)),
            m.FadeIn(result, shift=0.15 * m.UP),
            m.Indicate(result, color=GOOD, scale_factor=1.08),
        )
        self.stop(bigger)
        self.say(
            "[From Chinchilla's budget], [that's about a hundred billion parameters], [trained on about two trillion tokens.]",
            m.Indicate(base_text[0], color=COMPUTE, scale_factor=1.2),
            m.AnimationGroup(m.FadeIn(params, shift=0.15 * m.UP), m.FadeIn(params_word, shift=0.15 * m.UP)),
            m.FadeIn(tokens, shift=0.15 * m.LEFT),
        )
        self.say(
            "[The twenty twenty fits had said: put most of it into size]: [one point seven times the parameters, but only one point two times the tokens.]",
            m.FadeIn(old, shift=0.2 * m.UP),
            m.Indicate(old, color=INK, scale_factor=1.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ serving

    def serving(self):
        self.next_section(
            "Serving costs",
            notes="Training is paid once; inference costs about 2N FLOPs per generated token. Illustrative fit: half the parameters with 2.4x the tokens (about 96 per parameter) reaches the same loss for 1.2x the training compute and half the cost per token. Llama 3 8B: over 15T tokens (Meta, 2024), ~1,900 tokens per parameter.",
        )
        header = self.new_header("IN PRACTICE", "Training past twenty tokens")
        ground = -2.15
        side = 1.7
        best = box(side, side, [-6.1, ground, 0], GOOD, GOOD_FILL)
        half = box(side * SERVE_X, side * TOKENS_X, [-2.85, ground, 0], TEAL, TEAL_FILL)

        def caption(rect, name, color, sub):
            lines = m.VGroup(text(name, 22, color, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08)
            return lines.next_to(rect, m.DOWN, buff=0.2)

        best_text = caption(best, "compute-optimal", GOOD, "20 tokens / param")
        half_text = caption(half, "half the size", TEAL, f"≈ {HALF_RATIO:.0f} tokens / param")
        half_text.align_to(best_text, m.UP)
        same = pill("same loss", LOSS, PINK_FILL, 22).move_to([-4.55, 0.95, 0])

        panel = card(6.0, 4.15, EDGE, CARD, stroke=2).move_to([3.6, -0.1, 0])
        left = panel.get_left()[0] + 0.35
        bar_x, unit = left + 1.75, 2.3

        def cost_row(title, color, y, values):
            head = text(title, 24, color, weight=m.BOLD).move_to([0, y, 0]).align_to([left, 0, 0], m.LEFT)
            bars, labels, names = m.VGroup(), m.VGroup(), m.VGroup()
            for k, (v, c, name) in enumerate(zip(values, (GOOD, TEAL), ("optimal", "half-size"))):
                yy = y - 0.6 - 0.5 * k
                bar = m.Rectangle(width=unit * v, height=0.32).set_stroke(c, width=2).set_fill(c, opacity=0.75)
                bar.move_to([bar_x + unit * v / 2, yy, 0])
                bars.add(bar)
                names.add(text(name, 22, c, weight=m.SEMIBOLD).move_to([0, yy, 0]).align_to([left, 0, 0], m.LEFT))
                labels.add(code(f"{v:.1f}×", 22, INK).next_to(bar, m.RIGHT, buff=0.15))
            return head, bars, labels, names

        train = cost_row("training compute · paid once", COMPUTE, 1.5, (1.0, TRAIN_X))
        serve = cost_row("cost per token served · forever", TOKENS, -0.3, (1.0, SERVE_X))
        llama = m.VGroup(
            text("Llama 3 8B: over 15T tokens", 24, INK, weight=m.BOLD),
            text(f"≈ {round(LLAMA_RATIO, -2):,.0f} tokens per parameter · Meta, 2024", 20, MUTED),
        ).arrange(m.DOWN, buff=0.1)
        llama_box = card(llama.width + 0.6, llama.height + 0.4, TEAL, CARD, stroke=2).move_to([3.6, -3.05, 0])
        llama.move_to(llama_box)

        def show(row, k):
            head, bars, labels, names = row
            return m.AnimationGroup(m.GrowFromEdge(bars[k], m.LEFT), m.FadeIn(labels[k]), m.FadeIn(names[k]))

        self.say(
            "[But training is paid once], [while serving is paid on every token, for every user, and bigger models cost more to run.]",
            m.AnimationGroup(header, m.FadeIn(panel), m.FadeIn(train[0]), show(train, 0), m.GrowFromEdge(best, m.DOWN), m.FadeIn(best_text)),
            m.AnimationGroup(m.FadeIn(serve[0]), show(serve, 0)),
        )
        self.say(
            "[In our example fit, a model half the size] [reaches the same loss], [if it trains on about two and a half times the tokens.]",
            m.AnimationGroup(m.GrowFromEdge(half, m.DOWN), m.FadeIn(half_text)),
            m.FadeIn(same, shift=0.15 * m.DOWN),
            m.Indicate(half, color=TEAL, scale_factor=1.06),
        )
        self.say(
            "[Training costs about twenty percent more], [but every token it serves costs half as much.]",
            show(train, 1),
            show(serve, 1),
        )
        self.say(
            "[That's why many of today's models are trained far past twenty tokens per parameter.]",
            m.Indicate(half_text, color=TEAL, scale_factor=1.15),
        )
        self.say(
            "[One popular eight billion parameter model saw more than fifteen trillion tokens]: [nearly two thousand per parameter.]",
            m.AnimationGroup(m.FadeIn(llama_box), m.FadeIn(llama[0], shift=0.15 * m.UP)),
            m.FadeIn(llama[1], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ caveats

    def caveats(self):
        self.next_section(
            "The catch",
            notes="Scaling laws are empirical fits: exponents and constants depend on data, tokenizer and architecture (Kaplan et al. note N_c etc. have no fundamental meaning); trust them inside the measured range; loss does not predict specific abilities.",
        )
        header = self.new_header("THE CATCH", "Fits, not laws of nature")
        items = (
            (variant("chart", TEAL, TEAL_FILL), "Fitted, not derived", "the exponents depend on the data, the tokenizer and the architecture", TEAL),
            ("alert", "Trust the measured range", "curves must level off eventually, and good data can run out", BAD),
            (variant("sparkle", VIOLET, VIOLET_FILL), "Loss is not ability", "a lower loss doesn't say which new skills will appear", VIOLET),
        )
        rows = m.VGroup()
        for name, title, desc, color in items:
            frame = card(12.4, 1.45, color, CARD, stroke=2)
            words = m.VGroup(text(title, 28, INK, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            body = m.VGroup(icon(name, 0.85), words).arrange(m.RIGHT, buff=0.45)
            body.move_to(frame).align_to(frame, m.LEFT).shift(0.45 * m.RIGHT)
            rows.add(m.VGroup(frame, body))
        rows.arrange(m.DOWN, buff=0.35).move_to([0, -0.5, 0])

        self.say(
            "[One honest caveat]: [scaling laws are empirical fits, not laws of nature.]",
            header,
            m.FadeIn(rows[0], shift=0.2 * m.UP),
        )
        self.say(
            "[The numbers depend on the data, the tokenizer and the architecture], [and they're only trustworthy inside the range where they were measured.]",
            m.Indicate(rows[0][1][1][1], color=INK, scale_factor=1.04),
            m.FadeIn(rows[1], shift=0.2 * m.UP),
        )
        self.say(
            "[And a lower loss doesn't tell you exactly which new skills a model will gain.]",
            m.FadeIn(rows[2], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Fit, balance, predict.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Fit", "Balance", "Predict"), (LOSS, GOOD, COMPUTE))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Scaling laws turn a training budget into a forecast.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [fit], [balance], [predict.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Scaling laws turn a training budget into a forecast.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
