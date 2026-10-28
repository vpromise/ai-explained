"""AI, Explained · Episode 40 · State Space Models & Mamba

Running example: streaming a very long sequence through a model whose memory never grows.
Every picture is computed here with numpy (seeded): a diagonal linear recurrence
h_t = A h_{t-1} + B x_t on a spiky stream (its state, its output y_t = C h_t, and the same
output as one convolution), a parallel (Hillis-Steele) scan of eight affine steps checked
against the plain loop, a selective (input-dependent step size) recurrence against a fixed
one on a stream with one important token, and the best linear readout of twelve inputs from
a three-number state. The asserts tie every number the narration quotes to the computation.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the data

# the recurrence: three state numbers, a diagonal A (each number fades at its own rate)
T = 40
SPIKES = {3: 1.0, 15: 0.6, 28: 0.8}
X = np.zeros(T)
for _t, _v in SPIKES.items():
    X[_t] = _v
DECAY = np.array([0.55, 0.85, 0.97])  # the diagonal of A: fast, medium, slow
B_IN = np.ones(3)
C_OUT = np.array([0.5, 0.3, 0.2])


def run_lti(xs, a, b):
    h, out = np.zeros(len(a)), []
    for x in xs:
        h = a * h + b * x
        out.append(h.copy())
    return np.array(out)


H = run_lti(X, DECAY, B_IN)  # (T, 3)
Y = H @ C_OUT
KERNEL = np.array([C_OUT @ (DECAY**k * B_IN) for k in range(T)])  # C A^k B
assert np.allclose(np.convolve(X, KERNEL)[:T], Y)  # the loop and the convolution agree
LATER = 3 + 10  # ten steps after the first spike, before the second
assert X[4 : LATER + 1].sum() == 0
assert H[LATER, 0] < 0.01  # "the fast one keeps under one percent"
assert abs(H[LATER, 2] - 0.75) < 0.02  # "the slow one, about three quarters" (0.74)

# attention's cost over a million tokens (causal: token t compares with t tokens)
N_STREAM = 10**6
PAIRS = N_STREAM * (N_STREAM + 1) // 2
assert abs(PAIRS / 5e11 - 1) < 0.01  # "about half a trillion comparisons"

# the scan: eight affine steps h -> a h + b, combined (Hillis-Steele) in three rounds
_rng = np.random.default_rng(40)
SA = np.round(_rng.uniform(0.5, 0.95, 8), 2)
SB = np.round(_rng.uniform(-0.6, 0.9, 8), 2)


def combine(first, then):
    (a1, b1), (a2, b2) = first, then
    return (a2 * a1, a2 * b1 + b2)


LEVELS = [list(zip(SA, SB))]
_d = 1
while _d < 8:
    _prev = LEVELS[-1]
    LEVELS.append([combine(_prev[i - _d], _prev[i]) if i >= _d else _prev[i] for i in range(8)])
    _d *= 2
LOOP = []
_h = 0.0
for _a, _b in zip(SA, SB):
    _h = _a * _h + _b
    LOOP.append(_h)
assert len(LEVELS) - 1 == 3  # "after three rounds"
assert np.allclose([b for _, b in LEVELS[-1]], LOOP)  # "exactly what the loop computes"
ROUNDS = math.ceil(math.log2(N_STREAM))
assert ROUNDS == 20  # "a million steps need only twenty rounds"

# selection: one key token early, forty filler tokens after it. A = -1, B = 1, zero-order
# hold: carry = exp(-delta), write = 1 - exp(-delta) (Mamba's discretisation of this case)
_rng = np.random.default_rng(7)
KEY, AFTER = 3, 40
L = KEY + 1 + AFTER
XS = _rng.uniform(-0.6, 0.6, L)
XS[KEY] = 0.9
MARK = np.zeros(L)
MARK[KEY] = 1.0


def softplus(z):
    return np.log1p(np.exp(z))


DELTA_SEL = softplus(10 * MARK - 6)  # computed from the token: large for the key, tiny for filler
DELTA_FIX = np.full(L, 0.4)  # one step size for every token


def run_gate(delta):
    keep, h, out = np.exp(-delta), 0.0, []
    for t in range(L):
        h = keep[t] * h + (1 - keep[t]) * XS[t]
        out.append(h)
    return np.array(out)


H_SEL, H_FIX = run_gate(DELTA_SEL), run_gate(DELTA_FIX)


def kept(delta):
    """How much of the key token's value is still in the state at the end."""
    keep = np.exp(-delta)
    return float((1 - keep[KEY]) * np.prod(keep[KEY + 1 :]))


KEEP_SEL, KEEP_FIX = kept(DELTA_SEL), kept(DELTA_FIX)
assert L - KEY - 1 == 40  # "forty filler tokens follow"
assert KEEP_FIX < 0.01  # "the fixed model keeps almost nothing"
assert 0.85 < KEEP_SEL < 0.9  # "almost ninety percent" (89%)
assert DELTA_SEL[KEY] > 1000 * DELTA_SEL[KEY + 1]  # "a large delta ... the filler almost none"

# the catch: twelve numbers through the three-number state, then the best linear readout
_rng = np.random.default_rng(12)
XC = _rng.normal(0, 1, 12)
COPY_M = np.stack([DECAY ** (11 - t) * B_IN for t in range(12)], axis=1)  # state = COPY_M @ x
XR = np.linalg.pinv(COPY_M) @ (COPY_M @ XC)
FRACTION = float(XR @ XR / (XC @ XC))
assert np.linalg.matrix_rank(COPY_M) == 3
assert 0.15 < FRACTION < 0.25  # "recovers only about a fifth of the signal" (19%)

# the hook: an audio-like stream, and the state an exponential average would keep of it
_rng = np.random.default_rng(2026)
_i = np.arange(420)
HOOK = np.clip(0.55 * np.sin(2 * np.pi * _i / 17) * (0.6 + 0.4 * np.sin(2 * np.pi * _i / 97)) + 0.28 * _rng.normal(size=_i.size), -1, 1)
HOOK_H = run_lti(HOOK, DECAY, 1 - DECAY)

# ---------------------------------------------------------------------------- helpers

CHANNELS = (PINK, BLUE, LIME)  # fast, medium, slow state numbers


def formula(*parts, size=40):
    """A MathTex from (latex, colour) pairs, one part each."""
    t = m.MathTex(*[p for p, _ in parts], font_size=size, color=INK)
    for part, (_, c) in zip(t, parts):
        part.set_color(c)
    return t


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def polyline(points, color, width=4, opacity=1.0):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width, opacity=opacity)


def cite(s):
    return text(s, 20, MUTED)


def mix(a, b, t):
    """A colour between two hex colours."""
    t = float(np.clip(t, 0, 1))
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * t)


def stem(ax, t, v, color, width=3.5, radius=0.05):
    """A lollipop: a line from the axis to the value, and a dot on top."""
    base, top = ax.c2p(t, 0), ax.c2p(t, v)
    if abs(top[1] - base[1]) < 0.02:
        return m.Dot(base, radius=radius * 0.8, color=color)
    return m.VGroup(m.Line(base, top).set_stroke(color, width=width), m.Dot(top, radius=radius, color=color))


def bar(ax, t, v, color, half=0.4, opacity=0.9, least=0.012):
    """A bar standing on the axis at step t."""
    left, base = ax.c2p(t - half, 0), ax.c2p(t, 0)
    height = max(ax.c2p(t, v)[1] - base[1], least)
    rect = m.Rectangle(width=2 * (base[0] - left[0]), height=height, stroke_width=0).set_fill(color, opacity=opacity)
    return rect.move_to([base[0], base[1], 0], aligned_edge=m.DOWN)


def trace(ax, values, upto, color, width=4):
    """The line through values[0..upto] (upto may be fractional)."""
    k = int(np.floor(upto))
    points = [ax.c2p(t, values[t]) for t in range(k + 1)]
    if upto > k and k + 1 < len(values):
        a = upto - k
        points.append(ax.c2p(upto, (1 - a) * values[k] + a * values[k + 1]))
    if len(points) < 2:
        points.append(points[-1] + 0.001 * m.RIGHT)
    return polyline(points, color, width)


def at(values, upto):
    k = int(np.floor(upto))
    if k + 1 >= len(values):
        return values[-1]
    a = upto - k
    return (1 - a) * values[k] + a * values[k + 1]


# ---------------------------------------------------------------------------- the episode


class Ep40StateSpaceMamba(Episode):
    number = 40

    def lesson(self):
        self.hook()
        self.problem()
        self.recurrence()
        self.convolution()
        self.scan()
        self.selection()
        self.step_size()
        self.hardware()
        self.catch()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A stream a million steps long; a model with a memory that never grows. The waveform is seeded numpy noise and sines (illustrative); the three cells show an exponential-average state of it.")
        win = card(8.1, 2.3, EDGE, CARD).move_to([-2.45, 1.0, 0])
        x_left, x_right = win.get_left()[0] + 0.3, win.get_right()[0] - 0.3
        gap = 0.16
        visible = int((x_right - x_left) / gap)
        base_y, amp = win.get_y(), 0.85
        scroll = m.ValueTracker(0.0)
        scroll_end = float(len(HOOK) - visible - 3)

        def wave():
            s = scroll.get_value()
            first = int(np.floor(s))
            lines = m.VGroup()
            for i in range(first, first + visible + 2):
                x = x_left + (i - s) * gap
                if x < x_left - 1e-6 or x > x_right + 1e-6:
                    continue
                v = HOOK[i] * amp
                if abs(v) < 0.03:
                    v = 0.03
                lines.add(m.Line([x, base_y, 0], [x, base_y + v, 0]).set_stroke(SKY, width=3))
            return lines

        stream = m.always_redraw(wave)
        stream_label = text("the stream", 22, MUTED).next_to(win, m.UP, buff=0.15).align_to(win, m.LEFT)

        box = card(3.5, 2.3, PINK, CARD).move_to([4.65, 1.0, 0])
        box_title = text("the model", 26, INK, weight=m.BOLD).move_to(box.get_top() + 0.42 * m.DOWN)
        cell_x = [box.get_x() + 0.75 * (k - 1) for k in range(3)]

        def cells():
            newest = min(int(np.floor(scroll.get_value())) + visible, len(HOOK) - 1)
            h = HOOK_H[newest]
            return m.VGroup(
                *[
                    m.Square(0.58).set_stroke(PINK, width=2.5).set_fill(mix(PINK_FILL, PINK, 0.15 + 1.6 * abs(h[k])), opacity=1).move_to([cell_x[k], box.get_y() - 0.05, 0])
                    for k in range(3)
                ]
            )

        state = m.always_redraw(cells)
        state_label = text("memory: 3 numbers", 22, MUTED).move_to(box.get_bottom() + 0.38 * m.UP)
        feed = arrow([win.get_right()[0] + 0.05, 1.0, 0], [box.get_left()[0] - 0.05, 1.0, 0], MUTED)

        count = m.ValueTracker(0)
        number = m.DecimalNumber(0, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=36, color=INK)
        steps = text("steps so far:", 26, MUTED)
        number.set_value(1_000_000)
        counter = m.VGroup(steps, number).arrange(m.RIGHT, buff=0.3).move_to([-2.3, -0.6, 0])
        on_baseline(counter)
        number.set_value(0)
        number.add_updater(lambda d: d.set_value(int(count.get_value())))

        name = text("State space models", 54, INK, weight=m.BOLD)
        mamba = pill("Mamba", PINK, PINK_FILL, 30)
        title = m.VGroup(name, mamba).arrange(m.RIGHT, buff=0.6).move_to([0, -2.35, 0])
        mamba_cite = cite("Gu & Dao, 2023").next_to(mamba, m.DOWN, buff=0.2)
        halo = glow(PINK, radius=3.2).move_to(name)

        self.say(
            "So... [imagine a sequence that never stops arriving]: [audio, sensor readings, or a genome, a million steps long.]",
            m.AnimationGroup(m.FadeIn(win), m.FadeIn(stream_label), m.FadeIn(stream)),
            m.AnimationGroup(
                scroll.animate(run_time=5.5, rate_func=m.linear).set_value(scroll_end * 0.45),
                count.animate(run_time=5.5, rate_func=m.rush_into).set_value(1_000_000),
                m.FadeIn(counter, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[We want a model that reads it one step at a time], [carrying a small memory that never grows.]",
            m.AnimationGroup(m.GrowArrow(feed), m.FadeIn(box), m.FadeIn(box_title), m.FadeIn(state), scroll.animate(run_time=3.0, rate_func=m.linear).set_value(scroll_end * 0.7)),
            m.AnimationGroup(m.FadeIn(state_label, shift=0.15 * m.UP), scroll.animate(run_time=3.0, rate_func=m.linear).set_value(scroll_end)),
        )
        self.stop(stream, state, number)
        self.say(
            "[That's a state space model], [and Mamba is the best-known example.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(name, shift=0.3 * m.UP)),
            m.AnimationGroup(m.FadeIn(mamba, scale=1.2), m.FadeIn(mamba_cite)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the problem

    def problem(self):
        self.next_section("The problem", notes="Attention (Ep 37): the KV cache grows by one entry per token and step n compares with all n tokens; a million tokens cost n(n+1)/2 = 500,000,500,000 comparisons. An SSM keeps a fixed state: a million updates.")
        lp = card(6.2, 5.0, VIOLET, CARD, stroke=2).move_to([-3.4, -0.5, 0])
        rp = card(6.2, 5.0, PINK, CARD, stroke=2).move_to([3.4, -0.5, 0])
        lt = text("Attention", 28, VIOLET, weight=m.BOLD).move_to(lp.get_top() + 0.42 * m.DOWN)
        rt = text("State space model", 28, PINK, weight=m.BOLD).move_to(rp.get_top() + 0.42 * m.DOWN)
        row_y, cache_y, size = 0.35, -0.3, 0.46

        def px(cx, i):
            return cx + (i - 3.5) * 0.68

        def token(cx, i, opacity=1.0):
            return m.Square(size).set_stroke(SKY, width=2.5, opacity=opacity).set_fill(SKY_FILL, opacity=1).move_to([px(cx, i), row_y, 0])

        n = m.ValueTracker(4.0)

        def attention():
            k = int(np.floor(n.get_value() + 1e-6))
            group = m.VGroup()
            for i in range(k):
                group.add(token(-3.4, i))
                group.add(m.Square(0.36).set_stroke(VIOLET, width=2).set_fill(VIOLET_FILL, opacity=1).move_to([px(-3.4, i), cache_y, 0]))
            start = np.array([px(-3.4, k - 1) - 0.08, row_y + size / 2 + 0.04, 0])
            for j in range(k - 1):
                end = np.array([px(-3.4, j), row_y + size / 2 + 0.04, 0])
                group.add(m.ArcBetweenPoints(start, end, angle=0.36 * np.pi).set_stroke(VIOLET, width=2.5))
            group.add(m.Square(size + 0.14).set_stroke(INK, width=2.5).move_to([px(-3.4, k - 1), row_y, 0]))
            return group

        attn = m.always_redraw(attention)
        cache_note = text("KV cache: one entry per token", 22, MUTED).move_to([-3.4, -0.9, 0])
        step_note = text("step n compares with all n", 22, MUTED).move_to([-3.4, -1.32, 0])
        pairs = formula((r"\frac{n(n+1)}{2}", INK), (r"\approx", INK), (r"5 \times 10^{11}", BAD), size=36).move_to([-3.4, -1.98, 0])
        pairs_note = text("comparisons for n = 10⁶ tokens", 22, MUTED).move_to([-3.4, -2.65, 0])

        r_tokens = m.VGroup(*[token(3.4, i, 1.0) for i in range(8)])
        for tok in r_tokens[:-1]:
            tok.set_stroke(opacity=0.35).set_fill(opacity=0.45)
        r_ring = m.Square(size + 0.14).set_stroke(INK, width=2.5).move_to(r_tokens[-1])
        cells = m.VGroup(*[m.Square(0.42).set_stroke(PINK, width=2.5).set_fill(PINK_FILL, opacity=1) for _ in range(3)]).arrange(m.RIGHT, buff=0.12)
        cells.move_to([3.4, cache_y, 0])
        h_label = text("state", 22, PINK, weight=m.SEMIBOLD).next_to(cells, m.LEFT, buff=0.25)
        fold = arrow(r_tokens[-1].get_bottom() + 0.02 * m.DOWN, cells.get_right() + 0.08 * m.RIGHT, PINK, buff=0.05)
        state_note = text("state: 3 numbers, always", 22, MUTED).move_to([3.4, -0.9, 0])
        same_note = text("every step: the same work", 22, MUTED).move_to([3.4, -1.32, 0])
        linear = text("a million updates", 32, GOOD, weight=m.BOLD).move_to([3.4, -1.98, 0])
        linear_note = text("linear in the length", 22, MUTED).move_to([3.4, -2.65, 0])

        self.say(
            "[As we saw in episode thirty-seven, attention lets each new token look back at every token before it.]",
            m.AnimationGroup(self.new_header("THE PROBLEM", "Attention grows with the stream"), m.FadeIn(lp), m.FadeIn(lt), m.FadeIn(rp), m.FadeIn(rt), m.FadeIn(attn)),
        )
        self.say(
            "[So its key-value cache grows with every token], [and each step costs more than the last.]",
            m.AnimationGroup(n.animate(run_time=3.0, rate_func=m.linear).set_value(8.0), m.FadeIn(cache_note, shift=0.15 * m.UP)),
            m.FadeIn(step_note, shift=0.15 * m.UP),
        )
        self.stop(attn)
        self.say(
            "[Over a million tokens], [that's about half a trillion comparisons.]",
            m.FadeIn(pairs_note, shift=0.15 * m.UP),
            m.Write(pairs),
        )
        self.say(
            "[A state space model folds each token into a fixed-size state instead]: [constant memory, and the same cost at every step.]",
            m.AnimationGroup(m.FadeIn(r_tokens), m.FadeIn(r_ring), m.FadeIn(cells), m.FadeIn(h_label), m.GrowArrow(fold)),
            m.AnimationGroup(m.FadeIn(state_note, shift=0.15 * m.UP), m.FadeIn(same_note, shift=0.15 * m.UP)),
        )
        self.say(
            "[A million tokens, a million updates]: [linear in the length.]",
            m.FadeIn(linear, shift=0.15 * m.UP),
            m.FadeIn(linear_note),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the recurrence

    def recurrence(self):
        self.next_section("The recurrence", notes="h_t = A h_{t-1} + B x_t with A = diag(0.55, 0.85, 0.97), B = 1, run on a stream with spikes at steps 3, 15, 28 (illustrative input, computed state). Ten steps after a spike: 0.55^10 = 0.25%, 0.97^10 = 74%.")
        rec = formula(("h_t", PINK), ("=", INK), ("A", AMBER), ("h_{t-1}", PINK), ("+", INK), ("B", TEAL), ("x_t", SKY), size=66).move_to([0, 0.55, 0])
        self.rec = rec
        below = [(rec[0], "new state", PINK), (rec[3], "old state", PINK), (rec[6], "input", SKY)]
        above = [(rec[2], "carry-over", AMBER), (rec[5], "write", TEAL)]
        under = m.VGroup(*[text(s, 26, c, weight=m.SEMIBOLD).next_to(p, m.DOWN, buff=0.4) for p, s, c in below])
        over = m.VGroup(*[text(s, 26, c, weight=m.SEMIBOLD).next_to(p, m.UP, buff=0.4) for p, s, c in above])
        for group, y in ((under, min(u.get_y() for u in under)), (over, max(o.get_y() for o in over))):
            for label in group:
                label.set_y(y)
            on_baseline(group)
        diag = m.VGroup(
            text("A = diag(", 26, INK, weight=m.SEMIBOLD),
            *[text(f"{a:.2f}" + (", " if k < 2 else ""), 26, CHANNELS[k], weight=m.SEMIBOLD) for k, a in enumerate(DECAY)],
            text(")", 26, INK, weight=m.SEMIBOLD),
        ).arrange(m.RIGHT, buff=0.08)
        on_baseline(diag)
        diag.move_to([3.85, 1.85, 0])
        self.diag = diag

        inp = axes([0, T + 0.5, 5], [0, 1.2, 0.5], 8.6, 1.05).move_to([-2.15, 0.3, 0])
        inp_label = text("input x", 22, SKY, weight=m.SEMIBOLD).next_to(inp.c2p(T, 1.2), m.LEFT, buff=0.1).shift(0.05 * m.DOWN)
        st = axes([0, T + 0.5, 5], [0, 1.9, 0.5], 8.6, 2.35).move_to([-2.15, -2.05, 0])
        st_label = text("state h", 22, PINK, weight=m.SEMIBOLD).next_to(st.c2p(0, 1.9), m.RIGHT, buff=0.25).shift(0.05 * m.DOWN)
        self.inp, self.st = inp, st
        dots = m.VGroup(*[stem(inp, t, X[t], SKY) for t in range(T)])
        clock = m.ValueTracker(0.0)
        traces = [m.always_redraw(lambda k=k: trace(st, H[:, k], clock.get_value(), CHANNELS[k], width=4)) for k in range(3)]

        box = card(3.3, 4.35, EDGE, CARD).move_to([4.95, -1.05, 0])
        box_title = text("the state now", 24, INK, weight=m.BOLD).move_to(box.get_top() + 0.38 * m.DOWN)
        base_y, scale = -1.5, 1.0
        bx = [box.get_x() + 0.85 * (k - 1) for k in range(3)]

        def bars():
            h = at(H, clock.get_value())
            return m.VGroup(
                *[
                    m.Rectangle(width=0.5, height=max(scale * h[k], 0.012), stroke_width=0).set_fill(CHANNELS[k], opacity=0.9).move_to([bx[k], base_y, 0], aligned_edge=m.DOWN)
                    for k in range(3)
                ]
            )

        state_bars = m.always_redraw(bars)
        floor = m.Line([box.get_left()[0] + 0.35, base_y, 0], [box.get_right()[0] - 0.35, base_y, 0]).set_stroke(FAINT, width=2)
        rates = m.VGroup(*[text(f"×{a:.2f}", 22, CHANNELS[k], weight=m.SEMIBOLD).move_to([bx[k], base_y - 0.3, 0]) for k, a in enumerate(DECAY)])
        speeds = m.VGroup(text("fast", 20, MUTED).move_to([bx[0], base_y - 0.66, 0]), text("slow", 20, MUTED).move_to([bx[2], base_y - 0.66, 0]))
        memory = pill("memory: 3 numbers", PINK, PINK_FILL, 22).move_to([box.get_x(), box.get_bottom()[1] + 0.45, 0])
        self.box, self.box_title, self.floor, self.rates, self.speeds, self.memory = box, box_title, floor, rates, speeds, memory

        rings = m.VGroup(*[m.Circle(radius=0.13).set_stroke(INK, width=3).move_to(st.c2p(LATER, H[LATER, k])) for k in (0, 2)])
        fast_pct = text(f"{100 * H[LATER, 0]:.1f}%", 22, PINK, weight=m.BOLD).move_to([bx[0], base_y + scale * H[LATER, 0] + 0.25, 0])
        slow_pct = text(f"{100 * H[LATER, 2]:.0f}%", 22, LIME, weight=m.BOLD).move_to([bx[2], base_y + scale * H[LATER, 2] + 0.25, 0])

        self.say(
            "[Here's the core of the model: one update, applied once per token.]",
            m.AnimationGroup(self.new_header("THE RECURRENCE", "One small update per token"), m.Write(rec, run_time=1.6)),
        )
        self.say(
            "[On the left is the new state, built from the old one]: [a small vector that summarises the stream so far.]",
            m.AnimationGroup(m.FadeIn(under[0], shift=0.15 * m.UP), m.FadeIn(under[1], shift=0.15 * m.UP)),
            m.Circumscribe(m.VGroup(rec[0]), color=PINK, buff=0.1),
        )
        self.say(
            "[A carries the old state forward], [and B writes the new input into it.]",
            m.FadeIn(over[0], shift=0.15 * m.DOWN),
            m.AnimationGroup(m.FadeIn(over[1], shift=0.15 * m.DOWN), m.FadeIn(under[2], shift=0.15 * m.UP)),
        )
        self.say(
            "[Here, A is diagonal], [so each number in the state simply fades at its own rate.]",
            m.AnimationGroup(m.FadeOut(under), m.FadeOut(over), rec.animate.scale(50 / 66).move_to([-2.7, 1.85, 0]), m.FadeIn(diag, shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(box_title), m.Create(floor), m.FadeIn(rates), m.FadeIn(speeds)),
        )
        self.say(
            "[Let's run it, for example, on a stream with three spikes.]",
            m.AnimationGroup(
                m.Create(inp), m.FadeIn(inp_label), m.Create(st), m.FadeIn(st_label),
                m.LaggedStart(*[m.FadeIn(d) for d in dots], lag_ratio=0.03),
            ),
        )
        self.add(*traces, state_bars)
        self.say(
            "[When a spike arrives, every number jumps], [then fades at its own pace.]",
            clock.animate(run_time=1.6, rate_func=m.linear).set_value(4.0),
            clock.animate(run_time=2.4, rate_func=m.linear).set_value(float(LATER)),
        )
        self.say(
            "[Ten steps later, the fast one keeps under one percent], [and the slow one, about three quarters.]",
            m.AnimationGroup(m.Create(rings[0]), m.FadeIn(fast_pct, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(rings[1]), m.FadeIn(slow_pct, shift=0.15 * m.UP)),
        )
        self.say(
            "[The stream keeps coming], [but the memory never grows]: [three numbers, at step ten or step ten million.]",
            m.AnimationGroup(m.FadeOut(rings), m.FadeOut(fast_pct), m.FadeOut(slow_pct), clock.animate(run_time=4.0, rate_func=m.linear).set_value(float(T - 1))),
            m.FadeIn(memory, shift=0.15 * m.UP),
            m.Circumscribe(memory, color=PINK, buff=0.08),
        )
        self.stop(*traces, state_bars)
        self.traces, self.state_bars, self.dots, self.labels = traces, state_bars, dots, (inp_label, st_label)
        self.wait(0.3)

    # ------------------------------------------------------------------ the convolution

    def convolution(self):
        self.next_section("Unrolled: a convolution", notes="y_t = C h_t with C = (0.5, 0.3, 0.2); unrolled, y = K * x with K_k = C A^k B (asserted equal to the loop). With A, B, C fixed this is one convolution over the whole sequence (S4: Gu, Goel & Re, 2021, 'Efficiently Modeling Long Sequences with Structured State Spaces').")
        st, rec = self.st, self.rec
        out = formula(("y_t", GOOD), ("=", INK), ("C", GOOD), ("h_t", PINK), size=44)
        unrolled = formula(("=", INK), (r"\sum_{k \ge 0}", INK), ("C", GOOD), ("A^k", AMBER), ("B", TEAL), (r"\,x_{t-k}", SKY), size=44)
        small = rec.copy().scale(0.8).move_to([-4.95, 2.0, 0])
        out.move_to([-1.55, 2.0, 0])
        unrolled.next_to(out, m.RIGHT, buff=0.22)
        unrolled.shift((out[1].get_y() - unrolled[0].get_y()) * m.UP)
        y_line = polyline([st.c2p(t, Y[t]) for t in range(T)], GOOD, width=5)
        y_label = text("output y", 22, GOOD, weight=m.SEMIBOLD).next_to(self.labels[1], m.RIGHT, buff=0.4)
        y_dots = m.VGroup(*[m.Dot(st.c2p(t, Y[t]), radius=0.06, color=GOOD) for t in range(T)])
        y_dots.set_z_index(3)

        box = self.box
        k_title = text("the kernel", 24, INK, weight=m.BOLD).move_to(self.box_title)
        kx = axes([0, 16.5, 4], [0, 1.1, 0.5], 2.7, 2.2).move_to([box.get_x(), -0.75, 0])
        k_bars = m.VGroup(*[bar(kx, k + 0.5, KERNEL[k], GOOD, half=0.36) for k in range(16)])
        k_note = text("0 … 15 steps back", 20, MUTED).next_to(kx, m.DOWN, buff=0.22)
        fixed = pill("A, B, C fixed", GOOD, GOOD_FILL, 22).move_to(self.memory)
        parallel = pill("every output at once", GOOD, GOOD_FILL, 22).move_to([5.15, 2.0, 0])
        s4 = cite("S4 · Gu et al., 2021").next_to(parallel, m.DOWN, buff=0.18)
        self.kernel_bars = k_bars

        self.say(
            "[To get an output, C reads the state.]",
            m.AnimationGroup(
                self.new_header("TRAINING · 1", "Unroll it: one big convolution"),
                m.FadeOut(self.diag),
                m.Transform(rec, small),
                m.Write(out),
                *[tr.animate.set_stroke(opacity=0.22) for tr in self.traces],
                m.Create(y_line, run_time=2.0),
                m.FadeIn(y_label),
            ),
        )
        self.say(
            "[Unroll the loop, and each output becomes a weighted sum of past inputs], [with weights that fade like powers of A.]",
            m.Write(unrolled, run_time=2.0),
            m.AnimationGroup(
                m.FadeOut(self.state_bars), m.FadeOut(self.box_title), m.FadeOut(self.floor), m.FadeOut(self.rates), m.FadeOut(self.speeds), m.FadeOut(self.memory),
                m.FadeIn(k_title), m.Create(kx), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in k_bars], lag_ratio=0.08), m.FadeIn(k_note),
            ),
        )
        self.say(
            "[When A, B and C are fixed, those weights form one kernel], [so training is a single convolution over the whole sequence, in parallel.]",
            m.AnimationGroup(m.FadeIn(fixed, shift=0.15 * m.UP), m.Circumscribe(m.VGroup(kx, k_bars), color=GOOD, buff=0.12)),
            m.AnimationGroup(m.FadeIn(y_dots, scale=0.4), m.FadeIn(parallel, shift=0.15 * m.UP), m.FadeIn(s4)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the scan

    def scan(self):
        self.next_section("A parallel scan", notes="Eight affine steps h -> a h + b (seeded numpy); combining is associative, so a Hillis-Steele scan (Hillis & Steele, 1986) gives every prefix in log2(8) = 3 rounds, asserted equal to the loop. A million steps: 20 rounds. Work-efficient scans (Blelloch, 1990) keep total work O(n).")
        xs = [-4.15 + 1.32 * i for i in range(8)]
        ys = [1.15, -0.08, -1.31, -2.54]
        bw, bh = 1.08, 0.74
        comp = formula(
            ("(", INK), ("a_2", AMBER), (",", INK), ("b_2", TEAL), (")", INK), (r"\circ", INK), ("(", INK), ("a_1", AMBER), (",", INK), ("b_1", TEAL), (")", INK),
            ("=", INK), ("(", INK), ("a_2 a_1", AMBER), (",", INK), ("a_2 b_1 + b_2", TEAL), (")", INK), size=40,
        ).move_to([-1.7, 2.08, 0])
        refs = cite("Hillis & Steele, 1986 · Blelloch, 1990").move_to([4.45, 2.08, 0])

        def box(i, r, color=EDGE):
            return card(bw, bh, color, CARD, radius=0.14, stroke=2).move_to([xs[i], ys[r], 0])

        row0 = m.VGroup()
        for i in range(8):
            b = box(i, 0, SKY)
            nums = m.VGroup(code(f"×{SA[i]:.2f}", 20, AMBER), code(f"{SB[i]:+.2f}", 20, TEAL)).arrange(m.DOWN, buff=0.06).move_to(b)
            row0.add(m.VGroup(b, nums))
        labels = m.VGroup(
            *[text(s, 22, MUTED, weight=m.SEMIBOLD) for s in ("steps", "round 1", "round 2", "round 3")]
        )
        for lab, y in zip(labels, ys):
            lab.move_to([0, y, 0]).align_to([xs[0] - bw / 2 - 0.25, 0, 0], m.RIGHT)
        rounds = []
        for r in (1, 2, 3):
            d = 2 ** (r - 1)
            links = m.VGroup()
            boxes = m.VGroup()
            for i in range(8):
                top = np.array([xs[i], ys[r] + bh / 2, 0])
                links.add(m.Line([xs[i], ys[r - 1] - bh / 2, 0], top).set_stroke(FAINT, width=2.5))
                if i >= d:
                    links.add(m.Line([xs[i - d], ys[r - 1] - bh / 2, 0], top).set_stroke(TEAL, width=2.5))
                lo = max(0, i - (2**r - 1))
                if r < 3:
                    b = box(i, r)
                    lab = code(f"{lo + 1}–{i + 1}" if lo < i else f"{i + 1}", 20, MUTED).move_to(b)
                else:
                    b = box(i, r, PINK)
                    lab = code(f"{LOOP[i]:+.2f}", 22, PINK).move_to(b)
                boxes.add(m.VGroup(b, lab))
            rounds.append((links, boxes))
        check = pill("✓ the same as the step-by-step loop", GOOD, GOOD_FILL, 22).move_to([-1.9, -3.33, 0])
        million = text(f"a million steps: {ROUNDS} rounds", 24, INK, weight=m.SEMIBOLD).move_to([3.6, -3.33, 0])
        caption = text("each step: scale the state, then add", 22, MUTED).next_to(row0, m.UP, buff=0.18).align_to(row0, m.LEFT)

        self.say(
            "[But a convolution needs fixed parameters.] [A scan works even when they change at every step.]",
            self.new_header("TRAINING · 2", "A parallel scan"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(b, shift=0.15 * m.UP) for b in row0], lag_ratio=0.12), m.FadeIn(labels[0])),
        )
        self.say(
            "[Each step is a tiny update]: [scale the state, then add the new input.]",
            m.Circumscribe(row0[0], color=SKY, buff=0.08),
            m.FadeIn(caption, shift=0.1 * m.UP),
        )
        self.say(
            "[Two updates combine into one update of the same kind], [and the grouping doesn't matter.]",
            m.AnimationGroup(m.FadeOut(caption), m.Write(comp, run_time=2.0)),
            m.FadeIn(refs),
        )
        self.say(
            "[So we combine in parallel]: [neighbours one apart], [then two apart], [then four apart.]",
            m.Indicate(comp[5], color=TEAL, scale_factor=1.4),
            *[m.AnimationGroup(m.Create(lk), m.FadeIn(bx), m.FadeIn(labels[r + 1])) for r, (lk, bx) in enumerate(rounds)],
        )
        self.say(
            "[After three rounds, every box holds the state at its step], [exactly what the loop computes.]",
            m.LaggedStart(*[m.Indicate(b, color=PINK, scale_factor=1.08) for b in rounds[2][1]], lag_ratio=0.1),
            m.FadeIn(check, shift=0.15 * m.UP),
        )
        self.say(
            "[A million steps need only twenty rounds], [and work-efficient scans keep the total work linear.]",
            m.FadeIn(million, shift=0.15 * m.UP),
            m.Circumscribe(refs, color=MUTED, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ selection

    def stream_plots(self):
        xl, width = -4.85, 7.6
        cx = xl + width / 2
        inp = axes([0, L + 0.5, 5], [-0.7, 1.05, 0.5], width, 1.35).move_to([cx, 0.5, 0])
        dl = axes([0, L + 0.5, 5], [0, 4.5, 1.5], width, 0.75).move_to([cx, -0.8, 0])
        st = axes([0, L + 0.5, 5], [-0.6, 1.05, 0.5], width, 1.9).move_to([cx, -2.4, 0])
        labels = m.VGroup(
            text("tokens", 22, SKY, weight=m.SEMIBOLD).move_to([0, inp.c2p(0, 0)[1], 0]),
            text("step size Δ", 22, AMBER, weight=m.SEMIBOLD).move_to([0, dl.get_y(), 0]),
            text("state", 22, PINK, weight=m.SEMIBOLD).move_to([0, st.c2p(0, 0)[1], 0]),
        )
        for lab in labels:
            lab.align_to([xl - 0.3, 0, 0], m.RIGHT)
        return inp, dl, st, labels

    def selection(self):
        self.next_section("Selection", notes="Mamba (Gu & Dao, 2023, 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces'): B, C and the step size delta are functions of the input. Stream: three filler tokens, one key token (0.9), forty filler tokens (seeded uniform noise, illustrative). Fixed model: delta = 0.4 for every token.")
        inp, dl, st, labels = self.stream_plots()
        self.sel = dict(inp=inp, dl=dl, st=st, labels=labels)
        filler = m.VGroup(*[stem(inp, t, XS[t], SKY, width=3, radius=0.04) for t in range(L) if t != KEY])
        key = stem(inp, KEY, XS[KEY], GOOD, width=5, radius=0.07)
        key_label = text("key", 22, GOOD, weight=m.BOLD).next_to(inp.c2p(KEY, XS[KEY]), m.RIGHT, buff=0.15)
        fact = m.DashedLine(st.c2p(0, XS[KEY]), st.c2p(L, XS[KEY]), dash_length=0.1).set_stroke(GOOD, width=2.5)
        fact_label = text("the key's value", 20, GOOD).next_to(st.c2p(22, XS[KEY]), m.UP, buff=0.1)
        fixed_line = polyline([st.c2p(t, H_FIX[t]) for t in range(L)], MUTED, width=4)
        fixed_label = text("fixed", 22, MUTED, weight=m.BOLD).next_to(st.c2p(L, H_FIX[-1]), m.RIGHT, buff=0.2)
        sel = formula(("h_t", PINK), ("=", INK), (r"\bar{A}_t", AMBER), ("h_{t-1}", PINK), ("+", INK), (r"\bar{B}_t", TEAL), ("x_t", SKY), size=44).move_to([-2.6, 1.95, 0])
        marks = m.VGroup(*[m.Line(sel[i].get_corner(m.DL) + 0.1 * m.DOWN, sel[i].get_corner(m.DR) + 0.1 * m.DOWN).set_stroke(c, width=4) for i, c in ((2, AMBER), (5, TEAL))])
        mamba_cite = cite("Mamba · Gu & Dao, 2023").next_to(sel, m.RIGHT, buff=0.5)
        chips = m.VGroup(
            chip("B", TEAL, TEAL_FILL, width=0.8, height=0.62, size=26),
            chip("C", GOOD, GOOD_FILL, width=0.8, height=0.62, size=26),
            chip("Δ", AMBER, AMBER_FILL, width=0.8, height=0.62, size=26),
        ).arrange(m.RIGHT, buff=0.22)
        from_token = text("computed from each token", 22, MUTED)
        chooses = text("what to write, what to keep", 22, INK, weight=m.SEMIBOLD)
        panel = m.VGroup(from_token, chips, chooses).arrange(m.DOWN, buff=0.25).move_to([4.95, 0.35, 0])
        self.sel.update(sel=sel, marks=marks, panel=panel, fixed_label=fixed_label, cite=mamba_cite)

        self.say(
            "[Back to our stream.] [Say, for example, one early token really matters], [and forty filler tokens follow.]",
            m.AnimationGroup(self.new_header("MAMBA", "Selective state spaces"), m.Create(inp), m.FadeIn(labels[0])),
            m.AnimationGroup(m.FadeIn(key, shift=0.2 * m.UP), m.FadeIn(key_label)),
            m.LaggedStart(*[m.FadeIn(s) for s in filler], lag_ratio=0.04),
        )
        self.say(
            "[With fixed parameters, every token is written and faded the same way], [so the fact is soon washed out.]",
            m.AnimationGroup(m.Create(st), m.FadeIn(labels[2]), m.Create(fact), m.FadeIn(fact_label)),
            m.AnimationGroup(m.Create(fixed_line, run_time=2.5), m.FadeIn(fixed_label)),
        )
        self.say(
            "[Mamba, from twenty twenty-three, makes the update selective]: [the carry-over and the write now change with every token.]",
            m.AnimationGroup(m.Write(sel, run_time=1.6), m.FadeIn(mamba_cite)),
            m.Create(marks),
        )
        self.say(
            "[B, C and a step size, delta, are computed from each token], [so the model chooses what to write, and what to keep.]",
            m.AnimationGroup(m.FadeIn(from_token), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.25)),
            m.FadeIn(chooses, shift=0.15 * m.UP),
        )
        self.wait(0.3)

    def step_size(self):
        self.next_section("The step size", notes="Mamba discretises with the step size: A-bar = exp(delta A), A negative. Here A = -1, B = 1 (zero-order hold): carry exp(-delta), write 1 - exp(-delta). Delta = softplus(10 * key - 6): 4.0 for the key, 0.0025 for filler. After forty filler tokens the key's share of the state: fixed 4e-8, selective 89%.")
        s = self.sel
        inp, dl, st, labels = s["inp"], s["dl"], s["st"], s["labels"]
        disc = formula((r"\bar{A}_t", AMBER), ("=", INK), (r"e^{\Delta_t A}", AMBER), size=44).move_to([4.6, 1.95, 0])
        sign = text("A < 0", 22, MUTED).next_to(disc, m.DOWN, buff=0.22)
        large = pill("large Δ: Ā ≈ 0 · write", AMBER, AMBER_FILL, 22)
        tiny = pill("tiny Δ: Ā ≈ 1 · keep", AMBER, AMBER_FILL, 22)
        rules = m.VGroup(large, tiny).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([4.95, -0.2, 0])
        key_bar = bar(dl, KEY, DELTA_SEL[KEY], AMBER, half=0.35)
        filler_bars = m.VGroup(*[bar(dl, t, DELTA_SEL[t], AMBER, half=0.35, least=0.025) for t in range(L) if t != KEY])
        sel_line = polyline([st.c2p(t, H_SEL[t]) for t in range(L)], PINK, width=5)
        sel_label = text("selective", 22, PINK, weight=m.BOLD).next_to(st.c2p(L, H_SEL[-1]), m.RIGHT, buff=0.2)
        fixed_keep = text("keeps ≈ 0%", 22, MUTED).next_to(s["fixed_label"], m.RIGHT, buff=0.2)
        sel_keep = text(f"keeps {100 * KEEP_SEL:.0f}%", 22, PINK).next_to(sel_label, m.RIGHT, buff=0.2)
        on_baseline(m.VGroup(s["fixed_label"], fixed_keep))
        on_baseline(m.VGroup(sel_label, sel_keep))

        self.say(
            "[Delta sets the carry-over]: [the exponential of delta times A, where A is negative.]",
            m.AnimationGroup(self.new_header("MAMBA · STEP SIZE", "Delta decides: keep or write"), m.FadeOut(s["panel"]), m.FadeOut(s["cite"])),
            m.AnimationGroup(m.Write(disc, run_time=1.4), m.FadeIn(sign)),
        )
        self.say(
            "[A large delta wipes the old state and writes the token in.] [A tiny delta skips the token, and keeps the state.]",
            m.FadeIn(large, shift=0.15 * m.UP),
            m.FadeIn(tiny, shift=0.15 * m.UP),
        )
        self.say(
            "[In our stream, the key token gets a large delta], [and the filler almost none.]",
            m.AnimationGroup(m.Create(dl), m.FadeIn(labels[1]), m.GrowFromEdge(key_bar, m.DOWN)),
            m.LaggedStart(*[m.FadeIn(b) for b in filler_bars], lag_ratio=0.03),
        )
        self.say(
            "[Forty tokens later, the fixed model keeps almost nothing of the fact]; [the selective one keeps almost ninety percent.]",
            m.FadeIn(fixed_keep, shift=0.15 * m.LEFT),
            m.AnimationGroup(m.Create(sel_line, run_time=2.2), m.FadeIn(sel_label), m.FadeIn(sel_keep)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ hardware

    def hardware(self):
        self.next_section("Hardware-aware scan", notes="Input-dependent parameters rule out a fixed convolution kernel; Mamba computes the scan in one fused GPU kernel: parameters are loaded from HBM into SRAM, discretised and scanned there, outputs written back; the expanded state is never written to HBM, and intermediate states are recomputed in the backward pass (Gu & Dao, 2023).")
        left = card(4.1, 4.6, EDGE, CARD).move_to([-4.45, -0.55, 0])
        l_title = text("one fixed kernel?", 24, INK, weight=m.BOLD).move_to(left.get_top() + 0.42 * m.DOWN)
        kx = axes([0, 16.5, 4], [0, 1.1, 0.5], 2.9, 2.0).move_to([-4.45, -0.45, 0])
        k_bars = m.VGroup(*[bar(kx, k + 0.5, KERNEL[k], GOOD, half=0.36, opacity=0.6) for k in range(16)])
        cross = m.VGroup(
            m.Line(kx.get_corner(m.DL) + [-0.1, -0.1, 0], kx.get_corner(m.UR) + [0.1, 0.1, 0]),
            m.Line(kx.get_corner(m.UL) + [-0.1, 0.1, 0], kx.get_corner(m.DR) + [0.1, -0.1, 0]),
        ).set_stroke(BAD, width=7)
        cross.set_z_index(2)
        no_kernel = text("changes with every token", 22, BAD, weight=m.SEMIBOLD).move_to(left.get_bottom() + 0.5 * m.UP)

        sram = card(8.0, 2.0, PINK, CARD).move_to([2.55, 1.1, 0])
        sram_title = text("on-chip SRAM · small, fast", 24, PINK, weight=m.BOLD).move_to(sram.get_corner(m.UL) + [0.25, -0.36, 0], aligned_edge=m.LEFT)
        grid = m.VGroup(*[m.Square(0.28).set_stroke(PINK, width=1.5).set_fill(mix(PINK_FILL, PINK, 0.15 + 0.6 * ((3 * r + 5 * c) % 7) / 7), opacity=1) for r in range(3) for c in range(11)])
        grid.arrange_in_grid(rows=3, buff=0.06).move_to([0.95, 0.92, 0])
        grid_label = text("expanded state", 22, INK, weight=m.SEMIBOLD).next_to(grid, m.RIGHT, buff=0.45).shift(0.17 * m.UP)
        grid_note = text("never leaves the chip", 20, MUTED).next_to(grid_label, m.DOWN, buff=0.1).align_to(grid_label, m.LEFT)
        sweep = arrow(grid.get_corner(m.DL) + [0, -0.14, 0], grid.get_corner(m.DR) + [0, -0.14, 0], PINK, buff=0)

        hbm = card(8.0, 1.5, EDGE, CARD).move_to([2.55, -2.45, 0])
        hbm_title = text("GPU main memory (HBM) · large, slow", 22, MUTED, weight=m.SEMIBOLD).move_to(hbm.get_corner(m.UL) + [0.25, -0.3, 0], aligned_edge=m.LEFT)
        ins = m.VGroup(*[chip(s, c, f, width=0.72, height=0.56, size=24) for s, c, f in (("x", SKY, SKY_FILL), ("Δ", AMBER, AMBER_FILL), ("B", TEAL, TEAL_FILL), ("C", GOOD, GOOD_FILL))]).arrange(m.RIGHT, buff=0.18)
        ins.move_to([0.6, -2.72, 0])
        outs = chip("y", GOOD, GOOD_FILL, width=0.72, height=0.56, size=24).move_to([5.3, -2.72, 0])
        up = arrow([0.6, hbm.get_top()[1] + 0.05, 0], [0.6, sram.get_bottom()[1] - 0.05, 0], SKY, buff=0)
        down = arrow([5.3, sram.get_bottom()[1] - 0.05, 0], [5.3, hbm.get_top()[1] + 0.05, 0], GOOD, buff=0)
        up_label = text("load", 22, MUTED).next_to(up, m.LEFT, buff=0.15)
        down_label = text("write", 22, MUTED).next_to(down, m.RIGHT, buff=0.15)
        lesson = pill("backward pass: recompute states", PINK, PINK_FILL, 20).move_to([2.95, -0.8, 0])
        lesson.set_z_index(3)
        hw_cite = cite("Gu & Dao, 2023").move_to([-4.45, -3.3, 0])

        self.say(
            "[With parameters that change per token, there's no single kernel], [so no convolution.]",
            m.AnimationGroup(self.new_header("MAMBA · SPEED", "A hardware-aware scan"), m.FadeIn(left), m.FadeIn(l_title), m.Create(kx), m.FadeIn(k_bars)),
            m.AnimationGroup(m.Create(cross), m.FadeIn(no_kernel, shift=0.15 * m.UP)),
        )
        self.say(
            "[Mamba runs the scan instead, fused into one GPU kernel], [keeping the large expanded state in fast on-chip memory.]",
            m.AnimationGroup(m.FadeIn(sram), m.FadeIn(sram_title), m.FadeIn(hbm), m.FadeIn(hbm_title), m.FadeIn(hw_cite)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in grid], lag_ratio=0.015), m.FadeIn(grid_label), m.FadeIn(grid_note), m.GrowArrow(sweep)),
        )
        self.say(
            "[Only inputs and outputs travel to main memory], [and the backward pass recomputes states instead of storing them]: [the FlashAttention lesson again.]",
            m.AnimationGroup(m.FadeIn(ins), m.FadeIn(outs), m.GrowArrow(up), m.GrowArrow(down), m.FadeIn(up_label), m.FadeIn(down_label)),
            m.FadeIn(lesson, shift=0.15 * m.UP),
            m.AnimationGroup(pulse(up, SKY), pulse(down, GOOD)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section("The catch", notes="Twelve seeded random numbers pushed through the three-number state (A from the recurrence section, B = 1); the best linear readout (pseudo-inverse) recovers 19% of their energy. Copying: Jelassi et al., 2024, 'Repeat After Me: Transformers are Better than State Space Models at Copying'. Hybrids: Jamba (Lieber et al., 2024) uses one attention layer per seven Mamba layers. Facts checked October 2026.")
        left = card(6.3, 4.75, EDGE, CARD).move_to([-3.4, -0.45, 0])
        l_title = text("Copy back twelve numbers", 26, INK, weight=m.BOLD).move_to(left.get_top() + 0.42 * m.DOWN)
        ax = axes([0, 12.5, 4], [-2.0, 2.0, 1], 5.3, 2.5).move_to([-3.4, -0.6, 0])
        truth = m.VGroup(*[bar(ax, i + 0.5, XC[i], SKY, half=0.32, opacity=0.85) if XC[i] >= 0 else bar_down(ax, i + 0.5, XC[i], SKY) for i in range(12)])
        guess = m.VGroup(*[m.Dot(ax.c2p(i + 0.5, XR[i]), radius=0.09, color=PINK) for i in range(12)])
        guess_line = polyline([ax.c2p(i + 0.5, XR[i]) for i in range(12)], PINK, width=3)
        guess.set_z_index(3)
        legend = m.VGroup(
            m.VGroup(m.Square(0.22).set_stroke(width=0).set_fill(SKY, opacity=0.85), text("the inputs", 22, MUTED)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Dot(radius=0.09, color=PINK), text("best readout of the state", 22, MUTED)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.45).move_to([-3.4, 1.0, 0])
        score = pill(f"recovers {100 * FRACTION:.0f}% of the signal", PINK, PINK_FILL, 22).move_to([-3.4, -2.3, 0])
        copy_cite = cite("Jelassi et al., 2024").move_to([-3.4, -3.25, 0])

        result = pill("Attention wins at copying and recall", VIOLET, VIOLET_FILL, 22).move_to([3.4, 1.6, 0])
        stack = m.VGroup()
        for k in range(8):
            is_attn = k == 4
            color, fill = (VIOLET, VIOLET_FILL) if is_attn else (PINK, PINK_FILL)
            layer = card(3.4, 0.34, color, fill, radius=0.1, stroke=2)
            label = text("attention" if is_attn else "Mamba", 20, color, weight=m.SEMIBOLD).move_to(layer)
            stack.add(m.VGroup(layer, label))
        stack.arrange(m.UP, buff=0.07).move_to([2.45, -1.0, 0])
        brace = m.Brace(stack, m.RIGHT, buff=0.15).set_color(MUTED)
        ratio = m.VGroup(text("1 attention", 22, VIOLET, weight=m.BOLD), text("7 Mamba", 22, PINK, weight=m.BOLD)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        ratio.next_to(brace, m.RIGHT, buff=0.2)
        stack_title = text("a hybrid block", 22, MUTED, weight=m.SEMIBOLD).next_to(stack, m.UP, buff=0.15)
        jamba = cite("Jamba · Lieber et al., 2024").next_to(stack, m.DOWN, buff=0.22)

        self.say(
            "[Now the honest catch.] [Ask the model to copy back a long string, exactly.]",
            self.new_header("THE CATCH", "A fixed-size memory"),
            m.AnimationGroup(m.FadeIn(left), m.FadeIn(l_title), m.Create(ax), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in truth], lag_ratio=0.08), m.FadeIn(legend[0])),
        )
        self.say(
            "[Our three-number state, for example, cannot hold twelve numbers]: [even the best readout recovers only about a fifth of the signal.]",
            m.AnimationGroup(m.FadeIn(legend[1]), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in guess], lag_ratio=0.08), m.Create(guess_line)),
            m.AnimationGroup(m.FadeIn(score, shift=0.15 * m.UP), m.FadeIn(copy_cite)),
        )
        self.say(
            "[Attention can just look back], [so Transformers beat state space models at copying and exact recall, in theory and in experiments.]",
            m.FadeIn(result, shift=0.15 * m.UP),
            m.Circumscribe(result, color=VIOLET, buff=0.08),
        )
        self.say(
            "[So many recent models are hybrids]: [mostly Mamba layers, with a few attention layers mixed in.]",
            m.FadeIn(stack_title),
            m.LaggedStart(*[m.FadeIn(layer, shift=0.15 * m.UP) for layer in stack], lag_ratio=0.15),
        )
        self.say(
            "[Jamba, for example, uses one attention layer for every seven Mamba layers.]",
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(ratio, shift=0.15 * m.LEFT), m.FadeIn(jamba)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Recur, scan, select.")
        halo = glow(PINK, radius=4.0)
        words = on_baseline(m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Recur", "Scan", "Select"), (PINK, TEAL, AMBER))]).arrange(m.RIGHT, buff=1.3))
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A small state that chooses what to keep can read an endless stream.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [recur], [scan], [select.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A small state that chooses what to keep can read an endless stream.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)


def bar_down(ax, t, v, color, half=0.32, opacity=0.85):
    """A bar hanging below the axis at t (v < 0)."""
    left, base = ax.c2p(t - half, 0), ax.c2p(t, 0)
    height = max(base[1] - ax.c2p(t, v)[1], 0.012)
    rect = m.Rectangle(width=2 * (base[0] - left[0]), height=height, stroke_width=0).set_fill(color, opacity=opacity)
    return rect.move_to([base[0], base[1], 0], aligned_edge=m.UP)
