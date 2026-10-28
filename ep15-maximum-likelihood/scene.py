"""AI, Explained · Episode 15 · Maximum Likelihood

Running example: a coin of unknown bias theta, flipped ten times: seven heads, three tails
(illustrative: the sequence comes from a seeded generator). The likelihood
L(theta) = theta^7 (1 - theta)^3 is computed and plotted; its peak is at theta = 0.7. The log
turns the product into a sum and avoids underflow: ten thousand seeded flips multiply to 0.0
in floating point, while their log-likelihood is an ordinary number (about -6,109). Then the
reveal: for a line through noisy points (seeded, Gaussian noise, sigma = 0.5), maximising the
Gaussian log-likelihood is minimising squared error (Episode 2); yes/no labels are coin flips,
and give cross-entropy (Episode 16). Every number the narration quotes is asserted below.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the data

BLUE_FILL = "#14213D"  # heads (BLUE) tint; tails use PINK / PINK_FILL


def _ten_flips():
    """Ten flips of a 70% coin from a seeded generator: the first with seven heads that opens
    with a head and doesn't run three heads in a row at the start (so it reads as random)."""
    for seed in range(1000):
        flips = np.random.default_rng(seed).random(10) < 0.7
        if flips.sum() == 7 and flips[0] and not flips[1:3].all():
            return flips
    raise RuntimeError("no sequence found")


FLIPS = _ten_flips()
HEADS, TAILS = int(FLIPS.sum()), int((~FLIPS).sum())
assert (HEADS, TAILS) == (7, 3)  # "seven heads and three tails"


def lik(t):
    return t**HEADS * (1 - t) ** TAILS


def loglik(t):
    return HEADS * np.log(t) + TAILS * np.log(1 - t)


_GRID = np.linspace(0, 1, 100001)
PEAK = float(_GRID[np.argmax(lik(_GRID))])
assert abs(PEAK - 0.7) < 1e-4 and abs(PEAK - HEADS / 10) < 1e-4  # "peaks at exactly seventy percent: seven out of ten"
LOG_PEAK = float(_GRID[1:-1][np.argmax(loglik(_GRID[1:-1]))])
assert abs(LOG_PEAK - PEAK) < 1e-4  # "the peak stays in exactly the same place"

L_FAIR, L_HEAVY, L_BEST = float(lik(0.5)), float(lik(0.9)), float(lik(0.7))
assert 0.0009 < L_FAIR < 0.0011  # fair coin: "about one in a thousand"
assert 0.45 < L_HEAVY / L_FAIR < 0.55  # ninety percent: "about half of that"
assert 2 < L_BEST / L_FAIR < 2.5  # seventy percent: "more than twice the fair coin's"
assert 0.0019 < L_BEST < 0.0025  # "about two in a thousand"

# ten thousand flips of the same kind of coin (seeded), scored at theta = 0.7
_BIG = np.random.default_rng(0).random(10_000) < 0.7
BIG_P = np.where(_BIG, 0.7, 0.3)
BIG_PRODUCT = float(np.prod(BIG_P))
BIG_LOG10 = float(np.log10(BIG_P).sum())
BIG_LOG = float(np.log(BIG_P).sum())
BIG_EXP = int(np.floor(BIG_LOG10))
BIG_MANT = 10 ** (BIG_LOG10 - BIG_EXP)
BIG_ZEROS = -BIG_EXP - 1  # zeros between the decimal point and the first digit
assert BIG_PRODUCT == 0.0  # "a computer just rounds it to zero"
assert 2600 < BIG_ZEROS < 2700  # "over two thousand six hundred zeros after the decimal point"
assert 5900 < -BIG_LOG < 6300  # "about six thousand below zero"

# a line through noisy points: y = 0.75 x + 1 plus Gaussian noise (sd 0.5), seeded
LINE_B, TRUE_W, SIGMA = 1.0, 0.75, 0.5
XS = np.arange(1, 7, dtype=float)
YS = TRUE_W * XS + LINE_B + SIGMA * np.random.default_rng(4).normal(0, 1, len(XS))
W_BEST = float(np.dot(XS, YS - LINE_B) / np.dot(XS, XS))  # least squares with b fixed


def sse(w):
    return float(np.sum((YS - (w * XS + LINE_B)) ** 2))


def gauss_ll(w):
    r = YS - (w * XS + LINE_B)
    return float(np.sum(-(r**2) / (2 * SIGMA**2) - np.log(SIGMA * np.sqrt(2 * np.pi))))


_WS = np.linspace(W_BEST - 0.5, W_BEST + 0.5, 2001)
assert np.argmin([sse(w) for w in _WS]) == np.argmax([gauss_ll(w) for w in _WS])  # "the same line"
_RES = np.abs(YS - (W_BEST * XS + LINE_B)) / SIGMA
NEAR, FAR = int(np.argmin(_RES)), int(np.argmax(_RES))
assert _RES[NEAR] < 0.2 and _RES[FAR] > 1.5  # one point near the line, one out in the tail


# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def flip_chip(head, size=0.85, font=32):
    return chip("H" if head else "T", BLUE if head else PINK, BLUE_FILL if head else PINK_FILL, width=size, height=size, size=font)


def coin(radius=0.95, mark="?"):
    disc = m.Circle(radius=radius).set_stroke(AMBER, width=5).set_fill(AMBER_FILL, opacity=1)
    rim = m.Circle(radius=0.78 * radius).set_stroke(AMBER, width=2.5, opacity=0.6)
    sign = text(mark, int(76 * radius), AMBER, weight=m.BOLD).move_to(disc)
    return m.VGroup(disc, rim, sign)


def flip(c, run_time=0.32):
    """The coin turning over once (squash and back)."""
    return c.animate(rate_func=m.there_and_back, run_time=run_time).stretch(0.06, 1)


def mtex(*parts, size=40, color=INK):
    return m.MathTex(*parts, font_size=size, color=color)


def legend(symbol, words, color, size=34, word_size=24):
    return m.VGroup(mtex(symbol, size=size, color=color), text(words, word_size, MUTED)).arrange(m.RIGHT, buff=0.25)


def number(value, places, color, size=34):
    return m.DecimalNumber(value, num_decimal_places=places, mob_class=Mono, font_size=size, color=color)


def morph(src, dst, pairs, new=(), gone=(), **kw):
    """One formula becoming another, part by part: `pairs` of (source part, target part)
    indices move, `new` target parts fade in, `gone` source parts fade out."""
    return m.AnimationGroup(
        *[m.ReplacementTransform(src[i], dst[j]) for i, j in pairs],
        *[m.FadeIn(dst[j]) for j in new],
        *[m.FadeOut(src[i]) for i in gone],
        **kw,
    )


def mini_bell(color=VIOLET, width=1.0, height=0.55):
    xs = np.linspace(-1, 1, 60)
    pts = [[x * width / 2, height * np.exp(-((x * 2.4) ** 2) / 2) - height / 2, 0] for x in xs]
    curve = m.VMobject().set_points_as_corners(pts).set_stroke(color, width=4)
    base = m.Line([-width / 2, -height / 2, 0], [width / 2, -height / 2, 0]).set_stroke(MUTED, width=2.5)
    return m.VGroup(base, curve)


# ---------------------------------------------------------------------------- the episode


class Ep15MaximumLikelihood(Episode):
    number = 15

    def lesson(self):
        self.hook()
        self.experiment()
        self.likelihood()
        self.peak()
        self.log_trick()
        self.recipe()
        self.reveal()
        self.yes_no()
        self.big_picture()
        self.catch()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    def split(self, formula):
        """Put a formula's parts on stage one by one, so morph() can replace each."""
        self.remove(formula)
        self.add(*formula)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Why is the loss in Episode 2 squared? Maximum likelihood answers it.")
        tag = text("Episode 2 · the squared error", 24, MUTED, weight=m.SEMIBOLD).move_to([0, 2.3, 0])
        mse = mtex(r"L = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)", "^2", size=66).move_to([0, 0.95, 0])
        why = text("Why square it?", 48, AMBER, weight=m.BOLD).move_to([0, -0.65, 0])
        options = m.VGroup(
            pill("the gap", MUTED, CARD, 24),
            pill("the gap squared", AMBER, AMBER_FILL, 24),
            pill("the gap to the fourth power", MUTED, CARD, 24),
        ).arrange(m.RIGHT, buff=0.45).move_to([0, -1.95, 0])
        first = m.VGroup(tag, mse, why, options)

        halo = glow(AMBER, radius=3.6).move_to(0.4 * m.UP)
        title = text("Maximum likelihood", 84, INK, weight=m.BOLD).move_to([0, 0.45, 0])
        sub = text("where loss functions come from", 34, MUTED).next_to(title, m.DOWN, buff=0.5)

        self.say(
            "So... [in episode two, we measured a model's mistakes with the squared error.]",
            m.AnimationGroup(m.FadeIn(tag, shift=0.2 * m.DOWN), m.Write(mse), run_time=1.6),
        )
        self.say(
            "[But why do we square the gap?] [Why not the plain gap, or the gap to the fourth power?]",
            m.AnimationGroup(m.Indicate(mse[1], color=AMBER, scale_factor=1.6), m.FadeIn(why, shift=0.2 * m.UP)),
            m.LaggedStart(*[m.FadeIn(o, shift=0.2 * m.UP) for o in options], lag_ratio=0.3),
        )
        self.say(
            "[There's a deep answer, called maximum likelihood]: [it's where loss functions come from.]",
            m.Succession(
                m.FadeOut(first, run_time=0.5),
                m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(title, shift=0.3 * m.UP, scale=0.9), run_time=1.0),
            ),
            m.Write(sub),
        )
        self.wait(0.4)
        self.play(m.FadeOut(halo), m.FadeOut(title), m.FadeOut(sub), run_time=0.8)

    # ------------------------------------------------------------------ the experiment

    def experiment(self):
        self.next_section(
            "The experiment",
            notes="Illustrative data: ten flips from a seeded generator (a 70% coin), seven heads and three tails. The model: P(heads) = theta.",
        )
        pitch = 1.03
        self.chips = m.VGroup(*[flip_chip(h).move_to([(i - 4.5) * pitch, 1.55, 0]) for i, h in enumerate(FLIPS)])
        the_coin = coin().move_to([-4.6, -1.0, 0])
        tally = m.VGroup(pill(f"{HEADS} heads", BLUE, BLUE_FILL, 24), pill(f"{TAILS} tails", PINK, PINK_FILL, 24))
        tally.arrange(m.RIGHT, buff=0.4).move_to([0, 0.5, 0])

        model = card(7.4, 2.5, EDGE, CARD).move_to([2.35, -1.45, 0])
        rows = m.VGroup(mtex(r"P(\text{heads})", "=", r"\theta", size=44), mtex(r"P(\text{tails})", "=", r"1-\theta", size=44))
        rows.arrange(m.RIGHT, buff=0.8)
        rows[0][0].set_color(BLUE)
        rows[1][0].set_color(PINK)
        rows[0][2].set_color(SKY)
        rows[1][2].set_color(SKY)
        rows.move_to(model.get_center() + 0.42 * m.UP)
        knob = legend(r"\theta", "“theta”: the coin's bias, our one knob", SKY, 36, 24).move_to(model.get_center() + 0.6 * m.DOWN)

        question = mtex(r"\theta = \;?", size=48, color=SKY).move_to([-4.6, -2.55, 0])
        gut = text("gut feeling: 7 out of 10", 24, MUTED).move_to([-4.6, -3.25, 0])
        flips = m.Succession(*[flip(the_coin) for _ in range(8)])

        self.say(
            "[Let's start with the simplest experiment there is]: [we find a strange coin, and flip it ten times.]",
            m.AnimationGroup(self.new_header("THE EXPERIMENT", "Ten flips of a mystery coin"), m.FadeIn(the_coin, scale=0.7)),
            m.AnimationGroup(
                flips,
                m.LaggedStart(*[m.FadeIn(c, target_position=the_coin.get_center(), scale=0.4) for c in self.chips], lag_ratio=0.22),
                run_time=2.6,
            ),
        )
        self.say(
            "[We get, for example, seven heads and three tails.]",
            m.LaggedStart(m.FadeIn(tally[0], shift=0.2 * m.UP), m.FadeIn(tally[1], shift=0.2 * m.UP), lag_ratio=0.4),
        )
        self.say(
            "[Our model of the coin has one knob]: [theta, the chance that it lands heads.]",
            m.AnimationGroup(m.FadeIn(model), m.FadeIn(rows, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(knob, shift=0.2 * m.UP), m.Indicate(rows[0][2], color=SKY, scale_factor=1.4)),
        )
        self.say(
            "[What's the best value for theta?] [Your gut probably says seven out of ten.] [Let's see why the math agrees.]",
            m.FadeIn(question, shift=0.2 * m.UP),
            m.FadeIn(gut, shift=0.2 * m.UP),
            m.Circumscribe(model, color=AMBER, buff=0.06),
        )
        self.wait(0.3)
        self.play(*[m.FadeOut(mob) for mob in (the_coin, tally, model, rows, knob, question, gut)], run_time=0.7)

    # ------------------------------------------------------------------ the likelihood

    def likelihood(self):
        self.next_section("Likelihood", notes="L(theta) = theta^7 (1 - theta)^3: 0.00098 at 0.5, 0.00048 at 0.9, 0.00222 at 0.7 (computed).")
        chips = self.chips
        factors = m.VGroup(
            *[
                mtex(r"\theta" if h else r"1-\theta", size=34, color=BLUE if h else PINK).move_to([c.get_x(), 0.72, 0])
                for h, c in zip(FLIPS, chips)
            ]
        )
        heads = m.VGroup(*[f for f, h in zip(factors, FLIPS) if h])
        tails = m.VGroup(*[f for f, h in zip(factors, FLIPS) if not h])
        formula = mtex(r"L(\theta)", "=", r"\theta^{7}", r"(1-\theta)^{3}", size=62).move_to([-3.4, -0.6, 0])
        formula[0].set_color(VIOLET)
        formula[2].set_color(BLUE)
        formula[3].set_color(PINK)
        caption = text("independent flips: multiply", 24, MUTED).move_to([-3.4, -1.6, 0])
        name = pill("the likelihood", VIOLET, VIOLET_FILL, 26).move_to([-3.4, -2.55, 0])

        theta = m.ValueTracker(0.3)
        panel = card(5.6, 3.5, EDGE, CARD).move_to([3.5, -1.35, 0])
        x0, x1, rail_y = 1.25, 5.75, -0.8

        def at(t):
            return x0 + (x1 - x0) * t

        rail = m.Line([x0, rail_y, 0], [x1, rail_y, 0]).set_stroke(FAINT, width=4)
        ticks = m.VGroup(*[m.Line([at(v), rail_y - 0.1, 0], [at(v), rail_y + 0.1, 0]).set_stroke(FAINT, width=3) for v in (0, 0.5, 1)])
        tick_labels = m.VGroup(*[text(s, 22, MUTED).move_to([at(v), rail_y - 0.38, 0]) for s, v in (("0", 0), ("0.5", 0.5), ("1", 1))])
        marker = m.Triangle().set_stroke(width=0).set_fill(SKY, opacity=1).rotate(m.PI)
        marker.scale(0.16).move_to([at(0.3), rail_y + 0.24, 0])
        marker.add_updater(lambda mob: mob.set_x(at(theta.get_value())))
        t_num = number(0.3, 2, SKY, 36)
        t_num.add_updater(lambda d: d.set_value(theta.get_value()))
        t_row = m.VGroup(mtex(r"\theta =", size=40, color=SKY), t_num).arrange(m.RIGHT, buff=0.2).move_to([3.5, -0.05, 0])
        l_num = number(lik(0.3), 5, VIOLET, 36)
        l_num.add_updater(lambda d: d.set_value(lik(theta.get_value())))
        l_row = m.VGroup(mtex(r"L(\theta) =", size=40, color=VIOLET), l_num).arrange(m.RIGHT, buff=0.2).move_to([3.5, -1.85, 0])
        track = m.Rectangle(width=x1 - x0, height=0.3).set_stroke(width=0).set_fill(EDGE, opacity=1).move_to([3.5, -2.6, 0])

        def make_bar():
            w = max(0.02, (x1 - x0) * lik(theta.get_value()) / L_BEST)
            return m.Rectangle(width=w, height=0.3).set_stroke(width=0).set_fill(VIOLET, opacity=1).move_to(track).align_to(track, m.LEFT)

        bar = m.always_redraw(make_bar)
        slider = m.VGroup(rail, ticks, tick_labels)

        self.say(
            "[Pick any value of theta, and ask]: [how likely were these exact ten flips?]",
            m.AnimationGroup(
                self.new_header("LIKELIHOOD", "How likely is our data?"),
                m.FadeIn(panel),
                m.FadeIn(t_row),
                m.FadeIn(slider),
                m.FadeIn(marker),
            ),
            m.LaggedStart(*[m.Indicate(c, color=INK, scale_factor=1.15) for c in chips], lag_ratio=0.1),
        )
        self.say(
            "[Each head has probability theta, and each tail, one minus theta.] [The flips are independent, so we multiply.]",
            m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.DOWN) for f in factors], lag_ratio=0.12),
            m.AnimationGroup(
                m.FadeTransform(heads.copy(), formula[2]),
                m.FadeTransform(tails.copy(), formula[3]),
                m.FadeIn(formula[0:2]),
                m.FadeIn(caption, shift=0.15 * m.UP),
                run_time=1.4,
            ),
        )
        self.say(
            "[That product is called the likelihood]: [theta to the seventh, times one minus theta, cubed.]",
            m.AnimationGroup(m.FadeIn(name, scale=1.2), m.FadeIn(l_row), m.FadeIn(track), m.FadeIn(bar)),
            m.Succession(m.Indicate(formula[2], color=BLUE, scale_factor=1.2), m.Indicate(formula[3], color=PINK, scale_factor=1.2)),
        )
        self.say(
            "[For a fair coin, theta is fifty percent], [and the likelihood is about one in a thousand.]",
            theta.animate(run_time=1.2).set_value(0.5),
            m.Indicate(l_row, color=VIOLET, scale_factor=1.08),
        )
        self.say(
            "[At ninety percent, it drops to about half of that]: [a coin that loves heads rarely gives three tails.]",
            theta.animate(run_time=1.4).set_value(0.9),
            m.LaggedStart(*[m.Indicate(c, color=PINK, scale_factor=1.2) for c, h in zip(chips, FLIPS) if not h], lag_ratio=0.3),
        )
        self.say(
            "[At seventy percent, it's more than twice the fair coin's]: [about two in a thousand.]",
            theta.animate(run_time=1.4).set_value(0.7),
            m.Circumscribe(l_row, color=VIOLET, buff=0.1),
        )
        self.wait(0.3)
        self.stop(marker, t_num, l_num, bar)
        self.clear()

    # ------------------------------------------------------------------ the peak

    def peak(self):
        self.next_section(
            "Maximum likelihood",
            notes="L(theta) plotted from 0 to 1 (computed); peak at 0.7 = 7/10. Fisher, 1922 (On the mathematical foundations of theoretical statistics, Phil. Trans. R. Soc. A 222; introduced the term maximum likelihood estimate; see Aldrich 1997, Statistical Science 12(3); checked 2026-10-21). Likelihood is P(data | theta), not P(theta | data).",
        )
        ax = axes([0, 1, 0.1], [0, 0.0026, 0.0005], 7.4, 4.3).move_to([-2.0, -0.45, 0])
        x_ticks = m.VGroup(*[text(s, 22, MUTED).next_to(ax.c2p(v, 0), m.DOWN, buff=0.22) for s, v in (("0", 0), ("0.5", 0.5), ("1", 1))])
        y_ticks = m.VGroup(*[text(s, 20, MUTED).next_to(ax.c2p(0, v), m.LEFT, buff=0.18) for s, v in (("0.001", 0.001), ("0.002", 0.002))])
        x_name = mtex(r"\theta", size=36, color=SKY).next_to(ax.x_axis.get_end(), m.RIGHT, buff=0.15)
        y_name = text("likelihood", 24, VIOLET, weight=m.SEMIBOLD).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.2)
        frame = m.VGroup(ax, x_ticks, y_ticks, x_name, y_name)

        t = m.ValueTracker(0.02)
        partial = m.always_redraw(lambda: ax.plot(lik, x_range=[0, max(0.021, t.get_value())]).set_stroke(VIOLET, width=5))
        dot = m.always_redraw(lambda: m.Dot(ax.c2p(t.get_value(), lik(t.get_value())), radius=0.1, color=SKY).set_z_index(3))
        t_num = number(0.02, 2, SKY, 34)
        t_num.add_updater(lambda d: d.set_value(t.get_value()))
        l_num = number(lik(0.02), 5, VIOLET, 34)
        l_num.add_updater(lambda d: d.set_value(lik(t.get_value())))
        reads = m.VGroup(
            m.VGroup(mtex(r"\theta =", size=38, color=SKY), t_num).arrange(m.RIGHT, buff=0.2),
            m.VGroup(mtex(r"L =", size=38, color=VIOLET), l_num).arrange(m.RIGHT, buff=0.2),
        ).arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT).move_to([4.6, 1.0, 0])

        top = ax.c2p(PEAK, L_BEST)
        drop = m.DashedLine(ax.c2p(PEAK, 0), top, dash_length=0.1).set_stroke(GOOD, width=3)
        peak_tick = text("0.7", 22, GOOD, weight=m.BOLD).next_to(ax.c2p(PEAK, 0), m.DOWN, buff=0.22)
        hat = mtex(r"\hat{\theta} = 0.7", size=38, color=GOOD).next_to(top, m.UP, buff=0.3)
        ratio = mtex(r"\hat{\theta} = \frac{7\ \text{heads}}{10\ \text{flips}}", size=40, color=GOOD).move_to([4.6, -0.45, 0])
        mle_box = card(4.1, 1.45, GOOD, GOOD_FILL, stroke=2).move_to([4.6, -1.85, 0])
        mle_words = m.VGroup(text("maximum likelihood", 28, INK, weight=m.BOLD), text("estimation", 28, INK, weight=m.BOLD))
        mle_words.arrange(m.DOWN, buff=0.1).move_to(mle_box)
        fisher = text("Fisher, 1922", 20, MUTED).next_to(mle_box, m.DOWN, buff=0.18)
        note = m.VGroup(
            mtex(r"L(\theta) = P(\text{data} \mid \theta)", size=38, color=VIOLET),
            pill("not", BAD, BAD_FILL, 22),
            mtex(r"P(\theta \text{ is right})", size=38, color=MUTED),
        ).arrange(m.RIGHT, buff=0.4)
        note.move_to([-1.6, -3.5, 0])

        self.say(
            "[Now try every theta from zero to one], [and plot the likelihood.]",
            m.AnimationGroup(self.new_header("MAXIMUM LIKELIHOOD", "Find the top of the curve"), m.Create(ax), m.FadeIn(frame[1:]), m.FadeIn(reads)),
            m.AnimationGroup(m.FadeIn(dot), m.FadeIn(partial), t.animate(run_time=3.2, rate_func=m.linear).set_value(1.0)),
        )
        self.stop(partial)  # the whole curve stays while the dot moves back to the peak
        self.say(
            "[It peaks at exactly seventy percent]: [seven heads out of ten flips.]",
            m.AnimationGroup(t.animate(run_time=1.4).set_value(PEAK), m.Create(drop), m.FadeIn(peak_tick), m.FadeIn(hat, shift=0.2 * m.DOWN)),
            m.FadeIn(ratio, shift=0.2 * m.UP),
        )
        self.say(
            "[Picking the parameter that makes the data most likely is called maximum likelihood estimation.]",
            m.AnimationGroup(m.FadeIn(mle_box), m.FadeIn(mle_words, shift=0.2 * m.UP), m.FadeIn(fisher)),
        )
        self.say(
            "[One subtlety]: [it's the probability of the data given theta, not the probability that theta is right.]",
            m.Indicate(y_name, color=VIOLET, scale_factor=1.15),
            m.FadeIn(note, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.stop(dot, t_num, l_num)
        self.clear()

    # ------------------------------------------------------------------ the log trick

    def log_trick(self):
        self.next_section(
            "The log trick",
            notes="Ten thousand seeded flips scored at 0.7: the product underflows to 0.0 (true value about 4.8e-2654); the log-likelihood is about -6,109. Log is increasing, so the peak stays at 0.7 (computed).",
        )
        count = pill("10,000 flips", SKY, SKY_FILL, 26).move_to([0, 2.0, 0])
        product = code(" × ".join(f"{p:.1f}" for p in BIG_P[:8]) + " × …", 30, INK).move_to([0, 1.0, 0])
        result = mtex(rf"= {BIG_MANT:.1f} \times 10^{{{BIG_EXP}}}", size=58, color=VIOLET).move_to([0, -0.15, 0])
        zeros = text(f"{BIG_ZEROS:,} zeros after the decimal point", 26, MUTED).move_to([0, -1.05, 0])
        zero = pill("a computer stores it as 0.0", BAD, BAD_FILL, 28).move_to([0, -2.2, 0])
        problem = m.VGroup(count, product, result, zeros, zero)

        top = mtex(r"L(\theta)", "=", r"\theta^{7}", r"(1-\theta)^{3}", size=46).move_to([-3.5, 1.75, 0])
        top[0].set_color(VIOLET)
        top[2].set_color(BLUE)
        top[3].set_color(PINK)
        logged = mtex(r"\log L(\theta)", "=", r"7\log\theta", "+", r"3\log(1-\theta)", size=46).move_to([-3.5, 0.35, 0])
        logged[0].set_color(TEAL)
        logged[2].set_color(BLUE)
        logged[4].set_color(PINK)
        down = arrow(top.get_bottom() + 0.05 * m.DOWN, logged.get_top() + 0.05 * m.UP, MUTED, buff=0.1)
        down_label = text("take the log", 22, MUTED).next_to(down, m.RIGHT, buff=0.2)
        rule = text("the log of a product is the sum of the logs", 24, MUTED).move_to([-3.5, -0.55, 0])
        name = pill("the log-likelihood", TEAL, TEAL_FILL, 26).move_to([-3.5, -1.4, 0])
        big = m.VGroup(text("10,000 flips:", 26, MUTED), mtex(rf"\log L = {BIG_LOG:.1f}", size=40, color=TEAL))
        big.arrange(m.RIGHT, buff=0.25).move_to([-3.5, -2.4, 0])
        easy = pill("an ordinary number", GOOD, GOOD_FILL, 22).move_to([-3.5, -3.25, 0])

        up = axes([0, 1, 0.1], [0, 0.0026, 0.001], 5.2, 1.9).move_to([3.85, 1.2, 0])
        # the log-likelihood is negative: its axes run from -22 (bottom) to 0 (top), drawn
        # shifted by +22 so the horizontal axis sits at the bottom; tick labels give true values
        lo = axes([0, 1, 0.1], [0, 22, 5], 5.2, 2.3).move_to([3.85, -1.55, 0])
        c_up = up.plot(lik, x_range=[0, 1]).set_stroke(VIOLET, width=4)
        c_lo = lo.plot(lambda v: loglik(v) + 22, x_range=[0.05, 0.97]).set_stroke(TEAL, width=4)
        lo_ticks = m.VGroup(*[text(s, 20, MUTED).next_to(lo.c2p(0, v + 22), m.LEFT, buff=0.15) for s, v in (("0", 0), ("−20", -20))])
        up_name = text("likelihood", 22, VIOLET, weight=m.SEMIBOLD).next_to(up.y_axis.get_end(), m.RIGHT, buff=0.2)
        lo_name = text("log-likelihood", 22, TEAL, weight=m.SEMIBOLD).move_to(lo.c2p(0.42, 3))
        same = m.DashedLine(up.c2p(PEAK, L_BEST), lo.c2p(PEAK, 0), dash_length=0.1).set_stroke(GOOD, width=3)
        dots = m.VGroup(m.Dot(up.c2p(PEAK, L_BEST), radius=0.09, color=GOOD), m.Dot(lo.c2p(PEAK, loglik(PEAK) + 22), radius=0.09, color=GOOD))
        dots.set_z_index(3)
        same_label = text("same peak: θ = 0.7", 24, GOOD, weight=m.BOLD).next_to(lo, m.DOWN, buff=0.3)

        self.say(
            "[But there's a practical problem.] [Real data is big: say, ten thousand flips, and ten thousand numbers to multiply.]",
            m.AnimationGroup(self.new_header("THE LOG TRICK", "Products become sums"), m.FadeIn(count, shift=0.2 * m.DOWN)),
            m.AddTextLetterByLetter(product, time_per_char=0.03),
        )
        self.say(
            "[The answer has over two thousand six hundred zeros after the decimal point], [so a computer just rounds it to zero.]",
            m.AnimationGroup(m.FadeIn(result, shift=0.2 * m.UP), m.FadeIn(zeros)),
            m.Succession(m.FadeIn(zero, scale=1.2), m.Wiggle(zero, scale_value=1.06)),
        )
        self.say(
            "[The fix is to take the logarithm], [which turns a product into a sum.]",
            m.Succession(m.FadeOut(problem, run_time=0.5), m.FadeIn(top, shift=0.2 * m.UP, run_time=0.8)),
            m.AnimationGroup(
                m.GrowArrow(down),
                m.FadeIn(down_label),
                m.FadeTransform(top[0].copy(), logged[0]),
                m.FadeTransform(top[1].copy(), logged[1]),
                m.FadeTransform(top[2].copy(), logged[2]),
                m.FadeTransform(top[3].copy(), logged[3:5]),
                m.FadeIn(rule, shift=0.15 * m.UP),
                run_time=1.6,
            ),
        )
        self.say(
            "[The log-likelihood is seven log theta, plus three log of one minus theta.]",
            m.AnimationGroup(
                m.FadeIn(name, scale=1.2),
                m.Succession(m.Indicate(logged[2], color=BLUE, scale_factor=1.15), m.Indicate(logged[4], color=PINK, scale_factor=1.15)),
            ),
        )
        self.say(
            "[For ten thousand flips, the log-likelihood is about six thousand below zero]: [a perfectly ordinary number.]",
            m.FadeIn(big, shift=0.2 * m.UP),
            m.FadeIn(easy, scale=1.2),
        )
        self.say(
            "[And because the log always rises with its input], [the peak stays in exactly the same place.]",
            m.AnimationGroup(m.Create(up), m.Create(lo), m.FadeIn(lo_ticks), m.FadeIn(up_name), m.FadeIn(lo_name), m.Create(c_up), m.Create(c_lo), run_time=1.8),
            m.AnimationGroup(m.Create(same), m.FadeIn(dots, scale=0.5), m.FadeIn(same_label, shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the recipe

    def recipe(self):
        self.next_section("The recipe", notes="theta-hat = argmax over theta of the sum of log p(x_i | theta); flip the sign for a loss (the negative log-likelihood).")
        f = mtex(r"\hat{\theta}", "=", r"\arg\max_{\theta}", r"\sum_{i=1}^{n}", r"\log p(x_i \mid \theta)", size=66).move_to([0, 1.3, 0])
        g = mtex(r"\hat{\theta}", "=", r"\arg\min_{\theta}", "-", r"\sum_{i=1}^{n}", r"\log p(x_i \mid \theta)", size=66).move_to([0, 1.3, 0])
        for formula, (hat, op, total, score) in ((f, (0, 2, 3, 4)), (g, (0, 2, 4, 5))):
            formula[hat].set_color(GOOD)
            formula[op].set_color(AMBER if formula is f else BAD)
            formula[score].set_color(TEAL)
        g[3].set_color(BAD)

        rows = m.VGroup(
            m.VGroup(mtex(r"\log p(x_i \mid \theta)", size=36, color=TEAL), text("scores one data point: how likely the model finds it", 24, MUTED)),
            m.VGroup(mtex(r"\textstyle\sum", size=40, color=INK), text("adds the scores up over the whole dataset", 24, MUTED)),
            m.VGroup(mtex(r"\arg\max_{\theta}", size=36, color=AMBER), text("picks the θ with the biggest total", 24, MUTED)),
        )
        for row in rows:
            row.arrange(m.RIGHT, buff=0.35)
        rows.arrange(m.DOWN, buff=0.38, aligned_edge=m.LEFT).move_to([0, -0.85, 0])
        symbol_x = rows[0][0].get_x()
        for row in rows:  # symbols centred in one column, words left-aligned in the next
            row[0].set_x(symbol_x)
        column = rows[0][0].get_right()[0] + 0.4
        for row in rows:
            row[1].shift((column - row[1].get_left()[0]) * m.RIGHT)
        rows.move_to([0, -0.85, 0])
        coin_case = m.VGroup(
            text("for our coin:", 24, MUTED),
            flip_chip(True, 0.62, 24),
            mtex(r"\to\ \log\theta", size=34, color=BLUE),
            flip_chip(False, 0.62, 24),
            mtex(r"\to\ \log(1-\theta)", size=34, color=PINK),
        ).arrange(m.RIGHT, buff=0.25)
        coin_case[3].shift(0.35 * m.RIGHT)
        coin_case[4].shift(0.35 * m.RIGHT)
        coin_case.move_to([0, -3.05, 0])
        loss = pill("the negative log-likelihood: a loss function", BAD, BAD_FILL, 28).move_to([0, -0.9, 0])
        train = text("training minimizes it, just like in Episode 2", 24, MUTED).move_to([0, -1.9, 0])

        self.say(
            "[In general, maximum likelihood says]: [choose the theta that maximizes the sum of log p of x given theta.]",
            self.new_header("THE RECIPE", "Maximum likelihood in one line"),
            m.Write(f, run_time=2.2),
        )
        self.say(
            "[Each log term scores one data point]: [for our coin, the log of theta for a head, or of one minus theta for a tail.]",
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.2 * m.UP), m.Indicate(f[4], color=TEAL, scale_factor=1.1)),
            m.FadeIn(coin_case, shift=0.2 * m.UP),
        )
        self.say(
            "[The sum adds that up over the whole dataset], [and arg max picks the theta with the biggest total.]",
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.2 * m.UP), m.Indicate(f[3], color=INK, scale_factor=1.15)),
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.2 * m.UP), m.Indicate(f[2], color=AMBER, scale_factor=1.15)),
        )
        self.split(f)
        self.say(
            "[Flip the sign, and maximizing becomes minimizing]: [the negative log-likelihood is a loss function.]",
            m.AnimationGroup(
                m.FadeOut(rows),
                m.FadeOut(coin_case),
                morph(f, g, [(0, 0), (1, 1), (2, 2), (3, 4), (4, 5)], new=[3]),
                run_time=1.4,
            ),
            m.AnimationGroup(m.FadeIn(loss, scale=1.15), m.FadeIn(train, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the reveal: Gaussian noise

    def reveal(self):
        self.next_section(
            "Squared error from Gaussian noise",
            notes="Illustrative points: y = 0.75x + 1 plus Gaussian noise (sd 0.5), seed 4; intercept fixed, slope swept. Maximising the Gaussian log-likelihood = minimising squared error (asserted on a grid). Gauss, 1809 (Theoria motus corporum coelestium: with normal errors, least squares gives the maximum likelihood estimate; see Stigler 2007, arXiv:0804.2996; checked 2026-10-21).",
        )
        ax = axes([0, 7, 1], [0, 7.5, 1], 6.0, 4.8).move_to([-3.45, -0.55, 0])
        x_name = text("x", 24, MUTED).next_to(ax.x_axis.get_end(), m.DOWN, buff=0.15)
        y_name = text("y", 24, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.18)
        points = m.VGroup(*[m.Dot(ax.c2p(x, y), radius=0.09, color=AMBER) for x, y in zip(XS, YS)]).set_z_index(4)
        w = m.ValueTracker(W_BEST)
        bell_w = 0.6

        def make_line():
            v = w.get_value()
            return m.Line(ax.c2p(0.3, v * 0.3 + LINE_B), ax.c2p(6.8, v * 6.8 + LINE_B)).set_stroke(SKY, width=5).set_z_index(2)

        def make_bells():
            v = w.get_value()
            us = np.linspace(-3, 3, 61)
            group = m.VGroup()
            for x in XS:
                mu = v * x + LINE_B
                base = [ax.c2p(x, mu + u * SIGMA) for u in us]
                curve = [b + np.array([bell_w * np.exp(-u * u / 2), 0, 0]) for b, u in zip(base, us)]
                shape = m.VMobject().set_points_as_corners([base[0], *curve, base[-1], base[0]])
                shape.set_stroke(VIOLET, width=2.5, opacity=0.9).set_fill(VIOLET, opacity=0.12)
                group.add(shape)
            return group.set_z_index(1)

        def make_heights():
            v = w.get_value()
            group = m.VGroup()
            for x, y in zip(XS, YS):
                r = (y - (v * x + LINE_B)) / SIGMA
                a = ax.c2p(x, y)
                group.add(m.Line(a, a + np.array([max(0.03, bell_w * np.exp(-r * r / 2)), 0, 0])).set_stroke(VIOLET, width=6))
            return group.set_z_index(3)

        line = m.always_redraw(make_line)
        bells = m.always_redraw(make_bells)
        heights = m.always_redraw(make_heights)
        likely = text("likely", 22, GOOD, weight=m.BOLD).next_to(points[NEAR], m.UP, buff=1.05)
        unlikely = text("unlikely", 22, BAD, weight=m.BOLD).next_to(points[FAR], m.LEFT, buff=0.35).shift(0.35 * m.UP)

        p = mtex(r"p(y \mid \hat{y})", r"\propto", r"\exp\Big(", r"-\frac{(y-\hat{y})^2}{2\sigma^2}", r"\Big)", size=44).move_to([3.6, 1.55, 0])
        q = mtex(r"\log p(y \mid \hat{y})", "=", r"-\frac{(y-\hat{y})^2}{2\sigma^2}", "+", r"\text{const}", size=44).move_to([3.6, 1.55, 0])
        p[0].set_color(VIOLET)
        p[3].set_color(BAD)
        q[0].set_color(TEAL)
        q[2].set_color(BAD)
        keys = m.VGroup(
            m.VGroup(legend(r"\hat{y}", "the line's prediction", SKY, 38, 22), legend("y", "the real point", AMBER, 38, 22)).arrange(m.RIGHT, buff=0.5),
            legend(r"\sigma", "how spread out the noise is", VIOLET, 42, 22),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([3.7, 0.15, 0])

        e_num = number(sse(W_BEST), 2, BAD, 34)
        e_num.add_updater(lambda d: d.set_value(sse(w.get_value())))
        g_num = number(gauss_ll(W_BEST), 2, TEAL, 34)
        g_num.add_updater(lambda d: d.set_value(gauss_ll(w.get_value())))
        e_label = text("squared error", 26, BAD, weight=m.SEMIBOLD)
        g_label = text("log-likelihood", 26, TEAL, weight=m.SEMIBOLD)
        labels = m.VGroup(e_label, g_label).arrange(m.DOWN, buff=0.4, aligned_edge=m.LEFT).move_to([2.6, -1.35, 0])
        e_num.next_to(e_label, m.RIGHT, buff=0.5)
        g_num.next_to(g_label, m.RIGHT, buff=0.5)
        g_num.align_to(e_num, m.LEFT)
        scores = m.VGroup(labels, e_num, g_num)
        same = pill("max likelihood = min squared error", GOOD, GOOD_FILL, 24).move_to([3.7, -2.6, 0])
        gauss = text("Gauss, 1809", 20, MUTED).next_to(same, m.DOWN, buff=0.2)

        self.say(
            "[Now the big reveal.] [Here's a line through noisy points, just like in episode two.]",
            self.new_header("THE BIG REVEAL", "Squared error from Gaussian noise"),
            m.AnimationGroup(
                m.Create(ax),
                m.FadeIn(x_name),
                m.FadeIn(y_name),
                m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in points], lag_ratio=0.15),
                m.Create(line),
            ),
        )
        self.say(
            "[Assume each point is the line's prediction, plus a bit of Gaussian noise]: [a bell curve around the line.]",
            m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.7) for d in points], lag_ratio=0.15),
            m.Create(bells, run_time=1.6),
        )
        self.say(
            "[A point near the line lands on the bulge of its bell, so it's likely.] [A far point lands out in the thin tail.]",
            m.AnimationGroup(m.Create(heights), m.FadeIn(likely, shift=0.15 * m.DOWN), m.Indicate(points[NEAR], color=GOOD, scale_factor=1.8)),
            m.AnimationGroup(m.FadeIn(unlikely, shift=0.15 * m.RIGHT), m.Indicate(points[FAR], color=BAD, scale_factor=1.8)),
        )
        self.say(
            "[The bell's formula] [has the squared gap between the point and the prediction, up in its exponent.]",
            m.AnimationGroup(m.Write(p), m.FadeIn(keys, shift=0.15 * m.UP), run_time=1.4),
            m.Circumscribe(p[3], color=BAD, buff=0.08),
        )
        self.split(p)
        self.say(
            "[Take the log, and the exponential disappears]: [what's left is minus the squared gap, plus a constant.]",
            morph(p, q, [(0, 0), (1, 1), (3, 2)], new=[3, 4], gone=[2, 4], run_time=1.4),
            m.Indicate(q[2], color=BAD, scale_factor=1.12),
        )
        self.say(
            "[Add it up over every point], [and tilt the line: the two scores move as mirror images.]",
            m.FadeIn(scores, shift=0.2 * m.UP),
            m.Succession(
                w.animate(run_time=1.4).set_value(W_BEST + 0.22),
                w.animate(run_time=1.8).set_value(W_BEST - 0.18),
                w.animate(run_time=1.2).set_value(W_BEST),
            ),
        )
        self.say(
            "[The line with the smallest squared error is the line with the highest likelihood.]",
            m.AnimationGroup(m.FadeIn(same, scale=1.15), m.Indicate(labels, color=GOOD, scale_factor=1.1)),
        )
        self.say(
            "[So squared error isn't arbitrary]: [it's maximum likelihood, if the noise is Gaussian.]",
            m.Indicate(e_label, color=BAD, scale_factor=1.12),
            m.AnimationGroup(m.FadeIn(gauss), m.Circumscribe(same, color=GOOD, buff=0.08)),
        )
        self.wait(0.3)
        self.stop(line, bells, heights, e_num, g_num)
        self.clear()

    # ------------------------------------------------------------------ yes / no labels

    def yes_no(self):
        self.next_section("Yes or no labels", notes="A yes/no label is a coin flip with the model's predicted chance as its bias; the negative log-likelihood is binary cross-entropy (Episode 16).")
        pairs = m.VGroup(
            m.VGroup(chip("yes", BLUE, BLUE_FILL, width=1.3, height=0.72, size=28), text("is like", 24, MUTED), flip_chip(True, 0.72, 28)),
            m.VGroup(chip("no", PINK, PINK_FILL, width=1.3, height=0.72, size=28), text("is like", 24, MUTED), flip_chip(False, 0.72, 28)),
        )
        for pair in pairs:
            pair.arrange(m.RIGHT, buff=0.3)
        pairs.arrange(m.RIGHT, buff=1.6).move_to([0, 1.55, 0])
        model = legend(r"\hat{y}", "the model's predicted chance of yes: the coin's bias", SKY, 40, 24).move_to([0, 0.55, 0])
        c = mtex(r"\log p(y \mid \hat{y})", "=", r"y\log\hat{y}", "+", r"(1-y)\log(1-\hat{y})", size=52).move_to([0, -0.6, 0])
        d = mtex(r"\text{loss}", "=", "-", r"\Big[", r"y\log\hat{y}", "+", r"(1-y)\log(1-\hat{y})", r"\Big]", size=52).move_to([0, -0.6, 0])
        c[0].set_color(TEAL)
        c[2].set_color(BLUE)
        c[4].set_color(PINK)
        d[0].set_color(BAD)
        d[2].set_color(BAD)
        d[4].set_color(BLUE)
        d[6].set_color(PINK)
        cases = m.VGroup(
            m.VGroup(text("yes (y = 1):", 24, MUTED), mtex(r"\log\hat{y}", size=36, color=BLUE)).arrange(m.RIGHT, buff=0.2),
            m.VGroup(text("no (y = 0):", 24, MUTED), mtex(r"\log(1-\hat{y})", size=36, color=PINK)).arrange(m.RIGHT, buff=0.2),
        ).arrange(m.RIGHT, buff=1.2).move_to([0, -1.7, 0])
        name = m.VGroup(pill("binary cross-entropy", BAD, BAD_FILL, 26), pill("Episode 16", AMBER, AMBER_FILL, 22)).arrange(m.RIGHT, buff=0.4)
        name.move_to([0, -2.9, 0])

        self.say(
            "[What about yes-or-no labels, like spam or not spam?] [Each label is a coin flip], [and the model predicts its bias.]",
            m.AnimationGroup(self.new_header("YES OR NO", "Labels are coin flips"), m.FadeIn(pairs[0][0], shift=0.2 * m.UP), m.FadeIn(pairs[1][0], shift=0.2 * m.UP)),
            m.AnimationGroup(*[m.FadeIn(pr[1:], shift=0.2 * m.LEFT) for pr in pairs]),
            m.FadeIn(model, shift=0.2 * m.UP),
        )
        first, second = c[0:3], c[3:]
        self.say(
            "[For a yes, the log-likelihood is the log of the predicted chance]; [for a no, the log of one minus it]: [our coin formula, one flip at a time.]",
            m.AnimationGroup(m.Write(first), m.FadeIn(cases[0], shift=0.2 * m.UP)),
            m.AnimationGroup(m.Write(second), m.FadeIn(cases[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.Indicate(pairs[0][2], color=BLUE, scale_factor=1.2), m.Indicate(pairs[1][2], color=PINK, scale_factor=1.2)),
        )
        self.remove(first, second)
        self.add(*c)
        self.say(
            "[Flip the sign, and you get the cross-entropy loss]: [the star of episode sixteen.]",
            m.AnimationGroup(morph(c, d, [(0, 0), (1, 1), (2, 4), (3, 5), (4, 6)], new=[2, 3, 7]), m.FadeIn(name[0], scale=1.15), run_time=1.4),
            m.FadeIn(name[1], shift=0.2 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section("The big picture", notes="Choose the noise model; maximum likelihood gives the loss: Gaussian noise gives squared error, yes/no labels give cross-entropy.")
        heads = m.VGroup(text("if you assume the data is…", 26, MUTED), text("maximum likelihood gives you…", 26, MUTED))
        heads[0].move_to([-3.4, 1.75, 0])
        heads[1].move_to([3.4, 1.75, 0])

        def side(x, y, color, body, words, sub):
            box = card(5.7, 1.55, color, CARD, stroke=2).move_to([x, y, 0])
            label = m.VGroup(text(words, 28, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            inside = m.VGroup(body, label).arrange(m.RIGHT, buff=0.4).move_to(box) if body is not None else label.move_to(box)
            return m.VGroup(box, inside)

        flips = m.VGroup(flip_chip(True, 0.55, 22), flip_chip(False, 0.55, 22)).arrange(m.RIGHT, buff=0.12)
        left = m.VGroup(
            side(-3.4, 0.5, VIOLET, mini_bell(), "Gaussian noise", "around a line's predictions"),
            side(-3.4, -1.45, BLUE, flips, "yes-or-no labels", "coin flips with a predicted bias"),
        )
        right = m.VGroup(
            side(3.4, 0.5, BAD, None, "squared error", "Episode 2"),
            side(3.4, -1.45, BAD, None, "cross-entropy", "Episode 16"),
        )
        links = m.VGroup(*[arrow([l.get_right()[0], l.get_y(), 0], [r.get_left()[0], r.get_y(), 0], MUTED, buff=0.12) for l, r in zip(left, right)])
        tagline = text("Choose the noise, and the loss follows.", 30, INK, weight=m.SEMIBOLD).move_to([0, -3.15, 0])

        self.say(
            "[So here's the big idea.] [Choose how you think the data was made], [and maximum likelihood hands you the loss.]",
            m.AnimationGroup(self.new_header("THE BIG PICTURE", "Choose the noise, get the loss"), m.FadeIn(heads, shift=0.2 * m.DOWN)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.RIGHT) for c in left], lag_ratio=0.35),
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.GrowArrow(a), m.FadeIn(c, shift=0.2 * m.RIGHT)) for a, c in zip(links, right)], lag_ratio=0.35),
                m.FadeIn(tagline, shift=0.2 * m.UP),
            ),
        )
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section("The catch", notes="Three heads in three flips gives theta-hat = 1 (tails 'impossible'). Fixes: more data, or a prior belief via Bayes' rule (Episode 14), i.e. a MAP estimate.")
        the_coin = coin(0.8).move_to([-5.0, 0.55, 0])
        chips = m.VGroup(*[flip_chip(True, 0.8, 30) for _ in range(3)]).arrange(m.RIGHT, buff=0.2).move_to([-2.55, 0.55, 0])
        estimate = mtex(r"\hat{\theta}", "=", r"\tfrac{3}{3}", "=", "1", size=56, color=SKY).move_to([2.6, 0.55, 0])
        estimate[0].set_color(GOOD)
        never = pill("so tails is impossible?", BAD, BAD_FILL, 26).move_to([2.6, -0.65, 0])
        fixes = m.VGroup(pill("more data", GOOD, GOOD_FILL, 26), pill("a prior belief · Bayes' rule, Episode 14", GOOD, GOOD_FILL, 26))
        fixes.arrange(m.RIGHT, buff=0.5).move_to([0, -2.3, 0])
        fix_label = text("fixes:", 24, MUTED).next_to(fixes, m.UP, buff=0.3).align_to(fixes, m.LEFT)

        self.say(
            "[One catch]: [with very little data, maximum likelihood can be overconfident.]",
            self.new_header("THE CATCH", "Little data, big confidence"),
            m.FadeIn(the_coin, scale=0.7),
        )
        self.say(
            "[Three heads in three flips gives theta equal to one]: [a coin that could never land tails.]",
            m.AnimationGroup(
                m.Succession(*[flip(the_coin) for _ in range(3)]),
                m.LaggedStart(*[m.FadeIn(c, target_position=the_coin.get_center(), scale=0.4) for c in chips], lag_ratio=0.3),
                m.FadeIn(estimate, shift=0.2 * m.LEFT),
                run_time=1.8,
            ),
            m.Succession(m.FadeIn(never, scale=1.2), m.Wiggle(never, scale_value=1.06)),
        )
        self.say(
            "[More data helps], [and so does a prior belief, blended in with Bayes' rule from episode fourteen.]",
            m.AnimationGroup(m.FadeIn(fix_label), m.FadeIn(fixes[0], shift=0.2 * m.UP)),
            m.FadeIn(fixes[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Likelihood, log, loss.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Likelihood", "Log", "Loss"), (VIOLET, TEAL, BAD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[2].get_y()  # "Loss": no descender
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Behind every loss function is a guess about how your data was made.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [likelihood], [log], [loss.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Behind every loss function is a guess about how your data was made.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
