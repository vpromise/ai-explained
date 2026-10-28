"""AI, Explained · Episode 14 · Probability for ML

Running example: Episode 6's rare-disease screening test, revisited with the same numbers.
1,000 people, 10 of them sick. The test flags 8 of the 10 sick (it misses 2) and 40 of the 990
healthy (950 cleared):

    P(sick)               = 10 / 1000  = 1%          the prior
    P(positive | sick)    = 8 / 10     = 80%         the likelihood
    P(positive | healthy) = 40 / 990   ≈ 4%          the false-alarm rate
    P(positive)           = 48 / 1000  = 4.8%        the evidence
    P(sick | positive)    = 8 / 48     ≈ 17%         the posterior (Bayes' rule gives the same)
    with a 10% prior, the same test gives P(sick | positive) ≈ 69%

Distributions (illustrative, simulated with seeded numpy): a clinic tests 100 people a day,
each sick with probability 1%; 10,000 simulated days give 0 sick on 37% of days, 1 on 37%,
2 on 18%, 3 on 6%, 4 on 1.5%, 5 or more on 0.3% (the binomial distribution, which the
simulated shares match to the precision shown). Expectation: 100 × 1% = 1 a day. Ten
thousand tests a day (200,000 simulated days): bars around 100, close to a normal curve
with mean 100 and standard deviation √99 ≈ 9.95.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

PEOPLE, SICK = 1000, 10
HEALTHY = PEOPLE - SICK  # 990
TP, FN = 8, 2  # of the ten sick: flagged, missed
FP = 40  # healthy people flagged by mistake
TN = HEALTHY - FP  # 950
POSITIVE = TP + FP  # 48

P_SICK = SICK / PEOPLE  # the prior
P_POS_SICK = TP / SICK  # the likelihood
P_POS_HEALTHY = FP / HEALTHY  # the false-alarm rate
P_POS = POSITIVE / PEOPLE  # the evidence
P_SICK_POS = TP / POSITIVE  # the posterior, from the counts
BAYES = P_POS_SICK * P_SICK / P_POS  # the posterior, from the formula
COMMON = 0.10  # "one person in ten"


def posterior(prior):
    """P(sick | positive) for another prior, the test unchanged (80% caught, 40/990 false alarms)."""
    return P_POS_SICK * prior / (P_POS_SICK * prior + P_POS_HEALTHY * (1 - prior))


assert TP + FN == SICK and FP + TN == HEALTHY and TN == 950 and POSITIVE == 48
assert P_SICK == 0.01 and P_POS_SICK == 0.8 and round(100 * P_POS_HEALTHY) == 4
assert abs(P_POS - 0.048) < 1e-12 and abs(P_SICK_POS - 1 / 6) < 1e-12 and round(100 * P_SICK_POS) == 17
assert abs(BAYES - P_SICK_POS) < 1e-12  # the formula and the counts agree
assert abs(posterior(P_SICK) - P_SICK_POS) < 1e-12 and round(100 * posterior(COMMON)) == 69
assert HEALTHY / SICK == 99 and FP > TP  # "ninety-nine to one", false alarms outnumber real cases
assert round(100 * (1 - P_SICK_POS)) == 83  # the classifier's "healthy" bar

# a clinic: 100 tests a day, 10,000 simulated days
DAY_TESTS, DAYS, SIM_SEED = 100, 10_000, 41
DAILY = np.random.default_rng(SIM_SEED).binomial(DAY_TESTS, P_SICK, DAYS)
BINS = 6  # 0, 1, 2, 3, 4, and "5 or more"
CUM = np.cumsum(np.eye(BINS)[np.minimum(DAILY, BINS - 1)], axis=0)  # CUM[d - 1, k]: days so far with outcome k
FREQ = CUM[-1] / DAYS
PMF = np.array([math.comb(DAY_TESTS, k) * P_SICK**k * (1 - P_SICK) ** (DAY_TESTS - k) for k in range(DAY_TESTS + 1)])
EXACT = np.append(PMF[: BINS - 1], PMF[BINS - 1 :].sum())


def pct(v):
    return f"{100 * v:.0f}%" if v >= 0.05 else f"{100 * v:.1f}%"


LABELS = [pct(f) for f in FREQ]
assert LABELS == [pct(e) for e in EXACT] == ["37%", "37%", "18%", "6%", "1.5%", "0.3%"]
assert abs(FREQ.sum() - 1) < 1e-9 and abs(PMF.sum() - 1) < 1e-9
EXPECT = DAY_TESTS * P_SICK
assert EXPECT == 1 and abs(float((np.arange(DAY_TESTS + 1) * PMF).sum()) - 1) < 1e-9
assert round(float(DAILY.mean()), 1) == 1.0  # the simulated days average one sick person too

# many clinics: 10,000 tests a day, 200,000 simulated days
BIG_TESTS, BIG_DAYS = 10_000, 200_000
BIG = np.random.default_rng(SIM_SEED + 1).binomial(BIG_TESTS, P_SICK, BIG_DAYS)
BIG_MEAN = BIG_TESTS * P_SICK
BIG_SD = math.sqrt(BIG_TESTS * P_SICK * (1 - P_SICK))
K_LO, K_HI, K_ORIGIN = 66, 134, 62
BIG_FREQ = {k: float(np.mean(BIG == k)) for k in range(K_LO, K_HI + 1)}


def normal(x):
    return np.exp(-0.5 * ((x - BIG_MEAN) / BIG_SD) ** 2) / (BIG_SD * math.sqrt(2 * math.pi))


assert BIG_MEAN == 100 and round(float(BIG.mean())) == 100
assert max(abs(BIG_FREQ[k] - float(normal(k))) for k in BIG_FREQ) < 0.003  # the bars hug the bell curve

# ---------------------------------------------------------------------------- the crowd (Episode 6's layout)

COLS, ROWS, STEP = 40, 25, 0.13
DOT_R, RING_R = 0.04, 0.082


def crowd():
    """Who is sick, who is missed, which healthy people get flagged: spread out, no two
    marked people next to each other (the same draw as Episode 6, seed 6)."""
    rng = np.random.default_rng(6)
    sick = []
    for bc in range(5):
        for br in range(2):
            sick.append((br * 13 + int(rng.integers(1, 11))) * COLS + bc * 8 + int(rng.integers(1, 7)))
    taken = set(sick)
    flagged = []
    for i in rng.permutation(PEOPLE):
        i = int(i)
        if len(flagged) == FP:
            break
        r, c = divmod(i, COLS)
        near = {(r + dr) * COLS + (c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if i not in taken and not (near & taken) and 0 < c < COLS - 1:
            flagged.append(i)
            taken.add(i)
    missed = [sick[2], sick[7]]
    return sick, missed, sorted(flagged)


def person(i, center):
    r, c = divmod(i, COLS)
    pos = np.array(center) + np.array([(c - (COLS - 1) / 2) * STEP, ((ROWS - 1) / 2 - r) * STEP, 0])
    return m.Dot(pos, radius=DOT_R, color=MUTED).set_fill(MUTED, opacity=0.55).set_z_index(3)


def ring(dot, color, radius=RING_R, width=2.5):
    return m.Circle(radius=radius).set_stroke(color, width=width).move_to(dot.get_center()).set_z_index(3)


class People:
    """The 1,000 dots, the sick ones pink, split into the four groups of the confusion matrix."""

    def __init__(self, center):
        sick_ids, missed_ids, flagged_ids = crowd()
        self.all = m.VGroup(*[person(i, center) for i in range(PEOPLE)])
        self.sick = m.VGroup(*[self.all[i] for i in sick_ids])
        for d in self.sick:
            d.set_fill(PINK, opacity=1).scale(1.6)
        self.healthy = m.VGroup(*[d for i, d in enumerate(self.all) if i not in sick_ids])
        self.fn = m.VGroup(*[self.all[i] for i in missed_ids])
        self.tp = m.VGroup(*[self.all[i] for i in sick_ids if i not in missed_ids])
        self.fp = m.VGroup(*[self.all[i] for i in flagged_ids])
        assert (len(self.tp), len(self.fn), len(self.fp), len(self.healthy) - len(self.fp)) == (TP, FN, FP, TN)


# ---------------------------------------------------------------------------- helpers


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


def labelled(name, desc, color):
    return m.VGroup(text(name, 24, color, weight=m.BOLD), text(desc, 20, MUTED)).arrange(m.DOWN, buff=0.06)


def bar_rect(ax, x, half, h, color, opacity=0.85):
    """A bar standing on the axis at x (axis units), half-width `half`, height h (axis units)."""
    left, base = ax.c2p(x - half, 0), ax.c2p(x, 0)
    width = 2 * (base[0] - left[0])
    height = max(ax.c2p(x, h)[1] - base[1], 0.004)
    rect = m.Rectangle(width=width, height=height, stroke_width=0).set_fill(color, opacity=opacity)
    return rect.move_to([base[0], base[1], 0], aligned_edge=m.DOWN)


def morph(source, target, pairs=None, **kw):
    """TransformMatchingTex for these formulas (its key_map needs equal glyph counts): each part
    of `source` moves into the part of `target` written the same way, or paired in `pairs`
    (source LaTeX → target LaTeX); the rest fade. The source's parts must be in the scene on
    their own (`Ep14Probability.flatten`); the target's parts end up there the same way."""
    pairs = pairs or {}
    free = [t for t in target if len(t)]
    anims = []
    for s in source:
        if not len(s):
            continue
        want = pairs.get(s.tex_string, s.tex_string)
        match = next((t for t in free if t.tex_string == want), None)
        if match is None:
            anims.append(m.FadeOut(s))
        else:
            free.remove(match)  # written alike: glyph to glyph; paired: a cross-fade (glyph counts differ)
            anims.append(m.ReplacementTransform(s, match) if want == s.tex_string else m.FadeTransform(s, match))
    anims += [m.FadeIn(t) for t in free]
    return m.AnimationGroup(*anims, **kw)


# the bayes chain: the posterior, numbers first
LHS = r"P(\text{sick} \mid \text{positive})"


def chain(state):
    eq = ("=", INK)
    if state == 0:
        parts = [(LHS, GOOD), eq, (r"\frac{", INK), ("8", PINK), ("}{", INK), ("48", AMBER), ("}", INK)]
    elif state == 1:
        parts = [(LHS, GOOD), eq, (r"\frac{", INK), ("1000", MUTED), (r"\times", INK), (r"1\%", PINK), (r"\times", INK), (r"80\%", TEAL),
                 ("}{", INK), ("48", AMBER), ("}", INK)]
    elif state == 2:
        parts = [(LHS, GOOD), eq, (r"\frac{", INK), ("1000", MUTED), (r"\times", INK), (r"1\%", PINK), (r"\times", INK), (r"80\%", TEAL),
                 ("}{", INK), ("{1000}", MUTED), (r"{\times}", INK), (r"4.8\%", AMBER), ("}", INK)]
    else:
        parts = [(LHS, GOOD), eq, (r"\frac{", INK), (r"80\%", TEAL), (r"\times", INK), (r"1\%", PINK), ("}{", INK), (r"4.8\%", AMBER), ("}", INK),
                 (r"\approx", INK), (r"17\%", GOOD)]
    return formula(*parts, size=42)


class Ep14Probability(Episode):
    number = 14

    def lesson(self):
        self.hook()
        self.idea()
        self.distribution()
        self.expectation()
        self.bell_curve()
        self.conditional()
        self.flip()
        self.bayes()
        self.why_it_matters()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    def flatten(self, formula_mob):
        """Put a formula's parts into the scene on their own (no visible change), for morph()."""
        self.remove(formula_mob)
        self.add(*[p for p in formula_mob if len(p)])

    def fade_all_but(self, *keep, run_time=0.8):
        """Like clear(), but keeps everything that shares a member with `keep` (the crowd is
        split into sub-groups as it is animated)."""
        kept = {id(x) for k in keep for x in k.get_family()}
        gone = [
            mob
            for mob in self.mobjects[1:]
            if mob is not self.head and not ({id(x) for x in mob.get_family()} & kept)
        ]
        if gone:
            self.play(*[m.FadeOut(mob) for mob in gone], run_time=run_time)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A positive result from a test that catches 80% of sick people: are you 80% likely to be sick? (Episode 6's test.)")
        test = icon("ep14-test", 2.7).move_to([-3.7, 0.45, 0])
        line = test[3]  # the result line, shown on cue
        test.remove(line)
        halo = glow(AMBER, radius=2.4).move_to(test)
        result = pill("POSITIVE", AMBER, AMBER_FILL, 26).next_to(test, m.DOWN, buff=0.4)
        hero = text("80%", 150, TEAL, weight=m.BOLD).move_to([2.6, 1.15, 0])
        sub = text("of sick people test positive", 28, MUTED).next_to(hero, m.DOWN, buff=0.25)
        question = text("So is your chance of being sick 80%?", 30, INK, weight=m.SEMIBOLD).move_to([2.6, -0.95, 0])
        stamp = pill("✗  not even close", BAD, BAD_FILL, 28).move_to([2.6, -1.95, 0])
        words = ("putting a number on uncertainty", "updating it with evidence")
        today = m.VGroup(*[chip(w, SKY, SKY_FILL, width=text(w, 26).width + 0.8, height=0.8, size=26) for w in words])
        today.arrange(m.RIGHT, buff=0.5).move_to([0, -3.15, 0])

        self.say(
            "So... [imagine you take a test for a rare disease], [and it comes back positive.]",
            m.FadeIn(test, shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(line, scale=1.4), m.FadeIn(result, scale=1.2)),
        )
        self.say(
            "[The test catches eighty percent of sick people.] [So are you eighty percent likely to be sick?]",
            m.AnimationGroup(m.FadeIn(hero, shift=0.3 * m.UP, scale=0.85), m.FadeIn(sub)),
            m.FadeIn(question, shift=0.2 * m.UP),
        )
        self.say(
            "[Not even close.] [Today: how to put a number on uncertainty], [and how evidence should change it.]",
            m.Succession(m.FadeIn(stamp, scale=1.2), m.Wiggle(stamp, scale_value=1.06)),
            m.FadeIn(today[0], shift=0.3 * m.UP),
            m.FadeIn(today[1], shift=0.3 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ uncertainty as a number

    def idea(self):
        self.next_section(
            "Uncertainty as a number",
            notes="A probability runs from 0 (impossible) to 1 (certain); a fair coin's heads is 1/2. Episode 6's 1,000 people, 10 sick: P(sick) = 1%.",
        )
        x0, x1, sy = -5.5, 5.5, 1.45

        def sx(p):
            return x0 + (x1 - x0) * p

        scale = m.Line([x0, sy, 0], [x1, sy, 0]).set_stroke(FAINT, width=5)
        ticks = m.VGroup(*[m.Line([sx(p), sy - 0.13, 0], [sx(p), sy + 0.13, 0]).set_stroke(MUTED, width=3) for p in (0, 0.5, 1)])
        zero = m.VGroup(text("0", 26, INK, weight=m.BOLD), text("impossible", 22, MUTED)).arrange(m.DOWN, buff=0.08).next_to([sx(0), sy - 0.15, 0], m.DOWN, buff=0.1)
        one = m.VGroup(text("1", 26, INK, weight=m.BOLD), text("certain", 22, MUTED)).arrange(m.DOWN, buff=0.08).next_to([sx(1), sy - 0.15, 0], m.DOWN, buff=0.1)
        zero[1].align_to(zero[0], m.LEFT).shift(0.3 * m.LEFT)
        half = text("½", 28, BLUE, weight=m.BOLD).next_to([sx(0.5), sy - 0.15, 0], m.DOWN, buff=0.1)
        coin = icon("ep14-coin", 0.62).move_to([sx(0.5), sy + 0.62, 0])
        coin_label = text("a fair coin lands heads", 22, BLUE, weight=m.SEMIBOLD).next_to(coin, m.RIGHT, buff=0.22)
        coin_dot = m.Dot([sx(0.5), sy, 0], radius=0.11, color=BLUE).set_z_index(2)
        sick_dot = m.Dot([sx(P_SICK), sy, 0], radius=0.11, color=PINK).set_z_index(2)
        sick_label = text("a random person is sick", 22, PINK, weight=m.SEMIBOLD)
        sick_label.move_to([0, sy + 0.62, 0]).align_to([sx(P_SICK) - 0.25, 0, 0], m.LEFT)
        sick_tick = m.Line([sx(P_SICK), sy + 0.14, 0], [sx(P_SICK), sy + 0.42, 0]).set_stroke(PINK, width=3)

        people = People([-3.6, -1.45, 0])
        count = text("1,000 people", 44, INK, weight=m.BOLD).move_to([3.4, -0.3, 0])
        legend = m.VGroup(m.Dot(radius=0.09, color=PINK), text("10 are sick", 28, PINK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2).move_to([3.4, -1.05, 0])
        rule = formula((r"P(\text{sick})", PINK), ("=", INK), (r"\frac{10}{1000}", INK), ("=", INK), (r"1\%", PINK), size=44).move_to([3.4, -2.35, 0])
        what = text("P( … ): the probability of …", 20, MUTED).next_to(rule, m.DOWN, buff=0.25)

        self.say(
            "[A probability is a number from zero to one]: [zero means impossible], [one means certain.]",
            m.AnimationGroup(self.new_header("THE IDEA", "Uncertainty as a number"), m.Create(scale), m.FadeIn(ticks)),
            m.FadeIn(zero, shift=0.2 * m.UP),
            m.FadeIn(one, shift=0.2 * m.UP),
        )
        self.say(
            "[A fair coin landing heads sits right in the middle]: [one half.]",
            m.AnimationGroup(m.FadeIn(coin, shift=0.2 * m.DOWN), m.FadeIn(coin_label), m.FadeIn(coin_dot, scale=0.4)),
            m.FadeIn(half, shift=0.2 * m.UP),
        )
        self.say(
            "[Now, the screening test from episode six]: [one thousand people], [and ten of them are sick.]",
            m.FadeIn(people.healthy, lag_ratio=0.002, run_time=1.6),
            m.FadeIn(count, shift=0.2 * m.UP),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(d, scale=2.0) for d in people.sick], lag_ratio=0.1),
                m.FadeIn(legend, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[So the chance that a random person is sick is ten in a thousand]: [one percent.]",
            m.AnimationGroup(m.Write(rule[:3]), m.FadeIn(what)),
            m.AnimationGroup(
                m.FadeIn(rule[3:], shift=0.2 * m.RIGHT),
                m.FadeIn(sick_dot, scale=0.4),
                m.Create(sick_tick),
                m.FadeIn(sick_label, shift=0.2 * m.DOWN),
                m.Flash([sx(P_SICK), sy, 0], color=PINK, flash_radius=0.3, line_length=0.12),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ distributions

    def distribution(self):
        self.next_section(
            "Distributions",
            notes="For example (illustrative): a clinic tests 100 people a day, each sick with probability 1%. "
            "10,000 days simulated with numpy (seed 41): 37%, 37%, 18%, 6%, 1.5%, 0.3% for 0..4 and 5+ sick; "
            "these match the exact binomial probabilities to the precision shown.",
        )
        ax = axes([0, 6.5, 10], [0, 0.45, 0.1], 6.6, 3.8).move_to([-2.9, -0.15, 0])
        axis_y = ax.c2p(0, 0)[1]
        xs = [k + 0.75 for k in range(BINS)]
        ticks = m.VGroup(*[text("5+" if k == BINS - 1 else str(k), 24, MUTED, weight=m.SEMIBOLD).move_to([ax.c2p(x, 0)[0], axis_y - 0.62, 0]) for k, x in enumerate(xs)])
        x_label = text("sick people in a day", 22, MUTED).move_to([ax.c2p(3.25, 0)[0], axis_y - 1.08, 0])
        y_label = text("share of days", 22, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.18)
        days = m.ValueTracker(0.0)  # log10 of the days simulated so far

        def so_far():
            return int(round(10 ** days.get_value()))

        def make_bars():
            d = so_far()
            heights = CUM[d - 1] / max(d, 200)  # the first 200 days grow; then the shares settle
            return m.VGroup(*[bar_rect(ax, x, 0.31, h, PINK) for x, h in zip(xs, heights)])

        bars = m.always_redraw(make_bars)
        bars.set_z_index(1)
        tops = [bar_rect(ax, x, 0.31, f, PINK).get_top() for x, f in zip(xs, FREQ)]
        labels = m.VGroup(*[text(s, 22, INK, weight=m.SEMIBOLD).next_to(t, m.UP, buff=0.12) for s, t in zip(LABELS, tops)])

        clinic = icon("ep14-clinic", 1.15).move_to([3.9, 1.55, 0])
        info = m.VGroup(
            text("100 tests a day", 30, INK, weight=m.BOLD),
            text("each person: a 1% chance of being sick", 22, MUTED),
        ).arrange(m.DOWN, buff=0.15).move_to([3.9, 0.35, 0])
        number = m.DecimalNumber(1, num_decimal_places=0, group_with_commas=True, mob_class=Mono, font_size=34, color=INK)
        number.add_updater(lambda n: n.set_value(so_far()))
        readout = m.VGroup(text("days simulated", 24, MUTED, weight=m.SEMIBOLD), number).arrange(m.RIGHT, buff=0.3).move_to([3.9, -0.8, 0])
        name = pill("a probability distribution", PINK, PINK_FILL, 24).move_to([3.9, -1.85, 0])
        total = text("all the bars together add up to 1", 22, MUTED).move_to([3.9, -2.55, 0])
        self.chart = dict(ax=ax, xs=xs, ticks=ticks, x_label=x_label, y_label=y_label, bars=bars, labels=labels, axis_y=axis_y)

        self.say(
            "[For example, say a clinic tests a hundred people every day.] [How many of them will be sick?]",
            m.AnimationGroup(
                self.new_header("DISTRIBUTIONS", "A probability for every outcome"),
                m.FadeIn(clinic, shift=0.2 * m.UP),
                m.FadeIn(info, shift=0.2 * m.UP),
            ),
            m.AnimationGroup(m.Create(ax), m.FadeIn(ticks), m.FadeIn(x_label), m.FadeIn(y_label)),
        )
        self.add(bars)
        self.say(
            "[It changes from day to day.] [So let's simulate ten thousand days, and count the sick people each time.]",
            m.FadeIn(readout, shift=0.2 * m.UP),
            days.animate(run_time=4.2, rate_func=m.linear).set_value(math.log10(DAYS)),
        )
        self.stop(bars, number)
        self.say(
            "[Most days have no sick people, or just one]: [about thirty-seven percent each.] [Two sick people turn up on about eighteen percent of days.]",
            m.AnimationGroup(*[m.Indicate(bars[k], color=PINK, scale_factor=1.06) for k in (0, 1)]),
            m.AnimationGroup(m.FadeIn(labels[0], shift=0.15 * m.UP), m.FadeIn(labels[1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Indicate(bars[2], color=PINK, scale_factor=1.08), m.FadeIn(labels[2], shift=0.15 * m.UP)),
        )
        self.say(
            "[That's a probability distribution]: [every possible outcome, with its chance.] [Together, they add up to one.]",
            m.FadeIn(name, scale=1.15),
            m.LaggedStart(*[m.FadeIn(lab, shift=0.15 * m.UP) for lab in labels[3:]], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(total, shift=0.2 * m.UP), m.Circumscribe(bars, color=PINK, buff=0.12)),
        )
        self.wait(0.3)
        self.side = m.VGroup(clinic, info, readout, name, total)

    # ------------------------------------------------------------------ expectation

    def expectation(self):
        self.next_section(
            "Expectation",
            notes="E[X] = sum of x P(x): 0·37% + 1·37% + 2·18% + 3·6% + … ≈ 1 (exactly 100 × 1% = 1 for the binomial). The mean is the balance point.",
        )
        c = self.chart
        ax, xs, bars, labels, axis_y = c["ax"], c["xs"], c["bars"], c["labels"], c["axis_y"]
        rule = formula((r"E[X]", LIME), ("=", INK), (r"\sum", INK), ("x", INK), (r"\cdot", INK), (r"P(x)", PINK), size=48).move_to([3.7, 1.75, 0])
        legend = m.VGroup(
            m.VGroup(tex("x", 34), text("an outcome: sick people in a day", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(tex("P(x)", 34, PINK), text("its probability: a bar's height", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([3.7, 0.6, 0])
        row1 = tex(r"= 0 \times 37\% + 1 \times 37\%", 36)
        row2 = tex(r"+\; 2 \times 18\% + 3 \times 6\% + \dots", 36)
        rows = m.VGroup(row1, row2).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([3.7, -0.85, 0])
        result = text("≈ 1 sick person a day", 32, LIME, weight=m.BOLD).move_to([3.7, -2.0, 0])
        balance = text("the expectation: where the bars balance", 22, MUTED).move_to([3.7, -2.65, 0])
        tip = np.array([ax.c2p(xs[0] + EXPECT, 0)[0], axis_y - 0.04, 0])
        fulcrum = m.Polygon(tip, tip + np.array([-0.24, -0.4, 0]), tip + np.array([0.24, -0.4, 0]), stroke_width=0).set_fill(LIME, opacity=1)
        fulcrum.set_z_index(2)
        whole = m.VGroup(ax, bars, labels, c["ticks"], c["x_label"], c["y_label"])
        self.fulcrum = fulcrum

        def tilt(angle, run_time):
            return m.Rotate(whole, angle, about_point=tip, run_time=run_time, rate_func=m.smooth)

        self.say(
            "[On average, how many sick people does the clinic see in a day?]",
            m.AnimationGroup(self.new_header("EXPECTATION", "The probability-weighted average"), m.FadeOut(self.side)),
        )
        self.say(
            "[Take each outcome], [multiply it by its probability], [and add them all up.]",
            m.AnimationGroup(m.FadeIn(rule[3], scale=1.3), m.FadeIn(legend[0], shift=0.2 * m.UP), m.LaggedStart(*[m.Indicate(t, color=INK, scale_factor=1.3) for t in c["ticks"]], lag_ratio=0.12)),
            m.AnimationGroup(m.FadeIn(rule[4:], scale=1.3), m.FadeIn(legend[1], shift=0.2 * m.UP), m.LaggedStart(*[m.Indicate(b, color=PINK, scale_factor=1.05) for b in bars], lag_ratio=0.12)),
            m.FadeIn(rule[:3], shift=0.2 * m.RIGHT),
        )
        self.say(
            "[Zero times thirty-seven percent, plus one times thirty-seven percent, plus two times eighteen percent, and so on]: [about one.]",
            m.AnimationGroup(m.Write(row1, run_time=2.2), m.Succession(m.Wait(2.0), m.Write(row2, run_time=2.4))),
            m.FadeIn(result, shift=0.2 * m.UP),
        )
        self.say(
            "[That's the expectation: one sick person a day.] [It's the point where the bars would balance.]",
            m.AnimationGroup(m.Circumscribe(rule[0], color=LIME, buff=0.1), m.FadeIn(balance)),
            m.Succession(m.FadeIn(fulcrum, shift=0.2 * m.UP, run_time=0.5), tilt(0.045, 0.5), tilt(-0.08, 0.7), tilt(0.035, 0.5)),
        )
        self.wait(0.3)
        self.rule_side = m.VGroup(rule, legend, rows, result, balance)

    # ------------------------------------------------------------------ the bell curve

    def bell_curve(self):
        self.next_section(
            "The bell curve",
            notes="For example: 10,000 tests a day, 200,000 days simulated (numpy, seed 42). Bars around 100 = 10,000 × 1%, "
            "close to a normal curve (mean 100, sd √99 ≈ 9.95). The normal approximation to the binomial: de Moivre, 1733.",
        )
        c = self.chart
        big = axes([0, 78, 100], [0, 0.048, 0.01], 8.6, 3.8).move_to([-1.75, -0.15, 0])
        axis_y = big.c2p(0, 0)[1]

        def at(k):
            return k - K_ORIGIN

        bars = m.VGroup(*[bar_rect(big, at(k), 0.42, BIG_FREQ[k], PINK, opacity=0.7) for k in range(K_LO, K_HI + 1)]).set_z_index(1)
        ticks = m.VGroup(*[text(str(k), 24, MUTED, weight=m.SEMIBOLD).move_to([big.c2p(at(k), 0)[0], axis_y - 0.62, 0]) for k in (70, 85, 100, 115, 130)])
        x_label = text("sick people in a day, out of 10,000 tests", 22, MUTED).move_to([big.c2p(at(100), 0)[0], axis_y - 1.08, 0])
        y_label = text("share of days", 22, MUTED).next_to(big.y_axis.get_end(), m.RIGHT, buff=0.18)
        top = big.c2p(0, 0.048)[1]
        mean_line = m.DashedLine(big.c2p(at(100), 0), [big.c2p(at(100), 0)[0], top - 0.05, 0], dash_length=0.12).set_stroke(LIME, width=3.5).set_z_index(2)
        mean_label = text("E[X] = 10,000 × 1% = 100", 24, LIME, weight=m.BOLD).move_to([big.c2p(at(100), 0)[0], top + 0.28, 0])
        ks = np.linspace(K_LO - 1.5, K_HI + 1.5, 220)
        curve = m.VMobject().set_points_as_corners([big.c2p(at(k), float(normal(k))) for k in ks]).set_stroke(INK, width=5).set_z_index(3)
        name = pill("the normal distribution", INK, CARD, 24).move_to([4.75, 1.0, 0])
        source = text("de Moivre, 1733", 20, MUTED).next_to(name, m.DOWN, buff=0.18)
        uses = m.VGroup(*[chip(w, MUTED, CARD, width=text(w, 22, weight=m.SEMIBOLD).width + 0.6, height=0.62, size=22) for w in ("measurement noise", "random starting weights")])
        uses.arrange(m.DOWN, buff=0.22).move_to([4.75, -0.75, 0])

        self.say(
            "[Now picture many clinics, testing ten thousand people a day]: [the expectation grows to a hundred.]",
            m.AnimationGroup(
                self.new_header("THE BELL CURVE", "Many outcomes, one smooth shape"),
                m.FadeOut(self.rule_side),
                m.FadeOut(self.fulcrum),
                m.FadeOut(c["labels"]),
                m.ReplacementTransform(c["ax"], big),
                m.ReplacementTransform(c["bars"], bars),
                m.FadeTransform(c["ticks"], ticks),
                m.FadeTransform(c["x_label"], x_label),
                m.ReplacementTransform(c["y_label"], y_label),
                run_time=1.6,
            ),
            m.AnimationGroup(m.Create(mean_line), m.FadeIn(mean_label, shift=0.2 * m.DOWN)),
        )
        self.say(
            "[With so many outcomes, the bars blend into a smooth bell curve]: [the normal distribution.]",
            m.Create(curve, run_time=2.0),
            m.AnimationGroup(m.FadeIn(name, scale=1.1), m.FadeIn(source)),
        )
        self.say(
            "[You'll meet it all over machine learning], [from measurement noise to a network's random starting weights.]",
            m.ShowPassingFlash(curve.copy().set_stroke(LIME, width=9), time_width=0.5, run_time=1.4),
            m.LaggedStart(*[m.FadeIn(u, shift=0.2 * m.UP) for u in uses], lag_ratio=0.4),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ conditional probability

    def conditional(self):
        self.next_section(
            "Conditional probability",
            notes="Episode 6's counts: P(positive | sick) = 8/10 = 80%; P(positive | healthy) = 40/990 ≈ 4%. Conditioning = shrinking the crowd.",
        )
        people = People([-3.75, -0.2, 0])
        self.people = people
        tp_rings = m.VGroup(*[ring(d, AMBER, RING_R + 0.018) for d in people.tp])
        fp_rings = m.VGroup(*[ring(d, AMBER) for d in people.fp])
        self.rings = m.VGroup(*tp_rings, *fp_rings)
        legend = m.VGroup(
            m.VGroup(m.Dot(radius=0.09, color=PINK), text("sick", 24, PINK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.18),
            m.VGroup(m.Circle(radius=0.1).set_stroke(AMBER, width=3), text("tested positive", 24, AMBER, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.18),
        ).arrange(m.RIGHT, buff=0.6).move_to([-3.75, -2.35, 0])
        self.legend = legend

        f1 = formula(
            (r"P(\text{positive}", TEAL), (r"\mid", INK), (r"\text{sick})", TEAL), ("=", INK), (r"\frac{8}{10}", INK), ("=", INK), (r"80\%", TEAL), size=42
        ).move_to([3.15, 1.45, 0])
        given = text("the bar  |  means “given”", 22, MUTED).next_to(f1, m.DOWN, buff=0.3)
        f2 = formula(
            (r"P(\text{positive}", BAD), (r"\mid", INK), (r"\text{healthy})", BAD), ("=", INK), (r"\frac{40}{990}", INK), (r"\approx", INK), (r"4\%", BAD), size=42
        ).move_to([3.15, -0.55, 0])
        note = pill("given: shrink the crowd, then count", INK, CARD, 24).move_to([3.15, -2.35, 0])
        self.cond_side = m.VGroup(f1, given, f2, note)

        self.say(
            "[Back to our thousand people, who all take the test.] [It flags eight of the ten sick], [and forty of the healthy.]",
            m.AnimationGroup(
                self.new_header("CONDITIONAL PROBABILITY", "Probability, given what you know"),
                m.FadeIn(people.all, lag_ratio=0.001, run_time=1.4),
                m.FadeIn(legend[0], shift=0.2 * m.UP),
            ),
            m.AnimationGroup(m.LaggedStart(*[m.Create(r) for r in tp_rings], lag_ratio=0.15), m.FadeIn(legend[1], shift=0.2 * m.UP)),
            m.LaggedStart(*[m.Create(r) for r in fp_rings], lag_ratio=0.04, run_time=1.6),
        )
        self.say(
            "[Now look only at the sick]: [eight of those ten test positive.]",
            m.AnimationGroup(people.healthy.animate.set_fill(opacity=0.1), fp_rings.animate.set_stroke(opacity=0.12)),
            m.LaggedStart(*[m.Indicate(r, color=AMBER, scale_factor=1.6) for r in tp_rings], lag_ratio=0.1),
        )
        self.say(
            "[That's the probability of a positive test, given that you're sick]: [eighty percent.] [This bar means given.]",
            m.Write(f1[:3]),
            m.FadeIn(f1[3:], shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.Indicate(f1[1], color=INK, scale_factor=1.8), m.FadeIn(given, shift=0.15 * m.UP)),
        )
        self.say(
            "[Among the healthy, forty of nine hundred and ninety test positive anyway]: [about four percent.]",
            m.AnimationGroup(
                people.healthy.animate.set_fill(opacity=0.55),
                fp_rings.animate.set_stroke(opacity=1),
                people.sick.animate.set_fill(opacity=0.15),
                tp_rings.animate.set_stroke(opacity=0.15),
            ),
            m.Write(f2),
        )
        self.say(
            "[Given simply means: shrink the crowd to the people it's true for], [then count.]",
            m.FadeIn(note, scale=1.1),
            m.AnimationGroup(people.sick.animate.set_fill(opacity=1), tp_rings.animate.set_stroke(opacity=1), m.Indicate(f2[4], color=INK, scale_factor=1.2)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ flip the question: counting

    def flip(self):
        self.next_section(
            "Flip the question",
            notes="Of the 48 who test positive, 8 are sick: P(sick | positive) = 8/48 ≈ 17%, about 1 in 6. Healthy outnumber sick 99 to 1, so 40 false alarms beat 8 real cases.",
        )
        people = self.people
        center = np.array([2.25, 0.55, 0])
        cols, step = 8, 0.3
        positives = m.VGroup(*people.tp, *people.fp)
        target = m.VGroup()
        for k, d in enumerate(positives):
            r, col = divmod(k, cols)
            spot = center + np.array([(col - (cols - 1) / 2) * step, (2.5 - r) * step, 0])
            twin = d.copy().move_to(spot).scale(1.9 if k < TP else 2.0)
            if k >= TP:
                twin.set_fill(MUTED, opacity=0.85)
            target.add(twin)
        frame = card(cols * step + 0.35, 6 * step + 0.35, AMBER, CARD, radius=0.2, stroke=2.5).move_to(center).set_z_index(-1)
        frame.set_fill(CARD, opacity=0.6)
        frame_label = text("48 tested positive", 24, AMBER, weight=m.BOLD).next_to(frame, m.UP, buff=0.18)
        counts = m.VGroup(
            text("8 sick", 24, PINK, weight=m.BOLD),
            text("40 healthy", 24, MUTED, weight=m.BOLD),
        ).arrange(m.DOWN, buff=0.5, aligned_edge=m.LEFT).next_to(frame, m.RIGHT, buff=0.35)
        counts[0].set_y(center[1] + 2.5 * step)
        counts[1].set_y(center[1] - 0.5 * step)
        f3 = formula(
            (LHS, GOOD), ("=", INK), (r"\frac{", INK), ("8", PINK), ("}{", INK), ("48", AMBER), ("}", INK), (r"\approx", INK), (r"17\%", GOOD), size=42
        ).move_to([2.75, -1.3, 0])
        answer = pill("about 1 in 6, not 80%", GOOD, GOOD_FILL, 24).move_to([2.75, -2.35, 0])
        ratio = pill("healthy : sick = 99 : 1", MUTED, CARD, 22).move_to([-3.75, -3.3, 0])
        swamp = pill("40 false alarms  vs  8 real cases", BAD, BAD_FILL, 22).move_to([2.75, -3.3, 0])
        self.f3 = f3

        self.say(
            "[Now flip the question.] [You tested positive. What's the chance you're sick?]",
            m.AnimationGroup(self.new_header("FLIP THE QUESTION", "You tested positive. Are you sick?"), m.FadeOut(self.cond_side)),
            m.LaggedStart(*[m.Indicate(r, color=AMBER, scale_factor=1.5) for r in self.rings], lag_ratio=0.02),
        )
        self.say(
            "[Gather everyone who tested positive]: [forty-eight people.]",
            m.AnimationGroup(m.Transform(positives, target, run_time=1.4), m.FadeOut(self.rings, run_time=0.6), m.FadeIn(frame)),
            m.AnimationGroup(m.FadeIn(frame_label, shift=0.15 * m.DOWN), m.Indicate(frame, color=AMBER, scale_factor=1.03)),
        )
        self.say(
            "[Only eight of them are sick]: [eight in forty-eight, about seventeen percent.] [Not eighty.]",
            m.AnimationGroup(m.LaggedStart(*[m.Indicate(d, color=PINK, scale_factor=1.6) for d in positives[:TP]], lag_ratio=0.1), m.FadeIn(counts, shift=0.2 * m.LEFT)),
            m.Write(f3),
            m.FadeIn(answer, scale=1.2),
        )
        self.say(
            "[Why so low?] [The healthy outnumber the sick ninety-nine to one], [so even a few false alarms outnumber the real cases.]",
            m.Wiggle(answer, scale_value=1.05),
            m.FadeIn(ratio, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(swamp, shift=0.2 * m.UP), m.LaggedStart(*[m.Indicate(d, color=BAD, scale_factor=1.5) for d in positives[TP:]], lag_ratio=0.02)),
        )
        self.wait(0.3)
        self.fade_all_but(f3)

    # ------------------------------------------------------------------ bayes' rule

    def bayes(self):
        self.next_section(
            "Bayes' rule",
            notes="8 = 1000 × 1% × 80%; 48 = 1000 × 4.8%; the 1000s cancel: P(sick|positive) = P(positive|sick) P(sick) / P(positive) "
            "= 80% × 1% / 4.8% ≈ 17%. Prior, likelihood, evidence, posterior. Bayes' essay: published 1763.",
        )
        s0, s1, s2, s3 = (chain(k).move_to([0, 1.85, 0]) for k in range(4))
        for s in (s1, s2, s3):  # keep the left-hand side where it is
            s.shift((s0[0].get_left()[0] - s[0].get_left()[0]) * m.RIGHT)
        strikes = m.VGroup(*[m.Line(p.get_corner(m.DL) + [-0.05, -0.02, 0], p.get_corner(m.UR) + [0.05, 0.02, 0]).set_stroke(BAD, width=4) for p in (s2[3], s2[9])])

        rows = {}
        rows["named"] = formula(
            (LHS, GOOD), ("=", INK), (r"\frac{", INK), (r"P(\text{positive} \mid \text{sick})", TEAL), (r"\, P(\text{sick})", PINK), ("}{", INK), (r"P(\text{positive})", AMBER), ("}", INK), size=44
        ).move_to([0, -0.75, 0])
        rows["general"] = formula(
            (r"P(A \mid B)", GOOD), ("=", INK), (r"\frac{", INK), (r"P(B \mid A)", TEAL), (r"\, P(A)", PINK), ("}{", INK), (r"P(B)", AMBER), ("}", INK), size=52
        ).move_to([0, -0.75, 0])
        named, general = rows["named"], rows["general"]

        def roles(f):
            lhs, like, prior, ev = f[0], f[3], f[4], f[6]
            return {
                "posterior": ("after the test", GOOD, lhs, m.DOWN, -1.85),
                "likelihood": ("how well the test spots the sick", TEAL, like, m.UP, None),
                "prior": ("before the test", PINK, prior, m.UP, None),
                "evidence": ("how often anyone tests positive", AMBER, ev, m.DOWN, -1.85),
            }

        def place(f):
            out = {}
            for key, (desc, color, part, side, y) in roles(f).items():
                lab = labelled(key, desc, color)
                if side is m.UP:
                    lab.next_to(f[3:5], m.UP, buff=0.18).set_x(part.get_x())
                else:
                    lab.move_to([part.get_x(), y, 0])
                out[key] = lab
            return out

        labels = place(named)
        later = place(general)
        # keep the two upper labels apart (the prior is narrow)
        gap = labels["prior"].get_left()[0] - labels["likelihood"].get_right()[0]
        if gap < 0.3:
            labels["prior"].shift((0.3 - gap) * m.RIGHT)
        gap = later["prior"].get_left()[0] - later["likelihood"].get_right()[0]
        if gap < 0.3:
            later["prior"].shift((0.3 - gap) * m.RIGHT)
        where = text("A: sick      B: positive", 24, MUTED, weight=m.SEMIBOLD).move_to([0, -3.0, 0])
        source = text("Bayes, 1763", 20, MUTED).next_to(where, m.DOWN, buff=0.15)

        f3 = self.f3
        self.flatten(f3)
        self.say(
            "[Where does the eight come from?] [One percent of a thousand people are sick], [and eighty percent of those test positive.]",
            m.AnimationGroup(
                self.new_header("BAYES' RULE", "The same answer, as a formula"),
                m.VGroup(*f3[:7]).animate.move_to(s0),
                m.FadeOut(m.VGroup(*f3[7:])),
            ),
            morph(f3[:7], s1),
            m.Indicate(s1[7], color=TEAL, scale_factor=1.3),
        )
        self.say(
            "[And forty-eight is everyone who tests positive]: [forty-eight in a thousand.]",
            m.Indicate(s1[9], color=AMBER, scale_factor=1.3),
            morph(s1, s2),
        )
        self.say(
            "[The thousands cancel], [leaving three probabilities.]",
            m.LaggedStart(*[m.Create(s) for s in strikes], lag_ratio=0.4),
            m.Succession(
                m.AnimationGroup(morph(s2, s3), m.FadeOut(strikes), run_time=1.2),
                m.AnimationGroup(
                    m.TransformFromCopy(s3[0], named[0]),
                    m.FadeIn(named[1:3]),
                    m.TransformFromCopy(s3[3], named[3]),
                    m.TransformFromCopy(s3[5], named[4]),
                    m.FadeIn(named[5]),
                    m.TransformFromCopy(s3[7], named[6]),
                    m.FadeIn(named[7]),
                    run_time=1.3,
                ),
            ),
        )
        self.say(
            "[One percent is the prior]: what we believed before the test. [Eighty percent is the likelihood]: how well the test spots the sick.",
            m.AnimationGroup(m.FadeIn(labels["prior"], shift=0.15 * m.DOWN), m.Indicate(s3[5], color=PINK, scale_factor=1.3), m.Indicate(named[4], color=PINK, scale_factor=1.15)),
            m.AnimationGroup(m.FadeIn(labels["likelihood"], shift=0.15 * m.DOWN), m.Indicate(s3[3], color=TEAL, scale_factor=1.3), m.Indicate(named[3], color=TEAL, scale_factor=1.1)),
        )
        self.say(
            "[Forty-eight in a thousand is the evidence]: how often anyone tests positive. [And seventeen percent is the posterior]: our belief after the test.",
            m.AnimationGroup(m.FadeIn(labels["evidence"], shift=0.15 * m.UP), m.Indicate(s3[7], color=AMBER, scale_factor=1.3), m.Indicate(named[6], color=AMBER, scale_factor=1.15)),
            m.AnimationGroup(m.FadeIn(labels["posterior"], shift=0.15 * m.UP), m.Indicate(s3[10], color=GOOD, scale_factor=1.4), m.Indicate(named[0], color=GOOD, scale_factor=1.1)),
        )
        key_map = {LHS: r"P(A \mid B)", r"P(\text{positive} \mid \text{sick})": r"P(B \mid A)", r"\, P(\text{sick})": r"\, P(A)", r"P(\text{positive})": r"P(B)"}
        self.say(
            "[Written for any two events], [this is Bayes' rule.]",
            m.AnimationGroup(
                morph(named, general, key_map),
                *[labels[k].animate.move_to(later[k]) for k in labels],
                m.FadeIn(where, shift=0.2 * m.UP),
                run_time=1.4,
            ),
            m.AnimationGroup(m.Circumscribe(general, color=AMBER, buff=0.2), m.FadeIn(source)),
        )
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------ why it matters

    def why_it_matters(self):
        self.next_section(
            "Why it matters",
            notes="With a 10% prior and the same test (80% caught, 40/990 false alarms), P(sick | positive) ≈ 69%. "
            "A classifier outputs probabilities, e.g. 17% sick / 83% healthy (illustrative); Episode 16 covers softmax.",
        )
        lbox = card(6.2, 4.2, AMBER, CARD, stroke=2).move_to([-3.35, -0.35, 0])
        rbox = card(6.2, 4.2, VIOLET, CARD, stroke=2).move_to([3.35, -0.35, 0])
        lx0, lx1 = -6.0, -0.7
        ltitle = text("The prior matters", 28, INK, weight=m.BOLD).move_to([0, 1.3, 0]).align_to([lx0, 0, 0], m.LEFT)
        prior = m.ValueTracker(P_SICK)

        def row(y, words, fn, color):
            name = text(words, 22, MUTED).move_to([0, y, 0]).align_to([lx0, 0, 0], m.LEFT)
            track = m.Rectangle(width=lx1 - lx0, height=0.32).set_stroke(EDGE, width=2).set_fill(BG, opacity=1)
            track.move_to([lx0, y - 0.48, 0], aligned_edge=m.LEFT)

            def make_bar():
                w = max(0.03, (lx1 - lx0) * fn())
                return m.Rectangle(width=w, height=0.32, stroke_width=0).set_fill(color, opacity=0.9).move_to([lx0, y - 0.48, 0], aligned_edge=m.LEFT)

            def make_value():
                return text(f"{100 * fn():.0f}%", 26, color, weight=m.BOLD).move_to([0, y, 0]).align_to([lx1, 0, 0], m.RIGHT)

            return name, track, m.always_redraw(make_bar), m.always_redraw(make_value)

        top = row(0.6, "how common the disease is", lambda: prior.get_value(), PINK)
        bottom = row(-0.6, "sick, given a positive test", lambda: posterior(prior.get_value()), GOOD)
        for part in (*top[2:], *bottom[2:]):
            part.set_z_index(2)
        footnote = text("the same test: catches 80%, false alarms 4%", 20, MUTED).move_to([0, -1.9, 0]).align_to([lx0, 0, 0], m.LEFT)
        left = m.VGroup(lbox, ltitle, *top, *bottom, footnote)

        rtitle = text("A classifier's answer", 28, INK, weight=m.BOLD).move_to([0, 1.3, 0]).align_to([0.7, 0, 0], m.LEFT)
        test = icon("ep14-test", 1.0).move_to([1.35, 0.15, 0])
        model = icon("llm", 1.0).move_to([3.15, 0.15, 0])
        links = m.VGroup(arrow(test.get_right(), model.get_left(), MUTED, buff=0.15))
        test_label = text("your test", 20, MUTED).next_to(test, m.DOWN, buff=0.15)
        model_label = text("classifier", 20, MUTED).next_to(model, m.DOWN, buff=0.15)
        base_y, full = -0.55, 1.2
        out = m.VGroup()
        for k, (word, p, color) in enumerate((("sick", P_SICK_POS, GOOD), ("healthy", 1 - P_SICK_POS, MUTED))):
            x = 4.85 + 0.95 * k
            b = m.Rectangle(width=0.55, height=full * p, stroke_width=0).set_fill(color, opacity=0.9).move_to([x, base_y, 0], aligned_edge=m.DOWN)
            v = text(f"{100 * p:.0f}%", 22, color, weight=m.BOLD).next_to(b, m.UP, buff=0.1)
            w = text(word, 20, MUTED).move_to([x, base_y - 0.25, 0])
            out.add(m.VGroup(b, v, w))
        links.add(arrow(model.get_right(), [out.get_left()[0] - 0.05, model.get_y(), 0], MUTED, buff=0.15))
        what = text("a probability for each label, given the input", 22, MUTED).move_to([3.35, -1.3, 0])
        later = pill("how: Episode 16 · Softmax", VIOLET, VIOLET_FILL, 22).move_to([3.35, -1.95, 0])

        self.say(
            "[Here's why the prior matters.] [Say the disease were common]: [one person in ten.]",
            m.AnimationGroup(self.new_header("WHY IT MATTERS", "The prior, and a classifier's answer"), m.FadeIn(left, shift=0.2 * m.UP)),
            m.Indicate(top[0], color=PINK, scale_factor=1.06),
            prior.animate(run_time=2.4).set_value(COMMON),
        )
        self.say(
            "[The very same positive test would then mean a sixty-nine percent chance], [not seventeen.]",
            m.AnimationGroup(m.Indicate(bottom[3], color=GOOD, scale_factor=1.25), m.Circumscribe(bottom[1], color=GOOD, buff=0.08)),
            m.Indicate(bottom[0], color=INK, scale_factor=1.04),
        )
        self.say(
            "[In machine learning, a classifier gives answers like these]: [not just sick or healthy, but a probability.]",
            m.AnimationGroup(
                prior.animate.set_value(P_SICK),
                m.FadeIn(rbox), m.FadeIn(rtitle),
                m.FadeIn(test, shift=0.2 * m.RIGHT), m.FadeIn(test_label),
                m.FadeIn(model, scale=0.8), m.FadeIn(model_label), m.GrowArrow(links[0]),
                run_time=1.4,
            ),
            m.AnimationGroup(m.GrowArrow(links[1]), m.LaggedStart(*[m.FadeIn(o, shift=0.2 * m.UP) for o in out], lag_ratio=0.3), m.FadeIn(what)),
        )
        self.say(
            "[Episode sixteen will show how a network turns its raw scores into probabilities.]",
            m.FadeIn(later, scale=1.15),
        )
        self.wait(0.3)
        self.stop(*top[2:], *bottom[2:])
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Distributions, expectation, Bayes' rule.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 46, c, weight=m.BOLD) for w, c in zip(("Distributions", "Expectation", "Bayes' rule"), (PINK, LIME, GOOD))]).arrange(m.RIGHT, buff=0.95)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.22) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Put a number on uncertainty, then let the evidence update it.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [distributions], [expectation], [and Bayes' rule.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Put a number on your uncertainty, then let the evidence update it.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
