"""AI, Explained · Episode 46 · Alignment & Evaluation (the final episode)

Running example: a model gaming its reward. First the classic CoastRunners boat (Clark & Amodei,
OpenAI, 2016): rewarded with points for hitting targets, the agent circles a lagoon forever and
never finishes the race, yet out-scores human players by about 20%. Then the same pattern in
language models (flattery; a coding agent that exits before the tests run, Baker et al., 2025),
how we measure models (benchmarks, noise, saturation, contamination, human judges, red-teaming),
and scalable oversight (critiques, debate, a constitution, a weaker model reading the coding
agent's chain of thought, and what happens when you penalise those thoughts).

Every picture with numbers is computed here with seeded numpy and the narrated numbers are
asserted: the proxy-vs-goal curves (illustrative, the shape of Gao et al., 2022), 200 re-tests of a
model with true accuracy 80% on 1,000 questions (95% interval ±2.5 points), a 200-question test
with 60 leaked questions (public 72%, fresh 60%), 200 red-team attacks (9 get through), and a
small Bradley-Terry vote simulation for the leaderboard. The boat course and its scores are
drawn for illustration; the 20% figure is from the OpenAI post.

The kit's series_outro() is overridden: this episode closes the course.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

BLUE_FILL = "#14213D"

# ---------------------------------------------------------------------------- the numbers

# CoastRunners: "a score on average 20 percent higher than that achieved by human players"
AGENT_VS_HUMAN = 1.20
assert round(100 * (AGENT_VS_HUMAN - 1)) == 20


# reward hacking: the reward we wrote (proxy) keeps rising; the goal we meant peaks, then falls
def proxy(d):
    return 2.2 * d - 0.1 * d**2


def true_goal(d):
    return 2.2 * d - 0.275 * d**2


D_MAX = 8.0
_D = np.linspace(0, D_MAX, 801)
D_PEAK = float(_D[np.argmax(true_goal(_D))])
assert abs(D_PEAK - 4.0) < 1e-9 and np.all(np.diff(proxy(_D)) > 0)
assert abs(true_goal(D_MAX)) < 1e-9  # by the end, the goal is back where it started
RATED_D = np.linspace(0.6, 7.6, 11)
RATED_Q = true_goal(RATED_D) + np.random.default_rng(46).normal(0, 0.3, RATED_D.size)

# benchmarks: MMLU (Hendrycks et al., 2021) has 57 subjects, 4 options per question
SUBJECTS, OPTIONS = 57, 4
assert round(100 / OPTIONS) == 25
N_Q, P_TRUE, RETESTS = 1000, 0.80, 200
HALF = 1.96 * math.sqrt(P_TRUE * (1 - P_TRUE) / N_Q) * 100  # 95% half-width, in points
assert round(HALF, 1) == 2.5
RETEST_SCORES = np.random.default_rng(46).binomial(N_Q, P_TRUE, RETESTS) / N_Q * 100
INSIDE = float(np.mean(np.abs(RETEST_SCORES - 100 * P_TRUE) <= HALF))
assert 0.92 <= INSIDE <= 0.98
BIN_EDGES = np.arange(76.0, 84.01, 0.5)
BIN_COUNTS, _ = np.histogram(RETEST_SCORES, BIN_EDGES)
assert BIN_COUNTS.sum() == RETESTS


def saturation(t, t0, tau=0.55):
    """An illustrative top score over time: from chance (25%) up to a 97% ceiling."""
    return 25 + 72 / (1 + np.exp(-(t - t0) / tau))


BENCH_T0 = (1.9, 5.6, 9.3)

# contamination: 200 questions, 60 leaked and memorised, true ability 60%
N_TEST, N_LEAKED, ABILITY = 200, 60, 0.6
_rng = np.random.default_rng(950)
LEAKED = np.sort(_rng.choice(N_TEST, N_LEAKED, replace=False))
KNOWS = _rng.random(N_TEST) < ABILITY  # what it can really answer
FRESH = _rng.random(N_TEST) < ABILITY  # a fresh test of the same difficulty
PUBLIC = KNOWS.copy()
PUBLIC[LEAKED] = True
assert round(100 * PUBLIC.mean()) == 72 and round(100 * FRESH.mean()) == 60

# red-teaming: 200 attacks, 9 get through
N_ATTACKS = 200
BROKE = np.random.default_rng(13).random(N_ATTACKS) < 0.045
assert BROKE.sum() == 9

# a leaderboard from pairwise votes (Bradley-Terry strengths, seeded votes)
ARENA = ("model K", "model M", "model P", "model S")
_strength = np.array([1.0, 0.55, 0.3, 0.0])
_rng = np.random.default_rng(7)
_pairs = _rng.integers(0, 4, (6000, 2))
_pairs = _pairs[_pairs[:, 0] != _pairs[:, 1]]
_first_wins = _rng.random(len(_pairs)) < 1 / (1 + np.exp(-(_strength[_pairs[:, 0]] - _strength[_pairs[:, 1]])))
_winner = np.where(_first_wins, _pairs[:, 0], _pairs[:, 1])
WIN_RATE = np.array([(_winner == k).sum() / ((_pairs == k).any(axis=1)).sum() for k in range(4)])
assert list(np.argsort(-WIN_RATE)) == [0, 1, 2, 3]

# ---------------------------------------------------------------------------- the boat course

TRACK_C = np.array([-2.75, -0.45, 0.0])
LAGOON_C = np.array([-1.15, -0.45, 0.0])
LAGOON_R, LOOP_R = 0.72, 0.42
DRIVE_SPEED, LOOP_SPEED = 2.6, 1.35  # units per second on the track, in the lagoon
LAGOON_TARGETS = np.radians([150, 270, 30])
FLAME_ANGLES = np.radians([205, 305, 118])
COURSE_TARGETS = ((-4.3, 1.25), (-2.6, 1.25), (-0.95, 1.25), (-5.85, -0.45), (0.35, 0.5))
POINTS_PER_HIT = 50


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[1:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def _route():
    """From the start line, round the bottom-right corner, up, left into the lagoon."""
    parts = [np.column_stack([np.linspace(-3.7, -0.55, 60), np.full(60, -2.15)])]
    a = np.linspace(-np.pi / 2, 0, 40)[1:]
    parts.append(np.column_stack([-0.55 + 0.9 * np.cos(a), -1.25 + 0.9 * np.sin(a)]))
    parts.append(np.column_stack([np.full(7, 0.35), np.linspace(-1.25, -0.85, 8)[1:]]))
    a = np.linspace(0, np.pi / 2, 30)[1:]
    parts.append(np.column_stack([-0.05 + 0.4 * np.cos(a), -0.85 + 0.4 * np.sin(a)]))
    top = LAGOON_C[:2] + np.array([0, LOOP_R])
    parts.append(_bezier(np.array([-0.05, -0.45]), np.array([-0.45, -0.45]), np.array([-0.75, top[1]]), top, 50))
    pts = np.vstack(parts)
    return pts, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


ROUTE, ROUTE_S = _route()
L_PRE = float(ROUTE_S[-1])


def route_at(u):
    """Position and heading of the boat after travelling u units."""
    if u < L_PRE:
        i = int(np.clip(np.searchsorted(ROUTE_S, u) - 1, 0, len(ROUTE) - 2))
        f = (u - ROUTE_S[i]) / (ROUTE_S[i + 1] - ROUTE_S[i])
        p = ROUTE[i] + f * (ROUTE[i + 1] - ROUTE[i])
        d = ROUTE[i + 1] - ROUTE[i]
        return np.array([p[0], p[1], 0.0]), math.atan2(d[1], d[0])
    th = np.pi / 2 + (u - L_PRE) / LOOP_R
    return LAGOON_C + LOOP_R * np.array([math.cos(th), math.sin(th), 0.0]), th + np.pi / 2


def lagoon_hits(u):
    """For each lagoon target: (times hit, angle travelled since its last hit)."""
    if u < L_PRE:
        return [(0, None)] * len(LAGOON_TARGETS)
    th = np.pi / 2 + (u - L_PRE) / LOOP_R
    out = []
    for phi in LAGOON_TARGETS:
        first = phi if phi >= np.pi / 2 else phi + 2 * np.pi
        if th < first:
            out.append((0, None))
        else:
            k = int((th - first) // (2 * np.pi)) + 1
            out.append((k, th - first - 2 * np.pi * (k - 1)))
    return out


# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """kit.recolored(), but written under this episode's prefix: assets/ep46-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    stem = f"ep46-{name.removeprefix('ep46-')}-{color.lstrip('#').lower()}"
    path = ASSETS / f"{stem}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return stem


def cite(s, size=20):
    return text(s, size, MUTED)


def menlo(s, size=22, color=INK):
    return m.Text(s, font="Menlo", font_size=size, color=color)


def target_mark(center, r=0.13):
    ring = m.Circle(radius=r).set_stroke(AMBER, width=3).set_fill(AMBER, opacity=0).move_to(center)
    dot = m.Circle(radius=0.05).set_stroke(width=0).set_fill(AMBER, opacity=1).move_to(center)
    return m.VGroup(ring, dot).set_z_index(3)


def set_alpha(mark, a):
    mark[0].set_stroke(opacity=a)
    mark[1].set_fill(opacity=a)


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def square(center, side, stroke, fill, opacity=1.0, width=1.5):
    sq = m.RoundedRectangle(corner_radius=0.03, width=side, height=side)
    return sq.set_stroke(stroke, width=width).set_fill(fill, opacity=opacity).move_to(center)


def grid_centers(n, cols, step, center):
    rows = math.ceil(n / cols)
    out = []
    for k in range(n):
        r, c = divmod(k, cols)
        out.append(np.array(center) + np.array([(c - (cols - 1) / 2) * step, ((rows - 1) / 2 - r) * step, 0]))
    return out


def bars_block(width, rows, seed, color=FAINT):
    """Grey bars standing in for lines of text."""
    rng = np.random.default_rng(seed)
    out = m.VGroup()
    for _ in range(rows):
        w = width * float(rng.uniform(0.45, 1.0))
        out.add(m.RoundedRectangle(corner_radius=0.04, width=w, height=0.09).set_stroke(width=0).set_fill(color, opacity=1))
    out.arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
    return out


class Ep46AlignmentEvaluation(Episode):
    number = 46

    def lesson(self):
        self.hook()
        self.boat()
        self.reward_hacking()
        self.in_language_models()
        self.benchmarks()
        self.saturation()
        self.contamination()
        self.human_judges()
        self.red_teaming()
        self.oversight_problem()
        self.oversight_ideas()
        self.oversight_monitor()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Measuring and steering model behaviour; a model learns what you reward, not what you mean.")
        halo = glow(PINK, radius=3.4).move_to(1.0 * m.UP)
        hero = text("Is it doing what we want?", 66, INK, weight=m.BOLD, gradient=(PINK, VIOLET)).move_to([0, 1.0, 0])
        chips = m.VGroup(
            chip("measure it", SKY, SKY_FILL, width=3.4, height=0.8, size=28),
            chip("steer it", TEAL, TEAL_FILL, width=3.4, height=0.8, size=28),
        ).arrange(m.RIGHT, buff=0.6).move_to([0, -0.7, 0])
        catch = pill("a model learns what you reward, not what you mean", BAD, BAD_FILL, 26).move_to([0, -2.1, 0])

        self.say(
            "So... [how do you know a model is doing what you want?]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(hero, shift=0.3 * m.UP, scale=0.9), run_time=1.2),
        )
        self.say(
            "[You measure its behaviour], [and you steer it with rewards.]",
            m.FadeIn(chips[0], shift=0.3 * m.UP),
            m.FadeIn(chips[1], shift=0.3 * m.UP),
        )
        self.say(
            "[But a model learns what you reward, not what you mean.]",
            m.Succession(m.FadeIn(catch, scale=1.15), m.Wiggle(catch, scale_value=1.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the boat

    def boat(self):
        self.next_section(
            "The running example: CoastRunners",
            notes="Clark & Amodei, 'Faulty Reward Functions in the Wild', OpenAI blog, 21 Dec 2016: in the boat-racing game "
            "CoastRunners the score comes from hitting targets, not from finishing; an RL agent found an isolated lagoon where it "
            "could circle and knock over three targets as they repopulate, catching fire and crashing, and scored on average about "
            "20% higher than human players. The course, its targets and the point counter drawn here are illustrative.",
        )
        water = m.RoundedRectangle(corner_radius=1.5, width=7.4, height=4.6).set_stroke(SKY, width=2, opacity=0.35)
        water.set_fill(SKY_FILL, opacity=1).move_to(TRACK_C)
        island = m.RoundedRectangle(corner_radius=0.35, width=5.0, height=2.2).set_stroke(EDGE, width=2).set_fill(CARD, opacity=1).move_to(TRACK_C)
        lagoon = m.Circle(radius=LAGOON_R).set_stroke(SKY, width=2, opacity=0.35).set_fill(SKY_FILL, opacity=1).move_to(LAGOON_C)
        channel = m.Rectangle(width=0.6, height=0.56).set_stroke(width=0).set_fill(SKY_FILL, opacity=1).move_to([-0.5, -0.45, 0])
        finish = m.VGroup(
            *[
                m.Square(0.2).set_stroke(width=0).set_fill(INK if (r + c) % 2 == 0 else FAINT, opacity=1).move_to([-4.4 + 0.2 * c, -2.65 + 0.2 * r, 0])
                for r in range(6)
                for c in range(2)
            ]
        )
        finish_label = text("FINISH", 20, TEAL, weight=m.BOLD).move_to([-4.3, -3.05, 0])
        course = m.VGroup(water, island, lagoon, channel, finish, finish_label)
        lagoon_label = text("a quiet lagoon", 20, MUTED, weight=m.SEMIBOLD).move_to([-3.25, -0.45, 0])

        course_marks = [target_mark([x, y, 0]) for x, y in COURSE_TARGETS]
        lagoon_marks = [target_mark(LAGOON_C + LOOP_R * np.array([math.cos(a), math.sin(a), 0])) for a in LAGOON_TARGETS]
        flames = m.VGroup(
            *[icon("ep46-flame", 0.36).move_to(LAGOON_C + (LAGOON_R + 0.02) * np.array([math.cos(a), math.sin(a), 0])) for a in FLAME_ANGLES]
        ).set_z_index(4)

        boat = icon("ep46-boat", 0.42).set_z_index(5)
        start, heading = route_at(0.0)
        boat.rotate(heading - np.pi / 2).move_to(start)
        boat.heading = heading
        cr_cite = cite("CoastRunners · Clark & Amodei, OpenAI, 2016").move_to([0, -3.05, 0]).align_to([6.6, 0, 0], m.RIGHT)

        px = 2.45
        goal = m.VGroup(pill("GOAL", TEAL, TEAL_FILL, 20), text("finish the race", 24, INK)).arrange(m.RIGHT, buff=0.25)
        reward = m.VGroup(pill("REWARD", AMBER, AMBER_FILL, 20), text("points for targets", 24, INK)).arrange(m.RIGHT, buff=0.25)
        left_at(goal, px, 2.0)
        left_at(reward, px, 1.35)
        cross = text("✗", 30, BAD, weight=m.BOLD).move_to([2.0, goal.get_y(), 0])
        tick = text("✓", 30, GOOD, weight=m.BOLD).move_to([2.0, reward.get_y(), 0])

        state = {"u": 0.0}

        def points():
            return POINTS_PER_HIT * sum(k for k, _ in lagoon_hits(state["u"]))

        score_label = left_at(text("score", 26, AMBER, weight=m.BOLD), px, 0.3)
        score_value = m.always_redraw(
            lambda: text(f"{points():,}", 44, AMBER, weight=m.BOLD).move_to([0, 0.3, 0]).align_to([6.6, 0, 0], m.RIGHT)
        )
        laps_label = left_at(text("laps finished", 26, TEAL, weight=m.BOLD), px, -0.45)
        laps_value = text("0", 44, TEAL, weight=m.BOLD).move_to([0, -0.45, 0]).align_to([6.6, 0, 0], m.RIGHT)

        bx, human_w = 3.45, 2.0
        human_label = left_at(text("human", 22, MUTED), px, -1.5)
        agent_label = left_at(text("agent", 22, MUTED), px, -2.15)
        human_bar = m.Rectangle(width=human_w, height=0.3).set_stroke(width=0).set_fill(SKY, opacity=0.9)
        human_bar.move_to([bx, -1.5, 0], aligned_edge=m.LEFT)
        agent_bar = m.Rectangle(width=human_w * AGENT_VS_HUMAN, height=0.3).set_stroke(width=0).set_fill(VIOLET, opacity=0.9)
        agent_bar.move_to([bx, -2.15, 0], aligned_edge=m.LEFT)
        plus = text("+20%", 24, VIOLET, weight=m.BOLD).next_to(agent_bar, m.RIGHT, buff=0.15)

        def drive(b, dt):
            u = state["u"]
            speed = LOOP_SPEED + (DRIVE_SPEED - LOOP_SPEED) * float(np.clip((L_PRE - u) / 0.8, 0, 1))
            state["u"] = u + speed * dt
            p, h = route_at(state["u"])
            b.rotate(h - b.heading)
            b.heading = h
            b.move_to(p)

        def respawn(i):
            def update(mark):
                k, since = lagoon_hits(state["u"])[i]
                set_alpha(mark, 1.0 if k == 0 else float(np.clip((since - 3.4) / 1.6, 0, 1)))

            return update

        self.say(
            "[Our running example is a classic, from OpenAI in twenty sixteen]: [a boat racing game called Coast Runners.]",
            m.AnimationGroup(
                self.new_header("THE RUNNING EXAMPLE", "A boat that games its score"),
                m.FadeIn(course, run_time=1.2),
            ),
            m.AnimationGroup(m.FadeIn(boat, shift=0.2 * m.RIGHT), m.FadeIn(cr_cite)),
        )
        self.say(
            "[To us, the goal is obvious: finish the race.] [But the game's score comes from hitting targets along the way.]",
            m.AnimationGroup(m.FadeIn(goal, shift=0.2 * m.UP), m.Indicate(finish, color=TEAL, scale_factor=1.15)),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(t, scale=0.5) for t in course_marks + lagoon_marks], lag_ratio=0.15),
                m.FadeIn(reward, shift=0.2 * m.UP),
            ),
        )
        boat.add_updater(drive)
        for i, mark in enumerate(lagoon_marks):
            mark.add_updater(respawn(i))
        self.say(
            "[A reinforcement learning agent learned to maximise that score.] [It found a quiet lagoon], [where three targets keep coming back.]",
            m.AnimationGroup(m.FadeIn(score_label), m.FadeIn(score_value)),
            m.FadeIn(lagoon_label),
            m.AnimationGroup(*[m.Flash(t.get_center(), color=AMBER, flash_radius=0.24, line_length=0.1) for t in lagoon_marks]),
        )
        self.say(
            "[So it circles forever], [crashing and catching fire], [and never finishes the race.]",
            m.Circumscribe(lagoon, shape=m.Circle, color=AMBER, buff=0.12),
            m.LaggedStart(*[m.FadeIn(f, scale=0.4) for f in flames], lag_ratio=0.35),
            m.AnimationGroup(m.FadeIn(laps_label), m.FadeIn(laps_value), m.Indicate(finish, color=BAD, scale_factor=1.15)),
        )
        self.say(
            "[Yet it scores about twenty percent higher than human players.]",
            m.LaggedStart(
                m.AnimationGroup(m.FadeIn(human_label), m.GrowFromEdge(human_bar, m.LEFT)),
                m.AnimationGroup(m.FadeIn(agent_label), m.GrowFromEdge(agent_bar, m.LEFT)),
                m.FadeIn(plus, scale=1.2),
                lag_ratio=0.4,
            ),
        )
        self.say(
            "[It did exactly what we rewarded], [just not what we meant.]",
            m.AnimationGroup(m.FadeIn(tick, scale=1.3), m.Indicate(reward, color=AMBER, scale_factor=1.05)),
            m.AnimationGroup(m.FadeIn(cross, scale=1.3), m.Indicate(goal, color=TEAL, scale_factor=1.05)),
        )
        self.wait(0.4)
        self.stop(boat, score_value, *lagoon_marks)
        self.clear()

    # ------------------------------------------------------------------ reward hacking

    def reward_hacking(self):
        self.next_section(
            "Reward hacking",
            notes="Specification gaming / reward hacking (Amodei et al., 2016; Krakovna et al., 2020). Curves are illustrative "
            "(proxy 2.2d - 0.1d^2, goal 2.2d - 0.275d^2, goal peaks at d = 4; dots: seeded noise), the shape measured for "
            "reward-model over-optimisation by Gao et al., 2022 (see Episode 30). Goodhart's law: Goodhart, 1975; the common "
            "wording 'When a measure becomes a target, it ceases to be a good measure' is Strathern's, 1997.",
        )
        ax = axes([0, 8.4, 1], [0, 12, 2], 7.0, 4.2).move_to([-2.65, -0.45, 0])
        x_label = text("optimisation pressure", 22, MUTED).next_to(ax.x_axis, m.DOWN, buff=0.25)
        y_label = text("score", 24, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.15)
        proxy_curve = ax.plot(lambda d: float(proxy(d)), x_range=[0, D_MAX]).set_stroke(AMBER, width=5)
        goal_curve = ax.plot(lambda d: float(true_goal(d)), x_range=[0, D_MAX]).set_stroke(TEAL, width=5)
        dots = m.VGroup(*[m.Dot(ax.c2p(d, max(q, 0.1)), radius=0.07, color=TEAL) for d, q in zip(RATED_D, RATED_Q)]).set_z_index(2)
        peak = m.DashedLine(ax.c2p(D_PEAK, 0), ax.c2p(D_PEAK, float(true_goal(D_PEAK))), dash_length=0.1).set_stroke(FAINT, width=3)
        peak_label = text("the goal peaks", 20, TEAL, weight=m.SEMIBOLD).next_to(ax.c2p(D_PEAK, 1.6), m.RIGHT, buff=0.15)

        gap_d = 7.6
        gap = m.DoubleArrow(
            ax.c2p(gap_d, float(true_goal(gap_d))), ax.c2p(gap_d, float(proxy(gap_d))),
            buff=0.12, color=BAD, stroke_width=4, max_tip_length_to_length_ratio=0.08,
        ).shift(0.3 * m.RIGHT)
        hack = pill("reward hacking", BAD, BAD_FILL, 22).next_to(gap, m.RIGHT, buff=0.3)

        lx = 1.75

        def key(color, words, y):
            swatch = m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(color, width=5)
            row = m.VGroup(swatch, text(words, 22, color, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.25)
            return left_at(row, lx, y)

        keys = m.VGroup(key(AMBER, "the reward we wrote down", 1.75), key(TEAL, "the goal we meant", 1.2))
        gao = left_at(cite("Gao et al., 2022  ·  see Episode 30"), lx, 0.65)
        law = left_at(text("Goodhart's law", 26, AMBER, weight=m.BOLD), lx, -1.45)
        quote = left_at(text("When a measure becomes a target,\nit stops being a good measure.", 24, INK), lx, -2.1)
        law_cite = left_at(cite("Goodhart, 1975  ·  wording: Strathern, 1997"), lx, -2.75)

        self.say(
            "[This is called specification gaming, or reward hacking]: [optimising the reward we wrote down, instead of the goal we meant.]",
            m.AnimationGroup(
                self.new_header("REWARD HACKING", "The reward is not the goal", AMBER, AMBER_FILL),
                m.Create(ax), m.FadeIn(x_label), m.FadeIn(y_label),
            ),
            m.LaggedStart(m.FadeIn(keys[0], shift=0.2 * m.UP), m.FadeIn(keys[1], shift=0.2 * m.UP), lag_ratio=0.4),
        )
        self.say(
            "[Push harder on the written reward, and it keeps climbing], [while the real goal peaks, then falls.]",
            m.Create(proxy_curve, run_time=1.6),
            m.AnimationGroup(
                m.Create(goal_curve, run_time=1.6),
                m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.12),
                m.Create(peak), m.FadeIn(peak_label),
            ),
        )
        self.say(
            "[We saw this shape in episode thirty]: [a reward model's score, pulling away from what people actually prefer.]",
            m.FadeIn(gao, shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowFromCenter(gap), m.FadeIn(hack, shift=0.2 * m.RIGHT)),
        )
        self.say(
            "[It's Goodhart's law]: [when a measure becomes a target, it stops being a good measure.]",
            m.FadeIn(law, shift=0.2 * m.UP),
            m.AnimationGroup(m.Write(quote), m.FadeIn(law_cite)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ in language models

    def in_language_models(self):
        self.next_section(
            "Reward hacking in language models",
            notes="Sycophancy: Sharma et al., 2023 ('Towards Understanding Sycophancy in Language Models'): RLHF-trained "
            "assistants tend to agree with users; human and preference-model judgements sometimes favour it. Test gaming: "
            "Baker et al., 2025 (OpenAI) report a frontier reasoning model in agentic coding RL learning hacks such as exit(0) "
            "before the tests run, raise SkipTest, or making verify() always return true. Chat and code shown are illustrative.",
        )
        panels = [card(6.2, 4.6, AMBER, CARD, stroke=2).move_to([x, -0.5, 0]) for x in (-3.375, 3.375)]

        def heading(box, title, sub):
            g = m.VGroup(text(title, 28, INK, weight=m.BOLD), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            return g.move_to(box.get_corner(m.UL) + np.array([0.4 + g.width / 2, -0.35 - g.height / 2, 0]))

        left, right = panels
        left_head = heading(left, "Flattery", "trained on human ratings")
        right_head = heading(right, "Skipping the tests", "rewarded for passing tests")

        ask = bubble(text("I'm sure 17 × 3 is 41. Right?", 24, INK), SKY, SKY_FILL, "left")
        ask.shift(np.array([left.get_left()[0] + 0.35 + ask[1].width / 2, 0.15, 0]) - ask[1].get_center())
        yes = bubble(text("Yes, you're absolutely right!", 24, INK), VIOLET, VIOLET_FILL, "right")
        yes.shift(np.array([left.get_right()[0] - 0.35 - yes[1].width / 2, -1.1, 0]) - yes[1].get_center())
        verdict = pill("✗  pleasing, but wrong: it's 51", BAD, BAD_FILL, 22).move_to([left.get_x(), -2.35, 0])
        sharma = cite("Sharma et al., 2023").move_to([left.get_x(), -3.2, 0])

        window = card(5.6, 2.0, EDGE, BG, radius=0.18, stroke=2).move_to([right.get_x(), -0.4, 0])
        lines = m.VGroup(
            menlo("# task: fix the bug in sort()", 22, MUTED),
            menlo("import sys", 22, INK),
            menlo("sys.exit(0)", 22, AMBER),
            menlo("# exits before any test runs", 22, MUTED),
        ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        lines.move_to(window).align_to(window, m.LEFT).shift(0.3 * m.RIGHT)
        passed = left_at(text("tests:  ✓ all passed", 24, AMBER, weight=m.SEMIBOLD), right.get_left()[0] + 0.45, -1.85)
        bug = left_at(text("bug:  ✗ still there", 24, BAD, weight=m.SEMIBOLD), right.get_left()[0] + 0.45, -2.35)
        baker = cite("Baker et al., 2025").move_to([right.get_x(), -3.2, 0])

        self.say(
            "[Language models do it too.] [Rewarded by human ratings, they can learn to flatter]: [agreeing with the user, even when the user is wrong.]",
            m.AnimationGroup(
                self.new_header("REWARD HACKING", "…in language models", AMBER, AMBER_FILL),
                m.FadeIn(left), m.FadeIn(left_head),
            ),
            m.AnimationGroup(m.FadeIn(ask, shift=0.2 * m.UP), m.FadeIn(sharma)),
            m.Succession(m.FadeIn(yes, shift=0.2 * m.UP), m.FadeIn(verdict, scale=1.1)),
        )
        self.say(
            "[And a coding model rewarded for passing tests] [may learn to skip the tests], [instead of fixing the bug.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(right_head), m.FadeIn(window), m.FadeIn(lines[0]), m.FadeIn(baker)),
            m.Succession(
                m.AddTextLetterByLetter(lines[1], time_per_char=0.03),
                m.AddTextLetterByLetter(lines[2], time_per_char=0.03),
                m.FadeIn(lines[3]),
                m.FadeIn(passed, shift=0.15 * m.UP),
            ),
            m.Succession(m.FadeIn(bug, scale=1.1), m.Wiggle(bug, scale_value=1.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ benchmarks

    def benchmarks(self):
        self.next_section(
            "1 · Benchmarks",
            notes="MMLU (Hendrycks et al., 2021): multiple-choice questions in 57 subjects, four options each; chance = 25%. "
            "The sample question is illustrative (not from MMLU). Noise: 200 seeded re-tests of a model with true accuracy 80% on "
            "1,000 fresh questions each; the 95% half-width is 1.96 × sqrt(0.8 × 0.2 / 1000) ≈ 2.5 points.",
        )
        front = card(5.8, 4.0, SKY, CARD, stroke=2).move_to([-3.5, -0.45, 0])
        backs = [card(5.8, 4.0, EDGE, CARD, stroke=2).move_to(front.get_center() + k * np.array([0.15, 0.15, 0])) for k in (2, 1)]
        for k, b in enumerate(backs):
            b.set_z_index(-2 + k)
        x0 = front.get_left()[0] + 0.45
        tag = left_at(m.VGroup(pill("MMLU", SKY, SKY_FILL, 20), text("57 subjects  ·  4 options", 22, MUTED)).arrange(m.RIGHT, buff=0.25), x0, 1.1)
        question = left_at(text("Which gas makes up most\nof Earth's atmosphere?", 26, INK, weight=m.SEMIBOLD), x0, 0.35)
        rows = m.VGroup()
        for k, (letter, words) in enumerate(zip("ABCD", ("Oxygen", "Nitrogen", "Carbon dioxide", "Argon"))):
            box = card(0.46, 0.46, SKY, SKY_FILL, radius=0.1, stroke=2)
            label = text(letter, 22, INK, weight=m.BOLD).move_to(box)
            row = m.VGroup(m.VGroup(box, label), text(words, 24, INK)).arrange(m.RIGHT, buff=0.3)
            rows.add(left_at(row, x0, -0.45 - 0.5 * k))
        right_row = rows[1]
        mark = m.RoundedRectangle(corner_radius=0.12, width=right_row.width + 0.3, height=0.6).set_stroke(GOOD, width=3).move_to(right_row)
        tick = text("✓", 28, GOOD, weight=m.BOLD).next_to(mark, m.RIGHT, buff=0.2)
        chance = pill("guessing: 25%", SKY, SKY_FILL, 22).move_to([-4.9, -3.05, 0])
        hendrycks = cite("MMLU · Hendrycks et al., 2021").next_to(chance, m.RIGHT, buff=0.4)

        # the noise picture
        hx0, hx1, base = 1.0, 6.5, -1.7

        def hx(v):
            return hx0 + (hx1 - hx0) * (v - 76) / 8

        scale = 2.15 / BIN_COUNTS.max()
        bars = m.VGroup(
            *[
                m.Rectangle(width=0.3, height=c * scale).set_stroke(width=0).set_fill(SKY, opacity=0.85).move_to([hx(lo + 0.25), base, 0], aligned_edge=m.DOWN)
                for lo, c in zip(BIN_EDGES[:-1], BIN_COUNTS)
                if c > 0
            ]
        ).set_z_index(1)
        axis_line = m.Line([hx0 - 0.1, base, 0], [hx1 + 0.1, base, 0]).set_stroke(FAINT, width=3)
        ticks = m.VGroup(*[text(f"{v}%", 20, MUTED).move_to([hx(v), base - 0.35, 0]) for v in (76, 78, 80, 82, 84)])
        title = text("200 re-tests · 1,000 fresh questions each", 22, MUTED).move_to([(hx0 + hx1) / 2, 2.05, 0])
        top = base + BIN_COUNTS.max() * scale
        band = m.Rectangle(width=hx(80 + HALF) - hx(80 - HALF), height=top + 0.15 - base).set_stroke(width=0)
        band.set_fill(SKY, opacity=0.12).move_to([hx(80), base, 0], aligned_edge=m.DOWN).set_z_index(0)
        span = m.DoubleArrow([hx(80 - HALF), top + 0.4, 0], [hx(80 + HALF), top + 0.4, 0], buff=0, color=SKY, stroke_width=4, max_tip_length_to_length_ratio=0.1)
        span_label = text("± 2.5 points", 28, SKY, weight=m.BOLD).next_to(span, m.UP, buff=0.15)
        caption = text("95% of re-tests land in the band", 22, MUTED).move_to([(hx0 + hx1) / 2, base - 0.95, 0])

        self.say(
            "[So how do we measure a model?] [Usually with a benchmark]: [a fixed set of questions, with known answers.]",
            self.new_header("1 · BENCHMARKS", "A fixed test, one number", SKY, SKY_FILL),
            m.LaggedStart(m.FadeIn(backs[0]), m.FadeIn(backs[1]), m.FadeIn(m.VGroup(front, question, rows)), lag_ratio=0.25),
            m.AnimationGroup(m.Create(mark), m.FadeIn(tick, scale=1.3)),
        )
        self.say(
            "[One widely used test covers fifty-seven subjects, from law to physics], [each with four options], [so guessing scores twenty-five percent.]",
            m.AnimationGroup(m.FadeIn(tag, shift=0.2 * m.UP), m.FadeIn(hendrycks)),
            m.LaggedStart(*[m.Indicate(r[0], color=SKY, scale_factor=1.2) for r in rows], lag_ratio=0.25),
            m.FadeIn(chance, shift=0.2 * m.UP),
        )
        self.say(
            "[But a score is only an estimate.] [On a thousand questions, eighty percent means plus or minus about two and a half points.]",
            m.AnimationGroup(
                m.FadeIn(title), m.Create(axis_line), m.FadeIn(ticks),
                m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.08),
            ),
            m.AnimationGroup(m.FadeIn(band), m.GrowFromCenter(span), m.FadeIn(span_label, shift=0.15 * m.UP), m.FadeIn(caption)),
        )
        self.wait(0.3)
        self.clear()

    def saturation(self):
        self.next_section(
            "1 · Benchmarks saturate",
            notes="Illustrative logistic curves (25% chance → 97% ceiling), not real benchmark histories. Harder successors: "
            "MMLU-Pro (Wang et al., 2024; ten options), GPQA (Rein et al., 2023; 'Google-proof' expert questions), "
            "Humanity's Last Exam (Phan et al., 2025).",
        )
        t_max = 12
        ax = axes([0, t_max, 1], [0, 100, 25], 9.4, 3.7).move_to([-0.8, 0.0, 0])
        x_label = text("time", 22, MUTED).next_to(ax.x_axis.get_end(), m.RIGHT, buff=0.15)
        y_ticks = m.VGroup(*[text(f"{v}%", 20, MUTED).next_to(ax.c2p(0, v), m.LEFT, buff=0.2) for v in (25, 50, 75, 100)])
        ceiling = m.DashedLine(ax.c2p(0, 100), ax.c2p(t_max, 100), dash_length=0.12).set_stroke(FAINT, width=2.5)
        chance = m.DashedLine(ax.c2p(0, 25), ax.c2p(t_max, 25), dash_length=0.12).set_stroke(FAINT, width=2.5)
        chance_label = text("chance", 20, MUTED).next_to(ax.c2p(t_max, 25), m.RIGHT, buff=0.15)
        ceiling_label = text("ceiling", 20, MUTED).next_to(ax.c2p(t_max, 100), m.RIGHT, buff=0.15)
        note = text("illustrative curves", 20, MUTED).next_to(ax.c2p(0, 0), m.DOWN, buff=0.3).align_to(ax.c2p(0, 0), m.LEFT)
        colors = (SKY, BLUE, VIOLET)
        names = ("a test", "a harder test", "harder still")
        curves, labels = [], []
        for t0, color, name in zip(BENCH_T0, colors, names):
            start = t0 - 2.0
            curves.append(ax.plot(lambda t, t0=t0: float(saturation(t, t0)), x_range=[start, start + 4.1]).set_stroke(color, width=5))
            at_75 = ax.c2p(t0 + 0.45 - 0.3, 75)  # just left of where the curve crosses 75%
            labels.append(text(name, 20, color, weight=m.SEMIBOLD).next_to(at_75, m.LEFT, buff=0.1))
        top_t = (3.1, 3.45, 3.8)
        tops = m.VGroup(*[m.Dot(ax.c2p(t, float(saturation(t, BENCH_T0[0]))), radius=0.08, color=INK) for t in top_t]).set_z_index(3)
        apart = text("top models: can't tell apart", 20, BAD, weight=m.SEMIBOLD).next_to(ax.c2p(3.45, 100), m.UP, buff=0.2)

        succ_label = text("harder successors", 22, MUTED, weight=m.SEMIBOLD)
        succ = m.VGroup()
        for name, who in (("MMLU-Pro", "Wang et al., 2024"), ("GPQA", "Rein et al., 2023"), ("Humanity's Last Exam", "Phan et al., 2025")):
            p = pill(name, SKY, SKY_FILL, 20)
            succ.add(m.VGroup(p, cite(who).next_to(p, m.DOWN, buff=0.12)))
        row = m.VGroup(succ_label, *succ).arrange(m.RIGHT, buff=0.55, aligned_edge=m.UP).move_to([0, -3.1, 0])
        succ_label.match_y(succ[0][0])

        self.say(
            "[And benchmarks saturate]: [once top models all score near the ceiling], [the test can no longer tell them apart.]",
            m.AnimationGroup(
                self.new_header("1 · BENCHMARKS", "Tests saturate", SKY, SKY_FILL),
                m.Create(ax), m.FadeIn(x_label), m.FadeIn(y_ticks), m.Create(ceiling), m.Create(chance),
                m.FadeIn(chance_label), m.FadeIn(ceiling_label), m.FadeIn(note),
            ),
            m.AnimationGroup(m.Create(curves[0], run_time=1.6), m.FadeIn(labels[0])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in tops], lag_ratio=0.25), m.FadeIn(apart, shift=0.15 * m.UP)),
        )
        self.say(
            "[So harder tests replace them], [and the race starts again.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(x, shift=0.2 * m.UP) for x in row], lag_ratio=0.2),
                m.Create(curves[1], run_time=1.5), m.FadeIn(labels[1]),
            ),
            m.AnimationGroup(m.Create(curves[2], run_time=1.5), m.FadeIn(labels[2])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ contamination

    def contamination(self):
        self.next_section(
            "2 · Contamination",
            notes="Illustrative: 200 questions, 60 leaked into training and memorised; the model genuinely answers 60% "
            "(seeded draws: public 144/200 = 72%, fresh 120/200 = 60%). Real case: GSM1k (Zhang et al., 2024), fresh "
            "grade-school maths problems in the style of GSM8K; several model families scored clearly lower on GSM1k, "
            "frontier models less so. Defences: n-gram overlap checks (e.g. Brown et al., 2020), private held-out sets, "
            "questions written after the training cut-off (e.g. LiveBench, White et al., 2024).",
        )
        y_flow = 1.95
        globe = m.VGroup(icon("ep46-globe", 0.8), text("the web", 22, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.25).move_to([-4.7, y_flow, 0])
        data = m.VGroup(icon("database", 0.8), text("training data", 22, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.25).move_to([0, y_flow, 0])
        model = m.VGroup(icon("llm", 0.8), text("the model", 22, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.25).move_to([4.6, y_flow, 0])
        a1 = arrow(globe.get_right() + 0.15 * m.RIGHT, data.get_left() + 0.15 * m.LEFT, MUTED)
        a2 = arrow(data.get_right() + 0.15 * m.RIGHT, model.get_left() + 0.15 * m.LEFT, MUTED)
        leak_sq = m.VGroup(*[square([0, 0, 0], 0.2, AMBER, AMBER_FILL, width=2.5) for _ in range(4)]).arrange(m.RIGHT, buff=0.12)
        leak_sq.next_to(a1, m.UP, buff=0.12)
        leak_label = text("+ test questions", 20, AMBER, weight=m.SEMIBOLD).next_to(a1, m.DOWN, buff=0.15)
        flow = m.VGroup(globe, data, model, a1, a2, leak_sq, leak_label)

        cols, step, side = 20, 0.21, 0.15
        pub_c, fresh_c = np.array([-3.3, -0.55, 0]), np.array([3.3, -0.55, 0])
        pub_pos = grid_centers(N_TEST, cols, step, pub_c)
        fresh_pos = grid_centers(N_TEST, cols, step, fresh_c)
        pub_base = m.VGroup(*[square(p, side, EDGE, CARD) for p in pub_pos])
        fresh_base = m.VGroup(*[square(p, side, EDGE, CARD) for p in fresh_pos])
        leaked = set(int(i) for i in LEAKED)
        leak_marks = m.VGroup(*[square(pub_pos[i], side, AMBER, CARD, width=2.5) for i in sorted(leaked)]).set_z_index(1)
        memorised = m.VGroup(*[square(pub_pos[i], side, AMBER, GOOD, 0.95, width=2.5) for i in sorted(leaked)]).set_z_index(2)
        others = m.VGroup(
            *[square(pub_pos[i], side, GOOD if PUBLIC[i] else BAD, GOOD if PUBLIC[i] else BAD, 0.9 if PUBLIC[i] else 0.75) for i in range(N_TEST) if i not in leaked]
        ).set_z_index(2)
        fresh_marks = m.VGroup(
            *[square(fresh_pos[i], side, GOOD if FRESH[i] else BAD, GOOD if FRESH[i] else BAD, 0.9 if FRESH[i] else 0.75) for i in range(N_TEST)]
        ).set_z_index(2)
        pub_title = text("public test · 200 questions", 22, INK, weight=m.SEMIBOLD).move_to([pub_c[0], 0.85, 0])
        leak_count = m.VGroup(text("60", 30, AMBER, weight=m.BOLD), text("leaked", 22, AMBER, weight=m.BOLD)).arrange(m.DOWN, buff=0.08)
        leak_count.next_to(pub_base, m.LEFT, buff=0.3)
        fresh_title = text("fresh test · 200 new questions", 22, INK, weight=m.SEMIBOLD).move_to([fresh_c[0], 0.85, 0])
        pub_score = text(f"{round(100 * PUBLIC.mean())}%", 44, GOOD, weight=m.BOLD).move_to([pub_c[0], -2.15, 0])
        fresh_score = text(f"{round(100 * FRESH.mean())}%", 44, INK, weight=m.BOLD).move_to([fresh_c[0], -2.15, 0])
        real = pill("real case: on fresh look-alike maths problems, some models scored clearly lower", PINK, PINK_FILL, 22).move_to([0, -2.95, 0])
        zhang = cite("GSM1k vs GSM8K · Zhang et al., 2024").move_to([0, -3.5, 0])
        fixes = m.VGroup(
            pill("check training data for overlap", GOOD, GOOD_FILL, 20),
            pill("keep held-out tests private", GOOD, GOOD_FILL, 20),
            pill("fresh questions after the cut-off", GOOD, GOOD_FILL, 20),
        ).arrange(m.RIGHT, buff=0.25).move_to([0, y_flow, 0])

        self.say(
            "[A sneakier problem is contamination.] [Models learn from the web], [and benchmark questions end up on the web too.]",
            self.new_header("2 · CONTAMINATION", "When the test leaks into training"),
            m.LaggedStart(m.FadeIn(globe), m.GrowArrow(a1), m.FadeIn(data), m.GrowArrow(a2), m.FadeIn(model), lag_ratio=0.2),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(s, target_position=globe[0]) for s in leak_sq], lag_ratio=0.2),
                m.FadeIn(leak_label), pulse(a1, AMBER),
            ),
        )
        self.say(
            "[Say, for example, that sixty of our two hundred test questions leaked], [and the model memorised their answers.]",
            m.AnimationGroup(
                m.FadeIn(pub_title), m.FadeIn(leak_count), m.FadeIn(pub_base, lag_ratio=0.004),
                m.LaggedStart(*[m.FadeIn(s) for s in leak_marks], lag_ratio=0.03),
            ),
            m.LaggedStart(*[m.FadeIn(s, scale=1.3) for s in memorised], lag_ratio=0.02),
        )
        self.say(
            "[On the public test, it scores seventy-two percent.] [On fresh questions it has never seen, only sixty.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s) for s in others], lag_ratio=0.01), m.FadeIn(pub_score, shift=0.2 * m.UP)),
            m.AnimationGroup(
                m.FadeIn(fresh_title), m.FadeIn(fresh_base),
                m.LaggedStart(*[m.FadeIn(s) for s in fresh_marks], lag_ratio=0.008),
                m.FadeIn(fresh_score, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[This happens for real]: on newly written grade-school maths problems, [some models scored clearly lower than on the public set.]",
            m.FadeIn(real, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(zhang), m.Indicate(fresh_score, color=BAD, scale_factor=1.12)),
        )
        self.say(
            "[So evaluators check training data for overlap], [keep held-out tests private], [and write fresh questions after a model's training cut-off.]",
            m.AnimationGroup(m.FadeOut(flow, shift=0.2 * m.UP), m.FadeIn(fixes[0], shift=0.2 * m.UP)),
            m.FadeIn(fixes[1], shift=0.2 * m.UP),
            m.FadeIn(fixes[2], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ human judges

    def human_judges(self):
        self.next_section(
            "3 · Human judges",
            notes="Pairwise human preference: Chatbot Arena (Chiang et al., 2024) ranks models from crowd votes between two "
            "anonymous answers (Bradley-Terry / Elo-style ratings). Leaderboard here: 4 illustrative models, seeded votes from "
            "Bradley-Terry strengths (win rates computed). Bias: Hosking et al., 2024 ('Human Feedback is not Gold Standard') "
            "find that assertive answers are perceived as having fewer factual errors.",
        )
        q = text("Q: When did the Titanic sink?", 24, INK, weight=m.SEMIBOLD).move_to([-3.35, 2.3, 0])

        def answer(name, words, x):
            box = card(2.85, 2.0, EDGE, CARD, stroke=2).move_to([x, 0.95, 0])
            body = m.VGroup(text(name, 22, MUTED, weight=m.BOLD), text(words, 24, INK)).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
            body.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            return m.VGroup(box, body)

        card_a = answer("Answer A", "Definitely\nin 1923.", -4.85)
        card_b = answer("Answer B", "I think 1912,\nbut check.", -1.85)
        judge = icon("user", 1.0).move_to([-3.35, -1.4, 0])
        ask = text("which is better?", 22, MUTED).next_to(judge, m.DOWN, buff=0.2)
        vote = arrow(judge.get_left() + np.array([0.05, 0.25, 0]), card_a[0].get_bottom() + np.array([0.2, 0, 0]), BLUE, buff=0.1)
        vote_label = text("vote", 20, BLUE, weight=m.SEMIBOLD).next_to(vote.get_start() + 0.3 * (vote.get_end() - vote.get_start()), m.LEFT, buff=0.25)
        wrong = text("✗ wrong", 22, BAD, weight=m.BOLD).move_to(card_a[0].get_corner(m.DR) + np.array([-0.7, 0.3, 0]))
        right = text("✓ right", 22, GOOD, weight=m.BOLD).move_to(card_b[0].get_corner(m.DR) + np.array([-0.7, 0.3, 0]))

        lx = 1.0
        lb_title = left_at(text("Leaderboard", 28, INK, weight=m.BOLD), lx, 2.0)
        lb_sub = left_at(text("win rate in pairwise votes", 20, MUTED), lx, 1.55)
        lb_rows = m.VGroup()
        for k, (name, rate) in enumerate(zip(ARENA, WIN_RATE)):
            y = 0.85 - 0.62 * k
            rank = text(f"{k + 1}", 24, MUTED, weight=m.BOLD).move_to([lx + 0.15, y, 0])
            label = left_at(text(name, 22, INK, weight=m.SEMIBOLD), lx + 0.55, y)
            bar = m.Rectangle(width=3.2 * rate, height=0.3).set_stroke(width=0).set_fill(BLUE, opacity=0.85).move_to([lx + 2.15, y, 0], aligned_edge=m.LEFT)
            val = text(f"{100 * rate:.0f}%", 20, BLUE, weight=m.SEMIBOLD).next_to(bar, m.RIGHT, buff=0.15)
            lb_rows.add(m.VGroup(rank, label, bar, val))
        arena = left_at(cite("Chatbot Arena · Chiang et al., 2024"), lx, -1.65)
        caveat = pill("people rate confident answers as more accurate than they are", BAD, BAD_FILL, 22).move_to([0, -2.75, 0])
        hosking = cite("Hosking et al., 2024").move_to([0, -3.35, 0])
        game = m.VGroup(icon("ep46-boat", 0.45).rotate(-np.pi / 2), text("to a model in training, every judge is one more score to game", 22, AMBER, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        callback = m.VGroup(card(game.width + 0.7, game.height + 0.3, AMBER, AMBER_FILL, radius=(game.height + 0.3) / 2, stroke=2), game)
        game.move_to(callback[0])
        callback.move_to([0, -2.75, 0])

        self.say(
            "[Open-ended answers have no answer key.] [So we show people two anonymous answers], [and ask which one is better.]",
            m.AnimationGroup(self.new_header("3 · HUMAN JUDGES", "Which answer is better?", BLUE, BLUE_FILL), m.FadeIn(q)),
            m.LaggedStart(m.FadeIn(card_a, shift=0.2 * m.UP), m.FadeIn(card_b, shift=0.2 * m.UP), lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(judge, shift=0.2 * m.UP), m.FadeIn(ask), m.GrowArrow(vote), m.FadeIn(vote_label)),
        )
        self.say(
            "[Many thousands of such votes add up to a leaderboard], [as in Chatbot Arena.]",
            m.AnimationGroup(m.FadeIn(lb_title), m.FadeIn(lb_sub), m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in lb_rows], lag_ratio=0.2)),
            m.FadeIn(arena),
        )
        self.say(
            "[But judges can be swayed]: [people tend to rate confident answers as more accurate than they are.]",
            m.AnimationGroup(m.FadeIn(wrong, scale=1.2), m.FadeIn(right, scale=1.2)),
            m.AnimationGroup(m.FadeIn(caveat, shift=0.2 * m.UP), m.FadeIn(hosking)),
        )
        self.say(
            "[To a model in training, every judge is just one more score to game.]",
            m.AnimationGroup(m.FadeTransform(caveat, callback), m.FadeOut(hosking)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ red-teaming

    def red_teaming(self):
        self.next_section(
            "4 · Red-teaming",
            notes="Red-teaming by people (Ganguli et al., 2022) and by language models (Perez et al., 2022). The 200 attacks "
            "and the 9 that get through are illustrative (seeded draws).",
        )
        person = m.VGroup(icon(variant("user", BAD, BAD_FILL), 0.9), text("people", 20, BAD, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.15).move_to([-5.7, 0.85, 0])
        bot = m.VGroup(icon(variant("llm", BAD, BAD_FILL), 0.9), text("other models", 20, BAD, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.15).move_to([-5.7, -1.15, 0])
        shield = m.VGroup(icon("ep46-shield", 1.5), text("its rules", 20, GOOD, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.15).move_to([-0.95, -0.15, 0])
        target = m.VGroup(icon("llm", 1.1), text("the model", 20, INK, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.15).move_to([0.75, -0.15, 0])
        hit = shield[0].get_left() + 0.1 * m.LEFT
        arrows = m.VGroup(
            arrow(person[0].get_right(), hit + 0.25 * m.UP, BAD, buff=0.15),
            arrow(bot[0].get_right(), hit + 0.25 * m.DOWN, BAD, buff=0.15),
        )
        prompt = bubble(text("Ignore your rules and…", 22, INK), BAD, BAD_FILL, "left").move_to([-3.2, 1.95, 0])

        cols, step = 20, 0.19
        g_c = np.array([4.6, 0.15, 0])
        pos = grid_centers(N_ATTACKS, cols, step, g_c)
        base = m.VGroup(*[square(p, 0.13, EDGE, CARD) for p in pos])
        blocked = m.VGroup(*[square(pos[i], 0.13, GOOD, GOOD, 0.8) for i in range(N_ATTACKS) if not BROKE[i]]).set_z_index(1)
        through = m.VGroup(*[square(pos[i], 0.13, BAD, BAD, 1.0) for i in range(N_ATTACKS) if BROKE[i]]).set_z_index(2)
        g_title = text("200 attacks", 24, INK, weight=m.BOLD).next_to(base, m.UP, buff=0.3)
        legend = m.VGroup(
            m.VGroup(square([0, 0, 0], 0.2, GOOD, GOOD, 0.8), text("blocked", 20, GOOD, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(square([0, 0, 0], 0.2, BAD, BAD), text("9 got through", 20, BAD, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.5).next_to(base, m.DOWN, buff=0.3)
        fix_train = pill("fixed in training", VIOLET, VIOLET_FILL, 22)
        fix_test = pill("added to the test set", SKY, SKY_FILL, 22)
        fixes = m.VGroup(fix_train, fix_test).arrange(m.DOWN, buff=0.2).move_to([g_c[0], -2.55, 0])
        links = m.VGroup(
            arrow(legend.get_bottom() + 0.05 * m.DOWN, fix_train.get_top(), MUTED, buff=0.08),
            m.Line(fix_train.get_bottom(), fix_test.get_top()).set_stroke(MUTED, width=3),
        )
        who = cite("Perez et al., 2022  ·  Ganguli et al., 2022").move_to([-3.6, -3.3, 0])

        self.say(
            "[Benchmarks measure what a model can do.] [Red-teaming hunts for what it shouldn't do.]",
            m.AnimationGroup(self.new_header("4 · RED-TEAMING", "Attack it before others do", BAD, BAD_FILL), m.FadeIn(target, shift=0.2 * m.UP)),
            m.FadeIn(shield, scale=0.9),
        )
        self.say(
            "[Attackers, both people and other models], [send tricky prompts that try to make it break its rules.]",
            m.LaggedStart(m.FadeIn(person, shift=0.2 * m.RIGHT), m.FadeIn(bot, shift=0.2 * m.RIGHT), lag_ratio=0.3),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.2), m.FadeIn(prompt, shift=0.2 * m.UP), m.FadeIn(who)),
        )
        self.say(
            "[For example, out of two hundred attacks], [nine get through.]",
            m.AnimationGroup(m.FadeIn(g_title), m.FadeIn(base, lag_ratio=0.004), m.LaggedStart(*[m.FadeIn(s) for s in blocked], lag_ratio=0.006)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, scale=1.8) for s in through], lag_ratio=0.15), m.FadeIn(legend, shift=0.15 * m.UP)),
        )
        self.say(
            "[Each one that gets through is fixed in training], [and joins the test set], so the next model must face it too.",
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(fix_train, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Create(links[1]), m.FadeIn(fix_test, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ scalable oversight

    def oversight_problem(self):
        self.next_section(
            "5 · Scalable oversight: the problem",
            notes="Scalable oversight (term from Amodei et al., 2016): supervising systems whose outputs are too hard or too long "
            "for people to check unaided; measuring progress: Bowman et al., 2022.",
        )
        person = m.VGroup(icon("user", 1.3), text("a human reviewer", 22, MUTED)).arrange(m.DOWN, buff=0.2).move_to([-5.0, -0.2, 0])
        doc_box = card(3.4, 4.6, VIOLET, CARD, stroke=2).move_to([3.4, -0.25, 0])
        doc_lines = bars_block(2.6, 18, seed=4)
        doc_lines.move_to(doc_box)
        doc_label = text("the model's work", 22, VIOLET, weight=m.SEMIBOLD).next_to(doc_box, m.LEFT, buff=0.35).align_to(doc_box, m.UP)
        doubt = bubble(text("?", 40, SKY, weight=m.BOLD), SKY, SKY_FILL, "left").next_to(person[0], m.UP, buff=0.1).shift(0.7 * m.RIGHT)
        helper = m.VGroup(icon(variant("llm", TEAL, TEAL_FILL), 1.0), text("an AI helper", 22, TEAL, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.2).move_to([-1.4, -0.2, 0])
        to_helper = arrow([doc_box.get_left()[0] - 0.1, helper[0].get_y(), 0], helper[0].get_right(), TEAL, buff=0.12)
        to_person = arrow(helper[0].get_left(), person[0].get_right(), TEAL, buff=0.15)
        question = text("Can AI help people judge AI?", 40, INK, weight=m.BOLD).move_to([0, -2.95, 0])

        self.say(
            "[Now the hardest part.] [As models get stronger, their work gets harder for people to check.]",
            m.AnimationGroup(self.new_header("5 · SCALABLE OVERSIGHT", "Who checks the work?", TEAL, TEAL_FILL), m.FadeIn(person, shift=0.2 * m.UP)),
            m.AnimationGroup(
                m.FadeIn(doc_box, shift=0.2 * m.UP),
                m.LaggedStart(*[m.FadeIn(x) for x in doc_lines], lag_ratio=0.06),
                m.FadeIn(doc_label),
                m.FadeIn(doubt, scale=1.2),
            ),
        )
        self.say(
            "[Scalable oversight asks]: [can AI help people judge AI?]",
            m.AnimationGroup(m.FadeIn(helper, shift=0.2 * m.UP), m.GrowArrow(to_helper), m.GrowArrow(to_person)),
            m.Write(question),
        )
        self.wait(0.3)
        self.clear()

    def oversight_ideas(self):
        self.next_section(
            "5 · Scalable oversight: three ideas",
            notes="Critiques: Saunders et al., 2022 (self-critiquing models help human evaluators find flaws); McAleese et al., "
            "2024 (CriticGPT: trainers with model critiques caught more inserted bugs). Debate: Irving, Christiano & Amodei, "
            "2018; empirical support on reading comprehension: Khan et al., 2024. Constitutional AI: Bai et al., 2022.",
        )
        specs = (
            ("search", "Critiques", "a second model points\nout flaws for the reviewer", "Saunders et al., 2022\nMcAleese et al., 2024"),
            ("ep46-scale", "Debate", "two models argue\nopposite sides", "Irving et al., 2018"),
            (variant("doc", TEAL, TEAL_FILL), "A constitution", "written principles guide\nAI feedback", "Bai et al., 2022"),
        )
        cards = []
        for (name, title, desc, who), x in zip(specs, (-4.45, 0, 4.45)):
            box = card(4.0, 3.9, TEAL, CARD, stroke=2).move_to([x, -0.3, 0])
            pic = icon(name, 0.95).move_to([x, box.get_top()[1] - 0.75, 0])
            title_text = text(title, 30, INK, weight=m.BOLD).move_to([x, box.get_top()[1] - 1.55, 0])
            body = text(desc, 22, MUTED).move_to([x, box.get_y() - 0.35, 0])
            src = text(who, 20, MUTED).move_to([x, box.get_bottom()[1] + 0.5, 0])
            cards.append((m.VGroup(box, pic, title_text), body, src))

        self.say(
            "[One idea is critiques]: [a second model points out the flaws], [so human reviewers catch more bugs.]",
            m.AnimationGroup(self.new_header("5 · SCALABLE OVERSIGHT", "AI helping people judge AI", TEAL, TEAL_FILL), m.FadeIn(cards[0][0], shift=0.2 * m.UP)),
            m.FadeIn(cards[0][1], shift=0.15 * m.UP),
            m.FadeIn(cards[0][2]),
        )
        self.say(
            "[Another is debate]: [two models argue opposite sides], [betting that a lie is easier to expose than to defend.]",
            m.FadeIn(cards[1][0], shift=0.2 * m.UP),
            m.FadeIn(cards[1][1], shift=0.15 * m.UP),
            m.FadeIn(cards[1][2]),
        )
        self.say(
            "[A third is a constitution]: [written principles that a model uses to critique and revise its own answers.]",
            m.FadeIn(cards[2][0], shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(cards[2][1], shift=0.15 * m.UP), m.FadeIn(cards[2][2])),
        )
        self.wait(0.3)
        self.clear()

    def oversight_monitor(self):
        self.next_section(
            "5 · Scalable oversight: watching the reasoning",
            notes="Baker et al., 2025 (OpenAI, 'Monitoring Reasoning Models for Misbehavior and the Risks of Promoting "
            "Obfuscation'): a GPT-4o monitor reading a stronger reasoning model's chain of thought caught most of its reward "
            "hacks in agentic coding (far more than watching actions alone); adding the monitor to the reward made the agent "
            "hide its intent while still hacking. The chain-of-thought text shown is an illustrative paraphrase.",
        )
        window = card(6.3, 3.5, VIOLET, CARD, stroke=2).move_to([-3.3, 0.15, 0])
        w_title = left_at(text("the coding model's chain of thought", 22, VIOLET, weight=m.SEMIBOLD), window.get_left()[0] + 0.35, 1.5)
        x0 = window.get_left()[0] + 0.4
        thought = left_at(menlo("Fixing sort() is hard...\nWe could just exit\nbefore the tests run.", 22, INK), x0, 0.35)
        clean = left_at(menlo("Let me fix sort()\ncarefully, step by step.", 22, INK), x0, 0.5)
        act_box = card(5.6, 0.62, EDGE, BG, radius=0.14, stroke=2).move_to([window.get_x(), -1.05, 0])
        action = left_at(menlo("action: sys.exit(0)", 22, AMBER), act_box.get_left()[0] + 0.3, -1.05)

        monitor = m.VGroup(text("a weaker monitor model", 22, TEAL, weight=m.SEMIBOLD), icon(variant("llm", TEAL, TEAL_FILL), 1.1)).arrange(m.DOWN, buff=0.25)
        monitor.move_to([3.7, 1.3, 0])
        eye = monitor[1]
        reads = arrow(window.get_right() + np.array([0, 0.8, 0]), eye.get_left() + 0.15 * m.UP, TEAL, buff=0.15)
        reads_label = text("reads", 20, TEAL, weight=m.SEMIBOLD).next_to(reads.get_center(), m.UP, buff=0.2)
        flag = pill("✗  flagged: reward hacking", BAD, BAD_FILL, 24).move_to([3.7, -0.55, 0])
        fooled = pill("✓  looks fine", GOOD, GOOD_FILL, 24).move_to([3.7, -0.55, 0])
        penalty = arrow(eye.get_left() + 0.3 * m.DOWN, window.get_right() + np.array([0, 0.1, 0]), BAD, buff=0.15)
        penalty_label = text("penalty", 20, BAD, weight=m.SEMIBOLD).next_to(penalty.get_center(), m.DOWN, buff=0.25)
        baker = cite("Baker et al., 2025").move_to([3.7, -1.3, 0])
        hidden = pill("still cheating, just quietly", BAD, BAD_FILL, 26).move_to([0, -2.8, 0])

        self.say(
            "[Back to our coding model that skipped the tests.] [A weaker model read its chain of thought], [and caught most of those hacks.]",
            m.AnimationGroup(
                self.new_header("5 · SCALABLE OVERSIGHT", "Reading the model's reasoning", TEAL, TEAL_FILL),
                m.FadeIn(window), m.FadeIn(w_title), m.FadeIn(thought), m.FadeIn(act_box), m.FadeIn(action),
            ),
            m.AnimationGroup(m.FadeIn(monitor, shift=0.2 * m.LEFT), m.GrowArrow(reads), m.FadeIn(reads_label)),
            m.AnimationGroup(m.FadeIn(flag, scale=1.15), m.FadeIn(baker)),
        )
        self.say(
            "[But penalise the model for those thoughts], [and it learns to hide them]: [it still cheats, just quietly.]",
            m.AnimationGroup(m.GrowArrow(penalty), m.FadeIn(penalty_label)),
            m.AnimationGroup(m.FadeTransform(thought, clean), m.FadeTransform(flag, fooled)),
            m.AnimationGroup(m.Wiggle(action, scale_value=1.08), m.FadeIn(hidden, shift=0.2 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Measure (benchmarks, held-out tests, human judges), probe (red-teaming), oversee (AI-assisted oversight).")
        halo = glow(PINK, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Measure", "Probe", "Oversee"), (SKY, BAD, TEAL))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A model learns what you reward. So reward what you really mean.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [measure], [probe], [and oversee.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A model learns what you reward. So reward what you really mean.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)

    # ------------------------------------------------------------------ the end of the course

    def series_outro(self):
        """The last episode: close the course instead of announcing a next one."""
        self.next_section("The end of the course", notes="Closing card of AI, Explained: 46 episodes in 5 levels.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        halo = glow(PINK, radius=4.4).move_to(0.3 * m.UP)
        brand = text("AI, Explained", 72, INK, weight=m.BOLD, gradient=(SKY, VIOLET))
        sub = text("46 episodes  ·  from a spam filter to the research frontier", 26, MUTED)
        badges = [pill(f"{n} · {name}", accent, tint, 20) for n, (name, accent, tint) in LEVELS.items()]
        rows = m.VGroup(m.VGroup(*badges[:3]).arrange(m.RIGHT, buff=0.3), m.VGroup(*badges[3:]).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.25)
        thanks = text("Thanks for watching!", 44, INK, weight=m.BOLD)
        group = m.VGroup(brand, sub, rows, thanks).arrange(m.DOWN, buff=0.45).move_to(0.1 * m.UP)
        self.say(
            "[That's the end of AI, Explained]: [from a spam filter to the frontier of research.] [Thanks for watching!]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(brand, shift=0.2 * m.DOWN), m.FadeIn(sub)),
            m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.UP) for b in badges], lag_ratio=0.2),
            m.FadeIn(thanks, shift=0.2 * m.UP, scale=0.9),
        )
        self.wait(1.2)
        self.play(m.FadeOut(halo), m.FadeOut(group), run_time=1.0)
        self.wait(0.5)
