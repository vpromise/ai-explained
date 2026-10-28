"""AI, Explained · Episode 43 · Reasoning Models & Test-Time Compute

Thinking longer instead of being bigger. Running example, all film long: a math word problem with
a trap. "A snail is at the bottom of a 20 m well. Each day it climbs 5 m; each night it slides back
3 m. On which day does it get out?" The quick answer (2 m a day, 20 / 2 = day 10) is wrong; step by
step, after 8 days and nights it is at 16 m, and on day 9 it climbs to 21 m: out on day 9.

- Chain of thought (Wei et al., 2022; callback Episode 31): small steps, more tokens.
- Self-consistency (Wang et al., 2022): five sampled chains answer 9, 10, 9, 8, 9 (illustrative);
  the majority vote picks 9.
- The math (computed exactly below and asserted against the narration): each chain is right with
  p = 0.6, independently; worst case, every wrong chain gives the same wrong answer, so the right
  answer needs a strict majority of N (odd) votes:
      P_vote = sum_{k > N/2} C(N, k) p^k (1 - p)^(N - k)
      N = 1: 60%, 5: 68%, 11: 75%, 41: 90%, 101: 98%.
  When the errors agree (the trap answer is the most likely one, p = 0.4): N = 41 gives 10%.
- Best-of-N with a verifier (Cobbe et al., 2021). With a perfect checker the five chains win if any
  one is right: 1 - 0.4^5 = 98.98%, about 99% (versus 68% for voting). Verifier scores illustrative.
- Search over steps with a process reward model (Lightman et al., 2023). Scores illustrative.
- Reasoning models: reinforcement learning with verifiable rewards (term: Lambert et al., 2024,
  Tülu 3). OpenAI o1 (2024): large-scale RL on chains of thought. DeepSeek-R1 (DeepSeek-AI, 2025):
  rule-based accuracy and format rewards; in R1-Zero the response length grew during RL and
  reflection ("wait…") emerged. Chain-length bars are schematic.
- The price: 1,000 illustrative problems whose one-attempt success rates are drawn from
  Beta(0.5, 1) (seeded numpy), a perfect checker, 1,000 tokens per attempt; solved = mean of
  1 - (1 - p)^n: 1 attempt 36%, 8 attempts 72%, 64 attempts 89%.
- Hook: Snell et al., 2024 — at matched FLOPs, test-time compute let a smaller model beat a ~14x
  larger one on problems where the small model already had non-trivial success (easy/medium).

Colours: the problem (and the snail) SKY; the model and its reasoning chains VIOLET; thinking tokens
and votes AMBER; verifiers, checkers and rewards TEAL; right GOOD, wrong BAD; headers PINK (level 5).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

FACTS = "Facts checked 2026-10-21."

# ---------------------------------------------------------------------------- the numbers

WELL, CLIMB, SLIDE = 20, 5, 3
NET = CLIMB - SLIDE


def sawtooth(max_days=12):
    """The snail's height over time: (day, metres) corners, day d's climb peaking at d - 0.5."""
    pts, h = [(0.0, 0.0)], 0
    for day in range(1, max_days + 1):
        top = h + CLIMB
        pts.append((day - 0.5, top))
        if top >= WELL:
            return day, pts
        h = top - SLIDE
        pts.append((float(day), h))
    raise ValueError("never out")


DAY, SAW = sawtooth()
QUICK = WELL // NET
EXIT_T = 8 + 0.5 * (WELL - 16) / CLIMB  # when the day-9 climb passes the rim
assert DAY == 9 and QUICK == 10 and NET == 2
assert SAW[16] == (8.0, 16) and SAW[-1] == (8.5, 21) and 8 * NET == 16 and 16 + CLIMB == 21

ANSWERS = (9, 10, 9, 8, 9)  # five sampled chains (illustrative)
assert Counter(ANSWERS).most_common(1)[0] == (DAY, 3)

P = 0.6  # each chain's chance of being right


def vote_right(p, n):
    """Chance that more than half of n independent chains are right (n odd): the right answer
    wins even if every wrong chain gives the same wrong answer."""
    assert n % 2 == 1
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


NS = list(range(1, 102, 2))
MARKS = {1: 60, 5: 68, 11: 75, 41: 90, 101: 98}
TRAP_MARKS = {1: 40, 41: 10}
for _n, _pct in MARKS.items():
    assert round(100 * vote_right(P, _n)) == _pct
for _n, _pct in TRAP_MARKS.items():
    assert round(100 * vote_right(1 - P, _n)) == _pct
assert all(b > a for a, b in zip([vote_right(P, n) for n in NS], [vote_right(P, n) for n in NS[1:]]))

BEST5 = 1 - (1 - P) ** 5
assert round((1 - P) ** 5, 2) == 0.01 and round(100 * BEST5) == 99

SCORES = (0.92, 0.18, 0.71, 0.35, 0.83)  # a verifier's scores for the five chains (illustrative)
assert ANSWERS[SCORES.index(max(SCORES))] == DAY

EASE = np.random.default_rng(43).beta(0.5, 1.0, 1000)  # each problem's one-attempt success rate


def solved(n):
    return float(np.mean(1 - (1 - EASE) ** n))


COST_MARKS = {1: 36, 8: 72, 64: 89}
for _n, _pct in COST_MARKS.items():
    assert round(100 * solved(_n)) == _pct
assert (72 - 36, 89 - 72) == (36, 17)

# ---------------------------------------------------------------------------- helpers


def cite(s):
    return text(s, 20, MUTED)


def polyline(points, color, width=4):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width)


def recolored_ep43(name, color, fill):
    """kit.recolored(), written as assets/ep43-<name>-<colour>.svg (episode-owned files)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep43-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


class Chart:
    """Quiet axes (no ticks) with the origin at the lower-left corner."""

    def __init__(self, x_max, y_max, width, height, center):
        self.ax = m.Axes(
            x_range=[0, x_max, 1], y_range=[0, y_max, 1], x_length=width, y_length=height,
            axis_config={"color": FAINT, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18, "include_ticks": False},
        ).move_to(center)

    def p(self, x, y):
        return self.ax.c2p(x, y)

    def line(self, pts, color, width=4):
        return polyline([self.p(x, y) for x, y in pts], color, width)

    def xlabel(self, x, s):
        return text(s, 20, MUTED).move_to(self.p(x, 0) + 0.32 * m.DOWN)

    def ylabel(self, y, s):
        return text(s, 20, MUTED).next_to(self.p(0, y), m.LEFT, buff=0.18)

    def ticks(self, xs=(), ys=()):
        marks = [m.Line(self.p(x, 0) + 0.07 * m.DOWN, self.p(x, 0) + 0.07 * m.UP) for x in xs]
        marks += [m.Line(self.p(0, y) + 0.07 * m.LEFT, self.p(0, y) + 0.07 * m.RIGHT) for y in ys]
        return m.VGroup(*marks).set_stroke(FAINT, width=2.5)


def stack_lines(lines, gap):
    """Left-aligned lines with evenly spaced baselines (each line starts with a capital letter)."""
    base = lines[0][0].get_bottom()[1]
    for i, line in enumerate(lines):
        line.align_to(lines[0], m.LEFT)
        line.shift((base - i * gap - line[0].get_bottom()[1]) * m.UP)
    return lines


def chain_card(answer, seed, width, height=0.6):
    """One sampled chain of thought: a row of step marks, then its final answer in a chip."""
    box = card(width, height, VIOLET, CARD, radius=0.16, stroke=2)
    tag = chip(str(answer), VIOLET, VIOLET_FILL, width=0.78, height=height - 0.16, size=24)
    tag.move_to(box.get_right() + 0.51 * m.LEFT)
    rng = np.random.default_rng(seed)
    room = width - 0.78 - 0.12 - 0.45
    widths = []
    while sum(widths) + 0.08 * len(widths) < room:
        widths.append(float(rng.uniform(0.22, 0.7)))
    widths = widths[:-1]
    scale = (room - 0.08 * (len(widths) - 1)) / sum(widths)
    steps = m.VGroup(
        *[m.RoundedRectangle(corner_radius=0.06, width=w * scale, height=0.16).set_stroke(width=0).set_fill(VIOLET, opacity=0.55) for w in widths]
    ).arrange(m.RIGHT, buff=0.08)
    steps.move_to(box).align_to(box, m.LEFT).shift(0.22 * m.RIGHT)
    return m.VGroup(box, steps, tag)


def recolor_chip(c, color, fill):
    return c[2][0].animate.set_stroke(color).set_fill(fill, opacity=1)


def token_row(n, size=0.36, buff=0.1):
    return m.VGroup(
        *[m.RoundedRectangle(corner_radius=0.08, width=size, height=size).set_stroke(AMBER, width=2.5).set_fill(AMBER_FILL, opacity=1) for _ in range(n)]
    ).arrange(m.RIGHT, buff=buff)


def bar(width, color, height=0.26):
    return m.Rectangle(width=max(width, 0.05), height=height, stroke_width=0).set_fill(color, opacity=0.9)


# ---------------------------------------------------------------------------- the episode


class Ep43ReasoningTestTime(Episode):
    number = 43

    def lesson(self):
        self.hook()
        self.example()
        self.chain_of_thought()
        self.self_consistency()
        self.voting_math()
        self.catch()
        self.best_of_n()
        self.search()
        self.reasoning_models()
        self.price()
        self.recap()

    # ------------------------------------------------------------------ series card

    def series_intro(self):
        """The kit's series card, unchanged except that the heading is scaled to fit the frame: at
        the kit's 60 pt this title ran 0.57 units past both edges. Same narration as the kit's."""
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

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes="Two levers: a bigger model (training compute) or more thinking (test-time compute). "
            "Snell et al., 2024 (arXiv 2408.03314): at matched FLOPs, test-time compute let a smaller model "
            "outperform a ~14x larger one on problems where it had non-trivial success. " + FACTS,
        )
        halo = glow(PINK, radius=3.0).move_to([0, 2.25, 0])
        question = text("How do you get a better answer?", 50, INK, weight=m.BOLD).move_to([0, 2.3, 0])

        big_box = card(6.0, 3.5, VIOLET, CARD, stroke=2).move_to([-3.3, -0.4, 0])
        big = icon("llm", 1.7).move_to([-3.3, 0.4, 0])
        big_title = text("Make it bigger", 30, INK, weight=m.BOLD).move_to([-3.3, -1.25, 0])
        big_sub = text("more training compute", 22, MUTED).move_to([-3.3, -1.72, 0])

        think_box = card(6.0, 3.5, AMBER, CARD, stroke=2).move_to([3.3, -0.4, 0])
        small = icon("llm", 0.95).move_to([1.3, 0.4, 0])
        thoughts = token_row(8).next_to(small, m.RIGHT, buff=0.3)
        think_title = text("Let it think longer", 30, INK, weight=m.BOLD).move_to([3.3, -1.25, 0])
        think_sub = text("more test-time compute", 22, MUTED).move_to([3.3, -1.72, 0])

        result = pill("same compute: a smaller model + thinking beat a 14× larger one", GOOD, GOOD_FILL, 24).move_to([0, -2.8, 0])
        note = cite("on easy and medium problems · Snell et al., 2024").move_to([0, -3.45, 0])

        self.say(
            "So... [how do you get a better answer out of a language model?]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(question, shift=0.3 * m.UP), run_time=1.2),
        )
        self.say(
            "[For years, the answer was simple: make the model bigger.]",
            m.AnimationGroup(m.FadeIn(big_box), m.FadeIn(big, scale=0.6), m.FadeIn(big_title, shift=0.2 * m.UP), m.FadeIn(big_sub)),
        )
        self.say(
            "[Now there's a second lever]: [let it think longer before it answers.]",
            m.AnimationGroup(m.FadeIn(think_box), m.FadeIn(small, scale=0.6), m.FadeIn(think_title, shift=0.2 * m.UP), m.FadeIn(think_sub)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.RIGHT) for t in thoughts], lag_ratio=0.3),
        )
        self.say(
            "[In one study, at equal total compute, a smaller model that thought longer beat one fourteen times its size], "
            "[on easy and medium problems.]",
            m.FadeIn(result, shift=0.2 * m.UP),
            m.FadeIn(note),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the example

    def example(self):
        self.next_section(
            "The example",
            notes="A 20 m well, +5 m by day, -3 m by night. The quick answer, 20 / 2 = day 10, is wrong; "
            "the answer is day 9. The height curve is computed (sawtooth()).",
        )
        lines = m.VGroup(
            text("A snail is at the bottom of a 20 m well.", 25),
            text("Each day, it climbs 5 m.", 25),
            text("Each night, it slides back 3 m.", 25),
            text("On which day does it get out?", 25, SKY, weight=m.BOLD),
        )
        stack_lines(lines, 0.5)
        box = card(lines.width + 0.8, lines.height + 0.6, SKY, CARD, stroke=2).move_to([-3.5, 1.4, 0])
        lines.move_to(box)

        chart = Chart(10.8, 23.5, 5.3, 4.2, [3.95, -0.4, 0])
        self.well = chart
        frame = m.VGroup(
            chart.ax,
            chart.ticks(xs=[d - 0.5 for d in range(1, 11)], ys=(5, 10, 15, 20)),
            *[chart.xlabel(d - 0.5, str(d)) for d in range(1, 11)],
            *[chart.ylabel(v, str(v)) for v in (5, 10, 15, 20)],
            text("day", 22, MUTED).move_to(chart.p(5.4, 0) + 0.78 * m.DOWN),
            text("height (m)", 22, MUTED).next_to(chart.p(0, 23.5), m.UP, buff=0.12),
        )
        rim = m.DashedLine(chart.p(0, WELL), chart.p(10.5, WELL), dash_length=0.12).set_stroke(MUTED, width=2.5)
        rim_label = text("top of the well", 20, MUTED).next_to(chart.p(0.7, WELL), m.UP, buff=0.1, aligned_edge=m.LEFT)
        snail = icon("ep43-snail", 0.42)
        snail.move_to(chart.p(0, 0) + snail.height / 2 * m.UP).set_z_index(4)

        quick = bubble(text("2 m a day, so 20 ÷ 2 = day 10", 26, INK), VIOLET, VIOLET_FILL, tail="left").move_to([-3.6, -0.75, 0])
        wrong = pill("✗  WRONG", BAD, BAD_FILL, 26).move_to([-3.6, -1.9, 0])
        naive = m.DashedLine(chart.p(0, 0), chart.p(10, 20), dash_length=0.12).set_stroke(BAD, width=3.5)

        self.say(
            "[Our running example is a math word problem with a trap.]",
            m.AnimationGroup(
                self.new_header("THE EXAMPLE", "A math word problem with a trap"),
                m.Create(chart.ax), m.FadeIn(frame[1:]), m.FadeIn(snail, scale=0.6),
            ),
        )
        self.say(
            "[A snail is at the bottom of a twenty-metre well.] [Each day, it climbs five metres], [and each night, it slides back three.]",
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(lines[0], shift=0.2 * m.UP), m.Create(rim), m.FadeIn(rim_label)),
            m.FadeIn(lines[1], shift=0.2 * m.UP),
            m.FadeIn(lines[2], shift=0.2 * m.UP),
        )
        self.say("[On which day does it get out?]", m.FadeIn(lines[3], shift=0.2 * m.UP))
        self.say(
            "[The quick answer: it gains two metres a day, so twenty divided by two, day ten.] [Wrong.]",
            m.AnimationGroup(m.FadeIn(quick, shift=0.2 * m.UP), m.Create(naive, run_time=1.6)),
            m.Succession(m.FadeIn(wrong, scale=1.2), m.Wiggle(wrong, scale_value=1.08)),
        )
        self.wait(0.3)
        self.problem, self.snail = m.VGroup(box, lines), snail
        self.quick = m.VGroup(quick, wrong, naive)

    # ------------------------------------------------------------------ chain of thought

    def chain_of_thought(self):
        self.next_section(
            "Chain of thought",
            notes="Callback Episode 31. Chain-of-thought prompting: Wei et al., 2022. Three small steps: "
            "+2 m per day and night; 8 x 2 = 16 m; 16 + 5 = 21 m >= 20, out on day 9. Token bars are schematic.",
        )
        chart, snail = self.well, self.snail
        steps = m.VGroup(
            code("day + night: +5 − 3 = +2 m", 22, t2c={"day + night:": VIOLET}),
            code("after 8 nights: 8 × 2 = 16 m", 22, t2c={"after 8 nights:": VIOLET}),
            code("day 9: 16 + 5 = 21 m ≥ 20", 22, t2c={"day 9:": VIOLET}),
        ).arrange(m.DOWN, buff=0.24, aligned_edge=m.LEFT)
        steps.move_to([0, -0.62, 0]).align_to([-6.55, 0, 0], m.LEFT)
        answer = pill("✓  out on day 9", GOOD, GOOD_FILL, 26).move_to([0, -1.85, 0]).align_to([-6.55, 0, 0], m.LEFT)
        source = cite("Wei et al., 2022").move_to([0, -1.85, 0]).align_to([-0.35, 0, 0], m.RIGHT)

        saw8 = chart.line(SAW[:17], VIOLET, 4)
        saw9 = chart.line(SAW[16:], VIOLET, 4)
        lift = snail.height / 2 * m.UP
        path8 = saw8.copy().shift(lift)
        path9 = saw9.copy().shift(lift)
        exit_dot = m.Dot(chart.p(EXIT_T, WELL), radius=0.08, color=GOOD).set_z_index(5)

        names = m.VGroup(text("quick answer", 22, MUTED), text("chain of thought", 22, MUTED)).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        names.move_to([0, -2.95, 0]).align_to([-6.55, 0, 0], m.LEFT)
        x0 = -3.85
        quick_bar = bar(0.45, AMBER).move_to([x0, names[0].get_y(), 0], aligned_edge=m.LEFT)
        chain_bar = bar(3.1, AMBER).move_to([x0, names[1].get_y(), 0], aligned_edge=m.LEFT)
        tokens = text("tokens", 20, AMBER, weight=m.SEMIBOLD).next_to(chain_bar, m.RIGHT, buff=0.2)

        self.say(
            "[In episode thirty-one, we asked the model to think step by step.]",
            m.AnimationGroup(
                self.new_header("CHAIN OF THOUGHT", "Spend tokens on small steps"),
                m.FadeOut(self.quick), m.FadeIn(source),
            ),
        )
        self.say(
            "[Each full day and night, the snail gains two metres]: [after eight of them, it's at sixteen.]",
            m.FadeIn(steps[0], shift=0.2 * m.RIGHT),
            m.AnimationGroup(
                m.FadeIn(steps[1], shift=0.2 * m.RIGHT),
                m.Create(saw8, run_time=2.4, rate_func=m.linear),
                m.MoveAlongPath(snail, path8, run_time=2.4, rate_func=m.linear),
            ),
        )
        self.say(
            "[On day nine, it climbs five more, to twenty-one metres], [and it's out.]",
            m.AnimationGroup(
                m.FadeIn(steps[2], shift=0.2 * m.RIGHT),
                m.Create(saw9, run_time=1.0, rate_func=m.linear),
                m.MoveAlongPath(snail, path9, run_time=1.0, rate_func=m.linear),
            ),
            m.AnimationGroup(m.FadeIn(answer, scale=1.2), m.FadeIn(exit_dot), m.Flash(exit_dot.get_center(), color=GOOD, flash_radius=0.3)),
        )
        self.say(
            "[The chain costs more tokens], [but every step is small enough to get right.]",
            m.AnimationGroup(m.FadeIn(names), m.GrowFromEdge(quick_bar, m.LEFT), m.GrowFromEdge(chain_bar, m.LEFT), m.FadeIn(tokens)),
            m.Circumscribe(steps, color=VIOLET, buff=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ self-consistency

    def self_consistency(self):
        self.next_section(
            "Self-consistency",
            notes="Wang et al., 2022 (ICLR 2023): sample several chains of thought at temperature > 0, "
            "majority-vote the final answers. Callback Episode 33 (sampling). The five answers 9, 10, 9, 8, 9 "
            "are illustrative.",
        )
        snail = icon("ep43-snail", 0.75).move_to([-6.0, -0.05, 0])
        ask = text("which day?", 24, SKY, weight=m.BOLD).next_to(snail, m.DOWN, buff=0.2)
        temp = cite("temperature > 0").next_to(ask, m.DOWN, buff=0.2)
        source = m.VGroup(snail, ask)

        ys = (1.4, 0.5, -0.4, -1.3, -2.2)
        chains = [chain_card(a, 10 + i, 4.4).move_to([-1.75, y, 0]) for i, (a, y) in enumerate(zip(ANSWERS, ys))]
        start = np.array([-5.2, -0.3, 0])
        fans = [
            m.Arrow(start, c.get_left(), buff=0.12, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.12)
            for c in chains
        ]

        title = text("majority vote", 28, AMBER, weight=m.BOLD).move_to([4.15, 1.55, 0])
        source_note = cite("Wang et al., 2022").next_to(title, m.DOWN, buff=0.15)
        rows = {}
        for value, y, color in ((9, 0.2, GOOD), (10, -0.6, BAD), (8, -1.4, BAD)):
            rows[value] = text(f"day {value}", 26, color, weight=m.BOLD).move_to([0, y, 0]).align_to([1.75, 0, 0], m.LEFT)
        votes, fly = {v: m.VGroup() for v in rows}, []
        for c, a in zip(chains, ANSWERS):
            k = len(votes[a])
            sq = m.RoundedRectangle(corner_radius=0.08, width=0.42, height=0.42).set_stroke(AMBER, width=2.5).set_fill(AMBER_FILL, opacity=1)
            sq.move_to([3.55 + 0.55 * k, rows[a].get_y(), 0])
            votes[a].add(sq)
            fly.append(m.FadeIn(sq, target_position=c[2]))
        winner = pill("✓  day 9 · 3 of 5 votes", GOOD, GOOD_FILL, 26).move_to([4.15, -2.45, 0])

        self.say(
            "[Sampled with some temperature, as in episode thirty-three], [each chain comes out a little different.]",
            m.AnimationGroup(
                self.new_header("SELF-CONSISTENCY", "Sample several chains, then vote"),
                m.FadeIn(source, shift=0.2 * m.UP), m.FadeIn(temp),
            ),
            m.LaggedStart(*[m.AnimationGroup(m.GrowArrow(a), m.FadeIn(c, shift=0.2 * m.RIGHT)) for a, c in zip(fans, chains)], lag_ratio=0.2),
        )
        good = [c for c, a in zip(chains, ANSWERS) if a == DAY]
        bad = [c for c, a in zip(chains, ANSWERS) if a != DAY]
        self.say(
            "[Here, for example, three end at day nine], [the others at ten, or eight.]",
            m.AnimationGroup(*[recolor_chip(c, GOOD, GOOD_FILL) for c in good]),
            m.AnimationGroup(*[recolor_chip(c, BAD, BAD_FILL) for c in bad]),
        )
        self.say(
            "[Self-consistency takes a majority vote on the final answers]: [nine wins, with three votes out of five.]",
            m.AnimationGroup(
                m.FadeIn(title, shift=0.2 * m.UP), m.FadeIn(source_note), m.FadeIn(m.VGroup(*rows.values())),
                m.LaggedStart(*fly, lag_ratio=0.15),
            ),
            m.AnimationGroup(m.FadeIn(winner, scale=1.2), m.Circumscribe(m.VGroup(rows[9], votes[9]), color=GOOD, buff=0.12)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the math

    def voting_math(self):
        self.next_section(
            "Why voting helps",
            notes="Exact binomial: chains independent, each right with p = 0.6; worst case all wrong chains "
            "agree, so the right answer needs a strict majority (odd N). N = 1, 5, 11, 41, 101 give 60, 68, 75, "
            "90, 98% (asserted in the scene file).",
        )
        left = -6.55
        a1 = text("each chain is right with p = 0.6", 26, INK, t2c={"p = 0.6": GOOD})
        a2 = text("independently of the others", 22, MUTED)
        a3 = text("worst case: all wrong chains agree", 22, MUTED)
        for mob, y in ((a1, 2.2), (a2, 1.74), (a3, 1.32)):
            mob.move_to([0, y, 0]).align_to([left, 0, 0], m.LEFT)

        f = m.MathTex(r"P_{\text{vote}} =", r"\sum_{k > N/2}", r"\binom{N}{k}", r"\,p^{k}", r"\,(1-p)^{N-k}", font_size=40, color=INK)
        f.move_to([0, 0.3, 0]).align_to([left, 0, 0], m.LEFT)
        for part, color in zip(f[1:], (SKY, AMBER, GOOD, BAD)):
            part.set_color(color)

        def row(tex_s, color, words, y):
            sym = tex(tex_s, 28, color)
            sym.move_to([-5.75, y, 0])
            words = text(words, 22, MUTED).move_to([0, y, 0]).align_to([-4.85, 0, 0], m.LEFT)
            return m.VGroup(sym, words)

        legend = [
            row(r"\sum", SKY, "add up every k above half", -0.95),
            row(r"\binom{N}{k}", AMBER, "ways to pick which k are right", -1.67),
            row(r"p^{k}", GOOD, "chance those k are right", -2.39),
            row(r"(1-p)^{N-k}", BAD, "chance the rest are wrong", -3.08),
        ]

        chart = Chart(7.5, 1.15, 5.5, 4.2, [3.95, -0.5, 0])
        self.vote_chart = chart

        def u(n):
            return math.log2(n) + 0.5

        frame = m.VGroup(
            chart.ax,
            chart.ticks(xs=[u(n) for n in (1, 3, 10, 30, 100)], ys=(0.5, 1.0)),
            *[chart.xlabel(u(n), str(n)) for n in (1, 3, 10, 30, 100)],
            *[chart.ylabel(v, f"{round(100 * v)}%") for v in (0, 0.5, 1.0)],
            text("chains sampled (log scale)", 22, MUTED).move_to(chart.p(3.75, 0) + 0.8 * m.DOWN),
            text("vote is right", 22, MUTED).next_to(chart.p(0, 1.15), m.RIGHT, buff=0.25),
        )

        def seg(a, b, p=P, color=GOOD):
            return chart.line([(u(n), vote_right(p, n)) for n in NS if a <= n <= b], color, 4)

        def mark(n, p=P, color=GOOD, where=m.UP):
            dot = m.Dot(chart.p(u(n), vote_right(p, n)), radius=0.075, color=color).set_z_index(3)
            label = text(f"{round(100 * vote_right(p, n))}%", 22, color, weight=m.BOLD).next_to(dot, where, buff=0.12)
            return dot, label

        dots = {n: mark(n, where=m.DOWN if n == 101 else m.UP) for n in MARKS}
        self.vote_curve = m.VGroup(seg(1, 5), seg(5, 11), seg(11, 41), seg(41, 101))
        self.vote_left = m.VGroup(a1, a2, a3, *f, *legend)
        self.u, self.mark = u, mark

        self.say(
            "[How much does voting help? Say each chain is right sixty percent of the time], [independently of the others.]",
            m.AnimationGroup(
                self.new_header("THE MATH", "Why voting helps"),
                m.FadeIn(a1, shift=0.2 * m.UP), m.Create(chart.ax), m.FadeIn(frame[1:]),
            ),
            m.FadeIn(a2, shift=0.2 * m.UP),
        )
        self.say(
            "[To be safe, say all the wrong chains agree on one wrong answer]: [then day nine needs more than half the votes.]",
            m.FadeIn(a3, shift=0.2 * m.UP),
            m.AnimationGroup(m.Write(m.VGroup(f[0], f[1])), m.FadeIn(legend[0], shift=0.2 * m.DOWN)),
        )
        self.say(
            "[The binomial distribution counts the ways to choose which chains are right.]",
            m.AnimationGroup(m.Write(f[2]), m.FadeIn(legend[1], shift=0.2 * m.DOWN)),
        )
        self.say(
            "[Times the chance those are right], [times the chance the rest are wrong.]",
            m.AnimationGroup(m.Write(f[3]), m.FadeIn(legend[2], shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.Write(f[4]), m.FadeIn(legend[3], shift=0.2 * m.DOWN)),
        )
        c = self.vote_curve
        self.say(
            "[One chain: sixty percent.] [Five chains: sixty-eight.] [Eleven: seventy-five.]",
            m.AnimationGroup(m.FadeIn(dots[1][0], scale=0.5), m.FadeIn(dots[1][1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(c[0]), m.FadeIn(dots[5][0]), m.FadeIn(dots[5][1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(c[1]), m.FadeIn(dots[11][0]), m.FadeIn(dots[11][1], shift=0.15 * m.UP)),
        )
        self.say(
            "[Forty-one chains: ninety percent], [and it keeps climbing toward a hundred.]",
            m.AnimationGroup(m.Create(c[2]), m.FadeIn(dots[41][0]), m.FadeIn(dots[41][1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(c[3]), m.FadeIn(dots[101][0]), m.FadeIn(dots[101][1], shift=0.15 * m.UP)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section(
            "When errors agree",
            notes="Voting amplifies the most common answer. If the trap answer is more likely than the right "
            "one (right with p = 0.4), the same formula falls: N = 41 gives 10%. Correlated samples gain "
            "little for the same reason.",
        )
        chart, u, mark = self.vote_chart, self.u, self.mark
        left = -6.55

        def case(title_words, sub_words, color, y, t2c):
            box = card(6.1, 1.35, color, CARD, stroke=2).move_to([-3.5, y, 0])
            words = m.VGroup(
                text(title_words, 26, INK, weight=m.BOLD, t2c=t2c),
                text(sub_words, 22, color, weight=m.SEMIBOLD),
            ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
            words.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            return m.VGroup(box, words)

        ok = case("right 60% · trap 40%", "the vote climbs toward 100%", GOOD, 1.55, {"right 60%": GOOD, "trap 40%": BAD})
        trap = case("right 40% · trap 60%", "the vote sinks toward 0%", BAD, -0.1, {"right 40%": GOOD, "trap 60%": BAD})
        takeaway = m.VGroup(
            text("Voting amplifies the most common", 26, INK, weight=m.BOLD),
            text("answer, right or wrong.", 26, INK, weight=m.BOLD),
        ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        takeaway.move_to([0, -1.95, 0]).align_to([left + 0.05, 0, 0], m.LEFT)

        trap_curve = chart.line([(u(n), vote_right(1 - P, n)) for n in NS], BAD, 4)
        d1 = mark(1, 1 - P, BAD, m.DOWN)
        d41 = mark(41, 1 - P, BAD, m.UP)

        self.say(
            "[But voting only helps if the errors don't agree.]",
            m.AnimationGroup(
                self.new_header("THE CATCH", "When the errors agree", BAD, BAD_FILL),
                m.FadeOut(self.vote_left), m.FadeIn(ok, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[If the trap answer, day ten, comes up more often than day nine], [voting locks in the mistake.]",
            m.FadeIn(trap, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(d1[0]), m.FadeIn(d1[1]), m.Create(trap_curve, run_time=2.0)),
        )
        self.say(
            "[At forty percent per chain, forty-one votes are right only one time in ten.] "
            "[Voting amplifies the most common answer, right or wrong.]",
            m.AnimationGroup(m.FadeIn(d41[0], scale=0.5), m.FadeIn(d41[1], scale=1.2)),
            m.FadeIn(takeaway, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ best-of-N

    def best_of_n(self):
        self.next_section(
            "Best-of-N with a verifier",
            notes="Verifiers for math word problems: Cobbe et al., 2021 (GSM8K). With a perfect checker, "
            "five chains fail only if all are wrong: 1 - 0.4^5 = 98.98%, versus 68% for voting. The scores "
            "are illustrative. Learned verifiers are imperfect and can be gamed (reward hacking, Episode 46).",
        )
        ys = (1.7, 0.85, 0.0, -0.85, -1.7)
        chains = [chain_card(a, 10 + i, 3.9, 0.58).move_to([-4.7, y, 0]) for i, (a, y) in enumerate(zip(ANSWERS, ys))]
        for c, a in zip(chains, ANSWERS):
            color, fill = (GOOD, GOOD_FILL) if a == DAY else (BAD, BAD_FILL)
            c[2][0].set_stroke(color).set_fill(fill, opacity=1)

        check = icon(recolored_ep43("check", TEAL, TEAL_FILL), 0.5)
        verifier = m.VGroup(check, text("verifier", 26, TEAL, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18).move_to([-1.55, 2.4, 0])
        x0, track_w = -2.5, 1.3
        tracks, bars, values = [], [], []
        for c, s in zip(chains, SCORES):
            y = c.get_y()
            tracks.append(m.Rectangle(width=track_w, height=0.22).set_stroke(EDGE, width=2).set_fill(CARD, opacity=1).move_to([x0, y, 0], aligned_edge=m.LEFT))
            bars.append(bar(track_w * s, TEAL, 0.22).move_to([x0, y, 0], aligned_edge=m.LEFT).set_z_index(1))
            values.append(text(f"{s:.2f}", 20, TEAL, weight=m.SEMIBOLD).move_to([0, y, 0]).align_to([x0 + track_w + 0.18, 0, 0], m.LEFT))
        best = SCORES.index(max(SCORES))
        span = m.VGroup(chains[best], values[best])
        keep = m.RoundedRectangle(corner_radius=0.2, width=span.width + 0.3, height=chains[best].height + 0.24).move_to(span)
        keep.set_stroke(GOOD, width=4).set_fill(BG, opacity=0)
        keep_pill = pill("keep the best: day 9", GOOD, GOOD_FILL, 24).move_to([-3.75, -2.7, 0])
        source = cite("Cobbe et al., 2021").next_to(keep_pill, m.DOWN, buff=0.18)

        right = 0.95
        perfect = text("with a perfect checker:", 24, MUTED).move_to([0, 1.95, 0]).align_to([right, 0, 0], m.LEFT)
        g = m.MathTex(r"P_{\text{best}} =", r"1 -", r"(1-p)^{N}", font_size=44, color=INK)
        g.move_to([0, 1.15, 0]).align_to([right, 0, 0], m.LEFT)
        g[2].set_color(BAD)
        legend = m.VGroup(tex(r"(1-p)^{N}", 32, BAD), text("all N chains are wrong", 22, MUTED)).arrange(m.RIGHT, buff=0.3)
        legend.move_to([0, 0.25, 0]).align_to([right + 0.2, 0, 0], m.LEFT)
        numbers = m.VGroup(text("N = 5:", 24, MUTED), tex(r"1 - 0.4^{5} \approx 99\%", 44, GOOD)).arrange(m.RIGHT, buff=0.3)
        numbers.move_to([0, -0.75, 0]).align_to([right, 0, 0], m.LEFT)
        versus = text("majority vote, 5 chains: 68%", 24, AMBER, weight=m.SEMIBOLD).move_to([0, -1.6, 0]).align_to([right, 0, 0], m.LEFT)
        caveat = pill("✗  learned verifiers can be fooled", BAD, BAD_FILL, 24).move_to([0, -2.7, 0]).align_to([right, 0, 0], m.LEFT)

        self.say(
            "[Voting ignores the reasoning.] [A verifier reads each chain and scores it], [and we keep the best one.]",
            m.AnimationGroup(
                self.new_header("BEST-OF-N", "Let a verifier pick"),
                m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.RIGHT) for c in chains], lag_ratio=0.12),
            ),
            m.AnimationGroup(
                m.FadeIn(verifier, shift=0.2 * m.DOWN), m.FadeIn(source),
                m.LaggedStart(
                    *[m.AnimationGroup(m.FadeIn(t), m.GrowFromEdge(b, m.LEFT), m.FadeIn(v)) for t, b, v in zip(tracks, bars, values)],
                    lag_ratio=0.15,
                ),
            ),
            m.AnimationGroup(m.Create(keep), m.FadeIn(keep_pill, scale=1.15)),
        )
        self.say(
            "[With a perfect checker, we only lose if all five chains are wrong]: [point four to the fifth power.]",
            m.AnimationGroup(m.FadeIn(perfect), m.Write(g)),
            m.FadeIn(legend, shift=0.2 * m.DOWN),
        )
        self.say(
            "[So best of five is right about ninety-nine percent of the time], [against sixty-eight for voting.]",
            m.FadeIn(numbers, shift=0.2 * m.UP),
            m.FadeIn(versus, shift=0.2 * m.UP),
        )
        self.say(
            "[But real verifiers are learned models, and they can be fooled.]",
            m.Succession(m.FadeIn(caveat, scale=1.15), m.Wiggle(caveat, scale_value=1.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ search over steps

    def search(self):
        self.next_section(
            "Search over steps",
            notes="Process reward models score each step (Lightman et al., 2023, 'Let's Verify Step by Step'). "
            "Shown: greedy step-level search (beam width 1). Scores illustrative.",
        )
        root_box = card(1.8, 1.45, SKY, CARD, stroke=2).move_to([-5.95, -0.3, 0])
        root_body = m.VGroup(icon("ep43-snail", 0.5), text("which day?", 22, SKY, weight=m.BOLD)).arrange(m.DOWN, buff=0.15).move_to(root_box)
        root = m.VGroup(root_box, root_body)

        def step(words, score, x, y, width, color=VIOLET):
            box = card(width, 0.8, color, CARD, radius=0.18, stroke=2).move_to([x, y, 0])
            label = code(words, 22).move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            value = text(f"{score:.1f}", 22, TEAL, weight=m.BOLD).move_to(box).align_to(box, m.RIGHT).shift(0.25 * m.LEFT)
            return m.VGroup(box, label), value

        def edge(a, b):
            return m.Line(a.get_right(), b.get_left()).set_stroke(FAINT, width=3)

        l1 = [step("+2 m a day", 0.8, -2.95, 1.3, 3.3), step("20 ÷ 2 = 10", 0.1, -2.95, -0.3, 3.3), step("day 1: 5 m", 0.5, -2.95, -1.9, 3.3)]
        e1 = [edge(root_box, s[0][0]) for s in l1]
        l2 = [step("8 nights: 16 m", 0.9, 1.1, 1.3, 3.85), step("10 days: 20 m", 0.2, 1.1, -0.15, 3.85)]
        e2 = [edge(l1[0][0][0], s[0][0]) for s in l2]
        final_box = card(2.3, 0.8, GOOD, GOOD_FILL, radius=0.18, stroke=2.5).move_to([4.75, 1.3, 0])
        final = m.VGroup(final_box, text("✓  day 9", 26, INK, weight=m.BOLD).move_to(final_box))
        e3 = edge(l2[0][0][0], final_box)
        for mob in (*l2[1], e2[1]):
            mob.set_opacity(0.35)
        pruned = text("pruned", 20, BAD, weight=m.SEMIBOLD).next_to(l1[1][0], m.DOWN, buff=0.12)
        legend = text("scores: a process reward model (illustrative)", 22, TEAL).move_to([0, -2.95, 0])
        source = cite("Lightman et al., 2023").next_to(legend, m.DOWN, buff=0.15)

        self.say(
            "[We can also score every step, not just the final answer]: [that's a process reward model.]",
            m.AnimationGroup(self.new_header("SEARCH", "Score every step, not just the answer"), m.FadeIn(root, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(legend, shift=0.2 * m.UP), m.FadeIn(source)),
        )
        self.say(
            "[At each step, the model proposes a few next steps], [and the reward model gives each one a score.]",
            m.LaggedStart(*[m.AnimationGroup(m.Create(e), m.FadeIn(s[0], shift=0.2 * m.RIGHT)) for e, s in zip(e1, l1)], lag_ratio=0.25),
            m.LaggedStart(*[m.FadeIn(s[1], scale=1.3) for s in l1], lag_ratio=0.2),
        )
        self.say(
            "[The search keeps the best], [and prunes the tempting shortcut before it can spoil the answer.]",
            m.AnimationGroup(l1[0][0][0].animate.set_stroke(GOOD), e1[0].animate.set_stroke(GOOD)),
            m.AnimationGroup(
                *[mob.animate.set_opacity(0.35) for mob in (*l1[1], *l1[2], e1[1], e1[2])],
                m.FadeIn(pruned, shift=0.1 * m.DOWN),
            ),
        )
        self.say(
            "[It extends the best step the same way], [until it reaches day nine.]",
            m.AnimationGroup(*[m.AnimationGroup(m.Create(e), m.FadeIn(s[0], shift=0.2 * m.RIGHT), m.FadeIn(s[1])) for e, s in zip(e2, l2)]),
            m.AnimationGroup(m.Create(e3), m.FadeIn(final, shift=0.2 * m.RIGHT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ reasoning models

    def reasoning_models(self):
        self.next_section(
            "Reasoning models",
            notes="RL with verifiable rewards (term: Lambert et al., 2024, Tülu 3): rewards from a program — "
            "an answer key for math, unit tests for code. OpenAI o1 (Sept 2024): 'large-scale reinforcement "
            "learning' teaches the model to use its chain of thought; reward details unpublished. DeepSeek-R1 "
            "(DeepSeek-AI, Jan 2025; Nature, Sept 2025): rule-based accuracy and format rewards (GRPO); in "
            "R1-Zero (RL with no supervised examples) the average response length grew during training and "
            "reflection emerged. The length bars are schematic. " + FACTS,
        )
        y = 1.35
        task_box = card(2.3, 1.9, SKY, CARD, stroke=2).move_to([-5.75, y, 0])
        task_body = m.VGroup(
            icon("ep43-snail", 0.45),
            text("math: answer key", 20, MUTED),
            text("code: unit tests", 20, MUTED),
        ).arrange(m.DOWN, buff=0.14).move_to(task_box)
        task = m.VGroup(task_box, task_body)
        model = icon("llm", 1.05).move_to([-3.15, y, 0])
        to_model = arrow(task_box.get_right(), model.get_left(), MUTED, buff=0.1)
        tokens = token_row(9, size=0.3, buff=0.08).move_to([0, y, 0]).align_to([-2.35, 0, 0], m.LEFT)
        chain_label = text("long chain of thought", 20, AMBER, weight=m.SEMIBOLD).next_to(tokens, m.UP, buff=0.2)
        answer = chip("9", VIOLET, VIOLET_FILL, width=0.7, height=0.55, size=26).move_to([1.55, y, 0])
        check_box = card(3.2, 1.9, TEAL, CARD, stroke=2).move_to([4.85, y, 0])
        check_title = m.VGroup(icon(recolored_ep43("check", TEAL, TEAL_FILL), 0.45), text("checker", 24, TEAL, weight=m.BOLD)).arrange(m.RIGHT, buff=0.15)
        check_lines = m.VGroup(text("does the answer match?", 20, MUTED), text("do the tests pass?", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        m.VGroup(check_title, check_lines).arrange(m.DOWN, buff=0.18).move_to(check_box)
        to_check = arrow(answer.get_right(), check_box.get_left(), MUTED, buff=0.12)

        reward = pill("reward = 1", GOOD, GOOD_FILL, 24).move_to([4.85, -0.35, 0])
        rlvr = cite("RL with verifiable rewards · Lambert et al., 2024").move_to([0, -0.95, 0]).align_to([6.45, 0, 0], m.RIGHT)
        corner = np.array([model.get_x(), reward.get_y(), 0])
        back = m.Line(reward.get_left() + 0.1 * m.LEFT, corner).set_stroke(GOOD, width=4)
        up = m.Arrow(corner, model.get_bottom(), buff=0.1, color=GOOD, stroke_width=4, max_tip_length_to_length_ratio=0.3)
        update = text("make chains like this more likely", 22, GOOD, weight=m.SEMIBOLD).move_to([0.6, -0.02, 0])

        len_title = text("R1-Zero: chain length in training", 22, MUTED).move_to([0, -1.55, 0]).align_to([-6.55, 0, 0], m.LEFT)
        len_note = cite("schematic · DeepSeek-AI, 2025").next_to(len_title, m.RIGHT, buff=0.3)
        names = m.VGroup(*[text(s, 20, MUTED) for s in ("early", "middle", "late")])
        bars = m.VGroup()
        for i, (name, w) in enumerate(zip(names, (1.1, 2.3, 3.9))):
            yy = -2.05 - 0.4 * i
            name.move_to([0, yy, 0]).align_to([-6.55, 0, 0], m.LEFT)
            bars.add(bar(w, AMBER, 0.24).move_to([-5.3, yy, 0], aligned_edge=m.LEFT))
        quote = text("“Wait, let me check that step…”", 22, VIOLET, weight=m.SEMIBOLD).move_to([0, -3.35, 0]).align_to([-6.55, 0, 0], m.LEFT)

        o1 = pill("OpenAI o1 · 2024", VIOLET, VIOLET_FILL, 24).move_to([4.6, -1.8, 0])
        o1_note = cite("large-scale RL on chains of thought").next_to(o1, m.DOWN, buff=0.12)
        r1 = pill("DeepSeek-R1 · 2025", VIOLET, VIOLET_FILL, 24).move_to([4.6, -2.85, 0])
        r1_note = cite("an open recipe, rule-based rewards").next_to(r1, m.DOWN, buff=0.12)

        self.say(
            "[Reasoning models go one step further: they are trained to think.]",
            m.AnimationGroup(
                self.new_header("REASONING MODELS", "Trained with rewards you can check"),
                m.FadeIn(task, shift=0.2 * m.RIGHT), m.GrowArrow(to_model), m.FadeIn(model, scale=0.7),
            ),
        )
        self.say(
            "[The model writes a long chain of thought], [then an answer.]",
            m.AnimationGroup(m.FadeIn(chain_label), m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.RIGHT) for t in tokens], lag_ratio=0.25)),
            m.FadeIn(answer, scale=1.2),
        )
        self.say(
            "[The reward is verifiable: a program checks it.] [Does the answer match the key? Do the tests pass?]",
            m.AnimationGroup(m.GrowArrow(to_check), m.FadeIn(check_box), m.FadeIn(check_title)),
            m.FadeIn(check_lines, shift=0.15 * m.UP),
        )
        self.say(
            "[A correct answer earns a reward of one], [and training makes chains like that more likely.]",
            m.AnimationGroup(m.FadeIn(reward, scale=1.2), m.FadeIn(rlvr), answer[0].animate.set_stroke(GOOD).set_fill(GOOD_FILL, opacity=1)),
            m.AnimationGroup(
                m.Create(back), m.GrowArrow(up), m.FadeIn(update, shift=0.15 * m.UP),
                model.animate(rate_func=m.there_and_back, run_time=1.2).scale(1.12),
            ),
        )
        self.say(
            "[In one experiment with no worked examples to copy, the chains grew longer during training], "
            "[and the model learned to re-check its steps.]",
            m.AnimationGroup(
                m.FadeIn(len_title), m.FadeIn(len_note), m.FadeIn(names),
                m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in bars], lag_ratio=0.4),
            ),
            m.FadeIn(quote, shift=0.2 * m.UP),
        )
        self.say(
            "[OpenAI's first reasoning model, in twenty twenty-four, was trained with large-scale reinforcement learning.] "
            "[A year later, an open model's recipe was published, with rewards checked by simple rules.]",
            m.AnimationGroup(m.FadeIn(o1, shift=0.2 * m.UP), m.FadeIn(o1_note)),
            m.AnimationGroup(m.FadeIn(r1, shift=0.2 * m.UP), m.FadeIn(r1_note)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the price of thinking

    def price(self):
        self.next_section(
            "The price of thinking",
            notes="Illustrative: 1,000 problems with one-attempt success rates drawn from Beta(0.5, 1) "
            "(numpy seed 43), a perfect checker, 1,000 tokens per attempt; solved = mean of 1 - (1 - p)^n: "
            "36%, 72%, 89% at 1, 8, 64 attempts. OpenAI (2024) reported o1's accuracy rising smoothly with "
            "test-time compute on a log axis. " + FACTS,
        )
        chart = Chart(9.5, 1.15, 6.2, 4.2, [-3.1, -0.45, 0])

        def u(n):
            return math.log2(n) + 1.1

        frame = m.VGroup(
            chart.ax,
            chart.ticks(xs=[u(n) for n in (1, 4, 16, 64, 256)], ys=(0.5, 1.0)),
            *[chart.xlabel(u(n), f"{n}k") for n in (1, 4, 16, 64, 256)],
            *[chart.ylabel(v, f"{round(100 * v)}%") for v in (0, 0.5, 1.0)],
            text("thinking tokens per problem (log scale)", 22, MUTED).move_to(chart.p(4.75, 0) + 0.8 * m.DOWN),
            text("problems solved", 22, MUTED).next_to(chart.p(0, 1.15), m.RIGHT, buff=0.25),
        )
        caption = cite("for example: 1,000 problems, easy to very hard · a perfect checker · 1k tokens per attempt")
        caption.move_to([0, 2.35, 0]).align_to([-6.55, 0, 0], m.LEFT)

        def seg(a, b):
            return chart.line([(u(n), solved(n)) for n in range(a, b + 1)], GOOD, 4)

        curve = [seg(1, 8), seg(8, 64), seg(64, 256)]
        dots = {}
        for n in COST_MARKS:
            dot = m.Dot(chart.p(u(n), solved(n)), radius=0.075, color=GOOD).set_z_index(3)
            label = text(f"{COST_MARKS[n]}%", 22, GOOD, weight=m.BOLD).next_to(dot, m.DOWN if n == 1 else m.UP + 0.6 * m.LEFT, buff=0.14)
            dots[n] = (dot, label)
        rises, gains = [], []
        for a, b, words in ((1, 8, "+36 points"), (8, 64, "+17 points")):
            corner = chart.p(u(b), solved(a))
            run = m.DashedLine(chart.p(u(a), solved(a)), corner, dash_length=0.08).set_stroke(FAINT, width=2.5)
            lift = m.Line(corner, chart.p(u(b), solved(b))).set_stroke(GOOD, width=3)
            rises.append(m.VGroup(run, lift))
            gains.append(text(words, 20, GOOD, weight=m.SEMIBOLD).next_to(lift, m.RIGHT, buff=0.12))

        title = text("tokens spent", 26, AMBER, weight=m.BOLD).move_to([4.1, 1.8, 0])
        rows = []
        for n, y in zip(COST_MARKS, (0.9, -0.1, -1.1)):
            name = text(f"{n}k tokens", 22, MUTED).move_to([0, y, 0]).align_to([2.35, 0, 0], m.RIGHT)
            b = bar(3.3 * n / 64, AMBER, 0.3).move_to([2.6, y, 0], aligned_edge=m.LEFT)
            times = text(f"×{n}", 22, AMBER, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.15)
            rows.append((name, b, times))
        note = m.VGroup(
            text("reasoning models: accuracy rises roughly", 20, MUTED),
            text("with the log of thinking time · OpenAI, 2024", 20, MUTED),
        ).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        note.move_to([0, -2.35, 0]).align_to([1.3, 0, 0], m.LEFT)

        self.say(
            "[More thinking helps, but it isn't free.] [For example, take a thousand problems, from easy to very hard, and a perfect checker.]",
            m.AnimationGroup(self.new_header("THE PRICE", "More thinking, smaller gains"), m.Create(chart.ax), m.FadeIn(frame[1:])),
            m.FadeIn(caption),
        )
        self.say(
            "[One attempt each solves thirty-six percent.] [Eight attempts: seventy-two.] [Sixty-four: eighty-nine.]",
            m.AnimationGroup(m.FadeIn(dots[1][0], scale=0.5), m.FadeIn(dots[1][1]), m.FadeIn(title), m.FadeIn(rows[0][0]), m.GrowFromEdge(rows[0][1], m.LEFT)),
            m.AnimationGroup(m.Create(curve[0]), m.FadeIn(dots[8][0]), m.FadeIn(dots[8][1]), m.FadeIn(rows[1][0]), m.GrowFromEdge(rows[1][1], m.LEFT)),
            m.AnimationGroup(m.Create(curve[1]), m.FadeIn(dots[64][0]), m.FadeIn(dots[64][1]), m.FadeIn(rows[2][0]), m.GrowFromEdge(rows[2][1], m.LEFT)),
        )
        self.say(
            "[Each step costs eight times the tokens], [and buys a smaller gain.]",
            m.LaggedStart(*[m.FadeIn(r[2], shift=0.15 * m.RIGHT) for r in rows], lag_ratio=0.3),
            m.LaggedStart(*[m.AnimationGroup(m.Create(r), m.FadeIn(g, shift=0.15 * m.UP)) for r, g in zip(rises, gains)], lag_ratio=0.4),
        )
        self.say(
            "[Reasoning models show a similar shape: accuracy rises roughly with the logarithm of thinking time.]",
            m.AnimationGroup(m.Create(curve[2]), m.FadeIn(note, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Think (chain of thought), vote (self-consistency), verify (verifiers, search, verifiable rewards).")
        halo = glow(PINK, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Think", "Vote", "Verify"), (VIOLET, AMBER, TEAL))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.45 * m.UP)
        tagline = m.VGroup(
            text("A bigger model is one road to a better answer.", 30, MUTED),
            text("Thinking longer is the other.", 30, MUTED),
        ).arrange(m.DOWN, buff=0.2).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [think], [vote], [and verify.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A bigger model is one road to a better answer. Thinking longer is the other.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
