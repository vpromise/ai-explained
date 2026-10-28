"""AI, Explained · Episode 6 · Measuring a Model

Running example: a screening test for a rare disease, given to 1,000 people of whom 10 are
sick. A lazy model that always says "healthy" is 99% accurate and finds nobody. A real model
flags 8 of the 10 sick people (it misses 2) and 40 healthy people by mistake:

                    flagged   not flagged
    really sick     TP = 8    FN = 2        (10)
    really healthy  FP = 40   TN = 950      (990)

    accuracy  = (8 + 950) / 1000 = 95.8%   (lower than the lazy model's 99%, yet useful)
    precision = 8 / (8 + 40) = 8/48 = 1/6 ≈ 17%
    recall    = 8 / (8 + 2)  = 8/10 = 80%
    F1        = 2PR / (P + R) = 8/29 ≈ 0.28   (plain average ≈ 0.48)

Threshold trade-off: healthy risk scores spread like a bell curve (mean 0.25, sd 0.15), the
ten sick scores are listed below; the three thresholds are solved so that exactly
140 / 40 / 5 healthy people score above them:
    low threshold:  TP 10, FP 140 → precision 10/150 ≈ 7%,  recall 100%
    our threshold:  TP 8,  FP 40  → precision 8/48   ≈ 17%, recall 80%
    high threshold: TP 5,  FP 5   → precision 5/10   = 50%, recall 50%

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
TP, FN = 8, 2  # of the ten sick: found, missed
FP = 40  # healthy people flagged by mistake
TN = HEALTHY - FP  # 950
ACCURACY = (TP + TN) / PEOPLE
PRECISION = TP / (TP + FP)
RECALL = TP / (TP + FN)
F1 = 2 * PRECISION * RECALL / (PRECISION + RECALL)
AVERAGE = (PRECISION + RECALL) / 2
assert TP + FN == SICK and FP + TN == HEALTHY and TP + FN + FP + TN == PEOPLE
assert round(100 * ACCURACY, 1) == 95.8 and round(100 * ACCURACY) == 96
assert abs(PRECISION - 1 / 6) < 1e-12 and round(100 * PRECISION) == 17 and RECALL == 0.8
assert abs(F1 - 8 / 29) < 1e-12 and round(100 * F1) == 28 and round(100 * AVERAGE) == 48
assert round(2 * 0.17 * 0.80 / (0.17 + 0.80), 2) == 0.28  # the rounded arithmetic on screen

# the threshold picture
MU, SIGMA = 0.25, 0.15
SICK_SCORES = [0.435, 0.485, 0.54, 0.575, 0.61, 0.67, 0.73, 0.79, 0.85, 0.92]


def false_alarms(t):
    """Healthy people scoring above the threshold t (a smooth count)."""
    return HEALTHY * 0.5 * math.erfc((t - MU) / (SIGMA * math.sqrt(2)))


def threshold_for(count):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if false_alarms(mid) > count else (lo, mid)
    return (lo + hi) / 2


def counts(t):
    return sum(s > t for s in SICK_SCORES), round(false_alarms(t))


T_LOW, T_MID, T_HIGH = threshold_for(140), threshold_for(FP), threshold_for(5)
assert counts(T_MID) == (TP, FP) and counts(T_LOW) == (10, 140) and counts(T_HIGH) == (5, 5)
assert round(100 * 10 / 150) == 7 and 5 / (5 + 5) == 0.5

# the crowd: 1,000 dots
COLS, ROWS, STEP = 40, 25, 0.13
GRID_CENTER = np.array([-4.0, -0.15, 0])
DOT_R, RING_R = 0.04, 0.082


def crowd():
    """Who is sick, who is missed, which healthy people get flagged: spread out, no two
    marked people next to each other (so their rings never touch)."""
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


def person(i):
    r, c = divmod(i, COLS)
    pos = GRID_CENTER + np.array([(c - (COLS - 1) / 2) * STEP, ((ROWS - 1) / 2 - r) * STEP, 0])
    return m.Dot(pos, radius=DOT_R, color=MUTED).set_fill(MUTED, opacity=0.55).set_z_index(3)


def ring(dot, color, radius=RING_R, width=2.5):
    return m.Circle(radius=radius).set_stroke(color, width=width).move_to(dot.get_center()).set_z_index(3)


def outline(mob, color, buff=0.06, radius=0.22):
    box = m.RoundedRectangle(corner_radius=radius, width=mob.width + 2 * buff, height=mob.height + 2 * buff)
    return box.set_stroke(color, width=5).set_fill(BG, opacity=0).move_to(mob)


# the confusion matrix
CELL_W, CELL_H, CELL_GAP = 2.55, 1.55, 0.14
COL_X = (2.55, 2.55 + CELL_W + CELL_GAP)  # flagged, not flagged
ROW_Y = (0.4, 0.4 - CELL_H - CELL_GAP)  # really sick, really healthy
CELLS = {  # name: (row, column, count, words, colour, fill)
    "TP": (0, 0, TP, "true positives", GOOD, GOOD_FILL),
    "FN": (0, 1, FN, "false negatives", BAD, BAD_FILL),
    "FP": (1, 0, FP, "false positives", BAD, BAD_FILL),
    "TN": (1, 1, TN, "true negatives", GOOD, GOOD_FILL),
}
LEFT_X = -6.4  # left edge of the formulas beside the matrix


def landing(dots, center, cols, step, scale=1.0):
    """Copies of `dots` packed into a block around `center` (where they land in a cell)."""
    rows = math.ceil(len(dots) / cols)
    width = min(cols, len(dots))
    out = m.VGroup()
    for k, d in enumerate(dots):
        r, c = divmod(k, cols)
        spot = center + np.array([(c - (width - 1) / 2) * step, ((rows - 1) / 2 - r) * step, 0])
        out.add(d.copy().scale(scale).move_to(spot))
    return out


def recolored_ep06(name, color, fill):
    """kit.recolored(), but written as assets/ep06-<name>-<colour>.svg (episode-owned files)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    base = name if name.startswith("ep06-") else f"ep06-{name}"
    variant = f"{base}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


class Ep06MeasuringAModel(Episode):
    number = 6

    def lesson(self):
        self.hook()
        self.example()
        self.trap()
        self.matrix()
        self.precision()
        self.recall()
        self.trade_off()
        self.one_number()
        self.which_mistake()
        self.recap()

    # ------------------------------------------------------------------ helpers

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
        self.next_section("Hook", notes="A model that is 99% accurate can still be useless.")
        halo = glow(SKY, radius=3.4).move_to(1.0 * m.UP)
        hero = text("99%", 170, INK, weight=m.BOLD, gradient=(SKY, VIOLET)).move_to([0, 1.05, 0])
        sub = text("accurate", 40, MUTED, weight=m.SEMIBOLD).next_to(hero, m.DOWN, buff=0.3)
        useless = pill("…and useless?", BAD, BAD_FILL, 30).move_to([0, -1.45, 0])
        today = m.VGroup(
            chip("why accuracy can lie", SKY, SKY_FILL, width=4.6, height=0.8, size=26),
            chip("how to measure a model", SKY, SKY_FILL, width=4.6, height=0.8, size=26),
        ).arrange(m.RIGHT, buff=0.5).move_to([0, -2.85, 0])

        self.say(
            "So... [imagine a model that's right ninety-nine percent of the time.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(hero, shift=0.3 * m.UP, scale=0.85), m.FadeIn(sub), run_time=1.2),
        )
        self.say(
            "[Sounds great.] [But it might be completely useless.]",
            hero.animate(rate_func=m.there_and_back, run_time=0.8).scale(1.06),
            m.Succession(m.FadeIn(useless, scale=1.2), m.Wiggle(useless, scale_value=1.08)),
        )
        self.say(
            "Today: [why accuracy can lie], [and how to measure a model properly.]",
            m.FadeIn(today[0], shift=0.3 * m.UP),
            m.FadeIn(today[1], shift=0.3 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the example

    def example(self):
        self.next_section("The example", notes="1,000 people screened; 10 are sick, 990 healthy.")
        sick_ids, missed_ids, flagged_ids = crowd()
        self.people = m.VGroup(*[person(i) for i in range(PEOPLE)])
        self.sick = m.VGroup(*[self.people[i] for i in sick_ids])
        self.healthy = m.VGroup(*[d for i, d in enumerate(self.people) if i not in sick_ids])
        self.fn = m.VGroup(*[self.people[i] for i in missed_ids])
        self.tp = m.VGroup(*[self.people[i] for i in sick_ids if i not in missed_ids])
        self.fp = m.VGroup(*[self.people[i] for i in flagged_ids])
        self.tn = m.VGroup(*[d for i, d in enumerate(self.people) if i not in sick_ids and i not in flagged_ids])
        assert (len(self.tp), len(self.fn), len(self.fp), len(self.tn)) == (TP, FN, FP, TN)

        test = icon("ep06-test", 1.25).move_to([3.3, 1.35, 0])
        test_label = text("a screening test", 24, MUTED).next_to(test, m.DOWN, buff=0.22)
        crowd_label = text("1,000 people", 48, INK, weight=m.BOLD).move_to([3.3, -0.35, 0])
        legend = m.VGroup(
            m.VGroup(m.Dot(radius=0.09, color=PINK), text("10 sick", 28, PINK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2),
            m.VGroup(m.Dot(radius=0.09, color=MUTED).set_fill(MUTED, opacity=0.55), text("990 healthy", 28, MUTED, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2),
        ).arrange(m.RIGHT, buff=0.6).move_to([3.3, -1.3, 0])
        job = pill("the test's job: find the 10", PINK, PINK_FILL, 24).move_to([3.3, -2.45, 0])

        self.say(
            "[Our running example: a screening test for a rare disease.]",
            m.AnimationGroup(
                self.new_header("THE EXAMPLE", "Screening for a rare disease"),
                m.FadeIn(test, shift=0.2 * m.UP),
                m.FadeIn(test_label),
            ),
        )
        self.say(
            "[We test one thousand people.]",
            m.AnimationGroup(m.FadeIn(self.people, lag_ratio=0.002, run_time=1.8), m.FadeIn(crowd_label, shift=0.2 * m.UP)),
        )
        self.say(
            "[Only ten of them are actually sick.] [The test's job is to find those ten.]",
            m.AnimationGroup(
                *[d.animate.set_fill(PINK, opacity=1).scale(1.6) for d in self.sick],
                m.FadeIn(legend, shift=0.2 * m.UP),
            ),
            m.AnimationGroup(
                *[m.Flash(d.get_center(), color=PINK, flash_radius=0.2, line_length=0.1) for d in self.sick],
                m.FadeIn(job, scale=1.15),
            ),
        )
        self.wait(0.3)
        self.fade_all_but(self.people)

    # ------------------------------------------------------------------ the accuracy trap

    def trap(self):
        self.next_section("The accuracy trap", notes="Always saying 'healthy' scores 99% and finds no one.")
        lazy_box = card(5.7, 1.35, VIOLET, CARD, stroke=2).move_to([3.4, 1.55, 0])
        lazy_body = m.VGroup(
            icon("llm", 0.8),
            m.VGroup(text("A lazy model", 28, INK, weight=m.BOLD), text("always answers: healthy", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT),
        ).arrange(m.RIGHT, buff=0.35)
        lazy_body.move_to(lazy_box).align_to(lazy_box, m.LEFT).shift(0.35 * m.RIGHT)
        lazy = m.VGroup(lazy_box, lazy_body)

        right = text("990 right", 24, GOOD, weight=m.BOLD)
        wrong = text("10 wrong", 24, BAD, weight=m.BOLD)
        tally = m.VGroup(right, wrong).arrange(m.RIGHT, buff=0.7).move_to([GRID_CENTER[0], -2.3, 0])
        bad_rings = m.VGroup(*[ring(d, BAD, RING_R + 0.018, 3) for d in self.sick])

        label = text("accuracy", 32, SKY, weight=m.BOLD)
        rule = tex(r"= \frac{\text{correct}}{\text{everyone}}", 42)
        top = m.VGroup(label, rule).arrange(m.RIGHT, buff=0.25).move_to([3.4, 0.05, 0])
        numbers = tex(r"= \frac{990}{1000} = 99\%", 42).next_to(rule, m.DOWN, buff=0.35).align_to(rule, m.LEFT)
        found = pill("✗  finds 0 of the 10 sick", BAD, BAD_FILL, 24).move_to([3.4, -2.3, 0])
        note = text("When one outcome is rare, accuracy hides the mistakes that matter.", 26, BAD, weight=m.SEMIBOLD).move_to([0, -3.3, 0])

        self.say(
            "[First, meet a lazy model. It ignores the data, and always says healthy.]",
            m.AnimationGroup(
                self.new_header("THE ACCURACY TRAP", "A model that always says healthy", BAD, BAD_FILL),
                m.FadeIn(lazy, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[It's right about the nine hundred and ninety healthy people], [and wrong about the ten sick ones.]",
            m.AnimationGroup(self.healthy.animate.set_fill(GOOD, opacity=0.8), m.FadeIn(right, shift=0.2 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(r) for r in bad_rings], lag_ratio=0.1), m.FadeIn(wrong, shift=0.2 * m.UP)),
        )
        self.say(
            "[Accuracy is the share of answers that are correct]: [nine hundred and ninety out of one thousand.]",
            m.AnimationGroup(m.FadeIn(label), m.Write(rule)),
            m.Write(numbers),
        )
        self.say(
            "[That's ninety-nine percent accurate]... [and it found zero sick people.]",
            m.Circumscribe(numbers, color=SKY, buff=0.1),
            m.Succession(m.FadeIn(found, scale=1.2), m.Wiggle(found, scale_value=1.06)),
        )
        self.say(
            "[When one outcome is this rare, accuracy hides the mistakes that matter.]",
            m.FadeIn(note, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.fade_all_but(self.people)

    # ------------------------------------------------------------------ the confusion matrix

    def matrix(self):
        self.next_section("The confusion matrix", notes="TP 8, FN 2, FP 40, TN 950; accuracy 95.8%.")
        tp_rings = m.VGroup(*[ring(d, AMBER, RING_R + 0.018) for d in self.tp])
        fp_rings = m.VGroup(*[ring(d, AMBER) for d in self.fp])
        flag_key = m.VGroup(
            m.Circle(radius=0.1).set_stroke(AMBER, width=3), text("flagged by the test", 22, AMBER, weight=m.SEMIBOLD)
        ).arrange(m.RIGHT, buff=0.2).move_to([GRID_CENTER[0], -2.3, 0])

        self.boxes, self.contents, self.cells = {}, {}, {}
        for key, (row, col, count, words, color, fill) in CELLS.items():
            center = np.array([COL_X[col], ROW_Y[row], 0])
            box = card(CELL_W, CELL_H, color, fill, radius=0.2, stroke=2.5).move_to(center)
            number = text(f"{count}", 48, INK, weight=m.BOLD).move_to(center + 0.14 * m.UP)
            name = text(words, 20, color, weight=m.SEMIBOLD).move_to(center + 0.5 * m.DOWN)
            tag = text(key, 20, color, weight=m.BOLD).move_to(box.get_corner(m.UL) + np.array([0.32, -0.26, 0]))
            content = m.VGroup(number, name, tag).set_z_index(2)
            self.boxes[key], self.contents[key] = box, content
            self.cells[key] = m.VGroup(box, content)
        self.col_labels = m.VGroup(
            text("flagged", 24, AMBER, weight=m.BOLD).move_to([COL_X[0], ROW_Y[0] + CELL_H / 2 + 0.42, 0]),
            text("not flagged", 24, MUTED, weight=m.BOLD).move_to([COL_X[1], ROW_Y[0] + CELL_H / 2 + 0.42, 0]),
        )
        edge = COL_X[0] - CELL_W / 2 - 0.25
        self.row_labels = m.VGroup(
            text("really sick", 24, PINK, weight=m.BOLD).move_to([0, ROW_Y[0], 0]),
            text("really healthy", 24, MUTED, weight=m.BOLD).move_to([0, ROW_Y[1], 0]),
        )
        for lab in self.row_labels:
            lab.align_to([edge, 0, 0], m.RIGHT)
        frame = m.VGroup(*self.boxes.values(), self.col_labels, self.row_labels)

        def fly(dots, key, cols, step, scale, rings=None):
            center = self.boxes[key].get_center()
            target = landing(dots, center, cols, step, scale)
            parts = [m.Transform(dots, target, run_time=1.2)]
            if rings is not None:
                parts.append(m.FadeOut(rings, run_time=0.5))
            return m.AnimationGroup(*parts)

        def land(dots, key):
            return m.AnimationGroup(m.FadeOut(dots, scale=0.6), m.FadeIn(self.contents[key], scale=1.15), run_time=0.7)

        label = text("accuracy", 30, SKY, weight=m.BOLD)
        rule = tex(r"= \frac{\mathrm{TP} + \mathrm{TN}}{\text{everyone}}", 40)
        top = m.VGroup(label, rule).arrange(m.RIGHT, buff=0.25)
        top.move_to([0, 0.95, 0]).align_to([LEFT_X, 0, 0], m.LEFT)
        numbers = tex(r"= \frac{8 + 950}{1000}", 40).next_to(rule, m.DOWN, buff=0.4).align_to(rule, m.LEFT)
        result = tex(r"= 95.8\%", 48, SKY).next_to(numbers, m.DOWN, buff=0.35).align_to(rule, m.LEFT)
        compare = m.VGroup(
            pill("lazy model: 99%  ·  finds 0 of 10", BAD, BAD_FILL, 22),
            pill("this model: 95.8%  ·  finds 8 of 10", GOOD, GOOD_FILL, 22),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        compare.move_to([0, -2.75, 0]).align_to([LEFT_X, 0, 0], m.LEFT)

        self.say(
            "[Now let's try a real screening model.] [It flags eight of the ten sick people], [and misses two.]",
            m.AnimationGroup(
                self.new_header("THE CONFUSION MATRIX", "Sort every answer into four boxes"),
                self.healthy.animate.set_fill(MUTED, opacity=0.55),
            ),
            m.AnimationGroup(m.LaggedStart(*[m.Create(r) for r in tp_rings], lag_ratio=0.15), m.FadeIn(flag_key, shift=0.2 * m.UP)),
            m.AnimationGroup(*[m.Flash(d.get_center(), color=BAD, flash_radius=0.22, line_length=0.12) for d in self.fn]),
        )
        self.say(
            "[But it also flags forty healthy people by mistake.]",
            m.LaggedStart(*[m.Create(r) for r in fp_rings], lag_ratio=0.04, run_time=1.6),
        )
        self.say(
            "[Let's sort all one thousand people into four boxes.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(b, scale=0.9) for b in self.boxes.values()], lag_ratio=0.15),
                m.FadeIn(self.col_labels, shift=0.2 * m.DOWN),
                m.FadeIn(self.row_labels, shift=0.2 * m.RIGHT),
            ),
        )
        self.say("[Sick, and flagged]: [eight true positives.]", fly(self.tp, "TP", 8, 0.17, 1.0, tp_rings), land(self.tp, "TP"))
        self.say("[Sick, but missed]: [two false negatives.]", fly(self.fn, "FN", 2, 0.2, 1.0), land(self.fn, "FN"))
        self.say(
            "[Healthy, but flagged]: [forty false positives, or false alarms.]",
            fly(self.fp, "FP", 10, 0.12, 1.0, fp_rings),
            land(self.fp, "FP"),
        )
        self.say(
            "[And healthy, and cleared]: [nine hundred and fifty true negatives.]",
            m.AnimationGroup(fly(self.tn, "TN", 38, 0.05, 0.45), m.FadeOut(flag_key)),
            land(self.tn, "TN"),
        )
        self.say(
            "[This is a confusion matrix.] [Green boxes are right answers], [red boxes are mistakes.]",
            m.Circumscribe(m.VGroup(*self.boxes.values()), color=SKY, buff=0.15),
            m.AnimationGroup(*[m.Indicate(self.cells[k], color=GOOD, scale_factor=1.05) for k in ("TP", "TN")]),
            m.AnimationGroup(*[m.Indicate(self.cells[k], color=BAD, scale_factor=1.05) for k in ("FP", "FN")]),
        )
        self.say(
            "[Its accuracy]: [eight plus nine hundred and fifty, out of one thousand]. [Almost ninety-six percent.]",
            m.AnimationGroup(m.FadeIn(label), m.Write(rule)),
            m.Write(numbers),
            m.FadeIn(result, shift=0.2 * m.UP),
        )
        self.say(
            "[That's lower than the lazy model's ninety-nine], [yet this model actually finds sick people.]",
            m.FadeIn(compare[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(compare[1], shift=0.2 * m.UP), m.Indicate(self.cells["TP"], color=GOOD, scale_factor=1.06)),
        )
        self.wait(0.3)
        self.matrix_parts = m.VGroup(frame, *self.contents.values())
        self.fade_all_but(self.matrix_parts)

    # ------------------------------------------------------------------ precision

    def ratio(self, name, color, rule_tex, numbers_tex, result_words):
        label = text(name, 32, color, weight=m.BOLD)
        rule = tex(rule_tex, 42)
        top = m.VGroup(label, rule).arrange(m.RIGHT, buff=0.25)
        top.move_to([0, 0.95, 0]).align_to([LEFT_X, 0, 0], m.LEFT)
        numbers = tex(numbers_tex, 42).next_to(rule, m.DOWN, buff=0.4).align_to(rule, m.LEFT)
        result = text(result_words, 56, color, weight=m.BOLD).next_to(numbers, m.DOWN, buff=0.4).align_to(rule, m.LEFT)
        return label, rule, numbers, result

    def precision(self):
        self.next_section("Precision", notes="Of those flagged, how many were right: 8 / (8 + 40) = 1/6 ≈ 17%.")
        column = m.VGroup(self.boxes["TP"], self.boxes["FP"])
        frame = outline(column, AMBER)
        label, rule, numbers, result = self.ratio(
            "precision", AMBER,
            r"= \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}",
            r"= \frac{8}{8 + 40} = \frac{8}{48}",
            "≈ 17%",
        )
        note = pill("only 1 in 6 flagged people is really sick", AMBER, AMBER_FILL, 22)
        note.move_to([0, -2.95, 0]).align_to([LEFT_X - 0.1, 0, 0], m.LEFT)

        self.say(
            "[First question: of everyone the test flagged, how many were really sick?]",
            m.AnimationGroup(
                self.new_header("PRECISION", "Of those flagged, how many were right?", AMBER, AMBER_FILL),
                m.Create(frame),
                *[self.cells[k].animate.set_opacity(0.25) for k in ("FN", "TN")],
                self.col_labels[1].animate.set_opacity(0.4),
            ),
        )
        self.say(
            "[This is called precision]: [true positives, divided by true positives plus false positives.]",
            m.FadeIn(label, shift=0.2 * m.UP),
            m.Write(rule),
        )
        self.say(
            "[Eight, out of eight plus forty]: [about seventeen percent.]",
            m.AnimationGroup(
                m.Write(numbers),
                m.Indicate(self.contents["TP"][0], color=GOOD, scale_factor=1.2),
                m.Indicate(self.contents["FP"][0], color=BAD, scale_factor=1.2),
            ),
            m.FadeIn(result, shift=0.2 * m.UP, scale=1.1),
        )
        self.say("[So only one in six flagged people is actually sick.]", m.FadeIn(note, shift=0.2 * m.UP))
        self.wait(0.3)
        self.fade_all_but(self.matrix_parts)

    # ------------------------------------------------------------------ recall

    def recall(self):
        self.next_section("Recall", notes="Of the real cases, how many were found: 8 / (8 + 2) = 80%.")
        row = m.VGroup(self.boxes["TP"], self.boxes["FN"])
        frame = outline(row, PINK)
        label, rule, numbers, result = self.ratio(
            "recall", PINK,
            r"= \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}",
            r"= \frac{8}{8 + 2} = \frac{8}{10}",
            "= 80%",
        )
        lazy = pill("lazy model: recall = 0 / 10 = 0%", BAD, BAD_FILL, 22)
        lazy.move_to([0, -2.95, 0]).align_to([LEFT_X - 0.1, 0, 0], m.LEFT)

        self.say(
            "[Second question: of everyone who's really sick, how many did the test find?]",
            m.AnimationGroup(
                self.new_header("RECALL", "Of the real cases, how many were found?", PINK, PINK_FILL),
                m.Create(frame),
                self.cells["FN"].animate.set_opacity(1),
                self.cells["FP"].animate.set_opacity(0.25),
                self.col_labels[1].animate.set_opacity(1),
                self.row_labels[1].animate.set_opacity(0.4),
            ),
        )
        self.say(
            "[This is called recall]: [true positives, divided by true positives plus false negatives.]",
            m.FadeIn(label, shift=0.2 * m.UP),
            m.Write(rule),
        )
        self.say(
            "[Eight, out of eight plus two]: [eighty percent.]",
            m.AnimationGroup(
                m.Write(numbers),
                m.Indicate(self.contents["TP"][0], color=GOOD, scale_factor=1.2),
                m.Indicate(self.contents["FN"][0], color=BAD, scale_factor=1.2),
            ),
            m.FadeIn(result, shift=0.2 * m.UP, scale=1.1),
        )
        self.say(
            "[And the lazy model? Its recall is zero.] It never finds anyone.",
            m.Succession(m.FadeIn(lazy, scale=1.15), m.Wiggle(lazy, scale_value=1.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the trade-off

    def trade_off(self):
        self.next_section("The trade-off", notes="Low threshold: 10 caught, 140 false alarms. High: 5 and 5.")
        x0, x1 = -6.2, 1.6
        base, peak, sick_y, axis_y = -0.25, 1.95, -1.0, -1.7
        t = m.ValueTracker(T_MID)

        def sx(s):
            return x0 + (x1 - x0) * s

        def height(s):
            return base + peak * math.exp(-0.5 * ((s - MU) / SIGMA) ** 2)

        def area(s0, s1, color, opacity):
            ss = np.linspace(s0, s1, 70)
            pts = [[sx(s0), base, 0], *[[sx(s), height(s), 0] for s in ss], [sx(s1), base, 0]]
            return m.Polygon(*pts, stroke_width=0).set_fill(color, opacity=opacity)

        curve = m.VMobject().set_points_as_corners([[sx(s), height(s), 0] for s in np.linspace(0, 1, 120)])
        curve.set_stroke(MUTED, width=3)
        lane = m.Line([sx(0), base, 0], [sx(1), base, 0]).set_stroke(FAINT, width=2)
        healthy_label = text("990 healthy", 22, INK, weight=m.SEMIBOLD).move_to([sx(MU), base + 0.62, 0])
        dots = m.VGroup(*[m.Dot([sx(s), sick_y, 0], radius=0.065, color=PINK) for s in SICK_SCORES]).set_z_index(2)
        sick_label = text("10 sick", 22, PINK, weight=m.BOLD).move_to([0, sick_y, 0]).align_to([x0, 0, 0], m.LEFT)
        axis = arrow([x0, axis_y, 0], [x1 + 0.35, axis_y, 0], FAINT, buff=0)
        ticks = m.VGroup(
            text("low", 20, MUTED).move_to([sx(0), axis_y - 0.4, 0]),
            text("risk score", 22, MUTED, weight=m.SEMIBOLD).move_to([sx(0.5), axis_y - 0.4, 0]),
            text("high", 20, MUTED).move_to([sx(1) - 0.15, axis_y - 0.4, 0]),
        )
        scene_parts = m.VGroup(lane, curve, healthy_label, dots, sick_label, axis, ticks)

        clear_area = m.always_redraw(lambda: area(0, t.get_value(), MUTED, 0.2))
        alarm_area = m.always_redraw(lambda: area(t.get_value(), 1, BAD, 0.7))
        rings = m.always_redraw(
            lambda: m.VGroup(
                *[ring(d, GOOD if s > t.get_value() else BAD, 0.1, 3) for d, s in zip(dots, SICK_SCORES)]
            )
        )
        line = m.always_redraw(
            lambda: m.DashedLine([sx(t.get_value()), axis_y, 0], [sx(t.get_value()), 2.1, 0], dash_length=0.12).set_stroke(AMBER, width=4)
        )
        line_label = m.always_redraw(lambda: text("threshold", 22, AMBER, weight=m.BOLD).move_to([sx(t.get_value()), 2.38, 0]))
        flagged_hint = m.always_redraw(
            lambda: text("flagged →", 20, AMBER, weight=m.SEMIBOLD).next_to([sx(t.get_value()), 1.95, 0], m.RIGHT, buff=0.15)
        )

        # readouts on the right
        rx0, rx1 = 2.35, 6.5

        def value(fn, size, color, y):
            return m.always_redraw(lambda: text(fn(), size, color, weight=m.BOLD).move_to([0, y, 0]).align_to([rx1, 0, 0], m.RIGHT))

        def bar(fn, color, y):
            def make():
                w = max(0.02, (rx1 - rx0) * fn())
                return m.Rectangle(width=w, height=0.24, stroke_width=0).set_fill(color, opacity=0.9).move_to([rx0, y, 0], aligned_edge=m.LEFT)

            return m.always_redraw(make)

        def prec():
            tp, fp = counts(t.get_value())
            return tp / (tp + fp)

        def rec():
            return counts(t.get_value())[0] / SICK

        names = m.VGroup(
            text("sick people caught", 22, MUTED).move_to([0, 1.85, 0]),
            text("false alarms", 22, MUTED).move_to([0, 1.3, 0]),
            text("precision", 26, AMBER, weight=m.BOLD).move_to([0, 0.45, 0]),
            text("recall", 26, PINK, weight=m.BOLD).move_to([0, -0.85, 0]),
        )
        for n in names:
            n.align_to([rx0, 0, 0], m.LEFT)
        tracks = m.VGroup(
            *[
                m.Rectangle(width=rx1 - rx0, height=0.24).set_stroke(EDGE, width=2).set_fill(CARD, opacity=1).move_to([rx0, y, 0], aligned_edge=m.LEFT)
                for y in (-0.02, -1.32)
            ]
        )
        caught = value(lambda: f"{counts(t.get_value())[0]} of 10", 26, INK, 1.85)
        alarms = value(lambda: f"{counts(t.get_value())[1]}", 26, BAD, 1.3)
        p_val = value(lambda: f"{100 * prec():.0f}%", 30, AMBER, 0.45)
        r_val = value(lambda: f"{100 * rec():.0f}%", 30, PINK, -0.85)
        p_bar, r_bar = bar(prec, AMBER, -0.02), bar(rec, PINK, -1.32)
        missed_dots = [d for d, s in zip(dots, SICK_SCORES) if s < T_HIGH]

        self.say(
            "[But here's the catch: precision and recall pull against each other.]",
            m.AnimationGroup(
                self.new_header("THE TRADE-OFF", "Moving the threshold"),
                m.FadeIn(m.VGroup(names[2], names[3], tracks), shift=0.2 * m.UP),
                m.FadeIn(m.VGroup(p_val, r_val, p_bar, r_bar), shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[The model gives each person a risk score], [and flags everyone above a threshold.]",
            m.AnimationGroup(
                m.Create(axis), m.FadeIn(ticks), m.Create(lane), m.Create(curve), m.FadeIn(clear_area),
                m.FadeIn(healthy_label), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.1), m.FadeIn(sick_label),
            ),
            m.AnimationGroup(
                m.Create(line), m.FadeIn(line_label), m.FadeIn(flagged_hint), m.FadeIn(alarm_area), m.FadeIn(rings),
                m.FadeIn(m.VGroup(names[0], names[1], caught, alarms), shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[Lower the threshold], [and it catches all ten sick people]: [recall hits one hundred percent.]",
            t.animate(run_time=1.6).set_value(T_LOW),
            m.Indicate(names[0], color=GOOD, scale_factor=1.1),
            m.Indicate(names[3], color=PINK, scale_factor=1.15),
        )
        self.say(
            "[But it also flags one hundred and forty healthy people], [so precision drops to about seven percent.]",
            m.Indicate(names[1], color=BAD, scale_factor=1.15),
            m.Indicate(names[2], color=AMBER, scale_factor=1.15),
        )
        self.say(
            "[Raise the threshold], [and false alarms fall to just five]: [precision climbs to fifty percent.]",
            t.animate(run_time=2.0).set_value(T_HIGH),
            m.Indicate(names[1], color=BAD, scale_factor=1.15),
            m.Indicate(names[2], color=AMBER, scale_factor=1.15),
        )
        self.say(
            "[But now it misses half the sick people.] [Recall falls to fifty percent.]",
            m.AnimationGroup(*[m.Flash(d.get_center(), color=BAD, flash_radius=0.24, line_length=0.12) for d in missed_dots]),
            m.Indicate(names[3], color=PINK, scale_factor=1.15),
        )
        self.wait(0.3)
        for mob in (clear_area, alarm_area, rings, line, line_label, flagged_hint, caught, alarms, p_val, r_val, p_bar, r_bar):
            mob.clear_updaters()
        self.clear()

    # ------------------------------------------------------------------ one number: F1

    def one_number(self):
        self.next_section("One number: F1", notes="Harmonic mean: 2PR/(P+R) = 8/29 ≈ 0.28; plain average ≈ 0.48.")
        lx0, lx1, ly = -5.5, 5.5, -1.9

        def at(v):
            return lx0 + (lx1 - lx0) * v

        scale = m.Line([lx0, ly, 0], [lx1, ly, 0]).set_stroke(FAINT, width=3)
        ends = m.VGroup(
            text("0%", 20, MUTED).move_to([lx0, ly - 0.35, 0]),
            text("100%", 20, MUTED).move_to([lx1, ly - 0.35, 0]),
        )

        def marker(v, color, value_words, name_words, above, hollow=False):
            dot = m.Circle(radius=0.12).set_stroke(color, width=3).set_fill(BG if hollow else color, opacity=1).move_to([at(v), ly, 0])
            dot.set_z_index(2)
            val = text(value_words, 28, color, weight=m.BOLD)
            name = text(name_words, 20, color, weight=m.SEMIBOLD)
            if above:
                stack = m.VGroup(name, val).arrange(m.DOWN, buff=0.08).next_to(dot, m.UP, buff=0.15)
            else:
                stack = m.VGroup(val, name).arrange(m.DOWN, buff=0.08).next_to(dot, m.DOWN, buff=0.15)
            return m.VGroup(dot, stack)

        p_mark = marker(PRECISION, AMBER, "17%", "precision", True)
        r_mark = marker(RECALL, PINK, "80%", "recall", True)
        f_mark = marker(F1, VIOLET, "28%", "F1 score", False)
        a_mark = marker(AVERAGE, MUTED, "48%", "plain average", False, hollow=True)

        name = text("F1 score", 34, VIOLET, weight=m.BOLD)
        rule = tex(r"= \frac{2 \times P \times R}{P + R}", 46)
        top = m.VGroup(name, rule).arrange(m.RIGHT, buff=0.3).move_to([-3.1, 1.3, 0])
        mean = pill("the harmonic mean of precision and recall", VIOLET, VIOLET_FILL, 22).move_to([-3.1, -0.05, 0])
        numbers = tex(r"= \frac{2 \times 0.17 \times 0.80}{0.17 + 0.80} \approx 0.28", 42).move_to([3.3, 1.3, 0])

        self.say(
            "[Sometimes you want one number that balances both.]",
            m.AnimationGroup(
                self.new_header("ONE NUMBER", "The F1 score"),
                m.Create(scale), m.FadeIn(ends),
                m.FadeIn(p_mark, shift=0.2 * m.DOWN), m.FadeIn(r_mark, shift=0.2 * m.DOWN),
            ),
        )
        self.say(
            "[Take their harmonic mean]: [two times precision times recall, divided by their sum.]",
            m.AnimationGroup(m.FadeIn(name, shift=0.2 * m.UP), m.FadeIn(mean, shift=0.2 * m.UP)),
            m.Write(rule),
        )
        self.say(
            "[For our test, it's about twenty-eight percent], [much closer to the weaker number than a plain average.]",
            m.AnimationGroup(m.Write(numbers), m.FadeIn(f_mark, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(a_mark, shift=0.2 * m.UP), m.Indicate(p_mark, color=AMBER, scale_factor=1.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ which mistake costs more?

    def which_mistake(self):
        self.next_section("Which mistake costs more?", notes="Screening favours recall; a spam filter favours precision.")
        costs = m.VGroup(
            pill("a missed case", PINK, PINK_FILL, 26),
            text("or", 28, MUTED, weight=m.SEMIBOLD),
            pill("a false alarm", AMBER, AMBER_FILL, 26),
        ).arrange(m.RIGHT, buff=0.45).move_to([0, 1.55, 0])

        def panel(name, title, lines, verdict, color, fill, x):
            box = card(6.1, 3.45, color, CARD, stroke=2).move_to([x, -1.1, 0])
            words = m.VGroup(text(title, 30, INK, weight=m.BOLD), text(lines, 22, MUTED)).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
            mark = icon(name, 1.0)
            if mark.width > 1.0:
                mark.scale(1.0 / mark.width)
            body = m.VGroup(mark, words).arrange(m.RIGHT, buff=0.4).move_to(box.get_center() + 0.45 * m.UP)
            badge = pill(verdict, color, fill, 24).move_to(box.get_center() + 1.0 * m.DOWN)
            return m.VGroup(box, body[0], words[0]), words[1], badge

        screen, screen_why, screen_verdict = panel(
            recolored_ep06("ep06-test", PINK, PINK_FILL), "Disease screening",
            "A missed case can be dangerous.\nA false alarm means one more test.", "favour recall", PINK, PINK_FILL, -3.3,
        )
        spam, spam_why, spam_verdict = panel(
            recolored_ep06("mail", AMBER, AMBER_FILL), "Spam filter",
            "Losing a real email hurts.\nAn extra spam is a nuisance.", "favour precision", AMBER, AMBER_FILL, 3.3,
        )

        self.say(
            "[So which matters more? Ask which mistake costs more.]",
            m.AnimationGroup(
                self.new_header("IN PRACTICE", "Which mistake costs more?"),
                m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in costs], lag_ratio=0.25),
            ),
        )
        self.say(
            "[In disease screening, missing a sick person is the costly mistake], [so we favour recall], and confirm each flag with a follow-up test.",
            m.AnimationGroup(m.FadeIn(screen, shift=0.2 * m.UP), m.FadeIn(screen_why, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(screen_verdict, scale=1.2), m.Indicate(costs[0], color=PINK, scale_factor=1.1)),
        )
        self.say(
            "[For a spam filter, like in episode one], [losing a real email is worse], [so we favour precision.]",
            m.FadeIn(spam, shift=0.2 * m.UP),
            m.FadeIn(spam_why, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(spam_verdict, scale=1.2), m.Indicate(costs[2], color=AMBER, scale_factor=1.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Confusion matrix, precision, recall.")
        halo = glow(SKY, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Matrix", "Precision", "Recall"), (SKY, AMBER, PINK))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Don't just ask how often a model is right. Ask how it's wrong.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [the confusion matrix], [precision], [and recall.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Don't just ask how often a model is right. Ask how it's wrong.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
