"""AI, Explained · Episode 4 · Data: Features, Labels & Splits

Running example: a bank's past loan applications (income, debt, years employed → approved or
not). Features versus labels, the dataset as a table, a 70/15/15 train / validation / test
split and why the test set stays unseen (an exam you never studied the answers to), data
leakage (a "welcome letter sent" column that is only filled in after approval), and bias in
data (past decisions that treated two parts of town differently).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# past applications: income, debt, years employed, approved
ROWS = [
    ("$52,000", "$8,000", "6", True),
    ("$31,000", "$19,000", "1", False),
    ("$78,000", "$5,000", "12", True),
    ("$45,000", "$22,000", "3", False),
    ("$60,000", "$12,000", "8", True),
]
HEADS = ("Income", "Debt", "Years employed", "Approved?")
TRAIN, TRAIN_FILL = TEAL, TEAL_FILL
VALID, VALID_FILL = AMBER, AMBER_FILL
TEST, TEST_FILL = PINK, PINK_FILL


# ---------------------------------------------------------------------------- helpers


def sit(words, baseline):
    """Put a Text's letters on a baseline (descenders, "$" and "," hang below it)."""
    source = words.text.replace(" ", "").replace("\n", "")
    sitting = [g for g, c in zip(words, source) if (c.isalnum() or c in "?✓✗") and c not in "gjpqy,$"]
    if sitting:
        words.shift((baseline - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    else:
        words.set_y(baseline + 0.1)
    return words


def verdict(ok, size=24):
    return text("✓ Yes" if ok else "✗ No", size, GOOD if ok else BAD, weight=m.SEMIBOLD)


def cell(value, size=24):
    if isinstance(value, bool):
        return verdict(value, size)
    if value == "?":
        return text("?", size + 4, MUTED, weight=m.BOLD)
    return code(value, size, INK)


def data_table(heads, rows, widths, center, row_h=0.6, colors=None, size=24):
    """A dataset as a table. Returns (frame, header texts, rule, stripes, cells[r][c], xs, ys)."""
    colors = colors or [SKY] * (len(heads) - 1) + [GOOD]
    total = sum(widths)
    x0 = center[0] - total / 2
    xs = [x0 + sum(widths[:i]) + widths[i] / 2 for i in range(len(widths))]
    n = len(rows) + 1
    top = center[1] + n * row_h / 2
    ys = [top - (k + 0.5) * row_h for k in range(n)]
    frame = card(total + 0.3, n * row_h + 0.3, EDGE, CARD, radius=0.2, stroke=2).move_to(center)
    head = m.VGroup(*[sit(text(h, 22, c, weight=m.SEMIBOLD).set_x(x), ys[0] - 0.09) for h, c, x in zip(heads, colors, xs)])
    rule = m.Line([x0 + 0.1, ys[0] - row_h / 2, 0], [x0 + total - 0.1, ys[0] - row_h / 2, 0]).set_stroke(EDGE, width=2.5)
    stripes = m.VGroup()
    cells = []
    for r, row in enumerate(rows):
        y = ys[r + 1]
        stripe = m.RoundedRectangle(corner_radius=0.1, width=total - 0.12, height=row_h - 0.1, stroke_width=0)
        stripe.set_fill(EDGE, opacity=0.35 if r % 2 == 0 else 0.0).move_to([center[0], y, 0])
        stripes.add(stripe)
        cells.append(m.VGroup(*[sit(cell(v, size).set_x(x), y - 0.11) for v, x in zip(row, xs)]))
    for part in (head, rule, *cells):
        part.set_z_index(1)  # above the frame, also for cells that arrive by transform
    return frame, head, rule, stripes, cells, xs, ys


def mini_table(messy, color):
    """A small 'dataset' glyph: a card with a grid of cells; messy ones have gaps and errors."""
    box = card(2.3, 1.75, color, CARD, radius=0.2, stroke=2.5)
    grid = m.VGroup()
    rng = np.random.default_rng(4)
    for r in range(4):
        for c in range(3):
            w, h = 0.52, 0.22
            sq = m.RoundedRectangle(corner_radius=0.05, width=w, height=h, stroke_width=0)
            if r == 0:
                sq.set_fill(SKY if c < 2 else GOOD, opacity=0.9)
            else:
                sq.set_fill(MUTED, opacity=0.45)
            if messy and r > 0:
                k = rng.integers(0, 4)
                if (r, c) in ((1, 1), (3, 0)):
                    sq = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.05, width=w, height=h).set_stroke(FAINT, width=2), num_dashes=10)
                elif (r, c) in ((2, 2), (3, 2)):
                    sq.set_fill(BAD, opacity=0.9)
                elif k == 0:
                    sq.set_fill(MUTED, opacity=0.2)
            sq.move_to([(c - 1) * 0.62, (1.5 - r) * 0.32, 0])
            grid.add(sq)
    return m.VGroup(box, grid.move_to(box))


def person(ok, height=0.55):
    return icon("ep04-person-good" if ok else "ep04-person-bad", height)


class Ep04DataFeaturesLabels(Episode):
    number = 4

    def lesson(self):
        self.hook()
        self.features()
        self.dataset()
        self.splits()
        self.exam()
        self.leakage()
        self.bias()
        self.big_picture()
        self.recap()

    def regroup(self, group):
        """Swap the loose pieces on stage for one group that draws the same (no visible change)."""
        for mob in list(self.mobjects[1:]):
            if mob is not self.head:
                self.remove(mob)
        self.add(group)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Same model, different data, different results.")
        rows = []
        for y, messy, color, caption, result in (
            (1.05, False, GOOD, "clean data", pill("✓  sensible decisions", GOOD, GOOD_FILL, 22)),
            (-1.45, True, BAD, "messy data", pill("✗  costly mistakes", BAD, BAD_FILL, 22)),
        ):
            data = mini_table(messy, color).move_to([-4.5, y, 0])
            label = text(caption, 22, color, weight=m.SEMIBOLD).next_to(data, m.DOWN, buff=0.15)
            model = node("llm", "Loan model", "same recipe", VIOLET, VIOLET_FILL, width=2.6, height=1.95, label_size=24).move_to([0, y, 0])
            result.move_to([4.6, y, 0])
            feed = arrow(data.get_right(), model.get_left(), MUTED, buff=0.25)
            out = arrow(model.get_right(), result.get_left(), MUTED, buff=0.25)
            rows.append((data, label, model, result, feed, out))
        tagline = text("A model can only be as good as the examples it learns from.", 28, INK, weight=m.SEMIBOLD).move_to([0, -3.35, 0])

        self.say(
            "[Take two loan-approval models, built in exactly the same way.]",
            m.LaggedStart(*[m.FadeIn(r[2], shift=0.3 * m.UP) for r in rows], lag_ratio=0.35),
        )
        self.say(
            "[One makes sensible decisions.] [The other makes costly mistakes.]",
            m.AnimationGroup(m.GrowArrow(rows[0][5]), m.FadeIn(rows[0][3], shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(rows[1][5]), m.FadeIn(rows[1][3], shift=0.2 * m.RIGHT)),
        )
        self.say(
            "[So what's the difference?] [What they learned from: the data.]",
            m.Indicate(m.VGroup(rows[0][2], rows[1][2]), color=VIOLET, scale_factor=1.04),
            m.AnimationGroup(
                *[m.AnimationGroup(m.FadeIn(r[0], shift=0.3 * m.RIGHT), m.FadeIn(r[1]), m.GrowArrow(r[4])) for r in rows],
                m.AnimationGroup(pulse(rows[0][4], GOOD), pulse(rows[1][4], BAD)),
            ),
        )
        self.say(
            "[A model can only be as good as the examples it learns from.]",
            m.Write(tagline),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ features and labels

    def features(self):
        self.next_section("Features and labels", notes="One application: three features and a label.")
        bank = icon("ep04-bank", 1.35).move_to([-5.45, 0.35, 0])
        bank_label = text("the bank", 22, MUTED).next_to(bank, m.DOWN, buff=0.2)

        form = card(5.0, 4.0, SKY, CARD, radius=0.22, stroke=2.5).move_to([-0.8, -0.1, 0])
        stack = m.VGroup(*[card(5.0, 4.0, EDGE, CARD, radius=0.22, stroke=2).move_to(form.get_center() + k * np.array([-0.16, 0.16, 0])) for k in (2, 1)])
        title = text("Loan application", 28, INK, weight=m.BOLD)
        title.move_to(form.get_top() + 0.5 * m.DOWN).align_to(form, m.LEFT).shift(0.4 * m.RIGHT)
        left, right = form.get_left()[0] + 0.4, form.get_right()[0] - 0.4
        ys = [0.75, 0.1, -0.55]
        names, values = m.VGroup(), m.VGroup()
        for name, value, y in zip(HEADS[:3], ROWS[0][:3], ys):
            names.add(sit(text(name, 24, MUTED), y - 0.1).align_to([left, 0, 0], m.LEFT))
            values.add(sit(code(value, 26, INK), y - 0.1).align_to([right, 0, 0], m.RIGHT))
        divider = m.DashedLine([left, -1.0, 0], [right, -1.0, 0], dash_length=0.12).set_stroke(FAINT, width=2.5)
        answer_name = sit(text("Approved?", 24, MUTED), -1.6).align_to([left, 0, 0], m.LEFT)
        answer = sit(verdict(True, 26), -1.6).align_to([right, 0, 0], m.RIGHT)

        feat_brace = m.Brace(m.VGroup(*[m.Line([0, y + 0.22, 0], [0, y - 0.22, 0]) for y in ys]), m.RIGHT, buff=0).set_color(SKY)
        feat_brace.next_to(form, m.RIGHT, buff=0.25).set_y((ys[0] + ys[-1]) / 2)
        feat_words = m.VGroup(text("Features", 32, SKY, weight=m.BOLD), text("the inputs", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        feat_words.next_to(feat_brace, m.RIGHT, buff=0.25)
        label_brace = m.Brace(m.Line([0, -1.2, 0], [0, -1.8, 0]), m.RIGHT, buff=0).set_color(GOOD)
        label_brace.next_to(form, m.RIGHT, buff=0.25).set_y(-1.5)
        label_words = m.VGroup(text("Label", 32, GOOD, weight=m.BOLD), text("the answer to predict", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        label_words.next_to(label_brace, m.RIGHT, buff=0.25)

        self.say(
            "[Our running example: a bank deciding who gets a loan.]",
            m.AnimationGroup(
                self.new_header("FEATURES & LABELS", "Inputs and answers"),
                m.FadeIn(bank, shift=0.2 * m.UP),
                m.FadeIn(bank_label),
            ),
        )
        self.say(
            "[It keeps records of past applications.] [Here's one.]",
            m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.DOWN) for s in stack], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(form, shift=0.2 * m.DOWN), m.FadeIn(title)),
        )
        self.say(
            "It describes the applicant: [their income], [their debt], [and how many years they've been employed.]",
            *[m.AnimationGroup(m.FadeIn(n, shift=0.15 * m.RIGHT), m.FadeIn(v, shift=0.15 * m.LEFT)) for n, v in zip(names, values)],
        )
        self.say(
            "[These inputs are called features.]",
            m.AnimationGroup(m.GrowFromCenter(feat_brace), m.FadeIn(feat_words, shift=0.2 * m.LEFT), *[m.Indicate(v, color=SKY, scale_factor=1.08) for v in values]),
        )
        self.say(
            "[The record also holds the answer]: [was this loan approved?] [Yes.]",
            m.Create(divider),
            m.FadeIn(answer_name, shift=0.15 * m.RIGHT),
            m.FadeIn(answer, scale=1.3),
        )
        self.say(
            "[That answer is called the label]: what we want the model to predict for new applicants.",
            m.AnimationGroup(m.GrowFromCenter(label_brace), m.FadeIn(label_words, shift=0.2 * m.LEFT)),
        )
        # the values move on into the table: keep look-alike copies on stage, fade the rest
        self.card_values = [v.copy() for v in (*values, answer)]
        self.add(*self.card_values)
        self.wait(0.3)
        self.clear(*self.card_values)

    # ------------------------------------------------------------------ the dataset

    def dataset(self):
        self.next_section("The dataset", notes="Rows are examples; columns are features plus the label.")
        frame, head, rule, stripes, cells, xs, ys = data_table(HEADS, ROWS, [2.25, 2.1, 2.75, 2.3], [-1.15, 0.45, 0], row_h=0.6)
        more = m.VGroup(*[text("⋮", 26, MUTED).move_to([x, frame.get_bottom()[1] - 0.3, 0]) for x in xs])
        count = text("thousands of rows", 22, MUTED).next_to(more, m.DOWN, buff=0.12)
        self.table = m.VGroup(frame, stripes, rule, head, *cells)

        row_box = m.RoundedRectangle(corner_radius=0.12, width=frame.width + 0.2, height=0.62).set_stroke(INK, width=3).set_fill(opacity=0)
        row_box.move_to([frame.get_x(), ys[1], 0])
        row_note = m.VGroup(text("one row", 26, INK, weight=m.BOLD), text("= one example", 22, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        row_note.next_to(row_box, m.RIGHT, buff=0.3)
        top, bottom = frame.get_top()[1] - 0.05, frame.get_bottom()[1] + 0.05
        feat_box = m.RoundedRectangle(corner_radius=0.15, width=xs[2] + 2.75 / 2 - (xs[0] - 2.25 / 2) - 0.1, height=top - bottom).set_stroke(SKY, width=3).set_fill(opacity=0)
        feat_box.move_to([(xs[0] - 2.25 / 2 + xs[2] + 2.75 / 2) / 2, (top + bottom) / 2, 0])
        label_box = m.RoundedRectangle(corner_radius=0.15, width=2.2, height=top - bottom).set_stroke(GOOD, width=3).set_fill(opacity=0)
        label_box.move_to([xs[3], (top + bottom) / 2, 0])
        feat_note = text("columns = features", 24, SKY, weight=m.BOLD).move_to([feat_box.get_x(), -2.95, 0])
        label_note = text("last column = label", 24, GOOD, weight=m.BOLD).move_to([label_box.get_x(), -2.95, 0])

        wrong = sit(verdict(False), ys[3] - 0.11).set_x(xs[3])
        wrong_note = m.VGroup(text("wrong label", 26, BAD, weight=m.BOLD), text("= wrong lesson", 22, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        wrong_note.next_to(frame, m.RIGHT, buff=0.3).set_y(ys[3])

        self.say(
            "[Collect thousands of records like this], [and you get a dataset]: one big table.",
            m.AnimationGroup(
                self.new_header("THE DATASET", "A table of examples"),
                *[m.ReplacementTransform(v, c) for v, c in zip(self.card_values, cells[0])],
                m.FadeIn(frame), m.FadeIn(stripes), m.FadeIn(head), m.Create(rule),
            ),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in cells[1:]], lag_ratio=0.2),
                m.FadeIn(more), m.FadeIn(count),
            ),
        )
        self.say(
            "[Each row is one example]: one past application.",
            m.AnimationGroup(m.Create(row_box), m.FadeIn(row_note, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[Each column is a feature], [and the last column is the label.]",
            m.AnimationGroup(m.FadeOut(row_box), m.FadeOut(row_note), m.Create(feat_box), m.FadeIn(feat_note, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Create(label_box), m.FadeIn(label_note, shift=0.2 * m.UP)),
        )
        self.say(
            "[And quality matters.] [A wrong label, like a strong applicant marked as denied], [teaches the model the wrong lesson.]",
            m.AnimationGroup(m.FadeOut(feat_box), m.FadeOut(label_box), m.FadeOut(feat_note), m.FadeOut(label_note)),
            m.AnimationGroup(m.Indicate(cells[2][:3], color=SKY, scale_factor=1.08), m.FadeOut(cells[2][3], shift=0.2 * m.UP), m.FadeIn(wrong, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Circumscribe(wrong, color=BAD, buff=0.1), m.FadeIn(wrong_note, shift=0.2 * m.LEFT)),
        )
        self.wait(0.3)
        self.play(m.FadeOut(wrong_note), m.FadeOut(more), m.FadeOut(count), m.FadeOut(wrong), m.FadeIn(cells[2][3]), run_time=0.7)

    # ------------------------------------------------------------------ splits

    def splits(self):
        self.next_section("Splits", notes="Shuffle, then 70 / 15 / 15: training, validation, test.")
        n, w, gap, h = 20, 0.5, 0.07, 1.35
        approved = [True, False, True, False, True, True, True, False, True, True, False, True, True, False, True, True, False, True, False, True]
        slots = [(-(n - 1) / 2 + i) * (w + gap) for i in range(n)]
        y = 0.5
        slivers = m.VGroup()
        for i in range(n):
            bar = m.RoundedRectangle(corner_radius=0.08, width=w, height=h).set_stroke(EDGE, width=2).set_fill(EDGE, opacity=0.6)
            dot = m.Dot(radius=0.075, color=GOOD if approved[i] else BAD).move_to([0, h / 2 - 0.25, 0])
            slivers.add(m.VGroup(bar, dot).move_to([slots[i], y, 0]))
        order = [7, 15, 2, 11, 18, 4, 0, 13, 9, 19, 5, 16, 1, 10, 6, 17, 12, 3, 14, 8]  # slivers[i] moves to slot order[i]
        groups = {"train": [], "valid": [], "test": []}
        for i, s in enumerate(slivers):
            k = order[i]
            groups["train" if k < 14 else "valid" if k < 17 else "test"].append((s, k))
        offsets = {"train": -0.35, "valid": 0.0, "test": 0.35}
        styles = {"train": (TRAIN, TRAIN_FILL), "valid": (VALID, VALID_FILL), "test": (TEST, TEST_FILL)}

        def final_x(k):
            return slots[k] + offsets["train" if k < 14 else "valid" if k < 17 else "test"]

        centres = {
            "train": (final_x(0) + final_x(13)) / 2,
            "valid": (final_x(14) + final_x(16)) / 2,
            "test": (final_x(17) + final_x(19)) / 2,
        }
        captions = {}
        for key, name, share, rows in (("train", "Training", "70%", "7,000 rows"), ("valid", "Validation", "15%", "1,500 rows"), ("test", "Test", "15%", "1,500 rows")):
            color = styles[key][0]
            label = sit(text(name, 26, color, weight=m.BOLD).set_x(centres[key]), -0.72)
            pct = sit(text(share, 40, INK, weight=m.BOLD).set_x(centres[key]), -1.58)
            num = sit(text(rows, 22, MUTED).set_x(centres[key]), -2.18)
            captions[key] = (label, pct, num)
        total = text("10,000 records", 26, INK, weight=m.SEMIBOLD).move_to([centres["train"], 1.72, 0])
        lock = icon("ep04-lock", 0.62).move_to([centres["test"], 1.72, 0])
        locked = text("locked away", 22, TEST, weight=m.SEMIBOLD).next_to(lock, m.LEFT, buff=0.2)
        common = text("common choices: 70 / 15 / 15 or 80 / 10 / 10", 22, MUTED).move_to([0, -2.95, 0])

        def shuffle():
            return m.AnimationGroup(*[s.animate.move_to([slots[order[i]], y, 0]) for i, s in enumerate(slivers)], run_time=1.4)

        def cut():
            anims = []
            for key, members in groups.items():
                color, fill = styles[key]
                for s, k in members:
                    target = s.copy().move_to([slots[k] + offsets[key], y, 0])
                    target[0].set_stroke(color, width=2.5).set_fill(fill, opacity=1)
                    anims.append(m.Transform(s, target))
            return m.AnimationGroup(*anims, run_time=1.2)

        self.regroup(self.table)
        self.say(
            "[Before training, there's one crucial step: splitting the data.]",
            m.AnimationGroup(self.new_header("SPLITS", "Train, validation, test"), m.FadeTransform(self.table, slivers)),
        )
        self.say("[First we shuffle the rows], [then cut them into three groups.]", shuffle(), cut())
        self.say(
            "[Typically, about seventy percent becomes the training set]: the examples the model learns from.",
            m.AnimationGroup(*[m.FadeIn(c, shift=0.2 * m.UP) for c in captions["train"][:2]], m.Indicate(m.VGroup(*[s for s, _ in groups["train"]]), color=TRAIN, scale_factor=1.03)),
        )
        self.say(
            "[About fifteen percent is the validation set], for checking progress and tuning settings along the way.",
            m.AnimationGroup(*[m.FadeIn(c, shift=0.2 * m.UP) for c in captions["valid"][:2]]),
        )
        self.say(
            "[And the last fifteen percent is the test set], [locked away until the very end.]",
            m.AnimationGroup(*[m.FadeIn(c, shift=0.2 * m.UP) for c in captions["test"][:2]]),
            m.AnimationGroup(m.FadeIn(lock, scale=1.4), m.FadeIn(locked, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[With ten thousand records], [that's seven thousand to train on], [and fifteen hundred each to validate and test.]",
            m.FadeIn(total, shift=0.2 * m.DOWN),
            m.FadeIn(captions["train"][2], shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(captions["valid"][2], shift=0.2 * m.UP), m.FadeIn(captions["test"][2], shift=0.2 * m.UP), m.FadeIn(common)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the exam

    def exam(self):
        self.next_section("The exam analogy", notes="Homework, practice test, final exam.")
        roles = (
            ("ep04-book", "Homework", "the training set", ("study it, with the", "answers included"), TRAIN, TRAIN_FILL),
            ("ep04-quiz", "Practice test", "the validation set", ("check how it's going,", "then adjust"), VALID, VALID_FILL),
            ("ep04-lock", "Final exam", "the test set", ("questions never", "seen before"), TEST, TEST_FILL),
        )
        cards = m.VGroup()
        for x, (name, title, sub, desc, color, fill) in zip((-4.45, 0, 4.45), roles):
            box = card(4.0, 3.75, color, CARD, stroke=2.5).move_to([x, -0.05, 0])
            top = box.get_top()[1]
            body = m.VGroup(
                icon(name, 0.95).move_to([x, top - 0.82, 0]),
                sit(text(title, 30, color, weight=m.BOLD).set_x(x), top - 1.82),
                sit(text(sub, 24, INK, weight=m.SEMIBOLD).set_x(x), top - 2.3),
                *[sit(text(line, 22, MUTED).set_x(x), top - 2.85 - 0.37 * k) for k, line in enumerate(desc)],
            )
            cards.add(m.VGroup(box, body))
        start, end = cards[2][0].get_top() + 0.15 * m.UP + 0.6 * m.LEFT, cards[0][0].get_top() + 0.15 * m.UP + 0.6 * m.RIGHT
        leak = m.CurvedArrow(start, end, angle=0.55)
        leak.set_stroke(BAD, width=4).set_color(BAD)
        nothing = pill("✗  seen the answers? then a perfect score proves nothing", BAD, BAD_FILL, 22).move_to([0, -2.85, 0])
        honest = pill("✓  tune on validation, test once: an honest score on new data", GOOD, GOOD_FILL, 22).move_to([0, -2.85, 0])

        self.say(
            "[Why lock the test set away? Think of a student getting ready for an exam.]",
            self.new_header("THE IDEA", "An exam you never studied the answers to"),
        )
        self.say("[The training set is the homework]: practice questions, with the answers included.", m.FadeIn(cards[0], shift=0.3 * m.UP))
        self.say("[The validation set is a practice test], to see how the studying is going.", m.FadeIn(cards[1], shift=0.3 * m.UP))
        self.say("[And the test set is the final exam]: questions the student has never seen.", m.FadeIn(cards[2], shift=0.3 * m.UP))
        self.say(
            "[If they had seen the exam answers beforehand], [a perfect score would prove nothing.]",
            m.Create(leak),
            m.FadeIn(nothing, shift=0.2 * m.UP),
        )
        self.say(
            "[So we tune with the validation set], [and use the test set once, at the end], [for an honest score on new data.]",
            m.AnimationGroup(m.FadeOut(leak), m.Indicate(cards[1], color=VALID, scale_factor=1.04)),
            m.Indicate(cards[2], color=TEST, scale_factor=1.04),
            m.FadeTransform(nothing, honest),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ data leakage

    def leakage(self):
        self.next_section("Data leakage", notes="A column filled in after the decision gives the answer away.")
        heads = ("Income", "Debt", "Years employed", "Welcome letter", "Approved?")
        rows = [(*r[:3], "Yes" if r[3] else "No", r[3]) for r in ROWS[:4]]
        widths = [2.0, 2.0, 2.55, 2.6, 2.15]
        colors = [SKY, SKY, SKY, BAD, GOOD]
        frame, head, rule, stripes, cells, xs, ys = data_table(heads, rows, widths, [0, 0.95, 0], row_h=0.56, colors=colors)
        letter = m.VGroup(head[3], *[c[3] for c in cells])
        others = m.VGroup(*[h for k, h in enumerate(head) if k != 3], *[x for c in cells for k, x in enumerate(c) if k != 3])
        top, bottom = frame.get_top()[1] - 0.05, frame.get_bottom()[1] + 0.05
        letter_box = m.RoundedRectangle(corner_radius=0.15, width=widths[3] - 0.1, height=top - bottom).set_stroke(BAD, width=3).set_fill(opacity=0)
        letter_box.move_to([xs[3], (top + bottom) / 2, 0])
        after = text("filled in only after a loan is approved", 22, BAD, weight=m.SEMIBOLD).move_to([xs[3] - 0.6, -0.95, 0])

        new_y = -1.55
        new_label = text("a new applicant", 22, MUTED).move_to([0, -0.98, 0]).align_to(frame, m.LEFT).shift(0.15 * m.RIGHT)
        new_box = m.DashedVMobject(
            m.RoundedRectangle(corner_radius=0.15, width=frame.width, height=0.58).set_stroke(SKY, width=2.5), num_dashes=70
        ).move_to([0, new_y, 0])
        new_cells = m.VGroup(*[sit(cell(v).set_x(x), new_y - 0.11) for v, x in zip(("$49,000", "$10,000", "5", "?", "?"), xs)])
        new_cells[3].set_color(BAD)

        learner = icon("llm", 0.62)
        learns = text("learns:", 24, MUTED)
        shortcut = pill("letter sent  →  approved", BAD, BAD_FILL, 22)
        score = pill("test score: 99%", GOOD, GOOD_FILL, 22)
        lesson_row = m.VGroup(learner, learns, shortcut, score).arrange(m.RIGHT, buff=0.3)
        score.shift(0.4 * m.RIGHT)
        lesson_row.move_to([0, -2.45, 0])
        fails = pill("✗  no letter yet: the model fails", BAD, BAD_FILL, 22).move_to(score).align_to(score, m.LEFT)
        definition = text("Data leakage: a feature that secretly contains the answer", 28, INK, weight=m.SEMIBOLD, t2c={"Data leakage:": BAD})
        definition.move_to([0, -2.45, 0])
        fix = pill("✓  the fix: only use what you'd know at decision time", GOOD, GOOD_FILL, 22).move_to([0, -3.35, 0])

        self.say(
            "[Even with a clean split, there's a sneaky trap: data leakage.]",
            self.new_header("DATA LEAKAGE", "When a feature gives the answer away", BAD, BAD_FILL),
        )
        self.say(
            "[Imagine our table had one more column]: [welcome letter sent.]",
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(stripes), m.Create(rule), m.FadeIn(others)),
            m.AnimationGroup(m.FadeIn(letter, shift=0.2 * m.DOWN), m.Create(letter_box)),
        )
        self.say(
            "[But the bank only sends that letter after approving a loan.]",
            m.AnimationGroup(
                m.FadeIn(after, shift=0.2 * m.UP),
                m.LaggedStart(*[m.Indicate(m.VGroup(c[3], c[4]), color=BAD, scale_factor=1.08) for c in cells], lag_ratio=0.25),
            ),
        )
        self.say(
            "[So the model learns a shortcut]: [letter sent means approved], [and it scores ninety-nine percent on the test.]",
            m.AnimationGroup(m.FadeIn(learner, scale=0.7), m.FadeIn(learns)),
            m.FadeIn(shortcut, shift=0.2 * m.RIGHT),
            m.FadeIn(score, scale=1.3),
        )
        self.say(
            "[But a new applicant hasn't been sent any letter yet.] [The shortcut is gone, and the model fails.]",
            m.AnimationGroup(m.FadeOut(after), m.FadeIn(new_label), m.Create(new_box), m.FadeIn(new_cells, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Wiggle(new_cells[3], scale_value=1.4), m.FadeTransform(score, fails)),
        )
        self.say(
            "[That's data leakage]: [a feature that secretly contains the answer.] [The fix? Only use what you'd know at decision time.]",
            m.Circumscribe(letter_box, color=BAD, buff=0.05),
            m.AnimationGroup(*[m.FadeOut(x) for x in (learner, learns, shortcut, fails)], m.Write(definition)),
            m.FadeIn(fix, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ bias

    def bias(self):
        self.next_section("Bias in data", notes="Past decisions that treated two parts of town differently.")
        panels = m.VGroup()
        for x, name, approved in ((-3.3, "Neighbourhood A", 8), (3.3, "Neighbourhood B", 3)):
            box = card(6.1, 3.35, EDGE, CARD, stroke=2).move_to([x, 0.75, 0])
            title = text(name, 28, INK, weight=m.BOLD).move_to(box.get_top() + 0.42 * m.DOWN)
            sub = text("similar incomes and debts", 22, MUTED).next_to(title, m.DOWN, buff=0.12)
            people = m.VGroup(*[person(k < approved) for k in range(10)]).arrange_in_grid(rows=2, cols=5, buff=(0.32, 0.2))
            people.move_to(box.get_center() + 0.12 * m.DOWN)
            rate = text(f"{approved} of 10 approved", 24, GOOD if approved > 5 else BAD, weight=m.BOLD).next_to(people, m.DOWN, buff=0.22)
            panels.add(m.VGroup(box, title, sub, people, rate))

        learner = icon("llm", 0.62)
        learned = m.VGroup(learner, text("learns:", 24, MUTED), pill("Neighbourhood B  →  deny more often", BAD, BAD_FILL, 22)).arrange(m.RIGHT, buff=0.3)
        learned.move_to([0, -1.6, 0])
        blind = text("It can't tell that the pattern is unfair. It just copies the data.", 24, MUTED).move_to([0, -2.45, 0])

        gone = pill("neighbourhood column: deleted", MUTED, CARD, 22)
        proxy = pill("branch applied at", SKY, SKY_FILL, 22)
        link = arrow([0, 0, 0], [1.2, 0, 0], MUTED)
        proxy_row = m.VGroup(gone, link, proxy).arrange(m.RIGHT, buff=0.3).move_to([0, -1.7, 0])
        link.put_start_and_end_on(gone.get_right() + 0.1 * m.RIGHT, proxy.get_left() + 0.1 * m.LEFT)
        same = text("other features can carry the same information", 24, MUTED).move_to([0, -2.5, 0])
        proxy_row.add(same)

        checks = m.VGroup(
            pill("Who is in the data?", SKY, SKY_FILL, 22),
            pill("Who is missing?", SKY, SKY_FILL, 22),
            pill("Which past decisions does it repeat?", SKY, SKY_FILL, 22),
        ).arrange(m.RIGHT, buff=0.3).move_to([0, -2.0, 0])

        self.say(
            "[Data can hide a quieter problem, too: bias.]",
            self.new_header("BIAS IN DATA", "Models copy the patterns they're shown", BAD, BAD_FILL),
        )
        self.say(
            "[Suppose that in the past, the bank approved most applicants from one part of town], [but far fewer from another], [even with similar finances.]",
            m.AnimationGroup(m.FadeIn(panels[0][:2]), m.LaggedStart(*[m.FadeIn(p, scale=0.6) for p in panels[0][3]], lag_ratio=0.08), m.FadeIn(panels[0][4])),
            m.AnimationGroup(m.FadeIn(panels[1][:2]), m.LaggedStart(*[m.FadeIn(p, scale=0.6) for p in panels[1][3]], lag_ratio=0.08), m.FadeIn(panels[1][4])),
            m.AnimationGroup(m.FadeIn(panels[0][2], shift=0.1 * m.DOWN), m.FadeIn(panels[1][2], shift=0.1 * m.DOWN)),
        )
        self.say("[A model trained on those decisions learns the same pattern.]", m.FadeIn(learned, shift=0.2 * m.UP))
        self.say(
            "[It doesn't know the pattern is unfair.] [It just copies what the data shows.]",
            m.Wiggle(learned[2], scale_value=1.05),
            m.FadeIn(blind, shift=0.2 * m.UP),
        )
        self.say(
            "[And deleting that column may not help]: [other features, like which branch someone applied at, can stand in for it.]",
            m.AnimationGroup(m.FadeOut(learned), m.FadeOut(blind), m.FadeIn(gone, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(proxy_row[1]), m.FadeIn(proxy, shift=0.2 * m.LEFT), m.FadeIn(same)),
        )
        self.say(
            "[So look closely at your data]: [who is in it], [who is missing], [and which past decisions it repeats.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in proxy_row]),
            m.FadeIn(checks[0], shift=0.2 * m.UP),
            m.FadeIn(checks[1], shift=0.2 * m.UP),
            m.FadeIn(checks[2], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section("The big picture", notes="What good data looks like.")
        statement = text("A model only learns what its data shows it.", 30, INK, weight=m.SEMIBOLD).move_to([0, 2.05, 0])
        tiles = m.VGroup(
            tile("ep04-table", "Features & labels", "inputs you'd really have,\nanswers you can trust", SKY),
            tile("ep04-lock", "An honest split", "a test set the model\nnever sees while learning", TEST),
            tile("alert", "No leaks", "nothing that secretly\ngives the answer away", BAD),
            tile("ep04-person-good", "Fair data", "examples that reflect the\npeople the model affects", GOOD),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.45, 0.4)).move_to([0, -0.9, 0])

        self.say(
            "[So, why does data quality decide model quality?] [Because a model only learns what its data shows it.]",
            self.new_header("THE BIG PICTURE", "Good data in, good model out"),
            m.FadeIn(statement, shift=0.2 * m.UP),
        )
        self.say(
            "[Good data means clear features and correct labels], [and an honest split, with a test set kept unseen.]",
            m.FadeIn(tiles[0], shift=0.3 * m.UP),
            m.FadeIn(tiles[1], shift=0.3 * m.UP),
        )
        self.say(
            "[No leaked answers], [and examples that fairly represent the people the model will affect.]",
            m.FadeIn(tiles[2], shift=0.3 * m.UP),
            m.FadeIn(tiles[3], shift=0.3 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Features, labels, splits.")
        halo = glow(SKY, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Features", "Labels", "Splits"), (SKY, GOOD, AMBER))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A model is only as good as the data it learns from.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [features], [labels], [splits.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Remember: a model is only as good as the data it learns from.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
