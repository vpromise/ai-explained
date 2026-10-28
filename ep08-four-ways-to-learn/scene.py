"""AI, Explained · Episode 8 · Four Ways to Learn

Running example: a robot learning to walk, and what each way of learning would need to teach it.
Supervised learning (examples with labels → predict labels: the spam filter from Episode 1; a
person's demonstrations), unsupervised learning (no labels → structure: a shop's customers fall
into groups; the robot's raw sensor data), self-supervised learning (the data labels itself: a
hidden word, the next word, the robot's next sensor reading), reinforcement learning (agent,
environment, reward: a fall is −1, a step is +1). Then a side-by-side comparison, and how
chatbots chain three of them (pretraining, supervised fine-tuning, RL from human feedback).

One accent per paradigm, all film long: supervised AMBER, unsupervised TEAL, self-supervised
VIOLET, reinforcement PINK; rewards GOOD (+) / BAD (−); general headers SKY (Level 1).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

SUP = (AMBER, AMBER_FILL)
UNS = (TEAL, TEAL_FILL)
SELF = (VIOLET, VIOLET_FILL)
RL = (PINK, PINK_FILL)
PARADIGMS = (
    ("Supervised", "learns from labels", "ep08-tag", *SUP),
    ("Unsupervised", "finds patterns", "ep08-clusters", *UNS),
    ("Self-supervised", "fills in blanks", "ep08-blank", *SELF),
    ("Reinforcement", "earns rewards", "ep08-star", *RL),
)


# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """kit.recolored(), but written under this episode's prefix: assets/ep08-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    stem = f"ep08-{name.removeprefix('ep08-')}-{color.lstrip('#').lower()}"
    path = ASSETS / f"{stem}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return stem


def robot(height, color=SKY, fill=SKY_FILL):
    return icon("ep08-robot" if color == SKY else variant("ep08-robot", color, fill), height)


def word_chip(word, color, fill, size=28, height=0.72, pad=0.42):
    """A chip as wide as its word."""
    return chip(word, color, fill, width=max(text(word, size, weight=m.SEMIBOLD).width + pad, 0.8), height=height, size=size)


def panel(x, title, color, width=6.3, height=4.5, y=-0.3):
    box = card(width, height, color, CARD, stroke=2).move_to([x, y, 0]).set_z_index(-1)
    head = text(title, 24, color, weight=m.BOLD)
    head.move_to(box.get_corner(m.UL) + np.array([0.35 + head.width / 2, -0.42, 0]))
    return m.VGroup(box, head)


def email(subject, width=3.45):
    box = card(width, 0.6, SKY, SKY_FILL, radius=0.15, stroke=2)
    body = m.VGroup(icon("mail", 0.3), text(subject, 22, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.22 * m.RIGHT)
    return m.VGroup(box, body)


def hnode(pic, title, sub, color, fill, width=3.4, height=1.5):
    """A node with its icon at the left and its words at the right."""
    box = card(width, height, color, fill)
    words = m.VGroup(text(title, 26, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
    body = m.VGroup(pic, words).arrange(m.RIGHT, buff=0.3).move_to(box)
    return m.VGroup(box, body)


def trace(kind, seed, width=1.4, color=MUTED):
    """A short sensor signal: steady, rhythmic (stepping) or a sudden drop (falling)."""
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, 40)
    if kind == "steady":
        ys = 0.05 * np.sin(8 * u + seed) + rng.normal(0, 0.012, u.size)
    elif kind == "rhythm":
        ys = 0.2 * np.sin(2 * np.pi * 2.5 * u + seed)
    else:
        ys = 0.16 - 0.38 / (1 + np.exp(-(u - 0.5 - 0.05 * seed % 2) * 40)) + rng.normal(0, 0.012, u.size)
    points = [[width * (v - 0.5), y, 0] for v, y in zip(u, ys)]
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=3)


def bars(rows, color, x, y, scale=2.4):
    """Guesses with probabilities: word · bar · percent, one per row."""
    group = m.VGroup()
    for k, (word, p) in enumerate(rows):
        yy = y - 0.5 * k
        label = text(word, 22, INK, weight=m.SEMIBOLD)
        label.move_to([x, yy, 0]).align_to([x, 0, 0], m.RIGHT)
        bar = m.Rectangle(width=max(scale * p, 0.04), height=0.28, stroke_width=0).set_fill(color, opacity=0.9)
        bar.move_to([x + 0.18, yy, 0], aligned_edge=m.LEFT)
        pct = text(f"{round(100 * p)}%", 20, MUTED).next_to(bar, m.RIGHT, buff=0.15)
        group.add(m.VGroup(label, bar, pct))
    return group


def bottom(label, color, fill, size=22):
    return pill(label, color, fill, size).move_to([0, -3.25, 0])


# ---------------------------------------------------------------------------- the episode


class Ep08FourWaysToLearn(Episode):
    number = 8

    def lesson(self):
        self.hook()
        self.supervised()
        self.unsupervised()
        self.self_supervised()
        self.reinforcement()
        self.compare()
        self.chatbots()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A robot learning to walk; four ways a machine can learn.")
        halo = glow(SKY, radius=2.8).move_to([0, 1.0, 0])
        bot = robot(2.3).move_to([0, 1.05, 0])
        floor = m.Line([-1.7, bot.get_bottom()[1], 0], [1.7, bot.get_bottom()[1], 0]).set_stroke(FAINT, width=3)
        cards = m.VGroup()
        for name, sub, pic, color, fill in PARADIGMS:
            box = card(3.05, 1.95, color, fill)
            body = m.VGroup(icon(pic, 0.6), text(name, 26, color, weight=m.BOLD), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.14)
            cards.add(m.VGroup(box, body.move_to(box)))
        cards.arrange(m.RIGHT, buff=0.28).move_to([0, -2.25, 0])

        self.say(
            "[Imagine you want to teach a robot] [to walk.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(bot, shift=0.3 * m.UP), m.Create(floor)),
            m.Rotate(bot, -0.14, about_point=bot.get_bottom(), rate_func=m.there_and_back, run_time=0.8),
        )
        self.say(
            "Where would it learn from? There are four main ways: [supervised], [unsupervised], [self-supervised], [and reinforcement learning.]",
            *[m.AnimationGroup(m.FadeIn(c[0], shift=0.3 * m.UP), m.FadeIn(c[1][:2], shift=0.3 * m.UP)) for c in cards],
        )
        self.say(
            "[Each one learns from something different.] Let's see what each would need to teach our robot.",
            m.LaggedStart(*[m.FadeIn(c[1][2], shift=0.15 * m.UP) for c in cards], lag_ratio=0.25),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ supervised

    def supervised(self):
        self.next_section("1 · Supervised", notes="Examples with labels → predict labels. The spam filter from Episode 1; a person's demonstrations.")
        left = panel(-3.45, "Spam filter · Episode 1", AMBER)
        right = panel(3.45, "Robot · shown by a person", AMBER)
        person = icon("user", 0.48).move_to(right[0].get_corner(m.UR) + np.array([-0.55, -0.42, 0]))

        rows = m.VGroup()
        for k, (subject, label) in enumerate((("Free money inside", "spam"), ("Team lunch Friday", "not spam"), ("You are a winner", "spam"))):
            y = 0.82 - 0.72 * k
            mail = email(subject).move_to([-4.55, y, 0])
            tag = chip(label, AMBER, AMBER_FILL, width=1.55, height=0.6, size=22).move_to([-1.3, y, 0])
            rows.add(m.VGroup(mail, arrow(mail.get_right(), tag.get_left(), MUTED, buff=0.12), tag))
        divider = m.DashedLine([-6.2, -1.2, 0], [-0.7, -1.2, 0], dash_length=0.12).set_stroke(FAINT, width=2)
        new_mail = email("Claim your prize").move_to([-4.55, -1.78, 0])
        unknown = chip("?", EDGE, CARD, width=1.55, height=0.6, size=22).move_to([-1.3, -1.78, 0])
        guess = chip("spam", AMBER, AMBER_FILL, width=1.55, height=0.6, size=22).move_to(unknown)
        new_link = arrow(new_mail.get_right(), unknown.get_left(), MUTED, buff=0.12)
        predicted = text("predicted", 20, AMBER, weight=m.SEMIBOLD).next_to(guess, m.DOWN, buff=0.12)

        demos = m.VGroup()
        postures = ((-0.3, "leaning forward", "step forward"), (0.25, "leaning back", "lean in"), (0.0, "standing still", "lift a foot"))
        reach = 1.55 + max(text(p, 22).width for _, p, _ in postures) + 0.12
        for k, (tilt, posture, move) in enumerate(postures):
            y = 0.68 - 1.18 * k
            bot = robot(0.8, *SUP).rotate(tilt).move_to([0.95, y, 0])
            words = text(posture, 22, INK).move_to([0, y, 0]).align_to([1.55, 0, 0], m.LEFT)
            tag = chip(move, AMBER, AMBER_FILL, width=2.05, height=0.6, size=22).move_to([5.3, y, 0])
            demos.add(m.VGroup(bot, words, arrow([reach, y, 0], tag.get_left(), MUTED, buff=0.12), tag))

        catch = bottom("✗  someone must label every example", BAD, BAD_FILL)
        used = bottom("used for: spam filters · face recognition · medical scans", AMBER, AMBER_FILL)

        self.say("[First, supervised learning.]", self.new_header("1 · SUPERVISED", "Learning from labelled examples", *SUP))
        self.say(
            "[Remember our spam filter from episode one?] [Each email came with the right answer, a label]: [spam, or not spam.]",
            m.AnimationGroup(m.FadeIn(left), m.LaggedStart(*[m.FadeIn(r[0], shift=0.2 * m.RIGHT) for r in rows], lag_ratio=0.2)),
            m.LaggedStart(*[m.AnimationGroup(m.GrowArrow(r[1]), m.FadeIn(r[2], shift=0.2 * m.RIGHT)) for r in rows], lag_ratio=0.2),
            m.Indicate(m.VGroup(*[r[2] for r in rows]), color=AMBER, scale_factor=1.08),
        )
        self.say(
            "[The model learns to map each input to its label], [so it can label new emails it has never seen.]",
            m.AnimationGroup(*[pulse(r[1], AMBER) for r in rows], run_time=1.2),
            m.Succession(
                m.AnimationGroup(m.Create(divider), m.FadeIn(new_mail, shift=0.2 * m.RIGHT), m.GrowArrow(new_link), m.FadeIn(unknown)),
                m.AnimationGroup(m.FadeTransform(unknown, guess), m.FadeIn(predicted, shift=0.1 * m.UP)),
            ),
        )
        self.say(
            "[For our robot, the labels would be demonstrations]: [for each posture, a person shows the right move.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(person, scale=0.8)),
            m.LaggedStart(*[m.FadeIn(d, shift=0.2 * m.RIGHT) for d in demos], lag_ratio=0.3),
        )
        self.say(
            "[It works well, but someone has to label every example], [and that's slow and expensive.]",
            m.FadeIn(catch, shift=0.2 * m.UP),
            m.Wiggle(catch, scale_value=1.05),
        )
        self.say(
            "[Supervised learning is behind spam filters, face recognition, and medical scans.]",
            m.FadeTransform(catch, used),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ unsupervised

    def unsupervised(self):
        self.next_section("2 · Unsupervised", notes="No labels → find structure. A shop's customers form groups; the robot's raw sensor data.")
        left = panel(-3.45, "Shop customers · no labels", TEAL)
        right = panel(3.45, "Robot · raw sensor data", TEAL)
        ax = axes([0, 10, 2], [0, 10, 2], 4.6, 2.8).move_to([-3.35, -0.45, 0])
        x_label = text("visits", 20, MUTED).next_to(ax.x_axis.get_end(), m.DOWN, buff=0.3).shift(0.25 * m.LEFT)
        y_label = text("spend", 20, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.15)
        rng = np.random.default_rng(8)
        centres = ((1.9, 2.6), (2.8, 7.6), (8.2, 2.5))
        clusters = m.VGroup()
        for cx, cy in centres:
            dx = np.clip(rng.normal(0, 0.5, 11), -1.0, 1.0)
            dy = np.clip(rng.normal(0, 0.8, 11), -1.5, 1.5)
            clusters.add(m.VGroup(*[m.Dot(ax.c2p(cx + a, cy + b), radius=0.065, color=MUTED) for a, b in zip(dx, dy)]))
        clusters.set_z_index(2)
        rings = m.VGroup(
            *[
                m.DashedVMobject(m.Circle(radius=0.6).set_stroke(TEAL, width=2.5), num_dashes=24).move_to(ax.c2p(cx, cy))
                for cx, cy in centres
            ]
        )
        names = m.VGroup(
            text("occasional", 22, TEAL, weight=m.SEMIBOLD).next_to(rings[0], m.RIGHT, buff=0.15),
            text("big spenders", 22, TEAL, weight=m.SEMIBOLD).next_to(rings[1], m.RIGHT, buff=0.15),
            text("regulars", 22, TEAL, weight=m.SEMIBOLD).next_to(rings[2], m.UP, buff=0.1),
        )

        kinds = ("steady", "fall", "rhythm", "rhythm", "steady", "fall")
        slots = [(x, y) for y in (0.25, -0.95) for x in (1.45, 3.45, 5.45)]
        column = {"steady": 1.45, "rhythm": 3.45, "fall": 5.45}
        traces, targets, seen = m.VGroup(), [], {}
        for k, (kind, (x, y)) in enumerate(zip(kinds, slots)):
            box = card(1.6, 0.95, EDGE, CARD, radius=0.15, stroke=2)
            traces.add(m.VGroup(box, trace(kind, k, width=1.25).move_to(box)).move_to([x, y, 0]))
            row = seen.get(kind, 0)
            seen[kind] = row + 1
            targets.append([column[kind], (0.25, -0.95)[row], 0])
        groups = m.VGroup(
            *[
                m.DashedVMobject(m.RoundedRectangle(corner_radius=0.2, width=1.84, height=2.42).set_stroke(TEAL, width=2.5), num_dashes=34).move_to([x, -0.35, 0])
                for x in (1.45, 3.45, 5.45)
            ]
        )
        group_names = m.VGroup(*[text(f"group {g}", 22, TEAL, weight=m.SEMIBOLD).move_to([x, -1.88, 0]) for g, x in zip("ABC", (1.45, 3.45, 5.45))])
        catch = bottom("✗  finds patterns, but not which moves are good", BAD, BAD_FILL)
        used = bottom("used for: customer groups · unusual transactions · sorting photos", TEAL, TEAL_FILL)

        def sort():
            moves = []
            for t, target in zip(traces, targets):
                moves.append(t.animate.move_to(target))
                moves.append(t[1].animate.set_stroke(TEAL))
            return m.AnimationGroup(m.AnimationGroup(*moves), m.LaggedStart(*[m.Create(g) for g in groups], lag_ratio=0.2), m.FadeIn(group_names))

        self.say("[Second, unsupervised learning], where there are no labels at all.", self.new_header("2 · UNSUPERVISED", "Finding patterns without labels", *UNS))
        self.say(
            "[Say a shop has data on its customers]: [how often they visit, and how much they spend.]",
            m.AnimationGroup(m.FadeIn(left), m.Create(ax), m.FadeIn(x_label), m.FadeIn(y_label)),
            m.LaggedStart(*[m.FadeIn(d, scale=0.3) for g in clusters for d in g], lag_ratio=0.02),
        )
        self.say(
            "Nobody tells the model what to look for. [It finds the patterns by itself]: [three groups of similar customers.]",
            m.LaggedStart(*[g.animate.set_color(TEAL) for g in clusters], lag_ratio=0.3),
            m.LaggedStart(*[m.Create(r) for r in rings], lag_ratio=0.25),
        )
        self.say(
            "It's up to us to name them: [occasional shoppers], [big spenders], [and regulars.]",
            *[m.FadeIn(n, shift=0.15 * m.LEFT) for n in names],
        )
        self.say(
            "[Our robot could sort its raw sensor readings the same way], [finding moments that look alike], [but that alone won't teach it to walk.]",
            m.AnimationGroup(m.FadeIn(right), m.LaggedStart(*[m.FadeIn(t, scale=0.8) for t in traces], lag_ratio=0.12)),
            sort(),
            m.FadeIn(catch, shift=0.2 * m.UP),
        )
        self.say(
            "[Unsupervised learning is used to group customers, spot unusual transactions, and sort photos.]",
            m.FadeTransform(catch, used),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ self-supervised

    def self_supervised(self):
        self.next_section("3 · Self-supervised", notes="The data labels itself: a hidden word, the next word (language-model pretraining), the robot's next sensor reading.")
        words = ("The", "robot", "took", "a", "careful", "step", "forward")
        chips = m.VGroup(*[word_chip(w, SKY, SKY_FILL) for w in words]).arrange(m.RIGHT, buff=0.16).move_to([0, 1.55, 0])
        step, forward = chips[5], chips[6]
        mask = chip("?", VIOLET, VIOLET_FILL, width=step.width, height=0.72, size=28).move_to(step)
        mask_next = chip("?", VIOLET, VIOLET_FILL, width=forward.width, height=0.72, size=28).move_to(forward)
        slot_note = text("the hidden word", 20, MUTED).move_to([-3.6, 0.35, 0])
        hidden = step.copy().move_to([-3.6, -0.3, 0])
        guess_note = text("the model's guesses", 20, MUTED).move_to([2.6, 0.35, 0])
        guesses = bars((("step", 0.78), ("stride", 0.12), ("look", 0.05)), VIOLET, 1.8, -0.3)
        tick = text("✓", 30, GOOD, weight=m.BOLD).next_to(guesses[0], m.RIGHT, buff=0.25)
        free = bottom("✓  the label came free, from the text itself", GOOD, GOOD_FILL)
        next_note = text("next word?", 20, VIOLET, weight=m.SEMIBOLD).next_to(mask_next, m.DOWN, buff=0.15)
        next_guesses = bars((("forward", 0.64), ("again", 0.15), ("slowly", 0.08)), VIOLET, 1.8, -0.3)
        scale_pill = bottom("× billions of sentences: how language models are pretrained", VIOLET, VIOLET_FILL)

        # the robot predicts its own next sensor reading
        bot = robot(1.8, *SELF).move_to([-5.45, -0.45, 0])
        ax = axes([0, 10, 1], [-1, 1, 0.5], 9.0, 3.0).move_to([1.35, -0.45, 0])
        y_label = text("tilt sensor", 20, MUTED).next_to(ax.y_axis.get_end(), m.RIGHT, buff=0.15)
        x_label = text("time", 20, MUTED).next_to(ax.x_axis.get_end(), m.RIGHT, buff=0.12)

        def signal(t):
            return 0.5 * np.sin(1.25 * t) + 0.15 * np.sin(3.1 * t)

        def actual(t):
            return signal(t) - 0.42 * (np.maximum(t - 6, 0) / 3.6) ** 1.3

        past = ax.plot(signal, x_range=[0.2, 6]).set_stroke(SKY, width=4)
        now = m.DashedLine(ax.c2p(6, -1), ax.c2p(6, 1), dash_length=0.1).set_stroke(MUTED, width=2.5)
        now_label = text("now", 24, MUTED).next_to(now, m.UP, buff=0.1)
        predicted = m.DashedVMobject(ax.plot(signal, x_range=[6, 9.6]).set_stroke(VIOLET, width=4), num_dashes=16)
        happened = ax.plot(actual, x_range=[6, 9.6]).set_stroke(SKY, width=4)
        misses = m.VGroup(*[m.Line(ax.c2p(t, signal(t)), ax.c2p(t, actual(t))).set_stroke(BAD, width=3) for t in (7.2, 8.4, 9.6)])
        predicted_label = text("prediction", 20, VIOLET, weight=m.SEMIBOLD).move_to(ax.c2p(8.2, 0.95))
        happened_label = text("what happened", 20, SKY, weight=m.SEMIBOLD).move_to(ax.c2p(7.6, -0.95))
        future = bottom("the next moment is the label", VIOLET, VIOLET_FILL)
        used = bottom("used for: pretraining language, image and speech models", VIOLET, VIOLET_FILL)

        self.say(
            "[Third, self-supervised learning], a clever twist: the data labels itself.",
            self.new_header("3 · SELF-SUPERVISED", "The data labels itself", *SELF),
        )
        self.say(
            "[Take a sentence], [hide one word], [and ask the model to fill in the blank.]",
            m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in chips], lag_ratio=0.12),
            m.AnimationGroup(m.ReplacementTransform(step, hidden), m.FadeIn(mask, scale=0.8), m.FadeIn(slot_note)),
            m.AnimationGroup(m.FadeIn(guess_note), m.LaggedStart(*[m.FadeIn(g, shift=0.15 * m.RIGHT) for g in guesses], lag_ratio=0.2)),
        )
        self.say(
            "[The hidden word is the label], [so the model can check its own guess, and no person had to write a label.]",
            m.Circumscribe(hidden, color=VIOLET, buff=0.1),
            m.AnimationGroup(m.FadeIn(tick, scale=1.4), m.FadeIn(free, shift=0.2 * m.UP)),
        )
        self.say(
            "[Language models are pretrained this way]: [they predict the next word, across billions of sentences.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(guesses, tick, slot_note, free)),
                m.FadeOut(mask),
                hidden.animate.move_to(mask),
                m.FadeOut(forward),
                m.FadeIn(mask_next, scale=0.8),
                m.FadeIn(next_note),
            ),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(g, shift=0.15 * m.RIGHT) for g in next_guesses], lag_ratio=0.2), m.FadeIn(scale_pill, shift=0.2 * m.UP)),
        )
        self.wait(0.2)
        self.play(*[m.FadeOut(x) for x in (chips[:5], hidden, mask_next, next_note, guess_note, next_guesses, scale_pill)], run_time=0.7)
        self.say(
            "[Our robot can play the same game]: [predict its next sensor reading], [then compare it with what really happens.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.UP), m.Create(ax), m.FadeIn(y_label), m.FadeIn(x_label), m.Create(past), m.Create(now), m.FadeIn(now_label)),
            m.AnimationGroup(m.Create(predicted), m.FadeIn(predicted_label)),
            m.AnimationGroup(m.Create(happened), m.FadeIn(happened_label), m.LaggedStart(*[m.Create(x) for x in misses], lag_ratio=0.3)),
        )
        self.say("[So the future becomes the label], and the robot learns how its body behaves.", m.FadeIn(future, shift=0.2 * m.UP))
        self.say(
            "[Self-supervised learning is how language models, and many image and speech models, get their start.]",
            m.FadeTransform(future, used),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ reinforcement

    def reinforcement(self):
        self.next_section("4 · Reinforcement", notes="Agent, environment, reward. A fall is −1, a step is +1; total reward climbs with practice.")
        agent = hnode(robot(1.0, *RL), "Agent", "the robot", *RL).move_to([-3.95, 1.3, 0])
        world = hnode(icon("ep08-world", 0.9), "Environment", "the world", EDGE, CARD).move_to([-3.95, -1.95, 0])
        act = arrow([-3.0, agent.get_bottom()[1], 0], [-3.0, world.get_top()[1], 0], PINK, buff=0.1)
        act_label = text("action", 22, PINK, weight=m.SEMIBOLD).next_to(act, m.RIGHT, buff=0.18)
        back = arrow([-4.9, world.get_top()[1], 0], [-4.9, agent.get_bottom()[1], 0], MUTED, buff=0.1)
        back_label = text("reward", 22, INK, weight=m.SEMIBOLD).next_to(back, m.LEFT, buff=0.18).shift(0.25 * m.UP)
        plus = text("+1", 26, GOOD, weight=m.BOLD)
        minus = text("−1", 26, BAD, weight=m.BOLD)
        values = m.VGroup(plus, minus).arrange(m.RIGHT, buff=0.3).next_to(back_label, m.DOWN, buff=0.15)

        ground = 0.45
        floor = m.Line([0.2, ground, 0], [6.4, ground, 0]).set_stroke(FAINT, width=3)
        bot = robot(1.45, *RL)
        bot.move_to([1.1, ground + bot.height / 2, 0])
        fall_pivot = bot.get_corner(m.DR)
        minus_pop = text("−1", 40, BAD, weight=m.BOLD).move_to([2.35, 1.95, 0])
        bot2 = bot.copy()
        plus_pop = text("+1", 40, GOOD, weight=m.BOLD).move_to([3.2, 1.4, 0])

        chart = axes([0, 10.5, 1], [-4, 8, 2], 5.2, 2.0).move_to([3.4, -1.6, 0])
        tries = text("tries", 20, MUTED).next_to(chart.x_axis.get_end(), m.RIGHT, buff=0.12)
        total = text("total reward", 20, MUTED).next_to(chart.y_axis.get_end(), m.RIGHT, buff=0.15)
        scores = (-3.4, -2.6, -3.0, -1.4, 0.2, 1.0, 2.8, 3.6, 5.4, 7.2)
        dots = m.VGroup(*[m.Dot(chart.c2p(k + 1, s), radius=0.07, color=PINK) for k, s in enumerate(scores)]).set_z_index(2)
        climb = m.VMobject().set_points_as_corners([d.get_center() for d in dots]).set_stroke(PINK, width=3, opacity=0.7)

        catch = bottom("✗  the catch: millions of tries, often in a simulation", BAD, BAD_FILL)
        used = bottom("used for: game-playing AI · robot control · chatbots", PINK, PINK_FILL)

        def hop(mob, x):
            """A little jump to x (absolute, so several hops can be built at once)."""
            return mob.animate(path_arc=-0.9, run_time=0.55).move_to([x, mob.get_y(), 0])

        self.say(
            "[Fourth, reinforcement learning]: [how our robot can teach itself to walk.]",
            self.new_header("4 · REINFORCEMENT", "Learning from rewards", *RL),
            m.AnimationGroup(m.Create(floor), m.FadeIn(bot, shift=0.3 * m.DOWN)),
        )
        self.say(
            "[The robot is an agent], [taking actions in an environment.]",
            m.FadeIn(agent, shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(world, shift=0.2 * m.RIGHT), m.GrowArrow(act), m.FadeIn(act_label)),
        )
        self.say(
            "[After each action, it gets a reward]: [just a number, high for good, low for bad.]",
            m.AnimationGroup(m.GrowArrow(back), m.FadeIn(back_label)),
            m.LaggedStart(m.FadeIn(plus, scale=1.3), m.FadeIn(minus, scale=1.3), lag_ratio=0.4),
        )
        self.say(
            "[It tries a move, and falls]: [minus one.] [It tries again, and takes a step]: [plus one.]",
            m.Rotate(bot, -m.PI / 2, about_point=fall_pivot, run_time=0.9),
            m.AnimationGroup(m.FadeIn(minus_pop, scale=1.4), pulse(back, BAD), m.Indicate(minus, color=BAD, scale_factor=1.3)),
            m.Succession(m.AnimationGroup(m.FadeOut(bot), m.FadeOut(minus_pop), m.FadeIn(bot2), run_time=0.5), hop(bot2, 2.1)),
            m.AnimationGroup(m.FadeIn(plus_pop, scale=1.4), pulse(back, GOOD), m.Indicate(plus, color=GOOD, scale_factor=1.3)),
        )
        laps = m.Succession(*[m.Succession(pulse(act, PINK), pulse(back, GOOD)) for _ in range(2)])
        self.say(
            "No one shows it the right move. [It learns by trial and error], [repeating whatever earned a reward.]",
            laps,
            m.Circumscribe(plus_pop, color=GOOD, buff=0.12),
        )
        self.say(
            "[Over thousands of tries, its total reward climbs], [and wobbling turns into walking.]",
            m.AnimationGroup(
                m.Create(chart), m.FadeIn(tries), m.FadeIn(total),
                m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.15), m.Create(climb, run_time=2.0),
            ),
            m.Succession(m.FadeOut(plus_pop, run_time=0.3), hop(bot2, 3.2), hop(bot2, 4.3), hop(bot2, 5.4)),
        )
        self.say(
            "[The catch: it can take millions of tries, so robots often train in a simulation first.]",
            m.FadeIn(catch, shift=0.2 * m.UP),
        )
        self.say(
            "[Reinforcement learning trains game-playing programs, robots, and, as we'll see, chatbots.]",
            m.FadeTransform(catch, used),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ compare

    def compare(self):
        self.next_section("Side by side", notes="What each paradigm needs, and what it would need to teach the robot.")
        rows = (
            ("labelled examples", "a person's demos"),
            ("raw data", "its sensor logs"),
            ("the data itself", "its next reading"),
            ("rewards", "steps and falls"),
        )
        cards = m.VGroup()
        for (name, _, pic, color, fill), (needs, bot_needs) in zip(PARADIGMS, rows):
            box = card(3.05, 5.0, color, CARD, stroke=2)
            head = m.VGroup(icon(pic, 0.62), text(name, 26, color, weight=m.BOLD)).arrange(m.DOWN, buff=0.16)
            need = m.VGroup(text("learns from", 20, MUTED), text(needs, 24, INK, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.1)
            rule = m.Line([-1.2, 0, 0], [1.2, 0, 0]).set_stroke(EDGE, width=2)
            bot = m.VGroup(text("for the robot", 20, MUTED), text(bot_needs, 22, color, weight=m.SEMIBOLD), robot(0.8, color, fill)).arrange(m.DOWN, buff=0.12)
            body = m.VGroup(head, need, rule, bot).arrange(m.DOWN, buff=0.3).move_to(box)
            cards.add(m.VGroup(box, head, need, rule, bot))
        cards.arrange(m.RIGHT, buff=0.28).move_to([0, -0.5, 0])

        self.say(
            "[So how do the four compare?] It comes down to what each one learns from.",
            m.AnimationGroup(
                self.new_header("SIDE BY SIDE", "What each one learns from"),
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(c[0]), m.FadeIn(c[1], shift=0.2 * m.UP)) for c in cards], lag_ratio=0.2),
            ),
        )
        self.say(
            "[Supervised learning needs examples with labels.] [Unsupervised learning needs only raw data.]",
            *[m.AnimationGroup(m.FadeIn(c[2], shift=0.15 * m.UP), m.Create(c[3]), m.FadeIn(c[4], shift=0.15 * m.UP)) for c in cards[:2]],
        )
        self.say(
            "[Self-supervised learning makes its labels from the data itself.] [And reinforcement learning needs rewards.]",
            *[m.AnimationGroup(m.FadeIn(c[2], shift=0.15 * m.UP), m.Create(c[3]), m.FadeIn(c[4], shift=0.15 * m.UP)) for c in cards[2:]],
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ chatbots

    def chatbots(self):
        self.next_section(
            "How chatbots learn",
            notes="Facts as of October 2026, kept high-level: self-supervised pretraining → supervised fine-tuning → reinforcement learning from human feedback.",
        )
        stages = (
            ("1 · Pretraining", "self-supervised", ("predicts the next word", "in huge amounts of text"), variant("book", VIOLET, VIOLET_FILL), *SELF),
            ("2 · Fine-tuning", "supervised", ("learns from example", "conversations by people"), "ep08-chat", *SUP),
            ("3 · Human feedback", "reinforcement", ("people compare answers;", "it learns what they prefer"), "ep08-compare", *RL),
        )
        nodes = m.VGroup()
        for (title, kind, desc, pic, color, fill), x in zip(stages, (-4.6, 0, 4.6)):
            box = card(3.95, 3.5, color, CARD, stroke=2.5).move_to([x, 0.3, 0])
            head = m.VGroup(icon(pic, 0.8), text(title, 28, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.18)
            tag = pill(kind, color, fill, 20)
            lines = m.VGroup(*[text(d, 22, MUTED) for d in desc]).arrange(m.DOWN, buff=0.1)
            m.VGroup(head, tag, lines).arrange(m.DOWN, buff=0.22).move_to(box)
            nodes.add(m.VGroup(box, head, tag, lines))
        links = m.VGroup(*[arrow(a[0].get_right(), b[0].get_left(), MUTED, buff=0.1) for a, b in zip(nodes, nodes[1:])])
        start = chip("a next-word predictor", EDGE, CARD, width=4.2, height=0.7, size=24).move_to([-4.6, -2.45, 0])
        finish = chip("a helpful assistant", GOOD, GOOD_FILL, width=4.2, height=0.7, size=24).move_to([4.6, -2.45, 0])
        journey = arrow(start.get_right(), finish.get_left(), MUTED, buff=0.2)

        self.say(
            "[So how do today's chatbots learn?] Usually, with three of these, one after another.",
            self.new_header("PUTTING IT TOGETHER", "How a chatbot learns"),
        )
        self.say(
            "[First, pretraining]: [the model predicts the next word across huge amounts of text.]",
            m.AnimationGroup(m.FadeIn(nodes[0][0]), m.FadeIn(nodes[0][1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(nodes[0][2], scale=1.1), m.FadeIn(nodes[0][3], shift=0.1 * m.UP)),
        )
        self.say(
            "[Then, supervised fine-tuning]: [it learns from example conversations written by people.]",
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(nodes[1][0]), m.FadeIn(nodes[1][1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(nodes[1][2], scale=1.1), m.FadeIn(nodes[1][3], shift=0.1 * m.UP)),
        )
        self.say(
            "[Finally, reinforcement learning from human feedback]: [people compare its answers], [and it learns to give the kind they prefer.]",
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(nodes[2][0]), m.FadeIn(nodes[2][1], shift=0.2 * m.UP), m.FadeIn(nodes[2][2], scale=1.1)),
            m.FadeIn(nodes[2][3][0], shift=0.1 * m.UP),
            m.FadeIn(nodes[2][3][1], shift=0.1 * m.UP),
        )
        self.say(
            "[That's how a next-word predictor becomes a helpful assistant.]",
            m.AnimationGroup(
                m.FadeIn(start, shift=0.2 * m.RIGHT),
                m.Succession(pulse(links[0], AMBER), pulse(links[1], PINK), run_time=1.2),
                m.Succession(m.Wait(0.4), m.GrowArrow(journey), m.FadeIn(finish, shift=0.2 * m.RIGHT)),
            ),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Labels, patterns, blanks, rewards.")
        halo = glow(SKY, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 50, c, weight=m.BOLD) for w, c in zip(("Labels", "Patterns", "Blanks", "Rewards"), (AMBER, TEAL, VIOLET, PINK))]).arrange(m.RIGHT, buff=0.85)
        )
        mid = words[0].get_y()
        seps = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, seps).move_to(0.35 * m.UP)
        tagline = text("The best way to learn depends on what you can give the machine.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [labels], [patterns], [blanks], [rewards.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(seps[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.FadeIn(seps[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
            m.AnimationGroup(m.FadeIn(seps[2]), m.FadeIn(words[3], shift=0.3 * m.UP)),
        )
        self.say("[The best way to learn depends on what you can give the machine.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
