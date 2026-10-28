"""AI, Explained · Episode 1 · What is AI?

Running example: a spam filter — hand-written rules break; a model learns the rules from
labelled emails. Then: how AI, machine learning, deep learning and generative AI nest; a short
history; narrow versus general AI.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403


class Ep01WhatIsAI(Episode):
    number = 1

    def lesson(self):
        self.hook()
        self.old_way()
        self.idea()
        self.learning()
        self.family()
        self.history()
        self.narrow()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Machines doing things that usually need human intelligence.")
        halo = glow(SKY, radius=3.6).move_to(1.3 * m.UP)
        title = text("Artificial Intelligence", 72, INK, weight=m.BOLD, gradient=(SKY, VIOLET)).move_to([0, 1.55, 0])
        meaning = text("Machines doing things that usually need human intelligence.", 30, MUTED).move_to([0, 0.45, 0])
        examples = m.VGroup()
        for name, label, color, fill in (
            ("user", "recognise faces", SKY, SKY_FILL),
            ("doc", "translate languages", AMBER, AMBER_FILL),
            ("answer", "answer questions", GOOD, GOOD_FILL),
            ("code", "write code", VIOLET, VIOLET_FILL),
        ):
            box = card(2.9, 2.0, color, fill)
            body = m.VGroup(icon(name, 0.75), text(label, 22, INK, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.25)
            examples.add(m.VGroup(box, body.move_to(box)))
        examples.arrange(m.RIGHT, buff=0.35).move_to([0, -1.9, 0])

        self.say(
            "So, [what is artificial intelligence]?",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(title, shift=0.3 * m.UP), run_time=1.2),
        )
        self.say(
            "[Simply put, it's machines doing things that usually need human intelligence]:",
            m.Write(meaning),
        )
        self.say(
            "[recognising faces], [translating languages], [answering questions], [and even writing code.]",
            *[m.FadeIn(e, shift=0.3 * m.UP) for e in examples],
        )
        self.say("But how do you build a machine like that? Let's find out with a simple example.")
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the old way

    def old_way(self):
        self.next_section("The old way", notes="Hand-written spam rules break and pile up.")

        def email(subject, color=SKY, fill=SKY_FILL):
            box = card(5.9, 1.15, color, fill, radius=0.2)
            body = m.VGroup(icon("mail", 0.5), text(subject, 24, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.3)
            return m.VGroup(box, body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT))

        first = email("WIN FREE MONEY NOW!!!").move_to([-3.55, 1.5, 0])
        sneaky = email("Cl1ck for fr33 m0ney").move_to([-3.55, -0.6, 0])
        window = card(6.0, 4.6, AMBER, CARD, radius=0.22, stroke=2).move_to([3.3, -0.25, 0])
        name = code("spam_rules.py", 20, MUTED).next_to(window.get_corner(m.UL), m.DR, buff=0.28)
        rules = m.VGroup(
            code('if "free money" in mail: spam', 22, INK, t2c={'"free money"': AMBER}),
            code('if "winner" in mail: spam', 22, INK, t2c={'"winner"': AMBER}),
            code('if "!!!" in mail: spam', 22, INK, t2c={'"!!!"': AMBER}),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        rules.next_to(name, m.DOWN, buff=0.4).align_to(name, m.LEFT)
        more = m.VGroup(
            code('if "fr33" in mail: spam', 22, MUTED),
            code('if "m0ney" in mail: spam', 22, MUTED),
            code("# … 300 more rules", 22, FAINT),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        more.next_to(rules, m.DOWN, buff=0.22).align_to(rules, m.LEFT)
        caught = pill("✓  caught", GOOD, GOOD_FILL, 22).next_to(first, m.DOWN, buff=0.3)
        missed = pill("✗  missed: no rule matches", BAD, BAD_FILL, 22).next_to(sneaky, m.DOWN, buff=0.3)
        note = text("hand-written rules break, and pile up", 26, BAD, weight=m.SEMIBOLD).move_to([0, -3.35, 0])

        self.say(
            "[Let's start with a classic problem: spam email.]",
            m.AnimationGroup(
                self.new_header("THE OLD WAY", "Writing every rule by hand", BAD, BAD_FILL),
                m.FadeIn(first, shift=0.3 * m.RIGHT),
            ),
        )
        self.say(
            "[The old way was to write the rules by hand]: [if an email says free money, it's spam.]",
            m.AnimationGroup(m.FadeIn(window), m.FadeIn(name)),
            m.AnimationGroup(
                m.AddTextLetterByLetter(rules[0], time_per_char=0.03),
                m.FadeIn(rules[1:], shift=0.1 * m.DOWN),
                m.FadeIn(caught, shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[But spammers adapt.] [They swap letters for numbers], [and the rule misses it.]",
            m.Wiggle(first, scale_value=1.05),
            m.FadeIn(sneaky, shift=0.3 * m.RIGHT),
            m.FadeIn(missed, scale=1.2),
        )
        self.say(
            "[So you add another rule, and another], [until you're maintaining hundreds of them], and they still break.",
            m.LaggedStart(*[m.FadeIn(line, shift=0.1 * m.DOWN) for line in more[:2]], lag_ratio=0.4),
            m.AnimationGroup(m.FadeIn(more[2]), m.Write(note)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the idea

    def idea(self):
        self.next_section("The idea", notes="Traditional programming versus machine learning.")

        def box(label, color, fill):
            return chip(label, color, fill, width=2.3, height=0.85, size=28)

        def row(title, ins, engine, out, y):
            heading = text(title, 28, MUTED, weight=m.BOLD).move_to([-4.9, y + 0.95, 0]).align_to([-6.6, 0, 0], m.LEFT)
            a, b = ins
            a.move_to([-5.2, y, 0])
            plus = text("+", 40, MUTED, weight=m.BOLD).move_to([-3.55, y, 0])
            b.move_to([-1.9, y, 0])
            engine.move_to([1.35, y, 0])
            out.move_to([4.95, y, 0])
            into = arrow(b.get_right(), engine.get_left(), MUTED, buff=0.15)
            outof = arrow(engine.get_right(), out.get_left(), MUTED, buff=0.15)
            return heading, a, plus, b, into, engine, outof, out

        top = row(
            "Traditional programming",
            (box("Rules", AMBER, AMBER_FILL), box("Data", SKY, SKY_FILL)),
            node("code", "Program", "follows the rules", EDGE, CARD, width=2.6, height=1.7, label_size=24),
            box("Answers", GOOD, GOOD_FILL),
            1.15,
        )
        bottom = row(
            "Machine learning",
            (box("Data", SKY, SKY_FILL), box("Answers", GOOD, GOOD_FILL)),
            node("llm", "Learning", "finds the pattern", VIOLET, VIOLET_FILL, width=2.6, height=1.7, label_size=24),
            box("Rules", AMBER, AMBER_FILL),
            -1.95,
        )
        examples = text("thousands of emails, labelled spam or not spam", 22, SKY).next_to(bottom[3], m.DOWN, buff=0.25)
        examples.align_to(bottom[1], m.LEFT)
        model = text("= a model", 24, AMBER, weight=m.BOLD).next_to(bottom[7], m.DOWN, buff=0.22)

        self.say(
            "[Machine learning flips this around.]",
            self.new_header("THE IDEA", "Learn the rules from examples"),
        )
        self.say(
            "[In traditional programming, you give the computer rules and data], [and it gives you answers.]",
            m.AnimationGroup(m.FadeIn(top[0]), m.LaggedStart(*[m.FadeIn(x, shift=0.2 * m.RIGHT) for x in top[1:4]], lag_ratio=0.3)),
            m.AnimationGroup(m.GrowArrow(top[4]), m.FadeIn(top[5]), m.GrowArrow(top[6]), m.FadeIn(top[7], shift=0.2 * m.RIGHT)),
        )
        self.say(
            "[In machine learning, you give it data, and the answers]: [thousands of emails, already labelled spam or not spam.]",
            m.AnimationGroup(m.FadeIn(bottom[0]), m.LaggedStart(*[m.FadeIn(x, shift=0.2 * m.RIGHT) for x in bottom[1:4]], lag_ratio=0.3)),
            m.FadeIn(examples, shift=0.15 * m.UP),
        )
        self.say(
            "[And the computer works out the rules by itself.] [We call what it learns a model.]",
            m.AnimationGroup(m.GrowArrow(bottom[4]), m.FadeIn(bottom[5]), m.GrowArrow(bottom[6]), m.FadeIn(bottom[7], shift=0.2 * m.RIGHT), pulse(bottom[6], AMBER)),
            m.FadeIn(model, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ how it learns

    def learning(self):
        self.next_section("How it learns", notes="Labelled emails, learned word weights, a prediction.")
        labelled = m.VGroup()
        for subject, spam in (
            ("Claim your prize", True), ("Team lunch Friday", False), ("You are a winner", True),
            ("Invoice attached", False), ("Free money inside", True), ("Notes from today", False),
        ):
            color, fill = (BAD, BAD_FILL) if spam else (GOOD, GOOD_FILL)
            box = card(4.0, 0.62, color, CARD, radius=0.15, stroke=2)
            label = text(subject, 22, INK).move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            tag = text("spam" if spam else "ok", 20, color, weight=m.BOLD).move_to(box).align_to(box, m.RIGHT).shift(0.22 * m.LEFT)
            labelled.add(m.VGroup(box, label, tag))
        labelled.arrange(m.DOWN, buff=0.14).move_to([-4.55, -0.35, 0])

        brain = icon("llm", 1.5).move_to([-1.0, -0.35, 0])
        brain_label = text("model", 24, VIOLET, weight=m.BOLD).next_to(brain, m.DOWN, buff=0.2)
        feed = arrow(labelled.get_right(), brain.get_left(), MUTED, buff=0.2)

        weights = [("free", 2.1), ("winner", 1.7), ("prize", 1.4), ("lunch", -1.2), ("invoice", -0.9), ("notes", -0.7)]
        bars = m.VGroup()
        zero_x = 3.6
        for k, (word, w) in enumerate(weights):
            y = 1.25 - 0.55 * k
            color = BAD if w > 0 else GOOD
            bar = m.Rectangle(width=abs(w) * 1.15, height=0.36, stroke_width=0).set_fill(color, opacity=0.85)
            bar.move_to([zero_x, y, 0], aligned_edge=m.LEFT if w > 0 else m.RIGHT)
            label = text(word, 22, INK).move_to([zero_x, y, 0])
            label.next_to(bar, m.LEFT if w > 0 else m.RIGHT, buff=0.15)
            if w > 0:
                label.next_to([zero_x, y, 0], m.LEFT, buff=0.15)
            else:
                label.next_to([zero_x, y, 0], m.RIGHT, buff=0.15)
            bars.add(m.VGroup(bar, label))
        axis = m.Line([zero_x, 1.6, 0], [zero_x, -1.85, 0]).set_stroke(FAINT, width=2)
        legend = m.VGroup(
            text("points to spam", 20, BAD, weight=m.SEMIBOLD),
            text("points to normal mail", 20, GOOD, weight=m.SEMIBOLD),
        ).arrange(m.RIGHT, buff=0.6).move_to([zero_x, 2.05, 0])
        learned = arrow(brain.get_right(), [axis.get_x() - 2.6, -0.35, 0], VIOLET, buff=0.25)

        new_mail = m.VGroup(icon("mail", 0.45), text("Cl1ck for fr33 m0ney", 22, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.25)
        new_mail.move_to([-1.0, -2.95, 0])
        verdict = pill("97% spam", BAD, BAD_FILL, 26).move_to([3.6, -2.95, 0])
        judge = arrow(new_mail.get_right(), verdict.get_left(), MUTED, buff=0.25)

        self.say(
            "[Here's how it learns.]",
            self.new_header("HOW IT LEARNS", "Patterns from labelled examples"),
        )
        self.say(
            "[We show the model thousands of labelled emails.]",
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(e, shift=0.2 * m.RIGHT) for e in labelled], lag_ratio=0.15),
                m.GrowArrow(feed),
                m.FadeIn(brain, scale=0.8),
                m.FadeIn(brain_label),
            ),
        )
        self.say(
            "[It notices which words tend to show up in spam], [and which show up in normal mail.]",
            m.AnimationGroup(
                m.GrowArrow(learned),
                m.Create(axis),
                m.FadeIn(legend[0]),
                m.LaggedStart(*[m.GrowFromEdge(b[0], m.LEFT) for b in bars[:3]], lag_ratio=0.2),
                m.LaggedStart(*[m.FadeIn(b[1]) for b in bars[:3]], lag_ratio=0.2),
            ),
            m.AnimationGroup(
                m.FadeIn(legend[1]),
                m.LaggedStart(*[m.GrowFromEdge(b[0], m.RIGHT) for b in bars[3:]], lag_ratio=0.2),
                m.LaggedStart(*[m.FadeIn(b[1]) for b in bars[3:]], lag_ratio=0.2),
            ),
        )
        self.say(
            "[Then a brand new email arrives], [and the model makes its best guess: ninety-seven percent spam.]",
            m.FadeIn(new_mail, shift=0.3 * m.RIGHT),
            m.AnimationGroup(m.GrowArrow(judge), m.FadeIn(verdict, scale=1.3)),
        )
        self.say(
            "[No one wrote a rule for that email.] The model learned the pattern from the data.",
            m.Circumscribe(verdict, color=BAD, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the family

    def family(self):
        self.next_section("The family", notes="AI ⊃ machine learning ⊃ deep learning ⊃ generative AI.")
        rings = m.VGroup()
        names = m.VGroup()
        # each label sits deep enough that its ring's chord is wider than it, and above the next ring
        specs = [
            ("AI", SKY, 2.9, 0.5),
            ("Machine learning", TEAL, 2.25, 0.65),
            ("Deep learning", VIOLET, 1.6, 0.76),
            ("Generative AI", PINK, 0.95, 0.95),
        ]
        bottom = -3.4
        for label, color, r, depth in specs:
            ring = m.Circle(radius=r).set_stroke(color, width=3).set_fill(color, opacity=0.08)
            ring.move_to([-3.75, bottom + r, 0])
            rings.add(ring)
            names.add(text(label, 20 if r < 1 else 24, color, weight=m.BOLD).move_to([-3.75, bottom + 2 * r - depth, 0]))
        rows = m.VGroup()
        for (label, color, _, _), desc in zip(
            specs,
            (
                "any machine that acts smart, even\none following hand-written rules",
                "learns patterns from data,\nlike our spam filter",
                "machine learning with many-layered\nneural networks: faces, speech",
                "deep learning that creates new\ntext, images and code: chatbots",
            ),
        ):
            rows.add(
                m.VGroup(text(label, 28, color, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            )
        rows.arrange(m.DOWN, buff=0.38, aligned_edge=m.LEFT).move_to([3.4, -0.45, 0])

        self.say(
            "[So how do all these terms fit together?]",
            self.new_header("THE BIG PICTURE", "How the terms fit together"),
        )
        lines = (
            "[Artificial intelligence is the big umbrella]: any machine that acts smart, even one following hand-written rules.",
            "[Machine learning is the part of AI that learns from data], like our spam filter.",
            "[Deep learning is machine learning with many-layered neural networks.] It's behind face and speech recognition.",
            "[And generative AI uses deep learning to create new things]: text, images, and code. That's today's chatbots.",
        )
        for line, ring, name, row in zip(lines, rings, names, rows):
            self.say(line, m.AnimationGroup(m.Create(ring), m.FadeIn(name), m.FadeIn(row, shift=0.2 * m.LEFT)))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ history

    def history(self):
        self.next_section("A short history", notes="1956, 1997, 2012, 2017, 2022.")
        line = m.Line([-6.4, 0, 0], [6.4, 0, 0]).set_stroke(FAINT, width=3)
        events = [
            ("1956", "Dartmouth\nworkshop names\nthe field", SKY),
            ("1997", "Deep Blue beats\nthe chess\nchampion", MUTED),
            ("2012", "Deep learning\nwins at seeing\nimages", VIOLET),
            ("2017", "The Transformer\nis invented", AMBER),
            ("2022", "ChatGPT brings\nAI to everyone", PINK),
        ]
        marks = m.VGroup()
        for (year, what, color), x in zip(events, (-5.3, -2.65, 0, 2.65, 5.3)):
            dot = m.Dot([x, 0, 0], radius=0.13, color=color)
            label = text(year, 34, color, weight=m.BOLD).next_to(dot, m.UP, buff=0.35)
            desc = text(what, 22, INK).next_to(dot, m.DOWN, buff=0.35)
            marks.add(m.VGroup(dot, label, desc))

        self.say(
            "[Here's how we got here.]",
            m.AnimationGroup(self.new_header("A SHORT HISTORY", "From rules to learning"), m.Create(line)),
        )
        lines = (
            "[In nineteen fifty-six, a summer workshop at Dartmouth gave the field its name.]",
            "[In nineteen ninety-seven, IBM's Deep Blue beat the world chess champion], mostly with search and hand-crafted rules.",
            "[In twenty twelve, deep learning suddenly started winning at recognising images.]",
            "[In twenty seventeen came the Transformer], the design behind modern language models.",
            "[And in twenty twenty-two, ChatGPT brought it to everyone.]",
        )
        for text_line, mark in zip(lines, marks):
            self.say(text_line, m.AnimationGroup(m.FadeIn(mark[0], scale=0.4), m.FadeIn(mark[1], shift=0.2 * m.DOWN), m.FadeIn(mark[2], shift=0.2 * m.UP)))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ narrow vs general

    def narrow(self):
        self.next_section("Narrow vs general", notes="Today's AI is narrow; general AI is a research goal.")
        tiles = m.VGroup(
            tile("check", "Narrow AI: today", "Trained for specific tasks,\noften superhuman at them.", GOOD),
            tile(recolored("sparkle", VIOLET, VIOLET_FILL), "General AI: the goal", "Learns anything a person\ncan. Still an open problem.", VIOLET),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -0.2, 0])

        self.say(
            "[One last idea: today's AI is powerful, but narrow.]",
            self.new_header("TODAY'S AI", "Powerful, but narrow"),
        )
        self.say("[Each system is trained for specific tasks], and can be superhuman at them.", m.FadeIn(tiles[0], shift=0.3 * m.UP))
        self.say(
            "[A general AI, one that could learn anything a person can], is still an open goal for researchers.",
            m.FadeIn(tiles[1], shift=0.3 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Examples, patterns, predictions.")
        halo = glow(SKY, radius=4.0)
        words = on_baseline(
            m.VGroup(
                *[text(w, 56, c, weight=m.BOLD) for w, c in zip(("Examples", "Patterns", "Predictions"), (SKY, AMBER, GOOD))]
            ).arrange(m.RIGHT, buff=1.2)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Instead of writing the rules, machines learn them.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [examples], [patterns], [predictions.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Instead of writing the rules ourselves, we let machines learn them from examples.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
