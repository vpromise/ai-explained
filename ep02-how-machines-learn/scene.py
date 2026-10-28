"""AI, Explained · Episode 2 · How Machines Learn

Running example: predicting a house's price from its size with a line, price = w·size + b.
The loss (mean squared error), the loss valley, gradient descent step by step, the learning
rate, the training loop, and why the same recipe trains models with billions of parameters.

The data is built so the best slope (with b fixed at 40) is exactly 2: each descent step
halves the distance to it, and a 120 m² house is priced at 280 (thousand).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# houses: size (m²) → price (thousands); noise chosen so Σ size·noise = 0, so w* = 2 exactly
SIZES = np.array([50, 65, 80, 95, 110, 125, 140, 155], dtype=float)
_noise = np.array([15, -12, 10, -14, 8, -9, 12, 0], dtype=float)
_noise[-1] = -np.dot(SIZES[:-1], _noise[:-1]) / SIZES[-1]
B = 40.0
PRICES = 2 * SIZES + B + _noise
M2 = float(np.mean(SIZES**2))
START_W, BEST_W = 0.5, 2.0
STEPS = [START_W + (BEST_W - START_W) * (1 - 0.5**k) for k in range(1, 5)]  # halving the gap


def loss(w):
    return float(np.mean((w * SIZES + B - PRICES) ** 2))


class Ep02HowMachinesLearn(Episode):
    number = 2

    def lesson(self):
        self.w = m.ValueTracker(START_W)
        self.hook()
        self.task()
        self.measure()
        self.descend()
        self.learning_rate()
        self.loop()
        self.scale_up()
        self.recap()

    # ------------------------------------------------------------------ shared pieces

    def build_plot(self):
        """The houses, on axes at the left; the line and its errors follow the tracker w."""
        self.ax = axes([0, 180, 20], [0, 400, 50], 6.0, 4.3).move_to([-3.55, -0.65, 0])
        self.x_label = text("size", 22, MUTED).next_to(self.ax.x_axis.get_end(), m.DOWN, buff=0.18)
        self.y_label = text("price", 22, MUTED).next_to(self.ax.y_axis.get_end(), m.UP, buff=0.12)
        self.houses = m.VGroup(*[m.Dot(self.ax.c2p(x, y), radius=0.085, color=AMBER) for x, y in zip(SIZES, PRICES)])
        self.houses.set_z_index(2)

        def make_line():
            w = self.w.get_value()
            return m.Line(self.ax.c2p(10, w * 10 + B), self.ax.c2p(175, w * 175 + B)).set_stroke(SKY, width=5)

        def make_errors():
            w = self.w.get_value()
            return m.VGroup(
                *[
                    m.Line(self.ax.c2p(x, y), self.ax.c2p(x, w * x + B)).set_stroke(BAD, width=3, opacity=0.85)
                    for x, y in zip(SIZES, PRICES)
                ]
            )

        self.make_line, self.make_errors = make_line, make_errors

    def readouts(self):
        """w and the loss, live, above the plot."""
        w_value = m.DecimalNumber(START_W, num_decimal_places=2, font_size=34, color=SKY)
        w_value.add_updater(lambda d: d.set_value(self.w.get_value()))
        l_value = m.DecimalNumber(loss(START_W), num_decimal_places=0, font_size=34, color=BAD)
        l_value.add_updater(lambda d: d.set_value(loss(self.w.get_value())))
        row = m.VGroup(
            text("w =", 30, SKY, weight=m.BOLD), w_value, text("loss =", 30, BAD, weight=m.BOLD), l_value
        ).arrange(m.RIGHT, buff=0.2)
        row[2].shift(0.5 * m.RIGHT)
        row[3].next_to(row[2], m.RIGHT, buff=0.2)
        return row.move_to([-3.4, 2.2, 0])

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Guess, measure, adjust.")
        words = m.VGroup(
            *[text(w, 60, c, weight=m.BOLD) for w, c in zip(("Guess", "Measure", "Adjust"), (SKY, BAD, AMBER))]
        ).arrange(m.RIGHT, buff=1.2)
        on_baseline(words)
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        back = m.CurvedArrow(words[2].get_bottom() + 0.2 * m.DOWN, words[0].get_bottom() + 0.2 * m.DOWN, angle=-1.2)
        back.set_stroke(MUTED, width=4).set_color(MUTED)
        again = text("over and over", 26, MUTED).move_to([0, -2.2, 0])
        group = m.VGroup(words, links, back, again).move_to(0.2 * m.UP)

        self.say(
            "[How does a machine actually learn?] The short answer: [it guesses], [measures how wrong it was], [and adjusts.]",
            m.Wait(0.1),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Over and over again.] Let's watch that happen, one step at a time.", m.AnimationGroup(m.Create(back), m.FadeIn(again)))
        self.wait(0.3)
        self.play(m.FadeOut(group), run_time=0.8)

    # ------------------------------------------------------------------ the task and the model

    def task(self):
        self.next_section("The task", notes="Houses, a new house, a line with two knobs.")
        self.build_plot()
        new = self.ax.c2p(120, 0)
        marker = m.DashedLine(new, self.ax.c2p(120, 385), dash_length=0.1).set_stroke(MUTED, width=2.5)
        question = text("?", 40, INK, weight=m.BOLD).next_to(marker, m.UP, buff=0.08)
        new_label = text("new house", 20, MUTED).next_to(question, m.RIGHT, buff=0.15)
        self.marker = m.VGroup(marker, question, new_label)
        self.line = m.always_redraw(self.make_line)

        formula = tex(r"\text{price} = w \times \text{size} + b", 46, INK)
        formula.move_to([3.55, 1.1, 0])
        for part, color in ((r"w", SKY), (r"b", VIOLET)):
            pass  # coloured via the knobs below; the formula stays one colour for clarity
        knobs = m.VGroup(
            m.VGroup(text("w", 34, SKY, weight=m.BOLD), text("the slope: how fast price grows with size", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(text("b", 34, VIOLET, weight=m.BOLD), text("the starting price", 22, MUTED)).arrange(m.RIGHT, buff=0.3),
        ).arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT).move_to([3.55, -0.3, 0])
        params = pill("parameters: the knobs learning turns", SKY, SKY_FILL, 22).move_to([3.55, -1.6, 0])

        self.say(
            "[Say we want to predict the price of a house from its size.]",
            m.AnimationGroup(
                self.new_header("THE TASK", "Predict a price from a size"),
                m.Create(self.ax),
                m.FadeIn(self.x_label),
                m.FadeIn(self.y_label),
            ),
        )
        self.say(
            "[Here are some houses we've already seen]: their size, and what they sold for.",
            m.LaggedStart(*[m.FadeIn(h, scale=0.3) for h in self.houses], lag_ratio=0.15),
        )
        self.say("[A new house comes on the market.] What price should we expect?", m.AnimationGroup(m.Create(marker), m.FadeIn(question), m.FadeIn(new_label)))
        self.say(
            "[Our model is a straight line]: [price equals w times size, plus b.]",
            m.Create(self.line),
            m.Write(formula),
        )
        self.say(
            "[w and b are the model's parameters]: [two knobs that set the line.]",
            m.FadeIn(knobs, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(params), self.w.animate(rate_func=m.there_and_back).set_value(1.4)),
        )
        self.say(
            "[To keep things simple, we'll fix b, and only tune the slope.] [Learning means finding the w that predicts best.]",
            m.Indicate(knobs[1], color=VIOLET, scale_factor=1.05),
            m.Indicate(knobs[0], color=SKY, scale_factor=1.08),
        )
        self.formula = formula
        self.wait(0.3)
        self.play(*[m.FadeOut(mob) for mob in (formula, knobs, params)], run_time=0.7)

    # ------------------------------------------------------------------ the loss

    def measure(self):
        self.next_section("Measure: the loss", notes="Errors, squared, averaged.")
        errors = m.always_redraw(self.make_errors)
        errors.set_z_index(1)
        self.errors = errors
        mse = tex(r"L = \frac{1}{n}\sum_{i=1}^{n} \left(\hat{y}_i - y_i\right)^2", 44, INK).move_to([3.6, 0.9, 0])
        legend = m.VGroup(
            m.VGroup(tex(r"\hat{y}_i", 34, SKY), text("the line's prediction", 22, MUTED)).arrange(m.RIGHT, buff=0.25),
            m.VGroup(tex(r"y_i", 34, AMBER), text("the real price", 22, MUTED)).arrange(m.RIGHT, buff=0.25),
        ).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([3.6, -0.6, 0])
        self.read = self.readouts()
        better = text("a better line → a smaller loss", 24, GOOD, weight=m.SEMIBOLD).move_to([3.6, -1.9, 0])

        self.say(
            "[First, we need to measure how wrong a line is.]",
            self.new_header("STEP 1 · MEASURE", "The loss: how wrong are we?"),
        )
        self.say(
            "[For each house, the error is the gap between the real price and the line.]",
            m.Create(errors),
        )
        self.say(
            "[We square each gap, so big mistakes count more, and average them.] [That number is called the loss.]",
            m.Write(mse),
            m.AnimationGroup(m.FadeIn(legend, shift=0.2 * m.UP), m.FadeIn(self.read, shift=0.2 * m.DOWN)),
        )
        self.say(
            "[A better line has smaller gaps], [and a smaller loss.]",
            self.w.animate.set_value(1.3),
            m.FadeIn(better, shift=0.2 * m.UP),
        )
        self.say(
            "[But how do we find the best line, without guessing forever?]",
            m.AnimationGroup(self.w.animate.set_value(START_W), m.FadeOut(better)),
        )
        self.wait(0.2)
        self.play(m.FadeOut(mse), m.FadeOut(legend), run_time=0.7)

    # ------------------------------------------------------------------ gradient descent

    def descend(self):
        self.next_section("Adjust: gradient descent", notes="The loss valley, the slope, steps downhill.")
        lx = axes([0, 4, 1], [0, 40000, 10000], 5.2, 3.9).move_to([3.55, -0.5, 0])
        lx_label = text("slope w", 22, MUTED).next_to(lx.x_axis.get_end(), m.DOWN, buff=0.3).shift(0.35 * m.LEFT)
        ly_label = text("loss", 22, MUTED).next_to(lx.y_axis.get_end(), m.UP, buff=0.12)
        valley = lx.plot(loss, x_range=[0.2, 3.8], color=BAD)
        valley.set_stroke(BAD, width=4)
        best = m.Dot(lx.c2p(BEST_W, loss(BEST_W)), radius=0.08, color=GOOD)

        def make_ball():
            w = self.w.get_value()
            return m.Dot(lx.c2p(w, loss(w)), radius=0.14, color=SKY).set_z_index(3)

        def make_tangent():
            w = self.w.get_value()
            slope = 2 * (w - BEST_W) * M2
            d = min(0.45, 5000 / max(abs(slope), 1e-9))  # a steep tangent stays short
            return m.Line(lx.c2p(w - d, loss(w) - slope * d), lx.c2p(w + d, loss(w) + slope * d)).set_stroke(AMBER, width=4)

        ball = m.always_redraw(make_ball)
        tangent = m.always_redraw(make_tangent)
        rule = tex(r"w \leftarrow w - \eta\,\frac{\partial L}{\partial w}", 44, INK).move_to([3.55, 2.15, 0])
        rule_note = text("η: the learning rate (step size)", 20, MUTED).next_to(rule, m.DOWN, buff=0.15)
        name = pill("gradient descent", AMBER, AMBER_FILL, 22).next_to(lx, m.DOWN, buff=0.2).shift(1.0 * m.LEFT)
        prediction = 120 * BEST_W + B
        guess_dot = m.Dot(self.ax.c2p(120, prediction), radius=0.12, color=GOOD).set_z_index(3)
        guess = pill(f"≈ ${prediction:.0f}k", GOOD, GOOD_FILL, 22).next_to(guess_dot, m.LEFT, buff=0.2).shift(0.3 * m.UP)

        self.say(
            "[Picture the loss for every possible slope]: [it makes a valley.]",
            m.AnimationGroup(
                self.new_header("STEP 2 · ADJUST", "Walk downhill on the loss"),
                m.Create(lx),
                m.FadeIn(lx_label),
                m.FadeIn(ly_label),
            ),
            m.Create(valley),
        )
        self.say("[Our current line sits up here, high on the side of the valley.]", m.FadeIn(ball, scale=0.4))
        self.say(
            "[The slope of the valley at that point tells us which way is downhill.]",
            m.Create(tangent),
        )
        self.say(
            "[So we take a small step downhill]: [w becomes w, minus the learning rate times the slope.]",
            self.w.animate.set_value(STEPS[0]),
            m.AnimationGroup(m.Write(rule), m.FadeIn(rule_note)),
        )
        self.say(
            "[And again], [and again], [each step a little closer to the bottom.]",
            self.w.animate.set_value(STEPS[1]),
            self.w.animate.set_value(STEPS[2]),
            self.w.animate.set_value(STEPS[3]),
        )
        self.say(
            "[At the bottom the slope is flat, and the loss is as small as it gets.] [That's our best line.]",
            m.AnimationGroup(self.w.animate.set_value(BEST_W), m.FadeIn(best)),
            m.Flash(lx.c2p(BEST_W, loss(BEST_W)), color=GOOD, flash_radius=0.4),
        )
        self.say(
            "[Now we can price the new house]: [about two hundred and eighty thousand dollars.]",
            m.FadeIn(guess_dot, scale=0.4),
            m.FadeIn(guess, shift=0.2 * m.LEFT),
        )
        self.say("[This method is called gradient descent.]", m.FadeIn(name, scale=1.2))
        self.wait(0.3)
        self.stop(ball, tangent, self.line, self.errors, *self.read[1::2])
        self.clear()

    # ------------------------------------------------------------------ the learning rate

    def learning_rate(self):
        self.next_section("The learning rate", notes="Too small, too big, just right.")

        def panel(title, color, factor, start, steps, x):
            box = card(4.1, 4.3, color, CARD, stroke=2).move_to([x, -0.45, 0])
            heading = text(title, 26, color, weight=m.BOLD).move_to(box.get_top() + 0.45 * m.DOWN)
            ax = axes([-2.4, 2.4, 1], [0, 7, 1], 3.3, 2.7).move_to(box.get_center() + 0.3 * m.DOWN)
            bowl = ax.plot(lambda u: u**2, x_range=[-2.4, 2.4], color=MUTED).set_stroke(MUTED, width=3)
            us = [start]
            for _ in range(steps):
                us.append(us[-1] * factor)
            inside = [u for u in us if abs(u) <= 2.4]
            dots = m.VGroup(*[m.Dot(ax.c2p(u, u**2), radius=0.08, color=color) for u in inside])
            hops = m.VGroup(  # plain arcs: short hops with arrowheads turn into blobs
                *[
                    m.ArcBetweenPoints(a.get_center(), b.get_center(), angle=-0.8 if b.get_x() > a.get_x() else 0.8).set_stroke(color, width=3)
                    for a, b in zip(dots, dots[1:])
                ]
            )
            if len(inside) < len(us):  # it left the valley: one last arrow off the top
                last = dots[-1].get_center()
                out = arrow(last, last + np.array([0.7 if us[len(inside)] > 0 else -0.7, 0.45, 0]), color, buff=0.1)
                hops.add(out)
            return m.VGroup(box, heading, ax, bowl), dots, hops

        small = panel("Too small: crawls", SKY, 0.86, -1.9, 5, -4.45)
        big = panel("Too big: overshoots", BAD, -1.3, -1.25, 3, 0)
        right = panel("Just right: settles", GOOD, 0.45, -1.9, 5, 4.45)
        line = text("The learning rate is one of the most important settings in training.", 26, INK, weight=m.SEMIBOLD)
        line.move_to([0, -3.3, 0])

        def show(p):
            frame, dots, hops = p
            return m.AnimationGroup(
                m.FadeIn(frame, shift=0.2 * m.UP),
                m.Succession(*[m.FadeIn(dots[0], scale=0.5, run_time=0.3), *[m.AnimationGroup(m.Create(h), m.FadeIn(d, scale=0.5), run_time=0.45) for h, d in zip(hops, list(dots[1:]) + [m.VGroup()] * 2)]]),
            )

        self.say(
            "[How big should each step be?] That's set by the learning rate.",
            self.new_header("THE LEARNING RATE", "How big a step?"),
        )
        self.say("[Too small, and learning crawls.]", show(small))
        self.say("[Too big, and it overshoots], bouncing right out of the valley.", show(big))
        self.say("[Just right, and it settles quickly at the bottom.]", show(right))
        self.say("[Choosing it well is one of the most important settings in training.]", m.FadeIn(line, shift=0.2 * m.UP))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the training loop

    def loop(self):
        self.next_section("The training loop", notes="Predict, measure, find the slope, adjust; epochs.")
        spots = {"top": [0, 1.25, 0], "right": [4.0, -0.55, 0], "bottom": [0, -2.35, 0], "left": [-4.0, -0.55, 0]}
        steps = m.VGroup(
            node(recolored("answer", SKY, SKY_FILL), "Predict", "run the model", SKY, SKY_FILL, width=2.6, height=1.7, label_size=24).move_to(spots["top"]),
            node("alert", "Measure", "compute the loss", BAD, BAD_FILL, width=2.6, height=1.7, label_size=24).move_to(spots["right"]),
            node(recolored("chart", AMBER, AMBER_FILL), "Find the slope", "the gradient", AMBER, AMBER_FILL, width=2.6, height=1.7, label_size=24).move_to(spots["bottom"]),
            node(recolored("gear", VIOLET, VIOLET_FILL), "Adjust", "step downhill", VIOLET, VIOLET_FILL, width=2.6, height=1.7, label_size=24).move_to(spots["left"]),
        )
        links = m.VGroup(
            m.CurvedArrow(steps[0].get_right() + 0.1 * m.RIGHT, steps[1].get_top() + 0.1 * m.UP, angle=-0.9),
            m.CurvedArrow(steps[1].get_bottom() + 0.1 * m.DOWN, steps[2].get_right() + 0.1 * m.RIGHT, angle=-0.9),
            m.CurvedArrow(steps[2].get_left() + 0.1 * m.LEFT, steps[3].get_bottom() + 0.1 * m.DOWN, angle=-0.9),
            m.CurvedArrow(steps[3].get_top() + 0.1 * m.UP, steps[0].get_left() + 0.1 * m.LEFT, angle=-0.9),
        )
        for a in links:
            a.set_stroke(MUTED, width=4).set_color(MUTED)
        centre = m.VGroup(text("repeat", 30, INK, weight=m.BOLD), text("thousands of times", 22, MUTED)).arrange(m.DOWN, buff=0.12).move_to([0, -0.55, 0])
        epoch = text("one pass over all the training data = one epoch", 22, MUTED).move_to([0, -3.55, 0])
        laps = [
            m.AnimationGroup(*[m.ShowPassingFlash(a.copy().set_stroke(c, width=8), time_width=0.6) for a, c in zip(links, (SKY, BAD, AMBER, VIOLET))], lag_ratio=0.25, run_time=1.6)
            for _ in range(2)
        ]

        self.say(
            "[Put it all together, and you get the training loop.]",
            self.new_header("THE TRAINING LOOP", "Guess, measure, adjust, repeat"),
        )
        self.say(
            "[Make predictions], [measure the loss], [find the slope], [and adjust the parameters.]",
            *[m.AnimationGroup(m.FadeIn(s, shift=0.2 * m.UP), m.Create(a)) for s, a in zip(steps, links)],
        )
        self.say("[Then repeat, thousands or even millions of times.]", m.AnimationGroup(m.FadeIn(centre), m.Succession(*laps)))
        self.say("[Each full pass through the training data is called an epoch.]", m.FadeIn(epoch, shift=0.2 * m.UP))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ two knobs or billions

    def scale_up(self):
        self.next_section("Two knobs or billions", notes="The same recipe at scale; backpropagation later.")
        ours = card(4.2, 3.3, SKY, CARD, stroke=2).move_to([-4.3, -0.4, 0])
        ours_title = text("Our line", 28, SKY, weight=m.BOLD).move_to(ours.get_top() + 0.45 * m.DOWN)
        knobs = m.VGroup(
            m.VGroup(m.Circle(radius=0.42).set_stroke(SKY, width=4), text("w", 30, SKY, weight=m.BOLD)),
            m.VGroup(m.Circle(radius=0.42).set_stroke(VIOLET, width=4), text("b", 30, VIOLET, weight=m.BOLD)),
        ).arrange(m.RIGHT, buff=0.7).move_to(ours.get_center() + 0.1 * m.DOWN)
        ours_note = text("2 parameters", 22, MUTED).move_to(ours.get_bottom() + 0.45 * m.UP)

        grid = m.VGroup(*[m.Dot(radius=0.05, color=VIOLET) for _ in range(26 * 11)])
        grid.arrange_in_grid(rows=11, cols=26, buff=0.12).move_to([2.6, -0.35, 0])
        grid_title = text("A large language model", 28, VIOLET, weight=m.BOLD).next_to(grid, m.UP, buff=0.35)
        grid_note = text("billions of parameters", 22, MUTED).next_to(grid, m.DOWN, buff=0.3)
        same = pill("the same recipe: gradient descent", GOOD, GOOD_FILL, 22).move_to([0, -3.2, 0])
        later = pill("computing all those slopes: backpropagation · Episode 12", AMBER, AMBER_FILL, 20).move_to([0, -3.2, 0])
        wave = m.LaggedStart(*[d.animate(rate_func=m.there_and_back).set_color(GOOD).scale(1.6) for d in grid], lag_ratio=0.004, run_time=2.4)

        self.say(
            "[Here's the remarkable part.]",
            self.new_header("THE BIG PICTURE", "Two knobs, or billions"),
        )
        self.say(
            "[Our line had just two parameters.] [A large language model has billions], [but it learns with this very same recipe.]",
            m.AnimationGroup(m.FadeIn(ours), m.FadeIn(ours_title), m.FadeIn(knobs, scale=0.8), m.FadeIn(ours_note)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.2) for d in grid], lag_ratio=0.003), m.FadeIn(grid_title), m.FadeIn(grid_note)),
            m.FadeIn(same, shift=0.2 * m.UP),
        )
        self.say(
            "[Every parameter gets its own slope, and they all take a small step downhill together.]",
            wave,
        )
        self.say(
            "[Working out all those slopes efficiently is called backpropagation.] It's coming up later in the series.",
            m.FadeTransform(same, later),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Guess, measure, adjust.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Guess", "Measure", "Adjust"), (SKY, BAD, AMBER))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Learning is making mistakes a little smaller, one step at a time.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [guess], [measure], [adjust.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A machine learns by making its mistakes a little smaller, one step at a time.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
