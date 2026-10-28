"""AI, Explained · Episode 3 · Neural Networks

Running example: a handwritten "7" on an 8×8 grid of pixels. A neuron (weighted sum, bias,
ReLU), layers of neurons, a forward pass that answers "7", what layers learn (edges, strokes,
shapes), why the activation function matters, and how the network is trained.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

SEVEN = [
    "01111110",
    "00000110",
    "00001100",
    "00001000",
    "00011000",
    "00010000",
    "00110000",
    "00100000",
]
PROBS = {0: 0.0, 1: 0.03, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.94, 8: 0.0, 9: 0.03}


def pixel_grid(cell=0.42):
    """The "7": squares whose brightness is the pixel's value."""
    rng = np.random.default_rng(7)
    squares = m.VGroup()
    values = []
    for r, row in enumerate(SEVEN):
        for c, bit in enumerate(row):
            v = float(rng.uniform(0.8, 1.0)) if bit == "1" else 0.0
            values.append(v)
            sq = m.Square(side_length=cell).set_stroke(EDGE, width=1.5).set_fill(INK, opacity=v)
            sq.move_to([c * cell, -r * cell, 0])
            squares.add(sq)
    return squares, values


class Ep03NeuralNetworks(Episode):
    number = 3

    def lesson(self):
        self.hook()
        self.neuron()
        self.network()
        self.features()
        self.why_activation()
        self.training()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A seven is just a grid of numbers.")
        grid, values = pixel_grid(0.5)
        grid.move_to([-2.6, -0.2, 0])
        sample = [f"{v:.2f}" for v in values[:3]] + [f"{values[-1]:.2f}"]
        numbers = vector(SKY, sample, size=26).move_to([2.2, -0.2, 0])
        count = text("64 numbers", 26, SKY, weight=m.BOLD).next_to(numbers, m.DOWN, buff=0.35)
        link = arrow(grid.get_right(), numbers.get_left(), MUTED, buff=0.35)
        see = text("you see: a seven", 26, INK, weight=m.SEMIBOLD).next_to(grid, m.DOWN, buff=0.35)

        self.say(
            "[Look at this.] You instantly see a seven.",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, scale=0.6) for s in grid], lag_ratio=0.01), m.FadeIn(see)),
        )
        self.say(
            "[But to a computer, it's just a grid of numbers]: [how bright each pixel is.]",
            m.GrowArrow(link),
            m.AnimationGroup(m.FadeIn(numbers, shift=0.3 * m.LEFT), m.FadeIn(count)),
        )
        self.say("[A neural network turns numbers like these into an answer.] Let's build one, piece by piece.", m.Indicate(grid, color=SKY, scale_factor=1.04))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the neuron

    def neuron(self):
        self.next_section("The neuron", notes="Weighted sum, bias, ReLU.")
        xs = [0.9, 0.2, 0.7]
        ws = [0.8, -0.5, 0.3]
        bias = 0.1
        total = sum(x * w for x, w in zip(xs, ws)) + bias
        inputs = m.VGroup()
        for k, x in enumerate(xs):
            dot = m.Circle(radius=0.38).set_stroke(SKY, width=3).set_fill(SKY_FILL, opacity=1)
            label = text(f"{x}", 26, INK, weight=m.BOLD).move_to(dot)
            inputs.add(m.VGroup(dot, label).move_to([-5.2, 1.3 - 1.45 * k, 0]))
        names = m.VGroup(*[tex(f"x_{k + 1}", 30, SKY).next_to(i, m.LEFT, buff=0.2) for k, i in enumerate(inputs)])
        body = m.Circle(radius=0.75).set_stroke(VIOLET, width=4).set_fill(VIOLET_FILL, opacity=1).move_to([-1.6, -0.15, 0])
        sigma = tex(r"\Sigma", 52, INK).move_to(body)
        edges, weights = m.VGroup(), m.VGroup()
        for i, w in zip(inputs, ws):
            color = GOOD if w > 0 else BAD
            line = m.Line(i.get_right(), body.get_center(), buff=0).set_stroke(color, width=2 + 6 * abs(w))
            line.put_start_and_end_on(i.get_right(), body.get_center() + 0.75 * m.normalize(i.get_center() - body.get_center()))
            edges.add(line)
            d = m.normalize(line.get_end() - line.get_start())
            n = np.array([-d[1], d[0], 0]) * (1 if d[0] >= 0 else -1)  # the side above the line
            weights.add(text(f"{w:+.1f}", 24, color, weight=m.BOLD).move_to(line.point_from_proportion(0.45) + 0.32 * n))
        bias_label = m.VGroup(text("bias", 20, MUTED), text(f"+{bias}", 24, AMBER, weight=m.BOLD)).arrange(m.RIGHT, buff=0.15)
        bias_label.next_to(body, m.DOWN, buff=0.3)
        sum_line = code("0.9×0.8 + 0.2×(−0.5) + 0.7×0.3 + 0.1 = 0.93", 22, INK).move_to([0.9, -2.7, 0])

        relu_box = card(2.4, 2.2, AMBER, AMBER_FILL).move_to([1.65, -0.15, 0])
        rax = axes([-1, 1, 1], [0, 1, 1], 1.7, 1.1, color=MUTED).move_to(relu_box.get_center() + 0.25 * m.DOWN)
        relu = rax.plot(lambda u: max(0.0, u), x_range=[-1, 1], color=AMBER).set_stroke(AMBER, width=4)
        relu_name = text("ReLU", 24, AMBER, weight=m.BOLD).move_to(relu_box.get_top() + 0.35 * m.DOWN)
        to_relu = arrow(body.get_right(), relu_box.get_left(), MUTED, buff=0.1)
        out_dot = m.Circle(radius=0.52).set_stroke(GOOD, width=3).set_fill(GOOD_FILL, opacity=1).move_to([4.45, -0.15, 0])
        out_val = text(f"{total:.2f}", 26, INK, weight=m.BOLD).move_to(out_dot)
        out = arrow(relu_box.get_right(), out_dot.get_left(), MUTED, buff=0.1)
        out_name = text("output", 22, GOOD, weight=m.SEMIBOLD).next_to(out_dot, m.DOWN, buff=0.2)
        formula = tex(r"a = \mathrm{ReLU}\left(w_1x_1 + w_2x_2 + w_3x_3 + b\right)", 38, INK).move_to([2.3, 2.15, 0])

        self.say(
            "[The basic building block is a neuron.]",
            m.AnimationGroup(self.new_header("THE NEURON", "Weigh the inputs, add them up"), m.FadeIn(body, scale=0.6), m.FadeIn(sigma)),
        )
        self.say(
            "[It takes some inputs], [multiplies each one by a weight], [and adds them up, plus a bias.]",
            m.LaggedStart(*[m.FadeIn(m.VGroup(i, n), shift=0.2 * m.RIGHT) for i, n in zip(inputs, names)], lag_ratio=0.2),
            m.AnimationGroup(m.LaggedStart(*[m.Create(e) for e in edges], lag_ratio=0.2), m.FadeIn(weights)),
            m.AnimationGroup(m.FadeIn(bias_label, shift=0.2 * m.UP), m.Write(sum_line)),
        )
        self.say(
            "[The weights decide how much each input matters]: [positive weights push the neuron up], [negative ones push it down.]",
            m.Indicate(weights, color=INK, scale_factor=1.15),
            m.AnimationGroup(*[m.ShowPassingFlash(e.copy().set_stroke(GOOD, width=10), time_width=0.6) for e, w in zip(edges, ws) if w > 0]),
            m.AnimationGroup(*[m.ShowPassingFlash(e.copy().set_stroke(BAD, width=10), time_width=0.6) for e, w in zip(edges, ws) if w < 0]),
        )
        self.say(
            "[Then an activation function decides how strongly the neuron fires.] "
            "[The most common one, ReLU, simply turns negative numbers into zero.]",
            m.AnimationGroup(m.GrowArrow(to_relu), m.FadeIn(relu_box), m.FadeIn(relu_name), m.Create(rax), m.GrowArrow(out), m.FadeIn(out_dot), m.FadeIn(out_val), m.FadeIn(out_name)),
            m.Create(relu),
        )
        self.say(
            "[In math, the neuron's output is ReLU of the weighted sum, plus the bias.]",
            m.Write(formula),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the network

    def network(self):
        self.next_section("The network", notes="Layers; a forward pass; the answer.")
        grid, values = pixel_grid(0.24)
        grid.move_to([-5.6, -0.3, 0])
        rng = np.random.default_rng(3)

        def layer(x, n, color, top=1.75, bottom=-2.35, radius=0.17):
            ys = np.linspace(top, bottom, n)
            return m.VGroup(*[m.Circle(radius=radius).set_stroke(color, width=2.5).set_fill(BG, opacity=1).move_to([x, y, 0]) for y in ys])

        inp = layer(-3.55, 7, SKY)
        dots = text("⋮", 30, SKY).move_to([-3.55, -2.75, 0])
        h1 = layer(-1.6, 6, VIOLET, 1.55, -2.15)
        h2 = layer(0.35, 6, VIOLET, 1.55, -2.15)
        out = layer(2.3, 10, GOOD, 1.85, -2.45, radius=0.15)
        digits = m.VGroup(*[text(str(k), 22, INK, weight=m.SEMIBOLD).next_to(o, m.RIGHT, buff=0.22) for k, o in enumerate(out)])

        def links(a, b):
            return m.VGroup(*[m.Line(p.get_center(), q.get_center(), buff=0.17).set_stroke(MUTED, width=1.2, opacity=0.3) for p in a for q in b])

        e1, e2, e3 = links(inp, h1), links(h1, h2), links(h2, out)
        for e in (e1, e2, e3):
            e.set_z_index(-1)
        feed = arrow(grid.get_right(), inp.get_left(), MUTED, buff=0.2)
        bars = m.VGroup()
        for k, o in enumerate(out):
            p = PROBS[k]
            bar = m.Rectangle(width=max(0.03, 2.0 * p), height=0.22, stroke_width=0).set_fill(GOOD if k == 7 else MUTED, opacity=0.9)
            bar.move_to([2.95, o.get_y(), 0], aligned_edge=m.LEFT)
            bars.add(bar)
        answer = pill("7 · 94%", GOOD, GOOD_FILL, 26).move_to([6.05, out[7].get_y(), 0])
        hidden = m.VGroup(h1, h2)
        brace = m.Brace(hidden, m.UP, buff=0.2).set_color(VIOLET)
        brace_label = text("hidden layers", 22, VIOLET, weight=m.SEMIBOLD).next_to(brace, m.UP, buff=0.1)
        deep = pill("a deep neural network: many layers", VIOLET, VIOLET_FILL, 22).move_to([0, -3.35, 0])

        def light(group, levels, color):
            return m.AnimationGroup(*[c.animate.set_fill(color, opacity=0.25 + 0.75 * v) for c, v in zip(group, levels)])

        in_levels = [values[i] for i in (1, 6, 13, 20, 27, 35, 42)]
        self.say(
            "[One neuron can't do much. So we line many up into a layer], [and stack layers one after another.]",
            m.AnimationGroup(self.new_header("THE NETWORK", "Neurons in layers"), m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in h1], lag_ratio=0.1)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in h2], lag_ratio=0.1), m.Create(e2)),
        )
        self.say(
            "[The pixels go in at one end], one input for each.",
            m.AnimationGroup(m.FadeIn(grid), m.GrowArrow(feed), m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in inp], lag_ratio=0.1), m.FadeIn(dots), m.Create(e1)),
        )
        self.say(
            "[The last layer has one neuron for each digit, from zero to nine.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in out], lag_ratio=0.06), m.FadeIn(digits), m.Create(e3)),
        )
        self.say(
            "[Each layer passes its outputs forward to the next]...",
            m.Succession(
                light(inp, in_levels, SKY),
                m.ShowPassingFlash(e1.copy().set_stroke(SKY, width=2.5, opacity=0.9), time_width=0.5, run_time=0.7),
                light(h1, rng.uniform(0, 1, 6), VIOLET),
                m.ShowPassingFlash(e2.copy().set_stroke(VIOLET, width=2.5, opacity=0.9), time_width=0.5, run_time=0.7),
                light(h2, rng.uniform(0, 1, 6), VIOLET),
                m.ShowPassingFlash(e3.copy().set_stroke(GOOD, width=2.5, opacity=0.9), time_width=0.5, run_time=0.7),
            ),
        )
        self.say(
            "[and the brightest output is the answer]: [a seven, with ninety-four percent confidence.]",
            m.AnimationGroup(light(out, [PROBS[k] for k in range(10)], GOOD), m.LaggedStart(*[m.GrowFromEdge(b, m.LEFT) for b in bars], lag_ratio=0.05)),
            m.FadeIn(answer, scale=1.2),
        )
        self.say(
            "[The layers in the middle are called hidden layers.] [With many of them, it's a deep neural network.]",
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(brace_label)),
            m.FadeIn(deep, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ what layers learn

    def features(self):
        self.next_section("What layers learn", notes="Edges, strokes, shapes.")

        def tile(lines, color):
            box = card(1.25, 1.25, color, CARD, radius=0.15, stroke=2)
            marks = m.VGroup(*[m.Line(box.get_center() + np.array(a), box.get_center() + np.array(b)).set_stroke(color, width=6) for a, b in lines])
            return m.VGroup(box, marks)

        def arc_tile(color):
            box = card(1.25, 1.25, color, CARD, radius=0.15, stroke=2)
            mark = m.Arc(radius=0.33, start_angle=0.3, angle=4.2).set_stroke(color, width=6).move_to(box)
            return m.VGroup(box, mark)

        def loop_tile(color):
            box = card(1.25, 1.25, color, CARD, radius=0.15, stroke=2)
            mark = m.Ellipse(width=0.55, height=0.78).set_stroke(color, width=6).move_to(box)
            return m.VGroup(box, mark)

        columns = [
            ("Early layers", "short edges", SKY, [
                tile([((-0.35, 0, 0), (0.35, 0, 0))], SKY),
                tile([((-0.3, -0.3, 0), (0.3, 0.3, 0))], SKY),
                tile([((0, -0.35, 0), (0, 0.35, 0))], SKY),
            ]),
            ("Middle layers", "strokes and corners", VIOLET, [
                tile([((-0.35, 0.3, 0), (0.35, 0.3, 0)), ((0.35, 0.3, 0), (0.35, -0.3, 0))], VIOLET),
                tile([((0.35, 0.35, 0), (-0.2, -0.35, 0))], VIOLET),
                arc_tile(VIOLET),
            ]),
            ("Later layers", "whole shapes", GOOD, [
                tile([((-0.35, 0.33, 0), (0.33, 0.33, 0)), ((0.33, 0.33, 0), (-0.1, -0.38, 0))], GOOD),
                tile([((0, 0.38, 0), (0, -0.38, 0)), ((0, 0.38, 0), (-0.2, 0.18, 0))], GOOD),
                loop_tile(GOOD),
            ]),
        ]
        groups = m.VGroup()
        for x, (title, sub, color, tiles) in zip((-4.4, 0, 4.4), columns):
            stack = m.VGroup(*tiles).arrange(m.DOWN, buff=0.25)
            heading = m.VGroup(text(title, 28, color, weight=m.BOLD), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.1)
            groups.add(m.VGroup(heading, stack).arrange(m.DOWN, buff=0.35).move_to([x, -0.35, 0]))
        for g in groups[1:]:  # headings on one line, whatever their letters
            g[0].align_to(groups[0][0], m.UP)
        flows = m.VGroup(*[arrow(a[1].get_right(), b[1].get_left(), MUTED, buff=0.5) for a, b in zip(groups, groups[1:])])
        seven = groups[2][1][0]
        note = text("Nobody programs these features: the network discovers them.", 26, INK, weight=m.SEMIBOLD).move_to([0, -3.45, 0])

        self.say(
            "[So what do all these layers actually do?]",
            self.new_header("WHAT LAYERS LEARN", "Simple patterns build complex ones"),
        )
        self.say("[Early layers learn to spot simple things, like short edges.]", m.FadeIn(groups[0], shift=0.3 * m.UP))
        self.say("[Middle layers combine edges into strokes and corners.]", m.AnimationGroup(m.GrowArrow(flows[0]), m.FadeIn(groups[1], shift=0.3 * m.UP)))
        self.say(
            "[Later layers combine those into whole shapes], [like the top bar and slanted line of a seven.]",
            m.AnimationGroup(m.GrowArrow(flows[1]), m.FadeIn(groups[2], shift=0.3 * m.UP)),
            m.Circumscribe(seven, color=GOOD, buff=0.08),
        )
        self.say("[Nobody programs these features.] The network discovers them during training.", m.FadeIn(note, shift=0.2 * m.UP))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ why activation

    def why_activation(self):
        self.next_section("Why activation", notes="Without nonlinearity, layers collapse to a straight line.")
        rng = np.random.default_rng(11)
        inner = [(r * np.cos(t), r * np.sin(t)) for r, t in zip(rng.uniform(0, 0.62, 14), rng.uniform(0, 2 * np.pi, 14))]
        outer = [(r * np.cos(t), r * np.sin(t)) for r, t in zip(rng.uniform(1.1, 1.32, 18), np.linspace(0, 2 * np.pi, 18, endpoint=False))]

        def panel(x, title, color):
            box = card(5.8, 4.6, color, CARD, stroke=2).move_to([x, -0.45, 0])
            heading = text(title, 26, color, weight=m.BOLD).move_to(box.get_top() + 0.42 * m.DOWN)
            centre = box.get_center() + 0.05 * m.UP
            points = m.VGroup(
                *[m.Dot(centre + np.array([px, py, 0]), radius=0.08, color=GOOD) for px, py in inner],
                *[m.Dot(centre + np.array([px, py, 0]), radius=0.08, color=BAD) for px, py in outer],
            )
            return m.VGroup(box, heading), points, centre

        left, lpoints, lc = panel(-3.3, "Without activation", BAD)
        right, rpoints, rc = panel(3.3, "With activation", GOOD)
        straight = m.Line(lc + np.array([-1.9, -1.05, 0]), lc + np.array([1.9, 1.05, 0])).set_stroke(SKY, width=4)
        bent = m.DashedVMobject(m.Circle(radius=0.85).set_stroke(SKY, width=4), num_dashes=40).move_to(rc)
        no = pill("✗  only straight lines", BAD, BAD_FILL, 22).next_to(left[0].get_bottom(), m.UP, buff=0.25)
        yes = pill("✓  bends to fit the data", GOOD, GOOD_FILL, 22).next_to(right[0].get_bottom(), m.UP, buff=0.25)

        self.say(
            "[Why do we need that activation function at all?]",
            self.new_header("WHY ACTIVATION?", "Bending the boundary"),
        )
        self.say(
            "[Without it, stacking layers just gives you another straight line], [no matter how many layers you use.]",
            m.AnimationGroup(m.FadeIn(left), m.LaggedStart(*[m.FadeIn(p, scale=0.4) for p in lpoints], lag_ratio=0.02), m.Create(straight)),
            m.FadeIn(no, shift=0.2 * m.UP),
        )
        self.say(
            "[With it, the network can bend its boundary], [to fit almost any shape.]",
            m.AnimationGroup(m.FadeIn(right), m.LaggedStart(*[m.FadeIn(p, scale=0.4) for p in rpoints], lag_ratio=0.02)),
            m.AnimationGroup(m.Create(bent), m.FadeIn(yes, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ training

    def training(self):
        self.next_section("How it learns", notes="Labelled digits, loss going down, backpropagation later.")
        samples = m.VGroup()
        for d in ("7", "2", "1", "0", "4", "9"):
            box = card(1.15, 1.15, SKY, CARD, radius=0.15, stroke=2)
            glyph = text(d, 44, INK, weight=m.BOLD).move_to(box)
            tag = text(f"label: {d}", 20, MUTED).next_to(box, m.DOWN, buff=0.12)
            samples.add(m.VGroup(box, glyph, tag))
        samples.arrange_in_grid(rows=2, cols=3, buff=(0.3, 0.45)).move_to([-4.3, -0.5, 0])
        lx = axes([0, 10, 1], [0, 2.5, 0.5], 5.6, 3.6).move_to([2.9, -0.5, 0])
        lx_label = text("training steps", 22, MUTED).next_to(lx.x_axis.get_end(), m.DOWN, buff=0.3).shift(0.6 * m.LEFT)
        ly_label = text("loss", 22, MUTED).next_to(lx.y_axis.get_end(), m.UP, buff=0.12)
        curve = lx.plot(lambda t: 0.08 + 2.2 * np.exp(-0.55 * t) + 0.04 * np.sin(5 * t) * np.exp(-0.2 * t), x_range=[0, 9.8], color=BAD)
        curve.set_stroke(BAD, width=4)
        weights = pill("a small digit network: about 100,000 weights", VIOLET, VIOLET_FILL, 22).move_to([0, -3.35, 0])
        later = pill("finding all their slopes at once: backpropagation · Episode 12", AMBER, AMBER_FILL, 20).move_to([0, -3.35, 0])

        self.say(
            "[How does the network find good weights?] Exactly the way our line did in the last episode.",
            self.new_header("HOW IT LEARNS", "The same recipe, many more weights"),
        )
        self.say(
            "[Show it thousands of labelled digits], [measure the loss], [and nudge every weight a little downhill.]",
            m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.UP) for s in samples], lag_ratio=0.12),
            m.AnimationGroup(m.Create(lx), m.FadeIn(lx_label), m.FadeIn(ly_label)),
            m.Create(curve, run_time=2.5),
        )
        self.say("[Even a small digit network has around a hundred thousand weights.]", m.FadeIn(weights, shift=0.2 * m.UP))
        self.say(
            "[Working out the slope for every one of them at once is called backpropagation], and it gets its own episode later on.",
            m.FadeTransform(weights, later),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Neurons, layers, features.")
        halo = glow(VIOLET, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Neurons", "Layers", "Features"), (SKY, VIOLET, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Simple units, stacked deep, learn to see.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [neurons], [layers], [features.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Simple units, stacked deep, learn to see patterns that no one wrote down.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
