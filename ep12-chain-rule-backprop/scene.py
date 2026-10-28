"""AI, Explained · Episode 12 · The Chain Rule & Backpropagation

Running example: a tiny network worked by hand. One input x = 2, two hidden ReLU units, one
output, squared loss against a target of 3:

    z1 = w1·x,  z2 = w2·x,  h = ReLU(z),  y = v1·h1 + v2·h2,  L = (y − t)²
    weights (for example): w1 = 2, w2 = 1, v1 = 1, v2 = −1

The forward pass, the chain rule on a single chain (nudges multiply), the backward pass
multiplying local gradients along each path, the input's gradient as a sum where two paths
meet, one gradient-descent step (learning rate 0.01, for example), and the big picture
(about twice the forward pass's cost; PyTorch and JAX do it by automatic differentiation).

Every number on screen and in the narration is computed below, checked against numerical
(finite-difference) gradients with numpy, and asserted, so the film cannot drift from its words.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the network, by hand

X_IN, TARGET, LR = 2.0, 3.0, 0.01
START = {"w1": 2.0, "w2": 1.0, "v1": 1.0, "v2": -1.0}


def relu(z):
    return max(0.0, z)


def forward(p, x=X_IN):
    z1, z2 = p["w1"] * x, p["w2"] * x
    h1, h2 = relu(z1), relu(z2)
    y = p["v1"] * h1 + p["v2"] * h2
    return {"x": x, "z1": z1, "z2": z2, "h1": h1, "h2": h2, "y": y, "L": (y - TARGET) ** 2}


def backward(p, f):
    """Backpropagation by hand: each gradient = the gradient arriving × the local gradient."""
    g = {"L": 1.0, "y": 2 * (f["y"] - TARGET)}  # L = (y − t)²  ⇒  ∂L/∂y = 2(y − t)
    g["v1"], g["v2"] = g["y"] * f["h1"], g["y"] * f["h2"]  # ∂y/∂v = h
    g["h1"], g["h2"] = g["y"] * p["v1"], g["y"] * p["v2"]  # ∂y/∂h = v
    g["z1"], g["z2"] = g["h1"] * (f["z1"] > 0), g["h2"] * (f["z2"] > 0)  # ReLU: 1 if on, else 0
    g["w1"], g["w2"] = g["z1"] * f["x"], g["z2"] * f["x"]  # ∂z/∂w = x
    g["x_top"], g["x_bottom"] = g["z1"] * p["w1"], g["z2"] * p["w2"]  # ∂z/∂x = w, one per path
    g["x"] = g["x_top"] + g["x_bottom"]  # where paths meet, gradients add
    return g


def numeric_gradient(p, key, eps=1e-6):
    """Central differences on the loss: the honest check on the hand-made backward pass."""
    if key == "x":
        return (forward(p, X_IN + eps)["L"] - forward(p, X_IN - eps)["L"]) / (2 * eps)
    up, down = dict(p), dict(p)
    up[key] += eps
    down[key] -= eps
    return (forward(up)["L"] - forward(down)["L"]) / (2 * eps)


FWD = forward(START)
GRAD = backward(START, FWD)
NEW = {k: START[k] - LR * GRAD[k] for k in START}
FWD2 = forward(NEW)

for _k in (*START, "x"):
    assert np.isclose(numeric_gradient(START, _k), GRAD[_k], atol=1e-6), _k
# the narration's numbers
assert (FWD["z1"], FWD["z2"], FWD["h1"], FWD["h2"]) == (4, 2, 4, 2)  # "four on top, two below"
assert FWD["y"] == 2 and FWD["L"] == 1  # "that's two", "squared, is one"
assert GRAD["y"] == -2  # "two times the error: minus two"
assert (GRAD["v1"], GRAD["v2"]) == (-8, -4)  # "minus eight", "minus four"
assert (GRAD["h1"], GRAD["h2"]) == (-2, 2)  # "minus two", "plus two"
assert (GRAD["z1"], GRAD["z2"]) == (-2, 2)  # both ReLUs on: passed straight through
assert (GRAD["w1"], GRAD["w2"]) == (-4, 4)  # "minus four", "plus four"
assert (GRAD["x_top"], GRAD["x_bottom"], GRAD["x"]) == (-4, 2, -2)  # "minus four plus two, minus two"
assert all((NEW[k] > START[k]) == (GRAD[k] < 0) for k in START)  # negative gradients push weights up
assert np.allclose([NEW[k] for k in ("w1", "w2", "v1", "v2")], [2.04, 0.96, 1.08, -0.96])
assert 2 < FWD2["y"] < TARGET and round(FWD2["y"], 2) == 2.56  # "climbs from two toward three"
assert 0.15 < FWD2["L"] < 0.2  # "from one to under a fifth"

# the same step, repeated: the tiny network does learn (section notes)
_p = dict(START)
for _ in range(200):
    _g = backward(_p, forward(_p))
    _p = {k: _p[k] - LR * _g[k] for k in _p}
LOSS_AFTER_200 = forward(_p)["L"]
assert LOSS_AFTER_200 < 1e-4

# the chain-rule picture: a nudge to w moves y twice as much, a nudge to y moves L three times
DY_DW, DL_DY = 2, 3
assert DY_DW * DL_DY == 6  # "two times three: six times as much"

# ---------------------------------------------------------------------------- layout of the graph

R = 0.5  # node radius
COLS = {"x": -5.3, "z": -2.65, "h": 0.0, "y": 2.65, "L": 5.3}
TOP, MID, BOT = 1.35, -0.15, -1.65
WORK_Y = -3.12  # the line where formulas are worked out
POS = {
    "x": (COLS["x"], MID), "z1": (COLS["z"], TOP), "h1": (COLS["h"], TOP), "z2": (COLS["z"], BOT),
    "h2": (COLS["h"], BOT), "y": (COLS["y"], MID), "L": (COLS["L"], MID),
}
NAME_TEX = {"x": "x", "z1": "z_1", "h1": "h_1", "z2": "z_2", "h2": "h_2", "y": "y", "L": "L"}
EDGES = [("x", "z1"), ("x", "z2"), ("z1", "h1"), ("z2", "h2"), ("h1", "y"), ("h2", "y"), ("y", "L")]
WEIGHT_TEX = {"w1": "w_1", "w2": "w_2", "v1": "v_1", "v2": "v_2"}
WEIGHT_AT = {"w1": (-4.35, 1.2), "w2": (-4.35, -1.5), "v1": (1.6, 1.12), "v2": (1.6, -1.42)}
WEIGHT_SIDE = {"w1": 1, "w2": -1, "v1": 1, "v2": -1}  # where its gradient chip goes: above or below
WEIGHT_COLOR, GRAD_COLOR = VIOLET, AMBER  # parameters violet, gradients amber (the slope, as in Ep 2)


def num(v, plus=False):
    """A value as shown on screen: whole numbers plain, others to two decimals, a true minus."""
    v = float(v)
    s = f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}"
    if plus and v > 0:
        s = "+" + s
    return s.replace("-", "−")


def tnum(v):
    """A value for LaTeX."""
    v = float(v)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def grad_chip(value):
    return pill(num(value, plus=True), GRAD_COLOR, AMBER_FILL, 22)


def flow(edge, color, backward=False, width=8):
    a, b = edge.get_start(), edge.get_end()
    if backward:
        a, b = b, a
    return m.ShowPassingFlash(m.Line(a, b).set_stroke(color, width=width), time_width=0.7)


# ---------------------------------------------------------------------------- the episode


class Ep12ChainRuleBackprop(Episode):
    number = 12

    def lesson(self):
        self.work_mob = None
        self.hook()
        self.network()
        self.forward_pass()
        self.chain_rule()
        self.backward_pass()
        self.paths_meet()
        self.one_step()
        self.big_picture()
        self.recap()

    # ------------------------------------------------------------------ shared pieces

    def work(self, mob):
        """Swap the formula on the work line (bottom centre)."""
        mob.move_to([0, WORK_Y, 0])
        parts = [m.FadeIn(mob, shift=0.15 * m.UP)]
        if self.work_mob is not None:
            parts.insert(0, m.FadeOut(self.work_mob, shift=0.15 * m.UP))
        self.work_mob = mob
        return m.AnimationGroup(*parts)

    def build_graph(self):
        self.nodes, self.names, self.values, self.node_chips, self.weight_chips = {}, {}, {}, {}, {}
        for key, (x, y) in POS.items():
            color, fill = (BAD, BAD_FILL) if key == "L" else (SKY, SKY_FILL)
            self.nodes[key] = m.Circle(radius=R).set_stroke(color, width=3.5).set_fill(fill, opacity=1).move_to([x, y, 0])
        for key, node in self.nodes.items():
            name = tex(NAME_TEX[key], 34, BAD if key == "L" else SKY)
            if key == "x":
                name.next_to(node, m.LEFT, buff=0.22)
            elif key in ("z2", "h2"):
                name.next_to(node, m.DOWN, buff=0.14)
            else:
                name.next_to(node, m.UP, buff=0.14)
            self.names[key] = name
        self.edges = {}
        for a, b in EDGES:
            self.edges[(a, b)] = m.Arrow(
                self.nodes[a].get_center(), self.nodes[b].get_center(), buff=R + 0.05,
                color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.13,
            ).set_z_index(-1)
        self.relu_labels = m.VGroup(
            text("ReLU", 22, MUTED, weight=m.SEMIBOLD).move_to(self.edges[("z1", "h1")].get_center() + 0.3 * m.UP),
            text("ReLU", 22, MUTED, weight=m.SEMIBOLD).move_to(self.edges[("z2", "h2")].get_center() + 0.3 * m.DOWN),
        )
        self.weights = {k: self.weight_label(k, START[k]) for k in START}
        self.loss_label = tex(r"(y - t)^2", 30, MUTED).move_to(self.edges[("y", "L")].get_center() + 0.36 * m.UP)
        self.target = m.VGroup(text("target", 22, GOOD, weight=m.SEMIBOLD), tex("t = 3", 32, GOOD)).arrange(m.RIGHT, buff=0.18)
        self.target.next_to(self.nodes["L"], m.DOWN, buff=0.28)
        assert TARGET == 3

    def weight_label(self, key, value, color=WEIGHT_COLOR, like=None):
        label = tex(f"{WEIGHT_TEX[key]} = {tnum(value)}", 32, color).move_to([*WEIGHT_AT[key], 0])
        if like is not None:  # a new value grows outward, away from its edge
            label.align_to(like, m.RIGHT if key.startswith("w") else m.LEFT)
        return label

    def value(self, key, v, color=INK):
        words = text(num(v), 26, color, weight=m.BOLD)
        if words.width > 0.72:
            words.scale(0.72 / words.width)
        return words.move_to(self.nodes[key])

    def node_chip(self, key, v):
        x, y = POS[key]
        below = key in ("x", "z1", "h1", "y")
        return grad_chip(v).move_to([x, y + (-1 if below else 1) * (R + 0.36), 0])

    def weight_chip(self, key, v):
        x, y = WEIGHT_AT[key]
        return grad_chip(v).move_to([x, y + WEIGHT_SIDE[key] * 0.6, 0])

    def graph_parts(self):
        """Everything of the graph on stage now (for hiding it and bringing it back)."""
        parts = [*self.edges.values(), *self.nodes.values(), *self.names.values(), *self.values.values()]
        parts += [self.relu_labels, *self.weights.values(), self.loss_label, self.target]
        parts += [*self.node_chips.values(), *self.weight_chips.values()]
        return parts

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Every weight needs its gradient; millions of weights; backpropagation (Rumelhart, Hinton & Williams, Nature 1986).")
        sizes = [4, 6, 6, 6, 3]
        xs = np.linspace(-5.9, 0.6, len(sizes))
        layers = []
        for x, n in zip(xs, sizes):
            ys = -0.72 * (np.arange(n) - (n - 1) / 2) - 0.25
            layers.append(m.VGroup(*[m.Circle(radius=0.16).set_stroke(SKY, width=2.5).set_fill(BG, opacity=1).move_to([x, y, 0]) for y in ys]))
        edges = m.VGroup()
        for a, b in zip(layers, layers[1:]):
            for p in a:
                for q in b:
                    edges.add(m.Line(p.get_center(), q.get_center(), buff=0.16).set_stroke(MUTED, width=1.2, opacity=0.35))
        edges.set_z_index(-1)
        chosen = edges[4 * 6 + 2 * 6 + 3]  # second hidden layer, third neuron → fourth neuron of the next
        hot = m.Line(chosen.get_start(), chosen.get_end()).set_stroke(GRAD_COLOR, width=5)
        net = m.VGroup(edges, *layers)
        centre = -net.get_center()[0] * m.RIGHT
        net.shift(centre)  # starts centred, slides left when the question appears
        ask = tex(r"\frac{\partial L}{\partial w} = \;?", 52, GRAD_COLOR).move_to([4.3, 1.2, 0])
        ask_note = text("how the loss changes\nwhen this one weight changes", 22, MUTED).next_to(ask, m.DOWN, buff=0.3)
        many = pill("millions of weights", WEIGHT_COLOR, VIOLET_FILL, 24).move_to([4.3, -1.0, 0])
        all_at_once = text("every gradient, at once?", 28, INK, weight=m.SEMIBOLD).next_to(many, m.DOWN, buff=0.45)

        def wave(color, run_time):
            return m.LaggedStart(*[e.animate(rate_func=m.there_and_back).set_stroke(color, opacity=0.9, width=2.4) for e in edges], lag_ratio=0.01, run_time=run_time)

        halo = glow(AMBER, radius=4.0).move_to(0.3 * m.UP)
        title = text("Backpropagation", 84, INK, weight=m.BOLD)
        cite = text("Rumelhart et al., 1986", 22, MUTED)
        subtitle = text("the chain rule, run backward through the network", 30, AMBER, weight=m.SEMIBOLD)
        hero = m.VGroup(title, subtitle, cite).arrange(m.DOWN, buff=0.4).move_to(0.2 * m.UP)

        self.say(
            "[A neural network learns by gradient descent]: [every weight takes a small step downhill on the loss.]",
            m.AnimationGroup(m.FadeIn(edges), m.LaggedStart(*[m.FadeIn(layer, shift=0.2 * m.RIGHT) for layer in layers], lag_ratio=0.15)),
            wave(GRAD_COLOR, 2.2),
        )
        self.say(
            "[For that, each weight needs its own gradient]: [how much the loss changes when that one weight changes.]",
            m.Succession(net.animate(run_time=0.8).shift(-centre), m.AnimationGroup(m.Create(hot), m.FadeIn(ask, shift=0.2 * m.LEFT))),
            m.FadeIn(ask_note, shift=0.15 * m.UP),
        )
        self.say(
            "[Real networks have millions of weights.] [So how do we get every gradient at once?]",
            m.AnimationGroup(m.FadeIn(many, scale=1.2), wave(WEIGHT_COLOR, 2.0)),
            m.FadeIn(all_at_once, shift=0.2 * m.UP),
        )
        self.say(
            "[The answer is backpropagation]: [the chain rule from calculus, run backward through the network.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (edges, *layers, hot, ask, ask_note, many, all_at_once)],
                m.FadeIn(halo, scale=0.6), m.FadeIn(title, shift=0.3 * m.UP), m.FadeIn(cite),
            ),
            m.Write(subtitle),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the tiny network

    def network(self):
        self.next_section(
            "A tiny network",
            notes="Illustrative weights (said 'for example'): w1 = 2, w2 = 1 into two ReLU units, v1 = 1, v2 = −1 out; input 2, target 3, squared loss.",
        )
        self.build_graph()
        n, nm, e = self.nodes, self.names, self.edges

        def show(*keys):
            return m.LaggedStart(*[m.FadeIn(m.VGroup(n[k], nm[k]), scale=0.6) for k in keys], lag_ratio=0.15)

        inner = [e[k] for k in EDGES if k != ("y", "L")]
        self.say(
            "[Let's work it out by hand, on a tiny network]: [one input], [two hidden ReLU units], [and one output.]",
            self.new_header("THE NETWORK", "A tiny network, by hand"),
            show("x"),
            show("z1", "h1", "z2", "h2"),
            show("y"),
        )
        self.say(
            "[We draw it as a computational graph]: [each arrow is one simple step, like multiplying by a weight.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in inner], lag_ratio=0.12), m.FadeIn(self.relu_labels)),
            m.AnimationGroup(m.Indicate(e[("x", "z1")], color=WEIGHT_COLOR, scale_factor=1.06), flow(e[("x", "z1")], WEIGHT_COLOR)),
        )
        self.say(
            "It has four weights. [For example, two and one going in], [and one and minus one coming out.]",
            m.LaggedStart(m.FadeIn(self.weights["w1"], shift=0.15 * m.UP), m.FadeIn(self.weights["w2"], shift=0.15 * m.DOWN), lag_ratio=0.3),
            m.LaggedStart(m.FadeIn(self.weights["v1"], shift=0.15 * m.UP), m.FadeIn(self.weights["v2"], shift=0.15 * m.DOWN), lag_ratio=0.3),
        )
        self.say(
            "[At the end, a loss compares the output with the target, three.]",
            m.AnimationGroup(
                m.GrowArrow(e[("y", "L")]), m.FadeIn(m.VGroup(n["L"], nm["L"]), scale=0.6),
                m.FadeIn(self.loss_label), m.FadeIn(self.target, shift=0.15 * m.UP),
            ),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ forward pass

    def forward_pass(self):
        self.next_section("Forward pass", notes=f"z = ({tnum(FWD['z1'])}, {tnum(FWD['z2'])}), h = ({tnum(FWD['h1'])}, {tnum(FWD['h2'])}), y = {tnum(FWD['y'])}, L = {tnum(FWD['L'])}.")
        e = self.edges
        v = {k: self.value(k, FWD[k]) for k in POS}
        self.values = v
        z1 = tex(rf"z_1 = w_1\,x = {tnum(START['w1'])} \times {tnum(X_IN)} = {tnum(FWD['z1'])}", 36)
        z2 = tex(rf"z_2 = w_2\,x = {tnum(START['w2'])} \times {tnum(X_IN)} = {tnum(FWD['z2'])}", 36)
        relu_rule = tex(r"h = \mathrm{ReLU}(z) = \max(0,\ z)", 36)
        out = tex(rf"y = v_1 h_1 + v_2 h_2 = {tnum(START['v1'])} \times {tnum(FWD['h1'])} + ({tnum(START['v2'])}) \times {tnum(FWD['h2'])} = {tnum(FWD['y'])}", 36)
        loss = tex(rf"L = (y - t)^2 = ({tnum(FWD['y'])} - {tnum(TARGET)})^2 = {tnum(FWD['L'])}", 36)

        self.say(
            "[First, the forward pass, left to right. Our input is two.] [Times two gives four on top], [and times one gives two below.]",
            m.AnimationGroup(self.new_header("FORWARD PASS", "Compute every value, left to right"), m.FadeIn(v["x"], scale=0.5)),
            m.AnimationGroup(flow(e[("x", "z1")], SKY), m.FadeIn(v["z1"], scale=0.5), self.work(z1)),
            m.AnimationGroup(flow(e[("x", "z2")], SKY), m.FadeIn(v["z2"], scale=0.5), self.work(z2)),
        )
        self.say(
            "[ReLU keeps positive numbers as they are], [so the hidden units pass on four and two.]",
            m.AnimationGroup(self.work(relu_rule), m.Indicate(self.relu_labels, color=SKY)),
            m.AnimationGroup(flow(e[("z1", "h1")], SKY), flow(e[("z2", "h2")], SKY), m.FadeIn(v["h1"], scale=0.5), m.FadeIn(v["h2"], scale=0.5)),
        )
        self.say(
            "[The output is four times one, plus two times minus one]: [that's two.]",
            m.AnimationGroup(self.work(out), flow(e[("h1", "y")], SKY), flow(e[("h2", "y")], SKY)),
            m.FadeIn(v["y"], scale=0.5),
        )
        self.say(
            "[But the target is three.] [The loss is the error squared]: [two minus three, squared, is one.]",
            m.Indicate(self.target, color=GOOD, scale_factor=1.12),
            m.AnimationGroup(self.work(loss), flow(e[("y", "L")], SKY)),
            m.AnimationGroup(m.FadeIn(v["L"], scale=0.5), m.Indicate(self.nodes["L"], color=BAD, scale_factor=1.12)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the chain rule

    def chain_rule(self):
        self.next_section("The chain rule", notes="A chain w → y → L with illustrative local slopes 2 and 3: nudges multiply, 2 × 3 = 6.")
        box_y = 0.55
        boxes = m.VGroup()
        for x, sym, color, fill in ((-4.6, "w", WEIGHT_COLOR, VIOLET_FILL), (0.0, "y", SKY, SKY_FILL), (4.6, "L", BAD, BAD_FILL)):
            frame = card(1.7, 1.15, color, fill).move_to([x, box_y, 0])
            boxes.add(m.VGroup(frame, tex(sym, 52, color).move_to(frame)))
        links = m.VGroup(
            arrow(boxes[0].get_right(), boxes[1].get_left(), MUTED, buff=0.15),
            arrow(boxes[1].get_right(), boxes[2].get_left(), MUTED, buff=0.15),
        )
        times2 = text(f"×{DY_DW}", 32, GRAD_COLOR, weight=m.BOLD).next_to(links[0], m.UP, buff=0.18)
        times3 = text(f"×{DL_DY}", 32, GRAD_COLOR, weight=m.BOLD).next_to(links[1], m.UP, buff=0.18)
        nudge = 0.32
        bars, captions = m.VGroup(), m.VGroup()
        for b, factor, words in zip(boxes, (1, DY_DW, DY_DW * DL_DY), ("nudge", f"{DY_DW} × nudge", f"{DY_DW * DL_DY} × nudge")):
            bar = m.Rectangle(width=nudge * factor, height=0.2, stroke_width=0).set_fill(GRAD_COLOR, opacity=0.9)
            bar.move_to([b.get_left()[0], box_y - 1.0, 0], aligned_edge=m.LEFT)
            bars.add(bar)
            captions.add(text(words, 22, MUTED).next_to(bar, m.DOWN, buff=0.18).align_to(bar, m.LEFT))
        arc = m.CurvedArrow(boxes[0].get_top() + 0.12 * m.UP, boxes[2].get_top() + 0.12 * m.UP, angle=-0.9)
        arc.set_stroke(GRAD_COLOR, width=4).set_color(GRAD_COLOR)
        times6 = text(f"×{DY_DW * DL_DY}", 34, GRAD_COLOR, weight=m.BOLD).move_to([0, arc.get_top()[1] + 0.3, 0])

        formula = m.MathTex(
            r"\frac{\partial L}{\partial w}", "=", r"\frac{\partial L}{\partial y}", r"\cdot", r"\frac{\partial y}{\partial w}",
            font_size=56, color=INK,
        ).move_to([0, -2.05, 0])
        formula[2].set_color(GRAD_COLOR)
        formula[4].set_color(GRAD_COLOR)
        row_y = formula.get_bottom()[1] - 0.42
        numbers = m.MathTex(str(DY_DW * DL_DY), "=", str(DL_DY), r"\times", str(DY_DW), font_size=38, color=INK)
        for i, color in enumerate((INK, MUTED, GRAD_COLOR, MUTED, GRAD_COLOR)):  # each number under its term
            numbers[i].set_color(color).set_x(formula[i].get_x())
        numbers.set_y(row_y)
        local = text("local gradients", 24, GRAD_COLOR, weight=m.SEMIBOLD).next_to(formula, m.RIGHT, buff=0.6)

        stage = [*self.graph_parts(), self.work_mob]
        self.work_mob = None
        self.say(
            "[How does one weight affect that loss?] [Let's zoom in on a single chain.]",
            m.AnimationGroup(self.new_header("THE CHAIN RULE", "Nudges multiply along a chain"), *[m.FadeOut(mob) for mob in stage]),
            m.LaggedStart(*[m.FadeIn(b, scale=0.8) for b in boxes], m.GrowArrow(links[0]), m.GrowArrow(links[1]), lag_ratio=0.15),
        )
        self.say(
            "[A weight w changes the output y], [and the output changes the loss L.]",
            m.AnimationGroup(flow(links[0], SKY), m.Indicate(boxes[1], color=SKY, scale_factor=1.08)),
            m.AnimationGroup(flow(links[1], SKY), m.Indicate(boxes[2], color=BAD, scale_factor=1.08)),
        )
        self.say(
            "[Say a tiny nudge to w moves y twice as much], [and a nudge to y moves the loss three times as much.]",
            m.Succession(
                m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(captions[0])),
                m.AnimationGroup(m.FadeIn(times2, shift=0.15 * m.UP), m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(captions[1])),
            ),
            m.AnimationGroup(m.FadeIn(times3, shift=0.15 * m.UP), m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(captions[2])),
        )
        self.say(
            "[Then a nudge to w moves the loss two times three]: [six times as much.]",
            m.Create(arc),
            m.AnimationGroup(m.FadeIn(times6, scale=1.3), m.Indicate(captions[2], color=GRAD_COLOR, scale_factor=1.15)),
        )
        self.say(
            "[That's the chain rule.] [Written with gradients, it looks like this.]",
            m.Indicate(m.VGroup(times2, times3, times6), color=INK, scale_factor=1.15),
            m.Write(formula),
        )
        self.say(
            "[The gradient of the loss with respect to w] [equals its gradient with respect to the output], [times the output's gradient with respect to w.]",
            m.AnimationGroup(m.Indicate(formula[0], color=INK, scale_factor=1.1), m.TransformFromCopy(times6, numbers[0])),
            m.AnimationGroup(m.Indicate(formula[2], color=GRAD_COLOR, scale_factor=1.1), m.FadeIn(numbers[1]), m.TransformFromCopy(times3, numbers[2])),
            m.AnimationGroup(m.Indicate(formula[4], color=GRAD_COLOR, scale_factor=1.1), m.FadeIn(numbers[3]), m.TransformFromCopy(times2, numbers[4])),
        )
        self.say(
            "[Each factor is a local gradient]: [it only looks at one step of the chain.]",
            m.AnimationGroup(m.Circumscribe(formula[2], color=GRAD_COLOR, buff=0.08), m.Circumscribe(formula[4], color=GRAD_COLOR, buff=0.08), m.FadeIn(local, shift=0.2 * m.LEFT)),
            m.AnimationGroup(flow(links[0], GRAD_COLOR), flow(links[1], GRAD_COLOR), m.Indicate(times2, color=GRAD_COLOR), m.Indicate(times3, color=GRAD_COLOR)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ backward pass

    def backward_pass(self):
        self.next_section("Backward pass", notes="Gradients right to left: dL/dy = −2; v: −8, −4; h: −2, +2; through ReLU (on); w: −4, +4. Checked against finite differences.")
        e, g = self.edges, GRAD
        dy = tex(rf"\frac{{\partial L}}{{\partial y}} = 2\,(y - t) = 2 \times ({tnum(FWD['y'] - TARGET)}) = {tnum(g['y'])}", 36)
        sign = text("negative: raise y a little, and the loss goes down", 26, INK, weight=m.SEMIBOLD, t2c={"negative": GRAD_COLOR})
        local_v = tex(rf"\frac{{\partial y}}{{\partial v_1}} = h_1 = {tnum(FWD['h1'])} \qquad \frac{{\partial y}}{{\partial v_2}} = h_2 = {tnum(FWD['h2'])}", 36)
        dv1 = tex(rf"\frac{{\partial L}}{{\partial v_1}} = \frac{{\partial L}}{{\partial y}} \cdot \frac{{\partial y}}{{\partial v_1}} = {tnum(g['y'])} \times {tnum(FWD['h1'])} = {tnum(g['v1'])}", 36)
        dv2 = tex(rf"\frac{{\partial L}}{{\partial v_2}} = \frac{{\partial L}}{{\partial y}} \cdot \frac{{\partial y}}{{\partial v_2}} = {tnum(g['y'])} \times {tnum(FWD['h2'])} = {tnum(g['v2'])}", 36)
        dh1 = tex(rf"\frac{{\partial L}}{{\partial h_1}} = \frac{{\partial L}}{{\partial y}} \cdot v_1 = {tnum(g['y'])} \times {tnum(START['v1'])} = {tnum(g['h1'])}", 36)
        dh2 = tex(rf"\frac{{\partial L}}{{\partial h_2}} = \frac{{\partial L}}{{\partial y}} \cdot v_2 = {tnum(g['y'])} \times ({tnum(START['v2'])}) = {tnum(g['h2'])}", 36)
        relu_local = tex(r"\frac{\partial h}{\partial z} = 1 \ \text{if } z > 0, \ \text{else } 0", 36)
        local_w = tex(rf"\frac{{\partial z_1}}{{\partial w_1}} = \frac{{\partial z_2}}{{\partial w_2}} = x = {tnum(X_IN)}", 36)
        dw1 = tex(rf"\frac{{\partial L}}{{\partial w_1}} = \frac{{\partial L}}{{\partial z_1}} \cdot x = {tnum(g['z1'])} \times {tnum(X_IN)} = {tnum(g['w1'])}", 36)
        dw2 = tex(rf"\frac{{\partial L}}{{\partial w_2}} = \frac{{\partial L}}{{\partial z_2}} \cdot x = {tnum(g['z2'])} \times {tnum(X_IN)} = {tnum(g['w2'])}", 36)
        product = tex(
            rf"\frac{{\partial L}}{{\partial w_1}} = \frac{{\partial L}}{{\partial y}} \, \frac{{\partial y}}{{\partial h_1}} \, \frac{{\partial h_1}}{{\partial z_1}} \, \frac{{\partial z_1}}{{\partial w_1}}"
            rf" = ({tnum(g['y'])})({tnum(START['v1'])})(1)({tnum(X_IN)}) = {tnum(g['w1'])}",
            36,
        )
        assert g["y"] * START["v1"] * 1 * X_IN == g["w1"]

        ch = self.node_chips
        wc = self.weight_chips
        ch["y"] = self.node_chip("y", g["y"])
        wc["v1"], wc["v2"] = self.weight_chip("v1", g["v1"]), self.weight_chip("v2", g["v2"])
        ch["h1"], ch["h2"] = self.node_chip("h1", g["h1"]), self.node_chip("h2", g["h2"])
        ch["z1"], ch["z2"] = self.node_chip("z1", g["z1"]), self.node_chip("z2", g["z2"])
        wc["w1"], wc["w2"] = self.weight_chip("w1", g["w1"]), self.weight_chip("w2", g["w2"])
        stage = [mob for mob in self.graph_parts() if mob not in ch.values() and mob not in wc.values()]
        path = [e[("y", "L")], e[("h1", "y")], e[("z1", "h1")], e[("x", "z1")]]
        key_box = card(0.62, 0.38, GRAD_COLOR, AMBER_FILL, radius=0.19, stroke=2)
        legend = m.VGroup(key_box, text("gradient of the loss", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        legend.move_to([6.6, 2.35, 0], aligned_edge=m.RIGHT)
        self.legend = legend

        self.say(
            "[Now the backward pass.] [We start at the loss, and walk right to left.]",
            m.AnimationGroup(self.new_header("BACKWARD PASS", "Local gradients, right to left"), *[m.FadeIn(mob) for mob in stage]),
            m.AnimationGroup(m.Indicate(self.nodes["L"], color=BAD, scale_factor=1.15), flow(e[("y", "L")], GRAD_COLOR, backward=True)),
        )
        self.say(
            "[The loss is the error squared, so its gradient with respect to the output is two times the error]: [minus two.]",
            self.work(dy),
            m.AnimationGroup(m.FadeIn(ch["y"], scale=1.3), flow(e[("y", "L")], GRAD_COLOR, backward=True), m.FadeIn(legend, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[Negative means: raise the output a little, and the loss goes down.]",
            m.AnimationGroup(self.work(sign), m.Indicate(ch["y"], color=GRAD_COLOR, scale_factor=1.2)),
        )
        self.say(
            "[For an output weight, the local gradient is the hidden value it multiplies.]",
            m.AnimationGroup(self.work(local_v), m.Indicate(self.values["h1"], color=SKY, scale_factor=1.3), m.Indicate(self.values["h2"], color=SKY, scale_factor=1.3)),
        )
        self.say(
            "[So the top output weight gets minus two times four: minus eight.] [The bottom one gets minus two times two: minus four.]",
            m.AnimationGroup(self.work(dv1), flow(e[("h1", "y")], GRAD_COLOR, backward=True), m.FadeIn(wc["v1"], scale=1.3)),
            m.AnimationGroup(self.work(dv2), flow(e[("h2", "y")], GRAD_COLOR, backward=True), m.FadeIn(wc["v2"], scale=1.3)),
        )
        self.say(
            "[To reach the hidden units, we multiply by the output weights instead]: [minus two times one is minus two], [and minus two times minus one is plus two.]",
            m.AnimationGroup(m.Indicate(self.weights["v1"], color=WEIGHT_COLOR, scale_factor=1.15), m.Indicate(self.weights["v2"], color=WEIGHT_COLOR, scale_factor=1.15)),
            m.AnimationGroup(self.work(dh1), flow(e[("h1", "y")], GRAD_COLOR, backward=True), m.FadeIn(ch["h1"], scale=1.3)),
            m.AnimationGroup(self.work(dh2), flow(e[("h2", "y")], GRAD_COLOR, backward=True), m.FadeIn(ch["h2"], scale=1.3)),
        )
        self.say(
            "[ReLU's local gradient is one when its input is positive], [so both gradients pass straight through.]",
            m.AnimationGroup(self.work(relu_local), m.Indicate(self.relu_labels, color=GRAD_COLOR)),
            m.AnimationGroup(
                flow(e[("z1", "h1")], GRAD_COLOR, backward=True), flow(e[("z2", "h2")], GRAD_COLOR, backward=True),
                m.TransformFromCopy(ch["h1"], ch["z1"]), m.TransformFromCopy(ch["h2"], ch["z2"]),
            ),
        )
        self.say(
            "[Last, the weights going in. Their local gradient is the input, two.]",
            m.AnimationGroup(self.work(local_w), m.Indicate(self.values["x"], color=SKY, scale_factor=1.3)),
        )
        self.say(
            "[So the top weight gets minus two times two: minus four.] [The bottom weight gets two times two: plus four.]",
            m.AnimationGroup(self.work(dw1), flow(e[("x", "z1")], GRAD_COLOR, backward=True), m.FadeIn(wc["w1"], scale=1.3)),
            m.AnimationGroup(self.work(dw2), flow(e[("x", "z2")], GRAD_COLOR, backward=True), m.FadeIn(wc["w2"], scale=1.3)),
        )
        self.say(
            "[Every gradient is just the local gradients along its path, multiplied together.]",
            m.AnimationGroup(self.work(product), m.Succession(*[flow(p, GRAD_COLOR, backward=True, width=10) for p in path], run_time=2.4), m.Indicate(wc["w1"], color=GRAD_COLOR, scale_factor=1.2)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ where paths meet

    def paths_meet(self):
        self.next_section("Where paths meet", notes="The input feeds both hidden units: dL/dx = (−2)(2) + (2)(1) = −2. In a deeper net this sum flows on to earlier layers.")
        e, g = self.edges, GRAD
        top_part = grad_chip(g["x_top"]).move_to([POS["x"][0], POS["x"][1] + R + 0.36, 0])
        bottom_part = grad_chip(g["x_bottom"]).move_to([POS["x"][0], POS["x"][1] - R - 0.36, 0])
        self.node_chips["x"] = self.node_chip("x", g["x"])
        dx = tex(
            rf"\frac{{\partial L}}{{\partial x}} = \frac{{\partial L}}{{\partial z_1}}\, w_1 + \frac{{\partial L}}{{\partial z_2}}\, w_2"
            rf" = ({tnum(g['z1'])})({tnum(START['w1'])}) + ({tnum(g['z2'])})({tnum(START['w2'])}) = {tnum(g['x'])}",
            36,
        )
        deeper = text("in a deeper network, this sum keeps flowing back to earlier layers", 26, INK, weight=m.SEMIBOLD, t2c={"this sum": GRAD_COLOR})

        self.say(
            "[What about the input itself?] [It feeds both hidden units, so two paths lead back to it.]",
            m.AnimationGroup(self.new_header("WHERE PATHS MEET", "Gradients add up"), m.Indicate(self.nodes["x"], color=SKY, scale_factor=1.2)),
            m.AnimationGroup(flow(e[("x", "z1")], GRAD_COLOR, backward=True, width=10), flow(e[("x", "z2")], GRAD_COLOR, backward=True, width=10)),
        )
        self.say(
            "[The top path brings minus two times two: minus four.] [The bottom path brings two times one: plus two.]",
            m.AnimationGroup(m.TransformFromCopy(self.node_chips["z1"], top_part, path_arc=0.4), m.Indicate(self.weights["w1"], color=WEIGHT_COLOR, scale_factor=1.12)),
            m.AnimationGroup(m.TransformFromCopy(self.node_chips["z2"], bottom_part, path_arc=-0.4), m.Indicate(self.weights["w2"], color=WEIGHT_COLOR, scale_factor=1.12)),
        )
        self.say(
            "[Where paths meet, the gradients add]: [minus four plus two, minus two.]",
            self.work(dx),
            m.AnimationGroup(m.ReplacementTransform(m.VGroup(top_part, bottom_part), self.node_chips["x"]), m.Flash(self.nodes["x"].get_center(), color=GRAD_COLOR, flash_radius=0.7)),
        )
        self.say(
            "[In a deeper network, this input would be an earlier layer], [and this sum would keep flowing backward.]",
            m.Indicate(self.nodes["x"], color=SKY, scale_factor=1.2),
            m.AnimationGroup(self.work(deeper), m.Indicate(self.node_chips["x"], color=GRAD_COLOR, scale_factor=1.2)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ one step of gradient descent

    def one_step(self):
        self.next_section(
            "One step",
            notes=f"Learning rate {LR} (illustrative). New weights {', '.join(f'{k} = {NEW[k]:.2f}' for k in NEW)}; y {FWD['y']:.0f} → {FWD2['y']:.4f}; "
            f"loss {FWD['L']:.0f} → {FWD2['L']:.4f}. Repeating the step 200 times brings the loss to {LOSS_AFTER_200:.1e}.",
        )
        e = self.edges
        rule = m.MathTex(r"w", r"\leftarrow", r"w", "-", r"\eta", r"\frac{\partial L}{\partial w}", font_size=40, color=INK)
        rule[4].set_color(GOOD)
        rule[5].set_color(GRAD_COLOR)
        eta = text(f"η = {LR}, for example", 22, MUTED)
        new_w = {k: self.weight_label(k, NEW[k], GOOD, like=self.weights[k]) for k in NEW}
        new_v = {k: self.value(k, FWD2[k], GOOD if k in ("y", "L") else INK) for k in POS}
        assert num(FWD2["y"]) == "2.56" and num(FWD2["L"]) == "0.19"
        ups = [k for k in NEW if GRAD[k] < 0]
        downs = [k for k in NEW if GRAD[k] > 0]
        loop = text("forward  →  backward  →  update,  again and again", 26, INK, weight=m.SEMIBOLD, t2c={"forward": SKY, "backward": GRAD_COLOR, "update": GOOD})

        def change(keys):
            return m.AnimationGroup(
                *[m.ReplacementTransform(self.weights[k], new_w[k]) for k in keys],
                *[m.FadeOut(self.weight_chips[k], shift=0.2 * (m.UP if WEIGHT_SIDE[k] > 0 else m.DOWN)) for k in keys],
            )

        def run(keys, color=SKY):
            return m.AnimationGroup(*[flow(e[k], color) for k in keys])

        self.say(
            "[Now every weight has its gradient.] [Time for one step of gradient descent.]",
            m.AnimationGroup(self.new_header("ONE STEP", "Update every weight at once"), m.LaggedStart(*[m.Indicate(c, color=GRAD_COLOR, scale_factor=1.2) for c in self.weight_chips.values()], lag_ratio=0.2)),
            self.work(rule),
        )
        eta.next_to(rule, m.RIGHT, buff=0.6)
        self.say(
            "[Each weight moves against its gradient]: [new weight equals old weight, minus a small learning rate times the gradient.]",
            m.Indicate(rule[3], color=GRAD_COLOR, scale_factor=1.6),
            m.AnimationGroup(m.Indicate(m.VGroup(rule[4], rule[5]), color=GOOD, scale_factor=1.12), m.FadeIn(eta, shift=0.15 * m.LEFT)),
        )
        moves = [m.AnimationGroup(change(ups), *[m.FadeOut(c) for c in self.node_chips.values()]), m.AnimationGroup(change(downs), m.FadeOut(self.legend))]
        self.weights.update(new_w)
        self.weight_chips, self.node_chips = {}, {}
        self.say("[Negative gradients push weights up], [positive ones push them down.]", *moves)
        self.say(
            "[Run the forward pass again]: [the output climbs from two toward three], [and the loss drops from one to under a fifth.]",
            m.AnimationGroup(
                m.Succession(run([("x", "z1"), ("x", "z2")]), run([("z1", "h1"), ("z2", "h2")]), run([("h1", "y"), ("h2", "y")])),
                *[m.ReplacementTransform(self.values[k], new_v[k]) for k in ("z1", "z2", "h1", "h2")],
            ),
            m.AnimationGroup(m.ReplacementTransform(self.values["y"], new_v["y"]), m.Indicate(self.nodes["y"], color=GOOD, scale_factor=1.15)),
            m.AnimationGroup(run([("y", "L")]), m.ReplacementTransform(self.values["L"], new_v["L"]), m.Flash(self.nodes["L"].get_center(), color=GOOD, flash_radius=0.75)),
        )
        self.values.update({k: new_v[k] for k in ("z1", "z2", "h1", "h2", "y", "L")})
        self.say(
            "[Forward, backward, one small step.] [Repeat that thousands of times, and the network learns.]",
            m.Succession(
                run([("x", "z1"), ("x", "z2"), ("z1", "h1"), ("z2", "h2"), ("h1", "y"), ("h2", "y"), ("y", "L")]),
                m.AnimationGroup(*[flow(e[k], GRAD_COLOR, backward=True) for k in EDGES]),
                m.AnimationGroup(*[m.Indicate(self.weights[k], color=GOOD, scale_factor=1.12) for k in NEW]),
            ),
            m.AnimationGroup(self.work(loop), m.FadeOut(eta)),
        )
        self.wait(0.3)
        self.work_mob = None
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section(
            "The big picture",
            notes="Backward ≈ 2× forward FLOPs (forward ≈ 2N, backward ≈ 4N per token; Kaplan et al., 2020, 'Scaling Laws for Neural Language Models'). "
            "Automatic differentiation: PyTorch autograd (loss.backward()), JAX (jax.grad). Checked 2026-10-21.",
        )
        slabs = m.VGroup(*[card(0.62, 3.0, SKY, SKY_FILL, radius=0.15, stroke=2) for _ in range(6)]).arrange(m.RIGHT, buff=0.42).move_to([-3.55, 0.0, 0])
        joins = m.VGroup(*[arrow(a.get_right(), b.get_left(), FAINT, buff=0.06) for a, b in zip(slabs, slabs[1:])])
        neurons = m.VGroup(*[
            m.Circle(radius=0.12).set_stroke(SKY, width=2.5).set_fill(BG, opacity=1).move_to([s.get_x(), y, 0])
            for s in slabs for y in np.linspace(1.05, -1.05, 5)
        ]).set_z_index(1)
        top_y, bottom_y = slabs.get_top()[1] + 0.45, slabs.get_bottom()[1] - 0.45
        fwd = arrow([slabs.get_left()[0], top_y, 0], [slabs.get_right()[0], top_y, 0], SKY, buff=0)
        bwd = arrow([slabs.get_right()[0], bottom_y, 0], [slabs.get_left()[0], bottom_y, 0], GRAD_COLOR, buff=0)
        fwd_label = text("forward: values", 22, SKY, weight=m.SEMIBOLD).next_to(fwd, m.UP, buff=0.12)
        bwd_label = text("backward: gradients", 22, GRAD_COLOR, weight=m.SEMIBOLD).next_to(bwd, m.DOWN, buff=0.12)
        sweep = m.LaggedStart(*[s.animate(rate_func=m.there_and_back).set_fill(AMBER_FILL).set_stroke(GRAD_COLOR) for s in reversed(slabs)], lag_ratio=0.25, run_time=2.2)

        cost_title = text("cost of one training step", 24, INK, weight=m.BOLD).move_to([3.7, 1.9, 0])
        unit = 1.55
        cost_rows = m.VGroup()
        for name, factor, color, label in (("forward", 1, SKY, "1×"), ("backward", 2, GRAD_COLOR, "≈ 2×")):
            words = text(name, 22, MUTED)
            bar = m.Rectangle(width=unit * factor, height=0.36, stroke_width=0).set_fill(color, opacity=0.9)
            tag = text(label, 22, color, weight=m.BOLD)
            cost_rows.add(m.VGroup(words, bar, tag))
        for row, y in zip(cost_rows, (1.2, 0.55)):
            row[0].move_to([2.1, y, 0], aligned_edge=m.RIGHT)
            row[1].move_to([2.35, y, 0], aligned_edge=m.LEFT)
            row[2].next_to(row[1], m.RIGHT, buff=0.18)
        source = text("Kaplan et al., 2020", 20, MUTED).move_to([3.7, 0.0, 0])

        def code_card(lib, line, y):
            box = card(5.4, 0.82, EDGE, CARD, radius=0.18, stroke=2).move_to([3.7, y, 0])
            words = m.VGroup(text(lib, 22, INK, weight=m.BOLD), code(line, 22, GRAD_COLOR)).arrange(m.RIGHT, buff=0.35)
            words.move_to(box).align_to(box.get_left() + 0.3 * m.RIGHT, m.LEFT)
            return m.VGroup(box, words)

        torch = code_card("PyTorch", "loss.backward()", -1.0)
        jax = code_card("JAX", "jax.grad(loss_fn)", -2.0)
        autodiff = pill("automatic differentiation", GRAD_COLOR, AMBER_FILL, 24).move_to([3.7, -3.1, 0])

        self.say(
            "[Real networks run exactly this recipe], [with millions of weights instead of four.]",
            m.AnimationGroup(self.new_header("THE BIG PICTURE", "One backward sweep, every gradient"), m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.UP) for s in slabs], lag_ratio=0.12), m.FadeIn(joins), m.FadeIn(neurons)),
            m.AnimationGroup(m.GrowArrow(fwd), m.FadeIn(fwd_label), m.GrowArrow(bwd), m.FadeIn(bwd_label)),
        )
        self.say(
            "[Each gradient is computed once, and reused by every step behind it.]",
            m.AnimationGroup(sweep, flow(bwd, GRAD_COLOR, width=10)),
        )
        self.say(
            "[So the backward pass costs only about twice as much as the forward pass.]",
            m.AnimationGroup(
                m.FadeIn(cost_title),
                m.Succession(
                    m.AnimationGroup(m.FadeIn(cost_rows[0][0]), m.GrowFromEdge(cost_rows[0][1], m.LEFT), m.FadeIn(cost_rows[0][2])),
                    m.AnimationGroup(m.FadeIn(cost_rows[1][0]), m.GrowFromEdge(cost_rows[1][1], m.LEFT), m.FadeIn(cost_rows[1][2])),
                ),
                m.FadeIn(source),
            ),
        )
        self.say(
            "[You never do this by hand.] [Frameworks like PyTorch and JAX do it for you]: [it's called automatic differentiation.]",
            m.Indicate(slabs, color=INK, scale_factor=1.03),
            m.LaggedStart(m.FadeIn(torch, shift=0.2 * m.LEFT), m.FadeIn(jax, shift=0.2 * m.LEFT), lag_ratio=0.4),
            m.FadeIn(autodiff, scale=1.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Forward, backward, update.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Forward", "Backward", "Update"), (SKY, GRAD_COLOR, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Multiply along each path. Add where paths meet.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [forward], [backward], [update.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Multiply local gradients along each path, and add them where paths meet.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
