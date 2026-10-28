"""AI, Explained · Episode 21 · Recurrent Networks & LSTMs

Running example: predicting the next note of a short melody (invented for teaching, 16 notes in
C major that start and end on C). The film plays the first 15 notes and asks for the 16th.

A plain network sees a fixed window; a recurrent network reads one note at a time and carries a
hidden state, h_t = tanh(W·h_{t−1} + U·x_t + b), with the same weights at every step (unrolled
over time it is a deep network, one layer per note). Training through time multiplies one factor
per step, so the gradient vanishes (or explodes). The LSTM adds a cell state with an additive
path and three sigmoid gates (forget, input, output); a forget gate near one keeps the memory,
and its gradient, for many steps. Close: Transformers (Episode 24), state space models (Ep 40).

Genuine numbers (seeded numpy below, asserted against the narration):
- a 4-unit recurrent network with random weights (seed 11, W scaled to spectral norm 0.84)
  reads the 15 notes; backpropagating a unit-size gradient from step 15 to step 1 shrinks it by
  a factor of 0.26–0.57 per step (the three first: 0.42, 0.46, 0.57) to about 0.000011;
- 1.5^14 ≈ 292 (exploding, for example factors of one and a half);
- 0.99^14 ≈ 0.87 (an LSTM whose forget gate stays at 0.99, for example).
Illustrative (hand-set, said "for example"/"say"): the gate values over the melody, the guess
bars of the untrained readout (computed from a seeded readout matrix), the predicted final C.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the melody and the numbers

PITCHES = ["B", "C", "D", "E", "F", "G", "A"]  # low → high (B3 … A4)
MELODY = "C E G E F A G F E G F D E D B C".split()
SHOWN = 15  # the film plays 15 notes and asks for the 16th
assert len(MELODY) == 16 and MELODY[0] == MELODY[-1] == "C"

HID = 4
_rng = np.random.default_rng(11)
W_H = _rng.normal(0, 1, (HID, HID))
W_H *= 0.84 / np.linalg.norm(W_H, 2)
U_X = _rng.normal(0, 1, (HID, len(PITCHES)))
B_H = _rng.normal(0, 0.2, HID)


def one_hot(note):
    x = np.zeros(len(PITCHES))
    x[PITCHES.index(note)] = 1.0
    return x


STATES = []
_h = np.zeros(HID)
for _note in MELODY[:SHOWN]:
    _h = np.tanh(W_H @ _h + U_X @ one_hot(_note) + B_H)
    STATES.append(_h)

# backpropagation through time: a gradient of size one at step 15, carried back to step 1
_g = np.ones(HID) / 2
FACTORS = []
for _t in range(SHOWN - 1, 0, -1):
    _back = W_H.T @ ((1 - STATES[_t] ** 2) * _g)  # ∂h_t/∂h_{t−1} = diag(tanh′)·W
    FACTORS.append(float(np.linalg.norm(_back) / np.linalg.norm(_g)))
    _g = _back
FACTORS = np.array(FACTORS)  # FACTORS[0]: from note 15 back to note 14
REACH = np.concatenate([[1.0], np.cumprod(FACTORS)])  # REACH[k]: what is left k steps back
EXPLODE = 1.5**14
KEEP = 0.99**14

# readout (untrained, seeded): the network's guess for the third note after reading C, E
_V = np.random.default_rng(21).normal(0, 1.5, (len(PITCHES), HID))
_z = _V @ STATES[1]
GUESS = np.exp(_z - _z.max()) / np.exp(_z - _z.max()).sum()

# illustrative gate values for the melody section (hand-set, said "say ninety-nine percent")
_gr = np.random.default_rng(7)
INPUT_GATE = np.r_[0.95, _gr.uniform(0.02, 0.08, SHOWN - 1)]
FORGET_GATE = np.full(SHOWN - 1, 0.99)  # steps 2 … 15
OUTPUT_GATE = np.r_[_gr.uniform(0.04, 0.15, SHOWN - 1), 0.92]

# the narration's numbers
assert len(FACTORS) == 14  # "multiply all fourteen"
assert FACTORS.max() < 0.6 and 0.4 < FACTORS.mean() < 0.5  # "every factor is around a half, or less"
assert 0.5e-5 < REACH[-1] < 1.5e-5  # "about one part in a hundred thousand"
assert 280 < EXPLODE < 300  # "one and a half, fourteen times over, is almost three hundred"
assert round(100 * KEEP) == 87  # "still keeps eighty-seven percent"
assert np.all(np.abs(np.array(STATES)) < 1)  # tanh: "between minus one and one"


def fmt(p):
    """A gradient size with two significant figures, as plain decimals."""
    if p >= 0.01:
        return f"{p:.2f}"
    return f"{p:.{1 - int(np.floor(np.log10(p)))}f}"


def signed(v):
    return f"{v:+.2f}".replace("-", "−")


assert fmt(REACH[-1]) == "0.000011"

# accents for this film: notes SKY, hidden state VIOLET, cell state AMBER, gradient BAD,
# gates: forget PINK, input TEAL, output BLUE; results GOOD
BLUE_FILL = "#132140"


# ---------------------------------------------------------------------------- helpers


class Roll:
    """A piano roll: one lane per pitch, one column per note, the note's name in its block."""

    def __init__(self, notes, slots, col, row, center, labels=True, label_size=20):
        self.col, self.row = col, row
        n = len(PITCHES)
        width, height = slots * col, n * row
        cx, cy = center
        self.left, self.bottom = cx - width / 2, cy - height / 2
        self.top = self.bottom + height
        lane_w = len(notes) * col  # lanes under the played notes only; the slot after them stays clear
        self.lanes = m.VGroup(
            *[
                m.Rectangle(width=lane_w, height=row, stroke_width=0)
                .set_fill("#0F1628" if k % 2 == 0 else CARD, opacity=1)
                .move_to([self.left + lane_w / 2, self.bottom + (k + 0.5) * row, 0])
                for k in range(n)
            ]
        )
        self.keys = m.VGroup(
            *[text(p, 20, MUTED, weight=m.SEMIBOLD).move_to([self.left - 0.3, self.y(p), 0]) for p in PITCHES]
        )
        self.blocks = m.VGroup(*[self.block(i, p, SKY, SKY_FILL, labels, label_size) for i, p in enumerate(notes)])
        self.group = m.VGroup(self.lanes, self.keys, self.blocks)

    def x(self, i):
        return self.left + (i + 0.5) * self.col

    def y(self, pitch):
        return self.bottom + (PITCHES.index(pitch) + 0.5) * self.row

    def block(self, i, pitch, color, fill, labels=True, label_size=20):
        box = m.RoundedRectangle(corner_radius=0.07, width=self.col - 0.1, height=self.row - 0.07)
        box.set_stroke(color, width=2).set_fill(fill, opacity=1).move_to([self.x(i), self.y(pitch), 0])
        if not labels:
            return m.VGroup(box)
        return m.VGroup(box, text(pitch, label_size, INK, weight=m.BOLD).move_to(box))

    def slot(self, i):
        frame = m.RoundedRectangle(corner_radius=0.1, width=self.col - 0.06, height=self.top - self.bottom + 0.06)
        frame = m.DashedVMobject(frame.set_stroke(INK, width=2.5), num_dashes=30)
        frame.move_to([self.x(i), (self.top + self.bottom) / 2, 0])
        return frame


def state_column(values, center):
    """A hidden state: four numbers in violet boxes, brighter the bigger they are."""
    rows = m.VGroup()
    for v in values:
        box = m.RoundedRectangle(corner_radius=0.08, width=1.05, height=0.42)
        box.set_stroke(VIOLET, width=2).set_fill(VIOLET, opacity=0.08 + 0.5 * abs(v))
        rows.add(m.VGroup(box, code(signed(v), 20, INK).move_to(box)))
    return rows.arrange(m.DOWN, buff=0.08).move_to(center)


def note_chip(name, color=SKY, fill=SKY_FILL, size=0.62, font=22):
    return chip(name, color, fill, width=size, height=size * 0.9, size=font)


def link(a, b, color=MUTED, width=3.5):
    return m.Arrow(a, b, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.25)


def op_node(kind, center, color, radius=0.24):
    ring = m.Circle(radius=radius).set_stroke(color, width=3).set_fill(CARD, opacity=1).move_to(center)
    d = 0.11
    if kind == "x":
        marks = m.VGroup(m.Line([-d, -d, 0], [d, d, 0]), m.Line([-d, d, 0], [d, -d, 0]))
    else:
        d = 0.13
        marks = m.VGroup(m.Line([-d, 0, 0], [d, 0, 0]), m.Line([0, -d, 0], [0, d, 0]))
    marks.set_stroke(color, width=3.5).move_to(center)
    return m.VGroup(ring, marks).set_z_index(2)


def gate_box(symbol, color, fill, center, size=0.72):
    box = card(size, size, color, fill, radius=0.14, stroke=2.5).move_to(center)
    return m.VGroup(box, tex(symbol, 34, color).move_to(box)).set_z_index(2)


def cite(s):
    return text(s, 20, MUTED)


def ftex(*parts, size=46):
    """A formula with one part per string, to colour and indicate term by term."""
    return m.MathTex(*parts, font_size=size, color=INK)


# ---------------------------------------------------------------------------- the episode


class Ep21RNNsLSTMs(Episode):
    number = 21

    def lesson(self):
        self.hook()
        self.recurrent()
        self.unrolling()
        self.vanishing()
        self.cell_state()
        self.gates()
        self.melody_again()
        self.big_picture()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes="An invented 16-note melody in C (for example): starts on C, comes home to C. A fixed window of three notes cannot see how it began.",
        )
        roll = Roll(MELODY[:SHOWN], 16, 0.72, 0.44, (0.3, -0.15))
        slot = roll.slot(15)
        mark = text("?", 44, INK, weight=m.BOLD).move_to(slot)
        first = roll.blocks[0]
        rise = m.DashedLine(first.get_top() + 0.05 * m.UP, [roll.x(0), roll.top + 0.12, 0], dash_length=0.1).set_stroke(AMBER, width=2.5)
        arc = m.ArcBetweenPoints([roll.x(0), roll.top + 0.12, 0], [roll.x(15), roll.top + 0.12, 0], angle=-0.45)
        arc = m.DashedVMobject(arc.set_stroke(AMBER, width=2.5), num_dashes=40)
        began = text("remember how it began", 22, AMBER, weight=m.SEMIBOLD).move_to([(roll.x(0) + roll.x(15)) / 2, roll.top + 0.95, 0])
        window = m.DashedVMobject(
            m.RoundedRectangle(corner_radius=0.15, width=3 * roll.col + 0.12, height=roll.top - roll.bottom + 0.22).set_stroke(VIOLET, width=3),
            num_dashes=40,
        ).move_to([roll.x(13), (roll.top + roll.bottom) / 2, 0])
        window_label = text("a regular network: the last 3 notes", 22, VIOLET, weight=m.SEMIBOLD)
        window_label.next_to(window, m.UP, buff=0.15).align_to(window, m.RIGHT)
        need = pill("we need a network that reads one note at a time, and remembers", VIOLET, VIOLET_FILL, 22).move_to([0, -2.55, 0])

        self.say(
            "[Here's a short melody, one note at a time.]",
            m.AnimationGroup(
                self.new_header("THE TASK", "Predict the next note"),
                m.FadeIn(roll.lanes),
                m.FadeIn(roll.keys),
                m.LaggedStart(*[m.FadeIn(b, shift=0.15 * m.RIGHT) for b in roll.blocks], lag_ratio=0.15),
            ),
        )
        self.say("[What note comes next?]", m.AnimationGroup(m.Create(slot), m.Succession(m.FadeIn(mark, scale=1.4), m.Wiggle(mark))))
        self.say(
            "[To guess well, you have to remember the past]: [not just the last few notes], [but how the tune began.]",
            m.LaggedStart(*[m.Indicate(b, color=SKY, scale_factor=1.12) for b in roll.blocks], lag_ratio=0.06),
            m.Indicate(roll.blocks[12:], color=VIOLET, scale_factor=1.12),
            m.AnimationGroup(m.Circumscribe(first, color=AMBER, buff=0.06), m.Create(rise), m.Create(arc), m.FadeIn(began)),
        )
        self.say(
            "[A regular network, like the ones in episode three, takes a fixed number of inputs]: [say, the last three notes.]",
            m.FadeOut(m.VGroup(rise, arc, began)),
            m.AnimationGroup(m.Create(window), m.FadeIn(window_label, shift=0.15 * m.UP)),
        )
        self.say(
            "[Everything before that is simply out of sight.] [We need a network that reads one note at a time, and remembers.]",
            m.AnimationGroup(*[b.animate.set_opacity(0.25) for b in roll.blocks[:12]]),
            m.FadeIn(need, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the recurrent network

    def recurrent(self):
        self.next_section(
            "The recurrent network",
            notes="h_t = tanh(W·h_{t−1} + U·x_t + b) (Elman, 1990). A 4-number hidden state, seeded random weights; the note enters as a one-hot vector. The guess bars come from an untrained, seeded readout (illustrative).",
        )
        cell_box = card(2.3, 1.5, VIOLET, VIOLET_FILL).move_to([-0.4, 0, 0])
        cell = m.VGroup(cell_box, text("RNN cell", 26, INK, weight=m.BOLD).move_to(cell_box))
        old_pos, new_pos = np.array([-3.55, 0, 0]), np.array([2.75, 0, 0])
        zeros = state_column(np.zeros(HID), old_pos)
        h1 = state_column(STATES[0], new_pos)
        h2 = state_column(STATES[1], new_pos)
        old_label = m.VGroup(text("old state", 22, MUTED), tex("h_{t-1}", 32, VIOLET)).arrange(m.RIGHT, buff=0.15).next_to(zeros, m.UP, buff=0.22)
        new_label = m.VGroup(text("new state", 22, MUTED), tex("h_t", 32, VIOLET)).arrange(m.RIGHT, buff=0.15).next_to(h1, m.UP, buff=0.22)
        into = link([zeros.get_right()[0] + 0.12, 0, 0], [cell_box.get_left()[0] - 0.05, 0, 0])
        out = link([cell_box.get_right()[0] + 0.05, 0, 0], [h1.get_left()[0] - 0.12, 0, 0])

        row = m.VGroup(*[note_chip(n, size=0.5, font=20) for n in MELODY[:SHOWN]]).arrange(m.RIGHT, buff=0.08).move_to([0, -3.2, 0])
        slot_at = np.array([-0.4, -1.75, 0])
        in_c = note_chip("C").move_to(slot_at)
        in_e = note_chip("E").move_to(slot_at)
        in_arrow = link(slot_at + 0.32 * m.UP, cell_box.get_bottom() + 0.05 * m.DOWN, SKY)
        in_label = m.VGroup(text("note", 22, MUTED), tex("x_t", 32, SKY)).arrange(m.RIGHT, buff=0.15).next_to(in_c, m.LEFT, buff=0.3)

        formula = ftex("h_t", "=", r"\tanh(", "W", "h_{t-1}", "+", "U", "x_t", "+", "b", ")", size=46)
        formula[0].set_color(VIOLET)
        formula[4].set_color(VIOLET)
        formula[7].set_color(SKY)
        formula.move_to([-0.4, 2.05, 0])
        source = cite("Elman, 1990").next_to(formula, m.RIGHT, buff=0.6)

        order = list(reversed(PITCHES))  # high at the top
        bars = m.VGroup()
        for k, p in enumerate(order):
            yy = 0.9 - 0.3 * k
            label = text(p, 20, MUTED, weight=m.SEMIBOLD).move_to([4.55, yy, 0])
            prob = GUESS[PITCHES.index(p)]
            bar = m.Rectangle(width=max(0.04, 1.7 * prob / GUESS.max()), height=0.2, stroke_width=0).set_fill(SKY, opacity=0.85)
            bar.move_to([4.8, yy, 0], aligned_edge=m.LEFT)
            bars.add(m.VGroup(label, bar))
        guess_title = text("next note?", 22, SKY, weight=m.SEMIBOLD).move_to([5.45, 1.32, 0])
        readout = link([h2.get_right()[0] + 0.12, 0, 0], [4.3, 0, 0])

        self.say(
            "[A recurrent network reads the melody one note at a time.]",
            m.AnimationGroup(
                self.new_header("THE RECURRENT NETWORK", "A memory, updated note by note"),
                m.FadeIn(cell, scale=0.9),
                m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in row], lag_ratio=0.05),
                m.FadeIn(zeros),
                m.FadeIn(old_label),
                m.GrowArrow(into),
            ),
        )
        self.say(
            "[With each note], [it updates its hidden state]: [a short list of numbers that sums up the tune so far.]",
            m.AnimationGroup(m.FadeIn(in_c, target_position=row[0]), m.GrowArrow(in_arrow), m.FadeIn(in_label)),
            m.AnimationGroup(m.GrowArrow(out), m.FadeIn(h1, shift=0.3 * m.RIGHT), m.FadeIn(new_label)),
            m.Indicate(h1, color=VIOLET, scale_factor=1.06),
        )
        self.say(
            "[That new state is carried to the next step], [where it meets the next note.]",
            m.AnimationGroup(m.FadeOut(zeros), h1.animate.move_to(old_pos), run_time=1.0),
            m.AnimationGroup(m.FadeOut(in_c), m.FadeIn(in_e, target_position=row[1]), m.FadeIn(h2, shift=0.3 * m.RIGHT)),
        )
        self.say(
            "[In math]: [the new state] [is a weight matrix times the old state], [plus a second weight matrix times the note], [plus a bias.]",
            m.AnimationGroup(m.Write(formula), m.FadeIn(source)),
            m.Indicate(formula[0], color=VIOLET, scale_factor=1.3),
            m.AnimationGroup(m.Indicate(formula[3:5], color=VIOLET, scale_factor=1.2), m.Indicate(h1, color=VIOLET, scale_factor=1.05)),
            m.AnimationGroup(m.Indicate(formula[6:8], color=SKY, scale_factor=1.2), m.Indicate(in_e, color=SKY, scale_factor=1.15)),
            m.Indicate(formula[9], color=INK, scale_factor=1.4),
        )
        self.say(
            "[A hyperbolic tangent then squashes every number between minus one and one.]",
            m.AnimationGroup(m.Indicate(m.VGroup(formula[2], formula[10]), color=VIOLET, scale_factor=1.2), m.Indicate(h2, color=VIOLET, scale_factor=1.06)),
        )
        self.say(
            "[And a readout layer turns the state into a guess for the next note.]",
            m.AnimationGroup(m.GrowArrow(readout), m.FadeIn(guess_title), m.LaggedStart(*[m.GrowFromEdge(b[1], m.LEFT) for b in bars], lag_ratio=0.08), m.FadeIn(m.VGroup(*[b[0] for b in bars]))),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ unrolling

    def unrolling(self):
        self.next_section(
            "Unrolling",
            notes="One cell with a loop = a chain of copies over time sharing W and U. Each step's target is simply the next note (self-supervised, as in Episode 8).",
        )
        box = card(2.1, 1.35, VIOLET, VIOLET_FILL).move_to([0, 0.1, 0])
        compact_label = text("RNN cell", 26, INK, weight=m.BOLD).move_to(box)
        loop = m.CurvedArrow(box.get_right() + np.array([0, 0.3, 0]), box.get_right() + np.array([0, -0.3, 0]), angle=-4.4)
        loop.set_stroke(VIOLET, width=4).set_color(VIOLET)
        loop_label = tex("h_t", 32, VIOLET).next_to(loop, m.RIGHT, buff=0.15)
        c_in = note_chip("x", SKY, SKY_FILL).move_to([0, -1.6, 0])
        c_in_label = tex("x_t", 30, SKY).move_to(c_in)
        c_in = m.VGroup(c_in[0], c_in_label)
        c_in_arrow = link(c_in.get_top() + 0.05 * m.UP, box.get_bottom() + 0.05 * m.DOWN, SKY)
        compact = m.VGroup(box, compact_label, loop, loop_label, c_in, c_in_arrow)

        xs = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
        y0 = 0.15
        cells = m.VGroup()
        for k, x in enumerate(xs):
            b = card(1.25, 0.95, VIOLET, VIOLET_FILL).move_to([x, y0, 0])
            cells.add(m.VGroup(b, tex(f"h_{k + 1}", 32, VIOLET).move_to(b)))
        hops = m.VGroup(*[link([a + 0.66, y0, 0], [b - 0.66, y0, 0], VIOLET) for a, b in zip(xs, xs[1:])])
        w_labels = m.VGroup(*[tex("W", 30, INK).move_to([(a + b) / 2, y0 + 0.33, 0]) for a, b in zip(xs, xs[1:])])
        notes = m.VGroup(*[note_chip(n).move_to([x, -1.55, 0]) for n, x in zip(MELODY, xs)])
        ups = m.VGroup(*[link([x, -1.55 + 0.3, 0], [x, y0 - 0.5, 0], SKY) for x in xs])
        u_labels = m.VGroup(*[tex("U", 30, INK).move_to([x - 0.3, -0.8, 0]) for x in xs])
        more = m.VGroup(*[m.Dot([6.2 + 0.17 * k, y0, 0], radius=0.045, color=MUTED) for k in range(3)])
        slots = m.VGroup(*[chip("?", EDGE, CARD, width=0.62, height=0.56, size=22).move_to([x, 1.6, 0]) for x in xs])
        outs = m.VGroup(*[link([x, y0 + 0.5, 0], [x, 1.6 - 0.3, 0], MUTED) for x in xs])
        answers = m.VGroup(*[note_chip(n, GOOD, GOOD_FILL).move_to([x, 1.6, 0]) for n, x in zip(MELODY[1:], xs)])
        sources = [n.get_center() for n in notes[1:]] + [more.get_center()]  # each answer is the next note
        shared = pill("one set of weights, shared by every step", VIOLET, VIOLET_FILL, 22).move_to([0, -3.3, 0])
        brace = m.Brace(notes, m.DOWN, buff=0.15).set_color(VIOLET)
        brace_label = text("a deep network: one layer per note", 22, VIOLET, weight=m.SEMIBOLD).next_to(brace, m.DOWN, buff=0.1)

        self.say(
            "[Drawn compactly, it's a single cell with a loop]: [its state feeds back in at the next step.]",
            m.AnimationGroup(self.new_header("UNROLLING", "The same cell, at every step"), m.FadeIn(box), m.FadeIn(compact_label), m.FadeIn(c_in), m.GrowArrow(c_in_arrow)),
            m.AnimationGroup(m.Create(loop), m.FadeIn(loop_label), m.ShowPassingFlash(loop.copy().set_stroke(INK, width=8), time_width=0.6, run_time=1.2)),
        )
        self.say(
            "[Unroll the loop over time], [and you get a chain of copies, one per note.]",
            m.AnimationGroup(*[m.TransformFromCopy(box, c[0]) for c in cells], m.FadeOut(compact), run_time=1.2),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(c[1]) for c in cells], lag_ratio=0.1),
                m.LaggedStart(*[m.GrowArrow(h) for h in hops], lag_ratio=0.1),
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(n, shift=0.2 * m.UP), m.GrowArrow(u)) for n, u in zip(notes, ups)], lag_ratio=0.1),
                m.FadeIn(more),
            ),
        )
        self.say(
            "[Every copy uses the very same weights], [so there's just one set to learn, however long the melody is.]",
            m.AnimationGroup(m.FadeIn(w_labels), m.FadeIn(u_labels), m.Indicate(w_labels, color=VIOLET, scale_factor=1.3), m.Indicate(u_labels, color=SKY, scale_factor=1.3)),
            m.FadeIn(shared, shift=0.2 * m.UP),
        )
        self.say(
            "[At every step, it guesses the next note], [and the melody itself supplies the right answer.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(o) for o in outs], lag_ratio=0.08), m.LaggedStart(*[m.FadeIn(s, scale=0.8) for s in slots], lag_ratio=0.08)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeOut(s), m.FadeIn(a, target_position=src)) for s, src, a in zip(slots, sources, answers)], lag_ratio=0.15),
        )
        self.say(
            "[Unrolled like this, it's a deep network, with one layer per note.]",
            m.AnimationGroup(m.FadeOut(shared), m.GrowFromCenter(brace), m.FadeIn(brace_label, shift=0.1 * m.DOWN)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ vanishing gradients

    def vanishing(self):
        self.next_section(
            "Vanishing gradients",
            notes=(
                "Computed: the seeded 4-unit network reads the 15 notes; a unit gradient at step 15 is carried back with "
                "W^T·diag(1 − h²). Per-step factors 0.26–0.57 (first three 0.42, 0.46, 0.57); after 14 steps 0.000011. "
                "Exploding: 1.5^14 ≈ 292 (for example). Sources: Hochreiter 1991 (diploma thesis); Bengio, Simard & Frasconi 1994; "
                "clipping: Pascanu, Mikolov & Bengio 2013."
            ),
        )
        xs = [-6.02 + 0.86 * j for j in range(SHOWN)]
        y0 = 1.45
        cells = m.VGroup()
        for j, x in enumerate(xs):
            b = m.RoundedRectangle(corner_radius=0.1, width=0.6, height=0.6).set_stroke(VIOLET, width=2.5).set_fill(VIOLET_FILL, opacity=1).move_to([x, y0, 0])
            cells.add(m.VGroup(b, text(MELODY[j], 20, INK, weight=m.BOLD).move_to(b)))
        wires = m.VGroup(*[m.Line([a + 0.3, y0, 0], [b - 0.3, y0, 0]).set_stroke(MUTED, width=3) for a, b in zip(xs, xs[1:])])
        base_y, tall = -1.75, 2.3
        baseline = m.Line([-6.45, base_y, 0], [6.45, base_y, 0]).set_stroke(FAINT, width=2)
        bars = []
        for j in range(SHOWN):
            k = SHOWN - 1 - j  # steps back from the last note
            bar = m.Rectangle(width=0.4, height=max(0.025, tall * REACH[k]), stroke_width=0).set_fill(BAD, opacity=0.9)
            bars.append(bar.move_to([xs[j], base_y, 0], aligned_edge=m.DOWN))
        axis_label = text("gradient size", 20, MUTED).move_to([-5.35, 0.75, 0])
        factor_labels = m.VGroup(
            *[code(f"×{FACTORS[k]:.2f}", 20, BAD).move_to([(xs[SHOWN - 1 - k] + xs[SHOWN - 2 - k]) / 2, 0.7, 0]) for k in range(3)]
        )
        readouts = []
        anchor = np.array([-6.3, -2.45, 0])
        for j in range(SHOWN):
            k = SHOWN - 1 - j
            r = m.VGroup(text(f"gradient reaching note {j + 1}:", 24, MUTED), code(fmt(REACH[k]), 30, BAD)).arrange(m.RIGHT, buff=0.25)
            readouts.append(r.move_to(anchor, aligned_edge=m.LEFT))
        source = cite("Hochreiter, 1991 · Bengio et al., 1994").move_to([4.0, 2.3, 0])
        explain = text("each factor: the weights × the slope of tanh", 22, MUTED).move_to([0, -3.3, 0])

        K = m.ValueTracker(SHOWN - 1.0)  # where the gradient is: 0 … 14 (cell index)

        def position():
            k = K.get_value()
            lo = int(np.floor(k))
            hi = min(lo + 1, SHOWN - 1)
            return np.array([xs[lo] + (xs[hi] - xs[lo]) * (k - lo), y0 - 0.42, 0])

        def make_dot():
            return m.Dot(position(), radius=0.1, color=BAD).set_z_index(5)

        def make_bars():
            reached = int(np.ceil(K.get_value() - 1e-6))
            return m.VGroup(*[bars[j].copy() for j in range(reached, SHOWN)])

        def make_readout():
            return readouts[int(np.ceil(K.get_value() - 1e-6))].copy()

        dot = m.always_redraw(make_dot)
        live_bars = m.always_redraw(make_bars)
        live_readout = m.always_redraw(make_readout)

        vanished = pill("vanished", BAD, BAD_FILL, 22).move_to([-3.4, -0.75, 0])
        boom = m.VGroup(text("exploding · factors of 1.5:", 24, BAD, weight=m.SEMIBOLD), tex(r"1.5^{14} \approx 292", 36, BAD)).arrange(m.RIGHT, buff=0.25)
        boom.move_to([3.6, -2.45, 0])
        clip = m.VGroup(pill("exploding: clip them", AMBER, AMBER_FILL, 22), cite("Pascanu et al., 2013")).arrange(m.RIGHT, buff=0.25)
        redesign = pill("vanishing: needs a new design", BAD, BAD_FILL, 22)
        fixes = m.VGroup(clip, redesign).arrange(m.RIGHT, buff=0.7).move_to([0, -3.3, 0])

        sweep = m.ShowPassingFlash(
            m.Line([xs[-1], y0, 0], [xs[0], y0, 0]).set_stroke(BAD, width=10), time_width=0.5, run_time=1.6
        )
        self.say(
            "[To train it, we use backpropagation from episode twelve], [running backwards through time.]",
            m.AnimationGroup(
                self.new_header("VANISHING GRADIENTS", "Why long memories are hard to learn", BAD, BAD_FILL),
                m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in cells], lag_ratio=0.04),
                m.FadeIn(wires),
                m.Create(baseline),
            ),
            m.AnimationGroup(sweep, cells[-1][0].animate.set_stroke(BAD)),
        )
        self.say(
            "[Each step back], [the chain rule multiplies the gradient by one more factor]: [the weights, times the slope of the hyperbolic tangent.]",
            m.AnimationGroup(m.FadeIn(dot), m.FadeIn(live_bars), m.FadeIn(live_readout), m.FadeIn(axis_label), m.FadeIn(source)),
            m.AnimationGroup(K.animate(run_time=1.0).set_value(SHOWN - 2.0), m.FadeIn(factor_labels[0], shift=0.1 * m.DOWN)),
            m.FadeIn(explain, shift=0.2 * m.UP),
        )
        self.say(
            "[Here, every factor is around a half, or less.]",
            m.Succession(
                m.AnimationGroup(K.animate(run_time=0.8).set_value(SHOWN - 3.0), m.FadeIn(factor_labels[1], shift=0.1 * m.DOWN)),
                m.AnimationGroup(K.animate(run_time=0.8).set_value(SHOWN - 4.0), m.FadeIn(factor_labels[2], shift=0.1 * m.DOWN)),
            ),
        )
        self.say(
            "[Multiply all fourteen], [and only about one part in a hundred thousand reaches the first note.]",
            K.animate(run_time=2.6, rate_func=m.linear).set_value(0.0),
            m.Circumscribe(live_readout, color=BAD, buff=0.1),
        )
        self.say(
            "[The early notes barely affect learning]: [the gradient has vanished.]",
            m.AnimationGroup(*[c.animate.set_opacity(0.3) for c in cells[:8]]),
            m.FadeIn(vanished, scale=1.2),
        )
        self.say(
            "[If the factors are bigger than one, the opposite happens]: [one and a half, fourteen times over, is almost three hundred.]",
            m.Wait(0.1),
            m.FadeIn(boom, shift=0.2 * m.UP),
        )
        self.say(
            "[Exploding gradients can be clipped.] [Vanishing ones need a better design.]",
            m.AnimationGroup(m.FadeOut(explain), m.FadeIn(clip, shift=0.2 * m.UP)),
            m.FadeIn(redesign, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.stop(dot, live_bars, live_readout)
        self.clear()

    # ------------------------------------------------------------------ the LSTM cell state

    def build_lstm(self):
        """The LSTM cell, at its place in the gates section (x shifted right by DX in the first)."""
        belt_y = 0.75
        d = {}
        d["shell"] = card(7.6, 3.8, VIOLET, CARD, stroke=2.5).move_to([-1.8, -0.55, 0]).set_z_index(-1)
        d["name"] = text("LSTM cell", 20, VIOLET, weight=m.BOLD).move_to([-4.85, 1.07, 0])
        d["belt"] = m.Arrow([-6.45, belt_y, 0], [2.75, belt_y, 0], buff=0, color=AMBER, stroke_width=6, max_tip_length_to_length_ratio=0.04)
        d["c_old"] = tex("c_{t-1}", 32, AMBER).move_to([-6.15, belt_y + 0.4, 0])
        d["c_new"] = tex("c_t", 32, AMBER).move_to([2.45, belt_y + 0.4, 0])
        gy = -1.45
        d["forget_x"] = op_node("x", [-3.9, belt_y, 0], PINK)
        d["f"] = gate_box("f", PINK, PINK_FILL, [-3.9, gy, 0])
        d["f_up"] = link([-3.9, gy + 0.36, 0], [-3.9, belt_y - 0.24, 0], PINK)
        d["plus"] = op_node("+", [-1.7, belt_y, 0], AMBER)
        d["input_x"] = op_node("x", [-1.7, -0.25, 0], TEAL)
        d["i"] = gate_box("i", TEAL, TEAL_FILL, [-2.5, gy, 0])
        d["cand"] = gate_box(r"\tilde{c}", VIOLET, VIOLET_FILL, [-0.9, gy, 0])
        ix = np.array([-1.7, -0.25, 0])
        d["i_up"] = link([-2.5, gy + 0.36, 0], ix + 0.24 * m.normalize(np.array([-0.8, -0.84, 0])), TEAL)
        d["c_up"] = link([-0.9, gy + 0.36, 0], ix + 0.24 * m.normalize(np.array([0.8, -0.84, 0])), VIOLET)
        d["write_up"] = link([-1.7, -0.25 + 0.24, 0], [-1.7, belt_y - 0.24, 0], TEAL)
        # the output column
        d["tap"] = m.Line([0.85, belt_y, 0], [0.85, 0.38, 0]).set_stroke(AMBER, width=3.5)
        tanh_box = card(0.95, 0.46, VIOLET, VIOLET_FILL, radius=0.12, stroke=2.5).move_to([0.85, 0.15, 0])
        d["tanh"] = m.VGroup(tanh_box, tex(r"\tanh", 26, INK).move_to(tanh_box)).set_z_index(2)
        d["out_x"] = op_node("x", [0.85, -0.55, 0], BLUE)
        d["tanh_down"] = link([0.85, -0.08, 0], [0.85, -0.55 + 0.24, 0], VIOLET)
        d["o"] = gate_box("o", BLUE, BLUE_FILL, [0.85, gy, 0])
        d["o_up"] = link([0.85, gy + 0.36, 0], [0.85, -0.55 - 0.24, 0], BLUE)
        d["h_out"] = link([0.85 + 0.24, -0.55, 0], [2.45, -0.55, 0], VIOLET)
        d["h_new"] = tex("h_t", 32, VIOLET).move_to([2.8, -0.55, 0])
        # the inputs: one bus feeding every gate
        bus_y = -2.1
        d["bus"] = m.Line([-4.15, bus_y, 0], [0.85, bus_y, 0]).set_stroke(MUTED, width=3)
        d["feeds"] = m.VGroup(*[m.Line([x, bus_y, 0], [x, gy - 0.36, 0]).set_stroke(MUTED, width=3) for x in (-3.9, -2.5, -0.9, 0.85)])
        d["bus_label"] = tex(r"h_{t-1},\ x_t", 28, INK).move_to([-4.85, bus_y, 0])
        return d

    def cell_state(self):
        self.next_section(
            "Long short-term memory",
            notes=(
                "LSTM: Hochreiter & Schmidhuber, Neural Computation 1997 (cell state with a constant-error path, input and output gates). "
                "The forget gate was added by Gers, Schmidhuber & Cummins, 2000. c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t, with "
                "c̃_t = tanh(W_c·h_{t−1} + U_c·x_t + b_c); ⊙ multiplies element by element."
            ),
        )
        d = self.build_lstm()
        self.lstm = d
        dx = 1.78
        for mob in d.values():
            mob.shift(dx * m.RIGHT)
        source = cite("Hochreiter & Schmidhuber, 1997").move_to([4.6, -2.75, 0])
        formula = ftex("c_t", "=", "f_t", r"\odot", "c_{t-1}", "+", "i_t", r"\odot", r"\tilde{c}_t", size=46)
        for k, color in ((0, AMBER), (2, PINK), (4, AMBER), (6, TEAL), (8, VIOLET)):
            formula[k].set_color(color)
        formula.move_to([0, 2.05, 0])
        self.lstm_formula = formula
        key = pill("the memory path: scaled and added to, never squashed", GOOD, GOOD_FILL, 22).move_to([0, -3.35, 0])
        belt_line = m.Line(d["belt"].get_start(), d["belt"].get_end())

        self.say(
            "[In nineteen ninety-seven, a fix arrived]: [long short-term memory, or LSTM.]",
            self.new_header("LONG SHORT-TERM MEMORY", "A second memory, with a straight path"),
            m.AnimationGroup(m.FadeIn(d["shell"]), m.FadeIn(d["name"]), m.FadeIn(source)),
        )
        self.say(
            "[Besides its hidden state, it keeps a second memory, the cell state]: [a line that runs straight through every step.]",
            m.AnimationGroup(m.GrowArrow(d["belt"]), m.FadeIn(d["c_old"]), m.FadeIn(d["c_new"])),
            m.ShowPassingFlash(belt_line.copy().set_stroke(INK, width=10), time_width=0.5, run_time=1.4),
        )
        self.say(
            "[It changes in only two ways]: [a gate erases part of it], [and another gate adds something new.]",
            m.Indicate(d["belt"], color=AMBER, scale_factor=1.03),
            m.AnimationGroup(m.FadeIn(d["forget_x"], scale=0.6), m.FadeIn(d["f"], shift=0.2 * m.UP), m.GrowArrow(d["f_up"])),
            m.AnimationGroup(
                m.FadeIn(d["plus"], scale=0.6), m.FadeIn(d["input_x"], scale=0.6), m.FadeIn(d["i"], shift=0.2 * m.UP),
                m.FadeIn(d["cand"], shift=0.2 * m.UP), m.GrowArrow(d["i_up"]), m.GrowArrow(d["c_up"]), m.GrowArrow(d["write_up"]),
            ),
        )
        self.say(
            "[In math]: [the new cell state] [is the forget gate times the old cell state], [plus the input gate times a new candidate memory.]",
            m.Write(formula),
            m.AnimationGroup(m.Indicate(formula[0], color=AMBER, scale_factor=1.3), m.Indicate(d["c_new"], color=AMBER, scale_factor=1.3)),
            m.AnimationGroup(m.Indicate(formula[2:5], color=PINK, scale_factor=1.15), m.Indicate(d["forget_x"], color=PINK, scale_factor=1.3), m.Indicate(d["f"], color=PINK)),
            m.AnimationGroup(m.Indicate(formula[6:9], color=TEAL, scale_factor=1.15), m.Indicate(d["input_x"], color=TEAL, scale_factor=1.3), m.Indicate(d["cand"], color=VIOLET)),
        )
        self.say(
            "[That plus sign is the trick.] [The memory isn't squashed at every step], [so when the forget gate stays near one, information and gradients can flow through many steps.]",
            m.AnimationGroup(m.Indicate(formula[5], color=AMBER, scale_factor=1.6), m.Indicate(d["plus"], color=AMBER, scale_factor=1.4)),
            m.FadeIn(key, shift=0.2 * m.UP),
            m.Succession(*[m.ShowPassingFlash(belt_line.copy().set_stroke(AMBER, width=12), time_width=0.4, run_time=1.0) for _ in range(2)]),
        )
        self.wait(0.3)
        self.play(m.FadeOut(key), m.FadeOut(source), run_time=0.6)

    # ------------------------------------------------------------------ the gates

    def gates(self):
        self.next_section(
            "The gates",
            notes=(
                "Each gate: g_t = σ(W_g·h_{t−1} + U_g·x_t + b_g), values between 0 and 1, one per memory unit. "
                "Forget: what to erase; input: what to write; output: what to show, h_t = o_t ⊙ tanh(c_t). "
                "Forget gate: Gers, Schmidhuber & Cummins, 2000."
            ),
        )
        d = self.lstm
        dx = 1.78
        built = [d[k] for k in ("shell", "name", "belt", "c_old", "c_new", "forget_x", "f", "f_up", "plus", "input_x", "i", "cand", "i_up", "c_up", "write_up")]
        new_parts = [d[k] for k in ("tap", "tanh", "out_x", "tanh_down", "o", "o_up", "h_out", "h_new", "bus", "feeds", "bus_label")]
        for mob in new_parts:
            mob.shift(dx * m.LEFT)
        old = self.lstm_formula
        gate = ftex("f_t", "=", r"\sigma(", "W_f", "h_{t-1}", "+", "U_f", "x_t", "+", "b_f", ")", size=46)
        gate[0].set_color(PINK)
        gate[4].set_color(VIOLET)
        gate[7].set_color(SKY)
        gate.move_to([-1.8, 2.05, 0])
        source = cite("forget gate: Gers et al., 2000").move_to([4.9, 2.1, 0])
        output = ftex("h_t", "=", "o_t", r"\odot", r"\tanh(", "c_t", ")", size=46)
        output[0].set_color(VIOLET)
        output[2].set_color(BLUE)
        output[5].set_color(AMBER)
        output.move_to(gate)

        ax = axes([-6, 6, 2], [0, 1, 0.5], 2.9, 1.45).move_to([5.1, 0.55, 0])
        curve = ax.plot(lambda z: 1 / (1 + np.exp(-z)), x_range=[-6, 6]).set_stroke(INK, width=4)
        one = text("1", 20, MUTED).next_to(ax.c2p(-6, 1), m.LEFT, buff=0.15)
        zero = text("0", 20, MUTED).next_to(ax.c2p(-6, 0), m.LEFT, buff=0.15)
        s_title = text("sigmoid", 22, MUTED, weight=m.SEMIBOLD).next_to(ax, m.UP, buff=0.12)
        ends = m.VGroup(text("0 = closed", 20, MUTED), text("1 = open", 20, MUTED)).arrange(m.RIGHT, buff=0.5).next_to(ax, m.DOWN, buff=0.2)
        z = m.ValueTracker(0.0)
        dot = m.always_redraw(lambda: m.Dot(ax.c2p(z.get_value(), 1 / (1 + np.exp(-z.get_value()))), radius=0.1, color=AMBER).set_z_index(3))

        def row(symbol, words, color):
            return m.VGroup(tex(symbol, 34, color), text(words, 22, INK)).arrange(m.RIGHT, buff=0.25)

        legend = m.VGroup(
            row("f", "forget: what to erase", PINK),
            row("i", "input: what to write", TEAL),
            row("o", "output: what to show", BLUE),
        ).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        legend.move_to([3.45, -1.75, 0], aligned_edge=m.LEFT)
        out_path = m.VMobject().set_points_as_corners([[0.85, 0.75, 0], [0.85, -0.55, 0], [2.45, -0.55, 0]])

        self.say(
            "[Today's LSTM has three gates], [each a sigmoid of the current note and the previous hidden state], [with its own weights.]",
            m.AnimationGroup(
                self.new_header("THE GATES", "Erase, write, show"),
                *[mob.animate.shift(dx * m.LEFT) for mob in built],
                m.LaggedStart(*[m.FadeIn(mob) for mob in new_parts], lag_ratio=0.05),
                run_time=1.6,
            ),
            m.AnimationGroup(m.FadeOut(old), m.Write(gate), m.FadeIn(source), *[m.Indicate(d[k], color=INK, scale_factor=1.2) for k in ("f", "i", "o")]),
            m.AnimationGroup(m.Indicate(gate[3], color=PINK, scale_factor=1.4), m.Indicate(gate[6], color=PINK, scale_factor=1.4), m.Indicate(gate[9], color=PINK, scale_factor=1.4)),
        )
        self.say(
            "[A sigmoid squashes any number into the range zero to one]: [zero means the gate is closed], [one means it's fully open.]",
            m.AnimationGroup(m.Create(ax), m.Create(curve), m.FadeIn(one), m.FadeIn(zero), m.FadeIn(s_title), m.FadeIn(dot)),
            m.AnimationGroup(z.animate.set_value(-5.0), m.FadeIn(ends[0])),
            m.AnimationGroup(z.animate.set_value(5.0), m.FadeIn(ends[1])),
        )
        self.say(
            "[The forget gate decides what to erase from the cell state.]",
            m.AnimationGroup(m.FadeIn(legend[0], shift=0.15 * m.UP), m.Indicate(d["f"], color=PINK, scale_factor=1.3), m.Indicate(d["forget_x"], color=PINK, scale_factor=1.4)),
        )
        self.say(
            "[The input gate decides what new information to write.]",
            m.AnimationGroup(m.FadeIn(legend[1], shift=0.15 * m.UP), m.Indicate(d["i"], color=TEAL, scale_factor=1.3), m.Indicate(d["input_x"], color=TEAL, scale_factor=1.4)),
        )
        self.say(
            "[And the output gate decides what to show]: [the new hidden state is the output gate times the squashed cell state.]",
            m.AnimationGroup(m.FadeIn(legend[2], shift=0.15 * m.UP), m.Indicate(d["o"], color=BLUE, scale_factor=1.3), m.Indicate(d["out_x"], color=BLUE, scale_factor=1.4)),
            m.AnimationGroup(m.FadeOut(gate), m.Write(output), m.ShowPassingFlash(out_path.set_stroke(VIOLET, width=10), time_width=0.5, run_time=1.4)),
        )
        self.wait(0.3)
        self.stop(dot)
        self.clear()

    # ------------------------------------------------------------------ back to the melody

    def melody_again(self):
        self.next_section(
            "Back to the melody",
            notes=(
                "Illustrative gate values (hand-set): the input gate opens at note 1 and writes the home note C; the forget gate "
                "holds at 0.99 for 14 steps; the output gate opens at note 15. Computed: 0.99^14 = 0.869 ≈ 87% kept, versus "
                "0.000011 of the gradient in the plain network above. The predicted C is illustrative."
            ),
        )
        roll = Roll(MELODY[:SHOWN], 16, 0.66, 0.28, (0.55, 1.5), labels=False)
        slot = roll.slot(15)
        mark = text("?", 30, INK, weight=m.BOLD).move_to(slot)
        predicted = roll.block(15, "C", GOOD, GOOD_FILL, labels=False)
        predicted_label = text("C", 24, GOOD, weight=m.BOLD).next_to(predicted, m.RIGHT, buff=0.15)
        cols = [roll.x(i) for i in range(SHOWN)]
        belt_y = 0.05
        belt = m.Line([cols[0] - 0.3, belt_y, 0], [cols[-1] + 0.3, belt_y, 0]).set_stroke(AMBER, width=5)
        belt_label = text("cell state", 22, AMBER, weight=m.SEMIBOLD).move_to([-5.7, belt_y, 0])
        memo = chip("C", AMBER, AMBER_FILL, width=0.55, height=0.48, size=22).move_to([cols[0], belt_y, 0]).set_z_index(3)

        def strip(values, start, y, color, name):
            bars = m.VGroup()
            for k, v in enumerate(values):
                bar = m.Rectangle(width=0.36, height=max(0.03, 0.6 * v), stroke_width=0).set_fill(color, opacity=0.9)
                bars.add(bar.move_to([cols[start + k], y - 0.3, 0], aligned_edge=m.DOWN))
            base = m.Line([cols[0] - 0.3, y - 0.3, 0], [cols[-1] + 0.3, y - 0.3, 0]).set_stroke(FAINT, width=2)
            label = text(name, 22, color, weight=m.SEMIBOLD).move_to([-5.7, y, 0])
            return bars, m.VGroup(base, label)

        f_bars, f_frame = strip(FORGET_GATE, 1, -0.85, PINK, "forget")
        i_bars, i_frame = strip(INPUT_GATE, 0, -1.7, TEAL, "input")
        o_bars, o_frame = strip(OUTPUT_GATE, 0, -2.55, BLUE, "output")
        f_value = code("0.99", 20, PINK).next_to(f_bars, m.UP, buff=0.08)
        kept = m.VGroup(
            text("memory kept:", 24, MUTED), tex(r"0.99^{14} \approx 0.87", 34, AMBER),
            text("plain network:", 24, MUTED), code(fmt(REACH[-1]), 28, BAD),
        ).arrange(m.RIGHT, buff=0.25)
        kept[2].shift(0.5 * m.RIGHT)
        kept[3].next_to(kept[2], m.RIGHT, buff=0.25)
        kept.move_to([0.3, -3.35, 0])

        self.say(
            "[Back to our melody.] [At the first note, the input gate opens], [and writes the home note, C, into the cell state.]",
            m.AnimationGroup(
                self.new_header("BACK TO THE MELODY", "Remembering how it began"),
                m.FadeIn(roll.lanes), m.FadeIn(roll.keys), m.FadeIn(roll.blocks), m.Create(slot), m.FadeIn(mark),
                m.Create(belt), m.FadeIn(belt_label), m.FadeIn(f_frame), m.FadeIn(i_frame), m.FadeIn(o_frame),
            ),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in i_bars], lag_ratio=0.03), m.Indicate(i_bars[0], color=TEAL, scale_factor=1.3)),
            m.AnimationGroup(m.FadeIn(memo, target_position=roll.blocks[0]), m.Circumscribe(roll.blocks[0], color=AMBER, buff=0.05)),
        )
        self.say(
            "[For the next fourteen notes], [the forget gate stays near one, say ninety-nine percent], [and the input gate stays almost shut.]",
            m.LaggedStart(*[m.Indicate(b, color=SKY, scale_factor=1.15) for b in roll.blocks[1:]], lag_ratio=0.05),
            m.AnimationGroup(
                m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in f_bars], lag_ratio=0.05), m.FadeIn(f_value),
                memo.animate(run_time=2.4, rate_func=m.linear).move_to([cols[-1], belt_y, 0]),
            ),
            m.Indicate(i_bars[1:], color=TEAL, scale_factor=1.3),
        )
        self.say(
            "[Ninety-nine percent, fourteen times over, still keeps eighty-seven percent of the memory, and of its gradient.] [The plain network kept about one part in a hundred thousand.]",
            m.FadeIn(kept[:2], shift=0.2 * m.UP),
            m.FadeIn(kept[2:], shift=0.2 * m.UP),
        )
        self.say(
            "[At the last note, the output gate opens], [and the network can predict that the tune comes home, to C.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in o_bars], lag_ratio=0.03), m.Indicate(o_bars[-1], color=BLUE, scale_factor=1.3)),
            m.AnimationGroup(m.FadeOut(mark), m.FadeIn(predicted, target_position=memo), m.FadeIn(predicted_label), m.Flash(predicted.get_center(), color=GOOD, flash_radius=0.45)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section(
            "The big picture",
            notes=(
                "LSTM uses, e.g.: handwriting (Graves et al., 2009), speech (Graves et al., 2013), translation (Sutskever et al., 2014). "
                "Recurrence is sequential, so training cannot run over the time steps in parallel. Transformers (Vaswani et al., 2017; Episode 24) "
                "read all positions at once. State space models such as Mamba (Gu & Dao, 2023; Episode 40) use linear recurrences "
                "trained with parallel scans. Facts as of October 2026."
            ),
        )
        uses = (
            ("ep21-mic", "Speech recognition", "Graves et al., 2013"),
            ("ep21-translate", "Machine translation", "Sutskever et al., 2014"),
            ("ep21-pen", "Handwriting", "Graves et al., 2009"),
        )
        tiles = m.VGroup()
        for name, title, source in uses:
            box = card(4.3, 1.75, VIOLET, CARD, stroke=2)
            words = m.VGroup(text(title, 22, INK, weight=m.BOLD), cite(source)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            body = m.VGroup(icon(name, 0.85), words).arrange(m.RIGHT, buff=0.28).move_to(box)
            tiles.add(m.VGroup(box, body))
        tiles.arrange(m.RIGHT, buff=0.25).move_to([0, 1.45, 0])

        def panel(x, title, color):
            box = card(6.2, 2.45, EDGE, CARD, stroke=2).move_to([x, -1.05, 0])
            head = text(title, 24, color, weight=m.BOLD).move_to(box.get_top() + 0.38 * m.DOWN)
            return m.VGroup(box, head)

        left = panel(-3.35, "Recurrent: one step at a time", VIOLET)
        right = panel(3.35, "Transformer: every position at once", GOOD)
        lx = [-5.6 + 0.9 * k for k in range(6)]
        rx = [1.1 + 0.9 * k for k in range(6)]
        dot_y = -1.35
        l_dots = m.VGroup(*[m.Circle(radius=0.17).set_stroke(VIOLET, width=3).set_fill(BG, opacity=1).move_to([x, dot_y, 0]) for x in lx])
        l_links = m.VGroup(*[link([a + 0.19, dot_y, 0], [b - 0.19, dot_y, 0], VIOLET, width=3) for a, b in zip(lx, lx[1:])])
        r_dots = m.VGroup(*[m.Circle(radius=0.17).set_stroke(GOOD, width=3).set_fill(BG, opacity=1).move_to([x, dot_y, 0]) for x in rx])
        arcs = m.VGroup(
            *[
                m.ArcBetweenPoints([rx[a], dot_y + 0.19, 0], [rx[b], dot_y + 0.19, 0], angle=-0.9).set_stroke(GOOD, width=2, opacity=0.6)
                for a in range(6)
                for b in range(a + 1, 6)
            ]
        )
        slow = text("slow to train on long sequences", 20, BAD).move_to([-3.35, -1.95, 0])
        fast = text("trains in parallel · Episode 24", 20, GOOD).move_to([3.35, -1.95, 0])
        ssm = pill("state space models, like Mamba: recurrence returns · Episode 40", PINK, PINK_FILL, 22).move_to([0, -3.15, 0])
        sequential = m.Succession(*[m.AnimationGroup(d.animate(rate_func=m.there_and_back, run_time=0.3).set_fill(VIOLET, opacity=1)) for d in l_dots])

        self.say(
            "[For years, LSTMs powered speech recognition, machine translation, and handwriting recognition.]",
            m.AnimationGroup(self.new_header("THE BIG PICTURE", "Where recurrence went"), m.LaggedStart(*[m.FadeIn(t, shift=0.2 * m.UP) for t in tiles], lag_ratio=0.25)),
        )
        self.say(
            "[Their catch: they read one step at a time], [so training on long sequences is slow, and hard to run in parallel.]",
            m.AnimationGroup(m.FadeIn(left), m.LaggedStart(*[m.FadeIn(d, scale=0.6) for d in l_dots], lag_ratio=0.1), m.LaggedStart(*[m.GrowArrow(a) for a in l_links], lag_ratio=0.1)),
            m.AnimationGroup(sequential, m.FadeIn(slow, shift=0.1 * m.UP)),
        )
        self.say(
            "[Transformers, in episode twenty-four, read every position at once], [and replaced recurrent networks for most language tasks.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(r_dots, scale=0.6), m.Create(arcs, run_time=1.2)),
            m.AnimationGroup(m.AnimationGroup(*[d.animate(rate_func=m.there_and_back, run_time=0.8).set_fill(GOOD, opacity=1) for d in r_dots]), m.FadeIn(fast, shift=0.1 * m.UP)),
        )
        self.say(
            "[But recurrence is coming back]: [state space models, like Mamba, in episode forty, bring it back in a form that trains fast.]",
            m.Indicate(left[1], color=VIOLET, scale_factor=1.08),
            m.FadeIn(ssm, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Loop, fade, gate.")
        halo = glow(VIOLET, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Loop", "Fade", "Gate"), (VIOLET, BAD, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A loop gives a network a memory; gates let that memory last.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [loop], [fade], [gate.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A loop gives a network a memory], [and gates let that memory last.]", m.Write(glyphs(tagline, "A loop gives a network a memory;")), m.Write(glyphs(tagline, "gates let that memory last.")))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
