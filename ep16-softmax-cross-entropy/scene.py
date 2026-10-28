"""AI, Explained · Episode 16 · Softmax & Cross-Entropy

Running example: a classifier looks at a picture (of a cat) and gives three raw scores,
logits, for example z = [2.0, 1.0, 0.1] for cat, dog and bird.

- softmax: exponentiate (e^z ≈ 7.39, 2.72, 1.11), then normalise by the total (≈ 11.21):
  p ≈ [0.66, 0.24, 0.10]; one point more score = e times the share;
- temperature: softmax(z / T); T = 0.5 sharpens (cat 86 %), T = 5 flattens (40 / 33 / 27 %);
- cross-entropy for the true class: L = −log p(cat) ≈ 0.42; the −log curve punishes
  confident mistakes (p = 0.01 → 4.6);
- the gradient with respect to the logits is p − y = [−0.34, +0.24, +0.10]; one gradient
  step (learning rate 1, for example) lifts p(cat) to 77 % and cuts the loss to 0.26;
- the same softmax and cross-entropy train language models over their whole vocabulary.

Every number on screen and in the narration is computed below with numpy and asserted (the
gradient also against finite differences), so the film cannot drift from its words. The
logits are illustrative; the language-model bars are seeded random logits, for illustration.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- colours

# the three classes (category colours: never used for steps)
CAT, DOG, BIRD = PINK, BLUE, LIME
CAT_FILL, DOG_FILL, BIRD_FILL = PINK_FILL, "#13243A", "#1E2A0F"
NAMES = ("cat", "dog", "bird")
COLORS = (CAT, DOG, BIRD)
ICONS = ("ep16-cat", "ep16-dog", "ep16-bird")
# the concepts
SCORE, SCORE_FILL = SKY, SKY_FILL  # logits
PROB, PROB_FILL = AMBER, AMBER_FILL  # probabilities, softmax
TRUTH, TRUTH_FILL = GOOD, GOOD_FILL  # the true label, one-hot
LOSS, LOSS_FILL = BAD, BAD_FILL  # cross-entropy
GRAD, GRAD_FILL = VIOLET, VIOLET_FILL  # the gradient p − y

# ---------------------------------------------------------------------------- the numbers


def softmax(z, t=1.0):
    e = np.exp(np.asarray(z, dtype=float) / t)
    return e / e.sum()


def cross_entropy(z, true=0):
    return float(-np.log(softmax(z)[true]))


Z = np.array([2.0, 1.0, 0.1])  # the logits: cat, dog, bird (for example)
EXP = np.exp(Z)
TOTAL = float(EXP.sum())
P = softmax(Z)
COLD, HOT = 0.5, 5.0
P_COLD, P_HOT = softmax(Z, COLD), softmax(Z, HOT)
Y = np.array([1.0, 0.0, 0.0])  # the truth: a cat, one-hot
LOSS0 = cross_entropy(Z)
SURE_RIGHT, SURE_WRONG = 0.95, 0.01
GRADIENT = P - Y
LR = 1.0  # for example
Z_NEW = Z - LR * GRADIENT
P_NEW = softmax(Z_NEW)
LOSS_NEW = cross_entropy(Z_NEW)


def pct(p):
    return int(round(100 * float(p)))


# the narration's numbers
assert [round(float(v), 2) for v in P] == [0.66, 0.24, 0.10]  # "about 0.66, 0.24, 0.10"
assert [round(float(v), 1) for v in EXP] == [7.4, 2.7, 1.1]  # "seven point four, two point seven, one point one"
assert round(TOTAL, 1) == 11.2  # "about eleven point two"
assert [pct(v) for v in P] == [66, 24, 10] and sum(pct(v) for v in P) == 100  # "sixty-six, twenty-four, ten"
assert round(float(P[0] / P[1]), 1) == 2.7 and np.isclose(P[0] / P[1], np.e)  # "e times": "about two point seven"
assert pct(P_COLD[0]) == 86  # "cat jumps to eighty-six percent"
assert [pct(v) for v in P_HOT] == [40, 33, 27]  # "forty, thirty-three and twenty-seven percent"
assert round(LOSS0, 2) == 0.42  # "a loss of about zero point four two"
assert -np.log(SURE_RIGHT) < 0.06  # "costs almost nothing"
assert round(float(-np.log(SURE_WRONG)), 1) == 4.6  # "about four point six"
assert [round(float(v), 2) for v in GRADIENT] == [-0.34, 0.24, 0.10]  # "minus 0.34, plus 0.24, plus 0.1"
assert pct(P_NEW[0]) == 77 and round(LOSS_NEW, 2) == 0.26  # "seventy-seven percent", "0.42 to 0.26"
assert np.argmax(P_NEW) == 0 and Z_NEW[0] > Z[0] and np.all(Z_NEW[1:] < Z[1:])  # cat up, the others down
_h = 1e-6  # the gradient of cross-entropy really is p − y: central differences
_numeric = [(cross_entropy(Z + _h * e) - cross_entropy(Z - _h * e)) / (2 * _h) for e in np.eye(3)]
assert np.allclose(_numeric, GRADIENT, atol=1e-6)

# a language model's next word: seeded random logits over a slice of a vocabulary (illustrative)
_rng = np.random.default_rng(16)
VOCAB_Z = _rng.normal(0.0, 1.0, 48)
VOCAB_Z[17] = 4.2  # one clear favourite ("mat")
VOCAB_Z[34] = 3.1
VOCAB_Z[6] = 2.6
VOCAB_P = softmax(VOCAB_Z)
assert int(np.argmax(VOCAB_P)) == 17

# ---------------------------------------------------------------------------- helpers


def num(v, places=2):
    """A number for the screen, with a real minus sign."""
    return f"{v:.{places}f}".replace("-", "−")


def signed(v, places=2):
    return ("+" if v >= 0 else "−") + f"{abs(v):.{places}f}"


def bar_at(height, color, width, x, base, opacity=0.9):
    """A bar standing on y = base, centred on x."""
    rect = m.Rectangle(width=width, height=max(height, 0.02), stroke_width=0).set_fill(color, opacity=opacity)
    return rect.move_to([x, base, 0], aligned_edge=m.DOWN)


def class_names(xs, y, size=22):
    """cat, dog, bird under three columns, on one baseline."""
    names = m.VGroup(*[text(n, size, c, weight=m.SEMIBOLD).move_to([x, y, 0]) for n, c, x in zip(NAMES, COLORS, xs)])
    return on_baseline(names)


# ---------------------------------------------------------------------------- the episode


class Ep16SoftmaxCrossEntropy(Episode):
    number = 16

    def lesson(self):
        self.hook()
        self.softmax_steps()
        self.temperature()
        self.cross_entropy()
        self.gradient()
        self.big_picture()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="A picture, a network, three raw scores. The logits 2.0 / 1.0 / 0.1 are illustrative.")
        frame = card(2.6, 2.6, EDGE, CARD).move_to([-5.15, 0.3, 0])
        cat = icon("ep16-cat", 1.75).move_to(frame)
        caption = text("a picture", 22, MUTED).next_to(frame, m.DOWN, buff=0.25)

        def layer(x, n, spread, color, radius=0.16):
            ys = np.linspace(spread, -spread, n) + 0.3
            return m.VGroup(*[m.Circle(radius=radius).set_stroke(color, width=2.5).set_fill(BG, opacity=1).move_to([x, y, 0]) for y in ys])

        l1, l2 = layer(-2.95, 4, 1.35, SKY), layer(-1.65, 5, 1.7, VIOLET)
        l3 = layer(-0.35, 3, 1.2, SCORE, radius=0.2)

        def links(a, b):
            return m.VGroup(*[m.Line(p.get_center(), q.get_center(), buff=0.18).set_stroke(MUTED, width=1.3, opacity=0.35) for p in a for q in b])

        e1, e2 = links(l1, l2), links(l2, l3)
        for e in (e1, e2):
            e.set_z_index(-1)
        feed = arrow(frame.get_right(), [l1.get_left()[0], 0.3, 0], MUTED, buff=0.2)
        net = m.VGroup(e1, e2, l1, l2, l3)

        rows, chips = m.VGroup(), m.VGroup()
        for k, (node_, name, color, glyph) in enumerate(zip(l3, NAMES, COLORS, ICONS)):
            y = node_.get_y()
            pic = icon(glyph, 0.72).move_to([1.05, y, 0])
            label = text(name, 30, color, weight=m.BOLD).next_to(pic, m.RIGHT, buff=0.3)
            rows.add(m.VGroup(pic, label))
            box = card(1.45, 0.78, SCORE, SCORE_FILL, radius=0.18)
            chips.add(m.VGroup(box, code(num(Z[k], 1), 32, INK, weight=m.BOLD).move_to(box)).move_to([4.65, y, 0]))
        for r in rows[1:]:  # names on one baseline per row is automatic; align the names' left edges
            r[1].align_to(rows[0][1], m.LEFT)
        leads = m.VGroup(*[m.Line(n.get_right(), r[0].get_left(), buff=0.12).set_stroke(MUTED, width=2, opacity=0.6) for n, r in zip(l3, rows)])
        logits = pill("logits", SCORE, SCORE_FILL, 24).next_to(chips, m.UP, buff=0.35)
        anything = text("any number, even negative", 22, MUTED).next_to(chips, m.DOWN, buff=0.45)
        total = text(f"2.0 + 1.0 + 0.1 = {Z.sum():.1f}   ≠ 1", 24, INK, weight=m.SEMIBOLD, t2c={"≠ 1": LOSS}).next_to(anything, m.DOWN, buff=0.2)
        assert np.isclose(Z.sum(), 3.1)
        plan = m.VGroup(
            pill("softmax: scores → probabilities", PROB, PROB_FILL, 22),
            pill("cross-entropy: learning from them", LOSS, LOSS_FILL, 22),
        ).arrange(m.RIGHT, buff=0.5).move_to([0, -3.3, 0])

        self.say(
            "[Show a classifier a picture], [and it has to decide]: [cat, dog, or bird?]",
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(cat, scale=0.8), m.FadeIn(caption)),
            m.AnimationGroup(m.GrowArrow(feed), m.FadeIn(net)),
            m.AnimationGroup(m.LaggedStart(*[m.AnimationGroup(m.Create(l), m.FadeIn(r, shift=0.2 * m.RIGHT)) for l, r in zip(leads, rows)], lag_ratio=0.25)),
        )
        self.say(
            "[Its last layer gives one raw score for each class.] [For example: two for cat, one for dog, and zero point one for bird.]",
            m.AnimationGroup(
                m.ShowPassingFlash(e1.copy().set_stroke(SKY, width=2.5, opacity=0.9), time_width=0.5),
                m.ShowPassingFlash(e2.copy().set_stroke(SCORE, width=2.5, opacity=0.9), time_width=0.5),
                run_time=1.4,
            ),
            m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.LEFT) for c in chips], lag_ratio=0.45),
        )
        self.say(
            "[These raw scores are called logits.] [Bigger means more likely], but they are not probabilities yet.",
            m.FadeIn(logits, shift=0.2 * m.DOWN),
            m.Indicate(chips[0], color=SCORE, scale_factor=1.12),
        )
        self.say(
            "[They can be any number, even negative], [and they don't add up to one.]",
            m.FadeIn(anything, shift=0.15 * m.UP),
            m.FadeIn(total, shift=0.15 * m.UP),
        )
        self.say(
            "[Today: how softmax turns scores into probabilities], [and how cross-entropy helps the model learn from them.]",
            m.FadeIn(plan[0], shift=0.2 * m.UP),
            m.FadeIn(plan[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ softmax, step by step

    def softmax_steps(self):
        self.next_section(
            "Softmax",
            notes=f"e^z = {', '.join(f'{v:.3f}' for v in EXP)}; total {TOTAL:.3f}; p = {', '.join(f'{v:.4f}' for v in P)}. "
            "p(cat)/p(dog) = e exactly (scores differ by 1). Name 'softmax': J. S. Bridle, 1990 (Neurocomputing, NATO ASI F68).",
        )
        xs = (-4.55, 0.0, 4.55)
        offsets = (-1.05, 0.0, 1.05)
        tall = 1.85  # the tallest bar in each panel
        heights = (Z * tall / Z.max(), EXP * tall / EXP.max(), P * tall / P.max())
        assert np.allclose(heights[1], heights[2])  # normalising rescales: the shape stays the same
        titles = ("Scores", "Exponentiate", "Normalise")
        subs = (tex("z", 40, SCORE), tex(r"e^{z}", 40, INK), tex(rf"e^{{z}} \div {TOTAL:.1f}", 40, PROB))
        labels = ([num(v, 1) for v in Z], [num(v, 2) for v in EXP], [f"{pct(v)}%" for v in P])
        panels = []
        for x, title, sub, hs, labs in zip(xs, titles, subs, heights, labels):
            box = card(3.7, 4.3, EDGE, CARD, stroke=2).move_to([x, 0.3, 0])
            head = text(title, 26, INK, weight=m.BOLD).move_to(box.get_top() + 0.4 * m.DOWN)
            sub.next_to(head, m.DOWN, buff=0.14)
            base = box.get_bottom()[1] + 0.8
            bars = m.VGroup(*[bar_at(h, c, 0.62, x + dx, base) for h, c, dx in zip(hs, COLORS, offsets)]).set_z_index(2)
            vals = m.VGroup(*[code(s, 24, INK, weight=m.SEMIBOLD).next_to(b, m.UP, buff=0.12) for s, b in zip(labs, bars)]).set_z_index(2)
            names = class_names([x + dx for dx in offsets], base - 0.33, 20)
            panels.append(dict(box=box, head=head, sub=sub, bars=bars, vals=vals, names=names))
        steps = [
            arrow([panels[0]["box"].get_right()[0] + 0.02, 0.3, 0], [panels[1]["box"].get_left()[0] - 0.02, 0.3, 0], MUTED, buff=0.05),
            arrow([panels[1]["box"].get_right()[0] + 0.02, 0.3, 0], [panels[2]["box"].get_left()[0] - 0.02, 0.3, 0], MUTED, buff=0.05),
        ]
        ratio = pill("one point more score  →  e ≈ 2.7 times the share", PROB, PROB_FILL, 22).move_to([0, -2.45, 0])
        sum_line = code(" + ".join(num(v, 2) for v in EXP) + f" = {TOTAL:.2f}", 24, MUTED).move_to([0, -2.45, 0])
        formula = m.MathTex(r"\mathrm{softmax}(z)_i", "=", r"\frac{e^{z_i}}{\sum_j e^{z_j}}", font_size=42, color=INK)
        formula.move_to([-0.6, -2.85, 0])
        numerator, denominator, rule = formula[2][0:3], formula[2][3:8], formula[2][8]
        denominator.set_color(PROB)
        cite = text("the name: Bridle, 1990", 20, MUTED).next_to(formula, m.RIGHT, buff=0.7)

        def show(p, bars_from=None):
            parts = [m.FadeIn(p["box"]), m.FadeIn(p["head"]), m.FadeIn(p["sub"]), m.FadeIn(p["names"])]
            if bars_from is None:
                parts.append(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in p["bars"]], lag_ratio=0.2))
            else:
                parts.append(m.AnimationGroup(*[m.TransformFromCopy(a, b) for a, b in zip(bars_from, p["bars"])]))
            return m.AnimationGroup(*parts)

        one, two, three = panels
        self.say(
            "[Softmax works in two steps.] [Here are our three scores, as bars.]",
            self.new_header("SOFTMAX", "Exponentiate, then normalise"),
            m.AnimationGroup(show(one), m.FadeIn(one["vals"], shift=0.1 * m.UP)),
        )
        self.say(
            "[Step one: raise the number e, about two point seven, to the power of each score.]",
            m.AnimationGroup(m.GrowArrow(steps[0]), show(two, one["bars"]), run_time=1.6),
        )
        self.say(
            "[That gives about seven point four, two point seven, and one point one]: [all positive, whatever the scores were.]",
            m.LaggedStart(*[m.FadeIn(v, shift=0.1 * m.UP) for v in two["vals"]], lag_ratio=0.5),
            m.Indicate(two["bars"], color=INK, scale_factor=1.05),
        )
        self.say(
            "[A score one point higher gets e times the weight]: [so cat gets about two point seven times dog's share.]",
            m.AnimationGroup(m.Indicate(m.VGroup(one["vals"][0], one["vals"][1]), color=SCORE, scale_factor=1.15)),
            m.AnimationGroup(m.Indicate(m.VGroup(two["bars"][0], two["bars"][1]), color=INK, scale_factor=1.06), m.FadeIn(ratio, shift=0.2 * m.UP)),
        )
        self.say(
            "[Step two: divide each one by their total], [about eleven point two.]",
            m.AnimationGroup(m.FadeOut(ratio), m.GrowArrow(steps[1]), show(three, two["bars"]), run_time=1.6),
            m.FadeIn(sum_line, shift=0.15 * m.UP),
        )
        self.say(
            "[Now they add up to one]: [sixty-six percent cat, twenty-four percent dog, and ten percent bird.]",
            m.Indicate(three["sub"], color=PROB, scale_factor=1.15),
            m.LaggedStart(*[m.FadeIn(v, shift=0.1 * m.UP) for v in three["vals"]], lag_ratio=0.6),
        )
        self.say(
            "[In symbols, that's the softmax]: [e to the power of one score], [divided by the sum of e to the power of every score.]",
            m.AnimationGroup(m.FadeIn(formula[0]), m.FadeIn(formula[1]), m.FadeIn(cite)),
            m.AnimationGroup(m.FadeIn(numerator, shift=0.1 * m.DOWN), m.Indicate(two["vals"], color=INK, scale_factor=1.12)),
            m.AnimationGroup(m.Create(rule), m.FadeTransform(sum_line, denominator)),
        )
        self.say(
            "[The biggest score always gets the biggest share], [and no class ever gets exactly zero.]",
            m.AnimationGroup(*[m.Indicate(p["bars"][0], color=INK, scale_factor=1.06) for p in panels]),
            m.AnimationGroup(m.Indicate(three["bars"][2], color=INK, scale_factor=1.3), m.Indicate(three["vals"][2], color=BIRD, scale_factor=1.2)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ temperature

    def temperature(self):
        self.next_section(
            "Temperature",
            notes=f"softmax(z/T): T = {COLD} → {', '.join(f'{v:.3f}' for v in P_COLD)}; T = 1 → {', '.join(f'{v:.3f}' for v in P)}; "
            f"T = {HOT:g} → {', '.join(f'{v:.3f}' for v in P_HOT)}. Returns in Episode 33 (decoding and sampling).",
        )
        T = m.ValueTracker(1.0)
        xs, base, width, scale = (-4.95, -3.3, -1.65), -1.75, 0.95, 3.9

        def make_bars():
            p = softmax(Z, T.get_value())
            bars = m.VGroup(*[bar_at(v * scale, c, width, x, base) for v, c, x in zip(p, COLORS, xs)])
            vals = m.VGroup(*[code(f"{pct(v)}%", 26, INK, weight=m.SEMIBOLD).next_to(b, m.UP, buff=0.12) for v, b in zip(p, bars)])
            return m.VGroup(bars, vals)

        live = m.always_redraw(make_bars)
        floor = m.Line([-5.75, base, 0], [-0.85, base, 0]).set_stroke(FAINT, width=2.5)
        icons = m.VGroup(*[icon(g, 0.55).move_to([x, base - 0.45, 0]) for g, x in zip(ICONS, xs)])
        names = class_names(xs, base - 1.05, 22)

        formula = m.MathTex("p", "=", r"\mathrm{softmax}", r"(z / T)", font_size=48, color=INK).move_to([3.3, 1.85, 0])
        formula[0].set_color(PROB)
        formula[3].set_color(SCORE)
        t_label = text("temperature", 30, PROB, weight=m.BOLD)
        t_label.move_to([3.3, 0.75, 0]).shift(1.05 * m.LEFT)
        t_value = m.always_redraw(lambda: text(f"T = {T.get_value():.2f}", 30, PROB, weight=m.BOLD).next_to(t_label, m.RIGHT, buff=0.3))
        readout = m.VGroup(t_label, t_value)

        lo, hi, x0, x1, track_y = 0.35, 6.5, 1.0, 5.6, -0.45

        def at(t):
            return x0 + (np.log(t) - np.log(lo)) / (np.log(hi) - np.log(lo)) * (x1 - x0)

        track = m.Line([x0, track_y, 0], [x1, track_y, 0]).set_stroke(EDGE, width=8)
        ticks = m.VGroup(*[m.Line([at(t), track_y - 0.16, 0], [at(t), track_y + 0.16, 0]).set_stroke(MUTED, width=2.5) for t in (COLD, 1.0, HOT)])
        tick_labels = m.VGroup(*[text(s, 22, MUTED).move_to([at(t), track_y - 0.5, 0]) for s, t in (("0.5", COLD), ("1", 1.0), ("5", HOT))])
        ends = on_baseline(m.VGroup(
            text("sharper", 22, MUTED, weight=m.SEMIBOLD).next_to([x0, track_y, 0], m.UP, buff=0.3).align_to([x0, 0, 0], m.LEFT),
            text("flatter", 22, MUTED, weight=m.SEMIBOLD).next_to([x1, track_y, 0], m.UP, buff=0.3).align_to([x1, 0, 0], m.RIGHT),
        ))
        knob = m.always_redraw(lambda: m.Circle(radius=0.17).set_stroke(INK, width=3).set_fill(PROB, opacity=1).move_to([at(T.get_value()), track_y, 0]))
        slider = m.VGroup(track, ticks, tick_labels, ends)
        moods = m.VGroup(
            pill("low T: decisive", PROB, PROB_FILL, 22),
            pill("high T: hedges its bets", SCORE, SCORE_FILL, 22),
        ).arrange(m.RIGHT, buff=0.35).move_to([3.3, -1.75, 0])
        later = pill("picking the next word · Episode 33", TEAL, TEAL_FILL, 22).move_to([3.3, -2.95, 0])

        self.say(
            "[Softmax has one extra knob, called the temperature.]",
            m.AnimationGroup(
                self.new_header("TEMPERATURE", "Sharper or flatter"),
                m.Create(floor), m.FadeIn(icons), m.FadeIn(names), m.FadeIn(live),
            ),
        )
        self.say(
            "[Before the softmax, we divide every score by the temperature.] [At a temperature of one, nothing changes.]",
            m.Succession(m.Write(formula), m.Indicate(formula[3], color=SCORE, scale_factor=1.2)),
            m.AnimationGroup(m.FadeIn(readout), m.Create(track), m.FadeIn(ticks), m.FadeIn(tick_labels), m.FadeIn(ends), m.FadeIn(knob)),
        )
        self.say(
            "[Turn it down to one half], [and the gaps grow]: [cat jumps to eighty-six percent.]",
            m.Indicate(tick_labels[0], color=PROB, scale_factor=1.4),
            T.animate(run_time=1.6).set_value(COLD),
            m.Flash([xs[0], base + P_COLD[0] * scale + 0.3, 0], color=CAT, flash_radius=0.55),
        )
        self.say(
            "[Turn it up to five], [and the gaps shrink]: [forty, thirty-three, and twenty-seven percent.]",
            m.Indicate(tick_labels[2], color=PROB, scale_factor=1.4),
            T.animate(run_time=2.0).set_value(HOT),
            m.Indicate(ends[1], color=PROB, scale_factor=1.15),
        )
        self.say(
            "[A low temperature makes the model decisive]; [a high one makes it hedge its bets.]",
            m.FadeIn(moods[0], shift=0.2 * m.UP),
            m.FadeIn(moods[1], shift=0.2 * m.UP),
        )
        self.say(
            "[Chatbots turn this same knob when they pick their next word.] [More on that in episode thirty-three.]",
            T.animate(run_time=1.6).set_value(1.0),
            m.FadeIn(later, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        for mob in (live, knob, t_value):
            mob.clear_updaters()
        self.clear()

    # ------------------------------------------------------------------ cross-entropy

    def cross_entropy(self):
        self.next_section(
            "Cross-entropy",
            notes=f"L = −ln p(true). Cat: −ln {P[0]:.4f} = {LOSS0:.4f}. p = {SURE_RIGHT} → {-np.log(SURE_RIGHT):.3f}; "
            f"p = {SURE_WRONG} → {-np.log(SURE_WRONG):.3f}. Natural log. Averaged over a dataset this is the negative log-likelihood (Episode 15).",
        )
        xs, base, width, scale = (-5.55, -4.3, -3.05), -0.95, 0.75, 2.6
        bars = m.VGroup(*[bar_at(v * scale, c, width, x, base) for v, c, x in zip(P, COLORS, xs)])
        vals = m.VGroup(*[code(f"{v:.2f}", 24, INK, weight=m.SEMIBOLD).next_to(b, m.UP, buff=0.12) for v, b in zip(P, bars)])
        names = class_names(xs, base - 0.35, 22)
        column = m.VGroup(bars[0], vals[0], names[0])
        truth_box = m.RoundedRectangle(
            corner_radius=0.15, width=width + 0.5, height=column.height + 0.5, stroke_color=TRUTH, stroke_width=3,
        ).set_fill(opacity=0).move_to(column)
        truth = pill("the truth: cat", TRUTH, TRUTH_FILL, 22).next_to(truth_box, m.UP, buff=0.25)
        truth.align_to(truth_box, m.LEFT)
        formula = m.MathTex("L", "=", r"-\log", r"p_{\mathrm{correct}}", font_size=44, color=INK).move_to([-4.3, -2.25, 0])
        formula[0].set_color(LOSS)
        formula[3].set_color(PROB)
        worked = m.MathTex(r"-\log 0.66", r"\approx", f"{LOSS0:.2f}", font_size=40, color=INK).move_to([-4.3, -3.15, 0])
        worked[2].set_color(LOSS)

        ax = axes([0, 1, 0.25], [0, 5, 1], 5.3, 4.3)
        ax.shift(np.array([0.85, -2.35, 0]) - ax.c2p(0, 0))
        x_ticks = m.VGroup(*[text(s, 20, MUTED).next_to(ax.c2p(v, 0), m.DOWN, buff=0.15) for s, v in (("0", 0), ("0.5", 0.5), ("1", 1))])
        y_ticks = m.VGroup(*[text(s, 20, MUTED).next_to(ax.c2p(0, v), m.LEFT, buff=0.15) for s, v in (("2", 2), ("4", 4))])
        x_title = text("probability given to the right answer", 22, MUTED).move_to([ax.c2p(0.5, 0)[0], -3.25, 0])
        y_title = text("loss", 22, LOSS, weight=m.SEMIBOLD).next_to(ax.y_axis.get_end(), m.LEFT, buff=0.25)
        ps = np.geomspace(1.0, np.exp(-4.95), 160)  # drawn from "sure and right" towards zero
        curve = m.VMobject().set_points_smoothly([ax.c2p(p, -np.log(p)) for p in ps]).set_stroke(LOSS, width=5)

        def marker(p, color, words, at=None):
            """A dot on the curve; its label beside it, or at `at` with a thin leader line down to it."""
            point = ax.c2p(p, -np.log(p))
            dot = m.Dot(point, radius=0.11, color=color).set_z_index(3)
            if at is None:
                label = text(words, 22, color, weight=m.SEMIBOLD).next_to(dot, m.RIGHT, buff=0.2)
                return dot, label
            label = text(words, 22, color, weight=m.SEMIBOLD).move_to(at)
            leader = m.Line([point[0], label.get_bottom()[1] - 0.1, 0], point, buff=0.14).set_stroke(color, width=2, opacity=0.7)
            return dot, m.VGroup(leader, label)

        ours = marker(P[0], CAT, f"our cat: {LOSS0:.2f}", [ax.c2p(P[0], 0)[0], -1.25, 0])
        right = marker(SURE_RIGHT, TRUTH, f"sure and right: {-np.log(SURE_RIGHT):.2f}", [5.05, -0.55, 0])
        wrong = marker(SURE_WRONG, LOSS, f"sure and wrong: {-np.log(SURE_WRONG):.1f}")
        likelihood = pill("= maximum likelihood · Episode 15", PROB, PROB_FILL, 20).move_to([4.0, 0.55, 0])
        likelihood.shift(min(0.0, 6.7 - likelihood.get_right()[0]) * m.RIGHT)

        self.say(
            "[Now, how does the model learn?] [Say the picture really is a cat.]",
            m.AnimationGroup(
                self.new_header("CROSS-ENTROPY", "Scoring the guess"),
                m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.2), m.FadeIn(vals), m.FadeIn(names),
            ),
            m.AnimationGroup(m.Create(truth_box), m.FadeIn(truth, shift=0.2 * m.DOWN)),
        )
        self.say(
            "[Cross-entropy looks only at the probability given to the right answer], [and takes minus its log.]",
            m.Indicate(vals[0], color=PROB, scale_factor=1.3),
            m.Write(formula),
        )
        self.say(
            "[For our cat, that's minus the log of zero point six six]: [a loss of about zero point four two.]",
            m.FadeIn(m.VGroup(worked[0], worked[1]), shift=0.15 * m.UP),
            m.FadeIn(worked[2], scale=1.3),
        )
        self.say(
            "[Here's the whole curve.] [As the right answer's probability falls toward zero, the loss shoots up.]",
            m.AnimationGroup(m.Create(ax), m.FadeIn(x_ticks), m.FadeIn(y_ticks), m.FadeIn(x_title), m.FadeIn(y_title)),
            m.Create(curve, run_time=2.6),
        )
        self.say(
            "[Our cat sits down here.] [A confident, correct answer costs almost nothing.]",
            m.AnimationGroup(m.FadeIn(ours[0], scale=0.4), m.FadeIn(ours[1])),
            m.AnimationGroup(m.FadeIn(right[0], scale=0.4), m.FadeIn(right[1])),
        )
        self.say(
            "[But a confident wrong answer, giving the truth just one percent], [costs about four point six.]",
            m.FadeIn(wrong[0], scale=0.4),
            m.AnimationGroup(m.FadeIn(wrong[1], shift=0.2 * m.LEFT), m.Flash(wrong[0].get_center(), color=LOSS, flash_radius=0.4)),
        )
        self.say(
            "[Confident and wrong is punished hardest.] [Averaged over many pictures, minimising it is maximum likelihood, from episode fifteen.]",
            m.Wiggle(wrong[1]),
            m.FadeIn(likelihood, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the gradient

    def gradient(self):
        self.next_section(
            "The gradient",
            notes=f"dL/dz = p − y = {', '.join(signed(v, 4) for v in GRADIENT)} (checked by finite differences). "
            f"One step, learning rate {LR:g} (illustrative): z → {', '.join(f'{v:.3f}' for v in Z_NEW)}; "
            f"p(cat) {P[0]:.3f} → {P_NEW[0]:.3f}; loss {LOSS0:.3f} → {LOSS_NEW:.3f}.",
        )
        formula = m.MathTex(r"\frac{\partial L}{\partial z_i}", "=", "p_i", "-", "y_i", font_size=48, color=INK).move_to([-3.4, 1.95, 0])
        formula[0].set_color(GRAD)
        formula[2].set_color(PROB)
        formula[4].set_color(TRUTH)

        rows_y = (0.05, -0.85, -1.75)
        cols_x = (-4.05, -2.75, -1.05)
        row_icons = m.VGroup(*[icon(g, 0.55).move_to([-5.75, y, 0]) for g, y in zip(ICONS, rows_y)])
        heads = m.VGroup(tex("p", 40, PROB), tex("y", 40, TRUTH), tex("p - y", 40, GRAD))
        for h, x in zip(heads, cols_x):
            h.move_to([x, 0.8, 0])  # p and y both have descenders: centred, they share a baseline
        p_col = m.VGroup(*[code(f"{v:.2f}", 28, PROB, weight=m.SEMIBOLD).move_to([cols_x[0], y, 0]) for v, y in zip(P, rows_y)])
        y_col = m.VGroup(*[code(f"{v:.0f}", 28, TRUTH, weight=m.SEMIBOLD).move_to([cols_x[1], y, 0]) for v, y in zip(Y, rows_y)])
        g_col = m.VGroup(*[code(signed(v), 28, GRAD, weight=m.SEMIBOLD).move_to([cols_x[2], y, 0]) for v, y in zip(GRADIENT, rows_y)])
        one_hot = text("one-hot", 20, TRUTH, weight=m.SEMIBOLD).move_to([cols_x[1], -2.35, 0])

        xs, base, width, scale = (2.3, 3.75, 5.2), -1.45, 0.75, 0.95
        title = text("the scores z", 26, SCORE, weight=m.BOLD).move_to([3.75, 1.95, 0])
        bars = m.VGroup(*[bar_at(v * scale, c, width, x, base) for v, c, x in zip(Z, COLORS, xs)])
        vals = m.VGroup(*[code(num(v, 2), 24, INK, weight=m.SEMIBOLD).next_to(b, m.UP, buff=0.12) for v, b in zip(Z, bars)])
        names = class_names(xs, base - 0.35, 22)
        new_bars = m.VGroup(*[bar_at(v * scale, c, width, x, base) for v, c, x in zip(Z_NEW, COLORS, xs)])
        new_vals = m.VGroup(*[code(num(v, 2), 24, INK, weight=m.SEMIBOLD).next_to(b, m.UP, buff=0.12) for v, b in zip(Z_NEW, new_bars)])
        pushes = m.VGroup()
        for k, (b, x) in enumerate(zip(bars, xs)):
            top = b.get_top()[1]
            if k == 0:
                pushes.add(arrow([x + 0.68, top - 0.75, 0], [x + 0.68, top + 0.05, 0], TRUTH, buff=0))
            else:
                pushes.add(arrow([x + 0.68, top + 0.05, 0], [x + 0.68, top - 0.75, 0], LOSS, buff=0))
        descent = pill("gradient descent: step against the gradient", GRAD, GRAD_FILL, 20).move_to([-3.35, -3.15, 0])
        after = m.VGroup(
            text(f"p(cat):  {pct(P[0])}%  →  {pct(P_NEW[0])}%", 26, INK, weight=m.SEMIBOLD, t2c={f"{pct(P_NEW[0])}%": PROB}),
            text(f"loss:  {LOSS0:.2f}  →  {LOSS_NEW:.2f}", 26, INK, weight=m.SEMIBOLD, t2c={f"{LOSS_NEW:.2f}": LOSS}),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([3.75, -2.85, 0])

        self.say(
            "[Here's the neat part.] [How should each score change, to lower the loss?]",
            self.new_header("THE GRADIENT", "Predicted minus true"),
            m.AnimationGroup(m.FadeIn(title), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.2), m.FadeIn(vals), m.FadeIn(names)),
        )
        self.say(
            "[Put softmax and cross-entropy together], [and the gradient for each score is simply p minus y.]",
            m.FadeIn(m.VGroup(formula[0], formula[1]), shift=0.15 * m.UP),
            m.Write(m.VGroup(formula[2], formula[3], formula[4])),
        )
        self.say(
            "[p is the predicted probabilities.] [y is the truth, as a one-hot vector]: [one for cat, zero for the rest.]",
            m.AnimationGroup(m.Indicate(formula[2], color=PROB, scale_factor=1.3), m.FadeIn(row_icons), m.FadeIn(heads[0]), m.LaggedStart(*[m.FadeIn(v) for v in p_col], lag_ratio=0.2)),
            m.AnimationGroup(m.Indicate(formula[4], color=TRUTH, scale_factor=1.3), m.FadeIn(heads[1])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(v) for v in y_col], lag_ratio=0.2), m.FadeIn(one_hot)),
        )
        self.say(
            "[Subtract]: [cat gets minus zero point three four], [dog plus zero point two four, and bird plus zero point one.]",
            m.FadeIn(heads[2], shift=0.15 * m.DOWN),
            m.FadeIn(g_col[0], shift=0.15 * m.LEFT),
            m.LaggedStart(m.FadeIn(g_col[1], shift=0.15 * m.LEFT), m.FadeIn(g_col[2], shift=0.15 * m.LEFT), lag_ratio=0.6),
        )
        self.say(
            "[Gradient descent steps against the gradient]: [so the cat score goes up], [and the dog and bird scores go down.]",
            m.FadeIn(descent, shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(pushes[0]), m.Indicate(g_col[0], color=GRAD, scale_factor=1.2)),
            m.AnimationGroup(m.GrowArrow(pushes[1]), m.GrowArrow(pushes[2]), m.Indicate(m.VGroup(g_col[1], g_col[2]), color=GRAD, scale_factor=1.2)),
        )
        self.say(
            "[One step, with a learning rate of one, for example], [lifts cat to seventy-seven percent], [and cuts the loss from zero point four two to zero point two six.]",
            m.AnimationGroup(
                *[m.Transform(a, b) for a, b in zip(bars, new_bars)],
                *[m.FadeTransform(a, b) for a, b in zip(vals, new_vals)],
                m.FadeOut(pushes),
                run_time=1.4,
            ),
            m.FadeIn(after[0], shift=0.15 * m.UP),
            m.FadeIn(after[1], shift=0.15 * m.UP),
        )
        self.say(
            "[The push is just the error]: [the more wrong a prediction, the harder it pushes.]",
            m.Circumscribe(g_col, color=GRAD, buff=0.15),
            m.Indicate(g_col[0], color=GRAD, scale_factor=1.25),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section(
            "The big picture",
            notes="Language models end in a softmax over their vocabulary (tens of thousands of tokens or more) and are "
            "pretrained with cross-entropy on the next token (Episode 27). The 72 bars are seeded random logits, illustrative.",
        )
        left = card(3.9, 4.6, EDGE, CARD, stroke=2).move_to([-4.6, -0.25, 0])
        left_title = text("Our classifier", 28, INK, weight=m.BOLD).move_to(left.get_top() + 0.45 * m.DOWN)
        lx = (-5.65, -4.6, -3.55)
        lbase = left.get_bottom()[1] + 1.3
        lbars = m.VGroup(*[bar_at(v * 3.4, c, 0.62, x, lbase) for v, c, x in zip(P, COLORS, lx)])
        licons = m.VGroup(*[icon(g, 0.45).move_to([x, lbase - 0.35, 0]) for g, x in zip(ICONS, lx)])
        lnote = text("3 classes", 22, MUTED).move_to([left.get_x(), left.get_bottom()[1] + 0.3, 0])

        right = card(8.3, 4.6, EDGE, CARD, stroke=2).move_to([2.35, -0.25, 0])
        right_title = text("A language model", 28, INK, weight=m.BOLD).move_to(right.get_top() + 0.45 * m.DOWN)
        context = text("The cat sat on the …", 24, MUTED).next_to(right_title, m.DOWN, buff=0.15)
        rbase = right.get_bottom()[1] + 0.85
        n = len(VOCAB_P)
        step = 7.2 / n
        rx = [right.get_x() - 3.6 + step * (k + 0.5) for k in range(n)]
        rbars = m.VGroup(*[bar_at(v / VOCAB_P.max() * 2.2, PROB, step * 0.62, x, rbase, opacity=0.85) for v, x in zip(VOCAB_P, rx)])
        top = int(np.argmax(VOCAB_P))
        word = text("“mat”", 22, PROB, weight=m.SEMIBOLD).next_to(rbars[top], m.RIGHT, buff=0.15).align_to(rbars[top], m.UP)
        rnote = text("one bar for every word piece it knows: many thousands", 22, MUTED).move_to([right.get_x(), right.get_bottom()[1] + 0.3, 0])
        same = m.VGroup(
            pill("the same softmax", PROB, PROB_FILL, 22),
            pill("the same cross-entropy", LOSS, LOSS_FILL, 22),
        ).arrange(m.RIGHT, buff=0.4).move_to([0, -3.25, 0])

        self.say(
            "[Here's the bigger picture.] [Our classifier chose between three classes.]",
            self.new_header("THE BIG PICTURE", "Three classes, or a whole vocabulary"),
            m.AnimationGroup(m.FadeIn(left), m.FadeIn(left_title), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in lbars], lag_ratio=0.2), m.FadeIn(licons), m.FadeIn(lnote)),
        )
        self.say(
            "[A language model picks its next word from many thousands], [with the very same softmax], [trained by the very same cross-entropy.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(right_title), m.FadeIn(context), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in rbars], lag_ratio=0.04), m.FadeIn(word), m.FadeIn(rnote)),
            m.FadeIn(same[0], shift=0.2 * m.UP),
            m.FadeIn(same[1], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Logits, softmax, cross-entropy.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 52, c, weight=m.BOLD) for w, c in zip(("Logits", "Softmax", "Cross-entropy"), (SCORE, PROB, LOSS))]).arrange(m.RIGHT, buff=1.1)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Scores become probabilities, and the right answer gets pushed up.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [logits], [softmax], [cross-entropy.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Softmax turns scores into probabilities], [and cross-entropy pushes the right answer up.]", m.Write(tagline), m.Indicate(words[2], color=LOSS, scale_factor=1.06))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
