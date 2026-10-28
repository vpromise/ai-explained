"""AI, Explained · Episode 30 · RLHF & DPO

Running example: one prompt ("Explain recursion in one sentence.") and two answers, one preferred:
  A  "A function that solves a problem by calling itself on a smaller piece of it."  (preferred)
  B  "Recursion is when something is recursive."                                     (rejected)

Arc: picking the better of two is easy → preference data (prompt, preferred, rejected) → a reward
model and the Bradley–Terry probability σ(r_A − r_B) → PPO: maximise reward minus β·KL to the SFT
model (the trade-off, and reward hacking when the leash is loose) → DPO: the reward written with
the policy itself, so no reward model → the DPO loss, term by term → a worked example → side by
side → recap.

Colours, all film long: the prompt / people SKY · preferred GOOD · rejected BAD · the model being
trained VIOLET · the frozen SFT / reference model BLUE · the reward (model) AMBER · β and the KL
penalty, PPO PINK · DPO TEAL (also the level accent of every header).

Every number the narration quotes is computed below and asserted, so the scene cannot drift from
its words. All example numbers are illustrative (said "for example" in the narration):
  reward model:  r_A = 1.5, r_B = −0.5 → gap 2 → σ(2) = 0.881 ≈ 88%
  DPO, β = 0.1:  log π_ref(A) = −14, log π_ref(B) = −9; after training log π(A) = −10 (+4),
                 log π(B) = −15 (−6) → β·(4 − (−6)) = 1 → σ(1) = 0.731 → loss 0.313
                 (at the start π = π_ref → loss = log 2 = 0.693)
  PPO picture:   reward reachable at drift d (KL) r(d) = 2√d, penalty β·d → best drift 1/β²
                 (β 0.5 → 4, β 1 → 1, β 0.15 → as far as the plot goes); what people prefer
                 q(d) = 2√d − 0.06 d² peaks near d ≈ 4.1 and has collapsed by d = 10; ten "ratings"
                 drawn around it with seeded noise.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

BLUE_FILL = "#142040"

PROMPT = "Explain recursion in one sentence."
ANSWER_A = "A function that solves a problem by\ncalling itself on a smaller piece of it."
ANSWER_B = "Recursion is when something\nis recursive."


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------- the numbers

# the reward model's scores for A and B (illustrative)
R_A, R_B = 1.5, -0.5
GAP = R_A - R_B
P_A = float(sigmoid(GAP))
assert GAP == 2.0 and round(100 * P_A) == 88

# DPO, one comparison (illustrative sequence log-probabilities)
BETA = 0.1
REF_W, REF_L = -14.0, -9.0  # the frozen reference: log π_ref(A | x), log π_ref(B | x)
NEW_W, NEW_L = -10.0, -15.0  # our model after some training
RATIO_W, RATIO_L = NEW_W - REF_W, NEW_L - REF_L
Z = BETA * (RATIO_W - RATIO_L)
LOSS_START = float(-np.log(sigmoid(0.0)))
LOSS_AFTER = float(-np.log(sigmoid(Z)))
assert (RATIO_W, RATIO_L) == (4.0, -6.0) and RATIO_W - RATIO_L == 10.0 and abs(Z - 1.0) < 1e-12
assert abs(LOSS_START - np.log(2)) < 1e-12 and round(LOSS_START, 2) == 0.69
assert round(float(sigmoid(Z)), 2) == 0.73 and round(LOSS_AFTER, 2) == 0.31
assert round(float(-np.log(0.73)), 2) == 0.31  # the rounded arithmetic on screen

# PPO: the reward / penalty trade-off (illustrative curves)
D_MAX = 10.0
DS = np.linspace(0.0, D_MAX, 4001)


def reward_curve(d):
    return 2.0 * np.sqrt(d)


def quality(d):
    return 2.0 * np.sqrt(d) - 0.06 * np.asarray(d) ** 2


def best_drift(beta):
    return float(DS[np.argmax(reward_curve(DS) - beta * DS)])


BETA_MID, BETA_BIG, BETA_LOOSE = 0.5, 1.0, 0.15
assert abs(best_drift(BETA_MID) - 4.0) < 0.01 and abs(best_drift(BETA_BIG) - 1.0) < 0.01
assert best_drift(BETA_LOOSE) == D_MAX  # a loose leash: it drifts as far as it can
assert abs(float(DS[np.argmax(quality(DS))]) - 4.1) < 0.1  # people's preference peaks near β = 0.5's point
assert float(quality(D_MAX)) < 0.1 * float(reward_curve(D_MAX))  # ...and has collapsed at the far end
_rng = np.random.default_rng(30)
RATED_D = np.arange(0.5, D_MAX, 1.0)
RATED_Q = quality(RATED_D) + _rng.normal(0.0, 0.15, RATED_D.size)

PPO_MODELS = (("Policy", "being trained"), ("Reference", "frozen SFT copy"), ("Reward model", "gives the score"), ("Value model", "predicts the score"))
assert len(PPO_MODELS) == 4  # "four models in memory"


# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """kit.recolored(), but written under this episode's prefix: assets/ep30-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    stem = f"ep30-{name.removeprefix('ep30-')}-{color.lstrip('#').lower()}"
    path = ASSETS / f"{stem}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return stem


def model_icon(color, fill, height):
    return icon(variant("llm", color, fill), height)


def formula(*parts, size=40):
    """A MathTex from (latex, colour) pairs, one part each."""
    t = m.MathTex(*[p for p, _ in parts], font_size=size, color=INK)
    for part, (_, c) in zip(t, parts):
        part.set_color(c)
    return t


def cite(s, size=20):
    return text(s, size, MUTED)


def hnode(pic, title, sub, color, fill, width=3.4, height=1.5, title_size=26):
    """A node with its icon at the left and its words at the right."""
    box = card(width, height, color, fill)
    words = m.VGroup(text(title, title_size, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
    body = m.VGroup(pic, words).arrange(m.RIGHT, buff=0.3).move_to(box)
    return m.VGroup(box, body)


def mini(letter, words, color, fill, width=3.3, height=0.9, size=22):
    """A short answer: its letter in colour, then its words."""
    box = card(width, height, color, fill, radius=0.2, stroke=2.5)
    tag = text(letter, 28, color, weight=m.BOLD)
    body = text(words, size, INK)
    row = m.VGroup(tag, body).arrange(m.RIGHT, buff=0.28)
    row.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    return m.VGroup(box, tag, body)


def word_chip(word, color, fill, size=24, height=0.7, pad=0.5):
    """A chip as wide as its word."""
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def underline(mob, color, buff=0.12, width=4, y=None):
    y = (mob.get_bottom()[1] - buff) if y is None else y
    return m.Line([mob.get_left()[0], y, 0], [mob.get_right()[0], y, 0]).set_stroke(color, width=width)


def strike(mob, color=BAD, width=5):
    return m.Line(mob.get_corner(m.DL) + np.array([-0.08, 0.05, 0]), mob.get_corner(m.UR) + np.array([0.08, -0.05, 0])).set_stroke(color, width=width)


def bottom(label, color, fill, size=22, y=-3.25):
    return pill(label, color, fill, size).move_to([0, y, 0])


def panel(x, title, color, width=6.3, height=4.6, y=-0.35):
    box = card(width, height, color, CARD, stroke=2).move_to([x, y, 0]).set_z_index(-1)
    head = text(title, 28, color, weight=m.BOLD)
    head.move_to(box.get_corner(m.UL) + np.array([0.4 + head.width / 2, -0.5, 0]))
    return m.VGroup(box, head)


def bullet(words, color, size=24):
    dot = m.Dot(radius=0.07, color=color)
    return m.VGroup(dot, text(words, size, INK)).arrange(m.RIGHT, buff=0.25)


# ---------------------------------------------------------------------------- the episode


class Ep30RLHFDPO(Episode):
    number = 30

    def lesson(self):
        self.hook()
        self.preferences()
        self.reward_model()
        self.ppo_loop()
        self.ppo_tradeoff()
        self.dpo_idea()
        self.dpo_loss()
        self.worked_example()
        self.side_by_side()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="One prompt, two answers. Choosing the better of two is easy; that choice is the training signal.")
        person = icon("user", 0.8)
        ask = bubble(text(PROMPT, 30, INK, weight=m.SEMIBOLD), SKY, SKY_FILL, tail="left")
        prompt = m.VGroup(person, ask).arrange(m.RIGHT, buff=0.35).move_to([0, 2.3, 0])

        cards = []
        for letter, words, x in (("A", ANSWER_A, -3.45), ("B", ANSWER_B, 3.45)):
            box = card(6.3, 2.3, EDGE, CARD).move_to([x, -0.15, 0])
            tag = pill(f"Answer {letter}", MUTED, CARD, 20)
            tag.move_to(box.get_corner(m.UL) + np.array([0.35 + tag.width / 2, -0.32 - tag.height / 2, 0]))
            body = text(words, 26, INK)
            body.next_to(tag, m.DOWN, buff=0.28).align_to(tag, m.LEFT)
            cards.append((box, tag, body))
        (a_box, a_tag, a_body), (b_box, b_tag, b_body) = cards
        tick = icon("check", 0.62).move_to(a_box.get_corner(m.UR) + np.array([-0.5, -0.5, 0]))
        cross = icon("ep30-cross", 0.62).move_to(b_box.get_corner(m.UR) + np.array([-0.5, -0.5, 0]))

        hard = pill("writing a perfect answer: hard", MUTED, CARD, 22)
        easy = pill("picking the better of two: easy", AMBER, AMBER_FILL, 22)
        m.VGroup(hard, easy).arrange(m.RIGHT, buff=0.5).move_to([0, -2.2, 0])
        first = m.VGroup(text("first", 24, MUTED), pill("RLHF", PINK, PINK_FILL, 24)).arrange(m.RIGHT, buff=0.25)
        then = m.VGroup(text("then a shortcut", 24, MUTED), pill("DPO", TEAL, TEAL_FILL, 24)).arrange(m.RIGHT, buff=0.25)
        m.VGroup(first, then).arrange(m.RIGHT, buff=0.8).move_to([0, -3.3, 0])

        self.say(
            "So... [ask a model to explain recursion in one sentence.]",
            m.AnimationGroup(m.FadeIn(person, shift=0.2 * m.UP), m.FadeIn(ask, shift=0.2 * m.UP)),
        )
        self.say(
            "[It might say]: [a function that solves a problem by calling itself on a smaller piece of it.]",
            m.AnimationGroup(m.FadeIn(a_box), m.FadeIn(a_tag)),
            m.AddTextLetterByLetter(a_body, time_per_char=0.03),
        )
        self.say(
            "[Or it might say]: [recursion is when something is recursive.]",
            m.AnimationGroup(m.FadeIn(b_box), m.FadeIn(b_tag)),
            m.AddTextLetterByLetter(b_body, time_per_char=0.03),
        )
        self.say(
            "[Which is better? You knew at once.] [Writing a perfect answer is hard], [but picking the better of two is easy.]",
            m.AnimationGroup(
                a_box.animate.set_stroke(GOOD), m.FadeIn(tick, scale=1.3),
                b_box.animate.set_stroke(BAD), m.FadeIn(cross, scale=1.3),
            ),
            m.FadeIn(hard, shift=0.2 * m.UP),
            m.FadeIn(easy, shift=0.2 * m.UP),
        )
        self.say(
            "[Today: how a model learns from that one choice], [first with RLHF], [then with a shortcut called DPO.]",
            m.Indicate(easy, color=AMBER, scale_factor=1.08),
            m.FadeIn(first, shift=0.2 * m.UP),
            m.FadeIn(then, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ preference data

    def preferences(self):
        self.next_section(
            "1 · Preference data",
            notes="Start from the SFT model (Episode 29). It writes two answers per prompt; a person picks one. "
            "Each comparison is a triple (x, y_w, y_l). Real datasets hold thousands of them or more "
            "(InstructGPT's reward model used about 33k prompts, with rankings of 4–9 answers each: Ouyang et al., 2022).",
        )
        sft = node(variant("llm", BLUE, BLUE_FILL), "SFT model", "Episode 29", BLUE, BLUE_FILL, width=2.3, height=2.3).move_to([-5.55, 0.0, 0])
        a = mini("A", "calls itself on a smaller piece", GOOD, CARD, width=5.0, height=0.95).move_to([-0.85, 0.75, 0])
        b = mini("B", "when something is recursive", BAD, CARD, width=5.0, height=0.95).move_to([-0.85, -0.75, 0])
        a[0].set_stroke(EDGE)
        b[0].set_stroke(EDGE)
        links = m.VGroup(
            arrow(sft.get_right() + 0.35 * m.UP, a.get_left(), MUTED, buff=0.12),
            arrow(sft.get_right() + 0.35 * m.DOWN, b.get_left(), MUTED, buff=0.12),
        )
        tick = icon("check", 0.5).next_to(a, m.RIGHT, buff=0.22)
        cross = icon("ep30-cross", 0.5).next_to(b, m.RIGHT, buff=0.22)
        judge = icon("user", 0.75).move_to([-1.7, 2.0, 0])
        judge_note = text("a person picks the one they prefer", 22, SKY, weight=m.SEMIBOLD).next_to(judge, m.RIGHT, buff=0.25)

        box = card(3.7, 3.0, EDGE, CARD).move_to([4.75, 0.0, 0])
        title = text("one comparison", 22, MUTED, weight=m.SEMIBOLD).move_to(box.get_top() + 0.42 * m.DOWN)
        rows = m.VGroup(
            m.VGroup(tex("x", 36, SKY), text("the prompt", 22, INK)),
            m.VGroup(tex("y_w", 36, GOOD), text("preferred: A", 22, INK)),
            m.VGroup(tex("y_l", 36, BAD), text("rejected: B", 22, INK)),
        )
        for k, row in enumerate(rows):
            y = box.get_top()[1] - 1.05 - 0.62 * k
            row[0].move_to([box.get_left()[0] + 0.65, y, 0])
            row[1].move_to([0, y, 0]).align_to([box.get_left()[0] + 1.15, 0, 0], m.LEFT)
        to_record = arrow([tick.get_x() - 0.1, 0, 0], box.get_left(), MUTED, buff=0.12)
        shadows = m.VGroup(
            *[card(3.7, 3.0, EDGE, CARD, stroke=2).move_to(box.get_center() + k * np.array([0.12, 0.14, 0])).set_z_index(-2 - k) for k in (1, 2)]
        )
        many = text("× thousands of prompts", 22, MUTED, weight=m.SEMIBOLD).next_to(box, m.DOWN, buff=0.35)
        dataset = bottom("a preference dataset", SKY, SKY_FILL, 24, y=-3.0)

        self.say(
            "[We start from a fine-tuned assistant]: [the SFT model, from episode twenty-nine.]",
            m.AnimationGroup(self.new_header("1 · PREFERENCE DATA", "People pick the better answer"), m.FadeIn(sft, shift=0.2 * m.UP)),
            m.Circumscribe(sft, color=BLUE, buff=0.12),
        )
        self.say(
            "[For each prompt, it writes two answers], [and a person picks the one they prefer.]",
            m.AnimationGroup(m.GrowArrow(links[0]), m.GrowArrow(links[1]), m.FadeIn(a, shift=0.2 * m.RIGHT), m.FadeIn(b, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(
                m.FadeIn(judge, shift=0.2 * m.UP), m.FadeIn(judge_note),
                m.Succession(m.Wait(0.3), m.AnimationGroup(m.FadeIn(tick, scale=1.3), a[0].animate.set_stroke(GOOD), m.FadeIn(cross, scale=1.3), b[0].animate.set_stroke(BAD))),
            ),
        )
        self.say(
            "[That's one comparison]: [the prompt], [the preferred answer], [and the rejected one.]",
            m.AnimationGroup(m.GrowArrow(to_record), m.FadeIn(box), m.FadeIn(title)),
            *[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows],
        )
        self.say(
            "[Collect thousands of these], [and you have a preference dataset.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.UP) for s in shadows], lag_ratio=0.3), m.FadeIn(many, shift=0.15 * m.UP)),
            m.FadeIn(dataset, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the reward model

    def reward_model(self):
        self.next_section(
            "2 · Reward model",
            notes="A reward model maps (prompt, answer) to one score. Bradley–Terry (Bradley & Terry, 1952): "
            "P(A ≻ B) = σ(r_A − r_B). Example scores 1.5 and −0.5 are illustrative: gap 2 → σ(2) ≈ 0.88. "
            "Training maximises the probability of the choices people made (loss −log σ(r_w − r_l)).",
        )
        gap = m.ValueTracker(GAP)
        a = word_chip("answer A", GOOD, GOOD_FILL).move_to([-5.3, 2.15, 0])
        b = word_chip("answer B", BAD, BAD_FILL).move_to([-2.5, 2.15, 0])
        rm = hnode(icon("ep30-gauge", 0.95), "Reward model", "prompt + answer → a score", AMBER, AMBER_FILL, width=5.4, height=1.35).move_to([-3.9, 0.6, 0])
        ins = m.VGroup(
            arrow(a.get_bottom(), [a.get_x(), rm.get_top()[1], 0], MUTED, buff=0.1),
            arrow(b.get_bottom(), [b.get_x(), rm.get_top()[1], 0], MUTED, buff=0.1),
        )
        slots = {"A": [-5.3, -0.95, 0], "B": [-2.5, -0.95, 0]}
        outs = m.VGroup(
            arrow([slots["A"][0], rm.get_bottom()[1], 0], [slots["A"][0], -0.55, 0], AMBER, buff=0.1),
            arrow([slots["B"][0], rm.get_bottom()[1], 0], [slots["B"][0], -0.55, 0], AMBER, buff=0.1),
        )

        def score(name, color, value):
            def make():
                body = m.VGroup(tex(f"r_{name} =", 34, color), text(f"{value():.1f}".replace("-", "−"), 30, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18)
                return body.move_to(slots[name])

            return m.always_redraw(make)

        score_a = score("A", GOOD, lambda: 0.5 + gap.get_value() / 2)
        score_b = score("B", BAD, lambda: 0.5 - gap.get_value() / 2)
        assert (0.5 + GAP / 2, 0.5 - GAP / 2) == (R_A, R_B)

        bt = formula((r"P(A \succ B)", INK), ("=", INK), (r"\sigma(", INK), ("r_A", GOOD), ("-", INK), ("r_B", BAD), (")", INK), size=46).move_to([-3.9, -2.25, 0])
        bt_cite = cite("Bradley & Terry, 1952").next_to(bt, m.DOWN, buff=0.25)

        ax = axes([-4, 4, 1], [0, 1.15, 0.5], 5.4, 3.6).move_to([3.85, -0.45, 0])
        x_label = text("score gap", 20, MUTED).next_to(ax.x_axis, m.DOWN, buff=0.62)
        y_label = text("chance A is preferred", 20, MUTED).next_to(ax.y_axis.get_end(), m.UP, buff=0.15)
        curve = ax.plot(lambda z: float(sigmoid(z)), x_range=[-4, 4]).set_stroke(AMBER, width=4)

        def make_dot():
            g = gap.get_value()
            return m.Dot(ax.c2p(g, float(sigmoid(g))), radius=0.1, color=GOOD).set_z_index(3)

        def make_guides():
            g = gap.get_value()
            p = ax.c2p(g, float(sigmoid(g)))
            return m.VGroup(
                m.DashedLine(ax.c2p(0, float(sigmoid(g))), p, dash_length=0.08).set_stroke(GOOD, width=2.5),
                m.DashedLine(ax.c2p(g, 0), p, dash_length=0.08).set_stroke(GOOD, width=2.5),
            )

        def make_pct():
            g = gap.get_value()
            return text(f"{100 * float(sigmoid(g)):.0f}%", 24, GOOD, weight=m.BOLD).next_to(ax.c2p(0, float(sigmoid(g))), m.LEFT, buff=0.15)

        def make_gap():
            g = gap.get_value()
            return text(f"{g:.1f}", 22, GOOD, weight=m.BOLD).next_to(ax.c2p(g, 0), m.DOWN, buff=0.15)

        dot, guides, pct, gap_label = (m.always_redraw(f) for f in (make_dot, make_guides, make_pct, make_gap))
        up = m.VGroup(
            arrow([slots["A"][0] - 1.35, -1.3, 0], [slots["A"][0] - 1.35, -0.6, 0], GOOD, buff=0),
            arrow([slots["B"][0] + 1.4, -0.6, 0], [slots["B"][0] + 1.4, -1.3, 0], BAD, buff=0),
        )

        self.say(
            "[Classic RLHF first trains a reward model.]",
            m.AnimationGroup(self.new_header("2 · REWARD MODEL", "Turn preferences into a score"), m.FadeIn(rm, shift=0.2 * m.UP)),
        )
        self.say(
            "[It reads a prompt and an answer], [and returns one number: a score.]",
            m.AnimationGroup(m.FadeIn(a, shift=0.2 * m.DOWN), m.FadeIn(b, shift=0.2 * m.DOWN), m.GrowArrow(ins[0]), m.GrowArrow(ins[1])),
            m.AnimationGroup(m.GrowArrow(outs[0]), m.GrowArrow(outs[1]), m.Indicate(rm[1][0], color=AMBER, scale_factor=1.15)),
        )
        self.say(
            "[For example, answer A scores one point five], [and answer B, minus zero point five.]",
            m.FadeIn(score_a, shift=0.2 * m.DOWN),
            m.FadeIn(score_b, shift=0.2 * m.DOWN),
        )
        self.say(
            "[The Bradley-Terry model turns the gap into a probability]: [the sigmoid of the difference in scores.]",
            m.AnimationGroup(m.Write(bt), m.FadeIn(bt_cite)),
            m.AnimationGroup(m.Create(ax), m.FadeIn(x_label), m.FadeIn(y_label), m.Create(curve)),
        )
        self.say(
            "[Here the gap is two], [so A is preferred with a probability of about eighty-eight percent.]",
            m.AnimationGroup(m.FadeIn(gap_label), m.FadeIn(dot, scale=0.5), m.Indicate(bt[3:6], color=AMBER, scale_factor=1.1)),
            m.AnimationGroup(m.Create(guides), m.FadeIn(pct, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[Training makes this probability high on every pair people judged], [pushing preferred scores up and rejected ones down.]",
            m.Circumscribe(bt, color=AMBER, buff=0.12),
            m.AnimationGroup(gap.animate(run_time=2.0).set_value(2.9), m.GrowArrow(up[0]), m.GrowArrow(up[1])),
        )
        self.wait(0.3)
        self.stop(score_a, score_b, dot, guides, pct, gap_label)
        self.clear()

    # ------------------------------------------------------------------ PPO: the loop

    def ppo_loop(self):
        self.next_section(
            "3 · PPO: the loop",
            notes="The policy (initialised from the SFT model) writes answers, the frozen reward model scores them, "
            "PPO (Schulman et al., 2017) updates the policy toward higher scores; RLHF recipe as in InstructGPT "
            "(Ouyang et al., 2022; earlier: Christiano et al., 2017). The flattering answer and its score 9.7 are illustrative.",
        )
        policy = hnode(icon(variant("llm", VIOLET, VIOLET_FILL), 0.8), "Policy", "starts as the SFT model", VIOLET, VIOLET_FILL, width=4.25).move_to([-4.67, 1.0, 0])
        rm = hnode(icon("ep30-gauge", 0.8), "Reward model", "trained in step two", AMBER, AMBER_FILL, width=4.25).move_to([4.67, 1.0, 0])
        ans_box = card(3.4, 1.5, EDGE, CARD).move_to([0, 1.0, 0])
        good = text("A function that calls\nitself on a smaller piece.", 20, INK)
        odd = text("Great question!!!\nRecursion is so amazing!!!", 20, INK)
        for t in (good, odd):
            t.move_to(ans_box)
        links = m.VGroup(arrow(policy.get_right(), ans_box.get_left(), MUTED), arrow(ans_box.get_right(), rm.get_left(), MUTED))
        score = word_chip("score 1.5", AMBER, AMBER_FILL).next_to(rm, m.DOWN, buff=0.35)
        score_high = word_chip("score 9.7", AMBER, AMBER_FILL).move_to(score)
        back = m.CurvedArrow(score.get_left() + 0.1 * m.LEFT, policy.get_bottom() + 0.1 * m.DOWN, angle=-0.55)
        back.set_stroke(PINK, width=5).set_color(PINK)
        back_label = text("PPO: make high-scoring answers more likely", 22, PINK, weight=m.SEMIBOLD).move_to([0, -1.75, 0])
        who = cite("PPO: Schulman et al., 2017  ·  RLHF: Ouyang et al., 2022").move_to([0, -3.35, 0])
        fooled = bottom("✗  a high score for a useless answer", BAD, BAD_FILL, 22, y=-2.55)

        self.say(
            "[Next comes reinforcement learning]: [the model writes an answer], [and the reward model scores it.]",
            m.AnimationGroup(self.new_header("3 · PPO", "Chase the reward"), m.FadeIn(policy, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(ans_box), m.AddTextLetterByLetter(good, time_per_char=0.02)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(rm, shift=0.2 * m.UP), m.FadeIn(score, shift=0.2 * m.DOWN)),
        )
        self.say(
            "[An algorithm called PPO then nudges the model toward answers that score higher.]",
            m.AnimationGroup(m.Create(back), m.FadeIn(back_label), m.FadeIn(who), m.Succession(m.Wait(0.8), m.Indicate(policy, color=VIOLET, scale_factor=1.05))),
        )
        self.say(
            "[But chasing the score alone is risky]: [the model can drift into odd text that the reward model happens to love.]",
            m.Wiggle(policy, scale_value=1.04),
            m.AnimationGroup(
                m.FadeOut(good), m.FadeIn(odd), ans_box.animate.set_stroke(BAD),
                m.FadeOut(score), m.FadeIn(score_high, scale=1.3), m.FadeIn(fooled, shift=0.2 * m.UP),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ PPO: the trade-off

    def ppo_tradeoff(self):
        self.next_section(
            "3 · PPO: stay close",
            notes="Objective: maximise E[r] − β·KL(π ‖ π_SFT). Illustrative curves: reachable reward 2√KL, penalty β·KL, "
            "best drift 1/β² (β 0.5 → 4, β 1 → 1); 'what people prefer' rises then falls (dots: seeded noise around "
            "2√d − 0.06d²), the shape reported for reward-model over-optimisation (Gao et al., 2022). "
            "Reward hacking: Episode 46. Four models in PPO-style RLHF: policy, reference, reward model, value model.",
        )
        beta = m.ValueTracker(BETA_MID)
        obj = formula(
            (r"\max_{\pi}\;", INK), (r"\mathbb{E}\left[\, r(x, y) \,\right]", AMBER), ("-", INK),
            (r"\beta", PINK), (r"\,\mathrm{KL}\left(\pi \,\|\, \pi_{\text{SFT}}\right)", PINK), size=44,
        ).move_to([-1.2, 1.95, 0])
        r_note = text("the reward", 20, AMBER, weight=m.SEMIBOLD).next_to(obj[1], m.DOWN, buff=0.2)
        k_note = text("a penalty for drifting", 20, PINK, weight=m.SEMIBOLD).next_to(obj[3:], m.DOWN, buff=0.2)

        ax = axes([0, D_MAX, 1], [0, 7, 1], 7.4, 3.3).move_to([-2.25, -1.3, 0])
        x_label = text("drift from the SFT model (KL)", 20, MUTED).next_to(ax.x_axis, m.DOWN, buff=0.22)
        reward = ax.plot(lambda d: float(reward_curve(d)), x_range=[0, D_MAX]).set_stroke(AMBER, width=4)

        def make_penalty():
            b = beta.get_value()
            end = min(D_MAX, 7.0 / b)
            return m.Line(ax.c2p(0, 0), ax.c2p(end, b * end)).set_stroke(PINK, width=4)

        def make_best():
            d = best_drift(beta.get_value())
            top, low = ax.c2p(d, float(reward_curve(d))), ax.c2p(d, beta.get_value() * d)
            return m.VGroup(
                m.Line(low, top).set_stroke(GOOD, width=6),
                m.Dot(top, radius=0.1, color=GOOD).set_z_index(3),
            )

        def make_beta():
            return text(f"β = {beta.get_value():.2f}", 26, PINK, weight=m.BOLD).move_to([4.75, -2.55, 0])

        penalty, best, beta_label = m.always_redraw(make_penalty), m.always_redraw(make_best), m.always_redraw(make_beta)

        def key(swatch, words, color, y):
            swatch.move_to([3.15, y, 0])
            label = text(words, 22, color, weight=m.SEMIBOLD).next_to(swatch, m.RIGHT, buff=0.25)
            return m.VGroup(swatch, label)

        keys = m.VGroup(
            key(m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(AMBER, width=4), "reward model score", AMBER, 0.15),
            key(m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(PINK, width=4), "penalty: β × KL", PINK, -0.45),
            key(m.Line([0, -0.18, 0], [0, 0.18, 0]).set_stroke(GOOD, width=6), "best balance", GOOD, -1.05),
            key(m.Dot(radius=0.08, color=SKY), "what people prefer", SKY, -1.65),
        )
        for k in keys:
            k[1].align_to([3.65, 0, 0], m.LEFT)
        rated = m.VGroup(*[m.Dot(ax.c2p(d, max(q, 0.05)), radius=0.075, color=SKY) for d, q in zip(RATED_D, RATED_Q)]).set_z_index(2)
        trend = m.DashedVMobject(ax.plot(lambda d: float(quality(d)), x_range=[0, D_MAX]).set_stroke(SKY, width=3), num_dashes=40)
        hack_top, hack_low = ax.c2p(D_MAX, float(reward_curve(D_MAX))), ax.c2p(D_MAX, float(quality(D_MAX)))
        hack = m.DoubleArrow(hack_low, hack_top, buff=0.12, color=BAD, stroke_width=4, max_tip_length_to_length_ratio=0.08).shift(0.3 * m.RIGHT)
        hack_label = pill("reward hacking · Episode 46", BAD, BAD_FILL, 22).move_to([4.55, 1.0, 0])

        self.say(
            "[So the objective adds a penalty]: [beta times the KL divergence], [a measure of how far the model has drifted from the SFT model.]",
            m.AnimationGroup(self.new_header("3 · PPO", "Maximise the reward, but stay close"), m.Write(obj[:3]), m.FadeIn(r_note)),
            m.Write(obj[3:]),
            m.FadeIn(k_note, shift=0.15 * m.UP),
        )
        self.say(
            "[Here's the trade-off, for example.] [Drifting further earns more reward], [but the penalty grows too.]",
            m.AnimationGroup(m.Create(ax), m.FadeIn(x_label)),
            m.AnimationGroup(m.Create(reward), m.FadeIn(keys[0])),
            m.AnimationGroup(m.Create(penalty), m.FadeIn(keys[1]), m.FadeIn(beta_label)),
        )
        self.say(
            "[The best point is where reward beats the penalty by the most.] [A bigger beta keeps the model closer to where it started.]",
            m.AnimationGroup(m.FadeIn(best), m.FadeIn(keys[2])),
            beta.animate(run_time=2.0).set_value(BETA_BIG),
        )
        self.say(
            "[Loosen the leash], [and the model drifts as far as it can: its score keeps climbing], [while what people actually prefer falls.]",
            beta.animate(run_time=1.6).set_value(BETA_LOOSE),
            m.Indicate(keys[0], color=AMBER, scale_factor=1.06),
            m.AnimationGroup(m.FadeIn(keys[3]), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in rated], lag_ratio=0.12), m.Create(trend, run_time=2.0)),
        )
        self.say(
            "[That gap is called reward hacking], and we'll come back to it in episode forty-six.",
            m.AnimationGroup(m.GrowFromCenter(hack), m.FadeIn(hack_label, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.stop(penalty, best, beta_label)
        self.clear()

        nodes = m.VGroup()
        pics = (
            icon(variant("llm", VIOLET, VIOLET_FILL), 0.8),
            icon(variant("llm", BLUE, BLUE_FILL), 0.8),
            icon("ep30-gauge", 0.8),
            icon(variant("gear", PINK, PINK_FILL), 0.8),
        )
        for (title, sub), pic, (color, fill) in zip(PPO_MODELS, pics, ((VIOLET, VIOLET_FILL), (BLUE, BLUE_FILL), (AMBER, AMBER_FILL), (PINK, PINK_FILL))):
            box = card(3.0, 2.3, color, fill)
            body = m.VGroup(pic, text(title, 26, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.14).move_to(box)
            nodes.add(m.VGroup(box, body))
        nodes.arrange(m.RIGHT, buff=0.35).move_to([0, 0.0, 0])
        count = text("4 models in memory", 30, INK, weight=m.BOLD).next_to(nodes, m.UP, buff=0.5)
        knobs = bottom("…and many settings to tune", MUTED, CARD, 22, y=-2.3)

        self.say(
            "[PPO works, but it's heavy]: [four models in memory], [and many settings to tune.]",
            m.FadeIn(count, shift=0.2 * m.UP),
            m.LaggedStart(*[m.FadeIn(n, shift=0.2 * m.UP) for n in nodes], lag_ratio=0.2),
            m.FadeIn(knobs, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ DPO: the idea

    def dpo_idea(self):
        self.next_section(
            "4 · DPO: the idea",
            notes="Direct Preference Optimization (Rafailov et al., 2023, Stanford; 'Your Language Model is Secretly a "
            "Reward Model'). The KL-regularised objective has the closed-form optimum π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β), "
            "so r(x,y) = β log π*(y|x)/π_ref(y|x) + β log Z(x). In Bradley–Terry only reward differences for the same "
            "prompt appear, so β log Z(x) cancels.",
        )
        y1, y2 = 1.5, 0.1
        lab_rlhf = text("RLHF", 26, PINK, weight=m.BOLD).move_to([0, y1, 0]).align_to([-6.75, 0, 0], m.LEFT)
        lab_dpo = text("DPO", 26, TEAL, weight=m.BOLD).move_to([0, y2, 0]).align_to([-6.75, 0, 0], m.LEFT)
        prefs1 = word_chip("preference pairs", SKY, SKY_FILL).move_to([-3.85, y1, 0])
        rmc = word_chip("reward model", AMBER, AMBER_FILL).move_to([-0.75, y1, 0])
        ppoc = word_chip("PPO loop", PINK, PINK_FILL).move_to([1.95, y1, 0])
        tuned1 = word_chip("tuned model", VIOLET, VIOLET_FILL).move_to([5.1, y1, 0])
        row1 = m.VGroup(prefs1, rmc, ppoc, tuned1)
        links1 = m.VGroup(*[arrow(p.get_right(), q.get_left(), MUTED, buff=0.1) for p, q in zip(row1, row1[1:])])
        prefs2 = word_chip("preference pairs", SKY, SKY_FILL).move_to([-3.85, y2, 0])
        tuned2 = word_chip("tuned model", VIOLET, VIOLET_FILL).move_to([5.1, y2, 0])
        direct = arrow(prefs2.get_right(), tuned2.get_left(), TEAL, buff=0.1)
        direct_note = text("one simple loss", 20, TEAL, weight=m.SEMIBOLD).next_to(direct, m.UP, buff=0.12)
        who = cite("Rafailov et al., 2023").next_to(direct, m.DOWN, buff=0.15)

        implicit = formula(
            (r"\hat{r}(x, y)", AMBER), ("=", INK), (r"\beta", PINK),
            (r"\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}", VIOLET), ("+", INK), (r"\beta \log Z(x)", MUTED), size=46,
        ).move_to([0, -1.65, 0])
        note_y = implicit.get_bottom()[1] - 0.32
        ratio_note = text("our model vs reference", 20, VIOLET, weight=m.SEMIBOLD).move_to([implicit[3].get_x(), note_y, 0])
        z_note = text("depends only on the prompt", 20, MUTED, weight=m.SEMIBOLD).move_to([implicit[5].get_x(), note_y, 0])
        z_note.align_to(implicit[5], m.LEFT)
        z_gone = text("cancels in a comparison", 20, BAD, weight=m.SEMIBOLD).move_to(z_note).align_to(z_note, m.LEFT)
        z_strike = strike(implicit[4:])
        skipped = m.VGroup(rmc, links1[1], ppoc)
        skip_box = m.DashedVMobject(
            m.RoundedRectangle(corner_radius=0.2, width=skipped.width + 0.4, height=skipped.height + 0.36).set_stroke(BAD, width=3),
            num_dashes=44,
        ).move_to(skipped)
        skip_note = text("not needed", 20, BAD, weight=m.BOLD).next_to(skip_box, m.UP, buff=0.1)

        self.say(
            "[In twenty twenty-three, a team at Stanford found a shortcut]: [direct preference optimization, or DPO.]",
            m.AnimationGroup(
                self.new_header("4 · DPO", "Skip the reward model"),
                m.FadeIn(lab_rlhf), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in row1], lag_ratio=0.2),
                m.LaggedStart(*[m.GrowArrow(a) for a in links1], lag_ratio=0.2),
            ),
            m.AnimationGroup(m.FadeIn(lab_dpo), m.FadeIn(prefs2), m.GrowArrow(direct), m.FadeIn(tuned2), m.FadeIn(direct_note), m.FadeIn(who)),
        )
        self.say(
            "[That penalised objective has an exact best solution], [so the reward can be rewritten using the model itself.]",
            m.Circumscribe(m.VGroup(rmc, ppoc), color=PINK, buff=0.12),
            m.Write(implicit),
        )
        self.say(
            "[It's beta times the log of how much more likely our model makes the answer than the reference does], [plus a term that depends only on the prompt.]",
            m.AnimationGroup(m.Indicate(implicit[2:4], color=VIOLET, scale_factor=1.06), m.FadeIn(ratio_note, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Indicate(implicit[5], color=MUTED, scale_factor=1.1), m.FadeIn(z_note, shift=0.15 * m.UP)),
        )
        self.say(
            "[In a comparison, that term cancels]: [so no reward model], [and no reinforcement learning loop.]",
            m.AnimationGroup(m.Create(z_strike), m.FadeTransform(z_note, z_gone)),
            m.AnimationGroup(m.Create(skip_box), m.FadeIn(skip_note), rmc.animate.set_opacity(0.35)),
            m.AnimationGroup(ppoc.animate.set_opacity(0.35), links1[1].animate.set_opacity(0.35), pulse(direct, TEAL)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ DPO: the loss

    def dpo_loss(self):
        self.next_section(
            "5 · The DPO loss",
            notes="L_DPO = −log σ(β[log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)]) for one comparison "
            "(averaged over the dataset). β: strength of the implicit KL leash (commonly around 0.1).",
        )
        loss = formula(
            (r"\mathcal{L}_{\text{DPO}} =", INK), (r"-\log\sigma\Big(", INK), (r"\beta", PINK), (r"\Big[", INK),
            (r"\log\frac{\pi(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}", GOOD), ("-", INK),
            (r"\log\frac{\pi(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}", BAD), (r"\Big]\Big)", INK), size=46,
        ).move_to([0, 1.25, 0])
        line_y = loss.get_bottom()[1] - 0.14
        good_line, bad_line, sig_line = (underline(loss[k], c, y=line_y) for k, c in ((4, GOOD), (6, BAD), (1, INK)))
        good_note = m.VGroup(text("preferred answer", 22, GOOD, weight=m.BOLD), text("our model vs reference", 20, MUTED)).arrange(m.DOWN, buff=0.08)
        bad_note = m.VGroup(text("rejected answer", 22, BAD, weight=m.BOLD), text("our model vs reference", 20, MUTED)).arrange(m.DOWN, buff=0.08)
        sig_note = m.VGroup(text("gap → loss", 22, INK, weight=m.BOLD), text("Bradley–Terry", 20, MUTED)).arrange(m.DOWN, buff=0.08)
        good_note.next_to(good_line, m.DOWN, buff=0.2)
        bad_note.next_to(bad_line, m.DOWN, buff=0.2)
        sig_note.next_to(sig_line, m.DOWN, buff=0.2)
        beta_note = text("β: how far we may stray", 22, PINK, weight=m.BOLD)
        beta_note.move_to([loss[2].get_x(), loss.get_top()[1] + 0.35, 0])

        a = mini("A", "calls itself on a smaller piece", GOOD, GOOD_FILL, width=5.0, height=0.95).move_to([-3.3, -2.45, 0])
        b = mini("B", "when something is recursive", BAD, BAD_FILL, width=5.0, height=0.95).move_to([3.3, -2.45, 0])
        a_up = m.VGroup(arrow([0, -0.35, 0], [0, 0.35, 0], GOOD, buff=0), text("more likely", 22, GOOD, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15)
        b_down = m.VGroup(arrow([0, 0.35, 0], [0, -0.35, 0], BAD, buff=0), text("less likely", 22, BAD, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.15)
        a_up.next_to(a, m.UP, buff=0.22)
        b_down.next_to(b, m.UP, buff=0.22)
        rel = text("relative to the reference", 20, MUTED).move_to([0, -3.4, 0])

        self.say(
            "[Here is the DPO loss, for one comparison.]",
            m.AnimationGroup(self.new_header("5 · THE DPO LOSS", "Raise the winner, lower the loser"), m.Write(loss, run_time=2.0)),
        )
        self.say(
            "[The first log ratio asks]: [how much more likely is the preferred answer under our model than under the frozen reference?]",
            m.AnimationGroup(m.Indicate(loss[4], color=GOOD, scale_factor=1.06), m.Create(good_line)),
            m.FadeIn(good_note, shift=0.15 * m.UP),
        )
        self.say(
            "[The second asks the same about the rejected answer.]",
            m.AnimationGroup(m.Indicate(loss[6], color=BAD, scale_factor=1.06), m.Create(bad_line), m.FadeIn(bad_note, shift=0.15 * m.UP)),
        )
        self.say(
            "[Beta scales the difference], [and minus log sigmoid turns it into a loss]: [the same Bradley-Terry form we met earlier.]",
            m.AnimationGroup(m.Indicate(loss[2], color=PINK, scale_factor=1.3), m.FadeIn(beta_note, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.Indicate(loss[1], color=INK, scale_factor=1.08), m.Create(sig_line)),
            m.FadeIn(sig_note, shift=0.15 * m.UP),
        )
        self.say(
            "[To lower the loss], [push the preferred answer's probability up, relative to the reference], [and the rejected one's down.]",
            m.Circumscribe(loss, color=TEAL, buff=0.15),
            m.AnimationGroup(m.FadeIn(a, shift=0.2 * m.UP), m.FadeIn(a_up, shift=0.2 * m.UP), m.FadeIn(rel)),
            m.AnimationGroup(m.FadeIn(b, shift=0.2 * m.UP), m.FadeIn(b_down, shift=0.2 * m.DOWN)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ a worked example

    def worked_example(self):
        self.next_section(
            "6 · A worked example",
            notes="Illustrative log-probabilities, β = 0.1: reference −14 (A) and −9 (B; the short answer is more likely a priori, "
            "which DPO does not mind: only changes relative to the reference count). After training −10 and −15: log ratios +4 and −6, "
            "β·(4 − (−6)) = 1, σ(1) ≈ 0.73, loss ≈ 0.31 (from log 2 ≈ 0.69). The gradient's weight is σ(−margin): largest where "
            "the model still favours the rejected answer.",
        )
        cols = {"ref": -3.55, "ours": -1.95, "ratio": -0.4}
        y_head, y_a, y_b = 1.75, 1.0, 0.25
        heads = m.VGroup(
            text("reference", 22, BLUE, weight=m.BOLD).move_to([cols["ref"], y_head, 0]),
            text("our model", 22, VIOLET, weight=m.BOLD).move_to([cols["ours"], y_head, 0]),
            text("log ratio", 22, INK, weight=m.BOLD).move_to([cols["ratio"], y_head, 0]),
        )
        on_baseline(heads)
        caption = text("log probability of each answer", 20, MUTED).move_to([(cols["ref"] + cols["ours"]) / 2, y_head + 0.5, 0])
        rule = m.Line([-6.75, y_head - 0.33, 0], [0.4, y_head - 0.33, 0]).set_stroke(EDGE, width=2)
        names = m.VGroup(
            text("A · preferred", 22, GOOD, weight=m.BOLD).move_to([0, y_a, 0]).align_to([-6.75, 0, 0], m.LEFT),
            text("B · rejected", 22, BAD, weight=m.BOLD).move_to([0, y_b, 0]).align_to([-6.75, 0, 0], m.LEFT),
        )

        def num(v, color, y, col, sign=False):
            s = f"{v:+.0f}" if sign else f"{v:.0f}"
            return text(s.replace("-", "−"), 28, color, weight=m.BOLD).move_to([cols[col], y, 0])

        ref = m.VGroup(num(REF_W, INK, y_a, "ref"), num(REF_L, INK, y_b, "ref"))
        ours0 = m.VGroup(num(REF_W, INK, y_a, "ours"), num(REF_L, INK, y_b, "ours"))
        ours1 = m.VGroup(num(NEW_W, GOOD, y_a, "ours"), num(NEW_L, BAD, y_b, "ours"))
        ratio0 = m.VGroup(num(0, MUTED, y_a, "ratio"), num(0, MUTED, y_b, "ratio"))
        ratio1 = m.VGroup(num(RATIO_W, GOOD, y_a, "ratio", sign=True), num(RATIO_L, BAD, y_b, "ratio", sign=True))
        beta_pill = pill("β = 0.1", PINK, PINK_FILL, 22).move_to([0, -0.55, 0]).align_to([-6.75, 0, 0], m.LEFT)

        calc1 = formula((r"\beta", PINK), (r"\,[\,", INK), ("4", GOOD), ("-", INK), ("(-6)", BAD), (r"\,] = 0.1 \times 10 = 1", INK), size=36)
        calc2 = formula((r"\sigma(1) \approx 0.73", INK), size=36)
        calc3 = formula((r"\text{loss} = -\log 0.73 \approx 0.31", TEAL), size=36)
        for k, c in enumerate((calc1, calc2, calc3)):
            c.move_to([0, -1.35 - 0.72 * k, 0]).align_to([-6.75, 0, 0], m.LEFT)

        ax = axes([-3, 3, 1], [0, 3, 1], 5.4, 3.6).move_to([3.95, -0.45, 0])
        x_label = text("β × (preferred ratio − rejected ratio)", 20, MUTED).next_to(ax.x_axis, m.DOWN, buff=0.25)
        y_label = text("loss", 22, MUTED).next_to(ax.y_axis.get_end(), m.UP, buff=0.12)

        def dpo_loss(zz):
            return float(np.log1p(np.exp(-zz)))

        curve = ax.plot(dpo_loss, x_range=[-3, 3]).set_stroke(TEAL, width=4)
        steep = ax.plot(dpo_loss, x_range=[-3, -0.6]).set_stroke(BAD, width=8)
        z = m.ValueTracker(0.0)

        def make_dot():
            return m.Dot(ax.c2p(z.get_value(), dpo_loss(z.get_value())), radius=0.11, color=TEAL).set_z_index(3)

        def make_value():
            zz = z.get_value()
            return text(f"{dpo_loss(zz):.2f}", 26, TEAL, weight=m.BOLD).next_to(ax.c2p(zz, dpo_loss(zz)), m.RIGHT, buff=0.25).shift(0.28 * m.UP)

        dot, value = m.always_redraw(make_dot), m.always_redraw(make_value)
        push = text("biggest push", 24, BAD, weight=m.BOLD).move_to([5.5, 0.85, 0])
        push_note = m.VGroup(text("where the model", 20, MUTED), text("still prefers B", 20, MUTED)).arrange(m.DOWN, buff=0.06)
        push_note.next_to(push, m.DOWN, buff=0.12)
        push_arrow = arrow(push.get_left() + 0.05 * m.LEFT, ax.c2p(-1.45, dpo_loss(-1.45)) + np.array([0.2, 0.1, 0]), BAD, buff=0.12)

        self.say(
            "[Let's try numbers, for example, with beta at zero point one.]",
            m.AnimationGroup(
                self.new_header("6 · A WORKED EXAMPLE", "One comparison, by the numbers"),
                m.FadeIn(m.VGroup(caption, heads, rule, names, ref), shift=0.15 * m.UP),
                m.FadeIn(beta_pill, shift=0.15 * m.UP),
                m.Create(ax), m.FadeIn(x_label), m.FadeIn(y_label), m.Create(curve),
            ),
        )
        self.say(
            "[At the start, our model is the reference], [so both log ratios are zero], [and the loss is log two: about zero point six nine.]",
            m.FadeIn(ours0, target_position=ref),
            m.FadeIn(ratio0, shift=0.15 * m.LEFT),
            m.AnimationGroup(m.FadeIn(dot, scale=0.5), m.FadeIn(value)),
        )
        self.say(
            "[After some training], [the preferred answer's log probability has risen by four], [and the rejected one's has fallen by six.]",
            m.Indicate(heads[1], color=VIOLET, scale_factor=1.1),
            m.AnimationGroup(m.FadeTransform(ours0[0], ours1[0]), m.FadeTransform(ratio0[0], ratio1[0])),
            m.AnimationGroup(m.FadeTransform(ours0[1], ours1[1]), m.FadeTransform(ratio0[1], ratio1[1])),
        )
        self.say("[So the gap is zero point one times ten: exactly one.]", m.Write(calc1))
        self.say(
            "[The sigmoid of one is about zero point seven three], [so the loss falls to about zero point three one.]",
            m.Write(calc2),
            m.AnimationGroup(m.Write(calc3), z.animate(run_time=1.6).set_value(Z)),
        )
        self.say(
            "[The curve is steepest on the left], [where the model still favours the rejected answer], [so those pairs get the biggest push.]",
            m.Create(steep),
            m.AnimationGroup(m.FadeIn(push), m.FadeIn(push_note)),
            m.GrowArrow(push_arrow),
        )
        self.wait(0.3)
        self.stop(dot, value)
        self.clear()

    # ------------------------------------------------------------------ side by side

    def side_by_side(self):
        self.next_section(
            "RLHF or DPO?",
            notes="Kept conservative (facts as of October 2026): PPO-based RLHF trains an explicit reward model and samples "
            "fresh answers from the policy during training (on-policy); DPO trains offline on a fixed set of comparisons, "
            "with the policy and a frozen reference. Both learn from human (or other) preference comparisons.",
        )
        left = panel(-3.45, "RLHF with PPO", PINK)
        right = panel(3.45, "DPO", TEAL)
        l_items = m.VGroup(
            bullet("trains a separate reward model", PINK),
            bullet("samples fresh answers as it learns", PINK),
            bullet("four models, many settings", PINK),
        ).arrange(m.DOWN, buff=0.45, aligned_edge=m.LEFT)
        r_items = m.VGroup(
            bullet("one loss, straight on the pairs", TEAL),
            bullet("learns from a fixed dataset", TEAL),
            bullet("your model and a frozen reference", TEAL),
        ).arrange(m.DOWN, buff=0.45, aligned_edge=m.LEFT)
        l_items.move_to(left[0].get_center() + 0.3 * m.DOWN).align_to(left[0], m.LEFT).shift(0.45 * m.RIGHT)
        r_items.move_to(right[0].get_center() + 0.3 * m.DOWN).align_to(right[0], m.LEFT).shift(0.45 * m.RIGHT).align_to(l_items, m.UP)
        same = bottom("both learn from people picking the better of two answers", SKY, SKY_FILL, 22, y=-3.3)

        self.say(
            "[So, side by side.] [RLHF trains a reward model, then uses PPO], [sampling fresh answers as it learns.]",
            m.AnimationGroup(self.new_header("SIDE BY SIDE", "RLHF or DPO?"), m.FadeIn(left), m.FadeIn(right)),
            m.FadeIn(l_items[0], shift=0.15 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(l_items[1], shift=0.15 * m.RIGHT), m.FadeIn(l_items[2], shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[DPO uses one simple loss on a fixed set of comparisons], [with just a frozen reference beside it.] It's simpler and cheaper to run.",
            m.AnimationGroup(m.FadeIn(r_items[0], shift=0.15 * m.RIGHT), m.FadeIn(r_items[1], shift=0.15 * m.RIGHT)),
            m.FadeIn(r_items[2], shift=0.15 * m.RIGHT),
        )
        self.say(
            "[Either way, the signal is the same]: [people picking the better of two answers.]",
            m.AnimationGroup(m.Indicate(left[1], color=PINK, scale_factor=1.08), m.Indicate(right[1], color=TEAL, scale_factor=1.08)),
            m.FadeIn(same, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Compare, score, nudge.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Compare", "Score", "Nudge"), (SKY, AMBER, VIOLET))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Show a model which answer people prefer, and it learns to give more like it.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [compare], [score], [nudge.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Show a model which answer people prefer, and it learns to give more answers like it.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
