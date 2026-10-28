"""AI, Explained · Episode 29 · Fine-Tuning & Instruction Tuning

Running example: the same model, before and after, asked "How do I boil an egg?". Pretrained
only, it continues the document with more questions (an FAQ page); after supervised
fine-tuning on (instruction, response) pairs it answers. Then: what instruction data looks
like (thousands to millions of pairs; quality beats quantity, LIMA), the chat template
(system / user / assistant, marker tokens, generation continues after the assistant marker),
the loss on the assistant's tokens only (18 of the example's 34 tokens), the InstructGPT
recipe (SFT, then preference tuning: Episode 30), full fine-tuning versus adapters (Episode 41),
and two cautions.

One colour per role, all film long: system AMBER, user / instructions SKY, assistant /
responses VIOLET (the model), fine-tuning TEAL (the level accent), preference tuning PINK,
adapters AMBER; "not an answer" BAD, "an answer" GOOD.

Illustrative (made up for teaching, said with "for example"): the next-token probabilities
after the question, the per-token losses before and after training (seeded numpy), the
example pairs. Cited: InstructGPT compute and data sizes, the 1.3B-vs-175B preference result
(Ouyang et al., 2022), LIMA's 1,000 examples (Zhou et al., 2023). Computed and asserted: the
token counts, the compute share, the parameter ratio, the adapter share.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

SYS = (AMBER, AMBER_FILL)
USR = (SKY, SKY_FILL)
AST = (VIOLET, VIOLET_FILL)
TUNE = (TEAL, TEAL_FILL)
PREF = (PINK, PINK_FILL)
ROLE = {"system": SYS, "user": USR, "assistant": AST}

QUESTION = "How do I boil an egg?"
BASE_REPLY = ("How do I poach an egg?", "How do I fry an egg?", "How long do eggs keep?")
TUNED_REPLY = ("Put it in boiling water for", "about ten minutes, then", "cool it in cold water.")
SYSTEM = "You are a helpful assistant."
PLAIN_MONO = "Menlo"  # Fira Code draws "<|" and "|>" as ligature triangles; Menlo keeps them literal

# ---------------------------------------------------------------------------- the data

# illustrative next-token probabilities after the question (a softmax over a toy shortlist,
# the last logit standing for every other token)
NEXT_WORDS = ("How", "What", "Can", "Put")
_logits = np.array([2.1, 1.25, 0.6, 0.05, 1.9])
_p = np.exp(_logits) / np.exp(_logits).sum()
NEXT_P = _p[:4]
assert NEXT_P.argmax() == 0 and NEXT_P[:3].sum() > 5 * NEXT_P[3]  # "the likeliest next words start another question"

# the example as tokens: (piece, role, counted in the loss)
TOKENS = (
    [("<|system|>", "system", False)]
    + [(w, "system", False) for w in ("You", "are", "a", "helpful", "assistant", ".")]
    + [("<|user|>", "user", False)]
    + [(w, "user", False) for w in ("How", "do", "I", "boil", "an", "egg", "?")]
    + [("<|assistant|>", "assistant", False)]
    + [(w, "assistant", True) for w in ("Put", "it", "in", "boiling", "water", "for", "about", "ten", "minutes", ",", "then", "cool", "it", "in", "cold", "water", ".")]
    + [("<|end|>", "assistant", True)]
)
N_TOKENS = len(TOKENS)
N_COUNTED = sum(t[2] for t in TOKENS)
assert (N_COUNTED, N_TOKENS) == (18, 34)  # narration: "eighteen of the thirty-four"
ROWS = (7, 8, 8, 11)  # tokens per row on screen: system, user, assistant marker + 7, the rest
assert sum(ROWS) == N_TOKENS

# illustrative per-token losses on the answer, before and after fine-tuning (seeded)
_rng = np.random.default_rng(29)
LOSS_BEFORE = np.clip(_rng.normal(3.2, 0.7, N_COUNTED), 1.6, 4.6)
LOSS_AFTER = np.clip(_rng.normal(0.45, 0.2, N_COUNTED), 0.1, 1.0)
assert LOSS_AFTER.mean() < LOSS_BEFORE.mean() / 3

# InstructGPT (Ouyang et al., 2022): compute in petaflop/s-days, data, parameters
PRETRAIN_PFD, SFT_PFD = 3640, 4.9  # GPT-3 pretraining; the 175B SFT model
assert SFT_PFD / PRETRAIN_PFD < 0.01  # "well under one percent of the compute"
SFT_PROMPTS = 13_000  # "about thirteen thousand prompts" (the paper: "about 13k")
SMALL_B, BIG_B = 1.3, 175.0  # InstructGPT 1.3B (SFT + RLHF) preferred to GPT-3 175B
assert BIG_B / SMALL_B > 100  # "more than a hundred times larger"
LIMA_EXAMPLES = 1000  # Zhou et al., 2023: "about a thousand carefully chosen examples"

# full fine-tuning versus an adapter on one weight matrix
D, RANK = 4096, 8
FULL_WEIGHTS = D * D
ADAPTER_WEIGHTS = 2 * D * RANK
assert (FULL_WEIGHTS, ADAPTER_WEIGHTS) == (16_777_216, 65_536)
ADAPTER_SHARE = ADAPTER_WEIGHTS / FULL_WEIGHTS
assert ADAPTER_SHARE < 0.005  # "under half a percent as many numbers"


# ---------------------------------------------------------------------------- helpers


def variant(name, color, fill):
    """kit.recolored(), but written under this episode's prefix: assets/ep29-<icon>-<hex>.svg."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    stem = f"ep29-{name.removeprefix('ep29-')}-{color.lstrip('#').lower()}"
    path = ASSETS / f"{stem}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return stem


def panel(x, title, color, width=6.3, height=5.0, y=-0.3, title_color=None):
    box = card(width, height, color, CARD, stroke=2).move_to([x, y, 0]).set_z_index(-1)
    head = text(title, 24, title_color or color, weight=m.BOLD)
    head.move_to(box.get_corner(m.UL) + np.array([0.35 + head.width / 2, -0.42, 0]))
    return m.VGroup(box, head)


def lines(rows, size=22, color=INK, weight=m.NORMAL):
    """Left-aligned lines of text, one Text (so it can be typed letter by letter)."""
    return text("\n".join(rows), size, color, weight=weight)


def reply(rows, color, fill, tail="left", size=22, ink=INK):
    return bubble(lines(rows, size, ink), color, fill, tail)


def place(group, part, point):
    """Move a group so that one of its parts is centred at `point`."""
    return group.shift(np.array(point) - part.get_center())


def window(width, height, title):
    box = card(width, height, EDGE, CARD, radius=0.2, stroke=2)
    top, left = box.get_top()[1], box.get_left()[0]
    dots = m.VGroup(*[m.Dot(radius=0.07, color=c) for c in (BAD, AMBER, GOOD)]).arrange(m.RIGHT, buff=0.14)
    dots.move_to([left + 0.28 + dots.width / 2, top - 0.28, 0])
    name = code(title, 20, MUTED).move_to([box.get_x(), top - 0.28, 0])
    rule = m.Line([left, top - 0.56, 0], [box.get_right()[0], top - 0.56, 0]).set_stroke(EDGE, width=2)
    return m.VGroup(box, dots, name, rule)


def mark(s, size=22, color=INK, **kw):
    """Template text with literal <|markers|> (no ligatures)."""
    return m.Text(s, font=PLAIN_MONO, font_size=size, color=color, **kw)


def badge(label, color, fill, size=22, height=0.5):
    """A pill of fixed height whose words sit on a common baseline (pills of 'user' and
    'system' line up)."""
    words = text(label, size, color, weight=m.BOLD)
    box = card(words.width + 0.5, height, color, fill, radius=height / 2, stroke=2)
    words.move_to(box)
    sitting = [g for g, c in zip(words, label.replace(" ", "")) if c not in "gjpqy"] or list(words)
    baseline = box.get_center()[1] - 0.449 * size / 48 / 2
    words.shift((baseline - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    return m.VGroup(box, words)


def token(piece, color, fill, size=24, height=0.6):
    """A token chip: role markers in plain monospace, words baseline-aligned (kit.chip)."""
    if piece.startswith("<|"):
        label = mark(piece, 22, color)
        box = card(label.width + 0.3, height, color, fill, radius=0.14, stroke=2.5)
        return m.VGroup(box, label.move_to(box))
    if piece in ".,":  # drawn, not typed: a lone period or comma is too small for the text checker
        box = card(0.44, height, color, fill, radius=0.14, stroke=2.5)
        baseline = box.get_center()[1] - 0.449 * size / 48 / 2
        dot = m.Dot([box.get_x(), baseline + 0.045, 0], radius=0.045, color=INK)
        mark_shape = m.VGroup(dot)
        if piece == ",":
            mark_shape.add(m.Line(dot.get_center() + np.array([0.035, -0.01, 0]), dot.get_center() + np.array([-0.025, -0.12, 0])).set_stroke(INK, width=3.5))
        return m.VGroup(box, mark_shape)
    width = max(text(piece, size, weight=m.SEMIBOLD).width + 0.3, 0.44)
    return chip(piece, color, fill, width=width, height=height, size=size)


def bar(height, color, base, width=0.22):
    """A loss bar standing on the point `base`."""
    h = max(height, 0.02)
    return m.Rectangle(width=width, height=h, stroke_width=0).set_fill(color, opacity=0.9).move_to(base + np.array([0, h / 2, 0]))


def weight_grid(rows, cols, rng, color, cell=0.26, gap=0.07):
    grid = m.VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = m.Square(side_length=cell, stroke_width=0).set_fill(color, opacity=float(rng.uniform(0.25, 0.75)))
            sq.move_to([c * (cell + gap), -r * (cell + gap), 0])
            grid.add(sq)
    return grid


def assistant_icon(height=0.8):
    chip_icon = icon("llm", height)
    badge = icon("check", height * 0.42).move_to(chip_icon.get_corner(m.DR) + np.array([-0.05, 0.05, 0]) * height)
    return m.VGroup(chip_icon, badge)


def box_node(pic, label, sub, color, fill, width=3.5, height=2.35, label_size=26):
    """A node with its icon, label and sub-label at fixed heights (labels line up across nodes)."""
    box = card(width, height, color, fill)
    c = box.get_center()
    pic.move_to(c + np.array([0, 0.45, 0]))
    name = text(label, label_size, INK, weight=m.BOLD).move_to(c + np.array([0, -0.32, 0]))
    note = text(sub, 20, MUTED).move_to(c + np.array([0, -0.72, 0]))
    return m.VGroup(box, m.VGroup(pic, name, note))


def bottom(label, color, fill, size=22, y=-3.25):
    return pill(label, color, fill, size).move_to([0, y, 0])


# ---------------------------------------------------------------------------- the episode


class Ep29FineTuning(Episode):
    number = 29

    def lesson(self):
        self.hook()
        self.base_model()
        self.idea()
        self.data()
        self.template()
        self.loss()
        self.recipe()
        self.adapters()
        self.catch()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="The same model asked 'How do I boil an egg?', before and after fine-tuning. The base model's reply is illustrative.")
        sides = []
        for cx, title, title_color, rows, color, fill, ink, verdict in (
            (-3.45, "Before · pretrained only", MUTED, BASE_REPLY, EDGE, CARD, MUTED, pill("✗  more questions", BAD, BAD_FILL, 22)),
            (3.45, "After · fine-tuned", TEAL, TUNED_REPLY, VIOLET, VIOLET_FILL, INK, pill("✓  an answer", GOOD, GOOD_FILL, 22)),
        ):
            frame = panel(cx, title, EDGE if cx < 0 else TEAL, height=4.9, y=-0.2, title_color=title_color)
            ask = bubble(text(QUESTION, 24, INK, weight=m.SEMIBOLD), SKY, SKY_FILL, tail="right")
            place(ask, ask[1], [cx + 0.45, 1.1, 0])
            tip = ask[0].get_vertices()[2]
            person = icon("user", 0.6).move_to(tip + np.array([0.38, -0.25, 0]))
            answer = reply(rows, color, fill, "left", ink=ink)
            place(answer, answer[1], [cx - 0.05, -0.68, 0])
            tip = answer[0].get_vertices()[2]
            model = icon("llm", 0.72).move_to(tip + np.array([-0.42, -0.3, 0]))
            verdict.move_to([cx + 1.55, -2.2, 0])
            sides.append((frame, ask, person, answer, model, verdict))
        (lf, lq, lu, la, lm, lv), (rf, rq, ru, ra, rm, rv) = sides
        same = bottom("same model underneath  ·  fine-tuning made the difference", TEAL, TEAL_FILL)

        self.say(
            "[Ask a freshly pretrained language model a simple question]: [how do I boil an egg?]",
            m.AnimationGroup(self.new_header("BEFORE & AFTER", "One model, two answers"), m.FadeIn(lf), m.FadeIn(lm, shift=0.2 * m.UP), m.FadeIn(lu, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(lq[:2]), m.AddTextLetterByLetter(lq[2], time_per_char=0.03)),
        )
        self.say(
            "[It might reply with more questions]: [how do I poach an egg? How do I fry an egg?]",
            m.FadeIn(la[:2], shift=0.15 * m.UP),
            m.Succession(m.AddTextLetterByLetter(la[2], time_per_char=0.035), m.FadeIn(lv, shift=0.15 * m.UP)),
        )
        self.say(
            "[Now ask the same model after a little extra training.] [This time, it answers]: [put it in boiling water for about ten minutes.]",
            m.AnimationGroup(m.FadeIn(rf), m.FadeIn(rm, shift=0.2 * m.UP), m.FadeIn(ru, shift=0.2 * m.UP), m.FadeIn(rq, shift=0.15 * m.UP)),
            m.FadeIn(ra[:2], shift=0.15 * m.UP),
            m.Succession(m.AddTextLetterByLetter(ra[2], time_per_char=0.035), m.FadeIn(rv, shift=0.15 * m.UP)),
        )
        self.say(
            "[It's the same model underneath.] So what turned a text predictor into an assistant? [Fine-tuning.]",
            m.AnimationGroup(m.Indicate(lm, color=VIOLET, scale_factor=1.2), m.Indicate(rm, color=VIOLET, scale_factor=1.2)),
            m.FadeIn(same, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the base model

    def base_model(self):
        self.next_section("The base model", notes="Pretraining (Episode 27) teaches next-token prediction on web text; on FAQ pages a question is followed by another. Probabilities are illustrative.")
        task = pill("pretraining  ·  predict the next token of web text  ·  Episode 27", VIOLET, VIOLET_FILL, 20).move_to([0, 2.2, 0])
        win = window(6.3, 4.25, "eggs-faq.html").move_to([-3.35, -0.55, 0])
        left = win[0].get_left()[0] + 0.45
        top = win[0].get_top()[1]
        heading = m.VGroup(icon("ep29-egg", 0.5), text("Eggs: common questions", 24, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.22)
        heading.move_to([0, top - 1.05, 0]).align_to([left, 0, 0], m.LEFT)
        faq = (QUESTION, *BASE_REPLY, "Can I freeze eggs?")
        items = m.VGroup()
        for k, q in enumerate(faq):
            row = m.VGroup(m.Dot(radius=0.05, color=SKY if k == 0 else FAINT), text(q, 22, SKY if k == 0 else MUTED, weight=m.SEMIBOLD if k == 0 else m.NORMAL))
            row.arrange(m.RIGHT, buff=0.2).move_to([0, top - 1.75 - 0.5 * k, 0]).align_to([left + 0.05, 0, 0], m.LEFT)
            items.add(row)

        title = text("what comes next?", 26, INK, weight=m.BOLD).move_to([3.7, 1.15, 0])
        example = text("for example", 20, MUTED).next_to(title, m.DOWN, buff=0.12)
        bars = m.VGroup()
        for k, (word, p) in enumerate(zip(NEXT_WORDS, NEXT_P)):
            y = 0.05 - 0.58 * k
            color = VIOLET if word == "Put" else SKY
            label = text(word, 24, INK, weight=m.SEMIBOLD).move_to([0, y, 0]).align_to([2.0, 0, 0], m.RIGHT)
            b = m.Rectangle(width=6.0 * p, height=0.32, stroke_width=0).set_fill(color, opacity=0.9)
            b.move_to([2.2, y, 0], aligned_edge=m.LEFT)
            pct = text(f"{round(100 * p)}%", 20, MUTED).next_to(b, m.RIGHT, buff=0.15)
            bars.add(m.VGroup(label, b, pct))
        brace = m.Brace(m.VGroup(*[r[2] for r in bars[:3]], bars[0][1]), m.RIGHT, buff=0.15).set_color(SKY)
        brace_label = m.VGroup(text("another", 22, SKY, weight=m.SEMIBOLD), text("question", 22, SKY, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.06)
        brace_label.next_to(brace, m.RIGHT, buff=0.15)
        answer_note = text("an answer", 24, VIOLET, weight=m.SEMIBOLD).next_to(bars[3][2], m.RIGHT, buff=0.25)
        verdict = bottom("not broken: it continues the document, exactly as it was trained to", TEAL, TEAL_FILL)

        self.say(
            "[Why the questions?] [Pretraining taught the model one skill]: continue the text, one token at a time.",
            self.new_header("THE BASE MODEL", "A text predictor continues documents"),
            m.FadeIn(task, shift=0.2 * m.DOWN),
        )
        self.say(
            "[It learned from pages like this one], [where one question is often followed by another.]",
            m.AnimationGroup(m.FadeIn(win, shift=0.2 * m.UP), m.FadeIn(heading, shift=0.2 * m.UP)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in items], lag_ratio=0.25),
        )
        self.say(
            "So, for example, [after our question], [the likeliest next words start another question.]",
            m.AnimationGroup(m.Indicate(items[0], color=SKY, scale_factor=1.06), m.FadeIn(title), m.FadeIn(example)),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in bars], lag_ratio=0.2),
                m.Succession(m.Wait(0.6), m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(brace_label), m.FadeIn(answer_note))),
            ),
        )
        self.say(
            "It isn't broken. [It's doing exactly what it was trained to do]: [continue the document.]",
            m.Circumscribe(win[0], color=TEAL, buff=0.05),
            m.FadeIn(verdict, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the idea: SFT

    def idea(self):
        self.next_section(
            "Supervised fine-tuning",
            notes="Keep training the pretrained model, same next-token objective, on instruction-response pairs. Compute: GPT-3 pretraining 3,640 petaflop/s-days vs 4.9 for the 175B SFT model (Ouyang et al., 2022, sec. 5.1).",
        )
        n1 = box_node(icon("llm", 0.8), "Pretrained model", "continues text", *AST).move_to([-4.6, 1.05, 0])
        n2 = box_node(icon("database", 0.8), "Fine-tuning data", "instruction → response", *TUNE).move_to([0, 1.05, 0])
        n3 = box_node(assistant_icon(0.8), "Assistant", "follows instructions", GOOD, GOOD_FILL).move_to([4.6, 1.05, 0])
        a1 = arrow(n1[0].get_right(), n2[0].get_left(), MUTED, buff=0.1)
        a2 = arrow(n2[0].get_right(), n3[0].get_left(), MUTED, buff=0.1)
        same = pill("same objective: predict the next token", TEAL, TEAL_FILL, 22)
        new = pill("new data: instructions with good responses", TEAL, TEAL_FILL, 22)
        m.VGroup(same, new).arrange(m.RIGHT, buff=0.45).move_to([0, -0.85, 0])
        supervised = text("supervised: every example shows the right answer, like the labels in Episode 8", 22, MUTED).move_to([0, -1.62, 0])

        label_x = -2.7
        names = (text("GPT-3 pretraining", 22, INK, weight=m.SEMIBOLD), text("fine-tuning (InstructGPT, 175B)", 22, INK, weight=m.SEMIBOLD))
        rows = m.VGroup()
        for k, (name, value, color) in enumerate(zip(names, (PRETRAIN_PFD, SFT_PFD), (VIOLET, TEAL))):
            y = -2.32 - 0.5 * k
            name.move_to([0, y, 0]).align_to([label_x, 0, 0], m.RIGHT)
            b = m.Rectangle(width=max(6.2 * value / PRETRAIN_PFD, 0.05), height=0.3, stroke_width=0).set_fill(color, opacity=0.9)
            b.move_to([label_x + 0.2, y, 0], aligned_edge=m.LEFT)
            num = text(f"{value:,}" if value >= 100 else f"{value}", 22, color, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.15)
            rows.add(m.VGroup(name, b, num))
        unit = text("compute, petaflop/s-days  ·  Ouyang et al., 2022", 20, MUTED).move_to([0, -3.38, 0]).align_to([label_x + 0.2, 0, 0], m.LEFT)

        self.say(
            "[The fix is called supervised fine-tuning.]",
            m.AnimationGroup(self.new_header("THE IDEA", "Supervised fine-tuning"), m.FadeIn(n1, shift=0.2 * m.UP)),
        )
        self.say(
            "[Take the pretrained model], [keep training it on examples of instructions with good responses], [and out comes an assistant.]",
            m.Indicate(n1, color=VIOLET, scale_factor=1.05),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(n2, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(n3, shift=0.2 * m.UP), pulse(a1, TEAL)),
        )
        self.say(
            "[The training is the same next-token prediction as before.] [What changes is the data.]",
            m.FadeIn(same, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(new, shift=0.2 * m.UP), m.Indicate(n2, color=TEAL, scale_factor=1.05)),
        )
        self.say(
            "[It's supervised, like the labels in episode eight]: [every example shows the right answer.]",
            m.FadeIn(supervised, shift=0.15 * m.UP),
            m.Indicate(n2[1][2], color=TEAL, scale_factor=1.15),
        )
        self.say(
            "[And it's cheap.] [For one well-known model, it took well under one percent of the compute of pretraining.]",
            m.AnimationGroup(m.FadeIn(rows[0][0]), m.GrowFromEdge(rows[0][1], m.LEFT), m.FadeIn(rows[0][2])),
            m.AnimationGroup(m.FadeIn(rows[1][0]), m.GrowFromEdge(rows[1][1], m.LEFT), m.FadeIn(rows[1][2], scale=1.3), m.FadeIn(unit)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ instruction data

    def data(self):
        self.next_section(
            "Instruction data",
            notes="Example pairs are illustrative. Thousands to millions of pairs, written by people or drafted by models and checked. LIMA: 1,000 curated examples (Zhou et al., 2023). Flawed examples in the pile are random (seeded).",
        )
        examples = (
            (QUESTION, "Put it in boiling water for about ten minutes."),
            ("Sum up this email in one line.", "The team meeting moves to Friday at 3 pm."),
            ("Translate “thank you” into French.", "Merci."),
            ("Is 91 a prime number?", "No: 91 = 7 × 13."),
        )
        cards = m.VGroup()
        for k, (ask, ans) in enumerate(examples):
            x, y = (-3.35, 3.35)[k % 2], (1.3, -0.2)[k // 2]
            box = card(6.45, 1.25, EDGE, CARD, radius=0.2, stroke=2).move_to([x, y, 0])
            left = box.get_left()[0]
            rows = m.VGroup()
            for dy, words, color in ((0.27, ask, SKY), (-0.27, ans, VIOLET)):
                dot = m.Dot(radius=0.07, color=color).move_to([left + 0.32, y + dy, 0])
                line = text(words, 22, INK if color == VIOLET else SKY, weight=m.SEMIBOLD if color == SKY else m.NORMAL)
                line.move_to([0, y + dy, 0]).align_to([left + 0.55, 0, 0], m.LEFT)
                rows.add(m.VGroup(dot, line))
            cards.add(m.VGroup(box, rows))
        legend = m.VGroup(badge("instruction", SKY, SKY_FILL, 22), badge("response", VIOLET, VIOLET_FILL, 22)).arrange(m.RIGHT, buff=0.25)
        legend.move_to([0, 2.32, 0]).align_to([6.6, 0, 0], m.RIGHT)

        rng = np.random.default_rng(7)
        pile = m.VGroup()
        for r in range(4):
            for c in range(14):
                tile_box = m.RoundedRectangle(corner_radius=0.04, width=0.3, height=0.2).set_stroke(EDGE, width=1.5).set_fill(CARD, opacity=1)
                a = m.Line([-0.1, 0.04, 0], [0.08, 0.04, 0]).set_stroke(SKY, width=2)
                b = m.Line([-0.1, -0.04, 0], [0.04, -0.04, 0]).set_stroke(VIOLET, width=2)
                pile.add(m.VGroup(tile_box, a, b).move_to([-6.25 + 0.37 * c, -1.45 - 0.29 * r, 0]))
        flawed = rng.random(len(pile)) < 0.2
        bad = m.VGroup(*[p for p, f in zip(pile, flawed) if f])
        good = m.VGroup(*[p for p, f in zip(pile, flawed) if not f])
        pile_label = text("thousands to millions of pairs", 22, MUTED).move_to([pile.get_x(), -2.75, 0])

        sources = m.VGroup(
            m.VGroup(icon("user", 0.45), text("written by people", 22, INK)).arrange(m.RIGHT, buff=0.22),
            m.VGroup(icon("llm", 0.45), text("or drafted by models, then checked", 22, INK)).arrange(m.RIGHT, buff=0.22),
        ).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT).move_to([0, -1.85, 0]).align_to([1.0, 0, 0], m.LEFT)
        funnel = icon("ep29-funnel", 0.95).move_to([0.25, -1.85, 0])
        sloppy = text("✗  sloppy examples teach sloppy habits", 22, BAD, weight=m.SEMIBOLD).move_to([0, -1.45, 0]).align_to([1.15, 0, 0], m.LEFT)
        lima = text(f"✓  about {LIMA_EXAMPLES:,} careful examples", 22, GOOD, weight=m.SEMIBOLD).move_to([0, -2.1, 0]).align_to([1.15, 0, 0], m.LEFT)
        lima_sub = text("made a capable assistant", 22, INK).move_to([0, -2.5, 0]).align_to([1.55, 0, 0], m.LEFT)
        cite = text("LIMA · Zhou et al., 2023", 20, MUTED).move_to([0, -2.95, 0]).align_to([1.55, 0, 0], m.LEFT)
        quality = pill("quality beats quantity", TEAL, TEAL_FILL, 22).move_to([pile.get_x(), -3.3, 0])

        def show_card(c):
            return m.AnimationGroup(m.FadeIn(c[0]), m.FadeIn(c[1], shift=0.15 * m.RIGHT))

        self.say(
            "[What does this data look like?] [Each example pairs an instruction] [with the response we want.]",
            m.AnimationGroup(self.new_header("INSTRUCTION DATA", "Pairs of instruction and response"), m.FadeIn(legend)),
            m.AnimationGroup(m.FadeIn(cards[0][0]), m.FadeIn(cards[0][1][0], shift=0.15 * m.RIGHT)),
            m.FadeIn(cards[0][1][1], shift=0.15 * m.RIGHT),
        )
        self.say(
            "It covers many kinds of tasks: [summing up an email], [translating], [and a bit of arithmetic.]",
            *[show_card(c) for c in cards[1:]],
        )
        self.say(
            "[A dataset holds thousands to millions of these], [written by people, or drafted by models and then checked.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, scale=0.5) for p in pile], lag_ratio=0.02), m.FadeIn(pile_label)),
            m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.RIGHT) for s in sources], lag_ratio=0.4),
        )
        self.say(
            "[But quality beats quantity.] [Sloppy or wrong examples teach sloppy habits], [so teams filter hard.]",
            m.AnimationGroup(m.FadeOut(sources), m.FadeIn(quality, shift=0.2 * m.UP)),
            m.AnimationGroup(*[p[0].animate.set_stroke(BAD, width=2.5).set_fill(BAD_FILL, opacity=1) for p in bad], m.FadeIn(sloppy, shift=0.15 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(funnel, scale=0.8), m.FadeOut(bad, shift=0.3 * m.DOWN), *[p[0].animate.set_stroke(TEAL, width=2) for p in good]),
        )
        self.say(
            "[In one study, about a thousand carefully chosen examples] [made a surprisingly capable assistant.]",
            m.AnimationGroup(m.FadeIn(lima, shift=0.15 * m.RIGHT), m.FadeIn(cite)),
            m.AnimationGroup(m.FadeIn(lima_sub, shift=0.15 * m.RIGHT), m.Indicate(good, color=GOOD, scale_factor=1.04)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ chat templates

    def template(self):
        self.next_section(
            "Chat templates",
            notes="Roles (system, user, assistant) written as one string with marker tokens. The markers here are generic; real ones differ (ChatML <|im_start|>/<|im_end|>, Llama 2 [INST]...[/INST]).",
        )
        role_rows = m.VGroup()
        for y, role, rows in ((1.6, "system", (SYSTEM,)), (0.5, "user", (QUESTION,)), (-0.95, "assistant", TUNED_REPLY)):
            color, fill = ROLE[role]
            tag = badge(role, color, fill, 22)
            words = lines(rows, 22, INK)
            box = card(words.width + 0.5, words.height + 0.45, color, fill, radius=0.2, stroke=2)
            box.move_to([0, y, 0]).align_to([-4.75, 0, 0], m.LEFT)
            words.move_to(box)
            tag.move_to([0, box.get_top()[1] - 0.3, 0]).align_to([-4.95, 0, 0], m.RIGHT)
            role_rows.add(m.VGroup(tag, box, words))

        win = window(6.05, 4.15, "what the model reads").move_to([3.68, 0.42, 0])
        x0 = win[0].get_left()[0] + 0.32
        y0 = win[0].get_top()[1] - 0.92
        source = (
            ("<|system|>", AMBER), (SYSTEM, INK), ("<|user|>", SKY), (QUESTION, INK), ("<|assistant|>", VIOLET),
            ("Put it in boiling water for", INK), ("about ten minutes, then cool", INK), ("it in cold water.<|end|>", INK),
        )
        tlines = m.VGroup()
        for k, (s, color) in enumerate(source):
            line = mark(s, 22, color, t2c={"<|end|>": VIOLET}) if "<|end|>" in s else mark(s, 22, color)
            tlines.add(line.move_to([0, y0 - 0.4 * k, 0]).align_to([x0, 0, 0], m.LEFT))
        markers = m.VGroup(tlines[0], tlines[2], tlines[4], glyphs(tlines[7], "<|end|>"))
        retyped = m.VGroup(*[l.copy() for l in tlines[5:]])
        cursor = m.Rectangle(width=0.13, height=0.3, stroke_width=0).set_fill(VIOLET, opacity=0.9)
        cursor.move_to([x0 + 0.07, tlines[5].get_y(), 0])

        link = arrow([-0.5, 0.42, 0], [0.55, 0.42, 0], TEAL, buff=0.05)
        link_label = m.VGroup(text("chat", 22, TEAL, weight=m.SEMIBOLD), text("template", 22, TEAL, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.05)
        link_label.next_to(link, m.UP, buff=0.12)

        variants = m.VGroup(
            m.VGroup(text("ChatML", 22, MUTED, weight=m.SEMIBOLD), mark("<|im_start|>user … <|im_end|>", 22, MUTED)).arrange(m.RIGHT, buff=0.25),
            m.VGroup(text("Llama 2", 22, MUTED, weight=m.SEMIBOLD), mark("[INST] … [/INST]", 22, MUTED)).arrange(m.RIGHT, buff=0.25),
        ).arrange(m.RIGHT, buff=0.9).move_to([0, -2.55, 0])
        same = bottom("use the template the model was trained with", AMBER, AMBER_FILL)
        still = bottom("still next-token prediction, shaped into a conversation", TEAL, TEAL_FILL)

        def row_in(r):
            return m.AnimationGroup(m.FadeIn(r[0], shift=0.15 * m.RIGHT), m.FadeIn(r[1:], shift=0.15 * m.RIGHT))

        self.say(
            "[A conversation has roles.] [A system message sets the ground rules], [the user asks], [and the assistant answers.]",
            self.new_header("CHAT TEMPLATES", "Roles, written as one string"),
            row_in(role_rows[0]),
            row_in(role_rows[1]),
            row_in(role_rows[2]),
        )
        self.say(
            "[A chat template writes it all out as one string of text], [with special marker tokens between the turns.]",
            m.AnimationGroup(
                m.GrowArrow(link), m.FadeIn(link_label), m.FadeIn(win),
                m.LaggedStart(*[m.FadeIn(l, shift=0.1 * m.RIGHT) for l in tlines], lag_ratio=0.12),
            ),
            m.LaggedStart(*[m.Indicate(k, scale_factor=1.12) for k in markers], lag_ratio=0.25),
        )
        self.say(
            "[The markers differ from model to model], [and you should use the one the model was trained with.]",
            m.LaggedStart(*[m.FadeIn(v, shift=0.15 * m.UP) for v in variants], lag_ratio=0.3),
            m.FadeIn(same, shift=0.2 * m.UP),
        )
        self.say(
            "[At chat time, the app fills in the template up to the assistant marker], [and the model continues from there], [until it writes the end marker.]",
            m.AnimationGroup(m.FadeOut(tlines[5:]), m.FadeOut(role_rows[2][1:]), m.FadeIn(cursor), m.Indicate(tlines[4], color=VIOLET, scale_factor=1.1)),
            m.Succession(m.FadeOut(cursor, run_time=0.2), *[m.AddTextLetterByLetter(l, time_per_char=0.02) for l in retyped]),
            m.AnimationGroup(m.Circumscribe(glyphs(retyped[2], "<|end|>"), color=VIOLET, buff=0.06), m.FadeIn(role_rows[2][1:], shift=0.15 * m.RIGHT)),
        )
        self.say(
            "So it's still next-token prediction, [just shaped into a conversation.]",
            m.AnimationGroup(m.FadeOut(variants), m.FadeTransform(same, still)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the loss

    def loss(self):
        self.next_section(
            "The loss",
            notes="Loss usually computed only on the assistant's tokens (incl. the end marker): 18 of 34 here. Per-token losses are illustrative (seeded numpy, seed 29); the tokenisation is simplified.",
        )
        chips, plain = m.VGroup(), m.VGroup()
        for piece, role, _ in TOKENS:
            chips.add(token(piece, *ROLE[role]))
            plain.add(token(piece, EDGE, CARD))
        rows_y = (2.3, 1.5, 0.0, -1.75)
        start = 0
        for n, y in zip(ROWS, rows_y):
            for group in (chips, plain):
                row = group[start : start + n].arrange(m.RIGHT, buff=0.1)
                row.move_to([0, y, 0]).align_to([-6.5, 0, 0], m.LEFT)
            start += n
        counted = m.VGroup(*[c for c, t in zip(chips, TOKENS) if t[2]])
        context = m.VGroup(*[c for c, t in zip(chips, TOKENS) if not t[2]])
        end_chip = chips[-1]

        brace = m.Brace(m.VGroup(chips[:7], chips[7:15]), m.RIGHT, buff=0.25).set_color(MUTED)
        masked = m.VGroup(text("context only:", 22, INK, weight=m.SEMIBOLD), text("loss masked out", 22, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        masked.next_to(brace, m.RIGHT, buff=0.2)
        count = text(f"counted in the loss: {N_COUNTED} of {N_TOKENS} tokens", 24, INK, weight=m.SEMIBOLD, t2c={f"{N_COUNTED}": VIOLET})
        count.move_to([0, -2.85, 0])
        stop = text("learns when to stop", 22, VIOLET, weight=m.SEMIBOLD).next_to(end_chip, m.DOWN, buff=0.2)

        bases = [c.get_top() + np.array([0, 0.08, 0]) for c in counted]
        scale = 0.17
        before = m.VGroup(*[bar(scale * v, VIOLET, b) for v, b in zip(LOSS_BEFORE, bases)])
        after = m.VGroup(*[bar(scale * v, VIOLET, b) for v, b in zip(LOSS_AFTER, bases)])
        loss_label = m.VGroup(text("loss per answer token", 22, INK, weight=m.SEMIBOLD), text("for example", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        loss_label.move_to([0, 0.75, 0]).align_to([3.0, 0, 0], m.LEFT)
        surprised = text("it expected another question", 22, BAD, weight=m.SEMIBOLD).move_to([0, 0.0, 0]).align_to([3.0, 0, 0], m.LEFT)
        readout = text(
            f"average loss on the answer: {LOSS_BEFORE.mean():.1f}  →  {LOSS_AFTER.mean():.1f}", 22, MUTED,
            t2c={f"{LOSS_BEFORE.mean():.1f}": BAD, f"{LOSS_AFTER.mean():.1f}": GOOD},
        ).move_to([0, -3.4, 0])

        self.say(
            "[Now, the training itself.] [Here's our example as tokens], [coloured by role.]",
            self.new_header("THE LOSS", "Learn only from the answer"),
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in plain], lag_ratio=0.03),
            m.ReplacementTransform(plain, chips),
        )
        self.say(
            "[The system and user tokens are only context]: [their loss is usually masked out.]",
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(masked, shift=0.15 * m.LEFT)),
            context.animate.set_opacity(0.3),
        )
        self.say(
            "[Only the assistant's tokens count]: [eighteen of the thirty-four], [including the end marker, so the model also learns when to stop.]",
            m.Indicate(counted, color=VIOLET, scale_factor=1.06),
            m.FadeIn(count, shift=0.15 * m.UP),
            m.AnimationGroup(m.Circumscribe(end_chip, color=VIOLET, buff=0.08), m.FadeIn(stop, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[For each answer token, the loss measures how surprised the model was], [and at first it's high]: [this model expected another question.]",
            m.FadeIn(loss_label, shift=0.15 * m.LEFT),
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in before], lag_ratio=0.06),
            m.FadeIn(surprised, shift=0.15 * m.LEFT),
        )
        self.say(
            "[Training nudges the weights] [until the answer becomes the likely continuation.]",
            m.AnimationGroup(m.Transform(before, after, run_time=1.6), m.FadeOut(surprised)),
            m.FadeIn(readout, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the classic recipe

    def recipe(self):
        self.next_section(
            "The classic recipe",
            notes="InstructGPT (Ouyang et al., 2022): GPT-3 → SFT on labeler demonstrations (~13k training prompts) → reward model + PPO. The 1.3B PPO-ptx model's outputs were preferred to 175B GPT-3's. Circle areas are proportional to parameters.",
        )
        stages = (
            (icon("llm", 0.62), "1 · Pretrain", "GPT-3", ("next-token prediction", "on web text"), *AST),
            (icon(variant("user", TEAL, TEAL_FILL), 0.62), "2 · Fine-tune", "supervised", ("labelers write answers", "for ≈ 13k prompts"), *TUNE),
            (icon("ep29-rank", 0.62), "3 · Preferences", "Episode 30", ("people rank answers;", "it learns what they prefer"), *PREF),
        )
        nodes = m.VGroup()
        for (pic, title, tag, desc, color, fill), x in zip(stages, (-4.6, 0, 4.6)):
            box = card(3.95, 2.95, color, CARD, stroke=2.5).move_to([x, 0.95, 0])
            head = m.VGroup(pic, text(title, 28, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.14)
            tag_pill = pill(tag, color, fill, 20)
            desc_lines = m.VGroup(*[text(d, 22, MUTED) for d in desc]).arrange(m.DOWN, buff=0.08)
            m.VGroup(head, tag_pill, desc_lines).arrange(m.DOWN, buff=0.18).move_to(box)
            nodes.add(m.VGroup(box, head, tag_pill, desc_lines))
        links = m.VGroup(*[arrow(a[0].get_right(), b[0].get_left(), MUTED, buff=0.1) for a, b in zip(nodes, nodes[1:])])
        cite = text("InstructGPT  ·  Ouyang et al., 2022", 20, MUTED).move_to([0, -0.82, 0])

        big_r = 1.0
        small_r = big_r * np.sqrt(SMALL_B / BIG_B)
        y = -2.4
        big = m.Circle(radius=big_r).set_stroke(VIOLET, width=3).set_fill(VIOLET_FILL, opacity=1).move_to([1.7, y, 0])
        small = m.Circle(radius=small_r).set_stroke(TEAL, width=2).set_fill(TEAL, opacity=1).move_to([-1.3, y, 0])
        small_label = m.VGroup(text("InstructGPT · 1.3B", 26, TEAL, weight=m.BOLD), text("after both steps", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
        small_label.next_to(small, m.LEFT, buff=0.35)
        preferred = pill("✓  preferred by people", GOOD, GOOD_FILL, 20).next_to(small_label, m.DOWN, buff=0.18).align_to(small_label, m.RIGHT)
        big_label = m.VGroup(text("GPT-3 · 175B", 26, VIOLET, weight=m.BOLD), text("the original, untuned", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        big_label.next_to(big, m.RIGHT, buff=0.35).shift(0.3 * m.UP)
        ratio = text(f"{BIG_B / SMALL_B:.0f}× the parameters", 22, INK, weight=m.SEMIBOLD).next_to(big_label, m.DOWN, buff=0.18).align_to(big_label, m.LEFT)
        area = text("circle area ∝ parameters", 20, MUTED).move_to([0, -3.5, 0]).align_to(big_label, m.LEFT)
        result = m.VGroup(big, small, small_label, preferred, big_label, ratio, area)
        result.shift(-result.get_x() * m.RIGHT)

        self.say(
            "[The classic recipe is InstructGPT, from twenty twenty-two.]",
            m.AnimationGroup(
                self.new_header("THE CLASSIC RECIPE", "InstructGPT: fine-tune, then prefer"),
                m.FadeIn(cite),
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(n[0]), m.FadeIn(n[1][1], shift=0.15 * m.UP)) for n in nodes], lag_ratio=0.25),
                m.FadeIn(links),
            ),
        )
        self.say(
            "[Start with a pretrained model.] [Fine-tune it on answers written by people]: [about thirteen thousand prompts.]",
            m.AnimationGroup(m.FadeIn(nodes[0][1][0], scale=0.8), m.FadeIn(nodes[0][2:], shift=0.15 * m.UP)),
            m.AnimationGroup(pulse(links[0], TEAL), m.FadeIn(nodes[1][1][0], scale=0.8), m.FadeIn(nodes[1][2], scale=1.15)),
            m.FadeIn(nodes[1][3], shift=0.1 * m.UP),
        )
        self.say(
            "[Then tune it on which answers people prefer]: [reinforcement learning from human feedback, coming in episode thirty.]",
            m.AnimationGroup(pulse(links[1], PINK), m.FadeIn(nodes[2][1][0], scale=0.8), m.FadeIn(nodes[2][3], shift=0.1 * m.UP)),
            m.FadeIn(nodes[2][2], scale=1.15),
        )
        self.say(
            "[After both steps], [people preferred answers from a model with one point three billion parameters] [over the original, with one hundred seventy-five billion]: [more than a hundred times larger.]",
            m.Succession(pulse(links[0], TEAL), pulse(links[1], PINK)),
            m.AnimationGroup(m.FadeIn(small, scale=0.3), m.FadeIn(small_label, shift=0.15 * m.RIGHT), m.FadeIn(preferred, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowFromCenter(big), m.FadeIn(big_label, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(ratio, shift=0.15 * m.UP), m.FadeIn(area)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ full or adapters

    def adapters(self):
        self.next_section(
            "Full fine-tuning or adapters",
            notes="Full fine-tuning updates every weight (memory for gradients and optimizer state; one full copy per task). Adapters freeze the base weights and train a small add-on. Example: a rank-8 low-rank adapter on a 4096×4096 matrix trains 65,536 numbers (0.39%). Episode 41: LoRA.",
        )
        rng = np.random.default_rng(41)
        full = panel(-3.45, "Full fine-tuning", TEAL, height=4.7, y=-0.3)
        light = panel(3.45, "Adapters", AMBER, height=4.7, y=-0.3)
        grid_l = weight_grid(7, 12, rng, MUTED).move_to([-3.45, -0.45, 0])
        retuned = [float(rng.uniform(0.35, 0.95)) for _ in grid_l]
        grid_r = weight_grid(7, 12, rng, MUTED).move_to([3.2, -0.45, 0])
        lock = icon("lock", 0.5).next_to(light[1], m.RIGHT, buff=0.3)
        strip_b = weight_grid(7, 1, rng, AMBER).next_to(grid_r, m.RIGHT, buff=0.22)
        strip_a = weight_grid(1, 12, rng, AMBER).next_to(grid_r, m.UP, buff=0.18)
        note_l = text("every weight changes", 22, INK, weight=m.SEMIBOLD).move_to([-3.45, -2.0, 0])
        cost_l = text("lots of memory · a full copy per task", 22, BAD).move_to([-3.45, -2.38, 0])
        note_r = text("originals frozen", 22, INK, weight=m.SEMIBOLD).move_to([3.45, -2.0, 0])
        note_r2 = text("a small add-on is trained", 22, AMBER).move_to([3.45, -2.38, 0])
        lora = pill("Episode 41 · LoRA", AMBER, AMBER_FILL, 20)
        lora.move_to([0, light[1].get_y(), 0]).align_to(light[0].get_right() - np.array([0.3, 0, 0]), m.RIGHT)
        nums = m.VGroup(
            text(f"one {D} × {D} matrix: {FULL_WEIGHTS:,} weights", 22, MUTED),
            text(f"→  a rank-{RANK} adapter: {ADAPTER_WEIGHTS:,} ({100 * ADAPTER_SHARE:.2f}%)", 22, AMBER, weight=m.SEMIBOLD),
        ).arrange(m.RIGHT, buff=0.35).move_to([0, -3.2, 0])

        self.say(
            "[One more choice]: [how much of the model do we change?]",
            self.new_header("FULL OR LIGHT", "How much of the model to change"),
            m.AnimationGroup(m.FadeIn(full), m.FadeIn(light), m.FadeIn(grid_l), m.FadeIn(grid_r)),
        )
        self.say(
            "[Full fine-tuning updates every weight.] [It's the most flexible], [but it needs lots of memory, and a full copy of the model for every task.]",
            m.LaggedStart(*[c.animate.set_fill(TEAL, opacity=o) for c, o in zip(grid_l, retuned)], lag_ratio=0.01, run_time=1.6),
            m.FadeIn(note_l, shift=0.15 * m.UP),
            m.FadeIn(cost_l, shift=0.15 * m.UP),
        )
        self.say(
            "[Adapters freeze the original weights], [and train a small add-on instead.]",
            m.AnimationGroup(grid_r.animate.set_fill(FAINT, opacity=0.6), m.FadeIn(lock, scale=1.3), m.FadeIn(note_r, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(strip_a, m.LEFT), m.GrowFromEdge(strip_b, m.UP), m.FadeIn(note_r2, shift=0.15 * m.UP)),
        )
        self.say(
            "[For example, on one large weight matrix], [an adapter can train under half a percent as many numbers.] [More on that in episode forty-one, on LoRA.]",
            m.FadeIn(nums[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(nums[1], shift=0.15 * m.UP), m.Indicate(m.VGroup(strip_a, strip_b), color=AMBER, scale_factor=1.06)),
            m.FadeIn(lora, scale=1.15),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section(
            "Two cautions",
            notes="Fine-tuning mostly shapes format and manner; most knowledge comes from pretraining (hedged: LIMA's 'superficial alignment' view). Narrow, long fine-tuning can cause forgetting of other skills. Facts as of October 2026.",
        )
        manner = tile(variant("book", TEAL, TEAL_FILL), "Mostly manner", "format and tone; the facts\nmostly come from pretraining", TEAL)
        forget = tile("alert", "It can forget", "too long on a narrow task,\nand other skills fade", BAD)
        m.VGroup(manner, forget).arrange(m.RIGHT, buff=0.45).move_to([0, 1.25, 0])

        ask = m.VGroup(icon("ep29-egg", 0.55), chip(QUESTION, SKY, SKY_FILL, width=3.6, height=0.66, size=24)).arrange(m.RIGHT, buff=0.25)
        ask.move_to([0, -0.55, 0])
        before_words = lines(BASE_REPLY[:2], 22, MUTED)
        before_box = card(before_words.width + 0.5, before_words.height + 0.45, EDGE, CARD, radius=0.2, stroke=2)
        before = m.VGroup(before_box, before_words.move_to(before_box)).move_to([-3.6, -2.5, 0])
        after_words = lines(TUNED_REPLY, 22, INK)
        after_box = card(after_words.width + 0.5, after_words.height + 0.45, VIOLET, VIOLET_FILL, radius=0.2, stroke=2)
        after = m.VGroup(after_box, after_words.move_to(after_box)).move_to([3.6, -2.5, 0])
        go_left = arrow(ask[1].get_bottom() + np.array([-1.2, 0, 0]), before.get_top() + np.array([0.7, 0, 0]), MUTED, buff=0.12)
        go_right = arrow(ask[1].get_bottom() + np.array([1.2, 0, 0]), after.get_top() + np.array([-0.7, 0, 0]), TEAL, buff=0.12)
        before_tag = text("before", 22, MUTED, weight=m.SEMIBOLD).next_to(before, m.LEFT, buff=0.25).align_to(before, m.UP)
        after_tag = text("after", 22, TEAL, weight=m.SEMIBOLD).next_to(after, m.LEFT, buff=0.25).align_to(after, m.UP)

        self.say(
            "[Two cautions.] [Fine-tuning mostly shapes format and manner]: [most of the knowledge comes from pretraining.]",
            self.new_header("THE CATCH", "What fine-tuning does, and doesn't"),
            m.AnimationGroup(m.FadeIn(manner[0]), m.FadeIn(manner[1][0], scale=0.8), m.FadeIn(manner[1][1][0], shift=0.15 * m.UP)),
            m.FadeIn(manner[1][1][1], shift=0.15 * m.UP),
        )
        self.say(
            "[And train too long on a narrow task], [and the model can forget other skills.]",
            m.AnimationGroup(m.FadeIn(forget[0]), m.FadeIn(forget[1][0], scale=0.8), m.FadeIn(forget[1][1][0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(forget[1][1][1], shift=0.15 * m.UP), m.Circumscribe(forget[1][0], color=BAD, buff=0.1)),
        )
        self.say(
            "[So, before]: [a text predictor that met a question with more questions.] [After]: [an assistant that answers it.]",
            m.FadeIn(ask, shift=0.2 * m.DOWN),
            m.AnimationGroup(m.GrowArrow(go_left), m.FadeIn(before_tag), m.FadeIn(before, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(go_right), m.FadeIn(after_tag)),
            m.FadeIn(after, shift=0.15 * m.DOWN),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Examples, templates, answers.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Examples", "Templates", "Answers"), (TEAL, AMBER, VIOLET))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Show a text predictor good answers, in the right format, and it becomes an assistant.", 26, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [examples], [templates], [answers.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Show a text predictor good answers, in the right format, and it becomes an assistant.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
