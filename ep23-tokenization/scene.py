"""AI, Explained · Episode 23 · Tokenization

Running example: the word "unbelievable", and the tokens a language model reads instead of it.
Characters (a tiny vocabulary, long sequences), whole words (a huge vocabulary, unknown words)
and subwords in between; byte-pair encoding (BPE) run for real on a five-word toy corpus, merge
by merge, with the pair counts it chooses by; the vocabulary-size trade-off measured on that
same corpus; what tokens cost (prices, context windows, other languages, code); and why letter
puzzles trip models up.

Everything the toy tokenizer shows is computed below (learn_bpe); the narration quotes its
numbers and the asserts keep them honest. Ties between equally frequent pairs go to the pair
found first, scanning the corpus in order (only later, fast-forwarded merges have ties).
Real-tokenizer facts were measured with tiktoken 0.14.0 (encodings gpt2, cl100k_base,
o200k_base) on 2026-10-21 and are kept as constants.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

CHAR, CHAR_FILL = SKY, SKY_FILL  # one token per character
WORDS, WORDS_FILL = PINK, PINK_FILL  # one token per word
SUB, SUB_FILL = VIOLET, VIOLET_FILL  # subword tokens: the episode's answer
COUNT, COUNT_FILL = AMBER, AMBER_FILL  # counts: word frequencies, pair counts, costs

# ---------------------------------------------------------------------------- the toy tokenizer

CORPUS = [("unable", 8), ("believe", 9), ("unkind", 6), ("label", 7), ("reliable", 4)]  # illustrative counts
WORD, TYPO, GIBBERISH = "unbelievable", "unbelieveable", "xqz"


def pair_counts(rows):
    """count(a, b) = sum over words w of f(w) * n_w(a, b): every neighbouring pair, weighted by word count."""
    counts = Counter()
    for tokens, f in rows:
        for pair in zip(tokens, tokens[1:]):
            counts[pair] += f
    return counts


def merge(tokens, pair):
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def learn_bpe(corpus):
    """Byte-pair encoding on words: start from characters, merge the most frequent pair, repeat
    until every word is one token. Returns the merges, the corpus after each, and each ranking."""
    rows = [(list(w), f) for w, f in corpus]
    merges, states, rankings = [], [rows], []
    while True:
        counts = pair_counts(rows)
        if not counts:
            break
        ranking = sorted(counts.items(), key=lambda kv: -kv[1])  # stable: a tie goes to the pair found first
        pair, n = ranking[0]
        merges.append((pair, n))
        rankings.append(ranking)
        rows = [(merge(t, pair), f) for t, f in rows]
        states.append(rows)
    return merges, states, rankings


def tokenize(word, merges):
    """Apply learned merges in order; also return every distinct intermediate split."""
    tokens, steps = list(word), [list(word)]
    for pair, _ in merges:
        new = merge(tokens, pair)
        if new != tokens:
            steps.append(new)
        tokens = new
    return tokens, steps


def pair_marks(tokens, pair):
    marks, i = set(), 0
    while i < len(tokens) - 1:
        if (tokens[i], tokens[i + 1]) == pair:
            marks |= {i, i + 1}
            i += 2
        else:
            i += 1
    return marks


MERGES, STATES, RANKINGS = learn_bpe(CORPUS)
N_LEARN = 9  # the merges the film shows
LEARNED = MERGES[:N_LEARN]
LETTERS = sorted(set("".join(w for w, _ in CORPUS)))
VOCAB = LETTERS + ["".join(p) for p, _ in LEARNED]
TOKEN_ID = {t: i for i, t in enumerate(VOCAB)}
SPLIT, SPLIT_STEPS = tokenize(WORD, LEARNED)
TYPO_SPLIT, _ = tokenize(TYPO, LEARNED)
TOY_IDS = [TOKEN_ID[t] for t in SPLIT]
LENGTHS = [sum(len(t) * f for t, f in rows) for rows in STATES]  # corpus length in tokens after k merges
EL_TERMS = [(w, f, sum(1 for p in zip(w, w[1:]) if p == ("e", "l"))) for w, f in CORPUS if "el" in w]

assert len(WORD) == 12 and WORD.count("e") == 3  # "twelve letters", "count three"
assert all(w not in (WORD, TYPO) for w, _ in CORPUS) and len(CORPUS) == 5  # "five words", never "unbelievable"
assert len(LETTERS) == 11 and len(VOCAB) == 20  # "eleven tokens"; nine merges later, twenty
assert [n for _, n in MERGES[:3]] == [20, 19, 14] and MERGES[0][0] == ("e", "l")  # "twenty times"
assert all(RANKINGS[k][0][1] > RANKINGS[k][1][1] for k in range(3))  # the three detailed merges have clear winners
assert [(f, n) for _, f, n in EL_TERMS] == [(9, 1), (7, 1), (4, 1)] and sum(f * n for _, f, n in EL_TERMS) == 20
assert SPLIT == ["un", "believ", "able"]  # "a prefix, a stem and a suffix", "three tokens"
assert TYPO_SPLIT == ["un", "believ", "e", "able"]  # "one more piece"
assert all("".join(t) == w for (t, _), (w, _) in zip(STATES[N_LEARN], CORPUS))
assert any(t == ["un", "able"] for t, _ in STATES[N_LEARN]) and any(t[0] == "believ" for t, _ in STATES[N_LEARN])
assert LENGTHS[0] == 214 and 0.4 < LENGTHS[N_LEARN] / LENGTHS[0] < 0.5  # "two hundred and fourteen", "less than half as long"
assert len(MERGES) == 19 and LENGTHS[-1] == sum(f for _, f in CORPUS)  # at the end, every word is one token
assert all(b < a for a, b in zip(LENGTHS, LENGTHS[1:]))  # every merge shortens the corpus

# ---------------------------------------------------------------------------- real tokenizers (tiktoken 0.14.0, 2026-10-21)

REAL_SPLITS = [  # "unbelievable" at the start of a text (after a space, all three keep it whole)
    ("GPT-2", "2019", ["un", "bel", "iev", "able"]),
    ("GPT-4", "2023", ["un", "belie", "vable"]),
    ("GPT-4o", "2024", ["un", "bel", "ievable"]),
]
GPT2_IDS = [403, 6667, 11203, 540]
VOCAB_SIZES = [("GPT-2", 50257, "50,257", "2019"), ("GPT-4", 100277, "≈ 100k", "2023"), ("GPT-4o", 200019, "≈ 200k", "2024")]
ENGLISH_GPT2 = ["That", "␣is", "␣unbelievable", "!"]  # "That is unbelievable!"
GERMAN_GPT2 = ["D", "as", "␣is", "t", "␣un", "gl", "a", "ub", "lich", "!"]  # "Das ist unglaublich!"
LANGUAGES = [("English", 4, 4), ("German", 10, 4), ("Greek", 26, 9), ("Hindi", 29, 8)]  # (GPT-2, GPT-4o) tokens
INDENT_GPT2 = ["␣", "␣", "␣", "␣return", "␣a", "␣+", "␣b"]  # "    return a + b"
INDENT_GPT4O = ["␣␣␣", "␣return", "␣a", "␣+", "␣b"]
CODE_TOTALS = (16, 12)  # the whole two-line function, GPT-2 and GPT-4o

for _, _, parts in REAL_SPLITS:
    assert "".join(parts) == WORD
assert len(GPT2_IDS) == len(REAL_SPLITS[0][2])
assert "".join(ENGLISH_GPT2).replace("␣", " ") == "That is unbelievable!" and len(ENGLISH_GPT2) == LANGUAGES[0][1] == 4
assert "".join(GERMAN_GPT2).replace("␣", " ") == "Das ist unglaublich!" and len(GERMAN_GPT2) == LANGUAGES[1][1] == 10
assert min(n for name, n, _ in LANGUAGES if name in ("Greek", "Hindi")) > 25  # "more than twenty-five"
assert "".join(INDENT_GPT2) == "".join(INDENT_GPT4O) == "␣␣␣␣return␣a␣+␣b" and INDENT_GPT2[:3] == ["␣"] * 3
assert VOCAB_SIZES[0][1] == 50257  # "fifty thousand, two hundred and fifty-seven"

# ---------------------------------------------------------------------------- helpers

_ADV48 = (code("x" * 10, 48).width - code("x", 48).width) / 9  # one monospace character at size 48


def adv(size):
    return _ADV48 * size / 48


def baseline_of(center_y, size):
    return center_y - 0.115 * size / 26


def sit(label, s, baseline):
    """Put a Text's letters on a baseline (descenders and symbols excluded from the measure)."""
    chars = [c for c in s if not c.isspace()]
    sitting = [g for g, c in zip(label, chars) if c not in "gjpqy,␣()[]+=-_'\"…⋮"]
    if sitting:
        label.shift((baseline - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    return label


def hide(mob):
    mob.set_stroke(opacity=0)
    mob.set_fill(opacity=0)
    return mob


def spaces(n, size, color=INK):
    """Visible spaces (⎵ marks), drawn as strokes: a space glyph is too small to read."""
    w = adv(size)
    marks = m.VGroup()
    for i in range(n):
        x0, x1, y0, y1 = i * w + 0.18 * w, i * w + 0.82 * w, 0.0, 0.1 * size / 26
        marks.add(m.VMobject().set_points_as_corners([[x0, y1, 0], [x0, y0, 0], [x1, y0, 0], [x1, y1, 0]]).set_stroke(color, width=3))
    return marks


def token_chip(t, color=SUB, fill=SUB_FILL, size=26, h=0.56, pad=None, ink=INK):
    """A token in a rounded box, sized by its character count, letters on a common baseline."""
    pad = 0.3 * size / 26 if pad is None else pad
    box = card(len(t) * adv(size) + pad, h, color, fill, radius=0.12, stroke=2.2).set_z_index(1)
    if set(t) == {"␣"}:
        label = spaces(len(t), size, ink).move_to(box).shift(0.08 * size / 26 * m.DOWN)
    else:
        label = code(t, size, ink).move_to(box)
        sit(label, t, baseline_of(box.get_center()[1], size))
    return m.VGroup(box, label.set_z_index(2))


def sub_style(t):
    return (CHAR, CHAR_FILL) if len(t) == 1 else (SUB, SUB_FILL)


class Slots:
    """A word drawn as one slot per character: a box and the character. Any split of the word into
    tokens is a layout of the slots: a token's first slot carries its visible box, its other slots
    an invisible copy, and every character stays in its own slot. So merges are plain Transforms
    between layouts, and many can be chained in one line."""

    def __init__(self, word, y, x=0.0, anchor="center", size=26, h=0.56, gap=0.08, style=sub_style, tokens=None, plain=False):
        self.word, self.y, self.x, self.anchor = word, y, x, anchor
        self.size, self.h, self.gap, self.style = size, h, gap, style
        self.tokens = list(tokens) if tokens else list(word)
        self.mob = m.VGroup(*(self.plain_targets() if plain else self.targets(self.tokens)))

    def place(self, row):
        row.set_y(self.y)
        if self.anchor == "left":
            row.align_to([self.x, 0, 0], m.LEFT)
        else:
            row.set_x(self.x)
        return row

    def chips(self, tokens, style=None, marks=(), mark=COUNT):
        style = style or self.style
        out = []
        for i, t in enumerate(tokens):
            color, fill = style(t)
            out.append(token_chip(t, mark if i in marks else color, fill, self.size, self.h))
        self.place(m.VGroup(*out).arrange(m.RIGHT, buff=self.gap))
        return out

    def targets(self, tokens, style=None, marks=(), mark=COUNT):
        assert "".join(tokens) == self.word, (tokens, self.word)
        parts = []
        for (box, label), t in zip(self.chips(tokens, style, marks, mark), tokens):
            for k, c in enumerate(t):
                b = box.copy()
                if k:
                    hide(b)
                parts.append(m.VGroup(b, code(c, self.size, INK).move_to(label[k]).set_z_index(2)))
        return parts

    def plain_targets(self):
        label = self.place(code(self.word, self.size, INK))
        sit(label, self.word, baseline_of(self.y, self.size))
        parts = []
        for k, c in enumerate(self.word):
            box = hide(token_chip(c, *self.style(c), self.size, self.h)[0]).move_to([label[k].get_x(), self.y, 0])
            parts.append(m.VGroup(box, code(c, self.size, INK).move_to(label[k]).set_z_index(2)))
        return parts

    def to(self, tokens=None, style=None, marks=(), mark=COUNT, plain=False, **kw):
        if plain:
            new = self.plain_targets()
        else:
            tokens = list(tokens or self.tokens)
            new = self.targets(tokens, style, marks, mark)
            self.tokens = tokens
        return m.AnimationGroup(*[m.Transform(s, t) for s, t in zip(self.mob, new)], **kw)

    def first(self, i):
        return sum(len(t) for t in self.tokens[:i])

    def token(self, i):
        a = self.first(i)
        return self.mob[a : a + len(self.tokens[i])]

    def center(self, i):
        return self.mob[self.first(i)][0].get_center()


def chip_row(tokens, y, x, anchor="left", size=24, h=0.52, gap=0.06, color=SUB, fill=SUB_FILL, marks=(), mark=COUNT):
    row = m.VGroup(*[token_chip(t, mark if i in marks else color, fill, size, h) for i, t in enumerate(tokens)])
    row.arrange(m.RIGHT, buff=gap).set_y(y)
    return row.align_to([x, 0, 0], m.LEFT) if anchor == "left" else row.set_x(x)


def right_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def cost_tile(name, title, desc, color, width=6.1, height=1.75):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 28, INK, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
    glyph = icon(name, 0.95).move_to(box.get_left() + 0.85 * m.RIGHT)
    words.move_to(box).align_to(box.get_left() + 1.65 * m.RIGHT, m.LEFT)
    return m.VGroup(box, glyph, words)


# ---------------------------------------------------------------------------- the episode


class Ep23Tokenization(Episode):
    number = 23

    def lesson(self):
        self.hook()
        self.characters()
        self.words()
        self.subwords()
        self.bpe()
        self.vocab()
        self.costs()
        self.failures()
        self.recap()

    def regroup(self, *groups):
        """Pieces added one by one by an animation → one group per slot row (no visible change)."""
        for group in groups:
            for slot in group:
                self.remove(*slot)
            self.add(group)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes=f"The toy tokenizer built later in this episode splits 'unbelievable' as {' | '.join(SPLIT)}, "
            f"IDs {TOY_IDS} in its 20-token vocabulary (11 letters + 9 merges).",
        )
        cx = -2.3
        box = card(4.2, 1.0, SKY, SKY_FILL, radius=0.3).move_to([cx, 1.85, 0])
        typed = code(WORD, 34, INK).move_to(box)
        sit(typed, WORD, baseline_of(1.85, 34))
        chips = Slots(WORD, y=0.3, x=cx, size=30, h=0.66, gap=0.14, tokens=SPLIT)
        centers = [chips.center(i) for i in range(len(SPLIT))]
        ids = m.VGroup(*[token_chip(str(n), EDGE, CARD, 28, 0.62).move_to([c[0], -1.15, 0]) for n, c in zip(TOY_IDS, centers)])
        downs = m.VGroup(*[arrow([c[0], -0.08, 0], [c[0], -0.78, 0], MUTED, buff=0.04) for c in centers])
        captions = m.VGroup(*[text(s, 24, MUTED, weight=m.SEMIBOLD) for s in ("text", "tokens", "IDs")])
        for cap, y in zip(captions, (1.85, 0.3, -1.15)):
            right_at(cap, -4.75, y)
        model = icon("llm", 1.5).move_to([4.5, -1.15, 0])
        ask = text("?", 60, SKY, weight=m.BOLD).next_to(model, m.UP, buff=0.3)
        feed = arrow([ids.get_right()[0], -1.15, 0], model.get_left(), MUTED, buff=0.3)
        footnote = text("IDs from a toy vocabulary we'll build later in this episode", 20, MUTED).move_to([cx + 0.6, -1.9, 0])
        cut = arrow([0.3, 1.6, 0], [0.3, 0.6, 0], SUB, buff=0)
        cut_label = text("tokenization", 32, SUB, weight=m.BOLD).next_to(cut, m.RIGHT, buff=0.25)
        matters = m.VGroup(
            pill("what a model can do", GOOD, GOOD_FILL, 22), pill("what it costs", COUNT, COUNT_FILL, 22)
        ).arrange(m.RIGHT, buff=0.45).move_to([0, -2.9, 0])

        self.say(
            "[Type unbelievable into a chatbot.] [What does the language model actually read?]",
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(captions[0]), m.AddTextLetterByLetter(typed, time_per_char=0.06)),
            m.AnimationGroup(m.FadeIn(model, shift=0.2 * m.LEFT), m.FadeIn(ask, scale=0.6)),
        )
        self.say(
            "[It reads tokens]: [chunks of text, like these], [each one turned into an ID number.]",
            m.FadeIn(captions[1], shift=0.2 * m.RIGHT),
            m.AnimationGroup(
                *[m.TransformFromCopy(typed[i], chips.mob[i][1]) for i in range(len(WORD))],
                *[m.FadeIn(slot[0]) for slot in chips.mob],
            ),
            m.AnimationGroup(
                m.FadeIn(captions[2]),
                m.LaggedStart(*[m.GrowArrow(d) for d in downs], lag_ratio=0.2),
                m.LaggedStart(*[m.FadeIn(i, shift=0.2 * m.DOWN) for i in ids], lag_ratio=0.2),
                m.FadeIn(footnote),
            ),
        )
        self.regroup(chips.mob)
        self.say(
            "[How we cut text into tokens is called tokenization.] [It shapes what a model can do, and what it costs.]",
            m.AnimationGroup(m.GrowArrow(cut), m.Write(cut_label)),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.UP) for p in matters], lag_ratio=0.3),
                m.Succession(m.GrowArrow(feed), pulse(feed, SUB)),
                m.FadeOut(ask),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ option 1: characters

    def characters(self):
        self.next_section(
            "Option 1: characters",
            notes="Printable ASCII alone is 95 symbols, hence 'around a hundred'. A page of English is roughly "
            "500 words, about 3,000 characters: 'thousands' of tokens.",
        )
        word = Slots(WORD, y=1.75, x=0, size=34, h=0.74, gap=0.1, style=lambda t: (CHAR, CHAR_FILL), plain=True)
        count = text("12 tokens", 26, CHAR, weight=m.SEMIBOLD).move_to([0, 0.95, 0])

        left = card(6.0, 3.3, EDGE, CARD, radius=0.22, stroke=2).move_to([-3.25, -1.3, 0])
        left_title = m.VGroup(text("Vocabulary", 28, INK, weight=m.BOLD), pill("tiny", GOOD, GOOD_FILL, 20)).arrange(m.RIGHT, buff=0.3)
        left_at(left_title, -5.85, -0.0)
        symbols = m.VGroup(*[code(s, 24, MUTED) for s in ("a b c d … w x y z", "A B C D … W X Y Z", "0 1 2 3 … 6 7 8 9", ". , ! ? ; : ( ) …")])
        for s, y in zip(symbols, (-0.6, -1.05, -1.5, -1.95)):
            left_at(s, -5.85, y)
        hundred = text("around 100 tokens", 24, GOOD, weight=m.SEMIBOLD)
        left_at(hundred, -5.85, -2.5)

        right = card(6.0, 3.3, EDGE, CARD, radius=0.22, stroke=2).move_to([3.25, -1.3, 0])
        right_title = m.VGroup(text("Sequence length", 28, INK, weight=m.BOLD), pill("long", BAD, BAD_FILL, 20)).arrange(m.RIGHT, buff=0.3)
        left_at(right_title, 0.65, -0.0)
        per_word = m.VGroup(text("1 word", 26, MUTED), text("→  12 tokens", 26, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.3)
        per_page = m.VGroup(text("1 page", 26, MUTED), text("→  thousands", 26, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.3)
        left_at(per_word, 0.65, -0.65)
        left_at(per_page, 0.65, -1.3)
        rng = np.random.default_rng(23)
        ribbon = m.VGroup()
        for r in range(3):
            for c in range(26):
                sq = m.RoundedRectangle(corner_radius=0.025, width=0.15, height=0.15, stroke_width=0)
                sq.set_fill(CHAR, opacity=float(rng.uniform(0.25, 0.85))).move_to([0.73 + c * 0.2, -1.88 - r * 0.22, 0])
                ribbon.add(sq)

        self.say(
            "[Option one]: [give every character its own token.]",
            m.AnimationGroup(self.new_header("OPTION 1 · CHARACTERS", "One token per character"), m.FadeIn(word.mob)),
            m.AnimationGroup(word.to(list(WORD)), m.FadeIn(count, shift=0.15 * m.UP)),
        )
        self.say(
            "[The vocabulary is tiny]: [around a hundred letters, digits and symbols.]",
            m.AnimationGroup(m.FadeIn(left), m.FadeIn(left_title, shift=0.15 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.RIGHT) for s in symbols], lag_ratio=0.25), m.FadeIn(hundred)),
        )
        self.say(
            "[But sequences get long]: [our one word takes twelve tokens], [and a single page, thousands.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(right_title, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(per_word, shift=0.15 * m.RIGHT), m.Indicate(count, color=CHAR, scale_factor=1.2)),
            m.AnimationGroup(m.FadeIn(per_page, shift=0.15 * m.RIGHT), m.LaggedStart(*[m.FadeIn(s) for s in ribbon], lag_ratio=0.02)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ option 2: words

    def words(self):
        self.next_section(
            "Option 2: words",
            notes="Large English dictionaries list hundreds of thousands of word forms (the OED about 600,000). "
            "A word-level model maps anything outside its list to one unknown token, often written [UNK].",
        )
        x = -3.6
        word = Slots(WORD, y=1.65, x=x, size=32, h=0.72, style=lambda t: (WORDS, WORDS_FILL), plain=True)
        one = text("1 token", 24, WORDS, weight=m.SEMIBOLD).move_to([x, 0.9, 0])
        typo = code(TYPO, 32, INK).move_to([x, -0.3, 0])
        sit(typo, TYPO, baseline_of(-0.3, 32))
        typo[8].set_color(BAD)  # the extra e
        typo_name = text("a typo", 22, MUTED).next_to(typo, m.UP, buff=0.2)
        down = arrow([x, -0.7, 0], [x, -1.4, 0], BAD, buff=0)
        unk = token_chip("[UNK]", BAD, BAD_FILL, 30, 0.72).move_to([x, -1.85, 0])
        lost = text("its meaning is lost", 24, BAD, weight=m.SEMIBOLD).move_to([x, -2.75, 0])

        frame = card(6.0, 4.9, WORDS, CARD, radius=0.22, stroke=2).move_to([3.35, -0.25, 0])
        title = left_at(text("Word vocabulary", 26, INK, weight=m.BOLD), 0.75, 1.75)
        entries = ["a", "aardvark", "able", "⋮", "unbearable", WORD, "unbend", "⋮", "zygote"]
        listing = m.VGroup()
        for i, e in enumerate(entries):
            entry = code(e, 24, WORDS if e == WORD else MUTED)
            listing.add(left_at(entry, 0.85, 1.15 - 0.42 * i))
        big = m.VGroup(text("hundreds of", 32, INK, weight=m.BOLD), text("thousands", 32, INK, weight=m.BOLD), text("of English words", 22, MUTED))
        big.arrange(m.DOWN, buff=0.12).move_to([4.85, 0.75, 0])
        extras = m.VGroup(*[pill(f"+ {s}", WORDS, WORDS_FILL, 20) for s in ("names", "numbers", "new slang")])
        extras.arrange(m.DOWN, buff=0.22).move_to([4.85, -1.35, 0])

        self.say(
            "[Option two: one token per word.] [Now unbelievable is a single token.]",
            m.AnimationGroup(self.new_header("OPTION 2 · WORDS", "One token per word"), m.FadeIn(word.mob)),
            m.AnimationGroup(word.to([WORD]), m.FadeIn(one, shift=0.15 * m.UP)),
        )
        self.say(
            "[But the vocabulary explodes]: [English alone has hundreds of thousands of words], [plus names, numbers and new slang.]",
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(title)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(e, shift=0.1 * m.DOWN) for e in listing], lag_ratio=0.12), m.FadeIn(big, shift=0.2 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in extras], lag_ratio=0.3),
        )
        self.say(
            "[And a word that's missing from the list, like this typo], [becomes one unknown token]: [its meaning is lost.]",
            m.AnimationGroup(m.FadeIn(typo, shift=0.2 * m.DOWN), m.FadeIn(typo_name), m.Indicate(listing[4:7], color=BAD, scale_factor=1.05)),
            m.AnimationGroup(m.GrowArrow(down), m.FadeIn(unk, scale=0.7)),
            m.AnimationGroup(m.Wiggle(unk), m.FadeIn(lost, shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the middle: subwords

    def subwords(self):
        self.next_section(
            "Subwords",
            notes=f"Toy tokenizer splits (computed): {WORD} → {SPLIT}; the typo {TYPO} → {TYPO_SPLIT}. "
            "Byte-level tokenizers (GPT-2 onwards) can always fall back to single bytes, so no input is ever unknown.",
        )
        ends = m.VGroup(
            text("characters", 24, CHAR, weight=m.SEMIBOLD), text("subwords", 26, SUB, weight=m.BOLD), text("words", 24, WORDS, weight=m.SEMIBOLD)
        )
        for e, xx in zip(ends, (-4.4, 0, 4.4)):
            e.move_to([xx, 2.2, 0])
        links = m.VGroup(
            m.Line([ends[0].get_right()[0] + 0.25, 2.2, 0], [ends[1].get_left()[0] - 0.25, 2.2, 0]).set_stroke(FAINT, width=3),
            m.Line([ends[1].get_right()[0] + 0.25, 2.2, 0], [ends[2].get_left()[0] - 0.25, 2.2, 0]).set_stroke(FAINT, width=3),
        )
        labels = m.VGroup(*[text(s, 24, MUTED, weight=m.SEMIBOLD) for s in ("common words", "rarer words", "a typo")])
        for lab, y in zip(labels, (1.05, -0.45, -2.15)):
            right_at(lab, -4.0, y)
        common = chip_row(["the", "and", "is", "believe"], 1.05, -3.7, size=28, h=0.62, gap=0.25)
        each = text("1 token each", 22, MUTED).next_to(common, m.RIGHT, buff=0.4)
        rare = Slots(WORD, y=-0.45, x=-3.7, anchor="left", size=28, h=0.62, gap=0.16, plain=True)
        typo = Slots(TYPO, y=-2.15, x=-3.7, anchor="left", size=28, h=0.62, gap=0.16, plain=True)
        rare_split = Slots(WORD, y=-0.45, x=-3.7, anchor="left", size=28, h=0.62, gap=0.16, tokens=SPLIT)
        parts = m.VGroup(*[text(s, 22, MUTED).move_to([rare_split.center(i)[0], -1.12, 0]) for i, s in enumerate(("prefix", "stem", "suffix"))])
        pieces = text(f"{len(TYPO_SPLIT)} tokens", 22, MUTED)
        safe = pill("nothing is ever unknown", GOOD, GOOD_FILL, 20).move_to([4.55, -1.85, 0])
        gib = chip_row(list(GIBBERISH), -0.45, 4.55, anchor="center", size=28, h=0.62, gap=0.1, color=CHAR, fill=CHAR_FILL)
        gib_label = text("made-up letters", 22, MUTED).move_to([4.55, -1.12, 0])

        self.say(
            "[Subwords sit in between.] [Common words get a token of their own],",
            m.AnimationGroup(self.new_header("THE MIDDLE · SUBWORDS", "Pieces of words"), m.FadeIn(ends), m.Create(links)),
            m.AnimationGroup(m.FadeIn(labels[0]), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in common], lag_ratio=0.2), m.FadeIn(each)),
        )
        self.say(
            "[while rarer words are built from familiar pieces]: [a prefix], [a stem], [and a suffix.]",
            m.AnimationGroup(m.FadeIn(labels[1]), m.Succession(m.FadeIn(rare.mob, run_time=0.5), rare.to(SPLIT))),
            m.FadeIn(parts[0], shift=0.15 * m.UP),
            m.FadeIn(parts[1], shift=0.15 * m.UP),
            m.FadeIn(parts[2], shift=0.15 * m.UP),
        )
        self.say(
            "[Even our typo works now]: [it just takes one more piece.]",
            m.AnimationGroup(m.FadeIn(labels[2]), m.FadeIn(typo.mob)),
            m.AnimationGroup(typo.to(TYPO_SPLIT), m.FadeIn(pieces.next_to(typo.targets(TYPO_SPLIT)[-1], m.RIGHT, buff=0.4))),
        )
        self.say(
            "[Nothing is ever unknown]: [in the worst case, a word falls back to single characters.]",
            m.FadeIn(safe, shift=0.2 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.DOWN) for c in gib], lag_ratio=0.2), m.FadeIn(gib_label)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ byte-pair encoding

    def bpe_rows(self, rows, k, marks=False, mark=COUNT, style=None):
        """Every corpus row that merge k changes: highlight its pair (marks) or merge it."""
        pair = MERGES[k][0]
        anims = []
        for r, (before, _), (after, _) in zip(rows, STATES[k], STATES[k + 1]):
            assert r.tokens == before
            if before != after:
                anims.append(r.to(before, marks=pair_marks(before, pair), mark=mark) if marks else r.to(after))
        return m.AnimationGroup(*anims)

    def bars(self, k):
        rows = m.VGroup()
        for j, ((a, b), n) in enumerate(RANKINGS[k][:4]):
            y = (-0.72, -1.15, -1.58, -2.01)[j]
            color = COUNT if j == 0 else FAINT
            pair = m.VGroup(token_chip(a, *sub_style(a), 24, 0.4, pad=0.18), token_chip(b, *sub_style(b), 24, 0.4, pad=0.18))
            pair.arrange(m.RIGHT, buff=0.05)
            left_at(pair, 0.55, y)
            bar = m.Rectangle(width=n / 20 * 3.0, height=0.24, stroke_width=0).set_fill(color, opacity=0.9)
            left_at(bar, 2.6, y)
            num = code(str(n), 22, COUNT if j == 0 else MUTED).next_to(bar, m.RIGHT, buff=0.15)
            rows.add(m.VGroup(pair, bar, num))
        return rows

    def bpe(self):
        self.next_section(
            "Byte-pair encoding",
            notes="BPE for words: Sennrich et al., 2016 (from Gage's 1994 compression method). Toy corpus counts are "
            f"illustrative. Merges learned (pair, count): {[(''.join(p), n) for p, n in LEARNED]}. "
            "count(a, b) = Σ_w f(w)·n_w(a, b). Ties go to the pair found first (merges 5 and 7–9 have ties). "
            "Real splits of 'unbelievable' measured with tiktoken 0.14.0 on 2026-10-21 (gpt2, cl100k_base for GPT-4, "
            "o200k_base for GPT-4o); with a leading space all three keep it as one token.",
        )
        left_x = -6.3
        ys = (1.45, 0.7, -0.05, -0.8, -1.55)
        frame = card(6.3, 4.4, EDGE, CARD, radius=0.22, stroke=2).move_to([-3.6, 0.22, 0])
        title = m.VGroup(text("toy corpus", 24, INK, weight=m.BOLD), text("illustrative counts", 20, MUTED))
        title.arrange(m.RIGHT, buff=0.25)
        on_baseline(title)
        left_at(title, left_x, 2.05)
        rows = [Slots(w, y=y, x=left_x, anchor="left", size=24, h=0.5, gap=0.07, plain=True) for (w, _), y in zip(CORPUS, ys)]
        freqs = m.VGroup(*[code(f"×{f}", 24, COUNT).move_to([-1.15, y, 0]) for (_, f), y in zip(CORPUS, ys)])
        cite = right_at(text("Sennrich et al., 2016", 20, MUTED), 6.6, 2.62)

        strip_y, learned_y, strip_x = -2.62, -3.2, -4.75
        strip_label = left_at(text("vocabulary", 22, MUTED, weight=m.SEMIBOLD), left_x, strip_y)
        letters = chip_row(LETTERS, strip_y, strip_x, size=24, h=0.44, gap=0.06, color=CHAR, fill=CHAR_FILL)
        vcount = text(f"{len(LETTERS)} tokens", 22, CHAR, weight=m.SEMIBOLD).next_to(letters, m.RIGHT, buff=0.35)
        learned = chip_row(["".join(p) for p, _ in LEARNED], learned_y, strip_x, size=24, h=0.44, gap=0.06)
        vcount_after = text(f"{len(VOCAB)} tokens", 22, SUB, weight=m.SEMIBOLD).move_to(vcount, aligned_edge=m.LEFT)

        formula = m.MathTex(r"\text{count}(a,b)", "=", r"\sum_{w}", "f(w)", r"\cdot", "n_w(a,b)", font_size=36, color=INK)
        formula.move_to([3.6, 1.95, 0])
        formula[3].set_color(COUNT)
        formula[5].set_color(CHAR)

        def legend_row(sym, desc, color, y):
            return m.VGroup(left_at(tex(sym, 30, color), 0.6, y), left_at(text(desc, 22, MUTED), 2.05, y))

        legend = m.VGroup(
            legend_row("f(w)", "how often word w appears", COUNT, 1.15),
            legend_row("n_w(a,b)", "times the pair sits inside w", CHAR, 0.73),
            legend_row(r"\Sigma", "add it up over every word", INK, 0.31),
        )
        worked = tex(r"\text{count}(e,l) = 9\cdot 1 + 7\cdot 1 + 4\cdot 1 = 20", 30).move_to([3.6, -0.2, 0])
        bars = [self.bars(k) for k in range(3)]

        new_title = text("a word the corpus never saw", 22, MUTED).move_to([3.6, 1.95, 0])
        new = Slots(WORD, y=1.3, x=3.6, size=22, h=0.46, gap=0.06)
        three = text(f"{len(SPLIT)} tokens", 24, SUB, weight=m.SEMIBOLD).move_to([3.6, 0.62, 0])
        real = m.VGroup()
        for (name, year, parts), y in zip(REAL_SPLITS, (-0.2, -0.8, -1.4)):
            label = m.VGroup(text(name, 22, INK, weight=m.BOLD), text(year, 20, MUTED)).arrange(m.RIGHT, buff=0.18)
            on_baseline(label)
            real.add(m.VGroup(left_at(label, 0.5, y), chip_row(parts, y, 2.5, size=22, h=0.46, gap=0.06)))
        real_note = text("measured with tiktoken, word at the start of a text", 20, MUTED).move_to([3.6, -1.98, 0])

        self.say(
            "[So where do these pieces come from?] [Many language models learn them with byte-pair encoding, or BPE.]",
            self.new_header("BYTE-PAIR ENCODING · BPE", "Merge the most frequent pair"),
            m.AnimationGroup(m.FadeIn(cite), m.FadeIn(frame), m.FadeIn(title, shift=0.15 * m.UP)),
        )
        self.say(
            "[Start with a tiny corpus, for example these five words], [and how often each one appears.]",
            m.LaggedStart(*[m.FadeIn(r.mob, shift=0.15 * m.RIGHT) for r in rows], lag_ratio=0.2),
            m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.LEFT) for f in freqs], lag_ratio=0.2),
        )
        self.say(
            "[First, split every word into characters.] [That's our starting vocabulary: eleven tokens.]",
            m.AnimationGroup(*[r.to(list(r.word)) for r in rows]),
            m.AnimationGroup(m.FadeIn(strip_label), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in letters], lag_ratio=0.08), m.FadeIn(vcount)),
        )
        self.say(
            "[Now count every pair of neighbouring tokens], [weighting each word by how often it appears.]",
            m.Write(formula),
            m.LaggedStart(*[m.Indicate(f, color=COUNT, scale_factor=1.25) for f in freqs], lag_ratio=0.15),
        )
        self.say(
            "[The first term counts how often a word appears], [the second, how many times the pair sits inside it], "
            "[and the sum adds that up over every word.]",
            m.AnimationGroup(m.FadeIn(legend[0], shift=0.15 * m.RIGHT), m.Indicate(formula[3], color=COUNT, scale_factor=1.2)),
            m.AnimationGroup(m.FadeIn(legend[1], shift=0.15 * m.RIGHT), m.Indicate(formula[5], color=CHAR, scale_factor=1.2)),
            m.AnimationGroup(m.FadeIn(legend[2], shift=0.15 * m.RIGHT), m.Indicate(formula[2], color=INK, scale_factor=1.2)),
        )
        self.say(
            "[The top pair appears twenty times.] [We merge it into one new token, everywhere.]",
            m.AnimationGroup(self.bpe_rows(rows, 0, marks=True), m.FadeIn(bars[0], shift=0.15 * m.UP), m.Write(worked)),
            m.AnimationGroup(self.bpe_rows(rows, 0), m.FadeIn(learned[0], shift=0.2 * m.UP)),
        )
        self.say(
            "[Then we count again, and merge the next winner], [and the next.]",
            m.Succession(
                m.AnimationGroup(m.FadeOut(worked), m.FadeOut(bars[0]), m.FadeIn(bars[1]), self.bpe_rows(rows, 1, marks=True)),
                m.AnimationGroup(self.bpe_rows(rows, 1), m.FadeIn(learned[1], shift=0.2 * m.UP)),
            ),
            m.Succession(
                m.AnimationGroup(m.FadeOut(bars[1]), m.FadeIn(bars[2]), self.bpe_rows(rows, 2, marks=True)),
                m.AnimationGroup(self.bpe_rows(rows, 2), m.FadeIn(learned[2], shift=0.2 * m.UP)),
            ),
        )
        self.remove(worked, bars[0], bars[1])
        fast = [m.AnimationGroup(self.bpe_rows(rows, k), m.FadeIn(learned[k], shift=0.2 * m.UP), run_time=0.4) for k in range(3, N_LEARN)]
        familiar = []
        for r in rows:
            marks = {i for i, t in enumerate(r.tokens) if t in SPLIT}
            if marks:
                familiar.append(r.to(r.tokens, marks=marks, mark=GOOD))
        self.say(
            "[We repeat this six more times, merge after merge], [and familiar pieces appear.]",
            m.AnimationGroup(m.FadeOut(bars[2]), m.Succession(*fast)),
            m.AnimationGroup(*familiar, m.FadeTransform(vcount, vcount_after)),
        )
        self.remove(bars[2])
        steps = []
        for tokens in SPLIT_STEPS[1:]:
            steps.append(new.to(tokens, run_time=0.3))
        self.say(
            "[Now split a word the corpus never contained]: [unbelievable becomes three tokens.]",
            m.AnimationGroup(
                m.FadeOut(formula), m.FadeOut(legend), m.FadeIn(new_title), m.FadeIn(new.mob, shift=0.15 * m.DOWN),
                *[r.to(r.tokens) for r in rows],
            ),
            m.AnimationGroup(m.Succession(*steps), m.FadeIn(three, shift=0.15 * m.UP)),
        )
        self.say(
            "[Real tokenizers learn tens of thousands of merges from huge amounts of text], [so the exact split depends on each model's tokenizer.]",
            m.LaggedStart(*[m.FadeIn(r[0], shift=0.15 * m.RIGHT) for r in real], lag_ratio=0.3),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r[1], shift=0.15 * m.LEFT) for r in real], lag_ratio=0.3), m.FadeIn(real_note)),
        )
        self.wait(0.3)
        self.play(m.FadeOut(cite), run_time=0.3)
        self.clear()

    # ------------------------------------------------------------------ vocabulary size

    def vocab(self):
        self.next_section(
            "Vocabulary size",
            notes=f"Toy corpus length in tokens after 0…19 merges: {LENGTHS}. GPT-2: 50,257 tokens (Radford et al., 2019); "
            "GPT-4's cl100k_base 100,277; GPT-4o's o200k_base 200,019 (tiktoken n_vocab, 2026-10-21). "
            "Embedding-table rows are decorative (seeded numpy).",
        )
        x0 = 10
        ax = axes([x0, 32.5, 5], [0, 240, 50], 6.0, 3.8)
        ax.shift(np.array([-6.0, -2.2, 0]) - ax.c2p(x0, 0))
        xt = m.VGroup(*[text(str(v), 20, MUTED).next_to(ax.c2p(v, 0), m.DOWN, buff=0.18) for v in (11, 20, 30)])
        yt = m.VGroup(*[text(str(v), 20, MUTED).next_to(ax.c2p(x0, v), m.LEFT, buff=0.18) for v in (0, 100, 200)])
        x_title = text("vocabulary size", 22, MUTED).move_to([ax.c2p(20.5, 0)[0], -2.92, 0])
        y_title = left_at(text("corpus length in tokens", 22, MUTED), -6.1, 2.1)
        pts = [ax.c2p(len(LETTERS) + k, n) for k, n in enumerate(LENGTHS)]
        part_a = m.VMobject().set_points_as_corners(pts[: N_LEARN + 1]).set_stroke(SUB, width=4)
        part_b = m.VMobject().set_points_as_corners(pts[N_LEARN:]).set_stroke(SUB, width=4)
        marks_a = m.VGroup(*[m.Dot(p, radius=0.045, color=SUB) for p in pts[1 : N_LEARN + 1]])
        marks_b = m.VGroup(*[m.Dot(p, radius=0.045, color=SUB) for p in pts[N_LEARN + 1 :]])

        t = m.ValueTracker(0.0)

        def at(v):
            k = int(min(np.floor(v), len(pts) - 2))
            return pts[k] + (pts[k + 1] - pts[k]) * (v - k)

        def length_at(v):
            k = int(min(np.floor(v), len(LENGTHS) - 2))
            return LENGTHS[k] + (LENGTHS[k + 1] - LENGTHS[k]) * (v - k)

        mover = m.Dot(pts[0], radius=0.11, color=INK).set_z_index(3)
        mover.add_updater(lambda d: d.move_to(at(t.get_value())))
        r_names = m.VGroup(text("vocabulary", 22, MUTED), text("corpus", 22, MUTED))
        left_at(r_names[0], -2.75, 1.45)
        left_at(r_names[1], -2.75, 1.0)
        n_vocab = m.DecimalNumber(len(LETTERS), num_decimal_places=0, mob_class=Mono, font_size=26, color=SUB)
        n_len = m.DecimalNumber(LENGTHS[0], num_decimal_places=0, mob_class=Mono, font_size=26, color=COUNT)
        left_at(n_vocab, -1.15, 1.45)
        left_at(n_len, -1.15, 1.0)
        n_vocab.add_updater(lambda d: d.set_value(len(LETTERS) + round(t.get_value())))
        n_len.add_updater(lambda d: d.set_value(round(length_at(t.get_value()))))
        start_label = text("characters", 22, CHAR, weight=m.SEMIBOLD).next_to(pts[0], m.RIGHT, buff=0.3).shift(0.06 * m.UP)
        half = text("less than half", 22, COUNT, weight=m.SEMIBOLD).next_to(pts[N_LEARN], m.UR, buff=0.12)
        guide = m.DashedLine(ax.c2p(x0, LENGTHS[N_LEARN]), pts[N_LEARN], dash_length=0.08).set_stroke(COUNT, width=2.5)
        end_label = text("whole words", 22, WORDS, weight=m.SEMIBOLD).next_to(pts[-1], m.UP, buff=0.35).shift(0.45 * m.RIGHT)

        rng = np.random.default_rng(22)
        table = m.VGroup()
        for r in range(len(pts) + len(LETTERS) - 1):
            strip = m.VGroup(
                *[
                    m.Rectangle(width=0.27, height=0.075, stroke_width=0).set_fill(SUB, opacity=float(rng.uniform(0.15, 0.95))).move_to([2.3 + c * 0.29, 1.15 - r * 0.112, 0])
                    for c in range(10)
                ]
            )
            table.add(strip)
        assert len(table) == len(VOCAB) + len(MERGES) - N_LEARN == 30
        t_title = text("embedding table", 26, INK, weight=m.BOLD).move_to([3.6, 2.0, 0])
        t_sub = text("one row per token", 20, MUTED).move_to([3.6, 1.6, 0])
        brace = m.Brace(table, m.RIGHT, buff=0.12).set_color(MUTED)
        rows_label = text(f"{len(table)} rows", 22, INK, weight=m.SEMIBOLD).next_to(brace, m.RIGHT, buff=0.12)
        bigger = pill("bigger vocabulary, bigger model", COUNT, COUNT_FILL, 20).move_to([3.6, -2.75, 0])
        plot = [ax, xt, yt, x_title, y_title, part_a, part_b, marks_a, marks_b, mover, r_names, n_vocab, n_len, start_label, half, guide, end_label]

        def xs(v):
            return -5.6 + 11.2 * v / 250000

        line_y = -1.3
        scale = m.Line([xs(0), line_y, 0], [xs(250000), line_y, 0]).set_stroke(FAINT, width=3)
        ticks = m.VGroup()
        for v in range(0, 250001, 50000):
            ticks.add(m.Line([xs(v), line_y - 0.1, 0], [xs(v), line_y + 0.1, 0]).set_stroke(FAINT, width=3))
            ticks.add(text("0" if v == 0 else f"{v // 1000}k", 20, MUTED).move_to([xs(v), line_y - 0.42, 0]))
        band = m.Rectangle(width=xs(200000) - xs(30000), height=0.32, stroke_width=0).set_fill(SUB, opacity=0.22)
        band.move_to([(xs(30000) + xs(200000)) / 2, line_y, 0])
        statement = text("tens of thousands to a couple of hundred thousand tokens", 28, SUB, weight=m.BOLD).move_to([0, 1.45, 0])
        markers = []
        for name, value, shown, year in VOCAB_SIZES:
            source = "Radford et al., 2019" if name == "GPT-2" else year
            head = m.VGroup(text(name, 24, INK, weight=m.BOLD), code(shown, 22, SUB), text(source, 20, MUTED)).arrange(m.DOWN, buff=0.1)
            head.move_to([xs(value), 0.05, 0])
            stem = m.Line([xs(value), head.get_bottom()[1] - 0.1, 0], [xs(value), line_y + 0.05, 0]).set_stroke(MUTED, width=2.5)
            dot = m.Dot([xs(value), line_y, 0], radius=0.09, color=SUB).set_z_index(2)
            markers.append(m.VGroup(head, stem, dot))
        toy = left_at(text("our toy tokenizer: 30 tokens", 22, MUTED), xs(0) - 0.3, line_y - 1.0)
        toy_dot = m.Dot([xs(30), line_y, 0], radius=0.07, color=MUTED).set_z_index(2)

        self.say(
            "[How many merges should we learn?] [That choice sets the vocabulary size.]",
            m.AnimationGroup(self.new_header("VOCABULARY SIZE", "How many merges?"), m.Create(ax), m.FadeIn(xt), m.FadeIn(yt), m.FadeIn(y_title)),
            m.AnimationGroup(m.FadeIn(x_title, shift=0.15 * m.UP)),
        )
        self.say(
            "[Our corpus starts at eleven characters, and two hundred and fourteen tokens.]",
            m.AnimationGroup(m.FadeIn(mover, scale=0.5), m.FadeIn(r_names), m.FadeIn(n_vocab), m.FadeIn(n_len), m.FadeIn(start_label)),
        )
        self.say(
            "[Each merge adds a token and shortens the text]: [nine merges, and it's less than half as long.]",
            m.AnimationGroup(t.animate(rate_func=m.linear).set_value(N_LEARN), m.Create(part_a, rate_func=m.linear), m.LaggedStart(*[m.FadeIn(d) for d in marks_a], lag_ratio=1.0)),
            m.AnimationGroup(m.Create(guide), m.FadeIn(half, shift=0.15 * m.UP)),
        )
        self.say(
            "[Keep merging, and every word becomes one token]: [we're back to whole words.]",
            m.AnimationGroup(t.animate(rate_func=m.linear).set_value(len(LENGTHS) - 1), m.Create(part_b, rate_func=m.linear), m.LaggedStart(*[m.FadeIn(d) for d in marks_b], lag_ratio=1.0)),
            m.FadeIn(end_label, shift=0.15 * m.UP),
        )
        self.say(
            "[But as we saw in episode twenty-two, every token needs its own row in the embedding table], [so a bigger vocabulary means a bigger model.]",
            m.AnimationGroup(m.FadeIn(t_title), m.FadeIn(t_sub), m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.LEFT) for s in table], lag_ratio=0.06)),
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(rows_label), m.FadeIn(bigger, shift=0.15 * m.UP)),
        )
        for mob in (mover, n_vocab, n_len):
            mob.clear_updaters()
        self.say(
            "[Real models settle in between: tens of thousands to a couple of hundred thousand tokens.] [GPT two used fifty thousand, two hundred and fifty-seven.]",
            m.AnimationGroup(
                *[m.FadeOut(mob) for mob in (*plot, t_title, t_sub, table, brace, rows_label, bigger)],
                m.Succession(
                    m.Wait(0.5),
                    m.AnimationGroup(
                        m.Create(scale), m.FadeIn(ticks), m.FadeIn(band), m.FadeIn(statement, shift=0.15 * m.UP),
                        m.FadeIn(markers[1], shift=0.15 * m.DOWN), m.FadeIn(markers[2], shift=0.15 * m.DOWN), m.FadeIn(toy), m.FadeIn(toy_dot),
                    ),
                ),
            ),
            m.AnimationGroup(m.FadeIn(markers[0], shift=0.15 * m.DOWN), m.Flash(markers[0][2].get_center(), color=SUB, flash_radius=0.35)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ cost

    def costs(self):
        self.next_section(
            "Cost",
            notes="Rule of thumb for English: a token is about 4 characters or three quarters of a word (OpenAI's guidance). "
            "Token counts measured with tiktoken 0.14.0 (2026-10-21) for 'That is unbelievable!' and its German, Greek and "
            "Hindi translations: GPT-2 4/10/26/29, GPT-4o 4/4/9/8. Language gap in tokenizers: Petrov et al., 2023. "
            "The two-line function: GPT-2 16 tokens (three lone indentation spaces), GPT-4o 12.",
        )
        tiles = m.VGroup(
            cost_tile("ep23-tag", "Priced per token", "for what goes in and out", COUNT),
            cost_tile("ep23-window", "Context window", "a limit, counted in tokens", CHAR),
        ).arrange(m.RIGHT, buff=0.5).move_to([0, 1.45, 0])
        big = text("1 token ≈ ¾ of a word", 56, INK, weight=m.BOLD, t2c={"¾": COUNT}).move_to([0, -0.6, 0])
        small = code("1,000 tokens ≈ 750 words", 32, MUTED).move_to([0, -1.75, 0])
        note = text("a rough rule for English text", 22, MUTED).move_to([0, -2.55, 0])
        frame_a = [tiles, big, small, note]

        caption = m.VGroup(text("the same sentence, GPT-2's tokenizer (2019) ·", 24, MUTED), spaces(1, 24, INK), text("is a space", 24, MUTED))
        caption.arrange(m.RIGHT, buff=0.18)
        caption[1].align_to(caption[0], m.DOWN)
        caption.move_to([0, 2.25, 0])
        lang_rows = []
        for name, tokens, y in (("English", ENGLISH_GPT2, 1.5), ("German", GERMAN_GPT2, 0.72)):
            label = right_at(text(name, 24, INK, weight=m.SEMIBOLD), -4.6, y)
            chips = chip_row(tokens, y, -4.3, size=24, h=0.52)
            total = text(f"{len(tokens)} tokens", 24, SUB, weight=m.SEMIBOLD).next_to(chips, m.RIGHT, buff=0.35)
            lang_rows.append(m.VGroup(label, chips, total))
        unit = 7.0 / 29
        chart, newer = m.VGroup(), m.VGroup()
        for (name, old, new), y in zip(LANGUAGES, (-0.4, -1.08, -1.76, -2.44)):
            label = right_at(text(name, 22, INK), -4.6, y)
            b_old = left_at(m.Rectangle(width=old * unit, height=0.2, stroke_width=0).set_fill(SUB, opacity=0.9), -4.3, y + 0.13)
            n_old = code(str(old), 20, SUB).next_to(b_old, m.RIGHT, buff=0.12)
            b_new = left_at(m.Rectangle(width=new * unit, height=0.2, stroke_width=0).set_fill(TEAL, opacity=0.9), -4.3, y - 0.13)
            n_new = code(str(new), 20, TEAL).next_to(b_new, m.RIGHT, buff=0.12)
            chart.add(m.VGroup(label, b_old, n_old))
            newer.add(m.VGroup(b_new, n_new))
        legend_old = m.VGroup(m.Square(0.22, stroke_width=0).set_fill(SUB, 0.9), text("GPT-2 (2019)", 22, MUTED)).arrange(m.RIGHT, buff=0.15)
        legend_new = m.VGroup(m.Square(0.22, stroke_width=0).set_fill(TEAL, 0.9), text("GPT-4o (2024)", 22, MUTED)).arrange(m.RIGHT, buff=0.15)
        left_at(legend_old, 3.9, -0.6)
        left_at(legend_new, 3.9, -1.05)
        petrov = text("Petrov et al., 2023", 20, MUTED).move_to([4.95, -2.44, 0])
        frame_b = [caption, *lang_rows, chart, newer, legend_old, legend_new, petrov]

        first, body = code("def add(a, b):", 30, INK), code("return a + b", 30, INK)
        body.next_to(first, m.DOWN, buff=0.2).align_to(first, m.LEFT).shift(4 * adv(30) * m.RIGHT)
        indent = spaces(4, 30, COUNT).next_to(body, m.LEFT, buff=0.04).align_to(body, m.DOWN)
        listing = m.VGroup(first, body, indent)
        window = card(listing.width + 1.0, listing.height + 0.7, EDGE, CARD, radius=0.2, stroke=2).move_to([0, 1.45, 0])
        listing.move_to(window)
        code_rows = []
        for name, tokens, total, y in (("GPT-2", INDENT_GPT2, CODE_TOTALS[0], -0.35), ("GPT-4o", INDENT_GPT4O, CODE_TOTALS[1], -1.45)):
            label = right_at(text(name, 24, INK, weight=m.BOLD), -3.6, y)
            lone = {i for i, tok in enumerate(tokens) if set(tok) == {"␣"}}
            chips = chip_row(tokens, y, -3.3, size=26, h=0.56, marks=lone)
            count = text(f"{len(tokens)} tokens · whole function {total}", 22, MUTED).next_to(chips, m.RIGHT, buff=0.4)
            code_rows.append(m.VGroup(label, chips, count))
        second = text("the second line, token by token", 22, MUTED).move_to([0, 0.35, 0])

        self.say(
            "[Tokens are also how models are priced and limited]: [you pay per token], "
            "[and the context window, how much a model can read at once, is counted in tokens too.]",
            self.new_header("WHY IT MATTERS · COST", "Everything is counted in tokens"),
            m.FadeIn(tiles[0], shift=0.2 * m.UP),
            m.FadeIn(tiles[1], shift=0.2 * m.UP),
        )
        self.say(
            "[A rough rule for English: one token is about three quarters of a word], [so a thousand tokens is roughly seven hundred and fifty words.]",
            m.AnimationGroup(m.FadeIn(big, scale=0.9), m.FadeIn(note)),
            m.FadeIn(small, shift=0.15 * m.UP),
        )
        self.say(
            "[Other languages can cost more.] [In the GPT two tokenizer, this English sentence is four tokens], [and the same sentence in German is ten.]",
            m.AnimationGroup(*[m.FadeOut(mob) for mob in frame_a], m.FadeIn(caption, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(lang_rows[0][0]), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in lang_rows[0][1]], lag_ratio=0.2), m.FadeIn(lang_rows[0][2])),
            m.AnimationGroup(m.FadeIn(lang_rows[1][0]), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in lang_rows[1][1]], lag_ratio=0.1), m.FadeIn(lang_rows[1][2])),
        )
        self.say(
            "[In Greek or Hindi, it takes more than twenty-five.] [Newer tokenizers narrow the gap, but they don't close it.]",
            m.AnimationGroup(
                m.FadeIn(legend_old),
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2])) for r in chart], lag_ratio=0.2),
            ),
            m.AnimationGroup(
                m.FadeIn(legend_new), m.FadeIn(petrov),
                m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(r[0], m.LEFT), m.FadeIn(r[1])) for r in newer], lag_ratio=0.2),
            ),
        )
        self.say(
            "[Code can cost extra, too]: [older tokenizers spent a token on almost every space of indentation.]",
            m.AnimationGroup(*[m.FadeOut(mob) for mob in frame_b], m.Succession(m.Wait(0.4), m.AnimationGroup(m.FadeIn(window), m.FadeIn(listing)))),
            m.AnimationGroup(m.FadeIn(second), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.UP) for r in code_rows], lag_ratio=0.4)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ odd failures

    def failures(self):
        self.next_section(
            "Odd failures",
            notes="GPT-2's tokenizer: 'unbelievable' (start of text) = un | bel | iev | able, IDs 403, 6667, 11203, 540 "
            "(tiktoken 0.14.0, 2026-10-21). Letter counting, spelling and reversing are known weak spots of token-based models.",
        )
        user = icon("user", 0.95).move_to([-5.6, 1.95, 0])
        question = bubble(text("How many e's are in “unbelievable”?", 26, INK), SKY, SKY_FILL, tail="left")
        question.next_to(user, m.RIGHT, buff=0.35).shift(0.25 * m.UP)

        left = card(6.0, 3.1, EDGE, CARD, radius=0.22, stroke=2).move_to([-3.25, -1.0, 0])
        left_title = left_at(text("What you see", 26, INK, weight=m.BOLD), -5.85, 0.2)
        letters = code(WORD, 46, INK).move_to([-3.25, -0.85, 0])
        es = [letters[i] for i, c in enumerate(WORD) if c == "e"]
        answer = text("e appears 3 times", 26, COUNT, weight=m.SEMIBOLD).move_to([-3.25, -1.9, 0])

        right = card(6.0, 3.1, EDGE, CARD, radius=0.22, stroke=2).move_to([3.25, -1.0, 0])
        right_title = left_at(text("What the model sees", 26, INK, weight=m.BOLD), 0.65, 0.2)
        chips = chip_row(REAL_SPLITS[0][2], -0.55, 3.25, anchor="center", size=28, h=0.6, gap=0.35)
        ids = m.VGroup(*[token_chip(str(n), EDGE, CARD, 24, 0.54).move_to([c.get_x(), -1.5, 0]) for n, c in zip(GPT2_IDS, chips)])
        hint = text("GPT-2's tokenizer", 20, MUTED).move_to([3.25, -2.2, 0])
        weak = m.VGroup(*[pill(s, BAD, BAD_FILL, 22) for s in ("spelling", "counting letters", "reversing words")])
        weak.arrange(m.RIGHT, buff=0.4).move_to([0, -3.05, 0])

        self.say(
            "[Tokens also explain some odd mistakes.] [Ask a model how many times the letter e appears in unbelievable.]",
            self.new_header("WHY IT MATTERS · ODD FAILURES", "Tokens hide the letters"),
            m.AnimationGroup(m.FadeIn(user, shift=0.2 * m.RIGHT), m.FadeIn(question, shift=0.2 * m.UP)),
        )
        self.say(
            "[You see twelve letters], [and count three.]",
            m.AnimationGroup(m.FadeIn(left), m.FadeIn(left_title), m.FadeIn(letters, shift=0.15 * m.UP)),
            m.AnimationGroup(*[g.animate.set_color(COUNT) for g in es], m.FadeIn(answer, shift=0.15 * m.UP)),
        )
        self.say(
            "[The model sees a few token numbers]: [the letters are hidden inside them.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(right_title), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.DOWN) for c in chips], lag_ratio=0.2)),
            m.AnimationGroup(m.LaggedStart(*[m.TransformFromCopy(c, i) for c, i in zip(chips, ids)], lag_ratio=0.2), m.FadeIn(hint)),
        )
        self.say(
            "[It has to memorise how each token is spelled], [so spelling, counting letters and reversing words can trip up even strong models.]",
            m.Circumscribe(chips, color=SUB, buff=0.12),
            m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.UP) for p in weak], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Split, merge, count.")
        halo = glow(SUB, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Split", "Merge", "Count"), (CHAR, SUB, COUNT))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("A language model reads tokens, not words, and every token counts.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [split text into characters], [merge the most frequent pairs], [and count everything in tokens.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[A language model reads tokens, not words, and every token counts.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
