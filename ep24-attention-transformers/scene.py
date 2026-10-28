"""AI, Explained · Episode 24 · Attention & Transformers

A series remake of the standalone film ~/videos/transformer/ ("What is a Transformer?", 3:26):
the same running example, "The cat was tired, so it…", its attention arcs, weight bars, head
panels, block and next-word bars, rebuilt with the series kit, plus the attention formula worked
through with real numbers.

Everything plotted is computed below with seeded numpy, at toy size (4 numbers per vector):
- embeddings (seed 254978) plus a sinusoidal position signal (Vaswani et al.) × 0.5 → X;
- three random "learned" matrices give Q, K, V (rounded to one decimal, as shown);
- scores QKᵀ, ÷ √4, a causal mask, softmax per row → the attention table; its last row
  ("it") is the bars: cat 54 %, tired 21 %; the weighted sum of values is the new "it";
- one toy block: x + attention (the residual), then a ReLU feed-forward plus its residual;
- three illustrative heads: a pattern plus seeded noise, causal mask, softmax;
- next-token bars: softmax over 50,000 illustrative scores (4 named words + a seeded tail).
The asserts keep every number the narration quotes equal to the computed one.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# colours: one meaning each, for the whole episode
QUERY, QUERY_FILL = TEAL, TEAL_FILL
KEY, KEY_FILL = AMBER, AMBER_FILL
VALUE, VALUE_FILL = PINK, PINK_FILL
ATTN, ATTN_FILL = VIOLET, VIOLET_FILL  # attention: arcs, weights, the Attention step
FFN, FFN_FILL = BLUE, SKY_FILL  # the feed-forward step

# ---------------------------------------------------------------------------- the data

TOKENS = ["The", "cat", "was", "tired", "so", "it"]
XS = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
CAT, TIRED, IT = 1, 3, 5
D = 4  # numbers per vector (toy size)


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


_pos = np.arange(6)
POSITION = 0.5 * np.c_[np.sin(_pos), np.cos(_pos), np.sin(_pos / 100), np.cos(_pos / 100)]
_rng = np.random.default_rng(254978)
EMBED = _rng.normal(0, 1.0, (6, D))
X = EMBED + POSITION
W_Q, W_K, W_V = (_rng.normal(0, 0.45, (D, D)) for _ in range(3))
Q, K, V = (np.round(X @ w, 1) for w in (W_Q, W_K, W_V))
SCORES = Q @ K.T  # every query against every key
SCALED = SCORES / np.sqrt(D)
MASK = np.triu(np.ones((6, 6), dtype=bool), 1)  # a word may not look ahead
ATTN_TABLE = softmax(np.where(MASK, -np.inf, SCALED))
W_IT = ATTN_TABLE[IT]
PCT = np.round(100 * W_IT).astype(int)
NEW_IT = W_IT @ V
PRODUCTS = Q[IT] * K[CAT]
COSINES = V @ NEW_IT / np.linalg.norm(V, axis=1) / np.linalg.norm(NEW_IT)

assert np.sqrt(D) == 2  # "here, two"
assert np.argmax(SCORES[IT]) == CAT  # "cat scores highest"
assert np.all(PRODUCTS > 0) and abs(PRODUCTS.sum() - 3.52) < 1e-9 and abs(SCORES[IT, CAT] - 3.52) < 1e-9
assert np.argmax(W_IT) == CAT and PCT[CAT] == 54 and PCT[TIRED] == 21  # "fifty-four", "twenty-one"
assert np.argsort(-W_IT)[1] == TIRED and PCT.sum() == 100
assert W_IT[CAT] > 0.5  # "more than half of the new vector comes from the cat"
assert np.argmax(COSINES) == CAT and round(float(COSINES[CAT]), 2) == 0.99
assert np.allclose(ATTN_TABLE.sum(1), 1) and np.all(ATTN_TABLE[MASK] == 0)
assert np.abs(np.r_[Q.ravel(), K.ravel(), V.ravel()]).max() < 2

# one toy block for "it": attention plus the residual, then a ReLU feed-forward plus its residual
_ffn = np.random.default_rng(25)
W_1, W_2 = _ffn.normal(0, 0.5, (D, 8)), _ffn.normal(0, 0.5, (8, D))
AFTER_ATTN = X[IT] + NEW_IT
AFTER_BLOCK = AFTER_ATTN + np.maximum(AFTER_ATTN @ W_1, 0) @ W_2
assert np.abs(AFTER_BLOCK).max() < 10


# three illustrative heads: a pattern, a little seeded noise, the causal mask, softmax
def _head(bonus, seed):
    s = np.random.default_rng(seed).normal(0, 0.3, (6, 6))
    for (i, j), b in bonus.items():
        s[i, j] += b
    return softmax(np.where(MASK, -np.inf, s))


HEADS = [
    ("Head 1 · what “it” means", ATTN, _head({(5, 1): 3.0, (5, 3): 1.8, **{(k, k): 1.5 for k in range(5)}}, 1)),
    ("Head 2 · grammar", BLUE, _head({(1, 0): 2.5, (2, 1): 2.5, (3, 2): 2.5, (4, 3): 2.0, (5, 4): 1.5}, 2)),
    ("Head 3 · previous word", LIME, _head({(k, k - 1): 3.0 for k in range(1, 6)}, 3)),
]
assert np.argmax(HEADS[0][2][IT]) == CAT  # "one might link it to cat"
assert all(np.argmax(HEADS[2][2][k]) == k - 1 for k in range(1, 6))  # "the word just before"

# the next token: softmax over a 50,000-token vocabulary (illustrative scores)
VOCAB = 50_000
CANDIDATES = ["slept", "napped", "rested", "yawned"]
_tail = np.random.default_rng(33).normal(0, 1.0, VOCAB - len(CANDIDATES))
NEXT = softmax(np.r_[[13.0, 11.5, 10.9, 10.4], _tail])
NEXT_PCT = np.round(100 * NEXT[: len(CANDIDATES)]).astype(int)
REST_PCT = 100 - int(NEXT_PCT.sum())
assert np.argmax(NEXT) == 0 and NEXT_PCT[0] == 62  # "sixty-two percent"
assert abs(100 * NEXT[len(CANDIDATES) :].sum() - REST_PCT) < 1 and NEXT[len(CANDIDATES) :].max() < NEXT[3]

# ---------------------------------------------------------------------------- helpers


def ep_recolored(name, color, fill):
    """kit.recolored(), but written as assets/ep24-<name>-<color>.svg (this episode only adds
    ep24- files to the shared assets)."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"ep24-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


def fmt(v, digits):
    s = f"{v:.{digits}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def bracket(x, top, bottom, lip, color):
    corners = [[x + lip, top, 0], [x, top, 0], [x, bottom, 0], [x + lip, bottom, 0]]
    return m.VMobject().set_points_as_corners(corners).set_stroke(color, width=3)


def column(values, color, size=20, digits=2):
    """A vector as a bracketed column of all its numbers."""
    nums = m.VGroup(*[code(fmt(v, digits), size, color) for v in values]).arrange(m.DOWN, buff=0.1, aligned_edge=m.RIGHT)
    top, bottom = nums.get_top()[1] + 0.1, nums.get_bottom()[1] - 0.1
    return m.VGroup(
        bracket(nums.get_left()[0] - 0.12, top, bottom, 0.1, color), nums, bracket(nums.get_right()[0] + 0.12, top, bottom, -0.1, color)
    )


def row(values, color, xs, y, size=24, digits=1):
    """A vector as a bracketed row; each number right-aligned in its slot at xs."""
    nums = m.VGroup(*[code(fmt(v, digits), size, color).move_to([x, y, 0]).align_to([x + 0.3, 0, 0], m.RIGHT) for v, x in zip(values, xs)])
    top, bottom = y + 0.24, y - 0.24
    return m.VGroup(
        bracket(nums.get_left()[0] - 0.2, top, bottom, 0.1, color), nums, bracket(nums.get_right()[0] + 0.2, top, bottom, -0.1, color)
    )


def token_row(y, xs=XS, words=TOKENS, **kw):
    return m.VGroup(*[chip(w, **kw).move_to([x, y, 0]) for w, x in zip(words, xs)])


def arc(a, b, color, width, opacity=1.0, angle=0.8):
    """An arc from the top of `a` to the top of `b`, always bulging upward."""
    start, end = a.get_top() + 0.06 * m.UP, b.get_top() + 0.06 * m.UP
    bend = angle if start[0] > end[0] else -angle
    return m.ArcBetweenPoints(start, end, angle=bend).set_stroke(color, width=width, opacity=opacity)


def weighted_arcs(tokens, weights):
    return m.VGroup(*[arc(tokens[IT], tokens[i], ATTN, 2 + 16 * weights[i], opacity=0.35 + weights[i]) for i in range(IT)])


# ---------------------------------------------------------------------------- the episode


class Ep24AttentionTransformers(Episode):
    number = 24

    def lesson(self):
        self.hook()
        self.problem()
        self.idea()
        self.embed()
        self.qkv()
        self.scores()
        self.weights()
        self.blend()
        self.formula()
        self.heads()
        self.block()
        self.predict()
        self.why()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="'The cat was tired, so it…': we know 'it' is the cat and can guess the next word. GPT = Generative Pre-trained Transformer.")
        sentence = text("The cat was tired, so it…", 56, INK, weight=m.SEMIBOLD).move_to([-0.55, 1.2, 0])
        cat, it = glyphs(sentence, "cat"), glyphs(sentence, "it…")[:2]
        link = arc(it.copy().shift(0.1 * m.UP), cat.copy().shift(0.1 * m.UP), ATTN, 5, angle=0.9)
        means = text("= the cat", 26, ATTN, weight=m.BOLD).next_to(it, m.DOWN, buff=0.3)
        blank = chip("?", GOOD, GOOD_FILL, width=1.1).next_to(sentence, m.RIGHT, buff=0.35)
        name = text("attention", 26, ATTN, weight=m.SEMIBOLD).next_to(link, m.UP, buff=0.15)
        letters = m.VGroup(*[text(c, 110, col, weight=m.BOLD) for c, col in zip("GPT", (MUTED, MUTED, ATTN))])
        for letter, x in zip(letters, (-3.6, 0, 3.6)):
            letter.move_to([x, -1.25, 0])
        on_baseline(letters)
        names = m.VGroup(
            *[
                text(w, 28, col, weight=m.SEMIBOLD).move_to([x, -2.55, 0])
                for w, col, x in zip(("Generative", "Pre-trained", "Transformer"), (MUTED, MUTED, ATTN), (-3.6, 0, 3.6))
            ]
        )
        on_baseline(names)

        self.say("Read this sentence: [the cat was tired, so it...]", m.AddTextLetterByLetter(sentence, time_per_char=0.04))
        self.say(
            "[You know right away that it means the cat], [and you can probably guess what comes next.]",
            m.AnimationGroup(m.Create(link), m.FadeIn(means, shift=0.15 * m.UP)),
            m.FadeIn(blank, scale=1.2),
        )
        self.say(
            "[A Transformer does both, with one idea called attention.] [It's the T in GPT], the design behind most of today's language models.",
            m.AnimationGroup(m.ShowPassingFlash(link.copy().set_stroke(ATTN, width=10), time_width=0.6), m.FadeIn(name, shift=0.15 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(x, shift=0.3 * m.UP) for x in letters], lag_ratio=0.2), m.FadeIn(names)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the problem

    def problem(self):
        self.next_section("The problem", notes="Recurrent networks (Episode 21) pass a memory word by word: it fades over long text, and the steps can't run in parallel.")
        tokens = token_row(1.2)
        memory = m.VGroup(card(1.6, 1.05, EDGE, CARD, radius=0.2), text("memory", 20, MUTED), text("cat", 28, INK, weight=m.BOLD))
        memory[1].move_to(memory[0].get_center() + 0.24 * m.UP)
        memory[2].move_to(memory[0].get_center() + 0.17 * m.DOWN)
        memory.move_to([XS[1], -0.15, 0])
        hops = []
        for x, fade in zip(XS[2:], (0.7, 0.45, 0.27, 0.12)):
            step = (hops[-1] if hops else memory).copy().move_to([x, -0.15, 0])
            step[2].set_opacity(fade)
            hops.append(step)
        lost = text("it  =  ?", 30, BAD, weight=m.BOLD).move_to([XS[5], -1.25, 0])
        slow = pill("one word at a time: each one waits for the one before", BAD, BAD_FILL, 22).move_to([0, -2.6, 0])

        self.say(
            "[Before transformers], [language models read one word at a time], [passing a small memory along], like the recurrent networks of episode twenty-one.",
            self.new_header("THE PROBLEM", "One word at a time"),
            m.LaggedStart(*[m.FadeIn(t, shift=0.2 * m.RIGHT) for t in tokens], lag_ratio=0.25),
            m.FadeIn(memory, shift=0.3 * m.UP),
        )
        self.say(
            "[By the time they reach it, the cat has faded.] [And every word has to wait for the one before.]",
            m.Succession(*[m.Transform(memory, hop, run_time=0.6) for hop in hops], m.FadeIn(lost, scale=1.3)),
            m.FadeIn(slow, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.tokens = tokens
        self.clear(*tokens)

    # ------------------------------------------------------------------ the idea

    def idea(self):
        self.next_section(
            "The idea",
            notes="Vaswani et al., 2017, 'Attention Is All You Need' (NeurIPS 2017): the Transformer, built on attention, no recurrence.",
        )
        tokens = self.tokens
        web = m.VGroup(*[arc(tokens[j], tokens[i], ATTN, 2.5, opacity=0.55) for j in range(6) for i in range(j)])
        box = card(7.4, 1.45, EDGE, CARD)
        paper = m.VGroup(box, m.VGroup(text("“Attention Is All You Need”", 30, INK, weight=m.BOLD), text("Vaswani et al., 2017", 22, MUTED)).arrange(m.DOWN, buff=0.15).move_to(box))
        paper.move_to([0, -1.05, 0])
        caption = text("Attention: every word looks at the others, directly, all at once.", 28, INK, weight=m.SEMIBOLD, t2c={"Attention:": ATTN})
        caption.move_to([0, -2.85, 0])

        self.say(
            "[In twenty seventeen], [a paper called Attention Is All You Need changed that.]",
            self.new_header("THE IDEA", "Let every word look at the others"),
            m.FadeIn(paper, shift=0.3 * m.UP),
        )
        self.say(
            "[Every word can look directly at every other word, all at the same time.] [That's attention.]",
            m.LaggedStart(*[m.Create(a) for a in web], lag_ratio=0.05),
            m.Write(caption),
        )
        self.wait(0.3)
        self.clear(*tokens)

    # ------------------------------------------------------------------ embed

    def embed(self):
        self.next_section(
            "Embed",
            notes="Tokens (Episode 23) → embedding vectors (Episode 22) + a position signal. Toy: 4 numbers, seeded; position = the 2017 sinusoids × 0.5. Real models: hundreds to thousands of numbers.",
        )
        tokens = self.tokens
        vectors = m.VGroup(*[column(EMBED[k], SKY).move_to([x, -0.45, 0]) for k, x in enumerate(XS)])
        mixed = m.VGroup(*[column(X[k], SKY).move_to([x, -0.45, 0]) for k, x in enumerate(XS)])
        positions = m.VGroup(*[pill(f"pos {k + 1}", MUTED, CARD, 20).move_to([x, -2.15, 0]) for k, x in enumerate(XS)])
        toy = text("toy size: 4 numbers per vector · real models use hundreds or thousands", 22, MUTED).move_to([0, -3.3, 0])

        self.say(
            "[First, embedding.] [The text is split into tokens, as in episode twenty-three], [and each token becomes a vector, its embedding from episode twenty-two.]",
            self.new_header("EMBED", "Tokens become vectors"),
            m.LaggedStart(*[m.Indicate(t, color=SKY, scale_factor=1.1) for t in tokens], lag_ratio=0.12),
            m.AnimationGroup(m.LaggedStart(*[m.TransformFromCopy(t, v) for t, v in zip(tokens, vectors)], lag_ratio=0.12), m.FadeIn(toy)),
        )
        self.say(
            "[Every word is read at once], [so each vector also gets its position mixed in], to keep the word order.",
            m.AnimationGroup(*[m.Indicate(t, color=SKY, scale_factor=1.1) for t in tokens]),
            m.Succession(
                m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.UP) for p in positions], lag_ratio=0.1, run_time=1.0),
                m.AnimationGroup(
                    *[p.animate.move_to(v).scale(0.3).set_opacity(0) for p, v in zip(positions, vectors)],
                    *[m.ReplacementTransform(v, w) for v, w in zip(vectors, mixed)],
                    run_time=1.2,
                ),
            ),
        )
        self.remove(*positions)
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ query, key, value

    def qkv(self):
        self.next_section(
            "Query, key, value",
            notes="Each vector × three learned matrices W_Q, W_K, W_V (shared by every position) → query, key, value. Toy matrices: seeded random; q, k, v rounded to one decimal.",
        )
        word = chip("it").move_to([-6.0, 0, 0])
        x_col = column(X[IT], SKY).move_to([-4.3, 0, 0])
        to_x = arrow(word.get_right(), x_col.get_left(), MUTED, buff=0.12)
        ys = (1.55, 0.0, -1.55)
        roles = (
            ("q", "W_Q", QUERY, QUERY_FILL, Q[IT], "Query", "what I'm looking for"),
            ("k", "W_K", KEY, KEY_FILL, K[IT], "Key", "what I offer to others"),
            ("v", "W_V", VALUE, VALUE_FILL, V[IT], "Value", "what I pass on"),
        )
        slots = [0.0, 1.0, 2.0, 3.0]
        mats, fans, outs, rows, notes = m.VGroup(), m.VGroup(), m.VGroup(), m.VGroup(), m.VGroup()
        for y, (letter, mat, color, fill, values, title, desc) in zip(ys, roles):
            box = card(1.15, 0.95, color, fill, radius=0.18).move_to([-2.35, y, 0])
            mats.add(m.VGroup(box, tex(mat, 34, color).move_to(box)))
            fans.add(arrow(x_col.get_right(), box.get_left(), MUTED, buff=0.12))
            vec = row(values, color, slots, y, size=22)
            name = tex(letter, 38, color).next_to(vec, m.LEFT, buff=0.2)
            outs.add(arrow(box.get_right(), name.get_left(), MUTED, buff=0.12))
            rows.add(m.VGroup(name, vec))
            notes.add(
                m.VGroup(text(title, 26, color, weight=m.BOLD), text(desc, 22, MUTED))
                .arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
                .move_to([0, y, 0])
                .align_to([3.95, 0, 0], m.LEFT)
            )
        shared = text("every word gets its own three, from the same three matrices", 22, MUTED).move_to([0, -3.1, 0])

        self.say(
            "[Now, attention itself.] [Follow the word it.]",
            self.new_header("QUERY · KEY · VALUE", "Three vectors for every word"),
            m.AnimationGroup(m.FadeIn(word, shift=0.2 * m.RIGHT), m.GrowArrow(to_x), m.FadeIn(x_col)),
        )
        self.say(
            "[Its vector is multiplied by three learned matrices], [giving three new vectors.]",
            m.AnimationGroup(*[m.GrowArrow(f) for f in fans], *[m.FadeIn(x, scale=0.8) for x in mats]),
            m.AnimationGroup(*[m.GrowArrow(a) for a in outs], *[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in rows]),
        )
        self.say(
            "[A query: what this word is looking for.] [A key: what it offers to others.] [And a value: what it passes on.]",
            *[m.AnimationGroup(m.FadeIn(n, shift=0.2 * m.LEFT), m.Indicate(r, color=r[0].get_color(), scale_factor=1.05)) for n, r in zip(notes, rows)],
        )
        self.say("[Every word gets its own three, made with the same three matrices.]", m.FadeIn(shared, shift=0.2 * m.UP))
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ scores

    def scores(self):
        self.next_section(
            "Scores",
            notes=f"q(it)·k(cat) = {' + '.join(f'{p:.2f}' for p in PRODUCTS)} = {SCORES[IT, CAT]:.2f}. Scores of 'it' against every key: "
            + ", ".join(f"{t} {s:.2f}" for t, s in zip(TOKENS, SCORES[IT]))
            + ". Divided by √d = 2 (d = 4): Vaswani et al. scale by 1/√d_k so large dot products don't push softmax into flat regions.",
        )
        tokens = token_row(1.0)
        self.tokens = tokens
        it, cat = tokens[IT], tokens[CAT]
        self.arcs = m.VGroup(*[arc(it, tokens[i], ATTN, 2.5, opacity=0.5) for i in range(IT)])
        self.score_txt = m.VGroup(*[code(fmt(s, 2), 22, INK).move_to([x, 0.25, 0]) for s, x in zip(SCORES[IT], XS)])
        scaled = m.VGroup(*[code(fmt(s, 2), 22, INK).move_to([x, 0.25, 0]) for s, x in zip(SCALED[IT], XS)])

        slots = [-1.55, -0.45, 0.65, 1.75]
        q_row, k_row = row(Q[IT], QUERY, slots, -0.75), row(K[CAT], KEY, slots, -1.45)
        q_lab = code("q(it)", 22, QUERY).next_to(q_row, m.LEFT, buff=0.3)
        k_lab = code("k(cat)", 22, KEY).next_to(k_row, m.LEFT, buff=0.3)
        k_lab.align_to(q_lab, m.RIGHT)
        products = m.VGroup(*[code(f"{p:.2f}", 24, INK).move_to([x, -2.2, 0]).align_to([x + 0.3, 0, 0], m.RIGHT) for p, x in zip(PRODUCTS, slots)])
        pluses = m.VGroup(*[code("+", 24, MUTED).move_to([(a.get_right()[0] + b.get_left()[0]) / 2, -2.2, 0]) for a, b in zip(products, products[1:])])
        total = code(f"= {SCORES[IT, CAT]:.2f}", 24, ATTN).next_to(products, m.RIGHT, buff=0.3)
        times = code("q×k", 22, MUTED).next_to(products, m.LEFT, buff=0.3).align_to(q_lab, m.RIGHT)
        worked = m.VGroup(q_lab, q_row, k_lab, k_row, times, products, pluses, total)
        self.cap = text("÷ √d   ·   d = 4 numbers per vector, so ÷ 2", 22, MUTED).move_to([0, -3.3, 0])

        self.say(
            "[The query of it] [is compared with every word's key], [using the dot product from episode nine]: [multiply matching numbers, then add them up.]",
            m.AnimationGroup(self.new_header("SCORES", "Match the query against every key"), m.LaggedStart(*[m.FadeIn(t, shift=0.2 * m.DOWN) for t in tokens], lag_ratio=0.08)),
            m.AnimationGroup(m.Circumscribe(it, color=QUERY, buff=0.08), m.LaggedStart(*[m.Create(a) for a in reversed(self.arcs)], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(m.VGroup(q_lab, q_row), shift=0.2 * m.RIGHT), m.FadeIn(m.VGroup(k_lab, k_row), shift=0.2 * m.RIGHT)),
            m.Succession(
                m.AnimationGroup(m.FadeIn(times), m.LaggedStart(*[m.TransformFromCopy(m.VGroup(q_row[1][i], k_row[1][i]), products[i]) for i in range(D)], lag_ratio=0.3)),
                m.AnimationGroup(m.FadeIn(pluses), m.FadeIn(total, shift=0.2 * m.LEFT)),
            ),
        )
        self.say(
            "[A close match gives a high score], [and cat scores highest.]",
            m.AnimationGroup(
                m.TransformFromCopy(total, self.score_txt[CAT]),
                m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.DOWN) for i, s in enumerate(self.score_txt) if i != CAT], lag_ratio=0.1),
            ),
            m.AnimationGroup(m.Indicate(self.score_txt[CAT], color=ATTN, scale_factor=1.3), m.Indicate(cat, color=ATTN, scale_factor=1.12)),
        )
        self.say(
            "[Each score is then divided by the square root of the vector size, which is two here], [so that long vectors don't give extreme scores.]",
            m.AnimationGroup(*[m.Transform(s, t) for s, t in zip(self.score_txt, scaled)], m.FadeOut(worked), m.FadeIn(self.cap, shift=0.2 * m.UP)),
            m.Indicate(self.score_txt, color=INK, scale_factor=1.1),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ softmax

    def weights(self):
        self.next_section(
            "Softmax",
            notes="softmax (Episode 16) of the scaled scores: "
            + ", ".join(f"{t} {p}%" for t, p in zip(TOKENS, PCT))
            + ". Computed, not chosen.",
        )
        tokens = self.tokens
        base = -2.75
        self.bars, self.labels = m.VGroup(), m.VGroup()
        for w, p, x in zip(W_IT, PCT, XS):
            bar = m.Rectangle(width=0.8, height=max(0.05, 2.6 * w), stroke_width=0).set_fill(ATTN, opacity=0.85)
            bar.move_to([x, base, 0], aligned_edge=m.DOWN)
            self.bars.add(bar)
            self.labels.add(text(f"{p}%", 22, ATTN, weight=m.BOLD).next_to(bar, m.UP, buff=0.1))
        heavy = weighted_arcs(tokens, W_IT)
        cap = text("softmax · Episode 16: scores → weights that add up to 100%", 22, MUTED).move_to([0, -3.35, 0])

        self.say(
            "[Softmax, from episode sixteen], [turns the scores into weights that add up to one hundred percent.]",
            m.AnimationGroup(self.new_header("SOFTMAX", "Scores become weights"), m.FadeTransform(self.cap, cap)),
            m.AnimationGroup(
                m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in self.bars], lag_ratio=0.08),
                m.LaggedStart(*[m.TransformFromCopy(s, lab) for s, lab in zip(self.score_txt, self.labels)], lag_ratio=0.08),
            ),
        )
        self.cap = cap
        self.say(
            "[Cat gets fifty-four percent of the attention], [and tired gets twenty-one.]",
            m.AnimationGroup(
                m.Transform(self.arcs, heavy),
                m.Indicate(m.VGroup(self.bars[CAT], self.labels[CAT]), color=ATTN),
                m.Indicate(tokens[CAT], color=ATTN, scale_factor=1.12),
            ),
            m.AnimationGroup(m.Indicate(m.VGroup(self.bars[TIRED], self.labels[TIRED]), color=ATTN), m.Indicate(tokens[TIRED], color=ATTN, scale_factor=1.12)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ weighted sum

    def blend(self):
        self.next_section(
            "Weighted sum",
            notes=f"new 'it' = Σ weight × value = [{', '.join(f'{v:.2f}' for v in NEW_IT)}]; cosine similarity with cat's value {COSINES[CAT]:.3f} (Episode 9), the highest of the six.",
        )
        tokens = self.tokens
        weights = m.VGroup(*[code(f"×{w:.2f}", 22, ATTN).move_to([x, 0.25, 0]) for w, x in zip(W_IT, XS)])
        values = m.VGroup(*[column(V[i], VALUE, 20, 1).move_to([x, -0.8, 0]) for i, x in enumerate(XS)])
        flows = [
            m.ShowPassingFlash(self.arcs[i].copy().reverse_points().set_stroke(VALUE, width=7, opacity=1), time_width=0.5)
            for i in range(IT)
        ]
        out_col = column(NEW_IT, VALUE, 20, 2).move_to([-0.3, -2.62, 0])
        out_label = m.VGroup(text("new vector for “it”", 24, INK, weight=m.BOLD), text("Σ weight × value", 22, MUTED))
        out_label.arrange(m.DOWN, buff=0.1, aligned_edge=m.RIGHT).next_to(out_col, m.LEFT, buff=0.45)
        close = m.VGroup(text("cosine similarity with", 22, MUTED), text(f"cat's value: {COSINES[CAT]:.2f}", 22, VALUE, weight=m.BOLD))
        close.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(out_col, m.RIGHT, buff=0.45)

        self.say(
            "[Now every word passes on its value], [scaled by its weight], [and the results are added up.]",
            m.AnimationGroup(
                self.new_header("WEIGHTED SUM", "Blend the values"),
                m.FadeOut(self.bars),
                m.FadeOut(self.cap),
                m.FadeOut(self.score_txt),
                m.LaggedStart(*[m.FadeIn(v, shift=0.2 * m.UP) for v in values], lag_ratio=0.08),
            ),
            m.AnimationGroup(
                *[m.ReplacementTransform(lab, w) for lab, w in zip(self.labels, weights)],
                *[m.AnimationGroup(v[1].animate.set_opacity(0.3 + 0.7 * w / W_IT.max()), v[0].animate.set_stroke(opacity=0.3 + 0.7 * w / W_IT.max()), v[2].animate.set_stroke(opacity=0.3 + 0.7 * w / W_IT.max())) for v, w in zip(values, W_IT)],
            ),
            m.Succession(
                m.LaggedStart(*flows, lag_ratio=0.12),
                m.AnimationGroup(m.FadeIn(out_col, shift=0.2 * m.DOWN), m.FadeIn(out_label)),
            ),
        )
        self.say(
            "[More than half of the new vector for it comes from the cat.] [It now carries the cat's meaning.]",
            m.AnimationGroup(
                m.Indicate(values[CAT], color=VALUE, scale_factor=1.1),
                m.Indicate(weights[CAT], color=ATTN, scale_factor=1.2),
                m.FadeIn(close, shift=0.2 * m.LEFT),
            ),
            m.AnimationGroup(tokens[IT][0].animate.set_stroke(ATTN).set_fill(ATTN_FILL), m.Circumscribe(out_col, color=VALUE, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the formula

    def formula(self):
        self.next_section(
            "The formula",
            notes="Scaled dot-product attention, Attention(Q,K,V) = softmax(QKᵀ/√d)V (Vaswani et al., 2017; d = d_k, the key size). "
            "Decoder (chatbot) models mask later positions before the softmax. The table is the toy's real QKᵀ, then its masked softmax.",
        )
        f = m.MathTex(
            r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V",
            font_size=46,
            color=INK,
            substrings_to_isolate=[r"\mathrm{softmax}", "Q", "K", r"^{\top}", r"\sqrt{d}", "V"],
        ).move_to([-0.7, 1.95, 0])
        parts = {key: [p for p in f if p.tex_string == key] for key in ("Q", "K", "V", r"^{\top}", r"\sqrt{d}", r"\mathrm{softmax}")}
        for key, color in (("Q", QUERY), ("K", KEY), ("V", VALUE)):
            for p in parts[key]:
                p.set_color(color)
        qk = m.VGroup(parts["Q"][1], parts["K"][1], parts[r"^{\top}"][0])
        root, soft, val = parts[r"\sqrt{d}"][0], parts[r"\mathrm{softmax}"][0], parts["V"][1]
        credit = text("Vaswani et al., 2017", 20, MUTED).next_to(f, m.RIGHT, buff=0.5).shift(0.25 * m.DOWN)

        gx = [-4.45 + 0.9 * k for k in range(6)]
        gy = [0.45 - 0.52 * k for k in range(6)]
        cells, nums, after = m.VGroup(), m.VGroup(), m.VGroup()
        for i in range(6):
            for j in range(6):
                cell = m.Rectangle(width=0.86, height=0.48).set_stroke(EDGE, width=1.5).set_fill(CARD, opacity=1).move_to([gx[j], gy[i], 0])
                cells.add(cell)
                nums.add(code(fmt(SCORES[i, j], 2), 20, INK).move_to(cell))
                w = ATTN_TABLE[i, j]
                after.add(code(f"{round(100 * w)}%" if j <= i else "", 20, INK).move_to(cell))
        cols = m.VGroup(*[text(t, 24, KEY, weight=m.SEMIBOLD).move_to([x, 0.98, 0]) for t, x in zip(TOKENS, gx)])
        on_baseline(cols)
        rows_ = m.VGroup(*[text(t, 24, QUERY, weight=m.SEMIBOLD).move_to([0, y, 0]).align_to([gx[0] - 0.58, 0, 0], m.RIGHT) for t, y in zip(TOKENS, gy)])
        grid_note = text("rows: queries · columns: keys", 20, MUTED, t2c={"queries": QUERY, "keys": KEY}).move_to([gx[0] + 2.25, -2.62, 0])
        masked = [i * 6 + j for i in range(6) for j in range(6) if j > i]
        last_row = m.VGroup(*[cells[30 + j] for j in range(6)])

        def legend(term, words, y):
            return m.VGroup(term.move_to([1.65, y, 0]), text(words, 22, MUTED).move_to([0, y, 0]).align_to([2.55, 0, 0], m.LEFT))

        swatch = m.VGroup(m.Rectangle(width=0.6, height=0.42).set_stroke(EDGE, width=1.5).set_fill(BG, opacity=1))
        swatch.add(m.Line(swatch[0].get_corner(m.DL) + [0.12, 0.08, 0], swatch[0].get_corner(m.UR) + [-0.12, -0.08, 0]).set_stroke(FAINT, width=2))
        keys = (
            legend(m.MathTex(r"QK^{\top}", font_size=34, color=INK, tex_to_color_map={"Q": QUERY, "K": KEY}), "every query · every key", 0.75),
            legend(swatch, "no looking ahead (chatbots)", -0.1),
            legend(tex(r"\div\sqrt{d}", 34), "d = 4: keeps scores moderate", -0.95),
            legend(tex(r"\mathrm{softmax}", 34), "each row → weights", -1.8),
            legend(tex("V", 34, VALUE), "blend the values", -2.65),
        )

        self.say(
            "[All of this fits in one formula.] [Stack every query into a matrix Q], [every key into K], [and every value into V.]",
            m.AnimationGroup(self.new_header("THE FORMULA", "Scaled dot-product attention"), m.Write(f), m.FadeIn(credit)),
            m.Indicate(m.VGroup(*parts["Q"]), color=QUERY, scale_factor=1.25),
            m.Indicate(m.VGroup(*parts["K"]), color=KEY, scale_factor=1.25),
            m.Indicate(m.VGroup(*parts["V"]), color=VALUE, scale_factor=1.25),
        )
        self.say(
            "[Q times K transposed scores every query against every key]: [a table, with one row for each word.]",
            m.AnimationGroup(m.Indicate(qk, color=INK, scale_factor=1.2), m.FadeIn(keys[0], shift=0.2 * m.LEFT)),
            m.AnimationGroup(
                m.FadeIn(cols), m.FadeIn(rows_), m.FadeIn(grid_note),
                m.LaggedStart(*[m.FadeIn(m.VGroup(*cells[6 * i : 6 * i + 6], *nums[6 * i : 6 * i + 6])) for i in range(6)], lag_ratio=0.15),
            ),
        )
        self.say(
            "[In a chatbot, words may only look back], [so the scores for later words are blocked.]",
            m.FadeIn(keys[1], shift=0.2 * m.LEFT),
            m.AnimationGroup(*[m.AnimationGroup(cells[k].animate.set_fill(BG), nums[k].animate.set_opacity(0)) for k in masked]),
        )
        self.say(
            "[Divide by the square root of the vector size], [take the softmax of each row], [and multiply by V to blend the values.]",
            m.AnimationGroup(m.Indicate(root, color=INK, scale_factor=1.3), m.FadeIn(keys[2], shift=0.2 * m.LEFT)),
            m.AnimationGroup(
                m.Indicate(soft, color=ATTN, scale_factor=1.15),
                m.FadeIn(keys[3], shift=0.2 * m.LEFT),
                *[m.Transform(nums[k], after[k]) for k in range(36) if k not in masked],
                *[cells[i * 6 + j].animate.set_fill(m.interpolate_color(CARD, ATTN, 0.15 + 0.7 * ATTN_TABLE[i, j])) for i in range(6) for j in range(i + 1)],
            ),
            m.AnimationGroup(m.Indicate(val, color=VALUE, scale_factor=1.3), m.FadeIn(keys[4], shift=0.2 * m.LEFT)),
        )
        self.say(
            "[The bottom row is the one we just worked out for it.]",
            m.Circumscribe(last_row, color=ATTN, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ multi-head

    def heads(self):
        self.next_section(
            "Multi-head",
            notes="Several heads in parallel, each with its own W_Q, W_K, W_V; outputs concatenated and mixed by one more matrix (Vaswani et al. used 8 heads). "
            "The three patterns are illustrative: a pattern + seeded noise, masked softmax; lines where a weight ≥ 0.2.",
        )

        def panel(title, color, table):
            box = card(4.1, 3.9, color, CARD, stroke=2)
            name = text(title, 22, color, weight=m.BOLD).move_to(box.get_top() + 0.42 * m.DOWN)
            top = box.get_top()[1] - 1.0
            left = m.VGroup(*[text(w, 24, INK).move_to([0, top - 0.44 * k, 0]).align_to(box.get_center() + 1.0 * m.LEFT, m.RIGHT) for k, w in enumerate(TOKENS)])
            right = m.VGroup(*[text(w, 24, INK).move_to([0, top - 0.44 * k, 0]).align_to(box.get_center() + 1.0 * m.RIGHT, m.LEFT) for k, w in enumerate(TOKENS)])
            lines = m.VGroup(
                *[
                    m.Line(left[i].get_right() + 0.12 * m.RIGHT, right[j].get_left() + 0.12 * m.LEFT).set_stroke(color, width=1.5 + 7 * table[i, j], opacity=0.35 + 0.65 * table[i, j])
                    for i in range(6)
                    for j in range(i + 1)
                    if table[i, j] >= 0.2
                ]
            )
            return m.VGroup(m.VGroup(box, name, left, right), lines)

        built = m.VGroup(*[panel(title, color, table) for title, color, table in HEADS]).arrange(m.RIGHT, buff=0.3).move_to([0, -0.2, 0])
        joined = text("each head: its own queries, keys and values · outputs joined into one vector per word", 22, MUTED).move_to([0, -2.95, 0])

        self.say(
            "[Real models run several attention heads side by side], [each with its own queries, keys and values.]",
            m.AnimationGroup(self.new_header("MULTI-HEAD", "Several attentions side by side"), m.LaggedStart(*[m.FadeIn(b[0], shift=0.3 * m.UP) for b in built], lag_ratio=0.2)),
            m.FadeIn(joined, shift=0.2 * m.UP),
        )
        self.say(
            "[One might link it to cat], [another follow grammar], [another, the word just before.] [Their outputs are joined back together.]",
            *[m.LaggedStart(*[m.Create(x) for x in b[1]], lag_ratio=0.1) for b in built],
            m.Indicate(joined, color=INK, scale_factor=1.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the block

    def block(self):
        self.next_section(
            "The block",
            notes="Attention, then a position-wise feed-forward network; around each, a residual connection and LayerNorm (the 2017 'Add & Norm'; Episode 25). "
            f"Toy block for 'it': x + attention = [{', '.join(f'{v:.2f}' for v in AFTER_ATTN)}], after the feed-forward [{', '.join(f'{v:.2f}' for v in AFTER_BLOCK)}] "
            "(seeded ReLU network, no LayerNorm in the toy). Vaswani et al. stacked 6; large models stack dozens.",
        )
        y = -0.45
        att = node(ep_recolored("search", ATTN, ATTN_FILL), "Attention", "share context", ATTN, ATTN_FILL, width=2.3, height=1.75, label_size=24).move_to([-2.65, y, 0])
        ffn = node(ep_recolored("gear", FFN, FFN_FILL), "Feed-forward", "refine each word", FFN, FFN_FILL, width=2.3, height=1.75, label_size=24).move_to([1.55, y, 0])

        def plus(x):
            ring = m.Circle(radius=0.24).set_stroke(MUTED, width=3).set_fill(BG, opacity=1).move_to([x, y, 0])
            return m.VGroup(ring, text("+", 28, INK, weight=m.BOLD).move_to(ring))

        add1, add2 = plus(-0.55), plus(3.55)
        rail = att.get_top()[1] + 0.35

        def skip(x0, target):
            path = m.VMobject().set_points_as_corners([[x0, y, 0], [x0, rail, 0], [target.get_x(), rail, 0], [target.get_x(), target.get_top()[1] + 0.35, 0]])
            path.set_stroke(MUTED, width=3)
            tip = arrow([target.get_x(), target.get_top()[1] + 0.4, 0], target.get_top(), MUTED, buff=0.02)
            return m.VGroup(path, tip)

        skips = m.VGroup(skip(-4.1, add1), skip(0.05, add2))
        norms = m.VGroup(*[text("Add & Norm", 20, MUTED, weight=m.SEMIBOLD).next_to(a, m.DOWN, buff=0.22) for a in (add1, add2)])
        links = m.VGroup(
            arrow(att.get_right(), add1.get_left(), MUTED, buff=0.05),
            arrow(add1.get_right(), ffn.get_left(), MUTED, buff=0.05),
            arrow(ffn.get_right(), add2.get_left(), MUTED, buff=0.05),
        )
        frame = card(9.0, 3.0, ATTN, CARD, stroke=2.5).move_to([-0.05, y, 0]).set_z_index(-2)
        shadows = m.VGroup(
            *[
                card(9.0, 3.0, ATTN, CARD, stroke=2).set_stroke(opacity=0.55 - 0.15 * k).move_to(frame.get_center() + 0.26 * (k + 1) * m.UP).set_z_index(-3 - k)
                for k in range(3)
            ]
        )
        title = text("Transformer block", 26, ATTN, weight=m.BOLD).move_to([0, 2.3, 0])
        vec_in = column(X[IT], SKY).move_to([-6.0, y, 0])
        vec_out = column(AFTER_BLOCK, ATTN).move_to([5.75, y, 0])
        go_in = arrow(vec_in.get_right(), att.get_left(), MUTED, buff=0.1)
        go_out = arrow(add2.get_right(), vec_out.get_left(), MUTED, buff=0.1)
        layers = text("×  dozens of layers", 28, ATTN, weight=m.BOLD).move_to([0, -2.4, 0])
        later = pill("residuals and normalisation · Episode 25", MUTED, CARD, 20).move_to([0, -3.25, 0])
        self.block_out = vec_out

        self.say(
            "[After attention, each word's vector] [passes through a feed-forward network, the same small network for every word.]",
            m.AnimationGroup(self.new_header("THE BLOCK", "Attention, then feed-forward"), m.FadeIn(vec_in), m.GrowArrow(go_in), m.FadeIn(att, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(add1), m.GrowArrow(links[1]), m.FadeIn(ffn, shift=0.2 * m.UP), pulse(links[1], FFN)),
        )
        self.say(
            "[Around each step, a shortcut adds the input back in], [and layer norm keeps the numbers steady.] [Episode twenty-five covers both.]",
            m.AnimationGroup(m.Create(skips[0]), m.Create(skips[1]), m.GrowArrow(links[2]), m.FadeIn(add2)),
            m.FadeIn(norms, shift=0.15 * m.UP),
            m.FadeIn(later, shift=0.2 * m.UP),
        )
        self.say(
            "[Together, that's one transformer block.] [Big models stack dozens of them], [each one refining the meaning a little more.]",
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(title, shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, shift=0.2 * m.UP) for s in shadows], lag_ratio=0.3), m.FadeIn(layers)),
            m.AnimationGroup(m.GrowArrow(go_out), m.FadeIn(vec_out, shift=0.2 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ predict

    def predict(self):
        self.next_section(
            "Predict",
            notes="The last vector → a score for every token in the vocabulary → softmax → pick one, append, repeat. Illustrative: 50,000 tokens, "
            + ", ".join(f"{w} {p}%" for w, p in zip(CANDIDATES, NEXT_PCT))
            + f", the other {VOCAB - len(CANDIDATES):,} tokens {REST_PCT}% together (each under 0.01%). Training: the same next-token prediction on huge amounts of text (Episode 27).",
        )
        xs = [-5.4 + 1.8 * k for k in range(7)]
        tokens = token_row(1.55, xs=xs[:6])
        blank = chip("?", GOOD, GOOD_FILL).move_to([xs[6], 1.55, 0])
        last = column(AFTER_BLOCK, ATTN).move_to([-4.7, -1.0, 0])
        last_label = text("“it”, after the last block", 22, MUTED).next_to(last, m.DOWN, buff=0.3)
        heading = text("a score for every token it knows", 22, MUTED).move_to([1.25, 0.85, 0])
        rows = m.VGroup()
        for k, (word, p) in enumerate(zip(CANDIDATES, NEXT[: len(CANDIDATES)])):
            yk = 0.25 - 0.62 * k
            top = k == 0
            label = text(word, 26, INK if top else MUTED, weight=m.BOLD if top else m.NORMAL).move_to([0, yk, 0]).align_to([-0.75, 0, 0], m.RIGHT)
            bar = m.Rectangle(width=max(0.05, 6 * p), height=0.38, stroke_width=0).set_fill(GOOD if top else MUTED, opacity=0.85)
            bar.move_to([-0.55, yk, 0], aligned_edge=m.LEFT)
            value = text(f"{NEXT_PCT[k]}%", 22, GOOD if top else MUTED, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.15)
            rows.add(m.VGroup(label, bar, value))
        rest = text(f"+ {VOCAB - len(CANDIDATES):,} more tokens · {REST_PCT}% together", 22, MUTED).move_to([0, 0.25 - 0.62 * 4, 0]).align_to([-0.55, 0, 0], m.LEFT)
        feed = arrow(last.get_right(), [rows.get_left()[0] - 0.25, last.get_y(), 0], ATTN)
        picked = chip("slept", GOOD, GOOD_FILL).move_to(blank)
        again = text("↻  add it to the sentence, run the stack again", 26, GOOD, weight=m.SEMIBOLD).move_to([0, -3.3, 0])
        steps = m.VGroup(
            pill("predict the next token", GOOD, GOOD_FILL, 20),
            pill("check the real one", GOOD, GOOD_FILL, 20),
            pill("nudge every weight", GOOD, GOOD_FILL, 20),
        ).arrange(m.RIGHT, buff=0.75).move_to([0, -3.3, 0])
        step_links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.1) for a, b in zip(steps, steps[1:])])

        self.say(
            "[At the top of the stack, the last word's vector] [becomes a score for every token in the vocabulary.]",
            m.AnimationGroup(
                self.new_header("PREDICT", "One token at a time"),
                m.LaggedStart(*[m.FadeIn(t, shift=0.2 * m.DOWN) for t in (*tokens, blank)], lag_ratio=0.06),
                m.FadeIn(last, shift=0.2 * m.UP),
                m.FadeIn(last_label),
            ),
            m.AnimationGroup(
                m.GrowArrow(feed),
                m.FadeIn(heading),
                m.LaggedStart(*[m.FadeIn(r[:2], shift=0.2 * m.RIGHT) for r in rows], lag_ratio=0.15),
            ),
        )
        self.say(
            "[Softmax turns those scores into probabilities.] [Here, for example, slept comes out on top, at sixty-two percent.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r[2]) for r in rows], lag_ratio=0.1), m.FadeIn(rest)),
            m.Circumscribe(rows[0], color=GOOD, buff=0.1),
        )
        self.say(
            "[It's added to the sentence], [and the whole stack runs again, one token at a time.]",
            m.AnimationGroup(m.TransformFromCopy(rows[0][0], picked), m.FadeOut(blank)),
            m.FadeIn(again, shift=0.2 * m.UP),
        )
        self.say(
            "[Training is the same game]: [predict the next token on huge amounts of text], [and nudge every weight to do better.]",
            m.FadeOut(again, shift=0.2 * m.DOWN),
            m.AnimationGroup(m.FadeIn(steps[0], shift=0.2 * m.UP), m.GrowArrow(step_links[0]), m.FadeIn(steps[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(step_links[1]), m.FadeIn(steps[2], shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ why it won

    def why(self):
        self.next_section(
            "Why it won",
            notes="Parallel training (all positions at once), direct long-range links, steady gains with scale. Catch: attention compares every pair, so its cost grows with the square of the length (Episode 37: efficient attention).",
        )
        tiles = m.VGroup(
            tile("bolt", "Fast to train", "Every word at once: made\nfor parallel hardware.", AMBER),
            tile(ep_recolored("search", SKY, SKY_FILL), "Long reach", "Any word can look at any\nother, however far apart.", SKY),
            tile("chart", "Scales up", "Bigger models and more\ndata keep paying off.", GOOD),
            tile("alert", "The catch", "Cost grows with the square\nof the text length.", BAD),
        )
        tiles.arrange_in_grid(rows=2, cols=2, buff=(0.45, 0.4)).move_to([0, -0.3, 0])
        later = pill("Episode 37 · Efficient Attention", PINK, PINK_FILL, 20).move_to([0, -3.3, 0])

        self.say(
            "[Why did transformers take over?] [They train on every word in parallel], [any word can reach any other directly], [and they keep improving as they grow.]",
            self.new_header("WHY IT WON", "Parallel, far-reaching, scalable"),
            *[m.FadeIn(t, shift=0.3 * m.UP) for t in tiles[:3]],
        )
        self.say(
            "[The catch]: every word is compared with every other, so the cost grows with the square of the text length. [Episode thirty-seven shows how that's tamed.]",
            m.FadeIn(tiles[3], shift=0.3 * m.UP),
            m.FadeIn(later, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Embed, attend, predict.")
        halo = glow(ATTN, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Embed", "Attend", "Predict"), (SKY, ATTN, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        top_row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text(
            "Queries ask, keys answer, values carry the meaning.", 32, MUTED, t2c={"Queries": QUERY, "keys": KEY, "values": VALUE}
        ).next_to(top_row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [embed], [attend], [predict.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Queries ask, keys answer, and values carry the meaning], layer after layer.", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, top_row, tagline)], run_time=0.9)
