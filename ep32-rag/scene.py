"""AI, Explained · Episode 32 · Retrieval-Augmented Generation

A series remake of the standalone film ~/rag-video/rag.py ("What is RAG?", 3:47): the same
running example (a customer asks for the refund window for Enterprise plans; a plain model makes
up "90 days", RAG answers "30 days" with citations), its best pictures (the made-up answer, the
closed-/open-book exam, documents → chunks → embedding model → a map of meaning, the question's
star and its nearest chunks, the prompt.txt window, the cited answer, the five-node pipeline),
plus the math of "nearest" (cosine similarity, callback to Episode 9), callbacks to Episodes 22
(embeddings) and 31 (prompting), and the series' recap.

Everything plotted is computed below from seeded numpy toy vectors (256 numbers each, invented
for teaching): the cosine scores in the ranked list, the angles in the angle diagram, and the
2-D map (metric MDS of the vectors' distances, so close on the map means close in the space).
The asserts tie every number the narration quotes to the computation.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

R_COL, R_FILL = TEAL, TEAL_FILL  # retrieve
A_COL, A_FILL = AMBER, AMBER_FILL  # augment
G_COL, G_FILL = VIOLET, VIOLET_FILL  # generate
TOPICS = {"refunds": PINK, "pricing": BLUE, "shipping": LIME}

QUESTION = "What's the refund window\nfor Enterprise plans?"

# ---------------------------------------------------------------------------- the toy embeddings

SEED, DIM = 32, 256
_rng = np.random.default_rng(SEED)


def unit(v):
    return v / np.linalg.norm(v)


# a few directions of meaning; each text is a mix of them plus its own noise
_CONCEPTS = {k: unit(_rng.normal(size=DIM)) for k in ("refund", "price", "ship", "enterprise", "billing", "window", "ask", "o1", "o2", "o3", "o4")}
CHUNKS = [  # (topic, text as shown in lists, mix)
    ("refunds", "Enterprise: full refund\nwithin 30 days.", dict(refund=1.0, enterprise=0.4, window=0.4)),
    ("refunds", "Request refunds in the\nbilling portal.", dict(refund=1.0, billing=0.4)),
    ("refunds", "Annual plans are prorated\nafter 30 days.", dict(refund=0.8, window=0.35, price=0.35)),
    ("pricing", "Enterprise plan: billed\nper seat, yearly.", dict(price=1.0, enterprise=0.5, billing=0.3)),
    ("pricing", "Pro plan: $20 per user\nper month.", dict(price=1.0, billing=0.2)),
    ("pricing", "Discounts for nonprofits\nand schools.", dict(price=1.0)),
    ("shipping", "Orders ship within\n2 business days.", dict(ship=1.0, window=0.2)),
    ("shipping", "Free shipping on\norders over $50.", dict(ship=1.0)),
    ("shipping", "Track your parcel from\nthe order page.", dict(ship=1.0)),
]
OTHER_MIX = [dict(o1=1.0)] * 3 + [dict(o2=1.0)] * 2 + [dict(o3=1.0)] * 3 + [dict(o4=1.0)] * 2  # chunks of other documents


def embed(mix, noise=0.45):
    v = sum(w * _CONCEPTS[k] for k, w in mix.items())
    return unit(v + noise * np.linalg.norm(v) * unit(_rng.normal(size=DIM)))


VECS = np.array([embed(mix) for _, _, mix in CHUNKS] + [embed(mix, 0.6) for mix in OTHER_MIX])
QVEC = embed(dict(refund=1.0, enterprise=0.4, window=0.45, ask=0.35))
assert np.allclose(np.linalg.norm(VECS, axis=1), 1) and np.isclose(np.linalg.norm(QVEC), 1)
SIMS = VECS @ QVEC  # cosine similarity: every vector has length one
RANK = np.argsort(-SIMS)
TOP3 = [int(i) for i in RANK[:3]]
DEG = np.degrees(np.arccos(SIMS))
R1, R2, R3, P1, S1 = 0, 1, 2, 3, 6  # the chunks in the ranked list (document order)
LISTED = [R1, R2, R3, P1, S1]

assert TOP3 == [R1, R3, R2]  # "the three closest chunks are all about refunds" (R3 outranks R2: the list re-sorts)
assert round(SIMS[R1], 1) == 0.8 and round(DEG[R1]) == 38  # "about zero point eight", "about thirty-eight degrees"
assert round(SIMS[P1], 1) == 0.1 and SIMS[P1] == max(SIMS[3:6])  # "only about zero point one"
assert all(SIMS[i] < SIMS[R2] for i in range(3, len(SIMS)))  # nothing else comes near the refunds
assert sorted(LISTED, key=lambda i: -SIMS[i]) == [R1, R3, R2, P1, S1]


def mds(vectors, seed, steps=500):
    """Metric MDS (SMACOF): 2-D points whose distances match the vectors' distances as well as
    possible. Rotated so the refunds sit at the left, and reflected (if need be) so shipping sits low."""
    target = np.linalg.norm(vectors[:, None] - vectors[None], axis=2)
    x = np.random.default_rng(seed).normal(size=(len(vectors), 2))
    for _ in range(steps):
        d = np.linalg.norm(x[:, None] - x[None], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(d > 0, -target / d, 0.0)
        b[np.diag_indices_from(b)] = -b.sum(1)
        x = b @ x / len(vectors)
    x -= x.mean(0)
    c = x[:3].mean(0)
    a = np.radians(165) - np.arctan2(c[1], c[0])
    x = x @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
    if x[6:9, 1].mean() > 0:
        t = 2 * np.radians(165)
        x = x @ np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]]).T
    return x


_MAP = mds(np.vstack([VECS, QVEC]), SEED + 1)
MAP_SCALE = min(2.25 / np.abs(_MAP[:, 0]).max(), 1.75 / np.abs(_MAP[:, 1]).max())
MAP, QMAP = _MAP[:-1] * MAP_SCALE, _MAP[-1] * MAP_SCALE
assert set(np.argsort(np.linalg.norm(MAP - QMAP, axis=1))[:3]) == set(TOP3)  # the map agrees with the scores
LABEL_SHIFT = {"refunds": (0.0, 0.75), "pricing": (0.0, 0.68), "shipping": (-0.85, -0.25)}  # topic names beside their clusters
assert MAP[0:3, 0].max() < 0 < MAP[3:6, 0].min() and MAP[6:9, 1].max() < -1  # refunds left, pricing right, shipping low

# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def num(v, places=2):
    return f"{v:.{places}f}".replace("-", "−")


def callback(number, title):
    """A small muted pill: a reminder of an earlier episode."""
    return pill(f"Episode {number} · {title}", MUTED, CARD, 20)


def network(color, width=1.2, height=0.95):
    """A tiny neural network: three layers of nodes, fully connected."""
    layers = []
    for x, n in zip(np.linspace(-width / 2, width / 2, 3), (3, 4, 2)):
        layers.append([m.Dot([x, y, 0], radius=0.065, color=color) for y in np.linspace(-height / 2, height / 2, n)])
    edges = m.VGroup(
        *[
            m.Line(a.get_center(), b.get_center()).set_stroke(color, width=1.5, opacity=0.4)
            for left, right in zip(layers, layers[1:])
            for a in left
            for b in right
        ]
    )
    return m.VGroup(edges, m.VGroup(*[dot for layer in layers for dot in layer]))


def chunk_bar(color):
    return m.RoundedRectangle(corner_radius=0.08, width=1.1, height=0.26, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.3)


def compact_node(name, label, color, fill, width=1.9):
    box = card(width, 1.5, color, fill)
    body = m.VGroup(icon(name, 0.6), text(label, 22, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.14).move_to(box)
    return m.VGroup(box, body)


def values(v):
    """First three numbers and the last of a vector, as the kit's vector() shows them."""
    return [num(x) for x in (*v[:3], v[-1])]


# ---------------------------------------------------------------------------- the episode


class Ep32RAG(Episode):
    number = 32

    def lesson(self):
        self.problem()
        self.idea()
        self.indexing()
        self.retrieve()
        self.similarity()
        self.augment()
        self.generate()
        self.big_picture()
        self.why()
        self.recap()

    # ------------------------------------------------------------------ the problem

    def problem(self):
        self.next_section("The problem", notes="A plain model makes up a policy it never saw. The 90/30 days are illustrative.")
        user = icon("user", 1.1).move_to([-6.0, 1.0, 0])
        ask = bubble(text(QUESTION, 30), SKY, SKY_FILL, "left")
        ask.next_to(user, m.RIGHT, buff=0.35).shift(0.75 * m.UP)
        llm = icon("llm", 1.5).move_to([5.75, -0.9, 0])
        llm_label = text("LLM", 24, G_COL, weight=m.BOLD).next_to(llm, m.DOWN, buff=0.2)
        reply = bubble(text("Enterprise customers get a full\nrefund within 90 days!", 30), G_COL, G_FILL, "right")
        reply.next_to(llm, m.LEFT, buff=0.35).shift(0.75 * m.UP)
        stamp = pill("✗  MADE UP: the policy says 30 days", BAD, BAD_FILL, 24)
        stamp.next_to(reply, m.DOWN, buff=0.35).align_to(reply, m.LEFT).shift(0.3 * m.RIGHT)
        reasons = m.VGroup()
        for name, words, color in (("clock", "knowledge has a cutoff", SKY), ("lock", "never saw your docs", AMBER), ("alert", "fills gaps with guesses", BAD)):
            reasons.add(m.VGroup(icon(name, 0.55), text(words, 22, color, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2))
        reasons.arrange(m.RIGHT, buff=0.6).move_to([0, -3.05, 0])

        self.say(
            "[Imagine a customer asks a support chatbot]: [what's the refund window for Enterprise plans?]",
            m.AnimationGroup(self.new_header("THE PROBLEM", "A model only knows what it was trained on"), m.FadeIn(user, shift=0.3 * m.RIGHT)),
            m.FadeIn(ask, shift=0.3 * m.UP, scale=0.95),
        )
        self.say(
            "[The language model] [thinks for a moment], and answers with total confidence: "
            "[Enterprise customers get a full refund within ninety days!]",
            m.AnimationGroup(*[m.FadeIn(part, shift=0.2 * m.UP) for part in llm], m.FadeIn(llm_label), run_time=0.5),  # done before the sparkles turn
            m.AnimationGroup(m.Rotate(llm[14], angle=m.PI / 2), m.Rotate(llm[15], angle=-m.PI / 2)),
            m.FadeIn(reply, shift=0.3 * m.UP, scale=0.95),
        )
        self.say(
            "There's just one problem. [It made that up.] The real policy says thirty days, and the model has never seen it.",
            m.AnimationGroup(
                reply[0].animate.set_stroke(BAD),
                reply[1].animate.set_stroke(BAD),
                m.Succession(m.FadeIn(stamp, scale=1.4, run_time=0.5), m.Wiggle(stamp, scale_value=1.15, run_time=0.9)),
            ),
        )
        self.say(
            "[Its knowledge stops at a training cutoff], [it never read your private documents], [so it fills the gap with a guess that sounds right.]",
            *[m.FadeIn(r, shift=0.3 * m.UP) for r in reasons],
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ the idea

    def idea(self):
        self.next_section("The idea", notes="Closed-book versus open-book exam. The term RAG: Lewis et al., 2020 (NeurIPS). Callback: Episode 31, in-context learning. Facts checked 2026-10-21.")

        def panel(title, art, caption, verdict):
            box = card(6.2, 4.5, EDGE, CARD)
            body = m.VGroup(text(title, 32, INK, weight=m.BOLD), art, text(caption, 24, MUTED), verdict).arrange(m.DOWN, buff=0.34)
            return m.VGroup(box, body.move_to(box))

        open_art = m.VGroup(icon("book", 1.25), icon("llm", 1.45)).arrange(m.RIGHT, buff=0.45)
        closed = panel("Closed-book exam", icon("llm", 1.45), "answers from memory", pill("✗  guesses", BAD, BAD_FILL))
        opened = panel("Open-book exam", open_art, "looks up the facts, then answers", pill("✓  grounded", GOOD, GOOD_FILL))
        m.VGroup(closed, opened).arrange(m.RIGHT, buff=0.55).move_to(0.2 * m.DOWN)
        opened[1].shift((closed[1][0].get_y() - opened[1][0].get_y()) * m.UP)
        book = opened[1][1][0]
        source = text("Lewis et al., 2020", 20, MUTED).next_to(opened[0], m.DOWN, buff=0.15).align_to(opened[0], m.RIGHT)
        line = on_baseline(
            m.VGroup(
                text("Episode 31: a model learns from its prompt.", 26, INK, weight=m.SEMIBOLD, t2c={"Episode 31:": MUTED}),
                text("RAG puts the facts there.", 26, A_COL, weight=m.SEMIBOLD),
            ).arrange(m.RIGHT, buff=0.3)
        ).move_to([0, -3.3, 0])

        self.say(
            "[A plain language model is like a student in a closed-book exam.] It answers from memory, and [when it isn't sure, it guesses.]",
            m.AnimationGroup(self.new_header("THE IDEA", "Let the model look things up first"), m.FadeIn(m.VGroup(closed[0], closed[1][:3]), shift=0.4 * m.UP)),
            m.FadeIn(closed[1][3], scale=1.2),
        )
        self.say(
            "[Retrieval-augmented generation, or RAG, makes it an open-book exam]: "
            "[look up the relevant facts first], [then answer from them.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(opened[0], opened[1][0], opened[1][1][1]), shift=0.4 * m.UP), m.FadeIn(source)),
            m.FadeIn(book, shift=0.6 * m.RIGHT, scale=0.7),
            m.FadeIn(opened[1][2:], shift=0.2 * m.UP),
        )
        self.say(
            "[In episode thirty-one, we saw that a model can learn from its prompt], without changing a single weight. "
            "[RAG simply puts the right facts into that prompt.]",
            m.FadeIn(line[0], shift=0.2 * m.UP),
            m.FadeIn(line[1], shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ indexing

    def indexing(self):
        self.next_section(
            "Indexing",
            notes="Split, embed and store the documents, once, ahead of time. The documents and their chunks are illustrative; "
            f"the vectors are seeded toy vectors ({DIM} numbers each); the map is their metric MDS layout. "
            "Real embedding models output hundreds to thousands of numbers (e.g. 384, 768, 1536, 3072). Callback: Episode 22.",
        )
        specs = [("refunds.pdf", PINK), ("pricing.md", BLUE), ("shipping.html", LIME)]
        docs = m.VGroup()
        for name, col in specs:
            doc = icon("doc", 0.95)
            doc[2].set_fill(col, opacity=1)
            docs.add(m.VGroup(doc, code(name, 20, col)).arrange(m.DOWN, buff=0.14))
        docs.arrange(m.DOWN, buff=0.38).move_to([-5.75, -0.45, 0])

        chunks = m.VGroup()
        for (_, col), group in zip(specs, docs):
            chunks.add(m.VGroup(*[chunk_bar(col) for _ in range(3)]).arrange(m.DOWN, buff=0.1).move_to([-3.75, group[0].get_y(), 0]))

        embed_box = card(2.0, 2.5, SKY, SKY_FILL).move_to([-1.85, -0.45, 0])
        net = network(SKY).move_to(embed_box.get_center() + 0.35 * m.UP)
        embed_label = text("Embedding\nmodel", 22, INK, weight=m.SEMIBOLD).next_to(net, m.DOWN, buff=0.28)
        embedder = m.VGroup(embed_box, net, embed_label)
        vec = vector(SKY, values(VECS[R1]), 22).move_to([0.2, -0.45, 0])
        many = m.VGroup(text("hundreds to", 20, MUTED), text("thousands", 20, MUTED)).arrange(m.DOWN, buff=0.08).next_to(vec, m.DOWN, buff=0.3)

        caps = m.VGroup(
            text("1  Split", 24, SKY, weight=m.BOLD).move_to([-3.75, 2.45, 0]),
            text("2  Embed", 24, SKY, weight=m.BOLD).move_to([-1.85, 2.45, 0]),
            text("3  Store", 24, SKY, weight=m.BOLD).move_to([4.0, 2.45, 0]),
        )
        recall = callback(22, "Embeddings").move_to([-1.85, -2.95, 0])

        panel_box = card(5.5, 5.4, R_COL, CARD, stroke=2).move_to([4.0, -0.45, 0])
        panel_box.set_fill(CARD, opacity=0.6)
        panel_head = m.VGroup(icon("database", 0.5), text("Vector database", 24, INK, weight=m.BOLD))
        panel_head.arrange(m.RIGHT, buff=0.2).next_to(panel_box.get_corner(m.UL), m.DR, buff=0.28)
        self.plot = np.array([4.0, -0.75, 0])
        self.panel = m.VGroup(panel_box, panel_head)

        topic_of = [t for t, _, _ in CHUNKS]
        dots = [m.Dot(self.plot + [*MAP[i], 0], radius=0.12, color=TOPICS[topic_of[i]]) for i in range(len(CHUNKS))]
        others = m.VGroup(*[m.Dot(self.plot + [*MAP[i], 0], radius=0.09, color=FAINT) for i in range(len(CHUNKS), len(MAP))])
        labels = m.VGroup()
        for k, (topic, col) in enumerate(TOPICS.items()):
            centre = MAP[3 * k : 3 * k + 3].mean(0) + LABEL_SHIFT[topic]
            labels.add(text(topic, 20, col, weight=m.SEMIBOLD).move_to(self.plot + [*centre, 0]))
        caption = text(f"toy vectors, {DIM} numbers each, drawn in 2-D", 20, MUTED).move_to([4.0, -3.5, 0])
        for mob in (chunks, vec, others, labels, *dots):
            mob.set_z_index(2)  # drawn over the panel, wherever its group joined the scene
        first = chunks[0][0]
        rest = [(bar, dots[3 * i + j]) for i, bars in enumerate(chunks) for j, bar in enumerate(bars) if (i, j) != (0, 0)]

        self.say(
            "[First, we prepare the knowledge, once, ahead of time.] [For example, say we have a refund policy, a pricing page, and a shipping guide.]",
            self.new_header("STEP 0 · INDEX", "Prepare the documents, once"),
            m.LaggedStart(*[m.FadeIn(d, shift=0.3 * m.RIGHT) for d in docs], lag_ratio=0.25),
        )
        self.say(
            "[We split each document into chunks], a few sentences each: small enough to be specific, big enough to make sense on their own.",
            m.AnimationGroup(
                m.FadeIn(caps[0]),
                m.LaggedStart(*[m.ReplacementTransform(d[0].copy(), bars) for d, bars in zip(docs, chunks)], lag_ratio=0.25),
            ),
        )
        self.say(
            "[Next, an embedding model reads each chunk], [and turns it into a vector]: [a list of hundreds, or even thousands, of numbers.]",
            m.Succession(
                m.AnimationGroup(m.FadeIn(caps[1]), m.FadeIn(embedder, shift=0.2 * m.UP), run_time=0.9),
                first.animate(run_time=0.9).move_to(embed_box).scale(0.35).set_opacity(0),
            ),
            m.AnimationGroup(m.ShowPassingFlash(net[0].copy().set_stroke(SKY, width=3, opacity=1), time_width=0.6), m.FadeIn(vec, shift=0.5 * m.RIGHT)),
            m.FadeIn(many, shift=0.15 * m.UP),
        )
        self.remove(first)
        self.say(
            "[It's the idea from episode twenty-two], [applied to whole passages]: similar meaning, nearby vectors.",
            m.FadeIn(recall, shift=0.2 * m.UP),
            m.Circumscribe(vec, color=SKY, buff=0.12),
        )
        self.say(
            "[The vectors are stored in a vector database], [where each chunk becomes a point on a map of meaning.]",
            m.AnimationGroup(m.FadeIn(caps[2]), m.FadeIn(self.panel), m.FadeOut(many)),
            m.Succession(
                m.ReplacementTransform(vec, dots[R1], run_time=1.0),
                m.LaggedStart(*[m.ReplacementTransform(bar, dot, path_arc=-0.6) for bar, dot in rest], lag_ratio=0.12, run_time=2.4),
                m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in others], lag_ratio=0.08), m.FadeIn(caption), run_time=1.2),
            ),
        )
        self.say(
            "[Chunks about refunds end up close together], [and so do the ones about pricing], [and shipping.]",
            *[
                m.AnimationGroup(m.FadeIn(lab, shift=0.15 * m.UP), *[d.animate(rate_func=m.there_and_back).scale(1.5) for d in dots[3 * k : 3 * k + 3]])
                for k, lab in enumerate(labels)
            ],
        )
        self.wait(0.4)
        self.dots, self.labels, self.others, self.caption = dots, labels, others, caption
        self.play(*[m.FadeOut(mob) for mob in (docs, embedder, caps, recall)], run_time=0.8)

    # ------------------------------------------------------------------ retrieve

    def retrieve(self):
        self.next_section("Retrieve", notes="Embed the question with the same model; take the nearest chunks (the top 3 by cosine are the 3 refund chunks).")
        user = icon("user", 0.9).move_to([-6.3, 1.2, 0])
        ask = bubble(text(QUESTION, 26), SKY, SKY_FILL, "left")
        ask.next_to(user, m.RIGHT, buff=0.3).shift(0.7 * m.UP)
        qvec = vector(SKY, values(QVEC), 22).move_to([-3.6, -1.2, 0])
        qname = text("the question's vector", 22, SKY, weight=m.SEMIBOLD).next_to(qvec, m.DOWN, buff=0.3)
        star = m.Star(5, outer_radius=0.2, color=SKY, fill_opacity=1).move_to(self.plot + [*QMAP, 0])
        star_label = text("your question", 20, SKY, weight=m.SEMIBOLD).next_to(star, m.DOWN, buff=0.16)
        hits = [self.dots[i] for i in TOP3]
        links = m.VGroup(*[m.DashedLine(star.get_center(), d.get_center(), dash_length=0.08, buff=0.18).set_stroke(R_COL, width=3) for d in hits])
        rings = m.VGroup(*[m.Circle(radius=0.21, color=R_COL, stroke_width=3).move_to(d) for d in hits])
        nearest = text("top 3 nearest chunks", 22, R_COL, weight=m.SEMIBOLD).move_to(self.caption)
        for mob in (star, star_label, links, rings):
            mob.set_z_index(3)

        self.say(
            "[Now a real question comes in.] [The same embedding model turns it into a vector], [so it lands on the same map.]",
            m.AnimationGroup(
                self.new_header("R · RETRIEVE", "Find the chunks nearest the question"),
                m.FadeIn(user, shift=0.3 * m.RIGHT),
                m.FadeIn(ask, shift=0.3 * m.UP, scale=0.95),
            ),
            m.AnimationGroup(m.TransformFromCopy(ask[2], qvec), m.FadeIn(qname, shift=0.15 * m.UP)),
            m.AnimationGroup(m.TransformFromCopy(qvec, star, path_arc=-0.5), m.FadeIn(star_label, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[Then we search for the nearest points.] [The three closest chunks are all about refunds]: exactly what we need.",
            m.Broadcast(m.Circle(radius=1.3, color=R_COL, stroke_width=3), focal_point=star.get_center(), n_mobs=3),
            m.AnimationGroup(
                m.LaggedStart(*[m.Create(link) for link in links], lag_ratio=0.2),
                m.LaggedStart(*[m.Create(ring) for ring in rings], lag_ratio=0.2),
                *[d.animate.scale(1.3) for d in hits],
                m.FadeOut(self.caption),
                m.FadeIn(nearest),
            ),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ similarity

    def similarity(self):
        listed = ", ".join(f"{CHUNKS[i][1].replace(chr(10), ' ')} {SIMS[i]:.3f} ({DEG[i]:.1f}°)" for i in LISTED)
        self.next_section(
            "Similarity",
            notes=f"cos θ = q·d / (‖q‖‖d‖). Toy scores: {listed}. Top 3 of all {len(SIMS)} chunks: the refund chunks. "
            "Real vector databases use approximate nearest-neighbour indexes for large collections. Callback: Episode 9.",
        )
        recall = callback(9, "Vectors & Similarity").to_corner(m.UR, buff=0.5)
        formula = m.MathTex(
            r"\cos\theta", "=", r"\frac{", r"\mathbf{q}", r"\cdot", r"\mathbf{d}", "}{", r"\left\| \mathbf{q} \right\|", r"\left\| \mathbf{d} \right\|", "}",
            font_size=48, color=INK,
        ).move_to([-4.85, 1.7, 0])
        formula[0].set_color(AMBER)
        for k in (3, 7):
            formula[k].set_color(SKY)
        top_part = m.VGroup(formula[3], formula[4], formula[5])
        bottom_part = m.VGroup(formula[7], formula[8])
        x0 = formula.get_right()[0] + 0.4
        dot_label = m.VGroup(text("dot product", 22, INK, weight=m.SEMIBOLD), text("multiply pairs, add up", 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        len_label = m.VGroup(text("lengths", 22, INK, weight=m.SEMIBOLD), text("only direction counts", 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        dot_label.move_to([0, top_part.get_y() + 0.2, 0]).align_to([x0, 0, 0], m.LEFT)
        len_label.move_to([0, bottom_part.get_y() - 0.2, 0]).align_to([x0, 0, 0], m.LEFT)

        # the angle diagram: q fixed, d at an angle θ (and a length) that can change
        O, L = np.array([-6.3, -2.95, 0.0]), 2.55
        theta, size = m.ValueTracker(60.0), m.ValueTracker(1.0)

        def tip(deg, r):
            t = np.radians(deg)
            return O + r * np.array([np.cos(t), np.sin(t), 0.0])

        q_arrow = m.Arrow(O, O + L * m.RIGHT, buff=0, color=SKY, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        q_tag = tex(r"\mathbf{q}", 36, SKY).next_to(q_arrow.get_end(), m.RIGHT, buff=0.15)
        d_arrow = m.always_redraw(
            lambda: m.Arrow(O, tip(theta.get_value(), L * size.get_value()), buff=0, color=INK, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        )
        d_tag = tex(r"\mathbf{d}", 36, INK)
        d_tag.add_updater(lambda t: t.move_to(tip(theta.get_value() + 7, L * size.get_value() + 0.3)))
        arc = m.always_redraw(lambda: m.Arc(radius=0.7, start_angle=0, angle=np.radians(max(theta.get_value(), 0.5)), arc_center=O).set_stroke(AMBER, width=4))
        th_tag = tex(r"\theta", 30, AMBER)
        th_tag.add_updater(lambda t: t.move_to(tip(theta.get_value() / 2, 1.02)))
        legend = m.VGroup(
            text("q  the question", 20, SKY, weight=m.SEMIBOLD, t2c={"the question": MUTED}),
            text("d  a chunk", 20, INK, weight=m.SEMIBOLD, t2c={"a chunk": MUTED}),
        ).arrange(m.RIGHT, buff=0.5).move_to([-4.6, -3.5, 0])
        cos_label = tex(r"\cos\theta", 36, AMBER)
        cos_num = m.DecimalNumber(0.5, num_decimal_places=2, mob_class=Mono, font_size=44, color=AMBER)
        readout = m.VGroup(cos_label, cos_num).arrange(m.DOWN, buff=0.22).move_to([-2.0, -1.2, 0])

        def clean(v):
            return 0.0 if abs(v) < 0.005 else float(v)

        anchor = cos_num.get_center()
        cos_num.add_updater(lambda d: d.set_value(clean(np.cos(np.radians(theta.get_value())))).move_to(anchor))
        angle_note = text(f"θ ≈ {DEG[R1]:.0f}°", 24, AMBER, weight=m.SEMIBOLD).next_to(readout, m.DOWN, buff=0.3)
        movers = [d_arrow, d_tag, arc, th_tag, cos_num]

        # the ranked list: five of the chunks, in document order, then by score
        slots = [1.88, 0.96, 0.04, -1.1, -2.02]  # a wider gap under the top three
        BAR0, BAR_W = 4.1, 1.4
        rows, bars, scores, marks = [], [], [], []
        for k, i in enumerate(LISTED):
            topic, words, _ = CHUNKS[i]
            col = TOPICS[topic]
            box = card(6.55, 0.82, EDGE, CARD, radius=0.18, stroke=2).move_to([3.35, slots[k], 0])
            spot = m.Dot(radius=0.09, color=col).move_to([0.4, slots[k], 0])
            label = text(words, 20, INK).move_to([0, slots[k], 0]).align_to([0.68, 0, 0], m.LEFT)
            track = m.Line([BAR0, slots[k], 0], [BAR0 + BAR_W, slots[k], 0]).set_stroke(FAINT, width=2)
            rows.append(m.VGroup(box, spot, label, track))
            bar = m.Rectangle(width=max(BAR_W * SIMS[i], 0.03), height=0.24, stroke_width=0).set_fill(col, opacity=0.9)
            bars.append(bar.move_to([BAR0, slots[k], 0], aligned_edge=m.LEFT))
            marks.append(code("?", 22, MUTED).move_to([6.1, slots[k], 0]))
            scores.append(code(num(SIMS[i]), 22, INK).move_to([6.1, slots[k], 0]).align_to([6.45, 0, 0], m.RIGHT))
        for mob in (*bars, *scores):
            mob.set_z_index(1)
        heading = text(f"5 of the {len(SIMS)} chunks, scored against the question", 20, MUTED).move_to([3.35, 2.55, 0])
        order = sorted(range(5), key=lambda k: -SIMS[LISTED[k]])
        cut_y = (slots[2] + slots[3]) / 2
        cut = m.DashedLine([0.1, cut_y, 0], [6.6, cut_y, 0], dash_length=0.1).set_stroke(R_COL, width=3)
        cut_tag = text("top 3", 22, R_COL, weight=m.BOLD).next_to([0.1, cut_y, 0], m.LEFT, buff=0.2)
        ann = pill("millions of chunks?  approximate search keeps it fast", MUTED, CARD, 20).move_to([3.35, -3.15, 0])

        def score(k):  # introducers only: a FadeTransform's target would show from the start of its line
            return m.AnimationGroup(m.GrowFromEdge(bars[k], m.LEFT), m.FadeOut(marks[k], scale=0.5), m.FadeIn(scores[k], scale=0.5))

        self.say(
            "[But how do we measure nearest?] [With cosine similarity, from episode nine.]",
            m.AnimationGroup(
                self.new_header("THE MATH", "What “nearest” means"),
                m.FadeIn(heading),
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r, shift=0.2 * m.LEFT), m.FadeIn(mk, shift=0.2 * m.LEFT)) for r, mk in zip(rows, marks)], lag_ratio=0.15),
            ),
            m.AnimationGroup(m.FadeIn(recall, shift=0.2 * m.LEFT), m.Write(formula)),
        )
        self.say(
            "[On top, the dot product]: [multiply the two vectors number by number, and add up the results.]",
            m.AnimationGroup(m.Circumscribe(top_part, color=INK, buff=0.1), m.FadeIn(dot_label[0], shift=0.2 * m.LEFT)),
            m.AnimationGroup(
                m.FadeIn(dot_label[1], shift=0.2 * m.LEFT),
                m.GrowArrow(q_arrow), m.FadeIn(q_tag), m.FadeIn(d_arrow), m.FadeIn(d_tag), m.FadeIn(arc), m.FadeIn(th_tag), m.FadeIn(legend),
            ),
        )
        self.say(
            "[Below, the two lengths], [which divide out the size], so only the direction counts.",
            m.AnimationGroup(m.Circumscribe(bottom_part, color=INK, buff=0.1), m.FadeIn(len_label, shift=0.2 * m.LEFT), m.FadeIn(readout)),
            size.animate(run_time=2.0, rate_func=m.there_and_back).set_value(0.55),
        )
        self.say(
            "[Same direction gives one.] [At right angles, it's zero]: unrelated.",
            theta.animate(run_time=1.6).set_value(0.0),
            theta.animate(run_time=1.8).set_value(90.0),
        )
        self.say(
            "[For our toy vectors, the best refund chunk scores about zero point eight]: [an angle of about thirty-eight degrees.]",
            m.AnimationGroup(theta.animate(run_time=1.8).set_value(DEG[R1]), score(0)),
            m.FadeIn(angle_note, shift=0.15 * m.UP),
        )
        self.say(
            "[The pricing chunk that also says Enterprise scores only about zero point one]: [same word, different meaning.]",
            m.AnimationGroup(theta.animate(run_time=1.8).set_value(DEG[P1]), score(3), m.FadeOut(angle_note)),
            m.Circumscribe(rows[3], color=BLUE, buff=0.06),
        )
        self.say(
            "[Then we score every chunk the same way]: the three about refunds come out on top.",
            m.AnimationGroup(score(1), score(2), score(4)),
        )
        self.say(
            "[Rank them], [and keep the top few]: here, three. [With millions of chunks, approximate search keeps this fast.]",
            m.AnimationGroup(*[mob.animate.shift((slots[order.index(k)] - slots[k]) * m.UP) for k in (1, 2) for mob in (rows[k], bars[k], scores[k])]),
            m.AnimationGroup(m.Create(cut), m.FadeIn(cut_tag, shift=0.15 * m.RIGHT)),
            m.FadeIn(ann, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        for mob in movers:
            mob.clear_updaters()
        # the top three move on into the prompt
        self.top_rows = [rows[k] for k in order[:3]]
        self.clear(*self.top_rows)

    # ------------------------------------------------------------------ augment

    def augment(self):
        self.next_section("Augment", notes="Instructions + retrieved context + question = one prompt. Callback: Episode 31 (prompt structure).")

        def block(label, color, lines):
            head = code(label, 20, color, weight=m.BOLD)
            body = m.VGroup(*lines).arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
            return m.VGroup(head, body).arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT)

        rules = block("INSTRUCTIONS", MUTED, [code("Answer the question using only the context below."), code("Cite the chunks you used, like [1].")])
        context = block(
            "CONTEXT", A_COL, [code(f"[{n}] {CHUNKS[i][1].replace(chr(10), ' ')}", t2c={f"[{n}]": PINK}) for n, i in enumerate(TOP3, start=1)]
        )
        question = block("QUESTION", SKY, [code(QUESTION.replace("\n", " "))])
        page = m.VGroup(rules, context, question).arrange(m.DOWN, buff=0.32, aligned_edge=m.LEFT)

        window = card(10.8, page.height + 1.5, A_COL, CARD, radius=0.25, stroke=2).move_to([0, -0.45, 0])
        lights = m.VGroup(*[m.Dot(radius=0.075, color=c) for c in (BAD, A_COL, GOOD)]).arrange(m.RIGHT, buff=0.14)
        lights.next_to(window.get_corner(m.UL), m.DR, buff=0.3)
        name = code("prompt.txt", 20, MUTED).next_to(lights, m.RIGHT, buff=0.3)
        rule = m.Line(window.get_left(), window.get_right()).set_stroke(EDGE, width=2)
        rule.set_y(lights.get_y() - 0.3)
        chrome = m.VGroup(window, lights, name, rule)
        page.next_to(rule, m.DOWN, buff=0.35).align_to(window, m.LEFT).shift(0.45 * m.RIGHT)
        chrome.set_z_index(-1)  # the window opens behind the chunks flying into it
        recall = callback(31, "Prompting").move_to([0, lights.get_y(), 0]).align_to(window.get_right() + 0.3 * m.LEFT, m.RIGHT)

        self.say(
            "[Step two is augmentation.] [The retrieved chunks go into the prompt], [together with the question.]",
            self.new_header("A · AUGMENT", "Pack the chunks into the prompt"),
            m.AnimationGroup(
                m.FadeIn(chrome, scale=0.96),
                m.FadeIn(context[0]),
                *[m.AnimationGroup(m.FadeOut(r, target_position=line), m.FadeIn(line, target_position=r)) for r, line in zip(self.top_rows, context[1])],
            ),
            m.AnimationGroup(m.FadeIn(question[0]), m.FadeIn(question[1], shift=0.2 * m.UP)),
        )
        self.say(
            "[Then we add instructions on top]: [answer using only this context], [and cite your sources.]",
            m.FadeIn(rules[0]),
            m.AddTextLetterByLetter(rules[1][0], time_per_char=0.02),
            m.AddTextLetterByLetter(rules[1][1], time_per_char=0.02),
        )
        self.say(
            "[It's the prompt structure from episode thirty-one]: [instructions, context, question.] Everything the model needs is right in front of it.",
            m.FadeIn(recall, shift=0.2 * m.LEFT),
            m.Succession(*[m.Indicate(part[0], color=c, scale_factor=1.15) for part, c in ((rules, MUTED), (context, A_COL), (question, SKY))]),
        )
        self.wait(0.4)
        self.prompt = m.VGroup(chrome, page)
        self.play(m.FadeOut(recall), run_time=0.5)

    # ------------------------------------------------------------------ generate

    def generate(self):
        self.next_section("Generate", notes="The model answers from the context and cites chunks [1] and [3] (the 30-day rule and the billing portal).")
        doc = icon("prompt", 1.9)
        doc_label = m.VGroup(*[text(w, 22, A_COL, weight=m.SEMIBOLD) for w in ("augmented", "prompt")]).arrange(m.DOWN, buff=0.08)
        mini = m.VGroup(doc, doc_label).arrange(m.DOWN, buff=0.22).move_to([-5.45, 0, 0])
        llm = icon("llm", 1.9).move_to([-2.0, 0.15, 0])
        mini.shift((llm.get_y() - doc.get_y()) * m.UP)
        llm_label = text("LLM", 26, G_COL, weight=m.BOLD).next_to(llm, m.DOWN, buff=0.25)
        feed = arrow(doc.get_right(), llm.get_left(), A_COL, buff=0.2)

        cite_portal = f"[{TOP3.index(R2) + 1}]"
        cite_window = f"[{TOP3.index(R1) + 1}]"
        answer = f"Enterprise plans can be fully\nrefunded within 30 days {cite_window}.\nJust send the request through\nthe billing portal {cite_portal}."
        words = text(answer, 28, INK, t2c={cite_window: PINK, cite_portal: PINK, "30 days": GOOD})
        reply = bubble(words, G_COL, G_FILL, "left")
        reply.move_to([0, 0.4, 0]).align_to([0.35, 0, 0], m.LEFT)
        out = arrow(llm.get_right(), [reply[1].get_left()[0], llm.get_y(), 0], G_COL, buff=0.2)

        before = pill("✗  Without RAG: “90 days”, made up", BAD, BAD_FILL, 22)
        after = pill("✓  With RAG: “30 days”, cited from your docs", GOOD, GOOD_FILL, 22)
        m.VGroup(before, after).arrange(m.RIGHT, buff=0.45).move_to([0, -2.75, 0])

        self.say(
            "[The last step is generation.] [The augmented prompt] [goes to the language model], [which reads the context and writes its answer.]",
            self.new_header("G · GENERATE", "Answer from the context, with citations"),
            m.AnimationGroup(m.FadeOut(self.prompt, target_position=mini, scale=0.25), m.FadeIn(mini, target_position=self.prompt)),
            m.AnimationGroup(*[m.FadeIn(part, shift=0.2 * m.UP) for part in llm], m.FadeIn(llm_label), m.GrowArrow(feed), run_time=0.5),
            m.AnimationGroup(
                m.ShowPassingFlash(feed.copy().set_stroke(A_COL, width=8), time_width=0.6),
                m.Rotate(llm[14], angle=m.PI),
                m.Rotate(llm[15], angle=-m.PI),
                m.GrowArrow(out),
                m.FadeIn(reply[:2], shift=0.2 * m.UP),
            ),
        )
        self.say(
            "[Enterprise plans can be fully refunded within thirty days. Just send the request through the billing portal.]",
            m.AddTextLetterByLetter(words, time_per_char=0.03),
        )
        self.say(
            "[See those little numbers?] [They're citations]: they point back to the exact chunks, so anyone can check the answer.",
            m.Indicate(glyphs(words, cite_window), color=PINK, scale_factor=1.5),
            m.Indicate(glyphs(words, cite_portal), color=PINK, scale_factor=1.5),
        )
        self.say(
            "[Without RAG, we got a made-up ninety days.] [With RAG, we get thirty days], backed by a source.",
            m.FadeIn(before, shift=0.2 * m.UP),
            m.FadeIn(after, shift=0.2 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ the big picture

    def big_picture(self):
        self.next_section("The big picture", notes="Index once, offline; retrieve, augment and generate for every question.")
        steps = m.VGroup(
            node("user", "Question", "from a user", SKY, SKY_FILL),
            node("search", "Retrieve", "nearest chunks", R_COL, R_FILL),
            node("prompt", "Augment", "add context", A_COL, A_FILL),
            node("llm", "Generate", "LLM answers", G_COL, G_FILL),
            node("answer", "Answer", "with sources", GOOD, GOOD_FILL),
        )
        for x, step in zip((-5.55, -2.78, 0, 2.78, 5.55), steps):
            step.move_to([x, -0.8, 0])
        flow = m.VGroup(*[arrow(a.get_right(), b.get_left()) for a, b in zip(steps, steps[1:])])
        letters = m.VGroup(
            *[
                text(c, 40, col, weight=m.BOLD).next_to(steps[i], m.DOWN, buff=0.3)
                for i, (c, col) in enumerate(zip("RAG", (R_COL, A_COL, G_COL)), start=1)
            ]
        )
        online = text("ONLINE  ·  for every question", 22, MUTED, weight=m.SEMIBOLD).move_to([0, -3.45, 0])
        docs = compact_node("doc", "Your docs", EDGE, CARD).move_to([-5.55, 1.75, 0])
        store = compact_node("database", "Vector DB", R_COL, R_FILL).move_to([-2.78, 1.75, 0])
        index = arrow(docs.get_right(), store.get_left())
        lookup = arrow(store.get_bottom(), steps[1].get_top(), R_COL)
        offline = text("OFFLINE  ·  chunk, embed and store your docs, once", 22, MUTED, weight=m.SEMIBOLD)
        offline.move_to([0, 1.75, 0]).align_to([-1.2, 0, 0], m.LEFT)

        def rise(mob):
            return m.FadeIn(mob, shift=0.3 * m.UP)

        pulses = []
        for i, (a, col) in enumerate(zip(flow, (SKY, R_COL, A_COL, G_COL))):
            flash = [pulse(a, col)]
            if i == 0:
                flash.append(pulse(lookup, R_COL))
            pulses.append(m.AnimationGroup(*flash, steps[i + 1].animate(rate_func=m.there_and_back).scale(1.07), run_time=0.7))

        self.say(
            "[Let's put it all together.] [Ahead of time, your documents are chunked, embedded, and stored in a vector database.]",
            self.new_header("THE BIG PICTURE", "RAG, end to end"),
            m.LaggedStart(m.FadeIn(docs, shift=0.2 * m.RIGHT), m.GrowArrow(index), m.FadeIn(store, shift=0.2 * m.RIGHT), m.FadeIn(offline), lag_ratio=0.3),
        )
        self.say(
            "Then, for every [question], we [retrieve] the nearest chunks, [augment] the prompt with them, and [generate] [a grounded answer.]",
            m.AnimationGroup(rise(steps[0]), m.FadeIn(online)),
            m.AnimationGroup(m.GrowArrow(flow[0]), m.GrowArrow(lookup), rise(steps[1]), rise(letters[0])),
            m.AnimationGroup(m.GrowArrow(flow[1]), rise(steps[2]), rise(letters[1])),
            m.AnimationGroup(m.GrowArrow(flow[2]), rise(steps[3]), rise(letters[2])),
            m.AnimationGroup(m.GrowArrow(flow[3]), rise(steps[4])),
        )
        self.say("[Retrieve. Augment. Generate.] That's the whole loop.", m.Succession(*pulses, m.Circumscribe(steps[4], color=GOOD, buff=0.1)))
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ why, and the catch

    def why(self):
        self.next_section("Why RAG, and the catch", notes="Fresh, private, verifiable; only as good as its retrieval.")
        specs = [
            ("clock", "Always up to date", "Update the docs, not the\nmodel. No retraining.", SKY),
            ("lock", "Private data, too", "Knowledge stays in your\nown database.", AMBER),
            ("doc", "Verifiable", "Every answer cites the\nchunks it came from.", PINK),
            ("alert", "The catch", "Miss the right chunk, and\nthe model guesses again.", BAD),
        ]
        tiles = m.VGroup()
        for name, title, desc, col in specs:
            t = tile(name, title, desc, col)
            if name == "doc":
                t[1][0][2].set_fill(PINK, opacity=1)
            tiles.add(t)
        tiles.arrange_in_grid(rows=2, cols=2, buff=(0.45, 0.4)).move_to(0.3 * m.DOWN)
        advice = pill("✓  test the retrieval: is the right chunk in the top 3?", GOOD, GOOD_FILL, 22).move_to([0, -3.3, 0])

        self.say(
            "[So why do so many teams use RAG?] [It stays up to date]: when a policy changes, you update the documents, not the model.",
            self.new_header("WHY RAG?", "What you get, and the catch"),
            m.FadeIn(tiles[0], shift=0.3 * m.UP),
        )
        self.say(
            "[It works with private data] the model never saw, [and every answer can point to its source.]",
            m.FadeIn(tiles[1], shift=0.3 * m.UP),
            m.FadeIn(tiles[2], shift=0.3 * m.UP),
        )
        self.say(
            "[The catch: the answer is only as good as the retrieval.] If the right chunk isn't found, the model is back to guessing.",
            m.FadeIn(tiles[3], shift=0.3 * m.UP),
        )
        self.say("[So test the retrieval too]: is the right chunk in the top three?", m.FadeIn(advice, shift=0.2 * m.UP))
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Retrieve, augment, generate.")
        halo = glow(TEAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Retrieve", "Augment", "Generate"), (R_COL, A_COL, G_COL))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Look it up.  Put it in the prompt.  Answer from it.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [retrieve], [augment], [generate.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Look it up, put it in the prompt, and answer from it.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
