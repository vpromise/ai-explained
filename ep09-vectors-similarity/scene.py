"""AI, Explained · Episode 9 · Vectors & Similarity

Running example: three words, "king", "queen" and "apple", as toy vectors. In two dimensions
(pretend axes "royalty" and "food"): king [5, 1], queen [4, 2], apple [1, 5]. Then a third
number ("is it a person?"): king [5, 1, 4], queen [4, 2, 4], apple [1, 5, 0]. The numbers are
made up for teaching; real embeddings are learned and have hundreds or thousands of numbers.

The film builds a·b = ‖a‖‖b‖cos θ piece by piece: the dot product, the length (norm), the
catch (a stretched apple's dot product with king triples and beats queen's), cosine
similarity (direction only), how to read a cosine from 1 through 0 to −1, a third dimension,
and ranking by cosine as search engines and RAG do. Every number the narration quotes is
computed below with numpy and asserted, so the scene cannot drift from its words.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- colours

LIME_FILL = "#1E2A10"
WORDS = {"king": (SKY, SKY_FILL), "queen": (PINK, PINK_FILL), "apple": (LIME, LIME_FILL)}
DOT_C, LEN_C, COS_C = VIOLET, TEAL, AMBER  # dot product · length (norm) · cosine and angles

# ---------------------------------------------------------------------------- the data (toys)

KING = np.array([5.0, 1.0])
QUEEN = np.array([4.0, 2.0])
APPLE = np.array([1.0, 5.0])
STRETCH = 3.0
LONG = STRETCH * APPLE  # the stretched apple: same direction, three times as long
VEC = {"king": KING, "queen": QUEEN, "apple": APPLE}


def length(v):
    return float(np.linalg.norm(v))


def cosine(a, b):
    return float(np.dot(a, b) / (length(a) * length(b)))


def degrees(a, b):
    return float(np.degrees(np.arccos(np.clip(cosine(a, b), -1.0, 1.0))))


def heading(v):
    return float(np.arctan2(v[1], v[0]))


DOT_KQ, DOT_KA, DOT_KL = (float(np.dot(KING, v)) for v in (QUEEN, APPLE, LONG))
N_K, N_Q, N_A, N_L = (length(v) for v in (KING, QUEEN, APPLE, LONG))
COS_KQ, COS_KA, COS_KL = (cosine(KING, v) for v in (QUEEN, APPLE, LONG))
DEG_KQ, DEG_KA, DEG_KL = (degrees(KING, v) for v in (QUEEN, APPLE, LONG))

assert (DOT_KQ, DOT_KA, DOT_KL) == (22.0, 10.0, 30.0)  # "twenty-two", "only ten", "from ten to thirty"
assert DOT_KL == STRETCH * DOT_KA and DOT_KL > DOT_KQ > DOT_KA  # "tripled", "that beats queen's twenty-two"
assert np.isclose(N_K**2, 26) and round(N_K, 1) == 5.1  # "the square root of twenty-six: about five point one"
assert round(N_Q, 1) == 4.5 and np.isclose(N_A, N_K)  # "about four point five", "apple's, like king's"
assert round(COS_KQ, 2) == 0.96 and round(COS_KA, 2) == 0.38  # "zero point nine six", "zero point three eight"
assert np.isclose(COS_KL, COS_KA)  # "the stretched apple? Still zero point three eight."
assert round(DEG_KQ) == 15 and round(DEG_KA) == 67 and np.isclose(DEG_KL, DEG_KA)  # "fifteen", "sixty-seven degrees"

KING3 = np.array([5.0, 1.0, 4.0])
QUEEN3 = np.array([4.0, 2.0, 4.0])
APPLE3 = np.array([1.0, 5.0, 0.0])
VEC3 = {"king": KING3, "queen": QUEEN3, "apple": APPLE3}
COS3_KQ, COS3_KA = cosine(KING3, QUEEN3), cosine(KING3, APPLE3)
assert np.array_equal(KING3[:2], KING) and np.array_equal(QUEEN3[:2], QUEEN) and np.array_equal(APPLE3[:2], APPLE)
assert KING3[2] == QUEEN3[2] == 4 and APPLE3[2] == 0  # "king and queen both get a four, and apple gets a zero"
assert round(COS3_KQ, 2) == 0.98 and round(COS3_KA, 2) == 0.30  # "zero point nine eight", "zero point three"
assert COS3_KQ > COS_KQ and COS3_KA < COS_KA  # "even closer", "drops"

CANDIDATES = [("queen", QUEEN), ("apple", APPLE), ("apple ×3", LONG)]
BY_DOT = sorted(CANDIDATES, key=lambda c: -float(np.dot(KING, c[1])))
BY_COS = sorted(CANDIDATES, key=lambda c: -round(cosine(KING, c[1]), 9))  # the two apples tie; stable order
assert [n for n, _ in BY_DOT] == ["apple ×3", "queen", "apple"]  # "the stretched apple wins"
assert [n for n, _ in BY_COS] == ["queen", "apple", "apple ×3"]  # "queen comes first, with both apples tied"

EMBED = np.clip(np.random.default_rng(9).normal(0.0, 1.0, 300), -2.2, 2.2)  # illustrative values only
assert len(EMBED) == 300  # "three hundred numbers"; word2vec's Google News vectors have 300 per word


# ---------------------------------------------------------------------------- helpers


class Mono(m.Text):
    """Digits for DecimalNumber in the series' monospace font."""

    def __init__(self, s, **kw):
        kw.setdefault("font", MONO)
        super().__init__(s, **kw)


def mix(a, b, t):
    """A colour between two hex colours."""
    ca, cb = (np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) for c in (a, b))
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in ca + (cb - ca) * float(t))


def nums(v):
    return ", ".join(f"{x:g}" for x in v)


def pieces(parts, size=28, font=SANS, weight=m.NORMAL):
    """One Text from (string, colour) pieces, and the glyphs of each piece (spaces draw none).
    Strings here avoid ligature pairs (fi, fl, ->), so there is one glyph per visible character."""
    s = "".join(p for p, _ in parts)
    words = m.Text(s, font=font, font_size=size, color=INK, weight=weight)
    assert len(words) == sum(not ch.isspace() for ch in s), s
    groups, at = [], 0
    for p, color in parts:
        start = sum(not ch.isspace() for ch in s[:at])
        n = sum(not ch.isspace() for ch in p)
        group = words[start : start + n]
        if color:
            group.set_color(color)
        groups.append(group)
        at += len(p)
    return words, groups


def vector_row(name, v, size=30):
    """`king  = [5, 1]` in the word's colour, and the glyphs of each number."""
    color = WORDS[name][0]
    parts = [(f"{name:<5} = [", color)]
    for i, x in enumerate(v):
        parts += [(f"{x:g}", color), (", " if i < len(v) - 1 else "]", color)]
    words, groups = pieces(parts, size, MONO)
    return words, groups[1::2]


def word_tag(name, size=26):
    return text(name, size, WORDS[name][0], weight=m.BOLD)


def trace(start, end, color, width=10):
    """A light pulse along an arrow."""
    return m.ShowPassingFlash(m.Line(start, end).set_stroke(color, width=width), time_width=0.6)


def sit_on(mob, words, glyph=0):
    """Shift mob (digits) so its bottom sits on the baseline of a glyph of `words`."""
    return mob.shift((words[glyph].get_bottom()[1] - mob.get_bottom()[1]) * m.UP)


class Plane:
    """The toy meaning plane: royalty → right, food → up, equal units on both axes (so the
    angles on screen are the true angles)."""

    def __init__(self, origin, unit, xmax, ymax, step, xticks, yticks):
        self.o = np.array([origin[0], origin[1], 0.0])
        self.u = unit
        self.ax = axes([0, xmax, step], [0, ymax, step], xmax * unit, ymax * unit)
        self.ax.shift(self.o - self.ax.c2p(0, 0))
        self.ticks = m.VGroup(
            *[text(f"{v:g}", 20, MUTED).next_to(self.p((v, 0)), m.DOWN, buff=0.16) for v in xticks],
            *[text(f"{v:g}", 20, MUTED).next_to(self.p((0, v)), m.LEFT, buff=0.16) for v in yticks],
        )
        self.xl = text("royalty", 22, MUTED).next_to(self.ax.x_axis.get_end(), m.RIGHT, buff=0.15)
        self.yl = text("food", 22, MUTED).next_to(self.ax.y_axis.get_end(), m.RIGHT, buff=0.2)

    def p(self, v):
        return self.o + self.u * np.array([v[0], v[1], 0.0])

    def arrow(self, v, color, width=6):
        return m.Arrow(self.o, self.p(v), buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.2)

    def arc(self, radius, a, b, color=COS_C, width=4):
        return m.Arc(radius=radius, start_angle=heading(a), angle=heading(b) - heading(a), arc_center=self.o).set_stroke(color, width=width)


# 3-D: an orthographic view, turned about the vertical "person" axis, seen from a little above
O3, S3, TILT = np.array([-3.6, -1.85, 0.0]), 0.6, np.radians(20)
AXES3 = [((6.5, 0, 0), "royalty", m.RIGHT), ((0, 6.5, 0), "food", m.LEFT), ((0, 0, 6.0), "person", m.UP)]
TAGS3 = {"king": m.RIGHT, "queen": m.UP, "apple": m.UP}


def project(v, phi):
    x, y, z = v
    f = np.radians(phi)
    across, deep = x * np.cos(f) - y * np.sin(f), x * np.sin(f) + y * np.cos(f)
    return O3 + S3 * np.array([across, z * np.cos(TILT) + deep * np.sin(TILT), 0.0])


# ---------------------------------------------------------------------------- the episode


class Ep09VectorsSimilarity(Episode):
    number = 9

    def lesson(self):
        self.hook()
        self.vectors()
        self.dot()
        self.lengths()
        self.catch()
        self.cosine()
        self.reading()
        self.more_dims()
        self.search()
        self.recap()

    def stop(self, *mobs):
        for mob in mobs:
            mob.clear_updaters()

    def others(self, *keep):
        """Fade out everything on stage but `keep` (and the header): call before new_header()."""
        gone = [mob for mob in self.mobjects[1:] if mob not in keep and mob is not self.head]
        return m.AnimationGroup(*[m.FadeOut(mob) for mob in gone]) if gone else m.Wait(0.01)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="King, queen, apple: which words are alike? Words become lists of numbers (vectors). The teaser formula is explained in the cosine section.")
        spots = {"queen": -3.0, "king": 0.0, "apple": 5.0}
        chips = {w: chip(w, *WORDS[w], width=2.0, height=0.8, size=32).move_to([x, 1.55, 0]) for w, x in spots.items()}
        close = m.Line(chips["queen"].get_right(), chips["king"].get_left(), buff=0.12).set_stroke(GOOD, width=4)
        close_t = text("close", 22, GOOD, weight=m.SEMIBOLD).next_to(close, m.UP, buff=0.2)
        far = m.DashedLine(chips["king"].get_right(), chips["apple"].get_left(), buff=0.12, dash_length=0.12).set_stroke(MUTED, width=3)
        far_t = text("far", 22, MUTED, weight=m.SEMIBOLD).next_to(far, m.UP, buff=0.2)
        lists = {w: code(f"[{nums(VEC[w])}]", 30, WORDS[w][0]).next_to(chips[w], m.DOWN, buff=0.35) for w in spots}
        caption = text("A computer only works with numbers.", 26, MUTED).move_to([0, -0.55, 0])
        named = pill("a list of numbers = a vector", AMBER, AMBER_FILL, 24).move_to([0, -0.55, 0])
        formula = tex(r"\mathbf{a} \cdot \mathbf{b} = \left\| \mathbf{a} \right\| \, \left\| \mathbf{b} \right\| \cos\theta", 62, INK)
        formula.move_to([0, -2.3, 0])
        halo = glow(AMBER, radius=2.6).move_to(formula)

        self.say(
            "So... how can a computer tell that [king] is closer in meaning to [queen] [than to apple?]",
            m.FadeIn(chips["king"], shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(chips["queen"], shift=0.3 * m.UP), m.Create(close), m.FadeIn(close_t)),
            m.AnimationGroup(m.FadeIn(chips["apple"], shift=0.3 * m.UP), m.Create(far), m.FadeIn(far_t)),
        )
        self.say(
            "[Computers only understand numbers.] [So we turn each word into a list of numbers]: [a vector.]",
            m.FadeIn(caption, shift=0.2 * m.UP),
            m.LaggedStart(*[m.FadeIn(lists[w], shift=0.2 * m.DOWN) for w in spots], lag_ratio=0.3),
            m.FadeTransform(caption, named),
        )
        self.say(
            "By the end of this episode, [this little formula] will tell us which words are alike.",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.Write(formula, run_time=1.8)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ words as arrows

    def vectors(self):
        self.next_section(
            "Words as arrows",
            notes="Toy 2-D vectors, made up for teaching: king [5, 1], queen [4, 2], apple [1, 5]; 'royalty' and 'food' are pretend axis names. "
            "Real embeddings: hundreds to thousands of learned numbers (Episode 22).",
        )
        P = Plane((-6.05, -2.95), 0.8, 6, 6, 1, range(1, 6), range(1, 6))
        self.A = P
        self.arrows = {w: P.arrow(VEC[w], WORDS[w][0]) for w in WORDS}
        self.tags = {
            "king": word_tag("king").next_to(P.p(KING), m.RIGHT, buff=0.15),
            "queen": word_tag("queen").next_to(P.p(QUEEN), m.UR, buff=0.06),
            "apple": word_tag("apple").next_to(P.p(APPLE), m.RIGHT, buff=0.15),
        }
        rows, self.row_nums = {}, {}
        for w in WORDS:
            rows[w], self.row_nums[w] = vector_row(w, VEC[w])
        self.rows = rows
        m.VGroup(*rows.values()).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([3.4, 1.4, 0])
        right = m.DashedLine(P.o, P.p((KING[0], 0)), dash_length=0.12).set_stroke(SKY, width=3.5)
        up = m.DashedLine(P.p((KING[0], 0)), P.p(KING), dash_length=0.08).set_stroke(SKY, width=3.5)
        arc_kq = P.arc(2.0, KING, QUEEN)
        arc_ka = P.arc(1.1, KING, APPLE)
        idea = pill("similar meanings → similar directions", AMBER, AMBER_FILL, 24).move_to([3.4, -0.45, 0])
        toy = pill("toy numbers · made up for this example", MUTED, CARD, 20).move_to([3.4, -1.35, 0])
        box = card(6.0, 1.35, SKY, CARD, stroke=2).move_to([3.4, -2.65, 0])
        title = text("Real embeddings", 26, INK, weight=m.BOLD)
        ep22 = pill("Episode 22", SKY, SKY_FILL, 20)
        desc = text("hundreds or thousands of learned numbers", 22, MUTED)
        body = m.VGroup(m.VGroup(title, ep22).arrange(m.RIGHT, buff=0.3), desc).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        body.move_to(box)

        self.say(
            "[Let's give each word just two numbers]: [how royal it is], [and how much it's a food.]",
            m.AnimationGroup(self.new_header("VECTORS", "Words as arrows"), m.Create(P.ax), m.FadeIn(P.ticks)),
            m.AnimationGroup(m.FadeIn(P.xl, shift=0.2 * m.LEFT), trace(P.o, P.p((6, 0)), AMBER)),
            m.AnimationGroup(m.FadeIn(P.yl, shift=0.2 * m.DOWN), trace(P.o, P.p((0, 6)), AMBER)),
        )
        self.say(
            "[King gets five and one.] [Five steps right, one step up], [and we draw an arrow to that point.]",
            m.FadeIn(rows["king"], shift=0.2 * m.LEFT),
            m.Succession(m.Create(right, run_time=0.9), m.Create(up, run_time=0.5)),
            m.AnimationGroup(m.GrowArrow(self.arrows["king"]), m.FadeIn(self.tags["king"])),
        )
        self.say(
            "[Queen gets four and two], [and apple gets one and five.]",
            m.AnimationGroup(
                m.FadeOut(right), m.FadeOut(up), m.FadeIn(rows["queen"], shift=0.2 * m.LEFT),
                m.GrowArrow(self.arrows["queen"]), m.FadeIn(self.tags["queen"]),
            ),
            m.AnimationGroup(m.FadeIn(rows["apple"], shift=0.2 * m.LEFT), m.GrowArrow(self.arrows["apple"]), m.FadeIn(self.tags["apple"])),
        )
        self.say(
            "[King and queen point almost the same way], [while apple points somewhere else.] [Similar meanings, similar directions.]",
            m.Create(arc_kq),
            m.Create(arc_ka),
            m.FadeIn(idea, shift=0.2 * m.UP),
        )
        self.say("[A warning: these numbers are toys, made up for this example.]", m.FadeIn(toy, shift=0.2 * m.UP))
        self.say(
            "[Real embeddings have hundreds or thousands of numbers, learned from data.] [Episode twenty-two shows how.]",
            m.FadeIn(m.VGroup(box, title, desc), shift=0.2 * m.UP),
            m.FadeIn(ep22, scale=1.2),
        )
        self.wait(0.3)
        self.play(*[m.FadeOut(x) for x in (arc_kq, arc_ka, idea, toy, box, title, desc, ep22)], run_time=0.7)

    # ------------------------------------------------------------------ the dot product

    def product_row(self, other, y):
        """king · other = 5 × 4 + 1 × 2 = 22: the label, the four numbers, the signs, the result."""
        a, b = KING, VEC[other]
        ca, cb = SKY, WORDS[other][0]
        parts = [
            ("king", ca), (" · ", None), (other, cb), (" = ", None),
            (f"{a[0]:g}", ca), (" × ", None), (f"{b[0]:g}", cb), (" + ", None),
            (f"{a[1]:g}", ca), (" × ", None), (f"{b[1]:g}", cb), (" = ", None), (f"{np.dot(a, b):g}", DOT_C),
        ]
        words, g = pieces(parts, 30)
        words.move_to([3.4, y, 0])
        return m.VGroup(*g[0:4]), [g[4], g[6], g[8], g[10]], m.VGroup(g[5], g[7], g[9]), m.VGroup(g[11], g[12])

    def pairs(self, other, parts):
        """The numbers fly from the vectors into the sum: first pair, then second pair."""
        _, numbers, signs, _ = parts
        k, o = self.row_nums["king"], self.row_nums[other]
        return m.LaggedStart(
            m.AnimationGroup(m.TransformFromCopy(k[0], numbers[0]), m.TransformFromCopy(o[0], numbers[1]), m.FadeIn(signs[0])),
            m.AnimationGroup(m.FadeIn(signs[1]), m.TransformFromCopy(k[1], numbers[2]), m.TransformFromCopy(o[1], numbers[3]), m.FadeIn(signs[2])),
            lag_ratio=0.55,
            run_time=1.8,
        )

    def dot(self):
        self.next_section("The dot product", notes=f"king · queen = 5×4 + 1×2 = {DOT_KQ:g}; king · apple = 5×1 + 1×5 = {DOT_KA:g}.")
        P = self.A
        formula = m.MathTex(r"\mathbf{a} \cdot \mathbf{b}", "=", r"a_1 b_1", "+", r"a_2 b_2", font_size=46, color=INK).move_to([3.4, -0.15, 0])
        formula[0].set_color(DOT_C)
        legend = text("multiply the matching numbers, then add", 22, MUTED).next_to(formula, m.DOWN, buff=0.3)
        kq = self.product_row("queen", -1.6)
        ka = self.product_row("apple", -2.3)
        verdict = pill("more agreement → a bigger dot product", DOT_C, VIOLET_FILL, 22).move_to([3.4, -3.25, 0])

        self.say(
            "[To measure how much two arrows agree, start with the dot product.]",
            self.new_header("STEP 1 · DOT PRODUCT", "Multiply, then add"),
        )
        self.say(
            "[Multiply the first numbers together, then the second numbers], [and add the results.]",
            m.Write(formula, run_time=1.8),
            m.FadeIn(legend, shift=0.15 * m.UP),
        )
        self.say(
            "[King and queen]: [five times four, plus one times two], [makes twenty-two.]",
            m.AnimationGroup(m.FadeIn(kq[0]), trace(P.o, P.p(KING), SKY), trace(P.o, P.p(QUEEN), PINK)),
            self.pairs("queen", kq),
            m.FadeIn(kq[3], scale=1.3),
        )
        self.say(
            "[King and apple]: [five times one, plus one times five], [makes only ten.]",
            m.AnimationGroup(m.FadeIn(ka[0]), trace(P.o, P.p(KING), SKY), trace(P.o, P.p(APPLE), LIME)),
            self.pairs("apple", ka),
            m.FadeIn(ka[3], scale=1.3),
        )
        self.say(
            "[The more two arrows point the same way, the bigger their dot product.]",
            m.AnimationGroup(m.FadeIn(verdict, shift=0.2 * m.UP), m.Indicate(kq[3][1], color=DOT_C, scale_factor=1.3)),
        )
        self.wait(0.3)
        keep = [P.ax, P.ticks, P.xl, P.yl, *self.arrows.values(), *self.tags.values(), *self.rows.values()]
        self.play(self.others(*keep), run_time=0.7)

    # ------------------------------------------------------------------ length

    def norm_row(self, name):
        v = VEC[name]
        parts = [
            (f"‖{name}‖", WORDS[name][0]),
            (f" = √({v[0]:g}² + {v[1]:g}²) = √{v[0] ** 2 + v[1] ** 2:g} ≈ ", None),
            (f"{length(v):.2f}", LEN_C),
        ]
        return pieces(parts, 26)

    def lengths(self):
        self.next_section("Length: the norm", notes=f"Pythagoras: ‖king‖ = √26 ≈ {N_K:.2f}, ‖queen‖ = √20 ≈ {N_Q:.2f}, ‖apple‖ = √26 ≈ {N_A:.2f}.")
        P = self.A
        foot = P.p((KING[0], 0))
        leg_x = m.Line(P.o, foot).set_stroke(LEN_C, width=6)
        leg_y = m.Line(foot, P.p(KING)).set_stroke(LEN_C, width=6)
        lab_x = text(f"{KING[0]:g}", 24, LEN_C, weight=m.BOLD).next_to(leg_x, m.DOWN, buff=0.18)
        lab_y = text(f"{KING[1]:g}", 24, LEN_C, weight=m.BOLD).next_to(leg_y, m.RIGHT, buff=0.15).shift(0.12 * m.DOWN)
        formula = m.MathTex(r"\left\| \mathbf{a} \right\|", "=", r"\sqrt{a_1^2 + a_2^2}", font_size=46, color=INK)
        formula[0].set_color(LEN_C)
        name = pill("the norm", LEN_C, TEAL_FILL, 20)
        top = m.VGroup(formula, name).arrange(m.RIGHT, buff=0.45).move_to([3.4, -0.15, 0])
        legend = text("square each number, add them, take the square root", 22, MUTED).next_to(top, m.DOWN, buff=0.3)
        rows = {w: self.norm_row(w) for w in WORDS}
        m.VGroup(*[rows[w][0] for w in WORDS]).arrange(m.DOWN, buff=0.24, aligned_edge=m.LEFT).move_to([3.4, -2.3, 0])
        king_row = rows["king"][1]

        self.say(
            "[Next: the length of an arrow, also called its norm.]",
            m.AnimationGroup(self.new_header("STEP 2 · LENGTH", "How long is an arrow?"), m.FadeOut(P.ticks)),
        )
        self.say(
            "[It comes from Pythagoras]: [square each number, add them up, and take the square root.]",
            m.AnimationGroup(m.Create(leg_x), m.Create(leg_y), m.FadeIn(lab_x), m.FadeIn(lab_y)),
            m.AnimationGroup(m.Write(formula, run_time=1.8), m.FadeIn(name, scale=1.2), m.FadeIn(legend, shift=0.15 * m.UP)),
        )
        self.say(
            "[King's length is the square root of twenty-six]: [about five point one.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(king_row[0], king_row[1])), trace(P.o, P.p(KING), LEN_C)),
            m.FadeIn(king_row[2], scale=1.3),
        )
        self.say(
            "[Queen's is about four point five], [and apple's, like king's, about five point one.]",
            m.FadeIn(m.VGroup(*rows["queen"][1]), shift=0.15 * m.UP),
            m.FadeIn(m.VGroup(*rows["apple"][1]), shift=0.15 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the catch

    def catch(self):
        self.next_section(
            "The catch",
            notes=f"Apple stretched ×{STRETCH:g} = [3, 15]: same direction, king · apple goes {DOT_KA:g} → {DOT_KL:g}, beating queen's {DOT_KQ:g}.",
        )
        A = self.A
        B = Plane((-6.05, -3.2), 0.34, 12, 16, 5, (5, 10), (5, 10))
        self.B = B
        arrows = {w: B.arrow(VEC[w], WORDS[w][0], width=5).set_z_index(1) for w in WORDS}
        tags = {
            "king": word_tag("king", 24).next_to(B.p(KING), m.RIGHT, buff=0.12),
            "queen": word_tag("queen", 24).next_to(B.p(QUEEN), m.UR, buff=0.04),
            "apple": word_tag("apple", 24).next_to(B.p(APPLE), m.RIGHT, buff=0.2).shift(0.12 * m.DOWN),
        }
        self.arrowsB, self.tagsB = arrows, tags
        s = m.ValueTracker(1.0)
        dim = mix(LIME, BG, 0.45)
        self.long = m.always_redraw(lambda: B.arrow(s.get_value() * APPLE, dim, width=5))
        self.long_tag = text("apple ×3", 24, LIME, weight=m.BOLD).next_to(B.p(LONG), m.RIGHT, buff=0.15)
        zoom = m.AnimationGroup(
            m.ReplacementTransform(A.ax, B.ax), m.ReplacementTransform(A.xl, B.xl), m.ReplacementTransform(A.yl, B.yl),
            *[m.ReplacementTransform(self.arrows[w], arrows[w]) for w in WORDS],
            *[m.ReplacementTransform(self.tags[w], tags[w]) for w in WORDS],
            m.FadeIn(B.ticks),
            run_time=1.8,
        )

        row_q, gq = pieces([("king", SKY), (" · ", None), ("queen", PINK), (" = ", None), (f"{DOT_KQ:g}", DOT_C)], 32)
        label_a, _ = pieces([("king", SKY), (" · ", None), ("apple", LIME), (" = ", None)], 32)
        value = m.DecimalNumber(DOT_KA, num_decimal_places=0, mob_class=Mono, font_size=36, color=DOT_C)
        value.add_updater(lambda d: d.set_value(float(np.dot(KING, s.get_value() * APPLE))))
        value.next_to(label_a, m.RIGHT, buff=0.2)
        sit_on(value, label_a)
        row_a = m.VGroup(label_a, value)
        m.VGroup(row_q, row_a).arrange(m.DOWN, buff=0.4, aligned_edge=m.LEFT).move_to([3.0, -0.95, 0])
        beat = pill(f"{DOT_KL:g} > {DOT_KQ:g}: by dot product, the long apple wins", BAD, BAD_FILL, 22).move_to([3.4, -2.35, 0])
        mixes = m.VGroup(
            text("dot product =", 24, INK, weight=m.SEMIBOLD),
            pill("direction", GOOD, GOOD_FILL, 22),
            text("×", 28, MUTED),
            pill("length", BAD, BAD_FILL, 22),
        ).arrange(m.RIGHT, buff=0.25).move_to([3.4, -3.25, 0])

        gone = self.others(A.ax, A.xl, A.yl, *self.arrows.values(), *self.tags.values(), *self.rows.values())
        self.say(
            "[Now, here's the catch.] [Let's zoom out], [and stretch apple's arrow to three times its length.]",
            m.AnimationGroup(gone, self.new_header("THE CATCH", "Longer arrows, bigger dot products", BAD, BAD_FILL), m.FadeIn(row_q), m.FadeIn(row_a)),
            zoom,
            m.AnimationGroup(m.FadeIn(self.long), s.animate(run_time=2.4).set_value(STRETCH)),
        )
        value.clear_updaters()
        self.say(
            "[It points the same way, so it means the same thing.] [But its dot product with king just tripled, from ten to thirty.]",
            m.AnimationGroup(m.FadeIn(self.long_tag, shift=0.15 * m.RIGHT), trace(B.o, B.p(LONG), LIME)),
            m.Circumscribe(row_a, color=DOT_C, buff=0.12),
        )
        self.say(
            "[That beats queen's twenty-two.] [By this measure, apple now looks more like king than queen does.]",
            m.AnimationGroup(m.Indicate(gq[4], color=PINK, scale_factor=1.3), m.FadeIn(beat, shift=0.2 * m.UP)),
            m.Wiggle(beat, scale_value=1.05),
        )
        self.say(
            "[The dot product mixes up two things]: [direction], [and length.]",
            m.FadeIn(mixes[0], shift=0.2 * m.UP),
            m.FadeIn(mixes[1], shift=0.2 * m.UP),
            m.FadeIn(mixes[2:], shift=0.2 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ cosine similarity

    def cos_row(self, name, v, color):
        """queen    22 ÷ (5.10×4.47) ≈ 0.96  15°, with the '≈' in one column."""
        nb = f"{length(v):.2f}"
        parts = [
            (f"{name:<9}", color), (f"{np.dot(KING, v):>2.0f}", DOT_C), (" ÷ (", None), (f"{N_K:.2f}", LEN_C), ("×", None),
            (nb, LEN_C), (")" + " " * (5 - len(nb)) + " ≈ ", None), (f"{cosine(KING, v):.2f}", COS_C), (f"  {degrees(KING, v):.0f}°", MUTED),
        ]
        return pieces(parts, 22, MONO)[0]

    def cosine(self):
        self.next_section(
            "Cosine similarity",
            notes=f"a·b = ‖a‖‖b‖cos θ, so cos θ = a·b / (‖a‖‖b‖). With king: queen {COS_KQ:.3f} ({DEG_KQ:.1f}°), apple {COS_KA:.3f} ({DEG_KA:.1f}°), "
            f"stretched apple {COS_KL:.3f}.",
        )
        B = self.B
        f1 = m.MathTex(
            r"\mathbf{a} \cdot \mathbf{b}", "=", r"\left\| \mathbf{a} \right\|", r"\left\| \mathbf{b} \right\|", r"\cos\theta", font_size=50, color=INK
        ).move_to([1.9, 1.75, 0])
        f1[0].set_color(DOT_C)
        f1[2].set_color(LEN_C)
        f1[3].set_color(LEN_C)
        f1[4].set_color(COS_C)
        under = m.VGroup(
            text("dot product", 20, DOT_C, weight=m.SEMIBOLD).next_to(f1[0], m.DOWN, buff=0.3),
            text("lengths", 20, LEN_C, weight=m.SEMIBOLD).next_to(m.VGroup(f1[2], f1[3]), m.DOWN, buff=0.3),
            text("angle's cosine", 20, COS_C, weight=m.SEMIBOLD).next_to(f1[4], m.DOWN, buff=0.3),
        )
        for label in under[1:]:
            label.align_to(under[0], m.UP)
        for a, b in zip(under, under[1:]):  # keep the labels apart
            if b.get_left()[0] < a.get_right()[0] + 0.35:
                b.shift((a.get_right()[0] + 0.35 - b.get_left()[0]) * m.RIGHT)
        f2 = m.MathTex(
            r"\cos\theta", "=", r"\frac{", r"\mathbf{a} \cdot \mathbf{b}", "}{", r"\left\| \mathbf{a} \right\|", r"\left\| \mathbf{b} \right\|", "}",
            font_size=50, color=INK,
        ).move_to([1.9, 1.6, 0])
        f2[0].set_color(COS_C)
        f2[3].set_color(DOT_C)
        f2[5].set_color(LEN_C)
        f2[6].set_color(LEN_C)
        name = pill("cosine similarity", COS_C, AMBER_FILL, 24).move_to([5.25, 1.6, 0])
        title = text("compared with king", 20, MUTED).move_to([0, 0.2, 0])
        rows = m.VGroup(self.cos_row("queen", QUEEN, PINK), self.cos_row("apple", APPLE, LIME), self.cos_row("apple ×3", LONG, LIME))
        rows.arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([2.6, -1.0, 0])
        title.align_to(rows, m.LEFT)
        arc_kq = B.arc(1.15, KING, QUEEN)
        arc_ka = B.arc(0.7, KING, APPLE)
        only = pill("cosine sees direction only, not length", GOOD, GOOD_FILL, 22).move_to([2.6, -2.95, 0])

        keep = [B.ax, B.xl, B.yl, B.ticks, *self.arrowsB.values(), *self.tagsB.values(), self.long, self.long_tag]
        gone = self.others(*keep)
        self.say(
            "[Here's the fix.] [The dot product equals the two lengths multiplied], times the cosine of the angle between the arrows.",
            m.AnimationGroup(gone, self.new_header("STEP 3 · COSINE SIMILARITY", "Direction only")),
            m.AnimationGroup(m.Write(f1, run_time=2.6), m.LaggedStart(*[m.FadeIn(u, shift=0.15 * m.UP) for u in under], lag_ratio=0.7, run_time=3.6)),
        )
        self.say(
            "[Divide both sides by the lengths], [and the cosine stands on its own]: [that's cosine similarity.]",
            m.AnimationGroup(m.FadeOut(under), m.TransformMatchingTex(f1, f2, path_arc=-0.8), run_time=1.6),
            m.Indicate(f2[0], color=COS_C, scale_factor=1.2),
            m.FadeIn(name, shift=0.2 * m.LEFT),
        )
        self.say(
            "[For king and queen, it's about zero point nine six]: [an angle of about fifteen degrees.]",
            m.AnimationGroup(m.FadeIn(title), m.FadeIn(rows[0], shift=0.15 * m.UP)),
            m.Create(arc_kq),
        )
        self.say(
            "[For king and apple, only about zero point three eight]: [sixty-seven degrees apart.]",
            m.FadeIn(rows[1], shift=0.15 * m.UP),
            m.Create(arc_ka),
        )
        self.say(
            "[And the stretched apple? Still zero point three eight.] [Cosine only cares about direction.]",
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.15 * m.UP), trace(B.o, B.p(LONG), LIME)),
            m.FadeIn(only, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.stop(self.long)
        self.clear()

    # ------------------------------------------------------------------ reading a cosine

    def reading(self):
        self.next_section(
            "Reading a cosine",
            notes=f"cos θ runs from 1 (same direction) through 0 (right angles) to −1 (opposite); for arrows of length 1 it is the signed length "
            f"of one arrow's shadow on the other. King–queen {COS_KQ:.2f}, king–apple {COS_KA:.2f}.",
        )
        C, R = np.array([-3.4, -0.45, 0.0]), 2.3
        theta = m.ValueTracker(60.0)

        def tip(deg, r=R):
            t = np.radians(deg)
            return C + r * np.array([np.cos(t), np.sin(t), 0.0])

        def shadow_x():
            x = R * np.cos(np.radians(theta.get_value()))
            return x if abs(x) > 0.02 else 0.02

        ring = m.Circle(radius=R).set_stroke(FAINT, width=2).move_to(C)
        diameter = m.Line(C + R * m.LEFT, C + R * m.RIGHT).set_stroke(FAINT, width=2)
        a_arrow = m.Arrow(C, tip(0), buff=0, color=SKY, stroke_width=6, max_tip_length_to_length_ratio=0.15)
        a_tag = tex(r"\mathbf{a}", 40, SKY).move_to(tip(0) + np.array([0.3, -0.38, 0]))
        b_arrow = m.always_redraw(lambda: m.Arrow(C, tip(theta.get_value()), buff=0, color=AMBER, stroke_width=6, max_tip_length_to_length_ratio=0.15))
        b_tag = tex(r"\mathbf{b}", 40, AMBER)
        b_tag.add_updater(lambda t: t.move_to(tip(theta.get_value() + 9, R + 0.32)))
        angle = m.always_redraw(
            lambda: m.Arc(radius=0.6, start_angle=0, angle=np.radians(max(theta.get_value(), 0.5)), arc_center=C).set_stroke(AMBER, width=4)
        )
        drop = 0.16  # the shadow sits just under arrow a
        shadow = m.always_redraw(lambda: m.Line(C + drop * m.DOWN, C + np.array([shadow_x(), -drop, 0])).set_stroke(AMBER, width=9))
        plumb = m.always_redraw(
            lambda: m.DashedLine(tip(theta.get_value()), C + np.array([shadow_x(), -drop, 0]), dash_length=0.08).set_stroke(MUTED, width=2.5)
        )
        caption = text("arrows of length 1:  cosine = shadow length", 22, MUTED).move_to([-3.4, -3.25, 0])

        def clean(v):
            return 0.0 if abs(v) < 0.005 else float(v)

        deg_num = m.DecimalNumber(60, num_decimal_places=0, mob_class=Mono, font_size=44, color=AMBER)
        cos_num = m.DecimalNumber(0.5, num_decimal_places=2, mob_class=Mono, font_size=44, color=AMBER)
        deg_label = text("angle", 30, MUTED, weight=m.SEMIBOLD)
        cos_label = text("cosine", 30, MUTED, weight=m.SEMIBOLD)
        labels = m.VGroup(deg_label, cos_label).arrange(m.DOWN, buff=0.55, aligned_edge=m.LEFT).move_to([2.3, 1.45, 0])
        deg_num.next_to(deg_label, m.RIGHT, buff=0.4).align_to(m.RIGHT * 3.6, m.LEFT)
        cos_num.next_to(cos_label, m.RIGHT, buff=0.4).align_to(m.RIGHT * 3.6, m.LEFT)
        sit_on(deg_num, deg_label)
        sit_on(cos_num, cos_label, glyph=1)
        deg_sign = text("°", 40, AMBER)
        deg_num.add_updater(lambda d: d.set_value(theta.get_value()))
        cos_num.add_updater(lambda d: d.set_value(clean(np.cos(np.radians(theta.get_value())))))
        deg_sign.add_updater(lambda d: d.next_to(deg_num, m.RIGHT, buff=0.06).align_to(deg_num, m.UP))
        readouts = m.VGroup(labels, deg_num, cos_num, deg_sign)

        def g(v):
            return np.array([3.4 + 2.4 * v, -0.85, 0.0])

        bar = m.Line(g(-1), g(1)).set_stroke(MUTED, width=4)
        notches = m.VGroup(*[m.Line(g(v) + 0.12 * m.UP, g(v) + 0.12 * m.DOWN).set_stroke(MUTED, width=3) for v in (-1, 0, 1)])
        ends = m.VGroup(*[text(s, 22, MUTED).next_to(g(v), m.DOWN, buff=0.25) for v, s in ((-1, "−1"), (0, "0"), (1, "1"))])
        descs = {
            v: text(s, 22, INK, weight=m.SEMIBOLD).next_to(ends[i], m.DOWN, buff=0.18)
            for i, (v, s) in enumerate(((-1, "opposite"), (0, "at right angles"), (1, "same direction")))
        }
        for d in descs.values():
            d.align_to(descs[-1], m.UP)
        marker = m.always_redraw(
            lambda: m.Triangle().scale(0.12).rotate(np.pi).set_fill(AMBER, opacity=1).set_stroke(width=0).move_to(
                g(np.cos(np.radians(theta.get_value()))) + 0.32 * m.UP
            )
        )
        gauge = m.VGroup(bar, notches, ends)
        spots = {}
        for w, c in (("queen", COS_KQ), ("apple", COS_KA)):
            spot = m.Dot(g(c), radius=0.08, color=WORDS[w][0]).set_z_index(2)
            spots[w] = m.VGroup(spot, text(w, 22, WORDS[w][0], weight=m.BOLD).move_to(g(c) + 0.78 * m.UP))

        self.say(
            "[Here's a way to picture a cosine.] [For two arrows of length one, it's the length of the shadow one casts on the other.]",
            m.AnimationGroup(
                self.new_header("READING A COSINE", "From one, to zero, to minus one"),
                m.Create(ring), m.Create(diameter), m.GrowArrow(a_arrow), m.FadeIn(a_tag),
                m.FadeIn(b_arrow), m.FadeIn(b_tag), m.FadeIn(angle), m.FadeIn(readouts),
            ),
            m.AnimationGroup(m.FadeIn(shadow), m.FadeIn(plumb), m.FadeIn(caption, shift=0.15 * m.UP), m.Create(gauge), m.FadeIn(marker)),
        )
        self.say(
            "[Pointing the same way, the shadow is full length]: [a cosine of one.]",
            theta.animate(run_time=1.8).set_value(0),
            m.FadeIn(descs[1], shift=0.15 * m.UP),
        )
        self.say(
            "[At right angles, there's no shadow at all]: [zero.]",
            theta.animate(run_time=1.8).set_value(90),
            m.FadeIn(descs[0], shift=0.15 * m.UP),
        )
        self.say(
            "[Point them in opposite directions], [and the shadow flips]: [minus one.]",
            theta.animate(run_time=2.0).set_value(180),
            m.Indicate(cos_num, color=AMBER, scale_factor=1.2),
            m.FadeIn(descs[-1], shift=0.15 * m.UP),
        )
        self.say(
            "[King and queen sit near the top, at zero point nine six], [and king and apple at zero point three eight.]",
            m.AnimationGroup(theta.animate(run_time=2.0).set_value(DEG_KQ), m.FadeIn(spots["queen"], shift=0.15 * m.DOWN)),
            m.AnimationGroup(theta.animate(run_time=1.6).set_value(DEG_KA), m.FadeIn(spots["apple"], shift=0.15 * m.DOWN)),
        )
        assert round(float(np.cos(np.radians(DEG_KQ))), 2) == 0.96 and round(float(np.cos(np.radians(DEG_KA))), 2) == 0.38
        self.wait(0.3)
        self.stop(b_arrow, b_tag, angle, shadow, plumb, marker, deg_num, cos_num, deg_sign)
        self.clear()

    # ------------------------------------------------------------------ a third dimension

    def more_dims(self):
        self.next_section(
            "More dimensions",
            notes=f"Toy 3-D vectors (made up): king [5, 1, 4], queen [4, 2, 4], apple [1, 5, 0]; with king: queen {COS3_KQ:.3f}, apple {COS3_KA:.3f}. "
            "The 300-bar strip is illustrative (seeded random values). word2vec's Google News vectors have 300 numbers per word (Mikolov et al., 2013; "
            "code.google.com/archive/p/word2vec, GoogleNews-vectors-negative300; checked 2026-10-21).",
        )
        phi = m.ValueTracker(12.0)
        grow = {w: m.ValueTracker(0.0) for w in WORDS}

        def frame():
            f = phi.get_value()
            return m.VGroup(
                *[
                    m.Arrow(project((0, 0, 0), f), project(e, f), buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.06)
                    for e, _, _ in AXES3
                ]
            )

        def words3():
            f, group = phi.get_value(), m.VGroup()
            for w in WORDS:
                k = grow[w].get_value()
                if k > 0.02:
                    group.add(
                        m.Arrow(project((0, 0, 0), f), project(k * VEC3[w], f), buff=0, color=WORDS[w][0], stroke_width=6, max_tip_length_to_length_ratio=0.2)
                    )
            return group

        axes3 = m.always_redraw(frame)
        arrows3 = m.always_redraw(words3)
        arrows3.set_z_index(1)
        axis_tags = []
        for e, name, side in AXES3:
            tag = text(name, 22, MUTED)
            tag.add_updater(lambda t, e=e, side=side: t.next_to(project(e, phi.get_value()), side, buff=0.15))
            axis_tags.append(tag)
        tags3 = {}
        for w in WORDS:
            tag = word_tag(w, 24)
            tag.add_updater(lambda t, w=w: t.next_to(project(VEC3[w], phi.get_value()), TAGS3[w], buff=0.14))
            tags3[w] = tag

        rows = m.VGroup(*[vector_row(w, VEC3[w], 26)[0] for w in WORDS]).arrange(m.DOWN, buff=0.24, aligned_edge=m.LEFT).move_to([3.75, 1.75, 0])
        formula = m.MathTex(r"\mathbf{a} \cdot \mathbf{b}", "=", r"a_1 b_1", "+", r"a_2 b_2", "+", r"a_3 b_3", font_size=40, color=INK).move_to([3.75, 0.25, 0])
        formula[0].set_color(DOT_C)
        formula[6].set_color(AMBER)
        new_term = text("one more term", 20, AMBER, weight=m.SEMIBOLD).next_to(formula[6], m.DOWN, buff=0.22)
        heading3 = text("cosine with king:  2 numbers → 3", 20, MUTED)
        results = m.VGroup(
            pieces([("queen   ", PINK), (f"{COS_KQ:.2f}", MUTED), (" → ", MUTED), (f"{COS3_KQ:.2f}", AMBER)], 26, MONO)[0],
            pieces([("apple   ", LIME), (f"{COS_KA:.2f}", MUTED), (" → ", MUTED), (f"{COS3_KA:.2f}", AMBER)], 26, MONO)[0],
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        block = m.VGroup(heading3, results).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([3.75, -1.3, 0])

        top = 0.42
        step = 12.4 / len(EMBED)
        bars = m.VGroup(
            *[
                m.Rectangle(width=step * 0.6, height=max(abs(v) / 2.2 * top, 0.02))
                .set_stroke(width=0)
                .set_fill(SKY if v > 0 else mix(SKY, BG, 0.45), opacity=1)
                .move_to([-6.2 + step * (i + 0.5), -3.1 + np.sign(v) * max(abs(v) / 2.2 * top, 0.02) / 2, 0])
                for i, v in enumerate(EMBED)
            ]
        )
        strip_label = text("300 numbers · values made up", 22, MUTED).move_to([0, -2.4, 0]).align_to(m.LEFT * 6.2, m.LEFT)
        source = text("word2vec: 300 numbers per word · Mikolov et al., 2013", 20, MUTED).move_to([0, -2.4, 0]).align_to(m.RIGHT * 6.2, m.RIGHT)

        self.say(
            "[Real embeddings use far more than two numbers.] [Let's add a third: is the word a person?]",
            m.AnimationGroup(self.new_header("MORE DIMENSIONS", "Same math, more numbers"), m.FadeIn(axes3), m.FadeIn(axis_tags[0]), m.FadeIn(axis_tags[1])),
            m.AnimationGroup(m.FadeIn(axis_tags[2], shift=0.2 * m.DOWN), phi.animate(run_time=2.6).set_value(24)),
        )
        self.add(arrows3)
        self.say(
            "[King and queen both get a four], [and apple gets a zero.]",
            m.AnimationGroup(
                m.FadeIn(rows[0], shift=0.2 * m.LEFT), m.FadeIn(rows[1], shift=0.2 * m.LEFT),
                grow["king"].animate.set_value(1), grow["queen"].animate.set_value(1),
                m.FadeIn(tags3["king"]), m.FadeIn(tags3["queen"]),
            ),
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.2 * m.LEFT), grow["apple"].animate.set_value(1), m.FadeIn(tags3["apple"])),
        )
        self.say(
            "[The dot product simply gains a third term.]",
            m.AnimationGroup(m.Write(formula, run_time=1.6), m.FadeIn(new_term, shift=0.15 * m.UP), phi.animate(run_time=2.4).set_value(30)),
        )
        self.say(
            "[King and queen get even closer, at zero point nine eight], [while apple drops to zero point three.]",
            m.AnimationGroup(m.FadeIn(heading3), m.FadeIn(results[0], shift=0.15 * m.UP)),
            m.FadeIn(results[1], shift=0.15 * m.UP),
        )
        self.say(
            "[The math works the same with three hundred numbers, or three thousand.] We just can't draw it.",
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(b) for b in bars], lag_ratio=0.01, run_time=2.4),
                m.FadeIn(strip_label, shift=0.15 * m.UP), m.FadeIn(source, shift=0.15 * m.UP),
            ),
        )
        self.wait(0.3)
        self.stop(axes3, arrows3, *axis_tags, *tags3.values())
        self.clear()

    # ------------------------------------------------------------------ search

    def ranking(self, x, title, color, order, score, fmt):
        box = card(6.0, 3.2, color, CARD, stroke=2).move_to([x, -0.3, 0])
        head = text(title, 26, color, weight=m.BOLD).move_to(box.get_top() + 0.42 * m.DOWN)
        best = max(score(v) for _, v in order)
        left = box.get_left()[0]
        rows = m.VGroup()
        for i, (name, v) in enumerate(order):
            y = box.get_top()[1] - 1.2 - 0.68 * i
            c, f = WORDS[name.split()[0]]
            rank = text(f"{i + 1}", 24, MUTED, weight=m.BOLD).move_to([left + 0.45, y, 0])
            word = chip(name, c, f, width=1.9, height=0.58, size=24).move_to([left + 1.65, y, 0])
            w = 2.1 * score(v) / best
            bar = m.Rectangle(width=w, height=0.3).set_stroke(width=0).set_fill(color, opacity=0.85).move_to([left + 2.8 + w / 2, y, 0])
            num = code(fmt(score(v)), 24, color).move_to([box.get_right()[0] - 0.6, y, 0])
            rows.add(m.VGroup(rank, word, bar, num))
        return m.VGroup(box, head), rows

    def search(self):
        self.next_section(
            "Search by meaning",
            notes=f"Query 'king'. By dot product: apple ×3 {DOT_KL:g}, queen {DOT_KQ:g}, apple {DOT_KA:g}. By cosine: queen {COS_KQ:.2f}, apple and apple ×3 tie at {COS_KA:.2f}. "
            "Vector search in search engines and RAG (Episode 32) commonly ranks by cosine similarity (or by dot products of length-normalised vectors, the same thing).",
        )
        ask = text("Which word is most like", 30, INK, weight=m.SEMIBOLD)
        query = m.VGroup(ask, chip("king", SKY, SKY_FILL, width=1.5, height=0.7, size=28), text("?", 30, INK, weight=m.SEMIBOLD))
        query.arrange(m.RIGHT, buff=0.25).move_to([0, 2.05, 0])
        by_dot, dot_rows = self.ranking(-3.3, "ranked by dot product", DOT_C, BY_DOT, lambda v: float(np.dot(KING, v)), lambda s: f"{s:g}")
        by_cos, cos_rows = self.ranking(3.3, "ranked by cosine", COS_C, BY_COS, lambda v: cosine(KING, v), lambda s: f"{s:.2f}")
        bad = pill("✗ the long apple wins", BAD, BAD_FILL, 22).move_to([-3.3, -2.45, 0])
        good = pill("✓ queen wins", GOOD, GOOD_FILL, 22).move_to([3.3, -2.45, 0])
        used = pill("search engines and RAG (Episode 32) often rank by cosine similarity", AMBER, AMBER_FILL, 22).move_to([0, -3.3, 0])

        self.say(
            "[So why does this matter? Search.] [Ask a computer which word is most like king.]",
            self.new_header("THE BIG PICTURE", "Search by meaning"),
            m.FadeIn(query, shift=0.2 * m.DOWN),
        )
        self.say(
            "[Rank by dot product, and the stretched apple wins], [just because it's long.]",
            m.AnimationGroup(m.FadeIn(by_dot), m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in dot_rows], lag_ratio=0.3)),
            m.FadeIn(bad, shift=0.2 * m.UP),
        )
        self.say(
            "[Rank by cosine, and queen comes first], [with both apples tied.]",
            m.AnimationGroup(m.FadeIn(by_cos), m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in cos_rows], lag_ratio=0.3)),
            m.AnimationGroup(m.Indicate(cos_rows[1][3], color=COS_C), m.Indicate(cos_rows[2][3], color=COS_C), m.FadeIn(good, shift=0.2 * m.UP)),
        )
        self.say(
            "[That's why many search systems, including the retrieval-augmented generation of episode thirty-two, rank results by cosine similarity.]",
            m.FadeIn(used, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Arrows, angles, similarity.")
        halo = glow(AMBER, radius=4.0)
        words = on_baseline(
            m.VGroup(*[text(w, 58, c, weight=m.BOLD) for w, c in zip(("Arrows", "Angles", "Similarity"), (SKY, AMBER, GOOD))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.35 * m.UP)
        tagline = text("Similar meanings point in similar directions.", 32, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [arrows], [angles], [similarity.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Similar meanings point in similar directions.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
