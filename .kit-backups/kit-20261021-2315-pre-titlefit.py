"""The series kit for "AI, Explained": design system, components, voice, and the Episode base
class with the series intro and next-episode outro. Episodes import it:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kit import *  # noqa: F403

    class Ep01WhatIsAI(Episode):
        number = 1
        def lesson(self): ...

Run from an episode folder: `VOICE=draft manimgx inspect scene.py` (silent draft) or, with the
narrator serving from the series folder, `manimgx render scene.py`.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import manimgx as m
from manimgx.audio import Word

SERIES = Path(__file__).resolve().parent
ASSETS = SERIES / "assets"

# ManimGX's TransformFromCopy animates the target itself, so say() shows the finished target from
# the start of its spoken line: it pops in early, vanishes at its cue, then flies in. FadeIn from
# the source waits for its cue and looks nearly the same (the target glides out of the source),
# even when the source itself appears earlier in the line. Every episode gets this through kit.
def _from_copy(mobject, target_mobject, **kwargs):
    return m.FadeIn(target_mobject, target_position=mobject, **kwargs)


# FadeTransform and FadeTransformPieces show their target early the same way (found in Ep31);
# a cross-fade where each side glides toward the other waits for its cue and looks the same.
def _fade_transform(mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs):
    return m.AnimationGroup(
        m.FadeOut(mobject, target_position=target_mobject),
        m.FadeIn(target_mobject, target_position=mobject),
        **kwargs,
    )


m.TransformFromCopy = _from_copy
m.FadeTransform = _fade_transform
m.FadeTransformPieces = _fade_transform


# A line that introduces a group and also moves, rotates or highlights a PART of it shows that
# part alone from the start of the line: play() puts the object of every non-introducing animation
# on screen at once, and say() plays the whole line as one play (found in Ep32). Marking those
# animations as introducers keeps play() from adding their parts early; each part then appears
# with its group, and is animated on its own cue. Episode.say applies this.
def _leaves(animation):
    subs = getattr(animation, "animations", None)
    if subs:
        for sub in subs:
            yield from _leaves(sub)
    else:
        yield animation


def _cue_safe(scene, animations):
    on_screen = {id(x) for x in scene.get_mobject_family_members()}
    roots, family = set(), set()
    for animation in animations:
        for leaf in _leaves(animation):
            if leaf.is_introducer() and getattr(leaf, "mobject", None) is not None:
                roots.add(id(leaf.mobject))
                family |= {id(x) for x in leaf.mobject.get_family()}
    for animation in animations:
        for leaf in _leaves(animation):
            mob = getattr(leaf, "mobject", None)
            if mob is None or leaf.is_introducer() or id(mob) in roots or id(mob) in on_screen:
                continue
            # a part of something introduced in this line. (Not a GROUP whose parts are introduced:
            # marking a ReplacementTransform of such a group left its result behind after clear().)
            if id(mob) in family:
                leaf.introducer = True
    return animations


# The wrapper is compiled under ManimGX's package path: ManimGX labels each play with its first
# caller outside the package, so the label still points at the episode's own scene.py line.
_SAY = """
def say(self, script, *animations, voice=None):
    return _scene_say(self, script, *_cue_safe(self, animations), voice=voice)
"""
_say_namespace = {"_scene_say": m.Scene.say, "_cue_safe": _cue_safe}
exec(compile(_SAY, str(Path(m.__file__).parent / "_series_kit_say.py"), "exec"), _say_namespace)

# ---------------------------------------------------------------------------- design system

BG = "#0B1020"
CARD = "#121A2C"
EDGE = "#2A3550"
INK = "#E2E8F0"
MUTED = "#94A3B8"
FAINT = "#475569"
TEAL, TEAL_FILL = "#2DD4BF", "#0E2A2C"
AMBER, AMBER_FILL = "#FBBF24", "#2A2112"
VIOLET, VIOLET_FILL = "#C084FC", "#221A3A"
SKY, SKY_FILL = "#7DD3FC", "#13243A"
BAD, BAD_FILL = "#F87171", "#3A1418"
GOOD, GOOD_FILL = "#4ADE80", "#10291C"
PINK, PINK_FILL = "#F472B6", "#33142B"
BLUE, LIME = "#60A5FA", "#A3E635"

SANS = "Avenir Next"
MONO = "Fira Code"

m.config.background_color = BG

# the course: level → (name, accent, fill); episode → (title, level). See COURSE.md.
LEVELS = {
    1: ("FOUNDATIONS", SKY, SKY_FILL),
    2: ("THE MATH OF AI", AMBER, AMBER_FILL),
    3: ("DEEP LEARNING", VIOLET, VIOLET_FILL),
    4: ("LLMs & GENERATIVE AI", TEAL, TEAL_FILL),
    5: ("RESEARCH FRONTIERS", PINK, PINK_FILL),
}
EPISODES = {
    1: ("What is AI?", 1), 2: ("How Machines Learn", 1), 3: ("Neural Networks", 1),
    4: ("Data: Features, Labels & Splits", 1), 5: ("Overfitting & Generalization", 1),
    6: ("Measuring a Model", 1), 7: ("The Classic ML Toolkit", 1), 8: ("Four Ways to Learn", 1),
    9: ("Vectors & Similarity", 2), 10: ("Matrices as Transformations", 2),
    11: ("Derivatives & Gradients", 2), 12: ("The Chain Rule & Backpropagation", 2),
    13: ("Optimizers: SGD, Momentum, Adam", 2), 14: ("Probability for ML", 2),
    15: ("Maximum Likelihood", 2), 16: ("Softmax & Cross-Entropy", 2),
    17: ("Information Theory", 2), 18: ("Eigenvectors, SVD & PCA", 2),
    19: ("Activation Functions", 3), 20: ("Convolutional Networks", 3),
    21: ("Recurrent Networks & LSTMs", 3), 22: ("Embeddings", 3), 23: ("Tokenization", 3),
    24: ("Attention & Transformers", 3), 25: ("Residuals & Normalization", 3),
    26: ("Regularization in Practice", 3), 27: ("Pretraining a Language Model", 4),
    28: ("Scaling Laws", 4), 29: ("Fine-Tuning & Instruction Tuning", 4), 30: ("RLHF & DPO", 4),
    31: ("Prompting & In-Context Learning", 4), 32: ("Retrieval-Augmented Generation", 4),
    33: ("Decoding & Sampling", 4), 34: ("VAEs & GANs", 4), 35: ("Diffusion Models", 4),
    36: ("Multimodal Models", 4), 37: ("Efficient Attention", 5),
    38: ("Positional Encodings & Long Context", 5), 39: ("Mixture of Experts", 5),
    40: ("State Space Models & Mamba", 5), 41: ("LoRA & Parameter-Efficient Fine-Tuning", 5),
    42: ("Quantization & Distillation", 5), 43: ("Reasoning Models & Test-Time Compute", 5),
    44: ("Agents & Tool Use", 5), 45: ("Mechanistic Interpretability", 5),
    46: ("Alignment & Evaluation", 5),
}
_SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def spoken_title(title):
    """A title as the narrator should read it: no symbols."""
    return title.rstrip("?.").replace(" & ", " and ").replace(":", ",")


def say_number(n):
    """A number under a hundred as words, for the narrator."""
    return _SMALL[n] if n < 20 else _TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + _SMALL[n % 10])


# ---------------------------------------------------------------------------- the voice

NARRATOR = os.environ.get("NARRATOR_URL", "http://127.0.0.1:8765/speak")
VOICE_REF = ""  # a clean 10–20 s WAV of a voice to copy; "" = the built-in voice


def narrator(text, voice_ref=VOICE_REF, exaggeration=0.5, cfg_weight=0.3, temperature=0.8):
    """Chatterbox, run locally by the tutorial-video skill's tts_server.py, timed by Whisper.
    Start it from the series folder: ~/.claude/skills/tutorial-video/scripts/narrator.sh"""
    settings = dict(voice_ref=voice_ref, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    request = urllib.request.Request(
        NARRATOR, json.dumps({"text": text, **settings}).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as reply:
            said = json.load(reply)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"no narrator at {NARRATOR}: start ~/.claude/skills/tutorial-video/scripts/narrator.sh"
            " in the series folder (or render with VOICE=draft)"
        ) from error
    return m.Speech(base64.b64decode(said["audio"]), text=text, words=[Word(*w) for w in said["words"]])


def draft_voice(text, seconds_per_word=0.3):
    """A silent stand-in at about the real voice's pace: for layout and timing passes."""
    words = re.findall(r"\S+", text)
    rate, lead = 24000, 0.15
    timed = [Word(w, lead + i * seconds_per_word, lead + (i + 1) * seconds_per_word) for i, w in enumerate(words)]
    silence = np.zeros(int((2 * lead + seconds_per_word * len(words)) * rate), dtype=np.float32)
    return m.Speech(silence, text=text, words=timed, rate=rate)


# ---------------------------------------------------------------------------- components


def text(s, size=32, color=INK, weight=m.NORMAL, **kw):
    return m.Text(s, font=SANS, font_size=size, color=color, weight=weight, **kw)


def code(s, size=22, color=INK, **kw):
    return m.Text(s, font=MONO, font_size=size, color=color, **kw)


def tex(s, size=40, color=INK, **kw):
    """A formula (LaTeX math)."""
    return m.MathTex(s, font_size=size, color=color, **kw)


def glyphs(words, key):
    """The glyphs that draw the first `key` in a Text (spaces and newlines draw none)."""
    start = words.text.index(key)
    skip = sum(c.isspace() for c in words.text[:start])
    width = sum(not c.isspace() for c in key)
    return words[start - skip : start - skip + width]


def icon(name, height):
    return m.SVGMobject(str(ASSETS / f"{name}.svg"), height=height)


def recolored(name, color, fill):
    """An icon in another accent: bright colours → `color`, dark fills → `fill` (the stage colour
    stays). Written once as assets/<name>-<color>.svg; returns that name."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() == BG.upper():
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


def card(width, height, color=EDGE, fill=CARD, radius=0.25, stroke=2.5):
    return m.RoundedRectangle(
        corner_radius=radius, width=width, height=height,
        stroke_color=color, stroke_width=stroke, fill_color=fill, fill_opacity=1,
    )


def pill(label, color, fill, size=22):
    words = text(label, size, color, weight=m.BOLD)
    height = words.height + 0.3
    box = card(words.width + 0.55, height, color, fill, radius=height / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def chip(word, color=SKY, fill=SKY_FILL, width=1.5, height=0.72, size=28):
    """A word in a small rounded box, on the same baseline as its neighbours'."""
    box = card(width, height, color, fill, radius=0.18, stroke=2.5)
    label = text(word, size, INK, weight=m.SEMIBOLD).move_to(box)
    sitting = [g for g, c in zip(label, word.replace(" ", "")) if c not in "gjpqy"] or list(label)
    baseline = box.get_center()[1] - 0.449 * size / 48 / 2
    label.shift((baseline - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    return m.VGroup(box, label)


def header(tag, title, color, fill):
    row = m.VGroup(pill(tag, color, fill, 20), text(title, 34, INK, weight=m.BOLD))
    return row.arrange(m.RIGHT, buff=0.35).to_corner(m.UL, buff=0.5)


def bubble(words, color, fill, tail="left"):
    box = card(words.width + 0.7, words.height + 0.55, color, fill, radius=0.3)
    words.move_to(box)
    side = -1 if tail == "left" else 1
    base = box.get_bottom() + np.array([side * (box.width / 2 - 0.6), 0.15, 0])
    nub = m.Polygon(
        base + [-0.2, 0, 0], base + [0.2, 0, 0], base + [side * 0.5, -0.55, 0],
        stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1,
    )
    return m.VGroup(nub, box, words)


def node(name, label, sub, color, fill, width=2.1, height=2.1, label_size=26):
    box = card(width, height, color, fill)
    body = m.VGroup(icon(name, 0.8), text(label, label_size, INK, weight=m.BOLD), text(sub, 20, MUTED))
    body.arrange(m.DOWN, buff=0.13).move_to(box)
    return m.VGroup(box, body)


def tile(name, title, desc, color, width=6.3, height=2.1):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 30, INK, weight=m.BOLD), text(desc, 22, MUTED))
    words.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    body = m.VGroup(icon(name, 1.05), words).arrange(m.RIGHT, buff=0.45)
    body.move_to(box).align_to(box, m.LEFT).shift(0.45 * m.RIGHT)
    return m.VGroup(box, body)


def arrow(start, end, color=MUTED, buff=0.1):
    return m.Arrow(start, end, buff=buff, color=color, stroke_width=5, max_tip_length_to_length_ratio=0.3)


def pulse(a, color, width=10):
    return m.ShowPassingFlash(m.Line(a.get_start(), a.get_end()).set_stroke(color, width=width), time_width=0.7)


def on_baseline(words):
    for w in words[1:]:
        w.shift((words[0][0].get_bottom()[1] - w[0].get_bottom()[1]) * m.UP)
    return words


def glow(color, radius=3.0, layers=28):
    return m.VGroup(
        *[
            m.Circle(radius=radius * (1 - i / layers), stroke_width=0).set_fill(color, opacity=0.009)
            for i in range(layers)
        ]
    )


def backdrop():
    image = m.ImageMobject(str(ASSETS / "backdrop.png"))
    image.height = 8
    return image


def vector(color, values, size=20):
    """A column of numbers in brackets (first three, a vertical ellipsis, the last)."""
    shown = [*values[:3], "⋮", values[3]]
    nums = m.VGroup(*[code(v, size, color) for v in shown]).arrange(m.DOWN, buff=0.1, aligned_edge=m.RIGHT)
    nums[3].set_x(nums[2].get_x())
    top, bottom = nums.get_top()[1] + 0.1, nums.get_bottom()[1] - 0.1

    def bracket(x, lip):
        corners = [[x + lip, top, 0], [x, top, 0], [x, bottom, 0], [x + lip, bottom, 0]]
        return m.VMobject().set_points_as_corners(corners).set_stroke(color, width=3)

    return m.VGroup(bracket(nums.get_left()[0] - 0.12, 0.1), nums, bracket(nums.get_right()[0] + 0.12, -0.1))


def axes(x_range, y_range, x_length, y_length, color=FAINT):
    """Quiet axes for the dark stage: thin, no numbers (label them with text())."""
    return m.Axes(
        x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
        axis_config={"color": color, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18},
    )


# ---------------------------------------------------------------------------- the episode


class Episode(m.Scene):
    """An episode of the series: its intro card, its lesson (`lesson()`), the next-episode card. (Not `film`: a Scene keeps its Film there.)"""

    number = 0
    voice = draft_voice if os.environ.get("VOICE") == "draft" else narrator
    say = _say_namespace["say"]  # Scene.say, made cue-safe for parts (see _cue_safe)

    def construct(self) -> None:
        self.head = None
        self.add(backdrop().set_z_index(-100))
        self.series_intro()
        self.lesson()
        self.series_outro()

    def lesson(self) -> None:
        raise NotImplementedError

    @property
    def level(self):
        return LEVELS[EPISODES[self.number][1]]

    # helpers

    def new_header(self, tag, title, color=None, fill=None):
        """The animation that swaps the header: pass it to say() with a short [span]."""
        _, accent, tint = self.level
        new = header(tag, title, color or accent, fill or tint)
        parts = [m.FadeIn(new, shift=0.25 * m.UP)]
        if self.head:
            parts.insert(0, m.FadeOut(self.head, shift=0.25 * m.UP))
        self.head = new
        return m.AnimationGroup(*parts, run_time=0.9)

    def clear(self, *keep, run_time=0.8):
        gone = [mob for mob in self.mobjects[1:] if mob not in keep and mob is not self.head]
        if gone:
            self.play(*[m.FadeOut(mob) for mob in gone], run_time=run_time)

    def drop_header(self):
        """The animation that removes the header (for full-frame endings)."""
        old, self.head = self.head, None
        return m.FadeOut(old, shift=0.25 * m.UP) if old else m.Wait(0.01)

    # the series cards

    def series_intro(self):
        title, level = EPISODES[self.number]
        name, accent, tint = LEVELS[level]
        self.next_section(f"Episode {self.number} · {title}", notes="Series card.")
        halo = glow(accent, radius=4.2).move_to(0.2 * m.UP)
        brand = text("AI, Explained", 34, INK, weight=m.BOLD, gradient=(SKY, VIOLET))
        badge = pill(f"LEVEL {level} · {name}", accent, tint, 20)
        number = text(f"Episode {self.number:02d}", 30, MUTED, weight=m.SEMIBOLD)
        heading = text(title, 76 if len(title) < 22 else 60, INK, weight=m.BOLD)
        half = min(heading.width, 6) / 2
        rule = m.Line([-half, 0, 0], [half, 0, 0]).set_stroke(accent, width=4)
        card_group = m.VGroup(brand, badge, number, heading, rule).arrange(m.DOWN, buff=0.38).move_to(0.15 * m.UP)
        rule.next_to(heading, m.DOWN, buff=0.35)

        self.wait(0.3)
        self.say(
            f"[Welcome to AI, Explained.] [Episode {say_number(self.number)}]: [{spoken_title(title)}.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(brand, shift=0.2 * m.DOWN), run_time=1.0),
            m.AnimationGroup(m.FadeIn(badge, shift=0.2 * m.UP), m.FadeIn(number, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Write(heading), m.GrowFromCenter(rule)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, card_group)], run_time=0.8)

    def series_outro(self):
        upcoming = self.number + 1
        self.next_section("Next episode", notes="Next-episode card.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        if upcoming not in EPISODES:
            self.say("Thanks for watching!")
            return
        title, level = EPISODES[upcoming]
        name, accent, tint = LEVELS[level]
        halo = glow(accent, radius=4.0)
        kicker = text("Next time", 30, MUTED, weight=m.SEMIBOLD)
        number = pill(f"EPISODE {upcoming:02d} · LEVEL {level}", accent, tint, 20)
        heading = text(title, 64 if len(title) < 26 else 50, INK, weight=m.BOLD)
        group = m.VGroup(kicker, number, heading).arrange(m.DOWN, buff=0.4).move_to(0.2 * m.UP)
        self.say(
            f"[Next time, in episode {say_number(upcoming)}]: [{spoken_title(title)}.] Thanks for watching!",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(kicker), m.FadeIn(number, shift=0.2 * m.UP)),
            m.Write(heading),
        )
        self.wait(1.0)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)

