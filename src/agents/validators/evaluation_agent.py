"""
EvaluationAgent — LLM-based quality evaluation using DeepEval GEval.

Why DeepEval instead of a raw LLM prompt?
- GEval returns a structured score (0-1) + reason string, no JSON parsing needed
- Evaluation criteria are defined as natural-language strings (user-supplied via prompts)
- The same evaluation runs as a pytest metric in CI (deepeval is already a test dependency)
- Always uses gemini-2.0-flash-lite regardless of workflow — evaluation is a simpler task
  and doesn't need the higher-cost story creation models.

story_topics workflow: runs 8 parallel GEval metrics:
  NonToxicity, Bias, Completeness, Engagability, Trustworthiness,
  Latency (contextual relevance), Precision, Recall.
  Passes when the average score >= pass_threshold.

Other workflows: single GEval with workflow-specific criteria.

Usage:
    agent = EvaluationAgent(workflow_type="story_topics")
    result = await agent.evaluate(state)
    # result["evaluation"] = {"passed": True, "score": 0.82, "reason": "...", "metrics": {...}}
"""

import asyncio
import re
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from google import genai

from ...utils.logger import setup_logger
from ...utils.config import get_settings

# Bump when _TOPICS_CRITERIA or scoring logic changes so cached verdicts
# from prior rubric versions are ignored.
_EVAL_RUBRIC_VERSION = 1

logger = setup_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Gemini adapter for DeepEval — avoids the OpenAI dependency entirely.
# DeepEval passes the `model` arg to GEval. If it's a plain string DeepEval
# assumes OpenAI and demands OPENAI_API_KEY. Passing a DeepEvalBaseLLM
# instance bypasses that and routes all LLM calls through Gemini instead.
# ---------------------------------------------------------------------------

class _GeminiEvalModel(DeepEvalBaseLLM):
    """Routes DeepEval GEval LLM calls to Google Gemini via google.genai."""

    def __init__(self, model_name: str | None = None) -> None:
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self._model_name = model_name or settings.EVALUATION_MODEL

    def load_model(self):
        return self._client

    def generate(self, prompt: str, *args, **kwargs) -> str:
        """Sync — called by metric.measure() when run inside a thread executor."""
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Async — called by metric.a_measure() when awaited directly."""
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )
        return response.text

    def get_model_name(self) -> str:
        return self._model_name


# One shared instance per process — avoids re-creating the client on every metric.
# Both topic and activity evaluations primary on gemini-2.5-flash-lite (cheapest).
# The retry ladder falls back to gemini-2.5-flash (3-6× pricier but with higher
# capacity / different load characteristics) ONLY on transient errors. In steady
# state every call lands on flash-lite; flash is invoked once-per-failure at most.
_GEMINI_EVAL_MODEL = _GeminiEvalModel()
_GEMINI_ACTIVITIES_EVAL_MODEL = _GeminiEvalModel()   # same default — flash-lite
# Fallback used only when the primary eval model 503s twice in a row.
_GEMINI_EVAL_FALLBACK_MODEL = _GeminiEvalModel(model_name="gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Single-metric criteria (non-topics, non-story workflows)
# ---------------------------------------------------------------------------

_DEFAULT_CRITERIA: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Multi-metric criteria for `activities` workflow (WF5)
# Activities are run in parallel (mcq/art/moral/science) — evaluated per-type
# with the same metric set, since the judging dimensions are shared. Hard
# metrics gate safety (kids physically do these); soft metrics judge quality.
# ---------------------------------------------------------------------------

_ACTIVITY_CRITERIA: dict[str, str] = {
    "non_toxicity": (
        "Check the activity content for anything unsafe for the specified age: harsh "
        "words, slurs, profanity, violence, scary imagery, mature themes, or anything a "
        "parent would find inappropriate. For ages 3-6, even mild fear or dark imagery "
        "should lower the score. Mark high only if fully age-safe."
    ),
    "story_alignment": (
        "Does the activity clearly connect to the story it accompanies? An MCQ should "
        "ask about events/characters in the story; an art/science/moral activity should "
        "reinforce a theme, character, or concept from the story. Penalize generic "
        "activities that could apply to any story. Mark high if a child would recognise "
        "the activity as related to the story they just heard."
    ),
    "safety_of_execution": (
        "For physical activities (art/science/moral), is every listed material and step "
        "safe for an unsupervised young child? Penalize: sharp objects (scissors, knives), "
        "choking hazards (small beads, marbles for under-3), hot liquids, fire, toxic "
        "substances, foods a child might confuse with non-food. For MCQ (text-only) "
        "default to 1.0 unless the question itself promotes unsafe behavior. Mark high "
        "only if a parent would be comfortable leaving the child to do this alone."
    ),
    "instructions_clarity": (
        "Are the instructions/options/questions self-contained and unambiguous? A "
        "non-expert parent should be able to follow them without guessing or looking "
        "things up. For MCQ, the question must have one clearly correct answer and "
        "distractors that are wrong but plausible. For physical activities, materials "
        "list + steps must be complete and ordered. Mark high if there are no "
        "ambiguities or missing pieces."
    ),
    "engagability": (
        "Would a child of the specified age want to do this activity? Score high if "
        "the activity has a fun hook, a sense of play, a tangible outcome the child "
        "is excited about, or asks an interesting question. Score lower for dry, "
        "textbook-feeling activities with no spark of fun."
    ),
    # age_appropriateness is now Python-checked (_python_age_appropriateness) on the
    # flattened activity text — same vocab/sentence-length rubric as story eval.
    "educational_value": (
        "Does the activity reinforce a story element worth learning — the moral, a "
        "science concept, vocabulary, a social skill, or creative expression? Score "
        "high if a parent can clearly answer 'what did my child learn from this?' "
        "Score low for busywork with no learning takeaway."
    ),
}

_ACTIVITY_HARD_METRICS: dict[str, float] = {
    "non_toxicity":         0.85,
    "story_alignment":      0.6,
    "safety_of_execution":  0.9,
    "instructions_clarity": 0.7,
}
# age_appropriateness is Python-computed and injected post-gather.
_ACTIVITY_SOFT_METRICS = ("engagability", "age_appropriateness", "educational_value")


def _activity_to_text(activity_type: str, data) -> str:
    """Flatten an activity's payload (varying shapes per type) into a single readable
    text block for the GEval judge. Different activities have different field names —
    this helper centralises the per-type formatting."""
    if data is None:
        return ""
    if activity_type == "mcq":
        if not isinstance(data, list):
            return str(data)
        lines = []
        for i, q in enumerate(data, 1):
            if isinstance(q, dict):
                question = q.get("question") or q.get("text") or ""
                options = q.get("options") or []
                answer = q.get("answer") or q.get("correct") or ""
                lines.append(f"Q{i}: {question}")
                if options:
                    lines.append(f"  Options: {options}")
                if answer:
                    lines.append(f"  Answer: {answer}")
            else:
                lines.append(f"Q{i}: {q}")
        return "\n".join(lines)

    # art/moral/science all have similar dict-or-list-of-dicts shape with the same key set
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [data]
    else:
        return str(data)

    blocks = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            blocks.append(f"Item {i}: {item}")
            continue
        # `image` field is raw bytes — exclude from the judge text
        parts = [f"=== Item {i} ==="]
        for k, v in item.items():
            if k == "image":
                continue
            if k == "image_generation_prompt":
                parts.append(f"image_prompt: {v}")
            else:
                parts.append(f"{k}: {v}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Multi-metric criteria for `story` workflow
# Two-tier system: hard metrics gate safety, soft metrics judge quality.
# A story passes when: every hard metric ≥ its floor AND average soft ≥ pass_threshold.
# Per-metric reasons surface in `evaluation["metric_reasons"]` so the corrector
# can target the actual failures instead of guessing from a single score.
# ---------------------------------------------------------------------------

_STORY_CRITERIA: dict[str, str] = {
    "narrative_coherence": (
        "Story-internal logic only: does the story have a clear beginning, middle, end, "
        "with cause-and-effect between scenes, and emotional shifts that follow from "
        "events? Score high if a child can follow what happens and why. Score low for "
        "abrupt jumps, unexplained changes, or contradictions."
    ),
    "engagability": (
        "Would a child AND a parent be captivated on first listen? Score high "
        "(>=0.9) if the opening has a vivid hook, the story has a clear surprise moment, "
        "2-3 interactive beats inviting the child in, and emotional shifts shown through "
        "action. The story should make a parent feel 'oh really, I did not know that' at "
        "the science/wisdom moment. Score lower if flat, generic, or textbook-feeling."
    ),
    "educational_value": (
        "Are the 2 science_concepts woven INTO the narrative (a character observes, asks, "
        "or demonstrates them) — not just listed in the JSON field? AND is the moral a "
        "non-obvious insight worth a parent saying 'oh, really?' rather than a clichéd "
        "lesson? Score high only if both science concepts appear in the story text AND "
        "the moral is fresh. Score low if science_concepts are present in the JSON but "
        "absent from the story body."
    ),
    # age_appropriateness is now Python-computed (_python_age_appropriateness)
    # — average word length + sentence length vs an age band. Free, deterministic.
}

# Story has no hard-gate metrics — pass/fail is the soft average vs pass_threshold.
# age_appropriateness is injected post-gather as a Python check.
_STORY_SOFT_METRICS = ("narrative_coherence", "engagability", "educational_value", "age_appropriateness")


# ---------------------------------------------------------------------------
# Multi-metric criteria for `image` workflow (image-prompt evaluation)
# Same two-tier system as `story`. Judges the text PROMPT sent to FLUX —
# upstream of the actual pixel render. Two primary concerns:
#   1. The image must read clearly on the page (single focal subject, bright,
#      not cluttered or tiny-in-corner).
#   2. The image must be animated / cartoon style, not photorealistic.
# ---------------------------------------------------------------------------

_IMAGE_CRITERIA: dict[str, str] = {
    "non_toxicity": (
        "Check the image prompt for descriptors unsafe for the specified age: scary or "
        "violent imagery, dark/grim atmosphere, frightening creatures, mature themes, or "
        "anything a parent of a young child would find inappropriate. For ages 3-6, even "
        "'dark forest at night' or 'scary cave' should lower the score. Mark high only if "
        "the prompt describes a fully age-safe scene."
    ),
    # copyright_safety is now a Python keyword check (_python_copyright_safety) —
    # see _evaluate_image_prompt. Faster, deterministic, no LLM call.
    "visual_clarity": (
        "Will the rendered image read clearly on a children's book page or phone screen? "
        "Score high if the prompt specifies ONE clear focal subject (the main character "
        "or central object), the subject is foregrounded (not described as small, distant, "
        "or in a corner), the scene has 1-3 elements not 5+, and the lighting is described "
        "as bright/warm/soft (not dark, dim, shadowy, or moody). Penalize cluttered scenes, "
        "tiny subjects, or dark/low-contrast lighting that would render hard to see."
    ),
    # "animated_style": (
    #     "Does the prompt explicitly invoke an animated, cartoon, or children's-book "
    #     "illustration style? Required descriptors: at least one of 'cartoon', '3D animated', "
    #     "'children's book illustration', 'watercolour', 'storybook', 'animated movie style'. "
    #     "Penalize: 'photorealistic', 'photograph', 'cinematic', 'realistic', 'hyperrealistic', "
    #     "'gritty', 'film still', or anything that pushes FLUX toward a non-animated render."
    # ),
    # "kid_attractiveness": (
    #     "Would a young child find this image instantly appealing? Score high if the prompt "
    #     "describes warm/bright colours (golden, soft pastels, warm), a friendly character "
    #     "(big eyes, smile, gentle pose), playful or wonder-filled mood, and soft/round "
    #     "visual elements. Score lower for muted palettes, neutral expressions, static poses, "
    #     "or adult-coded aesthetics."
    # ),
    # "story_alignment": (
    #     "Does the image prompt match the story's main character, setting, and a key moment "
    #     "from the narrative? The character description in the prompt must match the story's "
    #     "character (species, age, distinctive features); the setting must be from the story; "
    #     "the action shown must be a recognisable scene from the story text. Score low if "
    #     "the prompt depicts a generic scene unrelated to the story."
    # ),
    # "compositional_completeness": (
    #     "Does the prompt include all standard composition elements: character description, "
    #     "what the character is doing, setting and key objects, lighting, art style, and mood? "
    #     "Score high if all six are present and concrete. Score low if any are missing or vague."
    # ),
}

_IMAGE_HARD_METRICS: dict[str, float] = {
    "non_toxicity": 0.85,
    "copyright_safety": 0.85,
    "visual_clarity": 0.7,
    # "animated_style": 0.7,
}
# _IMAGE_SOFT_METRICS = ("kid_attractiveness", "story_alignment", "compositional_completeness")


# ---------------------------------------------------------------------------
# Multi-metric criteria for `audio` workflow
# Coverage and bytes-present are deterministic Python checks — no LLM needed
# and we WANT them to be objective (you can't fuzz "did every paragraph get
# synthesised"). Soft metrics judge the story text's suitability for TTS,
# which is upstream of and predictive of narration quality.
# ---------------------------------------------------------------------------

_AUDIO_CRITERIA: dict[str, str] = {
    # tts_friendliness and narration_pacing are now Python regex/length checks —
    # see _python_tts_friendliness / _python_narration_pacing. Deterministic and free.
    "vocabulary_pronouncability": (
        "Are the words pronounceable by a TTS voice in the specified language? Penalize: "
        "untransliterated foreign-script words mid-sentence, made-up names that TTS will "
        "mangle (e.g. 'Xqzz'), heavy use of acronyms read letter-by-letter, or "
        "abbreviations that TTS won't expand correctly (Mr. vs Mister, etc. only matters "
        "when ambiguous). Score high for plain prose using normal words."
    ),
}

# Soft metrics averaged against pass_threshold.
# tts_friendliness + narration_pacing are Python-computed; vocabulary_pronouncability
# is LLM-judged. All three count toward the soft average.
_AUDIO_SOFT_METRICS = ("tts_friendliness", "narration_pacing", "vocabulary_pronouncability")

# Hard metrics are deterministic Python checks — they MUST hit their floor.
# `paragraph_coverage` is the headline requirement: audio must cover full story text.
_AUDIO_HARD_FLOORS: dict[str, float] = {
    "paragraph_coverage":      0.95,
    "audio_bytes_present":     1.0,
    "duration_plausibility":   0.7,
    "paragraph_integrity":     0.95,
}


def _python_paragraph_coverage(story_text: str, audio_timepoints: list | None) -> tuple[float, str]:
    """Fraction of story paragraphs that have a corresponding TTS timepoint entry."""
    paragraphs = [p.strip() for p in (story_text or "").split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0, "Story text contains no paragraphs to narrate."
    if not audio_timepoints:
        return 0.0, f"Story has {len(paragraphs)} paragraphs but audio_timepoints is empty."
    covered = len(audio_timepoints)
    ratio = min(1.0, covered / len(paragraphs))
    if ratio < 1.0:
        return round(ratio, 3), (
            f"Audio covers {covered}/{len(paragraphs)} paragraphs — narration is truncated."
        )
    return 1.0, f"Audio covers all {len(paragraphs)} story paragraphs."


def _python_audio_bytes_present(audio_bytes: bytes | None) -> tuple[float, str]:
    """1.0 if the WAV blob is plausibly non-trivial (>1 KiB), 0.0 otherwise."""
    if not audio_bytes or not isinstance(audio_bytes, (bytes, bytearray)):
        return 0.0, "audio_bytes is missing or not bytes."
    if len(audio_bytes) < 1024:
        return 0.0, f"audio_bytes too small ({len(audio_bytes)} B) — likely empty WAV."
    return 1.0, f"audio_bytes present ({len(audio_bytes)} B)."


def _python_paragraph_integrity(audio_timepoints: list | None) -> tuple[float, str]:
    """Per-paragraph audio integrity: every paragraph must have a non-trivial
    duration and timestamps must advance forward. Catches the case where TTS
    returned bytes + a timepoint entry for a paragraph but the segment is
    silent / zero-duration, or where paragraphs got glued in the wrong order.

    Aggregate: fraction of paragraphs that pass both checks. 1.0 = all clean."""
    if not audio_timepoints:
        return 0.0, "No timepoints to inspect."

    bad: list[str] = []
    prev_end = 0.0
    for i, tp in enumerate(audio_timepoints):
        try:
            start = float(tp.get("StartTimestamp", tp.get("start", 0.0)))
            end = float(tp.get("EndTimestamp", tp.get("end", 0.0)))
        except (TypeError, ValueError):
            bad.append(f"#{i}: unreadable timestamps")
            continue
        duration = end - start
        if duration < 0.5:
            bad.append(f"#{i}: duration {duration:.2f}s (silent/truncated)")
        elif start + 1e-3 < prev_end:
            bad.append(f"#{i}: starts {start:.2f}s before prev ends {prev_end:.2f}s")
        prev_end = end

    total = len(audio_timepoints)
    good = total - len(bad)
    score = round(good / total, 3) if total else 0.0
    if bad:
        return score, f"Paragraph integrity issues: {'; '.join(bad[:3])}"
    return 1.0, f"All {total} paragraph segments have valid duration and ordering."


def _python_duration_plausibility(
    story_text: str, audio_timepoints: list | None
) -> tuple[float, str]:
    """Does the reported audio duration roughly match the word count?

    English narration runs ~150 wpm = 2.5 words/sec ≈ 0.4 sec/word. Anything inside
    [0.5x, 2x] of expected is "plausible" (1.0); outside that, the score degrades
    linearly. Catches the case where TTS returned bytes but the duration is
    near-zero (truncated render) or absurdly long (looping/stuck)."""
    if not audio_timepoints:
        return 0.0, "No timepoints available to compute duration."
    try:
        last_ts = audio_timepoints[-1]
        end = float(last_ts.get("EndTimestamp", last_ts.get("end", 0.0)))
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0, "Could not read final timestamp from audio_timepoints."
    if end <= 0:
        return 0.0, f"Audio duration {end:.2f}s — likely empty render."

    word_count = len((story_text or "").split())
    if word_count == 0:
        return 1.0, "No source words to compare against."
    expected = word_count * 0.4  # seconds, English ~150 wpm
    ratio = end / expected
    if 0.5 <= ratio <= 2.0:
        return 1.0, f"Duration {end:.1f}s plausible for {word_count} words (expected ~{expected:.0f}s)."
    # Linear falloff outside the band; floor at 0
    if ratio < 0.5:
        score = max(0.0, ratio / 0.5)  # 0.25 ratio → 0.5 score; 0.0 → 0.0
        return round(score, 3), (
            f"Audio duration {end:.1f}s is too short for {word_count} words "
            f"(expected ~{expected:.0f}s) — likely truncated."
        )
    score = max(0.0, 2.0 / ratio)  # 4x → 0.5 score
    return round(score, 3), (
        f"Audio duration {end:.1f}s is too long for {word_count} words "
        f"(expected ~{expected:.0f}s) — likely stuck/looping."
    )


# ---------------------------------------------------------------------------
# Shared Python checks reused across image/audio/activity/story rubrics.
# Free, deterministic, predictable — used in place of GEval where the question
# is mechanical (regex match, lookup, count, ratio).
# ---------------------------------------------------------------------------

# Common children's-IP names whose appearance in an image prompt suggests
# copyright risk. Lowercased. Substring match — short tokens like "frozen"
# could false-positive on adjectives, so keep the list specific.
_COPYRIGHT_TOKENS = (
    "disney", "pixar", "marvel", "dc comics", "harry potter", "pokemon",
    "pokémon", "mickey mouse", "minnie mouse", "donald duck", "elsa", "anna",
    "spider-man", "spiderman", "batman", "superman", "iron man", "captain america",
    "star wars", "yoda", "darth vader", "shrek", "minions", "paw patrol",
    "peppa pig", "bluey", "doraemon", "barbie", "lego", "moana", "simba",
    "lion king", "toy story", "buzz lightyear", "woody", "nemo", "dory",
)

# Markdown / TTS-hostile artefacts. If the story text has these, the TTS
# engine will read them literally ("asterisk whoosh asterisk").
_TTS_HOSTILE_PATTERNS = (
    re.compile(r"\*[^*\n]+\*"),          # *bold* / *whoosh*
    re.compile(r"_[^_\n]+_"),            # _italic_
    re.compile(r"\[[^\]]+\]"),           # [stage direction]
    re.compile(r"\([^)]*(?:crash|bang|whoosh|boom|pop|swoosh|zoom)[^)]*\)", re.I),
    re.compile(r"^#{1,6}\s", re.M),      # markdown headers
    re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"),  # emoji ranges
)


def _python_copyright_safety(image_prompt: str) -> tuple[float, str]:
    """1.0 if no known copyrighted-IP tokens appear in the prompt; 0.0 if any do."""
    if not image_prompt:
        return 1.0, "Empty prompt — nothing to flag."
    lowered = image_prompt.lower()
    hits = [tok for tok in _COPYRIGHT_TOKENS if tok in lowered]
    if hits:
        return 0.0, f"Prompt mentions copyrighted IP tokens: {hits[:5]}"
    return 1.0, "No known copyrighted-IP tokens detected in prompt."


def _python_tts_friendliness(story_text: str) -> tuple[float, str]:
    """1.0 if no TTS-hostile artefacts (markdown, stage directions, emoji,
    sound-effect annotations) appear; degrades as more are found."""
    if not story_text:
        return 0.0, "Empty story text."
    findings: list[str] = []
    for pat in _TTS_HOSTILE_PATTERNS:
        m = pat.findall(story_text)
        if m:
            findings.append(f"{pat.pattern!r} → {len(m)} match(es)")
    if not findings:
        return 1.0, "No TTS-hostile artefacts found."
    # Each pattern hit knocks 0.25 off; floor at 0.
    score = max(0.0, 1.0 - 0.25 * len(findings))
    return round(score, 3), f"TTS-hostile artefacts: {'; '.join(findings[:3])}"


def _python_narration_pacing(story_text: str) -> tuple[float, str]:
    """Sentence- and paragraph-length sanity check for TTS narration.

    Penalizes: paragraphs over 6 sentences (too long for one breath), sentences
    over 30 words (run-on), or paragraphs with no terminal punctuation.
    1.0 if everything inside reasonable bounds."""
    if not story_text or not story_text.strip():
        return 0.0, "Empty story text."
    paragraphs = [p.strip() for p in story_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0, "Story has no paragraph breaks."
    issues: list[str] = []
    for i, p in enumerate(paragraphs, 1):
        sentences = re.split(r"[.!?]+\s+", p)
        sentences = [s for s in sentences if s.strip()]
        if len(sentences) > 6:
            issues.append(f"para {i}: {len(sentences)} sentences (>6)")
            continue
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        if long_sentences:
            issues.append(f"para {i}: {len(long_sentences)} sentence(s) >30 words")
    if not issues:
        return 1.0, f"All {len(paragraphs)} paragraphs are within TTS-friendly length bounds."
    # Each issue costs 0.15, floor 0.
    score = max(0.0, 1.0 - 0.15 * len(issues))
    return round(score, 3), f"Pacing issues: {'; '.join(issues[:3])}"


# Per-age expected word-length / sentence-length bands. Tuned for "read-aloud
# to child" rather than "child reads themselves" — slightly more lenient.
_AGE_BANDS: dict[str, dict[str, float]] = {
    "3-4":  {"max_avg_word_len": 4.5, "max_avg_sent_len": 12},
    "4-5":  {"max_avg_word_len": 4.7, "max_avg_sent_len": 14},
    "5-6":  {"max_avg_word_len": 5.0, "max_avg_sent_len": 16},
    "6-8":  {"max_avg_word_len": 5.5, "max_avg_sent_len": 18},
    "8-10": {"max_avg_word_len": 6.0, "max_avg_sent_len": 22},
}
_DEFAULT_AGE_BAND = {"max_avg_word_len": 5.5, "max_avg_sent_len": 18}


def _python_age_appropriateness(text: str, age: str) -> tuple[float, str]:
    """Cheap proxy for age-appropriate vocabulary/complexity.

    Looks at average word length (syllabic proxy) and average sentence length
    against an age band. Both inside band = 1.0; degrades linearly as each
    overshoots its limit."""
    if not text or not text.strip():
        return 0.0, "Empty content."
    band = _AGE_BANDS.get(str(age), _DEFAULT_AGE_BAND)

    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 1.0, "Non-alphabetic content — skipping age check."
    avg_word_len = sum(len(w) for w in words) / len(words)

    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.5, "Text has no sentence terminators."
    avg_sent_len = len(words) / len(sentences)

    word_ratio = min(1.0, band["max_avg_word_len"] / avg_word_len) if avg_word_len > 0 else 1.0
    sent_ratio = min(1.0, band["max_avg_sent_len"] / avg_sent_len) if avg_sent_len > 0 else 1.0
    score = round((word_ratio + sent_ratio) / 2, 3)
    if score >= 0.95:
        return 1.0, (
            f"avg word len {avg_word_len:.1f} / avg sent len {avg_sent_len:.1f} — "
            f"within age-{age} band."
        )
    return score, (
        f"avg word len {avg_word_len:.1f} (limit {band['max_avg_word_len']}); "
        f"avg sent len {avg_sent_len:.1f} (limit {band['max_avg_sent_len']}) — "
        f"may be advanced for age {age}."
    )


# ---------------------------------------------------------------------------
# Multi-metric criteria for story_topics (8 dimensions)
# ---------------------------------------------------------------------------

# GEval-judged criteria — only the subjective dimensions that genuinely need an LLM.
# `completeness` and `recall` were removed from this dict because they are
# deterministic checks (field presence, duplicate counting); they're now computed
# in Python below and merged into the final metric dict for backward compatibility.
_TOPICS_CRITERIA: dict[str, str] = {
    "non_toxicity": (
        "Check the title and description for content that is unsafe for the specified age. "
        "Penalize: harsh words, slurs, profanity, violence, scary imagery, mature themes, "
        "or anything a parent of a child this age would find inappropriate. "
        "For very young ages (3-6), even mild fear, conflict, or dark imagery should lower the score. "
        "Mark high (1.0) only if the content is fully age-safe."
    ),
    "bias": (
        "Judge the OUTPUT TEXT ONLY for biased or stereotyping language. Does the title "
        "or description reinforce a stereotype, use exclusionary or demeaning language, "
        "or portray any group unfairly? "
        "DO NOT penalize the topic for not mentioning the country, religion, or theme "
        "from the input — cultural fit is judged separately under 'latency'. A neutral, "
        "universal topic (e.g. an animal story) is NOT biased by virtue of being neutral. "
        "Mark 1.0 if the text contains nothing stereotyping, exclusionary, or demeaning."
    ),
    "engagability": (
        "Would a child of the specified age want to hear this story on first impression? "
        "Score >=0.7 if the title has any of: a vivid image, a named character, a hint of "
        "action/mystery, or a relatable feeling. Score 0.4-0.6 if it is a generic concept "
        "title (e.g. 'Sunshine and Shadows Play'). Score <0.4 only if it sounds like a "
        "textbook heading with no spark. Commit to a numeric score — never refuse."
    ),
    "trustworthiness": (
        "Judge the OUTPUT TEXT ONLY for honesty and child-safety. Penalize ONLY if the "
        "text itself contains: a factual error stated as fact (e.g. 'the sun is purple'), "
        "superstition presented as fact, harmful advice (e.g. 'hit your friend when angry'), "
        "or a misleading moral. A made-up character or fantasy scenario is fine — fantasy "
        "is not misinformation. "
        "DO NOT penalize the topic for not referencing the country or religion from the "
        "input — that is a relevance question handled by 'latency'. If there is nothing "
        "factually wrong or harmful in the output text, mark 1.0."
    ),
    "latency": (
        "Relevance check: could this topic PLAUSIBLY appear in a children's story set "
        "in the requested country/religion context? Be GENEROUS — universal themes "
        "(animals, nature, family, kindness, courage) fit virtually every culture and "
        "should score high. The topic does NOT need to name the country or religion "
        "explicitly; implicit cultural fit via tone, values, or setting is sufficient. "
        "Only penalize if the topic CONTRADICTS the requested context (e.g. a Christmas "
        "story when the context is Hindu, or a meat-eating moral when the context is "
        "Jain). A neutral animal/nature story works in any context — mark 1.0. "
        "Country/religion labels like 'universal_wisdom' are INTERNAL identifiers; "
        "do not expect them in the output."
    ),
    "precision": (
        "Does the title point to ONE clear story idea (not vague or scattered)? "
        "Mark high if a child could guess what the story is roughly about from title + description. "
        "A focused 5-6 word title with a clarifying description is enough — do not demand more."
    ),
}


def _python_completeness(topics: list[dict]) -> tuple[float, str]:
    """Field-presence check — no LLM needed."""
    if not topics:
        return 0.0, "No topics provided."
    missing = [
        t.get("title", "?") for t in topics
        if not (t.get("title") or "").strip() or not (t.get("description") or "").strip()
    ]
    if missing:
        return 0.0, f"Topics missing title or description: {missing}"
    return 1.0, f"All {len(topics)} topics have non-empty title + description."


def _python_recall(topics: list[dict]) -> tuple[float, str]:
    """Count + duplicate check — no LLM needed."""
    if not topics:
        return 0.0, "No topics."
    if len(topics) == 1:
        return 1.0, "Single-topic output; recall is not applicable."
    titles = [(t.get("title") or "").strip().lower() for t in topics]
    unique = set(titles)
    if len(unique) == len(titles):
        return 1.0, f"All {len(titles)} titles are distinct."
    dup_count = len(titles) - len(unique)
    score = max(0.0, 1.0 - dup_count / len(titles))
    return round(score, 3), f"{dup_count}/{len(titles)} duplicate titles detected."

# Minimum score (0-1) for evaluation to pass
PASS_THRESHOLD = 0.6

# Cap concurrent GEval calls to avoid hammering the evaluator model with rate-limit /
# 503 errors. With 8 metrics × N topics the unbounded fan-out can be 64+ in flight
# (each GEval also does internal LLM hops), which the upstream provider throttles.
_EVAL_CONCURRENCY = 4
_eval_semaphore: asyncio.Semaphore | None = None


def _get_eval_semaphore() -> asyncio.Semaphore:
    """Lazy semaphore so it binds to the running event loop on first use
    (each pytest-asyncio test gets its own loop)."""
    global _eval_semaphore
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _eval_semaphore is None or getattr(_eval_semaphore, "_loop", loop) is not loop:
        _eval_semaphore = asyncio.Semaphore(_EVAL_CONCURRENCY)
    return _eval_semaphore


def _is_transient_eval_error(exc: Exception) -> bool:
    """Gemini-side transient failures (overload, rate limit) that deserve one retry."""
    msg = str(exc)
    return any(tok in msg for tok in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


async def _run_geval_with_retry(
    name: str,
    criteria: str,
    test_case: LLMTestCase,
    threshold: float,
    sem: asyncio.Semaphore,
    is_hard: bool,
    log_prefix: str,
    eval_model: DeepEvalBaseLLM | None = None,
) -> tuple[str, float, str]:
    """Run one GEval metric with up to 2 retries on transient errors.

    Retry ladder:
      attempt 1 — primary model (gemini-2.5-flash-lite), no delay
      attempt 2 — primary again, 3s backoff (handles brief blip)
      attempt 3 — FALLBACK model (gemini-2.5-flash), 6s backoff
                  — higher-capacity endpoint, pricier but less load-shared

    Skip-policy after all attempts fail:
    - SOFT metrics: skip-as-pass (1.0) — soft metrics are quality-of-life and
      shouldn't fail a story because Gemini hiccupped.
    - HARD metrics: skip-as-FAIL (0.0) — hard metrics gate safety (toxicity,
      copyright, execution-safety, instructions); passing them silently when
      the judge didn't actually evaluate is a real risk.
    """
    primary = eval_model or _GEMINI_EVAL_MODEL
    last_error: Exception | None = None
    # (attempt_no, model_to_use, backoff_before_this_attempt)
    ladder = [
        (1, primary, 0.0),
        (2, primary, 3.0),
        (3, _GEMINI_EVAL_FALLBACK_MODEL, 6.0),
    ]
    for attempt, model, backoff in ladder:
        if backoff:
            await asyncio.sleep(backoff)
        try:
            metric = GEval(
                name=name,
                criteria=criteria,
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                model=model,
                threshold=threshold,
            )
            async with sem:
                await metric.a_measure(test_case)
            return name, round(metric.score, 3), metric.reason or ""
        except Exception as e:
            last_error = e
            # Non-transient errors abort the ladder immediately; no point
            # re-rolling the same bug or non-retryable response.
            if not _is_transient_eval_error(e):
                break
            if attempt < 3:
                model_label = "primary" if model is primary else "fallback"
                logger.info(
                    f"{log_prefix} Metric '{name}' transient error on {model_label} "
                    f"(attempt {attempt}); will retry: {e}"
                )
                continue
            break

    # All attempts exhausted
    if is_hard:
        logger.warning(
            f"{log_prefix} Hard metric '{name}' failed after retries — scoring 0.0 (skip-as-FAIL): {last_error}"
        )
        return name, 0.0, f"failed-after-retries: {last_error}"
    logger.warning(
        f"{log_prefix} Soft metric '{name}' failed after retries — scoring 1.0 (skip-as-pass): {last_error}"
    )
    return name, 1.0, f"skipped-after-retries: {last_error}"


class EvaluationAgent:
    """
    Evaluates generated content quality using DeepEval's GEval metric.

    Args:
        workflow_type: One of "story_topics", "story", "image", "audio", "activities".
                       Determines which evaluation criteria to use.
        pass_threshold: Score >= this value is considered passing. Default 0.6.
    """

    def __init__(self, workflow_type: str, pass_threshold: float = PASS_THRESHOLD):
        self.workflow_type = workflow_type
        self.pass_threshold = pass_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def evaluate(self, state: dict) -> dict:
        """
        Evaluates the content in state and returns an updated state dict with
        state["evaluation"] = {"passed": bool, "score": float, "reason": str}.

        For story_topics: also includes state["evaluation"]["metrics"] (per-dimension scores).
        """
        if self.workflow_type == "story_topics":
            return await self._evaluate_topics(state)
        if self.workflow_type == "story":
            return await self._evaluate_story(state)
        if self.workflow_type == "image":
            return await self._evaluate_image_prompt(state)
        if self.workflow_type == "audio":
            return await self._evaluate_audio(state)
        if self.workflow_type == "activities":
            return await self._evaluate_activities(state)
        return await self._evaluate_single(state)

    # ------------------------------------------------------------------
    # story_topics — 8 parallel GEval metrics
    # ------------------------------------------------------------------

    async def _evaluate_topics(self, state: dict) -> dict:
        topics = state.get("topics")
        if not topics:
            logger.warning("[story_topics] No topics found to evaluate.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No topics available for evaluation.",
                    "metrics": {},
                }
            }

        # Sampling strategy: LLM-judge ONE representative topic from the batch.
        # Rationale:
        #   - Topics in the same batch share the same prompt template, model, and
        #     generation pass — quality is highly correlated. Judging all N is
        #     redundant and 8 metrics × N topics overruns the evaluator model's
        #     rate limit. One sample gives a reliable quality signal.
        #   - `completeness` and `recall` still see the FULL list (Python checks
        #     below), so we don't lose coverage on the deterministic dimensions.
        # Pick the topic that's most likely to surface real issues: the longest
        # title+desc combined (deeper content = more material for the judge).
        sample_topic = max(
            topics,
            key=lambda t: len((t.get("title") or "") + (t.get("description") or "")),
        )

        age      = state.get("age", "3-4")
        language = state.get("language", "English")
        country  = state.get("country", "Any")
        religion = state.get("religion", "universal_wisdom")

        # ----------------------------------------------------------------
        # Eval-verdict cache: skip GEval entirely if this topic batch
        # already passed eval under the current rubric version.
        # Cache key derives from the sample topic's library doc fields
        # (theme, age, language-code, filter_value) so it's stable across
        # restarts. A failed verdict is NOT cached — re-eval gives the
        # corrector a fresh shot.
        # ----------------------------------------------------------------
        theme         = sample_topic.get("theme") or ""
        filter_value  = sample_topic.get("filter_value") or ""
        # Topics use the short language code in cache doc IDs (en/te); state
        # may still hold the long form ("English"). Strip to first two chars.
        lang_code     = (language or "en")[:2].lower()

        if theme and filter_value:
            # Lazy import to avoid pulling Firestore into modules that only
            # use the evaluator in unit tests.
            from ...services.database.firestore_service import FirestoreService
            firestore = FirestoreService()
            cached = await firestore.get_topic_eval_verdict(theme, age, lang_code, filter_value)
            if (
                cached
                and cached.get("passed") is True
                and cached.get("eval_version") == _EVAL_RUBRIC_VERSION
            ):
                logger.info(
                    f"[story_topics] Eval cache HIT (theme={theme}/{filter_value}) — "
                    f"skipping GEval. score={cached.get('score')}"
                )
                return {"evaluation": cached}

        title = sample_topic.get("title", "?")
        desc  = sample_topic.get("description", "?")
        request = (
            f"Generate one children's story topic for age {age} in {language} that fits "
            f"country '{country}' and wisdom/religion context '{religion}'. "
            f"It should have a vivid title and a short description hinting at the story."
        )
        test_case = LLMTestCase(
            input=request,
            actual_output=f"- {title}: {desc}",
        )

        sem = _get_eval_semaphore()

        llm_results = await asyncio.gather(
            *[
                _run_geval_with_retry(
                    name=n,
                    criteria=c,
                    test_case=test_case,
                    threshold=self.pass_threshold,
                    sem=sem,
                    is_hard=False,
                    log_prefix=f"[story_topics/{title}]",
                )
                for n, c in _TOPICS_CRITERIA.items()
            ]
        )

        metric_scores: dict[str, float] = {n: s for n, s, _ in llm_results}
        metric_reasons: dict[str, str] = {n: r for n, _, r in llm_results}

        # Python-computed deterministic metrics on the FULL topic list
        comp_score, comp_reason = _python_completeness(topics)
        rec_score, rec_reason   = _python_recall(topics)
        metric_scores["completeness"]  = comp_score
        metric_reasons["completeness"] = comp_reason
        metric_scores["recall"]        = rec_score
        metric_reasons["recall"]       = rec_reason

        avg_score = sum(metric_scores.values()) / len(metric_scores)
        passed = avg_score >= self.pass_threshold

        failed = [n for n, s in metric_scores.items() if s < self.pass_threshold]
        reason = (
            f"avg={avg_score:.3f} (sampled '{title}' from {len(topics)} topics). Failed: {failed}"
            if failed else
            f"avg={avg_score:.3f} (sampled '{title}' from {len(topics)} topics). All metrics passed."
        )

        logger.info(
            f"[story_topics] Evaluation {'PASSED' if passed else 'FAILED'} "
            f"avg={avg_score:.3f} metrics={metric_scores}"
        )

        verdict = {
            "passed": passed,
            "score": round(avg_score, 3),
            "reason": reason,
            "metrics": metric_scores,
            "metric_reasons": metric_reasons,
            "eval_version": _EVAL_RUBRIC_VERSION,
        }

        # Cache passing verdicts only — a failed verdict shouldn't pin the
        # corrector to its old answer on the next run.
        if passed and theme and filter_value:
            from ...services.database.firestore_service import FirestoreService
            firestore = FirestoreService()
            await firestore.save_topic_eval_verdict(theme, age, lang_code, filter_value, verdict)

        return {"evaluation": verdict}

    # ------------------------------------------------------------------
    # story — multi-metric GEval with hard/soft tiers
    # ------------------------------------------------------------------

    async def _evaluate_story(self, state: dict) -> dict:
        story = state.get("story") or {}
        story_text = story.get("story_text") or ""
        if not story_text.strip():
            logger.warning("[story] No story_text found to evaluate.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No story content available for evaluation.",
                    "metrics": {},
                }
            }

        topic = state.get("selected_topic") or {}
        age = state.get("age", "3-4")
        language = state.get("language", "English")

        # Reference (input) = the topic context the story was meant to deliver.
        # actual_output = the story + its required-field payload, so answer_relevance
        # can see whether moral/image_prompt/mcq_seeds/science_concepts/art_seed exist.
        topic_input = (
            f"Generate a children's story for age {age} in {language}.\n"
            f"Topic title: {topic.get('title', '?')}\n"
            f"Topic description: {topic.get('description', '?')}\n"
            f"Required moral: {topic.get('moral', '?')}\n"
            f"Story seed: {topic.get('story_seed', '?')}"
        )

        actual_output_parts = [f"STORY:\n{story_text}"]
        for field in ("moral", "image_prompt", "art_seed"):
            val = story.get(field)
            if val:
                actual_output_parts.append(f"{field.upper()}: {val}")
        if story.get("mcq_seeds"):
            actual_output_parts.append(f"MCQ_SEEDS: {story['mcq_seeds']}")
        if story.get("science_concepts"):
            actual_output_parts.append(f"SCIENCE_CONCEPTS: {story['science_concepts']}")
        actual_output = "\n\n".join(actual_output_parts)

        test_case = LLMTestCase(input=topic_input, actual_output=actual_output)
        sem = _get_eval_semaphore()

        results = await asyncio.gather(
            *[
                _run_geval_with_retry(
                    name=n,
                    criteria=c,
                    test_case=test_case,
                    threshold=self.pass_threshold,
                    sem=sem,
                    is_hard=False,
                    log_prefix="[story]",
                )
                for n, c in _STORY_CRITERIA.items()
            ]
        )
        metric_scores = {n: s for n, s, _ in results}
        metric_reasons = {n: r for n, _, r in results}

        # Python-checked age appropriateness: word/sentence length vs age band.
        age_score, age_reason = _python_age_appropriateness(story_text, age)
        metric_scores["age_appropriateness"] = age_score
        metric_reasons["age_appropriateness"] = age_reason

        soft_scores = [metric_scores[n] for n in _STORY_SOFT_METRICS if n in metric_scores]
        soft_avg = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
        passed = soft_avg >= self.pass_threshold

        reason = (
            f"soft-avg={soft_avg:.3f}"
            if passed else
            f"Soft-average {soft_avg:.3f} below threshold {self.pass_threshold}"
        )

        logger.info(
            f"[story] Evaluation {'PASSED' if passed else 'FAILED'} "
            f"soft_avg={soft_avg:.3f} metrics={metric_scores}"
        )

        return {
            "evaluation": {
                "passed": passed,
                "score": round(soft_avg, 3),
                "reason": reason,
                "metrics": metric_scores,
                "metric_reasons": metric_reasons,
            }
        }

    # ------------------------------------------------------------------
    # image — multi-metric GEval on the prompt sent to FLUX
    # ------------------------------------------------------------------

    async def _evaluate_image_prompt(self, state: dict) -> dict:
        image_prompt = state.get("image_prompt") or ""
        if not image_prompt.strip():
            logger.warning("[image] No image_prompt found to evaluate.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No image prompt available for evaluation.",
                    "metrics": {},
                }
            }

        age = state.get("age", "3-4")
        story_title = state.get("story_title") or ""
        story_text = state.get("story_text") or ""
        # Trim story_text so the judge gets enough signal for story_alignment without
        # blowing the context. First 400 chars is plenty for setting + opening scene.
        story_snippet = story_text[:400] + ("..." if len(story_text) > 400 else "")

        reference_input = (
            f"Children's story image for age {age}.\n"
            f"Story title: {story_title}\n"
            f"Story opening (truncated):\n{story_snippet}"
        )

        test_case = LLMTestCase(input=reference_input, actual_output=image_prompt)
        sem = _get_eval_semaphore()

        results = await asyncio.gather(
            *[
                _run_geval_with_retry(
                    name=n,
                    criteria=c,
                    test_case=test_case,
                    threshold=self.pass_threshold,
                    sem=sem,
                    is_hard=n in _IMAGE_HARD_METRICS,
                    log_prefix="[image]",
                )
                for n, c in _IMAGE_CRITERIA.items()
            ]
        )
        metric_scores = {n: s for n, s, _ in results}
        metric_reasons = {n: r for n, _, r in results}

        # Python-checked copyright safety: free, deterministic IP-token scan.
        cs_score, cs_reason = _python_copyright_safety(image_prompt)
        metric_scores["copyright_safety"] = cs_score
        metric_reasons["copyright_safety"] = cs_reason

        hard_failures = [
            (n, metric_scores[n], floor)
            for n, floor in _IMAGE_HARD_METRICS.items()
            if metric_scores.get(n, 0.0) < floor
        ]
        passed = not hard_failures

        hard_scores = [metric_scores[n] for n in _IMAGE_HARD_METRICS if n in metric_scores]
        hard_avg = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0

        if hard_failures:
            reason = "Hard-metric failures: " + ", ".join(
                f"{n}={s:.2f}<{floor}" for n, s, floor in hard_failures
            )
        else:
            reason = f"All hard metrics cleared; hard-avg={hard_avg:.3f}"

        logger.info(
            f"[image] Evaluation {'PASSED' if passed else 'FAILED'} "
            f"metrics={metric_scores}"
        )

        return {
            "evaluation": {
                "passed": passed,
                "score": round(hard_avg, 3),
                "reason": reason,
                "metrics": metric_scores,
                "metric_reasons": metric_reasons,
            }
        }

    # ------------------------------------------------------------------
    # audio — Python coverage/duration checks + GEval TTS-suitability checks
    # ------------------------------------------------------------------

    async def _evaluate_audio(self, state: dict) -> dict:
        story_text = state.get("story_text") or ""
        audio_bytes = state.get("audio_bytes")
        audio_timepoints = state.get("audio_timepoints")
        language = state.get("language", "English")
        age = state.get("age", "3-4")

        # --- Hard metrics: deterministic Python checks ---
        cov_score, cov_reason = _python_paragraph_coverage(story_text, audio_timepoints)
        bytes_score, bytes_reason = _python_audio_bytes_present(audio_bytes)
        dur_score, dur_reason = _python_duration_plausibility(story_text, audio_timepoints)
        intg_score, intg_reason = _python_paragraph_integrity(audio_timepoints)

        # Python-computed soft metrics (TTS-text quality)
        tts_score, tts_reason = _python_tts_friendliness(story_text)
        pacing_score, pacing_reason = _python_narration_pacing(story_text)

        metric_scores: dict[str, float] = {
            "paragraph_coverage":    cov_score,
            "audio_bytes_present":   bytes_score,
            "duration_plausibility": dur_score,
            "paragraph_integrity":   intg_score,
            "tts_friendliness":      tts_score,
            "narration_pacing":      pacing_score,
        }
        metric_reasons: dict[str, str] = {
            "paragraph_coverage":    cov_reason,
            "audio_bytes_present":   bytes_reason,
            "duration_plausibility": dur_reason,
            "paragraph_integrity":   intg_reason,
            "tts_friendliness":      tts_reason,
            "narration_pacing":      pacing_reason,
        }

        # If we have no story text at all, skip GEval — nothing meaningful to judge.
        if not story_text.strip():
            logger.warning("[audio] No story_text to evaluate TTS suitability.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No story_text available; cannot evaluate audio.",
                    "metrics": metric_scores,
                    "metric_reasons": metric_reasons,
                }
            }

        # --- Soft metrics: GEval text-suitability checks ---
        reference_input = (
            f"Story text intended for TTS narration. Language: {language}. Age: {age}."
        )
        test_case = LLMTestCase(input=reference_input, actual_output=story_text)
        sem = _get_eval_semaphore()

        # Audio LLM metrics are all soft (the hard tier is Python-computed
        # coverage/duration/integrity above), so skip-as-pass on transient eval
        # errors is safe — the safety gate doesn't depend on Gemini.
        soft_results = await asyncio.gather(
            *[
                _run_geval_with_retry(
                    name=n,
                    criteria=c,
                    test_case=test_case,
                    threshold=self.pass_threshold,
                    sem=sem,
                    is_hard=False,
                    log_prefix="[audio]",
                )
                for n, c in _AUDIO_CRITERIA.items()
            ]
        )
        for name, score, reason in soft_results:
            metric_scores[name] = score
            metric_reasons[name] = reason

        # --- Gating ---
        hard_failures = [
            (n, metric_scores[n], floor)
            for n, floor in _AUDIO_HARD_FLOORS.items()
            if metric_scores.get(n, 0.0) < floor
        ]
        soft_scores = [metric_scores[n] for n in _AUDIO_SOFT_METRICS if n in metric_scores]
        soft_avg = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
        soft_pass = soft_avg >= self.pass_threshold
        passed = (not hard_failures) and soft_pass

        if hard_failures:
            reason = "Hard-metric failures: " + ", ".join(
                f"{n}={s:.2f}<{floor}" for n, s, floor in hard_failures
            )
        elif not soft_pass:
            reason = f"Soft-average {soft_avg:.3f} below threshold {self.pass_threshold}"
        else:
            reason = f"All hard metrics cleared; soft-avg={soft_avg:.3f}"

        logger.info(
            f"[audio] Evaluation {'PASSED' if passed else 'FAILED'} "
            f"soft_avg={soft_avg:.3f} metrics={metric_scores}"
        )

        return {
            "evaluation": {
                "passed": passed,
                "score": round(soft_avg, 3),
                "reason": reason,
                "metrics": metric_scores,
                "metric_reasons": metric_reasons,
            }
        }

    # ------------------------------------------------------------------
    # activities — per-activity multi-metric GEval
    # ------------------------------------------------------------------

    async def _evaluate_activities(self, state: dict) -> dict:
        """Evaluate one or all activity types found in state["activities"].

        If state["activity_type"] is set, evaluate only that activity (used by the
        workflow's per-activity evaluate node). Otherwise evaluate all present
        activities and merge per-activity results."""
        activities = state.get("activities") or {}
        if not activities:
            logger.warning("[activities] No activities present to evaluate.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No activities available for evaluation.",
                    "per_activity": {},
                }
            }

        target = state.get("activity_type")
        types_to_eval = [target] if target else list(activities.keys())

        per_activity = {}
        for atype in types_to_eval:
            if atype not in activities:
                continue
            per_activity[atype] = await self._evaluate_one_activity(state, atype, activities[atype])

        # Aggregate: pass only if every evaluated activity passed
        scores = [r["score"] for r in per_activity.values() if r.get("score") is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        all_passed = bool(per_activity) and all(r.get("passed") for r in per_activity.values())
        failed_types = [t for t, r in per_activity.items() if not r.get("passed")]

        reason = (
            f"Activities passed: {list(per_activity.keys())}, avg={avg_score:.3f}"
            if all_passed else
            f"Activities failed: {failed_types}, avg={avg_score:.3f}"
        )

        score_summary = ", ".join(f"{t}={r.get('score')}" for t, r in per_activity.items())
        logger.info(
            f"[activities] Evaluation {'PASSED' if all_passed else 'FAILED'} "
            f"avg={avg_score:.3f} per_activity={{{score_summary}}}"
        )

        return {
            "evaluation": {
                "passed": all_passed,
                "score": round(avg_score, 3),
                "reason": reason,
                "per_activity": per_activity,
            }
        }

    async def _evaluate_one_activity(
        self, state: dict, activity_type: str, data
    ) -> dict:
        """Run the 7-metric activity rubric on a single activity."""
        story_text = state.get("story_text") or ""
        story_title = state.get("story_title") or ""
        age = state.get("age", "3-4")

        activity_text = _activity_to_text(activity_type, data)
        if not activity_text.strip():
            return {
                "passed": False,
                "score": 0.0,
                "reason": f"Activity '{activity_type}' has no content to evaluate.",
                "metrics": {},
                "metric_reasons": {},
            }

        story_snippet = story_text[:400] + ("..." if len(story_text) > 400 else "")
        reference_input = (
            f"Children's '{activity_type}' activity for age {age}, accompanying this story.\n"
            f"Story title: {story_title}\n"
            f"Story opening (truncated):\n{story_snippet}"
        )

        test_case = LLMTestCase(input=reference_input, actual_output=activity_text)
        sem = _get_eval_semaphore()

        # safety_of_execution is meaningful only for activities with physical materials.
        # MCQ is text-only; skip the metric entirely (saves one LLM call per MCQ eval).
        criteria_to_run = {
            n: c for n, c in _ACTIVITY_CRITERIA.items()
            if not (activity_type == "mcq" and n == "safety_of_execution")
        }

        results = await asyncio.gather(
            *[
                _run_geval_with_retry(
                    name=n,
                    criteria=c,
                    test_case=test_case,
                    threshold=self.pass_threshold,
                    sem=sem,
                    is_hard=n in _ACTIVITY_HARD_METRICS,
                    log_prefix=f"[activities/{activity_type}]",
                    # Activities use the higher-quota eval model — flash-lite
                    # 503s under the 28-call-per-WF5-pass burst.
                    eval_model=_GEMINI_ACTIVITIES_EVAL_MODEL,
                )
                for n, c in criteria_to_run.items()
            ]
        )
        metric_scores = {n: s for n, s, _ in results}
        metric_reasons = {n: r for n, _, r in results}

        # Python-checked age appropriateness on the flattened activity text.
        age_score, age_reason = _python_age_appropriateness(activity_text, age)
        metric_scores["age_appropriateness"] = age_score
        metric_reasons["age_appropriateness"] = age_reason

        # MCQ skips safety_of_execution by design — mark it skipped so downstream
        # consumers (UI / corrector) can tell the difference from a 0.0 fail.
        if activity_type == "mcq":
            metric_scores["safety_of_execution"] = 1.0
            metric_reasons["safety_of_execution"] = "Skipped: MCQ has no physical materials/steps."

        hard_failures = [
            (n, metric_scores[n], floor)
            for n, floor in _ACTIVITY_HARD_METRICS.items()
            if metric_scores.get(n, 0.0) < floor
        ]
        soft_scores = [metric_scores[n] for n in _ACTIVITY_SOFT_METRICS if n in metric_scores]
        soft_avg = sum(soft_scores) / len(soft_scores) if soft_scores else 0.0
        soft_pass = soft_avg >= self.pass_threshold
        passed = (not hard_failures) and soft_pass

        if hard_failures:
            reason = "Hard-metric failures: " + ", ".join(
                f"{n}={s:.2f}<{floor}" for n, s, floor in hard_failures
            )
        elif not soft_pass:
            reason = f"Soft-average {soft_avg:.3f} below threshold {self.pass_threshold}"
        else:
            reason = f"All hard metrics cleared; soft-avg={soft_avg:.3f}"

        return {
            "passed": passed,
            "score": round(soft_avg, 3),
            "reason": reason,
            "metrics": metric_scores,
            "metric_reasons": metric_reasons,
        }

    # ------------------------------------------------------------------
    # All other workflows — single GEval metric
    # ------------------------------------------------------------------

    async def _evaluate_single(self, state: dict) -> dict:
        content = self._extract_content(state)
        if content is None:
            logger.warning(f"[{self.workflow_type}] No content found to evaluate.")
            return {
                "evaluation": {
                    "passed": False,
                    "score": 0.0,
                    "reason": "No content available for evaluation.",
                }
            }

        criteria = _DEFAULT_CRITERIA.get(self.workflow_type)
        if criteria is None:
            logger.warning(f"[{self.workflow_type}] No criteria defined; skipping evaluation.")
            return {"evaluation": {"passed": True, "score": 0.0, "reason": "no criteria"}}
        prompt_context = state.get("story_text", state.get("selected_topic", ""))
        test_case = LLMTestCase(
            input=str(prompt_context),
            actual_output=str(content),
        )

        try:
            metric = GEval(
                name=f"{self.workflow_type}_quality",
                criteria=criteria,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                model=_GEMINI_EVAL_MODEL,
                threshold=self.pass_threshold,
            )
            await metric.a_measure(test_case)

            passed = metric.score >= self.pass_threshold
            result = {
                "passed": passed,
                "score": round(metric.score, 3),
                "reason": metric.reason or "",
            }
            logger.info(
                f"[{self.workflow_type}] Evaluation {'PASSED' if passed else 'FAILED'} "
                f"score={result['score']}"
            )
            return {"evaluation": result}

        except Exception as e:
            logger.error(f"[{self.workflow_type}] Evaluation error: {e}")
            return {
                "evaluation": {
                    "passed": True,
                    "score": 0.0,
                    "reason": f"Evaluation skipped due to error: {e}",
                }
            }

    def _extract_content(self, state: dict):
        """Extract the relevant content field from state depending on workflow type."""
        if self.workflow_type == "image":
            return state.get("image_prompt")
        if self.workflow_type == "audio":
            return state.get("story_text")
        return state.get("topics") or state.get("story")
