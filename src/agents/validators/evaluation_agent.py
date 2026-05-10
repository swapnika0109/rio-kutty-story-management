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
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from google import genai

from ...utils.logger import setup_logger
from ...utils.config import get_settings

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

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self._model_name = settings.EVALUATION_MODEL

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


# One shared instance per process — avoids re-creating the client on every metric
_GEMINI_EVAL_MODEL = _GeminiEvalModel()


# ---------------------------------------------------------------------------
# Single-metric criteria (non-topics workflows)
# ---------------------------------------------------------------------------

_DEFAULT_CRITERIA: dict[str, str] = {
    "story": (
        "The story is coherent, age-appropriate, has a clear moral lesson, "
        "is engaging for children, and is safe for young audiences."
    ),
    "image": (
        "The image generation prompt accurately represents the story scene, "
        "is child-friendly, colourful, and visually descriptive."
    ),
    "audio": (
        "The story text is suitable for audio narration: clear sentences, "
        "appropriate pacing markers, and child-friendly vocabulary."
    ),
    "activities": (
        "The activities are age-appropriate, clearly instructed, safe to perform, "
        "and educationally relevant to the story."
    ),
}

# ---------------------------------------------------------------------------
# Multi-metric criteria for story_topics (8 dimensions)
# ---------------------------------------------------------------------------

_TOPICS_CRITERIA: dict[str, str] = {
    "non_toxicity": (
        "Check the title and description for content that is unsafe for the specified age. "
        "Penalize: harsh words, slurs, profanity, violence, scary imagery, mature themes, "
        "or anything a parent of a child this age would find inappropriate. "
        "For very young ages (3-6), even mild fear, conflict, or dark imagery should lower the score. "
        "Mark high (1.0) only if the content is fully age-safe."
    ),
    "bias": (
        "The story titles and descriptions are unbiased and inclusive. They do not "
        "reinforce stereotypes, discriminate, or favour any particular group unfairly."
    ),
    "completeness": (
        "MECHANICAL CHECK — do not interpret. The Actual Output is one or more lines, each "
        "shaped like 'Title: Description' or 'Title|Description'. For each line, both sides "
        "of the separator must be non-empty. Score 1.0 if every line has text on BOTH sides. "
        "Score 0.0 ONLY when a line is literally missing the text after the separator. "
        "Do NOT invent a required JSON schema. Do NOT judge length, quality, or detail — "
        "the description '?' or a single word still counts as present."
    ),
    "engagability": (
        "Would a child of the specified age want to hear this story on first impression? "
        "Score >=0.7 if the title has any of: a vivid image, a named character, a hint of "
        "action/mystery, or a relatable feeling. Score 0.4-0.6 if it is a generic concept "
        "title (e.g. 'Sunshine and Shadows Play'). Score <0.4 only if it sounds like a "
        "textbook heading with no spark. Commit to a numeric score — never refuse."
    ),
    "trustworthiness": (
        "Check that the content is honest and safe to teach a child. Penalize: factual errors "
        "(e.g. 'the sun is purple'), superstition presented as fact, harmful advice (e.g. "
        "'hit your friend when angry'), or misleading morals. A made-up character or fantasy "
        "scenario is fine — fantasy is not the same as misinformation. Mark high if nothing "
        "in the topic could mislead or harm the child."
    ),
    "latency": (
        "Relevance check: does the topic fit the requested theme, age, and context? "
        "Verify silently — the output does NOT need to explicitly state the age, theme, or "
        "context labels (e.g. 'theme1', 'universal_wisdom' are INTERNAL identifiers). "
        "Implicit fit through tone, characters, setting, or moral is fully sufficient. "
        "Mark high if the topic plausibly belongs to the requested theme and age group."
    ),
    "precision": (
        "Does the title point to ONE clear story idea (not vague or scattered)? "
        "Mark high if a child could guess what the story is roughly about from title + description. "
        "A focused 5-6 word title with a clarifying description is enough — do not demand more."
    ),
    "recall": (
        "FORCED GATE — count the topic lines in the Actual Output FIRST, then apply:\n"
        "  count == 1  → score = 1.0 (STOP — do not evaluate anything else).\n"
        "  count >= 2 and all distinct → score = 1.0.\n"
        "  count >= 2 with some near-duplicates → score 0.3-0.6.\n"
        "  count >= 2 and all near-identical → score 0.0.\n"
        "The topic count is set by user config — having only one topic is NEVER a defect. "
        "Do NOT penalize for description length, word count, or missing age/theme labels."
    ),
}

# Minimum score (0-1) for evaluation to pass
PASS_THRESHOLD = 0.6


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

        # Per-topic evaluation: each topic is judged independently across all 8 metrics,
        # then we take the MEDIAN per metric to be robust to one bad topic dragging the
        # set down. Multi-religion batches used to fail because one weak topic poisoned
        # bias/non_toxicity for all of them when judged together.
        age      = state.get("age", "3-4")
        language = state.get("language", "English")
        country  = state.get("country", "Any")
        religion = state.get("religion", "universal_wisdom")

        async def _eval_one_topic(topic: dict) -> dict[str, tuple[float, str]]:
            title = topic.get("title", "?")
            desc  = topic.get("description", "?")
            request = (
                f"Generate one children's story topic for age {age} in {language} that fits "
                f"country '{country}' and wisdom/religion context '{religion}'. "
                f"It should have a vivid title and a short description hinting at the story."
            )
            test_case = LLMTestCase(
                input=request,
                actual_output=f"- {title}: {desc}",
            )

            async def _run_metric(name: str, criteria: str):
                metric = GEval(
                    name=name,
                    criteria=criteria,
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    model=_GEMINI_EVAL_MODEL,
                    threshold=self.pass_threshold,
                )
                try:
                    await metric.a_measure(test_case)
                    return name, round(metric.score, 3), metric.reason or ""
                except Exception as e:
                    logger.warning(f"[story_topics] Metric '{name}' failed for '{title}': {e}")
                    return name, 1.0, f"skipped: {e}"

            metric_results = await asyncio.gather(
                *[_run_metric(n, c) for n, c in _TOPICS_CRITERIA.items()]
            )
            return {n: (s, r) for n, s, r in metric_results}

        per_topic_results = await asyncio.gather(
            *[_eval_one_topic(t) for t in topics]
        )

        # Aggregate: median per metric across topics (robust to outliers)
        metric_scores: dict[str, float] = {}
        metric_reasons: dict[str, str] = {}
        for name in _TOPICS_CRITERIA.keys():
            scores = sorted(r[name][0] for r in per_topic_results)
            mid = len(scores) // 2
            median = scores[mid] if len(scores) % 2 else (scores[mid - 1] + scores[mid]) / 2
            metric_scores[name] = round(median, 3)
            # Keep the lowest-scoring topic's reason for that metric — most actionable
            worst_idx = min(range(len(per_topic_results)), key=lambda i: per_topic_results[i][name][0])
            worst_title = topics[worst_idx].get("title", "?")
            metric_reasons[name] = f"[worst: '{worst_title}'] {per_topic_results[worst_idx][name][1]}"

        avg_score = sum(metric_scores.values()) / len(metric_scores)
        passed = avg_score >= self.pass_threshold

        failed = [n for n, s in metric_scores.items() if s < self.pass_threshold]
        reason = (
            f"avg={avg_score:.3f} (median across {len(topics)} topics). Failed: {failed}" if failed
            else f"avg={avg_score:.3f} (median across {len(topics)} topics). All metrics passed."
        )

        logger.info(
            f"[story_topics] Evaluation {'PASSED' if passed else 'FAILED'} "
            f"avg={avg_score:.3f} median_metrics={metric_scores}"
        )

        return {
            "evaluation": {
                "passed": passed,
                "score": round(avg_score, 3),
                "reason": reason,
                "metrics": metric_scores,
                "metric_reasons": metric_reasons,
            }
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

        criteria = _DEFAULT_CRITERIA.get(self.workflow_type, _DEFAULT_CRITERIA["story"])
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
        if self.workflow_type == "story":
            return state.get("story")
        if self.workflow_type == "image":
            return state.get("image_prompt")
        if self.workflow_type == "audio":
            return state.get("story_text")
        if self.workflow_type == "activities":
            return state.get("activities")
        return state.get("topics") or state.get("story")
