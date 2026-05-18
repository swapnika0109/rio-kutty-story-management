"""
SelfCorrectionAgent — targeted content improvement using evaluation feedback.

Why self-correction before full re-generation?
- Re-generation is expensive (full LLM call with cold context).
- Self-correction sends the ORIGINAL content + evaluation issues and asks the LLM
  to fix only the specific problems identified. This is cheaper and faster.
- Max 2 correction attempts are allowed per content item; after that the workflow
  propagates the failure up to the retry/human-in-loop layer.

The agent is generic — it works for story topics, stories, image prompts, etc.
The calling workflow passes the appropriate model_override and prompt_key.
"""

import json
from ...services.ai_service import AIService
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()


class SelfCorrectionAgent:
    """
    Improves content by asking the LLM to address specific evaluation issues.

    Args:
        model_override: Gemini model to use. Should match the workflow's primary model.
        fallback_override: Fallback model if primary fails.
    """

    def __init__(
        self,
        model_override: str = None,
        fallback_override: str = None,
    ):
        self.ai_service = AIService()
        self.model_override = model_override
        self.fallback_override = fallback_override

    async def correct(self, state: dict, content_key: str) -> dict:
        """
        Generates a corrected version of state[content_key] using evaluation feedback.

        Args:
            state: Current workflow state.
                   Must contain state["evaluation"] with "reason" field.
            content_key: The state key holding the content to correct
                         (e.g. "topics", "story", "image_prompt").

        Returns:
            Partial state update with corrected content and incremented correction_attempts.
        """
        original_content = state.get(content_key)
        evaluation = state.get("evaluation", {})
        issues = evaluation.get("reason", "The content did not meet quality standards.")
        attempts = state.get("correction_attempts", 0)

        if original_content is None:
            logger.warning(f"[SelfCorrection] No content found at key '{content_key}', skipping.")
            return {"correction_attempts": attempts + 1}

        prompt = self._build_correction_prompt(
            original_content,
            issues,
            content_key,
            metric_reasons=evaluation.get("metric_reasons") or {},
            selected_topic=state.get("selected_topic") or {},
            age=state.get("age"),
            language=state.get("language"),
        )

        try:
            response = await self.ai_service.generate_content(
                prompt,
                model_override=self.model_override,
                fallback_override=self.fallback_override,
            )
            corrected = self._parse_response(response, original_content)
            logger.info(f"[SelfCorrection] Corrected '{content_key}' (attempt {attempts + 1})")
            return {
                content_key: corrected,
                "correction_attempts": attempts + 1,
                # Reset evaluation so the next workflow node re-evaluates
                "evaluation": None,
                "validated": False,
            }
        except Exception as e:
            logger.error(f"[SelfCorrection] Failed to correct '{content_key}': {e}")
            return {
                "correction_attempts": attempts + 1,
                "errors": {**state.get("errors", {}), "self_correction": str(e)},
            }

    def _build_correction_prompt(
        self,
        content,
        issues: str,
        content_key: str,
        *,
        metric_reasons: dict | None = None,
        selected_topic: dict | None = None,
        age: str | None = None,
        language: str | None = None,
    ) -> str:
        """Build the self-correction prompt.

        For `story` we inject the original topic (so the corrector knows the *target*
        moral/conflict) and a per-metric failure list (so it fixes the specific
        dimensions that scored low, not 'everything')."""
        content_str = json.dumps(content, ensure_ascii=False, indent=2) if not isinstance(content, str) else content
        metric_reasons = metric_reasons or {}

        if content_key == "topics":
            format_hint = (
                "\n\nOUTPUT FORMAT — return ONLY pipe-separated lines, one topic per line, no extra text:\n"
                "title1|description1\n"
                "title2|description2\n\n"
                "Requirements for each corrected topic:\n"
                "- Title: max 6 words, MUST feature a named character or vivid image (NOT a theme name)\n"
                "- Description: 12–20 words, contains an action verb and hints at a feeling or lesson\n"
                "- Each title must cover a different situation or character (diverse, not repetitive)\n"
                "- Do NOT reuse the original title verbatim — rewrite it so a child wants to hear the story\n"
            )
            return (
                f"You are a content quality editor for children's educational stories.\n\n"
                f"The following {content_key} was evaluated and found to have these issues:\n"
                f"{issues}\n\n"
                f"Original content:\n{content_str}\n\n"
                f"Please provide an improved version that fixes ALL identified issues."
                f"{format_hint}"
                f"Do not add any explanation or preamble."
            )

        if content_key == "story":
            topic = selected_topic or {}
            failure_lines = "\n".join(
                f"- {name}: {reason}" for name, reason in metric_reasons.items() if reason
            ) or f"- {issues}"

            return (
                "You are a content quality editor for children's educational stories.\n\n"
                "ORIGINAL TOPIC (the story MUST deliver this moral and conflict — do not drift):\n"
                f"  Title: {topic.get('title', '?')}\n"
                f"  Description: {topic.get('description', '?')}\n"
                f"  Required moral: {topic.get('moral', '?')}\n"
                f"  Story seed: {topic.get('story_seed', '?')}\n"
                f"  Age: {age or '?'}    Language: {language or '?'}\n\n"
                "EVALUATION FAILURES (fix ONLY these — leave passing parts unchanged):\n"
                f"{failure_lines}\n\n"
                "ORIGINAL STORY JSON:\n"
                f"{content_str}\n\n"
                "INSTRUCTIONS:\n"
                "- Return the SAME JSON schema with keys: story, image_prompt, mcq_seeds, "
                "art_seed, science_concepts, moral.\n"
                "- If 'topic_fidelity' failed: the central conflict in the story body MUST "
                "be the situation in the Story seed (e.g. if seed is 'turtle afraid of "
                "deep water', the obstacle must be deep water, not a fallen branch). "
                "Rewrite the conflict to match the seed exactly, and set the closing moral "
                "to the Required moral above.\n"
                "- If 'narrative_coherence' failed: tighten cause-and-effect between scenes; "
                "every emotional shift must follow from a visible event.\n"
                "- If 'educational_value' failed: weave both science_concepts INTO the story "
                "body through a character's observation or action — not just the JSON field.\n"
                "- If 'answer_relevance' failed: fill any missing or generic field with "
                "content drawn from the story body.\n"
                "- If 'non_toxicity' failed: remove the specific unsafe word/imagery only; "
                "this metric is purely about safety, not topic drift.\n"
                "- Do NOT use copyrighted characters, settings, or plots.\n"
                "- Keep the 480–520 word length and the vocabulary level for the specified age.\n"
                "- Preserve anything that was not flagged as a failure.\n\n"
                "Return ONLY the JSON. No text before or after."
            )

        return (
            f"You are a content quality editor for children's educational stories.\n\n"
            f"The following {content_key} was evaluated and found to have these issues:\n"
            f"{issues}\n\n"
            f"Original content:\n{content_str}\n\n"
            f"Please provide an improved version that fixes ALL identified issues.\n"
            f"Do not add any explanation or preamble."
        )

    def _parse_response(self, response: str, original_content):
        """
        Parse the corrected response, falling back to original on parse error.
        If the original was JSON (dict/list), try to parse the response as JSON.
        If the original was a string, return the response as-is.
        """
        if isinstance(original_content, (dict, list)):
            try:
                cleaned = response.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                logger.warning("[SelfCorrection] Could not parse corrected response as JSON, using raw text.")
                return response

            # Story prompts return the body under "story"; downstream pipeline expects
            # "story_text". Mirror StoryCreatorAgent._parse_story so the corrected
            # output passes structural validation.
            if isinstance(parsed, dict) and "story" in parsed and "story_text" not in parsed:
                parsed["story_text"] = parsed.pop("story")
            if isinstance(original_content, dict) and isinstance(parsed, dict):
                for k in ("age_group", "language", "title"):
                    if not parsed.get(k) and original_content.get(k):
                        parsed[k] = original_content[k]
            return parsed
        return response
