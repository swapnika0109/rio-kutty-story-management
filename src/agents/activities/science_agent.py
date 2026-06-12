from ...services.ai_service import AIService
from ...services.database.storage_bucket import StorageBucketService
from ...utils.logger import setup_logger
from ...prompts import get_registry
from . import _prepend_retry_feedback
import json
import uuid

logger = setup_logger(__name__)

class ScienceAgent:

    def __init__(self, prompt_version: str = "latest"):
        self.ai_service = AIService()
        self.storage = StorageBucketService()
        self.prompt_version = prompt_version
    
    async def generate(self, state: dict):
        """
        Generates Science based activities for the story.
        Uses science_concepts (list of {concept, explanation} dicts) from the story when available.
        Expected state: { "science_concepts": [...], "story_text": "...", "age": "3-4", ... }
        """
        logger.info("Starting Science based activities generation...")
        science_concepts = state.get("science_concepts") or []
        if science_concepts:
            story = "; ".join(
                f"{c.get('concept', '')}: {c.get('explanation', '')}"
                for c in science_concepts
                if isinstance(c, dict)
            )
        else:
            story = state.get("story_text", "")
        age = state.get("age", "3-4")
        language = state.get("language", "English")
        science_angle = state.get("science_angle", "")
        daily_life = state.get("daily_life_application", "")
        story_title = state.get("story_title", "")

        # Load prompt from registry
        registry = get_registry()
        prompt = registry.get_prompt(
            "science",
            version=self.prompt_version,
            age=age,
            story=story,
            language=language,
            science_angle=science_angle,
            daily_life_application=daily_life,
            story_title=story_title,
        )
        prompt = _prepend_retry_feedback(prompt, state, "science")

        try:
            response = await self.ai_service.generate_content(prompt)
            
            # Robust JSON extraction
            start_index = response.find('[')
            end_index = response.rfind(']')
            if start_index != -1 and end_index != -1:
                cleaned_text = response[start_index:end_index+1]
            else:
                cleaned_text = response.replace("```json", "").replace("```", "").strip()

            science_data = json.loads(cleaned_text)
            # Image generation is deferred to a post-evaluation node so we don't
            # burn FLUX credits on activities that fail eval and get regenerated.
            # NOTE: Don't add to "completed" here — only the save node should do that
            # after evaluation passes. The workflow uses "completed" to track which
            # activities have successfully passed eval, so marking completed at generation
            # breaks the retry-on-eval-fail logic.
            return {
                "activity_type": "science",
                "activities": {**state.get("activities", {}), "science": science_data}
            }
        except Exception as e:
            logger.error(f"Science based activities generation failed: {str(e)}")
            return {
                "errors": {**state.get("errors", {}), "science": str(e)}
            }

    async def generate_image(self, state: dict):
        """Generate + upload the science activity image. Runs only after the
        evaluation pass succeeds, so credits aren't spent on retried items."""
        activities = state.get("activities", {})
        science_data = activities.get("science")
        if not science_data:
            return {}
        try:
            image_bytes = await self.ai_service.generate_image(science_data[0].get("image_generation_prompt", ""))
            if image_bytes:
                filename = f"images/{uuid.uuid4()}.png"
                await self.storage.upload_file(filename, image_bytes, content_type="image/png")
                science_data[0]["image"] = filename
            else:
                science_data[0]["image"] = None
            return {"activities": {**activities, "science": science_data}}
        except Exception as e:
            logger.error(f"Science image generation failed: {e}")
            return {"errors": {**state.get("errors", {}), "science": str(e)}}
