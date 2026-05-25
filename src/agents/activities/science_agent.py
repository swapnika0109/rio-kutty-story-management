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

        # Load prompt from registry
        registry = get_registry()
        prompt = registry.get_prompt(
            "science",
            version=self.prompt_version,
            age=age,
            story=story,
            language=language
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
            # Generate image and upload immediately to GCS so PNG bytes never
            # enter LangGraph state (Firestore checkpoints cap at 1 MB).
            image_bytes = await self.ai_service.generate_image(science_data[0].get("image_generation_prompt", ""))
            if image_bytes:
                filename = f"images/{uuid.uuid4()}.png"
                await self.storage.upload_file(filename, image_bytes, content_type="image/png")
                science_data[0]["image"] = filename
            else:
                science_data[0]["image"] = None
            return {
                "activity_type": "science",
                "activities": {**state.get("activities", {}), "science": science_data},
                "completed": state.get("completed", []) + ["science"]
            }
        except Exception as e:
            logger.error(f"Science based activities generation failed: {str(e)}")
            return {
                "errors": {**state.get("errors", {}), "science": str(e)}
            }
