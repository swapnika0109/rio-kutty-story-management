import json
import uuid
from ...services.ai_service import AIService
from ...services.database.storage_bucket import StorageBucketService
from ...utils.logger import setup_logger
from ...prompts import get_registry
from . import _prepend_retry_feedback

logger = setup_logger(__name__)

class ArtAgent:
    def __init__(self, prompt_version: str = "latest"):
        self.ai_service = AIService()
        self.storage = StorageBucketService()
        self.prompt_version = prompt_version

    async def generate(self, state: dict):
        logger.info("Starting Art activity generation...")
        # Use art_seed (concise art direction from story) when available
        summary = state.get("art_seed") or state.get("story_text", "")
        age = state.get("age", "3-4")
        language = state.get("language", "English")

        # Load prompt from registry
        registry = get_registry()
        prompt = registry.get_prompt(
            "art",
            version=self.prompt_version,
            age=age,
            summary=summary,
            language=language
        )
        prompt = _prepend_retry_feedback(prompt, state, "art")

        try:
            response = await self.ai_service.generate_content(prompt)
            try:
                activity_data = json.loads(response)
            except json.JSONDecodeError:
                # Robust JSON extraction
                start_index = response.find('[')
                end_index = response.rfind(']')
                if start_index != -1 and end_index != -1:
                    cleaned_text = response[start_index:end_index+1]
                else:
                    cleaned_text = response.replace("```json", "").replace("```", "").strip()
                activity_data = json.loads(cleaned_text)
            
            # Generate the image and upload immediately to GCS so raw bytes never
            # enter LangGraph state. Putting hundreds of KB of PNG bytes into state
            # is what blows past Firestore's 1 MB checkpoint cap when multiple
            # activities run in parallel.
            image_bytes = await self.ai_service.generate_image(activity_data.get("image_generation_prompt", ""))
            if image_bytes:
                filename = f"images/{uuid.uuid4()}.png"
                await self.storage.upload_file(filename, image_bytes, content_type="image/png")
                activity_data["image"] = filename
            else:
                activity_data["image"] = None
            return {
                "activities": {**state.get("activities", {}), "art": activity_data},
                "completed": state.get("completed", []) + ["art"]
            }
        except Exception as e:
            logger.error(f"Art Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "art": str(e)}}
