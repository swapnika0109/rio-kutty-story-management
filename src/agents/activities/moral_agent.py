import json
import uuid
from ...services.ai_service import AIService
from ...services.database.storage_bucket import StorageBucketService
from ...utils.logger import setup_logger
from ...prompts import get_registry
from . import _prepend_retry_feedback

logger = setup_logger(__name__)

class MoralAgent:
    def __init__(self, prompt_version: str = "latest"):
        self.ai_service = AIService()
        self.storage = StorageBucketService()
        self.prompt_version = prompt_version

    async def _gen_and_upload(self, prompt_text: str) -> str | None:
        """Generate an image and upload it to GCS. Returns the GCS filename
        (or None on failure) so raw PNG bytes never enter LangGraph state."""
        image_bytes = await self.ai_service.generate_image(prompt_text)
        if not image_bytes:
            return None
        filename = f"images/{uuid.uuid4()}.png"
        await self.storage.upload_file(filename, image_bytes, content_type="image/png")
        return filename

    async def generate(self, state: dict):
        logger.info("Starting Moral activity generation...")
        # Use moral (the story's moral lesson string) when available
        story = state.get("moral") or state.get("story_text", "")
        age = state.get("age", "3-4")
        language = state.get("language", "English")

        # Load prompt from registry
        registry = get_registry()
        prompt = registry.get_prompt(
            "moral",
            version=self.prompt_version,
            age=age,
            story=story,
            language=language
        )
        prompt = _prepend_retry_feedback(prompt, state, "moral")

        try:
            response = await self.ai_service.generate_content(prompt)
            # Response is a dict: {"text": "...", "images": [...]}
            
            # Robust JSON extraction
            start_index = response.find('[')
            end_index = response.rfind(']')
            if start_index != -1 and end_index != -1:
                cleaned_text = response[start_index:end_index+1]
            else:
                cleaned_text = response.replace("```json", "").replace("```", "").strip()

            activity_data = json.loads(cleaned_text)
            # Generate images and upload immediately so PNG bytes never enter
            # LangGraph state (Firestore checkpoints cap at 1 MB).
            if len(activity_data) >= 2:
                activity_data[0]["image"] = await self._gen_and_upload(
                    activity_data[0].get("image_generation_prompt", "")
                )
                activity_data[1]["image"] = await self._gen_and_upload(
                    activity_data[1].get("image_generation_prompt", "")
                )
            else:
                activity_data[0]["image"] = await self._gen_and_upload(
                    activity_data[0].get("image_generation_prompt", "")
                )

            
            # If images were generated, attach them
            # We assume the prompt asked for 1 image which corresponds to the activity
            # if response["images"]:
                # For now, just taking the first image data
                # Ideally, we should upload this to cloud storage and get a URL here
                # But per your current flow, we will pass the binary data or handle it in the Saver.
                # Let's store it in a separate 'images' key in state for the saver to handle.
                # pass 

            return {
                "activities": {**state.get("activities", {}), "moral": activity_data},
                "completed": state.get("completed", []) + ["moral"]
            }
        except Exception as e:
            logger.error(f"Moral Agent failed: {e}")
            return {"errors": {**state.get("errors", {}), "moral": str(e)}}



   
