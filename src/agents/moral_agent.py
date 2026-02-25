import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger
from ..prompts import get_registry

logger = setup_logger(__name__)

class MoralAgent:
    def __init__(self, prompt_version: str = "latest"):
        self.ai_service = AIService()
        self.prompt_version = prompt_version

    async def generate(self, state: dict):
        logger.info("Starting Moral activity generation...")
        story = state.get("story_text", "")
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
            if len(activity_data) >= 2:
                image1 = await self.ai_service.generate_image(activity_data[0].get("image_generation_prompt", ""))
                activity_data[0]["image"] = image1
                image2 = await self.ai_service.generate_image(activity_data[1].get("image_generation_prompt", ""))
                activity_data[1]["image"] = image2
            else:
                image = await self.ai_service.generate_image(activity_data[0].get("image_generation_prompt", ""))
                activity_data[0]["image"] = image

            
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



   
