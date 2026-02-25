import json
from ..services.ai_service import AIService
from ..utils.logger import setup_logger
from ..prompts import get_registry

logger = setup_logger(__name__)

class MCQAgent:
    def __init__(self, prompt_version: str = "latest"):
        self.ai_service = AIService()
        self.prompt_version = prompt_version


    async def generate(self, state: dict):
        """
        Generates MCQs based on the story summary.
        Expected state: { "story_text": "...", "age": 5, ... }
        """
        logger.info("Starting MCQ generation...")
        summary = state.get("story_text", "")
        age = state.get("age", "3-4")
        language = state.get("language", "English")
        
        # Load prompt from registry
        registry = get_registry()
        prompt = registry.get_prompt(
            "mcq",
            version=self.prompt_version,
            age=age,
            summary=summary,
            language=language
        )

        try:
            response = await self.ai_service.generate_content(prompt)
            cleaned_text = response.replace("```json", "").replace("```", "").strip()
            mcq_data = json.loads(cleaned_text)

            return {
                "activity_type": "mcq",
                "activities": {**state.get("activities", {}), "mcq": mcq_data},
                "completed": state.get("completed", []) + ["mcq"]
            }
        except Exception as e:
            logger.error(f"MCQ generation failed: {str(e)}")
            return {
                "errors": {**state.get("errors", {}), "mcq": str(e)}
            }
