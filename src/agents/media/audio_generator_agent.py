"""
AudioGeneratorAgent — generates WAV narration for a children's story.

Splits the story into paragraphs, synthesizes each separately via Google TTS,
records per-paragraph timestamps, then combines into a single WAV file.

Voice and language are read from config (TTS_VOICE_NAME, TTS_LANGUAGE_CODE)
but can be overridden via the workflow config.
"""

from ...services.audio_service import AudioService
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()

# Display name → BCP-47 language code accepted by Google TTS
_LANG_TO_BCP47: dict[str, str] = {
    "english": "en-US",
    "telugu":  "te-IN",
    "en":      "en-US",
    "te":      "te-IN",
    "en-us":   "en-US",
    "te-in":   "te-IN",
}


def _to_bcp47(language: str) -> str:
    """Normalise any language value the system uses to a Google TTS BCP-47 code."""
    return _LANG_TO_BCP47.get(language.lower(), language)


def _split_paragraphs(text: str) -> list[str]:
    """Split story text into non-empty paragraphs on blank lines."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


class AudioGeneratorAgent:
    def __init__(self):
        self.audio_service = AudioService()

    async def generate(self, state: dict) -> dict:
        """
        Generates audio narration from story text split into paragraphs.

        Expected state fields:
            story_text: str — full story text to narrate
            language: str  — BCP-47 code (e.g. "en-US", "ta-IN", "hi-IN")
            voice: str     — TTS voice name (optional, falls back to config)

        Returns partial state update with:
            audio_bytes       — combined WAV bytes (or None on failure)
            audio_timepoints  — list of per-paragraph timing dicts (or None on failure)
        """
        story_text = state.get("story_text", "")
        language   = _to_bcp47(state.get("language", settings.TTS_LANGUAGE_CODE))
        voice      = state.get("voice", settings.TTS_VOICE_NAME)

        if not story_text:
            logger.warning("[AudioGenerator] Empty story_text, skipping audio generation")
            return {"errors": {**state.get("errors", {}), "audio_generator": "story_text is empty"}}

        paragraphs = _split_paragraphs(story_text)
        if not paragraphs:
            logger.warning("[AudioGenerator] No paragraphs found after splitting")
            return {"errors": {**state.get("errors", {}), "audio_generator": "no paragraphs found"}}

        logger.info(f"[AudioGenerator] Synthesizing {len(paragraphs)} paragraph(s) lang={language}")

        audio_bytes, audio_timepoints = await self.audio_service.synthesize_paragraphs(
            paragraphs=paragraphs,
            language_code=language,
            voice_name=voice,
        )

        if audio_bytes is None:
            logger.warning("[AudioGenerator] TTS returned None")
            return {
                "audio_bytes": None,
                "audio_timepoints": None,
                "errors": {**state.get("errors", {}), "audio_generator": "TTS returned None"},
            }

        logger.info(f"[AudioGenerator] Generated audio: {len(audio_bytes)} bytes, {len(audio_timepoints)} paragraphs")
        return {
            "audio_bytes": audio_bytes,
            "audio_timepoints": audio_timepoints,
            "validated": False,
            "evaluation": None,
        }
