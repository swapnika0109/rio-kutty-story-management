import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


def _strip_empty_env_vars() -> None:
    """Pydantic prefers env vars over .env file values, even when the env var is
    an empty string. That breaks runs invoked like `HF_TOKEN= pytest ...` where
    the user meant to inherit from .env. Drop empty env vars so Pydantic falls
    back to the .env file value."""
    for key in ("HF_TOKEN", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
        if key in os.environ and not os.environ[key].strip():
            del os.environ[key]


_strip_empty_env_vars()


class Settings(BaseSettings):
     # App Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Google Cloud
    GOOGLE_CLOUD_PROJECT : str = "riokutty"
    GOOGLE_CLOUD_BUCKET : str = "kutty_bucket"
    GOOGLE_API_KEY: str
    FIRESTORE_DATABASE: str = "(default)"
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None # Path to service account JSON
    
    # Cost & CO2 Optimization: AI Models
    # Default to Flash for speed and lower cost/CO2
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"       # WF5 activities (existing)
    GEMINI_FALLBACK_MODEL: str = "gemini-2.5-pro"   # WF5 fallback
    MULTIMODAL_MODEL: str = "gemini-2.5-flash-image"  # multimodal (existing)

    # WF1 — Story Topics (cheap model, topics are short/simple)
    STORY_TOPICS_MODEL: str = "gemini-2.5-flash-lite"
    STORY_TOPICS_FALLBACK_MODEL: str = "gemini-2.5-pro"

    # WF2 — Story Creator (higher quality model for narrative generation)
    STORY_CREATOR_MODEL: str = "gemini-2.5-flash-lite"
    STORY_CREATOR_FALLBACK_MODEL: str = "gemini-2.5-pro"

    # Evaluation model — always cheap; GEval scoring is a simpler task.
    # Per-workflow overrides below for workflows that fan out many concurrent
    # metric calls and overload the flash-lite quota.
    EVALUATION_MODEL: str = "gemini-2.5-flash-lite"

    # WF5 activities run 7 metrics × 4 activities = up to 28 GEval calls per
    # pipeline pass, each translating to 2-3 Gemini calls under the hood.
    # flash-lite throttles at this volume — use higher-quota flash here.
    ACTIVITIES_EVALUATION_MODEL: str = "gemini-2.5-flash"

    # WF3 — Image generation via HuggingFace InferenceClient
    # FLUX.1-schnell: 4 inference steps (vs 50 for dev), ~10x cheaper, Apache-2.0
    FLUX_IMAGE_MODEL: str = "black-forest-labs/FLUX.1-schnell"

    # Cost Optimization: Limits
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 2
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: int = 60
    RATE_LIMIT_TOKENS_PER_SECOND: float = 3.0
    RATE_LIMIT_BURST_CAPACITY: int = 6

    # Parallel workflow retries before escalating to human-in-the-loop.
    # Set to 4 to match the architectural plan: activities get 4 attempts to pass
    # evaluation before escalating to human-in-the-loop. With metric-specific retry
    # feedback, agents can target actual failures (e.g., "engagability too low" with
    # reason text), making retries genuinely corrective rather than blind re-rolls.
    PARALLEL_WORKFLOW_MAX_RETRIES: int = 4

    # Pub/Sub topic to notify admin when a parallel workflow needs human review
    # Format: projects/{project_id}/topics/{topic_name}
    HUMAN_LOOP_NOTIFICATION_TOPIC: str = "projects/riokutty/topics/story-agent"

    # WF4 — Audio (Google Cloud Text-to-Speech)
    TTS_LANGUAGE_CODE: str = "en-US"
    TTS_VOICE_NAME: str = "en-US-Standard-A"
    TTS_AUDIO_ENCODING: str = "MP3"

    # Prompt versioning (per agent)
    MCQ_PROMPT_VERSION: str = "latest"
    ART_PROMPT_VERSION: str = "latest"
    MORAL_PROMPT_VERSION: str = "latest"
    SCIENCE_PROMPT_VERSION: str = "latest"
    STORY_TOPICS_PROMPT_VERSION: str = "latest"
    STORY_CREATOR_PROMPT_VERSION: str = "latest"
    IMAGE_GENERATOR_PROMPT_VERSION: str = "latest"
    EVALUATION_PROMPT_VERSION: str = "latest"
    SELF_CORRECTION_PROMPT_VERSION: str = "latest"

    # WF1 — number of high-level topic names extracted per theme and sent to each prompt
    TOPICS_PER_THEME: int = 1

    # Langfuse — open-source LLM observability (free cloud tier: cloud.langfuse.com)
    # Set LANGFUSE_ENABLED=true and provide keys to activate tracing.
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"  # or your self-hosted URL

    # Performance & Scaling
    MAX_CONCURRENCY: int = 10
    HF_TOKEN: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache()
def get_settings():
    return Settings()