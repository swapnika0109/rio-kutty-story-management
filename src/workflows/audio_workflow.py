"""
WF4 — Audio Generator Workflow (compiled subgraph)

Flow:
    [generate_audio] → [validate_audio] → [evaluate_audio]
                                               ↓
                                 pass → [save_audio] → END (status="completed")
                                 fail → retry_count < 4?
                                           ↓ yes           ↓ no
                                      [generate_audio]   END (status="needs_human")

Single audio file per story in the story's requested language.
Triggered by master_workflow via asyncio.gather alongside WF3 and WF5.
"""

import uuid
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from ..models.state import AudioWorkflowState
from ..agents.media.audio_generator_agent import AudioGeneratorAgent
from ..agents.validators.evaluation_agent import EvaluationAgent
from ..services.database.firestore_service import FirestoreService
from ..services.database.storage_bucket import StorageBucketService
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()

MAX_RETRIES = settings.PARALLEL_WORKFLOW_MAX_RETRIES

# --- Component instances ---
audio_agent = AudioGeneratorAgent()
evaluator = EvaluationAgent(workflow_type="audio")
firestore = FirestoreService()
storage = StorageBucketService()


def _unpack_config(state: AudioWorkflowState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    return {
        **state,
        "story_id": cfg.get("story_id"),
        # Language from initial state (set by master from story data), fallback to config
        "language": state.get("language") or cfg.get("language", settings.TTS_LANGUAGE_CODE),
        "voice": state.get("voice") or settings.TTS_VOICE_NAME,
    }


# --- Nodes ---

async def generate_audio_node(state: AudioWorkflowState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    result = await audio_agent.generate(enriched)
    current = state.get("retry_count", 0)
    result["retry_count"] = current + 1
    return result


async def validate_audio_node(state: AudioWorkflowState, config: RunnableConfig) -> dict:
    """Structural validation: audio_bytes must be non-empty bytes."""
    audio_bytes = state.get("audio_bytes")
    if not audio_bytes or not isinstance(audio_bytes, bytes):
        logger.warning("[WF4] Structural validation failed: audio_bytes missing or empty")
        return {"validated": False}
    logger.info(f"[WF4] Structural validation passed: {len(audio_bytes)} bytes")
    return {"validated": True}


async def evaluate_audio_node(state: AudioWorkflowState, config: RunnableConfig) -> dict:
    """Evaluate the story text suitability for audio (not the audio bytes themselves)."""
    enriched = _unpack_config(state, config)
    return await evaluator.evaluate(enriched)


async def save_audio_node(state: AudioWorkflowState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    story_id = cfg.get("story_id")
    theme = cfg.get("theme", "theme1")
    audio_bytes = state.get("audio_bytes")
    audio_timepoints = state.get("audio_timepoints")
    language = state.get("language", settings.TTS_LANGUAGE_CODE)
    voice = state.get("voice", settings.TTS_VOICE_NAME)

    # Always WAV — paragraphs are combined as WAV regardless of TTS_AUDIO_ENCODING
    filename = f"story-audio/{uuid.uuid4()}.wav"
    audio_url = await storage.upload_file(filename, audio_bytes, content_type="audio/wav")

    if not audio_url:
        logger.error(f"[WF4] GCS upload failed for story_id={story_id}")
        return {
            "errors": {**state.get("errors", {}), "save_audio": "GCS upload returned no URL"},
            "status": "needs_human",
        }

    try:
        await firestore.save_story_audio(story_id, audio_url, language, voice, theme, audio_timepoints)
        logger.info(f"[WF4] Audio saved: {audio_url}")
        return {
            "audio_url":        audio_url,
            "audio_timepoints": audio_timepoints,
            "audio_bytes":      None,   # clear binary from checkpoint — already in GCS
            "completed":        ["audio"],
            "status":           "completed",
        }
    except Exception as e:
        logger.error(f"[WF4] Firestore update failed: {e}")
        return {
            "errors": {**state.get("errors", {}), "save_audio": str(e)},
            "status": "needs_human",
        }


# --- Routing ---

def route_after_validate(state: AudioWorkflowState) -> Literal["evaluate_audio", "generate_audio", "__end__"]:
    if state.get("validated"):
        return "evaluate_audio"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return END
    return "generate_audio"


def route_after_evaluate(
    state: AudioWorkflowState,
) -> Literal["save_audio", "generate_audio", "__end__"]:
    evaluation = state.get("evaluation") or {}
    if evaluation.get("passed"):
        return "save_audio"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return END
    return "generate_audio"


async def mark_needs_human_node(state: AudioWorkflowState, config: RunnableConfig) -> dict:
    logger.error(f"[WF4] Max retries ({MAX_RETRIES}) exhausted — needs human review")
    return {
        "status": "needs_human",
        "errors": {
            **state.get("errors", {}),
            "audio_workflow": f"Failed after {MAX_RETRIES} retries",
        },
    }


# --- Graph ---

workflow = StateGraph(AudioWorkflowState)

workflow.add_node("generate_audio", generate_audio_node)
workflow.add_node("validate_audio", validate_audio_node)
workflow.add_node("evaluate_audio", evaluate_audio_node)
workflow.add_node("save_audio", save_audio_node)
workflow.add_node("mark_needs_human", mark_needs_human_node)

workflow.set_entry_point("generate_audio")
workflow.add_edge("generate_audio", "validate_audio")
workflow.add_conditional_edges(
    "validate_audio",
    route_after_validate,
    {
        "evaluate_audio": "evaluate_audio",
        "generate_audio": "generate_audio",
        END: "mark_needs_human",
    },
)
workflow.add_conditional_edges(
    "evaluate_audio",
    route_after_evaluate,
    {
        "save_audio": "save_audio",
        "generate_audio": "generate_audio",
        END: "mark_needs_human",
    },
)
workflow.add_edge("save_audio", END)
workflow.add_edge("mark_needs_human", END)

# WF4 does not need persistent checkpointing — it runs in seconds, stores no
# human-interrupt state, and binary audio_bytes would exceed Firestore's 1 MB
# document limit.  MemorySaver is used so LangGraph's internal graph machinery
# still works, but nothing is written to Firestore.
audio_workflow = workflow.compile(checkpointer=MemorySaver())
