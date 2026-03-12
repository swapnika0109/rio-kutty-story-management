"""
Master Workflow — orchestrates the full story pipeline.

Execution order:
    1. [dispatch_media]     — WF3 (image) + WF4 (audio) in parallel via asyncio.gather
    2. [collect_media]      — check results; escalate HITL if any failed after 4 retries
       └→ needs_human → [handle_media_decision] → admin resumes with retry/skip/override
    3. [dispatch_activities] — WF5 (MCQ/Art/Science/Moral) with seeds from story JSON
    4. [collect_activities] — check WF5 result; HITL if needed
       └→ needs_human → [handle_activities_decision]
    5. [finalize]           — cleanup Firestore checkpoints, mark pipeline done

Why activities run AFTER image+audio:
- Activities use seeds (mcq_seeds, art_seed, science_concepts, moral) from the story.
- Image and audio must be saved first so the story record is complete before activities
  are linked to it.

Human-in-loop per workflow type:
- Image/audio failures use LangGraph interrupt() in collect_media.
  Admin can also bypass HITL by calling POST /generate-image/{story_id} or
  POST /generate-audio/{story_id} directly, then resuming the master with 'override'.
- Activity failures use LangGraph interrupt() in collect_activities.

Checkpoint cleanup:
- On successful finalization, all sub-thread checkpoints are deleted from Firestore
  since completed workflow state is persisted in the story documents.
"""

import asyncio
import json
import os
import random
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from google.cloud import pubsub_v1

from ..models.state import MasterWorkflowState
from ..workflows.image_workflow import image_workflow
from ..workflows.audio_workflow import audio_workflow
from ..workflows.activity_workflow import app_workflow as activity_workflow
from ..services.database.firestore_service import FirestoreService
from ..services.database.checkpoint_service import FirestoreCheckpointer
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()

firestore = FirestoreService()

# ---------------------------------------------------------------------------
# Voice pool — each story gets a randomly selected voice so the UI sounds
# different on every "next" swipe.
#
# Request JSON sends voice_type = "chirp" | "standard" (default: "standard").
# Full voice name is built as:  {lang_prefix}-{country}{suffix}
# e.g.  en-US-Standard-A  or  en-US-Chirp3-HD-Gacrux
# ---------------------------------------------------------------------------

# Language display name → (BCP-47 prefix, country code)
_LANG_MAP: dict[str, tuple[str, str]] = {
    "english": ("en", "US"),
    "telugu":  ("te", "IN"),
    "en":      ("en", "US"),
    "te":      ("te", "IN"),
}

_CHIRP_SUFFIXES: list[str] = [
    "-Chirp3-HD-Gacrux",
    "-Chirp3-HD-Callirrhoe",
    "-Chirp3-HD-Despina",
    "-Chirp3-HD-Iapetus",
    "-Chirp3-HD-Leda",
    "-Chirp3-HD-Zephyr",
    "-Chirp3-HD-Schedar",
    "-Chirp3-HD-Sadaltager",
    "-Chirp3-HD-Rasalgethi",
    "-Chirp3-HD-Umbriel",
    "-Chirp3-HD-Pulcherrima",
    "-Chirp3-HD-Charon",
    "-Chirp3-HD-Zubenelgenubi",
    "-Chirp3-HD-Achird",
    "-Chirp3-HD-Algenib",
    "-Chirp3-HD-Algieba",
    "-Chirp3-HD-Erinome",
]

_STANDARD_SUFFIXES: list[str] = [
    "-Standard-A",
    "-Standard-B",
    "-Standard-C",
    "-Standard-D",
]


def _pick_voice(language: str, voice_type: str | None) -> str:
    """
    Build a full BCP-47 voice name for the given language and voice type.

    voice_type: "chirp" → random Chirp3-HD voice
                "standard" | None → random Standard voice

    Full name format: {lang_prefix}-{country}{suffix}
    e.g. en-US-Standard-A  or  en-US-Chirp3-HD-Gacrux
    """
    lang_prefix, country = _LANG_MAP.get(language.lower(), ("en", "US"))
    suffixes = _CHIRP_SUFFIXES if (voice_type or "").lower() == "chirp" else _STANDARD_SUFFIXES
    suffix = random.choice(suffixes)
    return f"{lang_prefix}-{country}{suffix}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sub_thread_id(story_id: str, wf: str) -> str:
    """Each subgraph gets its own thread_id so their checkpoints don't collide."""
    return f"{story_id}_{wf}"


def _build_media_config(
    story_id: str, wf: str, story: dict, age: str, language: str, theme: str
) -> dict:
    """Build config.configurable for WF3/WF4 subgraph invocation."""
    return {
        "configurable": {
            "thread_id":   _sub_thread_id(story_id, wf),
            "story_id":    story_id,
            "story_text":  story.get("story_text", ""),
            "story_title": story.get("title", ""),
            "age":         age,
            "language":    language,
            "theme":       theme,
        }
    }


def _build_activities_config(
    story_id: str, story: dict, age: str, language: str
) -> dict:
    """Build config.configurable for WF5 subgraph invocation with story seeds."""
    return {
        "configurable": {
            "thread_id":        _sub_thread_id(story_id, "wf5"),
            "story_id":         story_id,
            "story_text":       story.get("story_text", ""),
            "age":              age,
            "language":         language,
            # Activity seeds from story JSON
            "mcq_seeds":        story.get("mcq_seeds", []),
            "art_seed":         story.get("art_seed", ""),
            "science_concepts": story.get("science_concepts", []),
            "moral":            story.get("moral", ""),
        }
    }


def _publish_hitl_notification(story_id: str, failed_workflows: list[dict], phase: str) -> None:
    """
    Publishes a Pub/Sub message to notify admin of workflows needing human review.
    Non-blocking — if publish fails, we log and continue (interrupt() still fires).
    """
    topic = settings.HUMAN_LOOP_NOTIFICATION_TOPIC
    if not topic:
        logger.warning("[Master] HUMAN_LOOP_NOTIFICATION_TOPIC not configured — skipping Pub/Sub")
        return

    try:
        publisher = pubsub_v1.PublisherClient()
        data = json.dumps({
            "story_id":        story_id,
            "phase":           phase,
            "failed_workflows": failed_workflows,
            "action_required": "Review and resume via POST /resume-workflow",
        }).encode("utf-8")
        future = publisher.publish(topic, data)
        future.result(timeout=5)
        logger.info(f"[Master] HITL notification published to {topic} (phase={phase})")
    except Exception as e:
        logger.error(f"[Master] Failed to publish HITL notification: {e}")


def _collect_thread_ids(story_id: str) -> list[str]:
    """Returns all sub-thread IDs for checkpoint cleanup."""
    return [
        f"{story_id}_master",
        _sub_thread_id(story_id, "wf2"),
        _sub_thread_id(story_id, "wf3"),
        _sub_thread_id(story_id, "wf4"),
        _sub_thread_id(story_id, "wf5"),
    ]


# ---------------------------------------------------------------------------
# Phase 1 nodes — Media (WF3 image + WF4 audio in parallel)
# ---------------------------------------------------------------------------

async def dispatch_media_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Dispatches WF3 (image) + WF4 (audio) in parallel via asyncio.gather.
    Activities (WF5) are NOT dispatched here — they run after both media workflows succeed.
    """
    cfg = config.get("configurable", {})
    story_id = state.get("story_id") or cfg.get("story_id")
    age      = cfg.get("age", "3-4")
    language = cfg.get("language", "English")
    theme    = cfg.get("theme", "theme1")
    voice    = _pick_voice(language, cfg.get("voice"))
    story    = state.get("story") or {}

    logger.info(f"[Master] Dispatching image+audio for story_id={story_id} theme={theme}")

    wf3_config = _build_media_config(story_id, "wf3", story, age, language, theme)
    wf4_config = _build_media_config(story_id, "wf4", story, age, language, theme)

    wf3_initial = {
        "story_text":  story.get("story_text", ""),
        "story_title": story.get("title", ""),
        "retry_count": 0,
        "status":      "pending",
        "completed":   [],
        "errors":      {},
    }
    wf4_initial = {
        "story_text":  story.get("story_text", ""),
        "language":    language,
        "voice":       voice,
        "retry_count": 0,
        "status":      "pending",
        "completed":   [],
        "errors":      {},
    }

    results = await asyncio.gather(
        image_workflow.ainvoke(wf3_initial, config=wf3_config),
        audio_workflow.ainvoke(wf4_initial, config=wf4_config),
        return_exceptions=True,
    )
    wf3_result, wf4_result = results

    def _status(result, wf_id: str) -> str:
        if isinstance(result, Exception):
            logger.error(f"[Master] {wf_id} raised exception: {result}")
            return "needs_human"
        return result.get("status", "needs_human")

    statuses = {
        "wf3": _status(wf3_result, "wf3"),
        "wf4": _status(wf4_result, "wf4"),
    }
    errors = {}
    for wf_id, result in [("wf3", wf3_result), ("wf4", wf4_result)]:
        if isinstance(result, Exception):
            errors[wf_id] = str(result)
        elif isinstance(result, dict) and result.get("errors"):
            errors[wf_id] = str(result["errors"])

    logger.info(f"[Master] Media results: {statuses}")
    return {"workflow_statuses": statuses, "errors": errors}


async def collect_media_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Checks image + audio workflow results.
    If any failed, notifies via Pub/Sub and suspends via interrupt() for HITL.
    Resume by calling POST /resume-workflow with decision: retry | skip | override.
    """
    statuses = state.get("workflow_statuses", {})
    failed = [
        {"workflow_id": wf_id, "error": state.get("errors", {}).get(wf_id, "unknown")}
        for wf_id in ("wf3", "wf4")
        if statuses.get(wf_id) == "needs_human"
    ]

    if not failed:
        logger.info("[Master] Image + audio completed successfully")
        return {}

    story_id = state.get("story_id")
    logger.warning(f"[Master] Media HITL: {[f['workflow_id'] for f in failed]}")
    _publish_hitl_notification(story_id, failed, phase="media")

    decision = interrupt({
        "message":         "Image or audio generation failed — human review required",
        "story_id":        story_id,
        "failed_workflows": failed,
        "instructions":    (
            "Call POST /resume-workflow with decision: 'retry', 'skip', or 'override'. "
            "You can also retry directly via POST /generate-image/{story_id} or "
            "POST /generate-audio/{story_id} then resume with 'override'."
        ),
    })

    human_decisions = {f["workflow_id"]: decision for f in failed}
    return {
        "human_decisions":   human_decisions,
        "workflow_statuses": {
            **statuses,
            **{f["workflow_id"]: "human_loop" for f in failed},
        },
    }


async def handle_media_decision_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """Applies admin decision for media HITL (skip/override marks as resolved)."""
    decisions  = state.get("human_decisions", {})
    statuses   = dict(state.get("workflow_statuses", {}))
    for wf_id, decision in decisions.items():
        if wf_id in ("wf3", "wf4"):
            statuses[wf_id] = "skipped" if decision in ("skip", "override") else statuses[wf_id]
            logger.info(f"[Master] Media decision for {wf_id}: {decision}")
    return {"workflow_statuses": statuses}


# ---------------------------------------------------------------------------
# Phase 2 nodes — Activities (WF5 runs AFTER image+audio are done)
# ---------------------------------------------------------------------------

async def dispatch_activities_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Dispatches WF5 (activities) after image+audio are complete.
    Activity seeds (mcq_seeds, art_seed, science_concepts, moral) are pulled from
    the story dict and passed via config so each agent uses concise, relevant input.
    """
    cfg      = config.get("configurable", {})
    story_id = state.get("story_id") or cfg.get("story_id")
    age      = cfg.get("age", "3-4")
    language = cfg.get("language", "English")
    story    = state.get("story") or {}

    logger.info(f"[Master] Dispatching activities for story_id={story_id}")

    wf5_config = _build_activities_config(story_id, story, age, language)
    wf5_initial = {
        "activities":  {},
        "images":      {},
        "completed":   [],
        "errors":      {},
        "retry_count": {},
        "status":      "pending",
    }

    try:
        wf5_result = await activity_workflow.ainvoke(wf5_initial, config=wf5_config)
        wf5_status = wf5_result.get("status", "needs_human")
        errors     = {}
        if wf5_result.get("errors"):
            errors["wf5"] = str(wf5_result["errors"])
    except Exception as e:
        logger.error(f"[Master] WF5 raised exception: {e}")
        wf5_status = "needs_human"
        errors     = {"wf5": str(e)}

    logger.info(f"[Master] Activities result: {wf5_status}")
    statuses = {**state.get("workflow_statuses", {}), "wf5": wf5_status}
    return {"workflow_statuses": statuses, "errors": {**state.get("errors", {}), **errors}}


async def collect_activities_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Checks WF5 result. If failed, fires HITL interrupt().
    """
    statuses = state.get("workflow_statuses", {})
    if statuses.get("wf5") != "needs_human":
        logger.info("[Master] Activities completed successfully")
        return {}

    story_id = state.get("story_id")
    failed   = [{"workflow_id": "wf5", "error": state.get("errors", {}).get("wf5", "unknown")}]
    logger.warning(f"[Master] Activities HITL for story_id={story_id}")
    _publish_hitl_notification(story_id, failed, phase="activities")

    decision = interrupt({
        "message":         "Activity generation failed — human review required",
        "story_id":        story_id,
        "failed_workflows": failed,
        "instructions":    "Call POST /resume-workflow with decision: 'retry', 'skip', or 'override'",
    })

    return {
        "human_decisions":   {**state.get("human_decisions", {}), "wf5": decision},
        "workflow_statuses": {**statuses, "wf5": "human_loop"},
    }


async def handle_activities_decision_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """Applies admin decision for activities HITL."""
    decisions = state.get("human_decisions", {})
    statuses  = dict(state.get("workflow_statuses", {}))
    if "wf5" in decisions:
        decision = decisions["wf5"]
        statuses["wf5"] = "skipped" if decision in ("skip", "override") else statuses["wf5"]
        logger.info(f"[Master] Activities decision: {decision}")
    return {"workflow_statuses": statuses}


# ---------------------------------------------------------------------------
# Finalize — cleanup checkpoints
# ---------------------------------------------------------------------------

async def finalize_node(state: MasterWorkflowState, config: RunnableConfig) -> dict:
    """
    Marks pipeline as complete and cleans up Firestore checkpoints.
    Checkpoint data is no longer needed once the full story (with image_url,
    audio_url, and activities) is persisted in the story document.
    """
    story_id = state.get("story_id")
    statuses = state.get("workflow_statuses", {})
    logger.info(f"[Master] Pipeline finalized for story_id={story_id}: {statuses}")

    # Clean up all checkpoints for this story's threads
    thread_ids = _collect_thread_ids(story_id)
    await firestore.delete_workflow_checkpoints(thread_ids)

    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_media(
    state: MasterWorkflowState,
) -> Literal["handle_media_decision", "dispatch_activities"]:
    """After media collection: handle HITL or proceed to activities."""
    statuses = state.get("workflow_statuses", {})
    if any(statuses.get(wf) == "human_loop" for wf in ("wf3", "wf4")):
        return "handle_media_decision"
    return "dispatch_activities"


def route_after_activities(
    state: MasterWorkflowState,
) -> Literal["handle_activities_decision", "finalize"]:
    """After activities collection: handle HITL or finalize."""
    if state.get("workflow_statuses", {}).get("wf5") == "human_loop":
        return "handle_activities_decision"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

master = StateGraph(MasterWorkflowState)

master.add_node("dispatch_media",              dispatch_media_node)
master.add_node("collect_media",               collect_media_node)
master.add_node("handle_media_decision",       handle_media_decision_node)
master.add_node("dispatch_activities",         dispatch_activities_node)
master.add_node("collect_activities",          collect_activities_node)
master.add_node("handle_activities_decision",  handle_activities_decision_node)
master.add_node("finalize",                    finalize_node)

master.set_entry_point("dispatch_media")
master.add_edge("dispatch_media", "collect_media")
master.add_conditional_edges(
    "collect_media",
    route_after_media,
    {"handle_media_decision": "handle_media_decision", "dispatch_activities": "dispatch_activities"},
)
master.add_edge("handle_media_decision", "dispatch_activities")
master.add_edge("dispatch_activities", "collect_activities")
master.add_conditional_edges(
    "collect_activities",
    route_after_activities,
    {"handle_activities_decision": "handle_activities_decision", "finalize": "finalize"},
)
master.add_edge("handle_activities_decision", "finalize")
master.add_edge("finalize", END)

if os.environ.get("USE_MEMORY_CHECKPOINTER", "false").lower() == "true":
    checkpointer = MemorySaver()
else:
    checkpointer = FirestoreCheckpointer()

master_workflow = master.compile(checkpointer=checkpointer)
