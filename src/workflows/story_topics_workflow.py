"""
WF1 — Story Topics Workflow

Flow:
    [generate_topics] → [validate_topics] → [evaluate_topics]
                                                    ↓
                                      pass → [save_topics] → [batch_create_stories] → END
                                      fail → correction_attempts < 2?
                                                ↓ yes            ↓ no
                                          [self_correct]       END (error)
                                                ↓
                                          [generate_topics] (loop)

batch_create_stories:
    For every topic title saved in save_topics, in bounded parallel:
      1. Generate a unique story_id (UUID)
      2. Run WF2 (story_creator_workflow) → creates the full story
      3. Run Master (WF3+WF4+WF5) → image, audio, activities
      4. Patch story_id back into the story_title_library_v1 entry for that title

Triggered by: POST /generate-topics
Ends: All stories (+ media) created and stored; story_ids saved in title library docs.

State key passed via config.configurable:
    story_id, age, language, theme (optional)
"""

import asyncio
import os
import uuid
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from ..models.state import StoryTopicsState
from ..agents.story.topics_creator_agent import TopicsCreatorAgent
from ..agents.story.self_correction_agent import SelfCorrectionAgent
from ..agents.validators.evaluation_agent import EvaluationAgent
from ..services.database.firestore_service import FirestoreService
from ..services.database.checkpoint_service import FirestoreCheckpointer
from ..utils.logger import setup_logger
from ..utils.config import get_settings

# _LANG_CODE_MAP mirrors the mapping in topics_creator_agent.py
_LANG_CODE_MAP: dict[str, str] = {
    "english": "en", "telugu": "te", "en": "en", "te": "te",
}

logger = setup_logger(__name__)
settings = get_settings()

# --- Component instances ---
topics_agent = TopicsCreatorAgent()
evaluator = EvaluationAgent(workflow_type="story_topics")
corrector = SelfCorrectionAgent(
    model_override=settings.STORY_TOPICS_MODEL,
    fallback_override=settings.STORY_TOPICS_FALLBACK_MODEL,
)
firestore = FirestoreService()

MAX_CORRECTION_ATTEMPTS = 2


def _unpack_config(state: StoryTopicsState, config: RunnableConfig) -> dict:
    """Merge state with read-only config.configurable fields."""
    cfg = config.get("configurable", {})
    return {
        **state,
        "story_id":    cfg.get("story_id"),
        "age":         cfg.get("age", "3-4"),
        "language":    cfg.get("language", "English"),
        "theme":       cfg.get("theme", ""),
        "religion":    cfg.get("religion", "universal_wisdom"),
        "preferences": cfg.get("preferences", ["Any"]),
        "country":     cfg.get("country", "Any"),
    }


# --- Nodes ---

async def generate_topics_node(state: StoryTopicsState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    return await topics_agent.generate(enriched)


async def validate_topics_node(state: StoryTopicsState, config: RunnableConfig) -> dict:
    """Structural validation: topics must be a non-empty list with required fields."""
    topics = state.get("topics")
    required = {"title", "theme", "moral", "description"}

    if not topics or not isinstance(topics, list) or len(topics) == 0:
        logger.warning("[WF1] Structural validation failed: empty or missing topics list")
        return {"validated": False}

    if not all(required.issubset(t.keys()) for t in topics):
        logger.warning("[WF1] Structural validation failed: missing required fields in topics")
        return {"validated": False}

    logger.info(f"[WF1] Structural validation passed: {len(topics)} topics")
    return {"validated": True}


async def evaluate_topics_node(state: StoryTopicsState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    return await evaluator.evaluate(enriched)


async def self_correct_topics_node(state: StoryTopicsState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    return await corrector.correct(enriched, content_key="topics")


async def save_topics_node(state: StoryTopicsState, config: RunnableConfig) -> dict:
    # Topics are already persisted per-theme inside TopicsCreatorAgent._generate_one()
    # (cached in the theme topic collections on first LLM call).
    # This node generates a topics_id for the batch and marks the evaluation-approved step.
    topics = state.get("topics", [])
    topics_id = str(uuid.uuid4())
    logger.info(
        f"[WF1] {len(topics)} topics evaluation-approved (topics_id={topics_id}), "
        "proceeding to batch story creation"
    )
    return {"completed": ["topics"], "story_ids": {"_topics_id": topics_id}}


async def batch_create_stories_node(state: StoryTopicsState, config: RunnableConfig) -> dict:
    """
    Creates a full story pipeline (WF2 → WF3+WF4+WF5) for every topic title.

    For each topic:
      1. Generates a unique story_id (UUID).
      2. Runs story_creator_workflow (WF2) with the topic as selected_topic.
      3. If WF2 produces a story, runs master_workflow (WF3+WF4+WF5) for image/audio/activities.
      4. Patches story_id back into the story_title_library_v1 document for that title.

    Runs with bounded concurrency (MAX_CONCURRENCY setting).
    Individual topic failures are logged but do not abort the batch.
    """
    # Local imports to avoid module-level circular imports between WF1/WF2/Master
    from ..workflows.story_creator_workflow import story_creator_workflow
    from ..workflows.master_workflow import master_workflow
    from ..utils.tracing import get_trace_callbacks

    cfg = config.get("configurable", {})
    age        = cfg.get("age", "3-4")
    language   = cfg.get("language", "English")
    voice_type = cfg.get("voice")
    lang_code  = _LANG_CODE_MAP.get(language.lower(), language[:2].lower())

    topics = state.get("topics") or []
    if not topics:
        logger.warning("[WF1/batch] No topics in state — skipping batch story creation")
        return {}

    # Retrieve the topics_id generated in save_topics_node
    topics_id = (state.get("story_ids") or {}).get("_topics_id") or str(uuid.uuid4())

    logger.info(f"[WF1/batch] Starting batch creation for {len(topics)} topics (topics_id={topics_id})")

    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENCY)
    story_id_map: dict[str, str] = {}

    async def _run_topic_pipeline(topic: dict) -> tuple[str, str, bool]:
        """Returns (title, story_id, success)."""
        title      = topic.get("title", "")
        theme      = topic.get("theme", "")
        filter_val = topic.get("filter_value", "")

        async with semaphore:
            # ----------------------------------------------------------------
            # Resume check: find any existing story for this topic so we don't
            # regenerate it.  Two ways the story_id can be known:
            #   1. topic dict has story_id (patched in by update_title_story_id)
            #   2. story already exists in Firestore by title (e.g. update_title_story_id
            #      was never called because the process crashed after WF2 saved)
            # ----------------------------------------------------------------
            existing_story_id = topic.get("story_id")
            existing = None

            if existing_story_id:
                existing = await firestore.get_story(existing_story_id, theme)

            if not existing and title:
                # Fallback: search by title in case story_id wasn't patched into topic doc
                existing = await firestore.get_story_by_title(title, theme)
                if existing:
                    existing_story_id = existing.get("story_id")
                    logger.info(
                        f"[WF1/batch] Found existing story by title '{title}' "
                        f"(story_id={existing_story_id}) — patching topic doc"
                    )
                    # Patch it back so future runs don't need the title query
                    try:
                        await firestore.update_title_story_id(
                            theme, age, lang_code, filter_val, title, existing_story_id
                        )
                    except Exception as _e:
                        logger.warning(f"[WF1/batch] Could not back-patch story_id for '{title}': {_e}")

            if existing_story_id and existing and existing.get("story_text", "").strip():
                needs_image      = not existing.get("image_url")
                needs_audio      = not existing.get("audio_url")
                done_activities  = existing.get("activities", {})
                activity_types   = {"mcq", "art", "science", "moral"}
                needs_activities = not activity_types.issubset(done_activities.keys())

                if not needs_image and not needs_audio and not needs_activities:
                    logger.info(f"[WF1/batch] Already complete — skipping: '{title}'")
                    return title, existing_story_id, True

                logger.info(
                    f"[WF1/batch] Resuming '{title}' ({existing_story_id}): "
                    f"image={needs_image} audio={needs_audio} activities={needs_activities}"
                )

                if needs_image or needs_audio:
                    master_config = {
                        "configurable": {
                            "thread_id": f"{existing_story_id}_master",
                            "story_id":  existing_story_id,
                            "age":       age,
                            "language":  language,
                            "theme":     theme,
                            "voice":     voice_type,
                        },
                        "callbacks": get_trace_callbacks(
                            name="master-pipeline",
                            metadata={"story_id": existing_story_id, "topic": title,
                                      "theme": theme, "age": age},
                            tags=["master", "resume", "batch"],
                            session_id=topics_id,
                        ),
                    }
                    master_initial = {
                        "story_id":            existing_story_id,
                        "topics":              None,
                        "story":               existing,
                        "workflow_statuses":   {},
                        "workflow_retries":    {},
                        "human_loop_requests": {},
                        "human_decisions":     {},
                        "errors":              {},
                    }
                    try:
                        await master_workflow.ainvoke(master_initial, config=master_config)
                    except Exception as e:
                        logger.error(f"[WF1/batch] Resume master failed for '{title}': {e}")
                elif needs_activities:
                    from ..workflows.activity_workflow import app_workflow as activity_workflow
                    wf5_config = {
                        "configurable": {
                            "thread_id":        f"{existing_story_id}_wf5",
                            "story_id":         existing_story_id,
                            "story_text":       existing.get("story_text", ""),
                            "age":              age,
                            "language":         language,
                            "mcq_seeds":        existing.get("mcq_seeds", []),
                            "art_seed":         existing.get("art_seed", ""),
                            "science_concepts": existing.get("science_concepts", []),
                            "moral":            existing.get("moral", ""),
                        },
                    }
                    wf5_initial = {
                        "activities":  {},
                        "images":      {},
                        "completed":   [],
                        "errors":      {},
                        "retry_count": {},
                        "status":      "pending",
                    }
                    try:
                        await activity_workflow.ainvoke(wf5_initial, config=wf5_config)
                    except Exception as e:
                        logger.error(f"[WF1/batch] Resume activities failed for '{title}': {e}")

                return title, existing_story_id, True

            # ----------------------------------------------------------------
            # Fresh run: no story yet — full WF2 → Master pipeline
            # ----------------------------------------------------------------
            story_id = str(uuid.uuid4())

            # --- WF2: Story Creator ---
            wf2_config = {
                "configurable": {
                    "thread_id": f"{story_id}_wf2",
                    "story_id":  story_id,
                    "age":       age,
                    "language":  language,
                    "topics_id": topics_id,
                    "theme":     theme,
                },
                "callbacks": get_trace_callbacks(
                    name="WF2-story",
                    metadata={"story_id": story_id, "topic": title, "theme": theme, "age": age},
                    tags=["wf2", "story", "batch"],
                    session_id=topics_id,
                ),
            }
            wf2_initial = {
                "selected_topic":      topic,
                "story":               None,
                "validated":           False,
                "evaluation":          None,
                "correction_attempts": 0,
                "completed":           [],
                "errors":              {},
            }
            try:
                wf2_result = await story_creator_workflow.ainvoke(wf2_initial, config=wf2_config)
            except Exception as e:
                logger.error(f"[WF1/batch] WF2 failed for '{title}': {e}")
                return title, story_id, False

            story = wf2_result.get("story") or {}
            if not story.get("story_text", "").strip():
                logger.warning(f"[WF1/batch] WF2 produced empty story for '{title}' — skipping media")
                return title, story_id, False

            # --- Master: WF3 + WF4 then WF5 ---
            master_config = {
                "configurable": {
                    "thread_id": f"{story_id}_master",
                    "story_id":  story_id,
                    "age":       age,
                    "language":  language,
                    "theme":     theme,
                    "voice":     voice_type,
                },
                "callbacks": get_trace_callbacks(
                    name="master-pipeline",
                    metadata={"story_id": story_id, "topic": title, "theme": theme, "age": age},
                    tags=["master", "image", "audio", "activities", "batch"],
                    session_id=topics_id,
                ),
            }
            master_initial = {
                "story_id":            story_id,
                "topics":              None,
                "story":               story,
                "workflow_statuses":   {},
                "workflow_retries":    {},
                "human_loop_requests": {},
                "human_decisions":     {},
                "errors":              {},
            }
            try:
                await master_workflow.ainvoke(master_initial, config=master_config)
            except Exception as e:
                logger.error(f"[WF1/batch] Master workflow failed for '{title}' ({story_id}): {e}")
                # Story was already saved by WF2 — don't block; record partial success

            # --- Update topic library doc with story_id ---
            try:
                await firestore.update_title_story_id(
                    theme, age, lang_code, filter_val, title, story_id
                )
            except Exception as e:
                logger.warning(f"[WF1/batch] Could not patch story_id into library for '{title}': {e}")

            logger.info(f"[WF1/batch] Pipeline done: story_id={story_id} '{title}'")
            return title, story_id, True

    results = await asyncio.gather(
        *[_run_topic_pipeline(t) for t in topics],
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, tuple):
            title, sid, success = result
            story_id_map[title] = sid
        else:
            logger.error(f"[WF1/batch] Unexpected exception during topic pipeline: {result}")

    success_count = sum(1 for r in results if isinstance(r, tuple) and r[2])
    logger.info(f"[WF1/batch] Completed {success_count}/{len(topics)} stories successfully")

    return {"story_ids": story_id_map, "completed": ["batch_stories"]}


# --- Routing ---

def route_after_validate(state: StoryTopicsState) -> Literal["evaluate_topics", "generate_topics"]:
    """After structural validation: proceed to eval if valid, else regenerate."""
    if state.get("validated"):
        return "evaluate_topics"
    return "generate_topics"


def route_after_evaluate(
    state: StoryTopicsState,
) -> Literal["save_topics", "self_correct_topics", "__end__"]:
    """After evaluation: save if passed, correct if attempts remain, else fail."""
    evaluation = state.get("evaluation") or {}
    if evaluation.get("passed"):
        return "save_topics"

    attempts = state.get("correction_attempts", 0)
    if attempts < MAX_CORRECTION_ATTEMPTS:
        return "self_correct_topics"

    logger.error("[WF1] Max correction attempts reached — workflow failed")
    return END


# --- Graph ---

workflow = StateGraph(StoryTopicsState)
workflow.add_node("generate_topics", generate_topics_node)
workflow.add_node("validate_topics", validate_topics_node)
workflow.add_node("evaluate_topics", evaluate_topics_node)
workflow.add_node("self_correct_topics", self_correct_topics_node)
workflow.add_node("save_topics", save_topics_node)
workflow.add_node("batch_create_stories", batch_create_stories_node)

workflow.set_entry_point("generate_topics")
workflow.add_edge("generate_topics", "validate_topics")
workflow.add_conditional_edges(
    "validate_topics",
    route_after_validate,
    {"evaluate_topics": "evaluate_topics", "generate_topics": "generate_topics"},
)
workflow.add_conditional_edges(
    "evaluate_topics",
    route_after_evaluate,
    {"save_topics": "save_topics", "self_correct_topics": "self_correct_topics", END: END},
)
workflow.add_edge("self_correct_topics", "generate_topics")
workflow.add_edge("save_topics", "batch_create_stories")
workflow.add_edge("batch_create_stories", END)

# Checkpointer: MemorySaver for dev, Firestore for prod
if os.environ.get("USE_MEMORY_CHECKPOINTER", "false").lower() == "true":
    checkpointer = MemorySaver()
else:
    checkpointer = FirestoreCheckpointer()

story_topics_workflow = workflow.compile(checkpointer=checkpointer)
