"""
WF2 — Story Creator Workflow

Flow:
    [generate_story] → [validate_story] → [evaluate_story]
                                                ↓
                                  pass → [save_story] → END
                                  fail → correction_attempts < 2?
                                            ↓ yes           ↓ no
                                      [self_correct]      END (error)
                                            ↓
                                      [generate_story] (loop)

Triggered by: POST /select-topic (human picks a topic from WF1 output)
Ends: Full story saved to Firestore `riostories_v3`; the Go client can then
      trigger POST /generate-media/{story_id} to start WF3+WF4+WF5 in parallel.

State key passed via config.configurable:
    story_id, age, language
State key passed in initial_state:
    selected_topic (the topic dict chosen by the human)
"""

import os
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from ..models.state import StoryCreatorState
from ..agents.story.story_creator_agent import StoryCreatorAgent
from ..agents.story.self_correction_agent import SelfCorrectionAgent
from ..agents.validators.evaluation_agent import EvaluationAgent
from ..services.database.firestore_service import FirestoreService
from ..services.database.checkpoint_service import FirestoreCheckpointer
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()

# --- Component instances ---
story_agent = StoryCreatorAgent()
evaluator = EvaluationAgent(workflow_type="story")
corrector = SelfCorrectionAgent(
    model_override=settings.STORY_CREATOR_MODEL,
    fallback_override=settings.STORY_CREATOR_FALLBACK_MODEL,
)
firestore = FirestoreService()

MAX_CORRECTION_ATTEMPTS = 2


def _unpack_config(state: StoryCreatorState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    return {
        **state,
        "story_id": cfg.get("story_id"),
        "age": cfg.get("age", "3-4"),
        "language": cfg.get("language", "English"),
    }


# --- Nodes ---

async def generate_story_node(state: StoryCreatorState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    return await story_agent.generate(enriched)


async def validate_story_node(state: StoryCreatorState, config: RunnableConfig) -> dict:
    """Structural validation: story must have required fields."""
    story = state.get("story")
    required = {"title", "story_text", "moral", "age_group", "language"}

    attempts = state.get("correction_attempts", 0)

    if not story or not isinstance(story, dict):
        logger.warning("[WF2] Structural validation failed: story is missing or not a dict")
        return {"validated": False, "correction_attempts": attempts + 1}

    if not required.issubset(story.keys()):
        missing = required - story.keys()
        logger.warning(f"[WF2] Structural validation failed: missing fields {missing}")
        return {"validated": False, "correction_attempts": attempts + 1}

    if not story.get("story_text", "").strip():
        logger.warning("[WF2] Structural validation failed: story_text is empty")
        return {"validated": False, "correction_attempts": attempts + 1}

    logger.info(f"[WF2] Structural validation passed: '{story.get('title')}'")
    return {"validated": True}


async def evaluate_story_node(state: StoryCreatorState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    return await evaluator.evaluate(enriched)


async def self_correct_story_node(state: StoryCreatorState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    return await corrector.correct(enriched, content_key="story")


async def save_story_node(state: StoryCreatorState, config: RunnableConfig) -> dict:
    cfg           = config.get("configurable", {})
    story_id      = cfg.get("story_id")
    topics_id     = cfg.get("topics_id")
    story         = state.get("story", {})
    selected_topic = state.get("selected_topic") or {}
    # Theme from config (batch flow) or from the selected topic (manual /select-topic flow)
    theme = (
        cfg.get("theme")
        or selected_topic.get("theme", "theme1")
    )
    topic_id          = selected_topic.get("topic_id")
    topic_document_id = selected_topic.get("topic_document_id")

    try:
        await firestore.save_story(
            story_id, story, theme,
            topics_id=topics_id,
            topic_id=topic_id,
            topic_document_id=topic_document_id,
        )
        logger.info(f"[WF2] Story saved: {theme}/{story_id} '{story.get('title')}'")
        return {"completed": ["story"]}
    except Exception as e:
        logger.error(f"[WF2] Failed to save story: {e}")
        return {"errors": {"save_story": str(e)}}


# --- Routing ---

def route_after_validate(state: StoryCreatorState) -> Literal["evaluate_story", "generate_story", "__end__"]:
    if state.get("validated"):
        return "evaluate_story"
    if state.get("correction_attempts", 0) >= MAX_CORRECTION_ATTEMPTS:
        logger.error("[WF2] Max structural validation attempts reached — workflow failed")
        return END
    return "generate_story"


def route_after_evaluate(
    state: StoryCreatorState,
) -> Literal["save_story", "self_correct_story", "__end__"]:
    evaluation = state.get("evaluation") or {}
    if evaluation.get("passed"):
        return "save_story"

    attempts = state.get("correction_attempts", 0)
    if attempts < MAX_CORRECTION_ATTEMPTS:
        return "self_correct_story"

    logger.error("[WF2] Max correction attempts reached — workflow failed")
    return END


# --- Graph ---

workflow = StateGraph(StoryCreatorState)

workflow.add_node("generate_story", generate_story_node)
workflow.add_node("validate_story", validate_story_node)
workflow.add_node("evaluate_story", evaluate_story_node)
workflow.add_node("self_correct_story", self_correct_story_node)
workflow.add_node("save_story", save_story_node)

workflow.set_entry_point("generate_story")
workflow.add_edge("generate_story", "validate_story")
workflow.add_conditional_edges(
    "validate_story",
    route_after_validate,
    {"evaluate_story": "evaluate_story", "generate_story": "generate_story"},
)
workflow.add_conditional_edges(
    "evaluate_story",
    route_after_evaluate,
    {"save_story": "save_story", "self_correct_story": "self_correct_story", END: END},
)
workflow.add_edge("self_correct_story", "generate_story")
workflow.add_edge("save_story", END)

if os.environ.get("USE_MEMORY_CHECKPOINTER", "false").lower() == "true":
    checkpointer = MemorySaver()
else:
    checkpointer = FirestoreCheckpointer()

story_creator_workflow = workflow.compile(checkpointer=checkpointer)
