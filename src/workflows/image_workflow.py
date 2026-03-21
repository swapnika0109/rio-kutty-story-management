"""
WF3 — Image Generator Workflow (compiled subgraph)

Flow:
    [generate_image] → [validate_image] → [save_image] → END (status="completed")
                                               ↓ fail
                                         retry_count < 4?
                                           ↓ yes           ↓ no
                                      [generate_image]   END (status="needs_human")

Evaluation and self-correction are disabled (commented out).

Why this is a compiled subgraph:
- Manages its own 4-retry loop internally
- Has its own state (ImageWorkflowState) and thread_id
- Can be tested standalone without the master workflow
- Master simply calls image_workflow.ainvoke(...) and checks state["status"]

Triggered by master_workflow via asyncio.gather alongside WF4 and WF5.
"""

import uuid
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from ..models.state import ImageWorkflowState
from ..agents.media.image_generator_agent import ImageGeneratorAgent
# from ..agents.validators.evaluation_agent import EvaluationAgent  # evaluation disabled
from ..services.database.firestore_service import FirestoreService
from ..services.database.storage_bucket import StorageBucketService
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()

MAX_RETRIES = settings.PARALLEL_WORKFLOW_MAX_RETRIES

# --- Component instances ---
image_agent = ImageGeneratorAgent()
# evaluator = EvaluationAgent(workflow_type="image")  # evaluation disabled
firestore = FirestoreService()
storage = StorageBucketService()


def _unpack_config(state: ImageWorkflowState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    return {
        **state,
        "story_id": cfg.get("story_id"),
        "age": cfg.get("age", "3-4"),
        "language": cfg.get("language", "English"),
    }


# --- Nodes ---

async def generate_image_node(state: ImageWorkflowState, config: RunnableConfig) -> dict:
    enriched = _unpack_config(state, config)
    result = await image_agent.generate(enriched)
    current = state.get("retry_count", 0)
    result["retry_count"] = current + 1
    return result


async def validate_image_node(state: ImageWorkflowState, config: RunnableConfig) -> dict:
    """Structural validation: image_bytes must be non-empty bytes."""
    image_bytes = state.get("image_bytes")
    if not image_bytes or not isinstance(image_bytes, bytes):
        logger.warning("[WF3] Structural validation failed: image_bytes missing or empty")
        return {"validated": False}
    logger.info(f"[WF3] Structural validation passed: {len(image_bytes)} bytes")
    return {"validated": True}


# async def evaluate_image_node(state: ImageWorkflowState, config: RunnableConfig) -> dict:
#     enriched = _unpack_config(state, config)
#     return await evaluator.evaluate(enriched)


async def save_image_node(state: ImageWorkflowState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    story_id = cfg.get("story_id")
    theme = cfg.get("theme", "theme1")
    image_bytes = state.get("image_bytes")
    image_prompt = state.get("image_prompt", "")

    filename = f"story-images/{uuid.uuid4()}.png"
    image_url = await storage.upload_file(filename, image_bytes, content_type="image/png")

    if not image_url:
        logger.error(f"[WF3] GCS upload failed for story_id={story_id}")
        return {
            "errors": {**state.get("errors", {}), "save_image": "GCS upload returned no URL"},
            "status": "needs_human",
        }

    try:
        await firestore.save_story_image(story_id, image_url, image_prompt, theme)
        logger.info(f"[WF3] Image saved: {image_url}")
        return {
            "image_url":   image_url,
            "image_bytes": None,   # clear binary from checkpoint — already in GCS
            "completed":   ["image"],
            "status":      "completed",
        }
    except Exception as e:
        logger.error(f"[WF3] Firestore update failed: {e}")
        return {
            "errors": {**state.get("errors", {}), "save_image": str(e)},
            "status": "needs_human",
        }


# --- Routing ---

def route_after_validate(state: ImageWorkflowState) -> Literal["save_image", "generate_image", "__end__"]:
    if state.get("errors"):
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return END
    if state.get("validated"):
        return "save_image"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return END
    return "generate_image"


# def route_after_evaluate(
#     state: ImageWorkflowState,
# ) -> Literal["save_image", "generate_image", "__end__"]:
#     evaluation = state.get("evaluation") or {}
#     if evaluation.get("passed"):
#         return "save_image"
#     if state.get("retry_count", 0) >= MAX_RETRIES:
#         return END
#     return "generate_image"


async def mark_needs_human_node(state: ImageWorkflowState, config: RunnableConfig) -> dict:
    """Final node when max retries exhausted — signals master to escalate to HITL."""
    logger.error(f"[WF3] Max retries ({MAX_RETRIES}) exhausted — needs human review")
    return {
        "status": "needs_human",
        "errors": {
            **state.get("errors", {}),
            "image_workflow": f"Failed after {MAX_RETRIES} retries",
        },
    }


# --- Graph ---

workflow = StateGraph(ImageWorkflowState)

workflow.add_node("generate_image", generate_image_node)
workflow.add_node("validate_image", validate_image_node)
# workflow.add_node("evaluate_image", evaluate_image_node)  # evaluation disabled
workflow.add_node("save_image", save_image_node)
workflow.add_node("mark_needs_human", mark_needs_human_node)

workflow.set_entry_point("generate_image")
workflow.add_edge("generate_image", "validate_image")
workflow.add_conditional_edges(
    "validate_image",
    route_after_validate,
    {
        "save_image":     "save_image",
        "generate_image": "generate_image",
        END:              "mark_needs_human",
    },
)
# workflow.add_conditional_edges(
#     "evaluate_image",
#     route_after_evaluate,
#     {
#         "save_image": "save_image",
#         "generate_image": "generate_image",
#         END: "mark_needs_human",
#     },
# )
workflow.add_edge("save_image", END)
workflow.add_edge("mark_needs_human", END)

# WF3 does not need persistent checkpointing — it runs in seconds, stores no
# human-interrupt state, and binary image_bytes would exceed Firestore's 1 MB
# document limit.  MemorySaver is used so LangGraph's internal graph machinery
# still works, but nothing is written to Firestore.
image_workflow = workflow.compile(checkpointer=MemorySaver())
