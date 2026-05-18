from typing import TypedDict, List, Dict, Any, Annotated
import operator
import os
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import Agents
from ..agents.activities.mcq_agent import MCQAgent
from ..agents.activities.art_agent import ArtAgent
from ..agents.activities.moral_agent import MoralAgent
from ..agents.activities.science_agent import ScienceAgent
from ..agents.validators.validator_agent import ValidatorAgent
from ..agents.validators.evaluation_agent import EvaluationAgent
from ..services.database.firestore_service import FirestoreService
from ..services.database.checkpoint_service import FirestoreCheckpointer
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)
settings = get_settings()

# Import shared reducer — defined once in models.state to avoid duplication
from ..models.state import merge_dicts

# State Definition for WF5 activities subgraph
class ActivityState(TypedDict):
    # story_id, story_text, age, language are in config.configurable (read-only)
    activities: Annotated[Dict[str, Any], merge_dicts]
    images: Annotated[Dict[str, str], merge_dicts]
    completed: Annotated[List[str], operator.add]
    errors: Annotated[Dict[str, str], merge_dicts]
    retry_count: Annotated[Dict[str, int], merge_dicts]
    # Subgraph result status reported back to master: "completed" | "needs_human"
    status: str

# Initialize Components
mcq_agent = MCQAgent(prompt_version=settings.MCQ_PROMPT_VERSION)
art_agent = ArtAgent(prompt_version=settings.ART_PROMPT_VERSION)
moral_agent = MoralAgent(prompt_version=settings.MORAL_PROMPT_VERSION)
science_agent = ScienceAgent(prompt_version=settings.SCIENCE_PROMPT_VERSION)
validator = ValidatorAgent()
evaluator = EvaluationAgent(workflow_type="activities")
firestore_service = FirestoreService()


def unpack_config(state: ActivityState, config: RunnableConfig):
    # Access read-only data from config
    cfg = config.get("configurable", {})
    # Merge state with config values so agents receive both story seeds and metadata
    return {
        **state,
        "story_id":        cfg.get("story_id"),
        "story_text":      cfg.get("story_text"),
        "age":             cfg.get("age"),
        "language":        cfg.get("language"),
        # Activity seeds — sourced from the story JSON, passed via master config
        "mcq_seeds":       cfg.get("mcq_seeds", []),
        "art_seed":        cfg.get("art_seed", ""),
        "science_concepts": cfg.get("science_concepts", []),
        "moral":           cfg.get("moral", ""),
    }

# --- Generation Nodes ---
async def generate_mcq_node(state: ActivityState, config: RunnableConfig):
    result = await mcq_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("mcq", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "mcq": current_retry + 1}
    return result

async def generate_art_node(state: ActivityState, config: RunnableConfig): 
    result = await art_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("art", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "art": current_retry + 1}
    return result

async def generate_moral_node(state: ActivityState, config: RunnableConfig): 
    result = await moral_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("moral", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "moral": current_retry + 1}
    return result

async def generate_science_node(state: ActivityState, config: RunnableConfig): 
    result = await science_agent.generate(unpack_config(state, config))
    current_retry = state.get("retry_count", {}).get("science", 0)
    result["retry_count"] = {**state.get("retry_count", {}), "science": current_retry + 1}
    return result

# --- Validation Nodes ---
def validate_mcq_node(state: ActivityState, config: RunnableConfig): 
    return validator.validate_mcq(unpack_config(state, config))

def validate_art_node(state: ActivityState, config: RunnableConfig): 
    return validator.validate_art(unpack_config(state, config))

def validate_science_node(state: ActivityState, config: RunnableConfig): 
    return validator.validate_science(unpack_config(state, config))

def validate_moral_node(state: ActivityState, config: RunnableConfig):
    return validator.validate_moral(unpack_config(state, config))

# --- Evaluation Nodes ---
# Each evaluates one activity type. On failure we increment retry_count and re-run
# the generator — same pattern as structural validation. The evaluator's per-activity
# rubric (toxicity, safety, story-alignment, instructions, engagability, etc.) is
# defined in EvaluationAgent.
#
# Activity generation runs in parallel (fan-out at `start`), so without coordination
# all four activities reach their evaluate node at roughly the same time. That fans
# out 4 × 7 = 28 GEval calls to the eval model in seconds and triggers Gemini 503
# "high demand" responses. We serialise evaluation across activities with this lock
# so only one activity evaluates at a time; the 7 metrics *within* an activity still
# parallelise (capped by _eval_semaphore inside the evaluator). Gen+val stay parallel.
#
# Lazy-init so the lock binds to the running event loop on first use. Each
# pytest-asyncio test gets its own loop; a module-level `asyncio.Lock()` created
# at import time can end up bound to a now-defunct loop and silently no-op.
_activity_eval_lock: asyncio.Lock | None = None


def _get_activity_eval_lock() -> asyncio.Lock:
    global _activity_eval_lock
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _activity_eval_lock is None or getattr(_activity_eval_lock, "_loop", loop) is not loop:
        _activity_eval_lock = asyncio.Lock()
    return _activity_eval_lock


async def _evaluate_activity(state: ActivityState, config: RunnableConfig, activity_type: str):
    """Run the evaluator on one activity; on fail, bump that activity's retry counter
    and set an evaluation_failed flag so the post-eval router can re-route to gen."""
    enriched = {**unpack_config(state, config), "activity_type": activity_type}
    # Serialise across activities so Gemini doesn't see a 28-call burst.
    lock = _get_activity_eval_lock()
    logger.info(f"[WF5/{activity_type}] Waiting for eval lock (queued={lock.locked()})")
    async with lock:
        logger.info(f"[WF5/{activity_type}] Acquired eval lock; starting evaluation")
        result = await evaluator.evaluate(enriched)
        logger.info(f"[WF5/{activity_type}] Released eval lock")
    evaluation = (result or {}).get("evaluation") or {}
    passed = evaluation.get("passed", False)
    # Always write the eval result (passed bool + score + reason) keyed by
    # `_eval_<type>` so the post-eval router can read the LATEST outcome.
    # Using merge_dicts on `activities` means last-write-wins per key — a pass
    # after a prior fail correctly overwrites the failure record.
    eval_record = {"passed": passed, "score": evaluation.get("score"), "reason": evaluation.get("reason")}
    if passed:
        logger.info(f"[WF5/{activity_type}] Evaluation PASSED score={evaluation.get('score')}")
        return {"activities": {f"_eval_{activity_type}": eval_record}}
    logger.warning(
        f"[WF5/{activity_type}] Evaluation FAILED score={evaluation.get('score')} "
        f"reason={evaluation.get('reason')}"
    )
    current_retry = state.get("retry_count", {}).get(activity_type, 0)
    return {
        "retry_count": {**state.get("retry_count", {}), activity_type: current_retry + 1},
        "activities": {f"_eval_{activity_type}": eval_record},
    }

async def evaluate_mcq_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "mcq")

async def evaluate_art_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "art")

async def evaluate_moral_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "moral")

async def evaluate_science_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "science")

# --- Save Nodes ---
async def save_mcq_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "mcq" in state.get("activities", {}):
        data = state["activities"]["mcq"]
        logger.info(f"Saving MCQ for story {db_data['story_id']}: {data}")
        await firestore_service.save_activity(db_data["story_id"], "mcq", data)
    return {}

async def save_art_node(state: ActivityState, config: RunnableConfig):
    # NOTE: image is already a GCS filename string by this point — the activity
    # agents upload during generation so raw PNG bytes never live in state.
    db_data = unpack_config(state, config)
    if "art" in state.get("activities", {}):
        data = state["activities"]["art"]
        payload = {"items": data}
        await firestore_service.save_activity(db_data["story_id"], "art", payload)
    return {}


async def save_science_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "science" in state.get("activities", {}):
        data = state["activities"]["science"][0]
        payload = {**data} if isinstance(data, dict) else {"items": data}
        await firestore_service.save_activity(db_data["story_id"], "science", payload)
    return {}


async def save_moral_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    if "moral" in state.get("activities", {}):
        data_list = state["activities"]["moral"]
        payloads = [
            ({**data} if isinstance(data, dict) else {"items": data})
            for data in data_list
        ]
        await firestore_service.save_activity(db_data["story_id"], "moral", payloads)
    return {}

# --- Routing Logic ---
# Max retries aligned with PARALLEL_WORKFLOW_MAX_RETRIES (4) so master can
# accurately decide on human-in-loop escalation when this subgraph returns
# status="needs_human".
MAX_ACTIVITY_RETRIES = settings.PARALLEL_WORKFLOW_MAX_RETRIES

def create_retry_logic(activity_type: str):
    def should_retry(state: ActivityState):
        if activity_type in state.get("errors", {}): return "fail"

        is_completed = activity_type in state.get("completed", [])
        retries = state.get("retry_count", {}).get(activity_type, 0)

        if is_completed:
            return "next"

        if retries < MAX_ACTIVITY_RETRIES:
            return "retry"
        return "fail"
    return should_retry


def create_post_eval_routing(activity_type: str):
    """After evaluate node: if eval passed, go to save. If failed, retry generator
    unless we've hit the retry cap."""
    def route(state: ActivityState):
        if activity_type in state.get("errors", {}):
            return "fail"
        eval_record = state.get("activities", {}).get(f"_eval_{activity_type}") or {}
        retries = state.get("retry_count", {}).get(activity_type, 0)
        if eval_record.get("passed"):
            return "save"
        if retries < MAX_ACTIVITY_RETRIES:
            return "retry"
        return "fail"
    return route

# Helper to Check Exists & Route (Runs AT RUNTIME)
async def route_start(state: ActivityState, config: RunnableConfig):
    story_id = config["configurable"]["story_id"]
    nodes_to_run = []
    
    # Map activity types to their node name prefixes
    type_to_prefix = {
        "mcq": "mcq",
        "art": "art",
        "moral": "mor",
        "science": "sci"
    }
    
    for activity_type, prefix in type_to_prefix.items():
        # Check if it exists in DB
        exists = await firestore_service.check_if_activity_exists(story_id, activity_type)
        
        if not exists:
            # If NOT exists, we want to run the generator
            nodes_to_run.append(f"gen_{prefix}")
        else:
            logger.info(f"Skipping {activity_type} for {story_id} - already exists.")
            
    return nodes_to_run

# Terminal node when activities exhaust retries — reports needs_human to master
def mark_activities_needs_human(state: ActivityState):
    failed = list(state.get("errors", {}).keys())
    logger.error(f"[WF5] Activities failed after {MAX_ACTIVITY_RETRIES} retries: {failed}")
    return {"status": "needs_human"}

# Mark all-completed terminal node
def mark_activities_completed(state: ActivityState):
    return {"status": "completed"}


# --- Graph Construction ---
workflow = StateGraph(ActivityState)

# Add Nodes
workflow.add_node("start", lambda s: s)  # Dummy start node
workflow.add_node("mark_needs_human", mark_activities_needs_human)
workflow.add_node("mark_completed", mark_activities_completed)

# Activity 1: MCQ
workflow.add_node("gen_mcq", generate_mcq_node)
workflow.add_node("val_mcq", validate_mcq_node)
workflow.add_node("eval_mcq", evaluate_mcq_node)
workflow.add_node("save_mcq", save_mcq_node)

# Activity 2: Art
workflow.add_node("gen_art", generate_art_node)
workflow.add_node("val_art", validate_art_node)
workflow.add_node("eval_art", evaluate_art_node)
workflow.add_node("save_art", save_art_node)

# Activity 3: Moral
workflow.add_node("gen_mor", generate_moral_node)
workflow.add_node("val_mor", validate_moral_node)
workflow.add_node("eval_mor", evaluate_moral_node)
workflow.add_node("save_mor", save_moral_node)

# Activity 4: Science
workflow.add_node("gen_sci", generate_science_node)
workflow.add_node("val_sci", validate_science_node)
workflow.add_node("eval_sci", evaluate_science_node)
workflow.add_node("save_sci", save_science_node)

# Entry & Fan-out (Dynamic)
workflow.set_entry_point("start")
workflow.add_conditional_edges(
    "start",
    route_start,
    ["gen_mcq", "gen_art", "gen_mor", "gen_sci"]
)

# Define Flows (Standardized: Gen -> Val -> Eval -> Retry/Save)
for key, prefix in [("mcq", "mcq"), ("art", "art"), ("moral", "mor"), ("science", "sci")]:
    gen   = f"gen_{prefix}"
    val   = f"val_{prefix}"
    ev    = f"eval_{prefix}"
    save  = f"save_{prefix}"

    workflow.add_edge(gen, val)
    # Structural validation: pass → evaluation; fail → regenerate or give up
    workflow.add_conditional_edges(
        val,
        create_retry_logic(key),
        {"next": ev, "retry": gen, "fail": "mark_needs_human"}
    )
    # LLM evaluation: pass → save; fail → regenerate or give up
    workflow.add_conditional_edges(
        ev,
        create_post_eval_routing(key),
        {"save": save, "retry": gen, "fail": "mark_needs_human"}
    )
    workflow.add_edge(save, END)

workflow.add_edge("mark_needs_human", END)
workflow.add_edge("mark_completed", END)

# Use persistent checkpointer in production, MemorySaver for local dev
# Set USE_MEMORY_CHECKPOINTER=true for local development
if os.environ.get("USE_MEMORY_CHECKPOINTER", "false").lower() == "true":
    logger.info("Using MemorySaver checkpointer (development mode)")
    checkpointer = MemorySaver()
else:
    logger.info("Using FirestoreCheckpointer (production mode)")
    checkpointer = FirestoreCheckpointer()

app_workflow = workflow.compile(checkpointer=checkpointer)