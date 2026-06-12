from typing import TypedDict, List, Dict, Any, Annotated
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

# Import shared reducers — defined once in models.state to avoid duplication
from ..models.state import merge_dicts, union_list

# Reducer for status: prefer "needs_human" over "completed" (error takes precedence)
def status_reducer(left: str, right: str) -> str:
    """Status reducer: "needs_human" takes precedence over "completed"."""
    if left == "needs_human" or right == "needs_human":
        return "needs_human"
    return right or left

# State Definition for WF5 activities subgraph
class ActivityState(TypedDict):
    # story_id, story_text, age, language are in config.configurable (read-only)
    activities: Annotated[Dict[str, Any], merge_dicts]
    images: Annotated[Dict[str, str], merge_dicts]
    # union_list (not operator.add): activities re-run on Gemini 503s, and a
    # plain-append accumulator re-added each completed step every retry until the
    # list hit ~470KB and blew the 1MB Firestore checkpoint limit. Dedup fixes it.
    completed: Annotated[List[str], union_list]
    errors: Annotated[Dict[str, str], merge_dicts]
    retry_count: Annotated[Dict[str, int], merge_dicts]
    # Subgraph result status reported back to master: "completed" | "needs_human"
    # Uses status_reducer so concurrent writes prefer "needs_human" (error state)
    status: Annotated[str, status_reducer]

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
        "mcq_seeds":              cfg.get("mcq_seeds", []),
        "art_seed":               cfg.get("art_seed", ""),
        "science_concepts":       cfg.get("science_concepts", []),
        "moral":                  cfg.get("moral", ""),
        # Topic-level anchors carried through for activity prompts
        "science_angle":          cfg.get("science_angle", ""),
        "daily_life_application": cfg.get("daily_life_application", ""),
        "story_title":            cfg.get("story_title", ""),
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
# Module-level lock. Python 3.10+ asyncio.Lock is loop-agnostic at construction
# time (binds on first acquire), so a single shared instance works across all
# WF5 invocations in the same event loop. Tests that need isolation should
# reset this fixture-scope.
#
# IMPORTANT: Do NOT re-introduce a "rebind to current loop" helper using
# getattr(lock, "_loop", default). On Python 3.11 asyncio.Lock has _loop=None
# at construction; that getattr returns None which compares unequal to the
# running loop, causing a fresh lock on EVERY call and breaking serialisation.
# This was the WF5 burst-503 cause in production.
_activity_eval_lock = asyncio.Lock()


def _get_activity_eval_lock() -> asyncio.Lock:
    return _activity_eval_lock


async def _evaluate_activity(state: ActivityState, config: RunnableConfig, activity_type: str):
    """Run the evaluator on one activity. Always writes the verdict under
    activities[_eval_<type>] so the post-eval router can decide retry vs save,
    and so the next gen pass can read metric_reasons to correct itself."""
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
    # Always write the eval result keyed by `_eval_<type>` so the post-eval
    # router can read the LATEST outcome and the next gen pass can read the
    # per-metric reasons to correct itself. Using merge_dicts on `activities`
    # means last-write-wins per key — a pass after a prior fail overwrites.
    per_activity = (evaluation.get("per_activity") or {}).get(activity_type, {})
    eval_record = {
        "passed": passed,
        "score": evaluation.get("score"),
        "reason": evaluation.get("reason"),
        # Both metrics (scores) and metric_reasons are stored so the retry
        # feedback prompt can show ONLY the metrics that actually failed
        # (filtered by score vs threshold, not by parsing reason strings).
        "metrics": per_activity.get("metrics") or {},
        "metric_reasons": per_activity.get("metric_reasons") or {},
    }
    if passed:
        logger.info(f"[WF5/{activity_type}] Evaluation PASSED score={evaluation.get('score')}")
        return {"activities": {f"_eval_{activity_type}": eval_record}}
    logger.warning(
        f"[WF5/{activity_type}] Evaluation FAILED score={evaluation.get('score')} "
        f"reason={evaluation.get('reason')}"
    )
    # Note: retry_count is bumped by the generator node on each re-entry, so we
    # do NOT bump it here — double-incrementing would skew the retry cap.
    return {"activities": {f"_eval_{activity_type}": eval_record}}

async def evaluate_mcq_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "mcq")

async def evaluate_art_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "art")

async def evaluate_moral_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "moral")

async def evaluate_science_node(state: ActivityState, config: RunnableConfig):
    return await _evaluate_activity(state, config, "science")

# --- Image Generation Nodes ---
# Images are generated AFTER evaluation passes so we don't burn FLUX credits
# on activities that fail eval and get regenerated. MCQ has no image.
async def image_art_node(state: ActivityState, config: RunnableConfig):
    return await art_agent.generate_image(unpack_config(state, config))

async def image_moral_node(state: ActivityState, config: RunnableConfig):
    return await moral_agent.generate_image(unpack_config(state, config))

async def image_science_node(state: ActivityState, config: RunnableConfig):
    return await science_agent.generate_image(unpack_config(state, config))

# --- Save Nodes ---
async def save_mcq_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    result = {}
    if "mcq" in state.get("activities", {}):
        data = state["activities"]["mcq"]
        logger.info(f"Saving MCQ for story {db_data['story_id']}: {data}")
        await firestore_service.save_activity(db_data["story_id"], "mcq", data)
        result["completed"] = state.get("completed", []) + ["mcq"]
    return result

async def save_art_node(state: ActivityState, config: RunnableConfig):
    # NOTE: image is already a GCS filename string by this point — the activity
    # agents upload during generation so raw PNG bytes never live in state.
    db_data = unpack_config(state, config)
    result = {}
    if "art" in state.get("activities", {}):
        data = state["activities"]["art"]
        payload = {"items": data}
        await firestore_service.save_activity(db_data["story_id"], "art", payload)
        result["completed"] = state.get("completed", []) + ["art"]
    return result


async def save_science_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    result = {}
    if "science" in state.get("activities", {}):
        data = state["activities"]["science"][0]
        payload = {**data} if isinstance(data, dict) else {"items": data}
        await firestore_service.save_activity(db_data["story_id"], "science", payload)
        result["completed"] = state.get("completed", []) + ["science"]
    return result


async def save_moral_node(state: ActivityState, config: RunnableConfig):
    db_data = unpack_config(state, config)
    result = {}
    if "moral" in state.get("activities", {}):
        data_list = state["activities"]["moral"]
        payloads = [
            ({**data} if isinstance(data, dict) else {"items": data})
            for data in data_list
        ]
        await firestore_service.save_activity(db_data["story_id"], "moral", payloads)
        result["completed"] = state.get("completed", []) + ["moral"]
    return result

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

    # If every activity already exists in DB, fan out to the terminal
    # `mark_completed` node so WF5 reports status="completed" to master.
    # Returning [] here lets LangGraph drop straight to END with the initial
    # status="pending" still in state — which master then mis-reads as success.
    if not nodes_to_run:
        logger.info(f"[WF5] All activities already exist for {story_id} — marking completed.")
        return ["mark_completed"]
    return nodes_to_run

# Terminal node when activities exhaust retries — reports needs_human to master
def mark_activities_needs_human(state: ActivityState):
    failed = list(state.get("errors", {}).keys())
    logger.error(f"[WF5] Activities failed after {MAX_ACTIVITY_RETRIES} retries: {failed}")
    return {"status": "needs_human"}

# Join node: waits for all parallel activity branches to either complete
# (be in activities/completed) or fail (be in errors). This ensures we don't prematurely
# terminate the subgraph when one branch finishes while others are still retrying.
def activities_join(state: ActivityState):
    """Synchronization point that waits for all activities to reach terminal state.

    An activity is terminal when:
    1. It's in 'completed' (passed eval and was saved)
    2. It's in 'errors' (failed generation/validation)
    3. It's in state['activities'] with an '_eval_X' record showing it failed eval after retries

    Once all are terminal, route to either mark_needs_human (if any errors) or mark_completed.
    """
    activities = state.get("activities", {})
    errors = state.get("errors", {})
    completed = state.get("completed", [])

    # Check each activity type
    pending = []
    for activity_type in ["mcq", "art", "moral", "science"]:
        in_error = activity_type in errors
        in_completed = activity_type in completed
        eval_record = activities.get(f"_eval_{activity_type}", {})

        # Activity is terminal if in errors, completed, or has a failed eval record
        is_terminal = in_error or in_completed or bool(eval_record and not eval_record.get("passed"))

        if not is_terminal:
            pending.append(activity_type)

    if pending:
        logger.debug(f"[join] Waiting on activities: {pending}")
        return state  # Keep waiting - return unchanged state

    # All activities are terminal
    logger.info("[join] All activities terminal; routing to completion")

    # If any errors, route to needs_human; otherwise mark_completed
    if errors:
        logger.error(f"[join] Activities have errors: {list(errors.keys())} - need human intervention")
        return {"status": "needs_human"}
    return {"status": "completed"}

# Mark all-completed terminal node. Only sets status="completed" if no
# parallel branch has already set "needs_human" — any failure wins over success.
def mark_activities_completed(state: ActivityState):
    if state.get("status") == "needs_human":
        return {}
    return {"status": "completed"}


# --- Graph Construction ---
workflow = StateGraph(ActivityState)

# Add Nodes
workflow.add_node("start", lambda s: s)  # Dummy start node
workflow.add_node("join", activities_join)  # Wait for all branches to reach terminal state
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
workflow.add_node("img_art", image_art_node)
workflow.add_node("save_art", save_art_node)

# Activity 3: Moral
workflow.add_node("gen_mor", generate_moral_node)
workflow.add_node("val_mor", validate_moral_node)
workflow.add_node("eval_mor", evaluate_moral_node)
workflow.add_node("img_mor", image_moral_node)
workflow.add_node("save_mor", save_moral_node)

# Activity 4: Science
workflow.add_node("gen_sci", generate_science_node)
workflow.add_node("val_sci", validate_science_node)
workflow.add_node("eval_sci", evaluate_science_node)
workflow.add_node("img_sci", image_science_node)
workflow.add_node("save_sci", save_science_node)

# Entry & Fan-out (Dynamic)
workflow.set_entry_point("start")
workflow.add_conditional_edges(
    "start",
    route_start,
    ["gen_mcq", "gen_art", "gen_mor", "gen_sci", "mark_completed"]
)

# Define Flows (Standardized: Gen -> Val -> Eval -> [Image] -> Save)
# MCQ has no image; art/moral/science route through an image node AFTER eval
# passes so we don't burn FLUX credits on retried activities.
for key, prefix in [("mcq", "mcq"), ("art", "art"), ("moral", "mor"), ("science", "sci")]:
    gen   = f"gen_{prefix}"
    val   = f"val_{prefix}"
    ev    = f"eval_{prefix}"
    save  = f"save_{prefix}"
    has_image = key != "mcq"
    post_eval_pass = f"img_{prefix}" if has_image else save

    workflow.add_edge(gen, val)
    # Structural validation: pass → evaluation; fail → regenerate or give up
    workflow.add_conditional_edges(
        val,
        create_retry_logic(key),
        {"next": ev, "retry": gen, "fail": "join"}  # Failed validation → join (wait for others)
    )
    # LLM evaluation: pass → image-gen (or save for MCQ); fail → regenerate or give up
    workflow.add_conditional_edges(
        ev,
        create_post_eval_routing(key),
        {"save": post_eval_pass, "retry": gen, "fail": "join"}  # Failed eval → join (wait for others)
    )
    if has_image:
        workflow.add_edge(f"img_{prefix}", save)
    # Every save converges on join node (synchronization point) so we wait for
    # all branches to reach terminal state before moving to mark_completed.
    # This prevents early termination when one branch completes while others retry.
    workflow.add_edge(save, "join")

# Join routes based on whether there are errors
def route_from_join(state: ActivityState):
    if state.get("status") == "needs_human":
        return "needs_human"
    return "completed"

workflow.add_conditional_edges(
    "join",
    route_from_join,
    {"needs_human": "mark_needs_human", "completed": "mark_completed"}
)
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