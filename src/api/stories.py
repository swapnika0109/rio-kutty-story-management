import asyncio
import base64
import json
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from ..workflows.story_topics_workflow import story_topics_workflow
from ..workflows.story_creator_workflow import story_creator_workflow
from ..services.database.firestore_service import FirestoreService
from ..utils.logger import setup_logger
from ..utils.tracing import build_trace_config

logger = setup_logger(__name__)

router = APIRouter(tags=["stories"])


class GenerateTopicsRequest(BaseModel):
    country: Optional[str] = "Any"
    religion: Optional[list[str]] = ["Any"]
    age: str
    language: str = "en"
    theme: Optional[str] = None
    voice: Optional[str] = None
    preferences: Optional[list] = ["Any"]


class SelectTopicRequest(BaseModel):
    story_id: str
    selected_topic: dict


async def _run_topics_workflow(request: GenerateTopicsRequest):
    try:
        session_id = str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": f"{session_id}_wf1",
                "country": request.country,
                "religion": request.religion,
                "preferences": request.preferences,
                "age": request.age,
                "language": request.language,
                "theme": request.theme or "",
                "voice": request.voice or "",
            },
            **build_trace_config(
                name="WF1-topics",
                metadata={"age": request.age, "language": request.language,
                          "country": request.country, "theme": request.theme, "voice": request.voice},
                tags=["wf1", "topics"],
                session_id=session_id,
            ),
        }
        initial_state = {
            "topics": None,
            "validated": False,
            "evaluation": None,
            "correction_attempts": 0,
            "completed": [],
            "errors": {},
        }
        await story_topics_workflow.ainvoke(initial_state, config=config)
        logger.info(f"WF1 completed for topics")
    except Exception as e:
        logger.exception(f"WF1 failed for topics: {e}")


async def _run_story_workflow(story_id: str, selected_topic: dict, age: str, language: str):
    try:
        config = {
            "configurable": {
                "thread_id": f"{story_id}_wf2",
                "story_id": story_id,
                "age": age,
                "country": selected_topic.get("country", "Any"),
                "religion": selected_topic.get("religion", ["Any"]),
                "preferences": selected_topic.get("preferences", ["Any"]),
                "language": language,
                "voice": selected_topic.get("voice", ""),
            },
            **build_trace_config(
                name="WF2-story",
                metadata={"story_id": story_id, "topic": selected_topic.get("title"),
                          "theme": selected_topic.get("theme"), "age": age, "voice": selected_topic.get("voice")},
                tags=["wf2", "story"],
                session_id=story_id,
            ),
        }
        initial_state = {
            "selected_topic": selected_topic,
            "story": None,
            "validated": False,
            "evaluation": None,
            "correction_attempts": 0,
            "completed": [],
            "errors": {},
        }
        await story_creator_workflow.ainvoke(initial_state, config=config)
        logger.info(f"WF2 completed for story {story_id}")
    except Exception as e:
        logger.exception(f"WF2 failed for story {story_id}: {e}")


@router.post("/generate-topics", status_code=202)
async def generate_topics(request: GenerateTopicsRequest, background_tasks: BackgroundTasks):
    """WF1: Generate story topic options. Human selects one via POST /select-topic."""
    logger.info(f"WF1 triggered for topics")
    background_tasks.add_task(_run_topics_workflow, request)
    return {"status": "accepted", "message": "Story topics generation started"}


@router.post("/pubsub/generate-topics", status_code=204)
async def pubsub_generate_topics(request: Request, background_tasks: BackgroundTasks):
    """
    Pub/Sub push endpoint for topic generation.

    GCP Pub/Sub push subscription should point to:
        POST https://<your-cloud-run-url>/pubsub/generate-topics

    Message body must be a base64-encoded JSON matching GenerateTopicsRequest:
        {
          "age": "3-4",
          "language": "en",
          "country": "India",
          "religion": ["Hindu"],
          "theme": "2",
          "voice": null,
          "preferences": ["Any"]
        }
    """
    try:
        envelope = await request.json()
        message = envelope.get("message", {})
        data = message.get("data", "")
        payload = json.loads(base64.b64decode(data).decode("utf-8"))
        topics_request = GenerateTopicsRequest(**payload)
    except Exception as e:
        logger.error(f"[PubSub] Failed to parse message: {e}")
        # Return 400 so Pub/Sub does NOT retry a malformed message
        raise HTTPException(status_code=400, detail=f"Invalid Pub/Sub message: {e}")

    logger.info(f"[PubSub] generate-topics received: age={topics_request.age} theme={topics_request.theme}")
    background_tasks.add_task(_run_topics_workflow, topics_request)
    return  # 204 No Content — tells Pub/Sub the message was acknowledged


@router.post("/select-topic", status_code=202)
async def select_topic(request: SelectTopicRequest, background_tasks: BackgroundTasks):
    """Human picks a topic from WF1 output → triggers WF2 story generation."""
    logger.info(f"Topic selected for story_id={request.story_id}: {request.selected_topic.get('title', '?')}")
    db = FirestoreService()

    try:
        await db.set_selected_topic(request.story_id, request.selected_topic)
    except Exception as e:
        logger.error(f"Failed to record topic selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    story = await db.get_story(request.story_id)
    age = story.get("age", "3-4") if story else "3-4"
    language = story.get("language", "en") if story else "en"

    background_tasks.add_task(_run_story_workflow, request.story_id, request.selected_topic, age, language)
    return {"status": "accepted", "message": "Story generation started", "story_id": request.story_id}


# ---------------------------------------------------------------------------
# Pipeline resume
# A client interrupted before the master pipeline reached `finalize` can resume
# by sending the thread_id (== topic_id) stored in pending_workflows. The server
# inspects checkpoint state for the WF2 and master threads, decides which stage
# is still pending, and re-invokes the right workflow with the same thread_id
# so LangGraph picks up from the last completed node.
# ---------------------------------------------------------------------------

class ResumePipelineRequest(BaseModel):
    thread_id: str   # canonical pipeline root — equal to topic_id / story_id


# Per-thread_id locks so two concurrent /resume-pipeline requests for the same
# pipeline cannot race on checkpoint writes. In-process only (single-instance
# servers); for multi-instance deployments a Firestore-backed lease would be
# required, but the realistic risk here is double-clicks / accidental retries
# from one client.
_resume_locks: dict[str, asyncio.Lock] = {}
_resume_locks_guard = asyncio.Lock()


async def _get_resume_lock(thread_id: str) -> asyncio.Lock:
    async with _resume_locks_guard:
        lock = _resume_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            _resume_locks[thread_id] = lock
        return lock


async def _resume_pipeline(thread_id: str):
    """Background worker: figure out where the pipeline left off and continue."""
    lock = await _get_resume_lock(thread_id)
    if lock.locked():
        logger.info(f"[Resume] thread_id={thread_id} already resuming — skipping duplicate request")
        return
    async with lock:
        try:
            await _resume_pipeline_inner(thread_id)
        finally:
            # Drop the lock entry so the dict doesn't grow unboundedly across
            # the lifetime of the process.
            async with _resume_locks_guard:
                _resume_locks.pop(thread_id, None)


async def _resume_pipeline_inner(thread_id: str):
    """Inner implementation — runs under per-thread lock."""
    # Local imports — these modules pull heavy deps (langgraph, deepeval) that
    # we don't want to load at FastAPI startup.
    from ..workflows.story_creator_workflow import story_creator_workflow
    from ..workflows.master_workflow import master_workflow
    from ..utils.tracing import build_trace_config

    db = FirestoreService()
    pending = await db.get_pending_workflow(thread_id)
    if not pending:
        logger.warning(f"[Resume] No pending_workflows entry for thread_id={thread_id}")
        return

    topic    = pending.get("topic") or {}
    meta     = pending.get("meta") or {}
    age      = meta.get("age", "3-4")
    language = meta.get("language", "English")
    theme    = meta.get("theme", "")
    voice    = meta.get("voice")
    topics_id = meta.get("topics_id")
    story_id = thread_id   # canonical equality

    wf2_config = {
        "configurable": {
            "thread_id": f"{story_id}_wf2",
            "story_id":  story_id,
            "age":       age,
            "language":  language,
            "topics_id": topics_id,
            "theme":     theme,
        },
        **build_trace_config(
            name="WF2-story-resume",
            metadata={"story_id": story_id, "topic": topic.get("title"), "theme": theme, "age": age},
            tags=["wf2", "story", "resume"],
            session_id=topics_id or story_id,
        ),
    }
    master_config = {
        "configurable": {
            "thread_id": f"{story_id}_master",
            "story_id":  story_id,
            "age":       age,
            "language":  language,
            "theme":     theme,
            "voice":     voice,
        },
        **build_trace_config(
            name="master-pipeline-resume",
            metadata={"story_id": story_id, "topic": topic.get("title"), "theme": theme, "age": age},
            tags=["master", "resume"],
            session_id=topics_id or story_id,
        ),
    }

    # Probe master first — if it has any state, the pipeline was past WF2.
    try:
        master_state = await master_workflow.aget_state(master_config)
    except Exception as e:
        logger.error(f"[Resume] Failed to read master checkpoint for {story_id}: {e}")
        master_state = None

    if master_state is not None and master_state.next:
        logger.info(f"[Resume] {story_id}: resuming master at nodes={master_state.next}")
        try:
            # Passing None tells LangGraph to resume the existing checkpoint as-is
            # (no new input to seed the entry node).
            await master_workflow.ainvoke(None, config=master_config)
        except Exception as e:
            logger.exception(f"[Resume] Master resume failed for {story_id}: {e}")
        return

    # Master had no pending state — either it never started or it already finished.
    # Either way, inspect WF2's checkpoint to decide.
    try:
        wf2_state = await story_creator_workflow.aget_state(wf2_config)
    except Exception as e:
        logger.error(f"[Resume] Failed to read WF2 checkpoint for {story_id}: {e}")
        wf2_state = None

    if wf2_state is not None and wf2_state.next:
        logger.info(f"[Resume] {story_id}: resuming WF2 at nodes={wf2_state.next}, then starting master")
        try:
            await story_creator_workflow.ainvoke(None, config=wf2_config)
        except Exception as e:
            logger.exception(f"[Resume] WF2 resume failed for {story_id}: {e}")
            return

    # WF2 either resumed-to-completion or had no pending nodes already.
    # Master never started (otherwise we'd have taken the master-resume branch
    # above) — kick it off with the persisted story.
    story = await db.get_story(story_id, theme)
    if not story or not (story.get("story_text") or "").strip():
        logger.warning(f"[Resume] {story_id}: no usable story doc — cannot start master")
        return

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
    logger.info(f"[Resume] {story_id}: starting master fresh after WF2")
    try:
        await master_workflow.ainvoke(master_initial, config=master_config)
    except Exception as e:
        logger.exception(f"[Resume] Master start-fresh failed for {story_id}: {e}")


@router.post("/resume-pipeline", status_code=202)
async def resume_pipeline(request: ResumePipelineRequest, background_tasks: BackgroundTasks):
    """Resume an interrupted WF2→Master pipeline by its thread_id (== topic_id).

    Returns 404 if the thread_id is unknown (already completed or never registered).
    Returns 202 once the resume has been queued; actual work runs in the background.
    """
    db = FirestoreService()
    pending = await db.get_pending_workflow(request.thread_id)
    if not pending:
        raise HTTPException(
            status_code=404,
            detail=f"No pending workflow found for thread_id={request.thread_id}",
        )

    background_tasks.add_task(_resume_pipeline, request.thread_id)
    return {
        "status": "accepted",
        "message": "Pipeline resume started",
        "thread_id": request.thread_id,
    }
