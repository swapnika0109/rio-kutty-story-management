from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..workflows.master_workflow import master_workflow
from ..workflows.image_workflow import image_workflow
from ..workflows.audio_workflow import audio_workflow
from ..services.database.firestore_service import FirestoreService
from ..utils.logger import setup_logger
from ..utils.config import get_settings
from ..utils.tracing import get_trace_callbacks

logger = setup_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["media"])


class GenerateMediaRequest(BaseModel):
    story_id: str
    age: str
    language: str = "en"
    voice_type: Optional[str] = None  # "chirp" | "standard" (default: "standard")


class ResumeWorkflowRequest(BaseModel):
    thread_id: str
    decision: str  # "retry" | "skip" | "override"


class RegenerateImageRequest(BaseModel):
    age: Optional[str] = "3-4"
    language: Optional[str] = "en"


class RegenerateAudioRequest(BaseModel):
    language: Optional[str] = None   # defaults to story language
    voice: Optional[str] = None      # defaults to settings.TTS_VOICE_NAME


# ---------------------------------------------------------------------------
# Master workflow (image + audio → activities)
# ---------------------------------------------------------------------------

async def _run_master_workflow(request: GenerateMediaRequest):
    try:
        db = FirestoreService()
        story = await db.get_story(request.story_id)
        if not story:
            logger.error(f"Story {request.story_id} not found for media generation")
            return
        theme = story.get("theme", "theme1")
        config = {
            "configurable": {
                "thread_id": f"{request.story_id}_master",
                "story_id":  request.story_id,
                "age":       request.age,
                "language":  story.get("language", request.language),
                "theme":     theme,
                "voice":     request.voice_type,
            },
            "callbacks": get_trace_callbacks(
                name="master-pipeline",
                metadata={"story_id": request.story_id, "theme": theme,
                          "title": story.get("title"), "age": request.age},
                tags=["master", "image", "audio", "activities"],
                session_id=request.story_id,
            ),
        }
        initial_state = {
            "story_id":          request.story_id,
            "story":             story,
            "topics":            None,
            "workflow_statuses": {},
            "workflow_retries":  {},
            "human_loop_requests": {},
            "human_decisions":   {},
            "errors":            {},
        }
        await master_workflow.ainvoke(initial_state, config=config)
        logger.info(f"Master workflow completed for story {request.story_id}")
    except Exception as e:
        logger.exception(f"Master workflow failed for story {request.story_id}: {e}")


@router.post("/generate-media/{story_id}", status_code=202)
async def generate_media(story_id: str, request: GenerateMediaRequest, background_tasks: BackgroundTasks):
    """Triggers the full media pipeline: WF3 (image) + WF4 (audio) then WF5 (activities)."""
    request.story_id = story_id
    logger.info(f"Media generation triggered for story_id={story_id}")
    background_tasks.add_task(_run_master_workflow, request)
    return {"status": "accepted", "message": "Media generation started", "story_id": story_id}


# ---------------------------------------------------------------------------
# Individual image retrigger (human-in-loop bypass for WF3)
# ---------------------------------------------------------------------------

async def _run_image_workflow(story_id: str, age: str, language: str):
    try:
        db = FirestoreService()
        story = await db.get_story(story_id)
        if not story:
            logger.error(f"Story {story_id} not found for image generation")
            return
        theme = story.get("theme", "theme1")
        config = {
            "configurable": {
                "thread_id":   f"{story_id}_wf3",
                "story_id":    story_id,
                "story_text":  story.get("story_text", ""),
                "story_title": story.get("title", ""),
                "age":         age,
                "language":    story.get("language", language),
                "theme":       theme,
            },
            "callbacks": get_trace_callbacks(
                name="WF3-image",
                metadata={"story_id": story_id, "theme": theme, "title": story.get("title")},
                tags=["wf3", "image", "manual-retry"],
                session_id=story_id,
            ),
        }
        initial_state = {
            "story_text":  story.get("story_text", ""),
            "story_title": story.get("title", ""),
            "retry_count": 0,
            "status":      "pending",
            "completed":   [],
            "errors":      {},
        }
        result = await image_workflow.ainvoke(initial_state, config=config)
        status = result.get("status", "unknown")
        logger.info(f"Image workflow completed for story {story_id}: {status}")
    except Exception as e:
        logger.exception(f"Image workflow failed for story {story_id}: {e}")


@router.post("/generate-image/{story_id}", status_code=202)
async def generate_image(
    story_id: str,
    request: RegenerateImageRequest,
    background_tasks: BackgroundTasks,
):
    """
    Manually triggers WF3 (image generation) for a specific story.
    Use this to retry after a HITL interrupt, then resume the master workflow
    via POST /resume-workflow with decision='override'.
    """
    logger.info(f"Image retrigger for story_id={story_id}")
    background_tasks.add_task(
        _run_image_workflow, story_id, request.age, request.language
    )
    return {"status": "accepted", "message": "Image generation started", "story_id": story_id}


# ---------------------------------------------------------------------------
# Individual audio retrigger (human-in-loop bypass for WF4)
# ---------------------------------------------------------------------------

async def _run_audio_workflow(story_id: str, language: Optional[str], voice: Optional[str]):
    try:
        db = FirestoreService()
        story = await db.get_story(story_id)
        if not story:
            logger.error(f"Story {story_id} not found for audio generation")
            return
        theme    = story.get("theme", "theme1")
        lang     = language or story.get("language", settings.TTS_LANGUAGE_CODE)
        tts_voice = voice or settings.TTS_VOICE_NAME
        config = {
            "configurable": {
                "thread_id": f"{story_id}_wf4",
                "story_id":  story_id,
                "theme":     theme,
            },
            "callbacks": get_trace_callbacks(
                name="WF4-audio",
                metadata={"story_id": story_id, "theme": theme,
                          "language": lang, "title": story.get("title")},
                tags=["wf4", "audio", "manual-retry"],
                session_id=story_id,
            ),
        }
        initial_state = {
            "story_text":  story.get("story_text", ""),
            "language":    lang,
            "voice":       tts_voice,
            "retry_count": 0,
            "status":      "pending",
            "completed":   [],
            "errors":      {},
        }
        result = await audio_workflow.ainvoke(initial_state, config=config)
        status = result.get("status", "unknown")
        logger.info(f"Audio workflow completed for story {story_id}: {status}")
    except Exception as e:
        logger.exception(f"Audio workflow failed for story {story_id}: {e}")


@router.post("/generate-audio/{story_id}", status_code=202)
async def generate_audio(
    story_id: str,
    request: RegenerateAudioRequest,
    background_tasks: BackgroundTasks,
):
    """
    Manually triggers WF4 (audio generation) for a specific story.
    Use this to retry after a HITL interrupt, then resume the master workflow
    via POST /resume-workflow with decision='override'.
    """
    logger.info(f"Audio retrigger for story_id={story_id}")
    background_tasks.add_task(
        _run_audio_workflow, story_id, request.language, request.voice
    )
    return {"status": "accepted", "message": "Audio generation started", "story_id": story_id}


# ---------------------------------------------------------------------------
# Resume (HITL) and status
# ---------------------------------------------------------------------------

@router.post("/resume-workflow", status_code=202)
async def resume_workflow(request: ResumeWorkflowRequest):
    """Resumes a human-in-loop workflow after admin review. decision: retry | skip | override"""
    from langgraph.types import Command
    logger.info(f"Resuming workflow thread_id={request.thread_id} with decision={request.decision}")
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        await master_workflow.ainvoke(None, config=config, command=Command(resume=request.decision))
        return {"status": "resumed", "thread_id": request.thread_id, "decision": request.decision}
    except Exception as e:
        logger.error(f"Resume failed for thread_id={request.thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow-status/{story_id}")
async def workflow_status(story_id: str):
    """Returns status of all workflows for a story."""
    db = FirestoreService()
    return await db.get_workflow_status(story_id)
