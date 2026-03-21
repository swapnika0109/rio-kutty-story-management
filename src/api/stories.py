from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from ..workflows.story_topics_workflow import story_topics_workflow
from ..workflows.story_creator_workflow import story_creator_workflow
from ..services.database.firestore_service import FirestoreService
from ..utils.logger import setup_logger
from ..utils.tracing import get_trace_callbacks

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
            "callbacks": get_trace_callbacks(
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
            "callbacks": get_trace_callbacks(
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
