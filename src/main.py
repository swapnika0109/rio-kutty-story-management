from fastapi import FastAPI, BackgroundTasks, HTTPException, Response, status
from pydantic import BaseModel
import uvicorn
import base64
import json
import os
from .workflows.activity_workflow import app_workflow
from .services.database.firestore_service import FirestoreService
from .utils.logger import setup_logger

logger = setup_logger(__name__)

logger.info("Starting application...")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete.")

class ActivityRequest(BaseModel):
    story_id: str
    age: int
    language: str = "en"

class PubSubMessage(BaseModel):
    message: dict = None
    subscription: str = None
    data: str = None

async def run_workflow(request: ActivityRequest):
    """Background task to run the LangGraph workflow"""
    try:
        # Get the story from the database
        story = await FirestoreService().get_story(request.story_id)
        if not story:
            logger.error(f"Story with ID {request.story_id} not found")
            raise Exception(f"Story {request.story_id} not found")

        config = {
            "configurable": {
                "thread_id": request.story_id,
                "story_id": request.story_id,
                "story_text": story.get("story_text", ""),
                "age": request.age,
                "language": story.get("language", "en"),
            }
        }
        
        initial_state = {
            "activities": {},
            "completed": [],
            "errors": {},
            "retry_count": {}
        }
        
        # Invoke the workflow
        # Note: In production, use a persistent checkpointer (e.g. Postgres) 
        # instead of MemorySaver to handle restarts.
        await app_workflow.ainvoke(initial_state, config=config)
        logger.info(f"Workflow completed for story {request.story_id}")
        
    except Exception as e:
        logger.exception(f"Workflow failed for story {request.story_id}: {e}")
        

@app.post("/generate-activities")
async def generate_activities(request: ActivityRequest, background_tasks: BackgroundTasks):
    """
    Endpoint called by the Go backend.
    Returns immediately (202 Accepted) and processes in background.
    """
    logger.info(f"Received request for story {request.story_id}")
    background_tasks.add_task(run_workflow, request)
    return {"status": "accepted", "message": "Activity generation started", "story_id": request.story_id}

@app.post("/pubsub-handler")
async def pubsub_handler(pubsub_msg: PubSubMessage, background_tasks: BackgroundTasks):
    """
    Endpoint called by the Go backend.
    Returns immediately (202 Accepted) and processes in background.
    """
    logger.info(f"Received request for pubsub activity generation {pubsub_msg}")
    
    # Handle both wrapped and direct data formats
    data = None
    if pubsub_msg.data:
        data = pubsub_msg.data
    elif pubsub_msg.message and "data" in pubsub_msg.message:
        data = pubsub_msg.message["data"]
        
    if not data:
        logger.error("No data found in pubsub message")
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    
    try:
        decoded_data = base64.b64decode(data).decode("utf-8")
        logger.info(f"Received Message: {decoded_data}")
        data_json = json.loads(decoded_data)
        
        # Parse activity request
        activity_request = ActivityRequest(
            story_id=data_json["story_id"],
            age=data_json["age"],
            language=data_json.get("language", "en")
        )
        
        background_tasks.add_task(run_workflow, activity_request)
        return Response(status_code=status.HTTP_202_ACCEPTED)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}. Data: {decoded_data}")
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Error processing pubsub message: {e}")
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/task-status/{story_id}")
async def get_task_status(story_id: str):
    """
    Get the status of activity generation tasks for a story.
    Returns information about which activities are completed, pending, or failed.
    """
    logger.info(f"Checking task status for story {story_id}")
    
    task_status = await FirestoreService().get_task_status(story_id)
    
    if not task_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Story {story_id} not found"
        )
    
    return task_status

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)