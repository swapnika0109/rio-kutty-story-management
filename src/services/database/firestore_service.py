from google.cloud import firestore
from ...utils.config import get_settings
from ...utils.logger import setup_logger
from google.cloud.firestore_v1.base_query import FieldFilter
settings = get_settings()
logger = setup_logger(__name__)

class FirestoreService:
    def __init__(self):
        self._db = None

    @property
    def db(self):
        if self._db is None:
            project = settings.GOOGLE_CLOUD_PROJECT
            database = settings.FIRESTORE_DATABASE
            
            logger.info(f"Initializing Firestore client for project: {project}, database: {database}")
            
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                logger.info(f"Using service account key from: {settings.GOOGLE_APPLICATION_CREDENTIALS}")
                self._db = firestore.Client.from_service_account_json(
                    json_credentials_path=settings.GOOGLE_APPLICATION_CREDENTIALS,
                    project=project, 
                    database=database
                )
            else:
                logger.info("Using default Google Application Credentials (ADC)")
                # When running in Cloud Run, we should try to let it auto-detect project if not explicitly set
                self._db = firestore.Client(
                    project=project, 
                    database=database
                )
        return self._db
            
    async def check_if_activity_exists(self, story_id: str, activity_type: str):
        """
        Checks if an activity already exists for a story.
        """
        try:
            activity_query = self.db.collection('activities_v1') \
                .where(filter=FieldFilter("story_id", "==", story_id)) \
                .where(filter=FieldFilter("type", "==", activity_type)) \
                .limit(1)
            doc_stream = activity_query.stream()
            first_doc = next(doc_stream, None)
            if first_doc:
                return first_doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Firestore check if activity exists failed: {str(e)}")
            return None

    async def save_activity(self, story_id: str, activity_type: str, activity_data: list):
        """
        Saves an activity and updates the story status in a single batch.
        """
        try:

            #check if activity_data has already for this tory id and this activity type
            activity_query = self.db.collection('activities_v1').where("story_id", "==", story_id).where("type", "==", activity_type).limit(1)
            docs = list(activity_query.stream())
            if docs:
                logger.info(f"Activity with type {activity_type} already exists for story {story_id}")
                return

            # 1. Find the story document by querying the 'story_id' field
            # (Since document ID != story_id)
            story_query = self.db.collection('riostories_v3').where("story_id", "==", story_id).limit(1)
            docs = list(story_query.stream())
            
            if not docs:
                logger.error(f"Story with ID {story_id} not found in riostories_v3")
                raise Exception(f"Story {story_id} not found")
            
            story_ref = docs[0].reference

            # 2. Prepare Batch
            batch = self.db.batch()
            
            # 3. Save the actual activity data
            activity_ref = self.db.collection('activities_v1').document() # Auto-ID
            
            # Handle different data types for activity_data
            if isinstance(activity_data, list):
                data_to_save = {'items': activity_data}
            elif isinstance(activity_data, dict):
                data_to_save = activity_data
            else:
                data_to_save = {'data': activity_data}

            batch.set(activity_ref, {
                **data_to_save,
                'story_id': story_id,
                'type': activity_type,
                'created_at': firestore.SERVER_TIMESTAMP
            })

            # 4. Update the parent story status
            batch.update(story_ref, {
                f'activities.{activity_type}': 'ready',
                'updated_at': firestore.SERVER_TIMESTAMP
            })

            # Commit both atomically
            batch.commit()
            logger.info(f"Saved {activity_type} for story {story_id}")
            
        except Exception as e:
            logger.error(f"Firestore save failed: {str(e)}")
            raise e

    async def get_story(self, story_id: str):
        """
        Gets a story from the database by querying the story_id field.
        """
        try:
            query = self.db.collection('riostories_v3').where("story_id", "==", story_id).limit(1)
            docs = list(query.stream())
            
            if not docs:
                return None
                
            return docs[0].to_dict()

        except Exception as e:
            logger.error(f"Firestore get failed: {str(e)}")
            return None

    async def get_task_status(self, story_id: str):
        """
        Gets the status of activity generation tasks for a story.
        Returns information about which activities are completed, pending, or failed.
        """
        try:
            # Get the story document
            story = await self.get_story(story_id)
            if not story:
                return None
            
            # Get all activity types that should be generated
            activity_types = ["mcq", "art", "moral", "science"]
            
            # Check activities field from story
            story_activities = story.get("activities", {})
            
            # Build status response
            status = {
                "story_id": story_id,
                "activities": {}
            }
            
            for activity_type in activity_types:
                activity_status = story_activities.get(activity_type, "pending")
                status["activities"][activity_type] = activity_status
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get task status: {str(e)}")
            return None