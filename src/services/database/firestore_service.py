from google.cloud import firestore
from ...utils.config import get_settings
from ...utils.logger import setup_logger
from google.cloud.firestore_v1.base_query import FieldFilter

settings = get_settings()
logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Collection name maps
# Topics:  planet_protectors_topics / mindful_topics / chill_stories_topics
# Stories: planet_protectors_stories / mindful_stories / chill_stories
# Activities: activities_v1 (unchanged, tagged with story_id)
# ---------------------------------------------------------------------------

_TOPIC_COLLECTIONS = {
    "theme1": "planet_protectors_topics",
    "theme2": "mindful_topics",
    "theme3": "chill_stories_topics",
}

_STORY_COLLECTIONS = {
    "theme1": "planet_protectors_stories",
    "theme2": "mindful_stories",
    "theme3": "chill_stories",
}


class FirestoreService:
    def __init__(self):
        self._db = None

    @property
    def db(self):
        if self._db is None:
            project = settings.GOOGLE_CLOUD_PROJECT
            database = settings.FIRESTORE_DATABASE
            logger.info(f"Initializing Firestore: project={project} database={database}")
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                self._db = firestore.Client.from_service_account_json(
                    json_credentials_path=settings.GOOGLE_APPLICATION_CREDENTIALS,
                    project=project,
                    database=database,
                )
            else:
                self._db = firestore.Client(project=project, database=database)
        return self._db

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topic_collection(theme: str) -> str:
        return _TOPIC_COLLECTIONS.get(theme, "planet_protectors_topics")

    @staticmethod
    def _story_collection(theme: str) -> str:
        return _STORY_COLLECTIONS.get(theme, "planet_protectors_stories")

    # ------------------------------------------------------------------
    # Activities  (activities_v1 — unchanged)
    # ------------------------------------------------------------------

    async def check_if_activity_exists(self, story_id: str, activity_type: str):
        try:
            query = (
                self.db.collection("activities_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .where(filter=FieldFilter("type", "==", activity_type))
                .limit(1)
            )
            doc = next(query.stream(), None)
            return doc.to_dict() if doc else None
        except Exception as e:
            logger.error(f"check_if_activity_exists failed: {e}")
            return None

    async def save_activity(self, story_id: str, activity_type: str, activity_data) -> None:
        """
        Saves an activity to activities_v1 and sets activities.{type}='ready'
        on the story document. Story is looked up across all theme collections
        using story_id as the document ID (O(1) direct reads, no query).
        """
        try:
            # Deduplicate
            dup_q = (
                self.db.collection("activities_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .where(filter=FieldFilter("type", "==", activity_type))
                .limit(1)
            )
            if next(dup_q.stream(), None):
                logger.info(f"Activity {activity_type} already exists for story {story_id}")
                return

            # Find story ref — story_id IS the doc ID in theme collections
            story_ref = None
            for col in _STORY_COLLECTIONS.values():
                ref = self.db.collection(col).document(story_id)
                if ref.get().exists:
                    story_ref = ref
                    break

            if story_ref is None:
                logger.error(f"Story {story_id} not found in any theme collection")
                raise ValueError(f"Story {story_id} not found")

            if isinstance(activity_data, list):
                data_to_save = {"items": activity_data}
            elif isinstance(activity_data, dict):
                data_to_save = activity_data
            else:
                data_to_save = {"data": activity_data}

            batch = self.db.batch()
            activity_ref = self.db.collection("activities_v1").document()
            batch.set(activity_ref, {
                **data_to_save,
                "story_id": story_id,
                "type": activity_type,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            batch.update(story_ref, {
                f"activities.{activity_type}": "ready",
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            batch.commit()
            logger.info(f"Saved activity {activity_type} for story {story_id}")
        except Exception as e:
            logger.error(f"save_activity failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Stories  (rio_stories_theme{N} — story_id as doc ID)
    # ------------------------------------------------------------------

    async def get_story(self, story_id: str, theme: str | None = None) -> dict | None:
        """
        Direct doc lookup (story_id = doc ID). Searches the given theme collection
        first; if not found (or no theme given) tries all 3.
        """
        try:
            cols = [self._story_collection(theme)] if theme else list(_STORY_COLLECTIONS.values())
            for col in cols:
                doc = self.db.collection(col).document(story_id).get()
                if doc.exists:
                    return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"get_story failed: {e}")
            return None

    async def get_story_by_title(self, title: str, theme: str | None = None) -> dict | None:
        """
        Queries story collections for a doc whose 'title' field matches exactly.
        Returns the first match, or None. Used to detect already-generated stories
        when the topic doc hasn't had story_id patched in yet.
        """
        try:
            cols = [self._story_collection(theme)] if theme else list(_STORY_COLLECTIONS.values())
            for col in cols:
                docs = (
                    self.db.collection(col)
                    .where("title", "==", title)
                    .limit(1)
                    .get()
                )
                if docs:
                    return docs[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"get_story_by_title failed for '{title}': {e}")
            return None

    async def save_story(
        self,
        story_id: str,
        story: dict,
        theme: str,
        topics_id: str | None = None,
        topic_id: str | None = None,
        topic_document_id: str | None = None,
    ) -> None:
        """Saves/upserts story to the theme story collection with story_id as document ID."""
        try:
            col = self._story_collection(theme)
            payload = {
                **story,
                "story_id":   story_id,
                "theme":      theme,
                "title":      story.get("title", ""),
                "description": story.get("description", story.get("story_seed", "")),
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
            if topics_id:
                payload["topics_id"] = topics_id
            if topic_id:
                payload["topic_id"] = topic_id
            if topic_document_id:
                payload["topic_document_id"] = topic_document_id
            self.db.collection(col).document(story_id).set(payload, merge=True)
            logger.info(f"[Firestore] Story saved: {col}/{story_id}")
        except Exception as e:
            logger.error(f"save_story failed: {e}")
            raise

    async def save_story_image(
        self, story_id: str, image_url: str, generation_prompt: str, theme: str
    ) -> None:
        """Updates image_url and image_prompt on the story doc."""
        try:
            col = self._story_collection(theme)
            self.db.collection(col).document(story_id).update({
                "image_url":    image_url,
                "image_prompt": generation_prompt,
                "updated_at":   firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"[Firestore] image_url saved on {col}/{story_id}")
        except Exception as e:
            logger.error(f"save_story_image failed: {e}")
            raise

    async def save_story_audio(
        self,
        story_id: str,
        audio_url: str,
        language: str,
        voice: str,
        theme: str,
        audio_timepoints: list | None = None,
    ) -> None:
        """Updates audio_url, voice, and audio_timepoints on the story doc."""
        try:
            col = self._story_collection(theme)
            story_update = {
                "audio_url":  audio_url,
                "voice":      voice,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
            if audio_timepoints:
                story_update["audio_timepoints"] = audio_timepoints
            self.db.collection(col).document(story_id).update(story_update)
            logger.info(f"[Firestore] audio_url saved on {col}/{story_id}")
        except Exception as e:
            logger.error(f"save_story_audio failed: {e}")
            raise

    # ------------------------------------------------------------------
    # WF1 session log  (story_topics_v1 — reference only, not the title lib)
    # ------------------------------------------------------------------

    async def save_story_topics(self, story_id: str, topics: list) -> None:
        try:
            self.db.collection("story_topics_v1").document().set({
                "story_id": story_id,
                "topics": topics,
                "selected_topic": None,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"[Firestore] Topics session saved for story_id={story_id}")
        except Exception as e:
            logger.error(f"save_story_topics failed: {e}")
            raise

    async def get_story_topics(self, story_id: str) -> dict | None:
        try:
            q = (
                self.db.collection("story_topics_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .limit(1)
            )
            docs = list(q.stream())
            return docs[0].to_dict() if docs else None
        except Exception as e:
            logger.error(f"get_story_topics failed: {e}")
            return None

    async def set_selected_topic(self, story_id: str, selected_topic: dict) -> None:
        try:
            q = (
                self.db.collection("story_topics_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .limit(1)
            )
            docs = list(q.stream())
            if not docs:
                raise ValueError(f"No topics doc for story_id={story_id}")
            docs[0].reference.update({
                "selected_topic": selected_topic,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
        except Exception as e:
            logger.error(f"set_selected_topic failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Topic Library  (planet_protectors_topics / mindful_topics / chill_stories_topics)
    # Doc ID: {age_norm}__{lang}__{filter_value_norm}
    # Each doc contains a topics_id UUID and a list of topic dicts.
    # ------------------------------------------------------------------

    @staticmethod
    def _library_doc_id(age: str, lang: str, filter_value: str) -> str:
        """Deterministic Firestore-safe doc ID."""
        import re
        norm = re.sub(r"[^a-z0-9]+", "_", filter_value.lower()).strip("_")
        return f"{age.replace('-', '_')}__{lang}__{norm}"

    async def get_title_library_entry(
        self, theme: str, age: str, lang: str, filter_value: str
    ) -> list | None:
        """
        Returns cached topics from the theme topic collection, or None on miss.
        Re-injects filter_type and filter_value (stored at doc level) into each
        topic dict so WF1 batch routing still has them in the flat topic list.
        """
        try:
            col = self._topic_collection(theme)
            doc_id = self._library_doc_id(age, lang, filter_value)
            doc = self.db.collection(col).document(doc_id).get()
            if doc.exists:
                data = doc.to_dict()
                topics = data.get("topics", [])
                doc_filter_type  = data.get("filter_type", "")
                doc_filter_value = data.get("filter_value", filter_value)
                doc_theme        = data.get("theme", theme)
                import uuid as _uuid
                # Re-inject doc-level fields so the in-memory topic list is complete.
                # Back-fill topic_id for legacy docs that predate this field and persist
                # it immediately so the same UUID is returned on every subsequent read.
                needs_backfill = False
                enriched = []
                for t in topics:
                    if not t.get("topic_id"):
                        t = {**t, "topic_id": str(_uuid.uuid4())}
                        needs_backfill = True
                    enriched.append({
                        **t,
                        "theme":        doc_theme,
                        "filter_type":  doc_filter_type,
                        "filter_value": doc_filter_value,
                    })

                if needs_backfill:
                    # Write back only the topics array so topic_ids are stable forever.
                    clean = [{k: v for k, v in t.items() if k not in {"theme", "filter_type", "filter_value"}} for t in enriched]
                    self.db.collection(col).document(doc_id).update({"topics": clean})
                    logger.info(f"[Firestore] Back-filled topic_ids in {col}/{doc_id}")

                logger.info(f"[Firestore] Cache hit: {col}/{doc_id} ({len(enriched)} topics)")
                return enriched
            return None
        except Exception as e:
            logger.error(f"get_title_library_entry failed: {e}")
            return None

    async def get_all_topic_titles(self, age: str, lang: str) -> set[str]:
        """
        Returns a set of all topic titles already stored across all theme collections
        for the given age + language. Used globally to prevent duplicate titles.
        """
        titles: set[str] = set()
        try:
            doc_id_prefix = self._library_doc_id(age, lang, "").rstrip("_")
            for col in _TOPIC_COLLECTIONS.values():
                docs = self.db.collection(col).list_documents()
                for doc_ref in docs:
                    doc_id = doc_ref.id
                    if not doc_id.startswith(doc_id_prefix):
                        continue
                    doc = doc_ref.get()
                    if doc.exists:
                        for t in doc.to_dict().get("topics", []):
                            if t.get("title"):
                                titles.add(t["title"].lower())
        except Exception as e:
            logger.error(f"get_all_topic_titles failed: {e}")
        return titles

    async def save_title_library_entry(
        self,
        theme: str,
        age: str,
        lang: str,
        filter_type: str,
        filter_value: str,
        titles: list,
        topics_id: str | None = None,
    ) -> None:
        """
        Upserts generated topics into the theme topic collection.
        filter_type and filter_value are stored once at the doc level; they are
        stripped from each topic item to avoid repetition.
        """
        import uuid as _uuid
        # Strip per-topic fields stored at doc level to avoid repetition.
        # Preserve story_id and topic_id — they must survive re-saves.
        _strip = {"filter_type", "filter_value"}
        clean_topics = [{k: v for k, v in t.items() if k not in _strip} for t in titles]
        try:
            col = self._topic_collection(theme)
            doc_id = self._library_doc_id(age, lang, filter_value)
            doc_ref = self.db.collection(col).document(doc_id)

            # Preserve topic_id and story_id from existing docs so IDs are stable across re-saves.
            existing_doc = doc_ref.get()
            if existing_doc.exists:
                existing_by_title = {
                    t.get("title"): t
                    for t in existing_doc.to_dict().get("topics", [])
                }
                for t in clean_topics:
                    existing = existing_by_title.get(t.get("title"), {})
                    if existing.get("topic_id"):
                        t["topic_id"] = existing["topic_id"]
                    if existing.get("story_id"):
                        t["story_id"] = existing["story_id"]

            doc_ref.set({
                "topics_id":    topics_id or str(_uuid.uuid4()),
                "theme":        theme,
                "age":          age,
                "language":     lang,
                "filter_type":  filter_type,
                "filter_value": filter_value,
                "topics":       clean_topics,
                "created_at":   firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"[Firestore] Topics saved: {col}/{doc_id} ({len(clean_topics)} topics)")
        except Exception as e:
            logger.error(f"save_title_library_entry failed: {e}")
            raise

    async def update_title_story_id(
        self,
        theme: str,
        age: str,
        lang: str,
        filter_value: str,
        title_text: str,
        story_id: str,
    ) -> None:
        """Patches story_id onto a specific topic entry in the theme topic collection."""
        try:
            col = self._topic_collection(theme)
            doc_id = self._library_doc_id(age, lang, filter_value)
            doc_ref = self.db.collection(col).document(doc_id)
            doc = doc_ref.get()
            if not doc.exists:
                logger.warning(f"[Firestore] Topic doc not found: {col}/{doc_id}")
                return
            topics = list(doc.to_dict().get("topics", []))
            for t in topics:
                if t.get("title") == title_text:
                    t["story_id"] = story_id
                    doc_ref.update({"topics": topics, "updated_at": firestore.SERVER_TIMESTAMP})
                    logger.info(f"[Firestore] story_id={story_id} patched in {col}/{doc_id}")
                    return
            logger.warning(f"[Firestore] Title '{title_text}' not found in {col}/{doc_id}")
        except Exception as e:
            logger.error(f"update_title_story_id failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Checkpoint cleanup
    # ------------------------------------------------------------------

    async def delete_workflow_checkpoints(self, thread_ids: list[str]) -> None:
        """
        Deletes workflow_checkpoints documents for the given thread_ids.
        Called after a full story pipeline completes successfully.
        """
        try:
            batch = self.db.batch()
            deleted = 0
            for thread_id in thread_ids:
                q = (
                    self.db.collection("workflow_checkpoints")
                    .where(filter=FieldFilter("thread_id", "==", thread_id))
                )
                docs = list(q.stream())
                for doc in docs:
                    batch.delete(doc.reference)
                    deleted += 1
            if deleted:
                batch.commit()
                logger.info(f"[Firestore] Cleaned {deleted} checkpoints for {len(thread_ids)} threads")
        except Exception as e:
            logger.error(f"delete_workflow_checkpoints failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_workflow_status(self, story_id: str) -> dict:
        try:
            story = await self.get_story(story_id)
            if not story:
                return {"story_id": story_id, "status": "not_found"}
            return {
                "story_id":       story_id,
                "theme":          story.get("theme"),
                "wf2_story":      "completed" if story.get("story_text") else "pending",
                "wf3_image":      "completed" if story.get("image_url") else "pending",
                "wf4_audio":      "completed" if story.get("audio_url") else "pending",
                "wf5_activities": story.get("activities", {}),
            }
        except Exception as e:
            logger.error(f"get_workflow_status failed: {e}")
            return {"story_id": story_id, "status": "error", "error": str(e)}
