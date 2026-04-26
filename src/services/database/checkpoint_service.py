"""
Firestore-backed Checkpoint Service for LangGraph workflows.

Persists workflow state to Firestore for crash recovery and resumption.

Usage:
    from src.services.database.checkpoint_service import FirestoreCheckpointer
    
    checkpointer = FirestoreCheckpointer()
    workflow = graph.compile(checkpointer=checkpointer)
"""

from typing import Any, Optional, Iterator, List, Tuple
from datetime import datetime, timezone
import json
import pickle
import base64

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from google.cloud import firestore

from ...utils.config import get_settings
from ...utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class FirestoreCheckpointer(BaseCheckpointSaver):
    """
    Firestore-backed checkpoint saver for LangGraph.
    
    Stores checkpoints in a Firestore collection with the following structure:
    
    Collection: workflow_checkpoints
    Document ID: {thread_id}_{checkpoint_id}
    
    Document fields:
        - thread_id: str
        - checkpoint_id: str
        - parent_checkpoint_id: Optional[str]
        - checkpoint: str (base64 encoded pickle)
        - metadata: dict
        - created_at: timestamp
    
    Args:
        collection_name: Firestore collection name for checkpoints
        ttl_days: Auto-delete checkpoints after N days (default: 7)
    """
    
    serde = JsonPlusSerializer()
    
    def __init__(
        self,
        collection_name: str = "workflow_checkpoints",
        ttl_days: int = 7
    ):
        super().__init__()
        self.collection_name = collection_name
        self.ttl_days = ttl_days
        self._client: Optional[firestore.AsyncClient] = None
    
    @property
    def client(self) -> firestore.AsyncClient:
        """Lazy initialization of Firestore client."""
        if self._client is None:
            project  = settings.GOOGLE_CLOUD_PROJECT
            database = settings.FIRESTORE_DATABASE
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                self._client = firestore.AsyncClient.from_service_account_json(
                    settings.GOOGLE_APPLICATION_CREDENTIALS,
                    project=project,
                    database=database,
                )
            else:
                self._client = firestore.AsyncClient(project=project, database=database)
        return self._client
    
    def _get_collection(self):
        """Get the checkpoints collection reference."""
        return self.client.collection(self.collection_name)
    
    def _make_doc_id(self, thread_id: str, checkpoint_id: str) -> str:
        """Create a unique document ID."""
        return f"{thread_id}_{checkpoint_id}"
    
    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serialize checkpoint to base64 string for Firestore storage."""
        pickled = pickle.dumps(checkpoint)
        return base64.b64encode(pickled).decode("utf-8")
    
    def _deserialize_checkpoint(self, data: str) -> Checkpoint:
        """Deserialize checkpoint from base64 string."""
        pickled = base64.b64decode(data.encode("utf-8"))
        return pickle.loads(pickled)
    
    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """
        Get the latest checkpoint for a thread.
        
        Args:
            config: Config dict with configurable.thread_id
        
        Returns:
            CheckpointTuple or None if no checkpoint exists
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        try:
            if checkpoint_id:
                # Get specific checkpoint
                doc_id = self._make_doc_id(thread_id, checkpoint_id)
                doc = await self._get_collection().document(doc_id).get()
                
                if not doc.exists:
                    return None
                
                data = doc.to_dict()
            else:
                # Get latest checkpoint for thread
                query = (
                    self._get_collection()
                    .where(filter=firestore.FieldFilter("thread_id", "==", thread_id))
                    .order_by("created_at", direction=firestore.Query.DESCENDING)
                    .limit(1)
                )

                try:
                    docs = await query.get()
                except Exception as e:
                    # Firestore requires a composite index for (thread_id == X) + order_by(created_at).
                    # In dev/local environments, fall back to fetching and sorting client-side.
                    msg = str(e).lower()
                    if "requires an index" not in msg and "create it here" not in msg:
                        raise

                    logger.warning(
                        "Firestore composite index missing for checkpoints query; falling back to client-side sorting. "
                        "Create the composite index for production performance."
                    )
                    docs = await (
                        self._get_collection()
                        .where(filter=firestore.FieldFilter("thread_id", "==", thread_id))
                        .get()
                    )
                
                if not docs:
                    return None

                if len(docs) == 1:
                    data = docs[0].to_dict()
                else:
                    def _created_at(doc) -> datetime:
                        doc_dict = doc.to_dict() or {}
                        return doc_dict.get("created_at") or datetime.fromtimestamp(0, tz=timezone.utc)

                    newest_doc = max(docs, key=_created_at)
                    data = newest_doc.to_dict()
            
            checkpoint = self._deserialize_checkpoint(data["checkpoint"])
            metadata = CheckpointMetadata(**data.get("metadata", {}))
            
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": data["thread_id"],
                        "checkpoint_id": data["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config={
                    "configurable": {
                        "thread_id": data["thread_id"],
                        "checkpoint_id": data.get("parent_checkpoint_id"),
                    }
                } if data.get("parent_checkpoint_id") else None,
            )
        except Exception as e:
            logger.error(f"Error getting checkpoint for thread {thread_id}: {e}")
            return None
    
    async def alist(
        self,
        config: Optional[dict] = None,
        *,
        filter: Optional[dict] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Config with thread_id
            filter: Optional filter criteria
            before: Get checkpoints before this config
            limit: Max number of checkpoints to return
        """
        if not config:
            return
        
        thread_id = config["configurable"]["thread_id"]
        
        try:
            query = (
                self._get_collection()
                .where(filter=firestore.FieldFilter("thread_id", "==", thread_id))
                .order_by("created_at", direction=firestore.Query.DESCENDING)
            )

            try:
                if limit:
                    query = query.limit(limit)
                docs = await query.get()
            except Exception as e:
                msg = str(e).lower()
                if "requires an index" not in msg and "create it here" not in msg:
                    raise

                logger.warning(
                    "Firestore composite index missing for checkpoints query; falling back to client-side sorting. "
                    "Create the composite index for production performance."
                )
                docs = await (
                    self._get_collection()
                    .where(filter=firestore.FieldFilter("thread_id", "==", thread_id))
                    .get()
                )

                docs.sort(
                    key=lambda d: (d.to_dict() or {}).get("created_at") or datetime.fromtimestamp(0, tz=timezone.utc),
                    reverse=True,
                )
                if limit:
                    docs = docs[:limit]
            
            for doc in docs:
                data = doc.to_dict()
                checkpoint = self._deserialize_checkpoint(data["checkpoint"])
                metadata = CheckpointMetadata(**data.get("metadata", {}))
                
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": data["thread_id"],
                            "checkpoint_id": data["checkpoint_id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config={
                        "configurable": {
                            "thread_id": data["thread_id"],
                            "checkpoint_id": data.get("parent_checkpoint_id"),
                        }
                    } if data.get("parent_checkpoint_id") else None,
                )
        except Exception as e:
            logger.error(f"Error listing checkpoints for thread {thread_id}: {e}")
    
    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict] = None,
    ) -> dict:
        """
        Save a checkpoint.
        
        Args:
            config: Config with thread_id
            checkpoint: The checkpoint to save
            metadata: Checkpoint metadata
            new_versions: Optional version info
        
        Returns:
            Updated config with checkpoint_id
        """
        thread_id = config["configurable"]["thread_id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        
        # Generate new checkpoint ID
        checkpoint_id = checkpoint["id"]
        
        try:
            doc_id = self._make_doc_id(thread_id, checkpoint_id)
            
            doc_data = {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "checkpoint": self._serialize_checkpoint(checkpoint),
                "metadata": metadata.__dict__ if hasattr(metadata, "__dict__") else dict(metadata),
                "created_at": datetime.now(timezone.utc),
            }
            
            await self._get_collection().document(doc_id).set(doc_data)
            
            logger.debug(f"Saved checkpoint {checkpoint_id} for thread {thread_id}")
            
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }
        except Exception as e:
            logger.error(f"Error saving checkpoint for thread {thread_id}: {e}")
            raise
    
    async def aput_writes(
        self,
        config: dict,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store intermediate writes (for pending operations).
        
        This is used for storing writes that haven't been committed to a checkpoint yet.
        """
        # For simplicity, we don't persist intermediate writes
        # They will be re-computed on recovery
        pass
    
    # Sync versions (required by base class, but we use async)
    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """Sync version - use aget_tuple for async."""
        raise NotImplementedError("Use aget_tuple for async operations")
    
    def list(
        self,
        config: Optional[dict] = None,
        *,
        filter: Optional[dict] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Sync version - use alist for async."""
        raise NotImplementedError("Use alist for async operations")
    
    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict] = None,
    ) -> dict:
        """Sync version - use aput for async."""
        raise NotImplementedError("Use aput for async operations")
    
    def put_writes(
        self,
        config: dict,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Sync version - use aput_writes for async."""
        raise NotImplementedError("Use aput_writes for async operations")
    
    async def adelete_thread(self, thread_id: str) -> None:
        """
        Delete all checkpoints for a thread.
        
        Args:
            thread_id: The thread ID to delete checkpoints for
        """
        try:
            query = self._get_collection().where(filter=firestore.FieldFilter("thread_id", "==", thread_id))
            docs = await query.get()
            
            batch = self.client.batch()
            for doc in docs:
                batch.delete(doc.reference)
            
            await batch.commit()
            logger.info(f"Deleted {len(docs)} checkpoints for thread {thread_id}")
        except Exception as e:
            logger.error(f"Error deleting checkpoints for thread {thread_id}: {e}")
            raise
    
    async def cleanup_old_checkpoints(self, days: Optional[int] = None) -> int:
        """
        Delete checkpoints older than N days.
        
        Args:
            days: Number of days (default: self.ttl_days)
        
        Returns:
            Number of deleted checkpoints
        """
        days = days or self.ttl_days
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff = cutoff.replace(day=cutoff.day - days)
        
        try:
            query = self._get_collection().where(filter=firestore.FieldFilter("created_at", "<", cutoff))
            docs = await query.get()
            
            if not docs:
                return 0
            
            batch = self.client.batch()
            for doc in docs:
                batch.delete(doc.reference)
            
            await batch.commit()
            logger.info(f"Cleaned up {len(docs)} old checkpoints")
            return len(docs)
        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")
            raise
