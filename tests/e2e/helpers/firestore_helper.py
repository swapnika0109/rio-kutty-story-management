"""
Firestore assertion and cleanup helpers for E2E tests.
All operations target the `rio-test` named database.
"""

from __future__ import annotations

import asyncio
from typing import Any

from google.cloud import firestore as _firestore

_THEME_TOPIC_COLLECTIONS = {
    "planet_protectors": "planet_protectors_topics",
    "mindful": "mindful_topics",
    "chill": "chill_stories_topics",
}

_THEME_STORY_COLLECTIONS = {
    "planet_protectors": "planet_protectors_stories",
    "mindful": "mindful_stories",
    "chill": "chill_stories",
}


def topic_collection(theme: str) -> str:
    return _THEME_TOPIC_COLLECTIONS[theme]


def story_collection(theme: str) -> str:
    return _THEME_STORY_COLLECTIONS[theme]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def assert_doc_fields(doc_data: dict, required_fields: list[str]) -> None:
    """Assert that all required fields are present and non-None in doc_data."""
    missing = [f for f in required_fields if doc_data.get(f) is None]
    assert not missing, f"Missing fields in Firestore doc: {missing}\nDoc: {doc_data}"


def assert_field_type(doc_data: dict, field: str, expected_type: type) -> None:
    value = doc_data.get(field)
    assert isinstance(value, expected_type), (
        f"Field '{field}' expected {expected_type.__name__}, got {type(value).__name__}: {value!r}"
    )


def assert_field_startswith(doc_data: dict, field: str, prefix: str) -> None:
    value = doc_data.get(field, "")
    assert isinstance(value, str) and value.startswith(prefix), (
        f"Field '{field}' expected to start with {prefix!r}, got {value!r}"
    )


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

async def delete_topic_doc(db: _firestore.AsyncClient, theme: str, topics_id: str) -> None:
    col = topic_collection(theme)
    await db.collection(col).document(topics_id).delete()


async def delete_story_doc(db: _firestore.AsyncClient, theme: str, story_id: str) -> None:
    col = story_collection(theme)
    await db.collection(col).document(story_id).delete()


async def delete_activities_for_story(db: _firestore.AsyncClient, story_id: str) -> None:
    col_ref = db.collection("activities_v1")
    docs = col_ref.where("story_id", "==", story_id).stream()
    batch = db.batch()
    count = 0
    async for doc in docs:
        batch.delete(doc.reference)
        count += 1
        if count % 500 == 0:
            await batch.commit()
            batch = db.batch()
    if count % 500 != 0:
        await batch.commit()


async def delete_checkpoint(db: _firestore.AsyncClient, thread_id: str) -> None:
    await db.collection("workflow_checkpoints").document(thread_id).delete()


async def bulk_cleanup(
    db: _firestore.AsyncClient,
    *,
    topic_ids: list[tuple[str, str]],   # [(theme, topics_id), ...]
    story_ids: list[tuple[str, str]],   # [(theme, story_id), ...]
    activity_story_ids: list[str],      # story_ids whose activities to delete
    checkpoint_thread_ids: list[str],
) -> None:
    """Delete all test-created docs. Teardown order: activities → checkpoints → stories → topics."""
    await asyncio.gather(
        *[delete_activities_for_story(db, sid) for sid in activity_story_ids],
        *[delete_checkpoint(db, tid) for tid in checkpoint_thread_ids],
    )
    await asyncio.gather(
        *[delete_story_doc(db, theme, sid) for theme, sid in story_ids],
    )
    await asyncio.gather(
        *[delete_topic_doc(db, theme, tid) for theme, tid in topic_ids],
    )
