"""
E2E tests for WF1 — Story Topics workflow.
Covers: happy path, deduplication (US2), caching (US2).
"""

from __future__ import annotations

import uuid
import pytest

from src.workflows.story_topics_workflow import story_topics_workflow
from tests.e2e.helpers.firestore_helper import (
    topic_collection,
    assert_doc_fields,
    assert_field_type,
)


THEMES = ["planet_protectors", "mindful", "chill"]

# WF1 uses theme keys like "theme1"/"theme2"/"theme3" internally;
# map display names used in tests to those keys.
_THEME_KEY = {
    "planet_protectors": "theme1",
    "mindful": "theme2",
    "chill": "theme3",
}


async def _run_wf1(theme: str, story_id: str | None = None) -> dict:
    """Run WF1 for a theme, return the final state."""
    sid = story_id or str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": sid,
            "story_id": sid,
            "age": "3-4",
            "language": "English",
            "theme": _THEME_KEY[theme],
        }
    }
    state = await story_topics_workflow.ainvoke({}, config=config)
    return state


# ---------------------------------------------------------------------------
# US1: Happy Path
# ---------------------------------------------------------------------------

# class TestWF1HappyPath:

    @pytest.mark.e2e
    @pytest.mark.parametrize("theme", THEMES)
    async def test_wf1_happy_path(
        self,
        theme,
        firestore_test_client,
        created_topic_ids,
    ):
        sid = str(uuid.uuid4())
        state = await _run_wf1(theme, sid)

        topics_id = state.get("topics_id") or sid
        created_topic_ids.append((theme, topics_id))

        col = topic_collection(theme)
        doc = await firestore_test_client.collection(col).document(topics_id).get()
        assert doc.exists, f"Expected topics doc to exist: {col}/{topics_id}"

        data = doc.to_dict()
        assert_doc_fields(data, ["topics_id", "topics"])
        assert_field_type(data, "topics", list)
        assert len(data["topics"]) > 0, "topics array must be non-empty"


# ---------------------------------------------------------------------------
# US2: Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    @pytest.mark.e2e
    @pytest.mark.parametrize("theme", THEMES)
    async def test_wf1_no_duplicate_titles(
        self,
        theme,
        firestore_test_client,
        created_topic_ids,
    ):
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())

        state1 = await _run_wf1(theme, sid1)
        state2 = await _run_wf1(theme, sid2)

        topics_id1 = state1.get("topics_id") or sid1
        topics_id2 = state2.get("topics_id") or sid2
        created_topic_ids.extend([(theme, topics_id1), (theme, topics_id2)])

        col = topic_collection(theme)

        doc1 = await firestore_test_client.collection(col).document(topics_id1).get()
        doc2 = await firestore_test_client.collection(col).document(topics_id2).get()

        titles1 = {t if isinstance(t, str) else t.get("title", "") for t in doc1.to_dict().get("topics", [])}
        titles2 = {t if isinstance(t, str) else t.get("title", "") for t in doc2.to_dict().get("topics", [])}

        duplicates = titles1 & titles2
        assert not duplicates, (
            f"Duplicate topic titles found across two WF1 calls for theme '{theme}': {duplicates}"
        )

    @pytest.mark.e2e
    @pytest.mark.parametrize("theme", THEMES)
    async def test_wf1_topics_cached_in_firestore(
        self,
        theme,
        firestore_test_client,
        created_topic_ids,
    ):
        sid = str(uuid.uuid4())
        state = await _run_wf1(theme, sid)
        topics_id = state.get("topics_id") or sid
        created_topic_ids.append((theme, topics_id))

        col = topic_collection(theme)
        doc = await firestore_test_client.collection(col).document(topics_id).get()
        assert doc.exists, f"Topics doc must be cached in Firestore after WF1: {col}/{topics_id}"
        data = doc.to_dict()
        assert_doc_fields(data, ["topics_id", "topics"])
        assert len(data.get("topics", [])) > 0
