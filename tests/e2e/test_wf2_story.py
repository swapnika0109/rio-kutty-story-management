"""
E2E tests for WF2 — Story Creator workflow.
Covers: story creation and Firestore persistence (US1).
"""

from __future__ import annotations

import uuid
import pytest

from src.workflows.story_creator_workflow import story_creator_workflow
from tests.e2e.helpers.firestore_helper import (
    story_collection,
    assert_doc_fields,
    assert_field_type,
)

THEMES = ["planet_protectors", "mindful", "chill"]

_THEME_KEY = {
    "planet_protectors": "theme1",
    "mindful": "theme2",
    "chill": "theme3",
}

_SAMPLE_TOPIC = {
    "title": "The Brave Little Turtle",
    "theme": None,  # filled per-test
    "moral": "Courage helps us overcome our fears.",
    "description": "A turtle learns to swim past the scary deep end.",
    "code": "EN",
    "country": "USA",
    "religion": "none",
    "source": "original",
    "story_seed": "A turtle afraid of deep water finds a friend who helps.",
}


async def _run_wf2(theme: str, story_id: str, topics_id: str) -> dict:
    topic = {**_SAMPLE_TOPIC, "theme": _THEME_KEY[theme]}
    config = {
        "configurable": {
            "thread_id": story_id,
            "story_id": story_id,
            "topics_id": topics_id,
            "age": "3-4",
            "language": "English",
            "theme": _THEME_KEY[theme],
            "selected_topic": topic,
        }
    }
    state = await story_creator_workflow.ainvoke(
        {"selected_topic": topic}, config=config
    )
    return state


class TestWF2StoryPersisted:

    @pytest.mark.e2e
    @pytest.mark.parametrize("theme", THEMES)
    async def test_wf2_story_persisted(
        self,
        theme,
        firestore_test_client,
        created_story_ids,
    ):
        story_id = str(uuid.uuid4())
        topics_id = str(uuid.uuid4())
        created_story_ids.append((theme, story_id))

        await _run_wf2(theme, story_id, topics_id)

        col = story_collection(theme)
        doc = await firestore_test_client.collection(col).document(story_id).get()
        assert doc.exists, f"Story doc must exist after WF2: {col}/{story_id}"

        data = doc.to_dict()
        assert_doc_fields(data, ["title", "description", "moral"])
        assert_field_type(data, "title", str)
        assert_field_type(data, "description", str)
        assert len(data["title"]) > 0, "title must be non-empty"
        assert len(data["description"]) > 0, "description must be non-empty"
