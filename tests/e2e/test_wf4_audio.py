"""
E2E tests for WF4 — Audio Generator workflow.
Covers: audio generation stored under test/audio/ (US1).
"""

from __future__ import annotations

import uuid
import pytest

from src.workflows.audio_workflow import audio_workflow
from tests.e2e.helpers.firestore_helper import story_collection, assert_field_startswith
from tests.e2e.helpers.storage_helper import assert_blob_exists, gcs_path_to_blob_name

THEMES = ["planet_protectors", "mindful", "chill"]

_THEME_KEY = {
    "planet_protectors": "theme1",
    "mindful": "theme2",
    "chill": "theme3",
}

_SAMPLE_STORY_TEXT = (
    "Once upon a time, a brave little turtle named Teo lived by a sparkling pond. "
    "Teo was afraid of the deep water, but one day, a kind fish named Finn helped "
    "Teo learn to swim. Together they explored every corner of the pond."
)


async def _run_wf4(theme: str, story_id: str) -> dict:
    config = {
        "configurable": {
            "thread_id": f"{story_id}_audio",
            "story_id": story_id,
            "theme": _THEME_KEY[theme],
            "language": "English",
            "voice_type": "standard",
            "story_text": _SAMPLE_STORY_TEXT,
            "storage_prefix": "test",
        }
    }
    state = await audio_workflow.ainvoke({}, config=config)
    return state


class TestWF4AudioGenerated:

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.parametrize("theme", THEMES)
    async def test_wf4_audio_generated(
        self,
        theme,
        firestore_test_client,
        test_bucket,
        created_story_ids,
        created_gcs_blobs,
    ):
        story_id = str(uuid.uuid4())
        created_story_ids.append((theme, story_id))

        await _run_wf4(theme, story_id)

        col = story_collection(theme)
        doc = await firestore_test_client.collection(col).document(story_id).get()
        assert doc.exists, f"Story doc must exist after WF4: {col}/{story_id}"

        data = doc.to_dict()
        assert_field_startswith(data, "audio_url", "gs://kutty_bucket/test/audio/")

        blob_name = gcs_path_to_blob_name(data["audio_url"])
        created_gcs_blobs.append(blob_name)
        assert_blob_exists(test_bucket, blob_name)
