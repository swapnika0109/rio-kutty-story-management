"""
E2E tests for WF3 — Image Generator workflow.
Covers: image generation stored under test/images/ (US1).
"""

from __future__ import annotations

import uuid
import pytest

from src.workflows.image_workflow import image_workflow
from tests.e2e.helpers.firestore_helper import story_collection, assert_field_startswith
from tests.e2e.helpers.storage_helper import assert_blob_exists, gcs_path_to_blob_name

THEMES = ["planet_protectors", "mindful", "chill"]

_THEME_KEY = {
    "planet_protectors": "theme1",
    "mindful": "theme2",
    "chill": "theme3",
}


async def _run_wf3(theme: str, story_id: str) -> dict:
    config = {
        "configurable": {
            "thread_id": f"{story_id}_image",
            "story_id": story_id,
            "theme": _THEME_KEY[theme],
            "image_prompt": "A brave little turtle swimming in a pond surrounded by lily pads",
            "storage_prefix": "test",
        }
    }
    state = await image_workflow.ainvoke({}, config=config)
    return state


class TestWF3ImageGenerated:

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.parametrize("theme", THEMES)
    async def test_wf3_image_generated(
        self,
        theme,
        firestore_test_client,
        test_bucket,
        created_story_ids,
        created_gcs_blobs,
    ):
        story_id = str(uuid.uuid4())
        created_story_ids.append((theme, story_id))

        await _run_wf3(theme, story_id)

        col = story_collection(theme)
        doc = await firestore_test_client.collection(col).document(story_id).get()
        assert doc.exists, f"Story doc must exist after WF3: {col}/{story_id}"

        data = doc.to_dict()
        assert_field_startswith(data, "image_url", "gs://kutty_bucket/test/images/")

        blob_name = gcs_path_to_blob_name(data["image_url"])
        created_gcs_blobs.append(blob_name)
        assert_blob_exists(test_bucket, blob_name)
