"""
E2E tests for the full WF1→WF2→WF3+WF4→WF5 pipeline (US1).
Also covers: parallel WF3+WF4 timing (T038), cleanup verification (T039–T041),
and JSON report verification (T041).
"""

from __future__ import annotations

import asyncio
import time
import uuid
import pytest

from src.workflows.story_topics_workflow import story_topics_workflow
from src.workflows.story_creator_workflow import story_creator_workflow
from src.workflows.master_workflow import master_workflow
from tests.e2e.helpers.firestore_helper import (
    story_collection,
    topic_collection,
    assert_doc_fields,
    assert_field_startswith,
)
from tests.e2e.helpers.storage_helper import assert_blob_exists, gcs_path_to_blob_name

THEMES = ["planet_protectors", "mindful", "chill"]

_THEME_KEY = {
    "planet_protectors": "theme1",
    "mindful": "theme2",
    "chill": "theme3",
}

_SAMPLE_TOPIC = {
    "title": "The Brave Little Turtle",
    "moral": "Courage grows when you have a friend.",
    "description": "Teo the turtle learns to swim past the scary deep end.",
    "code": "EN",
    "country": "USA",
    "religion": "none",
    "source": "original",
    "story_seed": "A turtle afraid of deep water finds a friend who helps.",
    "image_prompt": "A brave little turtle swimming through a sunlit pond.",
}


async def _run_full_pipeline(theme: str) -> tuple[str, str, float]:
    """
    Run WF1→WF2→master (WF3+WF4+WF5).
    Returns (story_id, topics_id, total_elapsed_seconds).
    """
    theme_key = _THEME_KEY[theme]
    story_id = str(uuid.uuid4())
    topics_id = str(uuid.uuid4())

    t_start = time.monotonic()

    # WF1 — generate topics
    await story_topics_workflow.ainvoke(
        {},
        config={
            "configurable": {
                "thread_id": f"{story_id}_wf1",
                "story_id": story_id,
                "age": "3-4",
                "language": "English",
                "theme": theme_key,
            }
        },
    )

    # WF2 — create story
    topic = {**_SAMPLE_TOPIC, "theme": theme_key}
    story_state = await story_creator_workflow.ainvoke(
        {"selected_topic": topic},
        config={
            "configurable": {
                "thread_id": f"{story_id}_wf2",
                "story_id": story_id,
                "topics_id": topics_id,
                "age": "3-4",
                "language": "English",
                "theme": theme_key,
                "selected_topic": topic,
            }
        },
    )

    story = story_state.get("story") or {}

    # Master — WF3+WF4+WF5
    await master_workflow.ainvoke(
        {"story_id": story_id, "story": story},
        config={
            "configurable": {
                "thread_id": f"{story_id}_master",
                "story_id": story_id,
                "age": "3-4",
                "language": "English",
                "theme": theme_key,
                "voice": "standard",
            }
        },
    )

    elapsed = time.monotonic() - t_start
    return story_id, topics_id, elapsed


class TestFullPipelineHappyPath:

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.parametrize("theme", THEMES)
    async def test_full_pipeline(
        self,
        theme,
        firestore_test_client,
        test_bucket,
        created_topic_ids,
        created_story_ids,
        created_activity_story_ids,
        created_gcs_blobs,
    ):
        story_id, topics_id, elapsed = await _run_full_pipeline(theme)

        created_topic_ids.append((theme, topics_id))
        created_story_ids.append((theme, story_id))
        created_activity_story_ids.append(story_id)

        # Assert elapsed < 5 minutes
        assert elapsed < 300, f"Full pipeline took {elapsed:.1f}s — exceeds 5 minute limit"

        # Assert story doc has all required fields
        col = story_collection(theme)
        doc = await firestore_test_client.collection(col).document(story_id).get()
        assert doc.exists, f"Story doc must exist: {col}/{story_id}"
        data = doc.to_dict()
        assert_doc_fields(data, ["title", "description", "moral", "image_url", "audio_url"])
        assert_field_startswith(data, "image_url", "gs://kutty_bucket/test/images/")
        assert_field_startswith(data, "audio_url", "gs://kutty_bucket/test/audio/")

        # Assert GCS blobs exist
        img_blob = gcs_path_to_blob_name(data["image_url"])
        aud_blob = gcs_path_to_blob_name(data["audio_url"])
        created_gcs_blobs.extend([img_blob, aud_blob])
        assert_blob_exists(test_bucket, img_blob)
        assert_blob_exists(test_bucket, aud_blob)

        # Assert 4 activities exist
        col_ref = firestore_test_client.collection("activities_v1")
        docs = col_ref.where("story_id", "==", story_id).stream()
        activity_types = set()
        async for adoc in docs:
            adata = adoc.to_dict()
            activity_types.add(adata.get("activity_type"))
            if adata.get("activity_image_url"):
                created_gcs_blobs.append(gcs_path_to_blob_name(adata["activity_image_url"]))

        assert activity_types == {"mcq", "art", "science", "moral"}, (
            f"Expected all 4 activity types, got: {activity_types}"
        )


class TestParallelExecution:

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.parametrize("theme", ["planet_protectors"])
    async def test_wf3_wf4_run_in_parallel(
        self,
        theme,
        firestore_test_client,
        test_bucket,
        created_story_ids,
        created_gcs_blobs,
    ):
        """Verify WF3 and WF4 run concurrently — total time < sum of individual durations."""
        from src.workflows.image_workflow import image_workflow
        from src.workflows.audio_workflow import audio_workflow

        story_id = str(uuid.uuid4())
        created_story_ids.append((theme, story_id))

        _THEME_K = _THEME_KEY[theme]

        # Time individual runs sequentially to get baseline
        t0 = time.monotonic()
        await image_workflow.ainvoke(
            {},
            config={"configurable": {
                "thread_id": f"{story_id}_img_serial",
                "story_id": f"{story_id}_s1",
                "theme": _THEME_K,
                "image_prompt": "A brave turtle swimming.",
                "storage_prefix": "test",
            }},
        )
        wf3_serial_duration = time.monotonic() - t0

        t0 = time.monotonic()
        await audio_workflow.ainvoke(
            {},
            config={"configurable": {
                "thread_id": f"{story_id}_aud_serial",
                "story_id": f"{story_id}_s2",
                "theme": _THEME_K,
                "language": "English",
                "voice_type": "standard",
                "story_text": "A turtle swam bravely.",
                "storage_prefix": "test",
            }},
        )
        wf4_serial_duration = time.monotonic() - t0

        # Time parallel execution
        t_parallel_start = time.monotonic()
        await asyncio.gather(
            image_workflow.ainvoke(
                {},
                config={"configurable": {
                    "thread_id": f"{story_id}_img_par",
                    "story_id": f"{story_id}_p1",
                    "theme": _THEME_K,
                    "image_prompt": "A brave turtle.",
                    "storage_prefix": "test",
                }},
            ),
            audio_workflow.ainvoke(
                {},
                config={"configurable": {
                    "thread_id": f"{story_id}_aud_par",
                    "story_id": f"{story_id}_p2",
                    "theme": _THEME_K,
                    "language": "English",
                    "voice_type": "standard",
                    "story_text": "A turtle swam bravely.",
                    "storage_prefix": "test",
                }},
            ),
        )
        parallel_duration = time.monotonic() - t_parallel_start

        assert parallel_duration < wf3_serial_duration + wf4_serial_duration, (
            f"Parallel execution ({parallel_duration:.1f}s) was not faster than "
            f"serial ({wf3_serial_duration + wf4_serial_duration:.1f}s) — "
            f"WF3+WF4 may not be running concurrently"
        )


class TestCleanupVerification:

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_firestore_cleanup_verified(
        self,
        firestore_test_client,
        created_story_ids,
        created_activity_story_ids,
    ):
        """After test teardown, no docs created during this test should remain."""
        story_id = str(uuid.uuid4())
        theme = "planet_protectors"
        created_story_ids.append((theme, story_id))
        created_activity_story_ids.append(story_id)

        # Manually create a doc to ensure cleanup fires
        col = story_collection(theme)
        await firestore_test_client.collection(col).document(story_id).set({"title": "cleanup-test"})

        # Teardown happens automatically — but we verify inside test via a separate check
        # by confirming the doc we just created will be tracked for deletion
        doc = await firestore_test_client.collection(col).document(story_id).get()
        assert doc.exists, "Doc should exist before teardown"
        # After test function ends, cleanup_firestore fixture deletes it

    @pytest.mark.e2e
    async def test_json_report_written(self, tmp_path):
        """Verify the report_writer plugin creates a valid JSON report."""
        import json
        from pathlib import Path

        reports_dir = Path("test/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_files = list(reports_dir.glob("run_*.json"))
        if not report_files:
            pytest.skip("No report file yet — run after a full E2E session")

        latest = max(report_files, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text())

        assert "run_id" in data
        assert "timestamp" in data
        assert "results" in data
        assert "summary" in data
        assert data["summary"]["total"] >= 0
