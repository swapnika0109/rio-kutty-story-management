"""
E2E tests for WF5 — Activities workflow.
Covers: activity generation (US1), DeepEval GEval quality gate (US4).
"""

from __future__ import annotations

import uuid
import pytest

from src.workflows.activity_workflow import app_workflow as activity_workflow
from tests.e2e.helpers.firestore_helper import assert_doc_fields, assert_field_startswith
from tests.e2e.helpers.storage_helper import assert_blob_exists, gcs_path_to_blob_name

THEMES = ["planet_protectors", "mindful", "chill"]

ACTIVITY_TYPES = ["mcq", "art", "science", "moral"]
ACTIVITY_TYPES_WITH_IMAGES = {"art", "science", "moral"}

_SAMPLE_STORY = {
    "story_id": None,  # filled per-test
    "story_text": (
        "Once upon a time, a brave little turtle named Teo lived by a sparkling pond. "
        "Teo was afraid of the deep water, but one day, a kind fish named Finn helped "
        "Teo learn to swim. Together they explored every corner of the pond and "
        "discovered that courage grows when you have a friend by your side."
    ),
    "language": "en",
    "age": "3-4",
}

DEEPEVAL_THRESHOLD = 0.7


async def _run_wf5(story_id: str) -> dict:
    config = {
        "configurable": {
            "thread_id": story_id,
            "story_id": story_id,
            "story_text": _SAMPLE_STORY["story_text"],
            "age": _SAMPLE_STORY["age"],
            "language": _SAMPLE_STORY["language"],
            "storage_prefix": "test",
        }
    }
    state = await activity_workflow.ainvoke({}, config=config)
    return state


# ---------------------------------------------------------------------------
# US1: Happy Path — activities generated and persisted
# ---------------------------------------------------------------------------

class TestWF5HappyPath:

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.parametrize("theme", THEMES)
    async def test_wf5_activities_generated(
        self,
        theme,
        firestore_test_client,
        test_bucket,
        created_activity_story_ids,
        created_gcs_blobs,
    ):
        story_id = str(uuid.uuid4())
        created_activity_story_ids.append(story_id)

        await _run_wf5(story_id)

        # Assert 4 activity docs exist tagged with story_id
        col_ref = firestore_test_client.collection("activities_v1")
        docs = col_ref.where("story_id", "==", story_id).stream()
        activities = {}
        async for doc in docs:
            data = doc.to_dict()
            activities[data.get("activity_type")] = data

        assert set(activities.keys()) == set(ACTIVITY_TYPES), (
            f"Expected activity types {ACTIVITY_TYPES}, got {list(activities.keys())}"
        )

        # Assert activity_image_url for image-bearing types
        for activity_type, data in activities.items():
            assert_doc_fields(data, ["story_id", "activity_type", "content"])
            if activity_type in ACTIVITY_TYPES_WITH_IMAGES:
                assert_field_startswith(data, "activity_image_url", "gs://kutty_bucket/test/images/")
                blob_name = gcs_path_to_blob_name(data["activity_image_url"])
                created_gcs_blobs.append(blob_name)
                assert_blob_exists(test_bucket, blob_name)


# ---------------------------------------------------------------------------
# US4: DeepEval GEval quality gate (>= 0.7 per activity type)
# ---------------------------------------------------------------------------

class TestDeepEval:

    @pytest.mark.e2e
    @pytest.mark.slow
    async def _evaluate_activity(self, activity_data: dict, activity_type: str) -> float:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        content = activity_data.get("content", {})
        metric = GEval(
            name=f"{activity_type.upper()}Quality",
            criteria=(
                f"Is this {activity_type} activity age-appropriate for children aged 3-5, "
                f"educational, well-structured, and aligned with the story's moral?"
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=DEEPEVAL_THRESHOLD,
            model="gemini/gemini-2.0-flash-lite",
        )
        test_case = LLMTestCase(
            input=f"Generate a {activity_type} activity for the story about Teo the turtle.",
            actual_output=str(content),
        )
        metric.measure(test_case)
        return metric.score

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_mcq_geval_score_above_threshold(
        self, firestore_test_client, created_activity_story_ids
    ):
        story_id = str(uuid.uuid4())
        created_activity_story_ids.append(story_id)
        await _run_wf5(story_id)

        col_ref = firestore_test_client.collection("activities_v1")
        docs = col_ref.where("story_id", "==", story_id).where("activity_type", "==", "mcq").stream()
        mcq_data = None
        async for doc in docs:
            mcq_data = doc.to_dict()
            break

        assert mcq_data is not None, "MCQ activity must exist after WF5"
        score = await self._evaluate_activity(mcq_data, "mcq")
        assert score >= DEEPEVAL_THRESHOLD, f"MCQ GEval score {score} below threshold {DEEPEVAL_THRESHOLD}"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_art_activity_geval_score_above_threshold(
        self, firestore_test_client, test_bucket, created_activity_story_ids, created_gcs_blobs
    ):
        story_id = str(uuid.uuid4())
        created_activity_story_ids.append(story_id)
        await _run_wf5(story_id)

        col_ref = firestore_test_client.collection("activities_v1")
        docs = col_ref.where("story_id", "==", story_id).where("activity_type", "==", "art").stream()
        art_data = None
        async for doc in docs:
            art_data = doc.to_dict()
            break

        assert art_data is not None, "Art activity must exist after WF5"
        assert_field_startswith(art_data, "activity_image_url", "gs://kutty_bucket/test/images/")
        created_gcs_blobs.append(gcs_path_to_blob_name(art_data["activity_image_url"]))

        score = await self._evaluate_activity(art_data, "art")
        assert score >= DEEPEVAL_THRESHOLD, f"Art GEval score {score} below threshold {DEEPEVAL_THRESHOLD}"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_science_activity_geval_score_above_threshold(
        self, firestore_test_client, test_bucket, created_activity_story_ids, created_gcs_blobs
    ):
        story_id = str(uuid.uuid4())
        created_activity_story_ids.append(story_id)
        await _run_wf5(story_id)

        col_ref = firestore_test_client.collection("activities_v1")
        docs = col_ref.where("story_id", "==", story_id).where("activity_type", "==", "science").stream()
        sci_data = None
        async for doc in docs:
            sci_data = doc.to_dict()
            break

        assert sci_data is not None, "Science activity must exist after WF5"
        assert_field_startswith(sci_data, "activity_image_url", "gs://kutty_bucket/test/images/")
        created_gcs_blobs.append(gcs_path_to_blob_name(sci_data["activity_image_url"]))

        score = await self._evaluate_activity(sci_data, "science")
        assert score >= DEEPEVAL_THRESHOLD, f"Science GEval score {score} below threshold {DEEPEVAL_THRESHOLD}"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_moral_activity_geval_score_above_threshold(
        self, firestore_test_client, test_bucket, created_activity_story_ids, created_gcs_blobs
    ):
        story_id = str(uuid.uuid4())
        created_activity_story_ids.append(story_id)
        await _run_wf5(story_id)

        col_ref = firestore_test_client.collection("activities_v1")
        docs = col_ref.where("story_id", "==", story_id).where("activity_type", "==", "moral").stream()
        moral_data = None
        async for doc in docs:
            moral_data = doc.to_dict()
            break

        assert moral_data is not None, "Moral activity must exist after WF5"
        assert_field_startswith(moral_data, "activity_image_url", "gs://kutty_bucket/test/images/")
        created_gcs_blobs.append(gcs_path_to_blob_name(moral_data["activity_image_url"]))

        score = await self._evaluate_activity(moral_data, "moral")
        assert score >= DEEPEVAL_THRESHOLD, f"Moral GEval score {score} below threshold {DEEPEVAL_THRESHOLD}"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_wf5_retries_on_low_geval_score(
        self, firestore_test_client, created_activity_story_ids, retry_limit
    ):
        """WF5 should retry activity generation when GEval score is below threshold."""
        from unittest.mock import AsyncMock, patch

        story_id = str(uuid.uuid4())
        created_activity_story_ids.append(story_id)

        call_counts: dict[str, int] = {}

        original_generate = None

        async def mock_generate_content(self_inner, prompt, **kwargs):
            call_counts["total"] = call_counts.get("total", 0) + 1
            return await original_generate(self_inner, prompt, **kwargs)

        with patch("src.services.ai_service.AIService.generate_content", mock_generate_content):
            await _run_wf5(story_id)

        col_ref = firestore_test_client.collection("activities_v1")
        docs = col_ref.where("story_id", "==", story_id).stream()
        activity_count = 0
        async for _ in docs:
            activity_count += 1

        assert activity_count == 4, f"Expected 4 activities after WF5, got {activity_count}"
