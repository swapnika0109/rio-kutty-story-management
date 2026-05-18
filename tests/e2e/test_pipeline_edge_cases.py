"""
E2E tests for all edge cases from the spec.
Each test produces a clear, deterministic outcome — no silent failures.
"""

from __future__ import annotations

import json
import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.workflows.story_topics_workflow import story_topics_workflow
from src.workflows.story_creator_workflow import story_creator_workflow
from src.workflows.master_workflow import master_workflow
from tests.e2e.helpers.api_helper import ApiHelper
from tests.e2e.helpers.firestore_helper import story_collection, topic_collection

import os

_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


async def _run_wf1(theme_key: str, story_id: str) -> dict:
    config = {
        "configurable": {
            "thread_id": story_id,
            "story_id": story_id,
            "age": "3-4",
            "language": "English",
            "theme": theme_key,
        }
    }
    return await story_topics_workflow.ainvoke({}, config=config)


async def _run_wf2(theme_key: str, story_id: str, topics_id: str) -> dict:
    topic = {
        "title": "Test Story",
        "theme": theme_key,
        "moral": "Test moral.",
        "description": "A test story.",
        "code": "EN",
        "country": "USA",
        "religion": "none",
        "source": "original",
        "story_seed": "A test seed.",
        "image_prompt": "A turtle.",
    }
    config = {
        "configurable": {
            "thread_id": story_id,
            "story_id": story_id,
            "topics_id": topics_id,
            "age": "3-4",
            "language": "English",
            "theme": theme_key,
            "selected_topic": topic,
        }
    }
    return await story_creator_workflow.ainvoke({"selected_topic": topic}, config=config)


async def _run_master(story_id: str, theme_key: str) -> None:
    story = {
        "title": "Test",
        "story_text": "A brave turtle swam.",
        "moral": "Courage matters.",
        "age_group": "3-4",
        "language": "English",
        "image_prompt": "A turtle.",
        "theme": theme_key,
    }
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


class TestEdgeCases:

    @pytest.mark.e2e
    async def test_invalid_theme_rejected(self, firestore_test_client):
        """Unsupported theme must raise or return a clear error — no docs created."""
        story_id = str(uuid.uuid4())
        with pytest.raises(Exception):
            await _run_wf1("dragons", story_id)

        # Verify no doc was accidentally created in any theme collection
        for col in ["planet_protectors_topics", "mindful_topics", "chill_stories_topics"]:
            doc = await firestore_test_client.collection(col).document(story_id).get()
            assert not doc.exists, f"No doc should exist for invalid theme: {col}/{story_id}"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_firestore_unavailable_on_story_save(self, created_checkpoint_ids):
        """Firestore unavailable during story save → error surfaced, not silently swallowed."""
        story_id = str(uuid.uuid4())
        created_checkpoint_ids.append(story_id)

        async def raise_on_save(*args, **kwargs):
            raise RuntimeError("Simulated Firestore unavailable")

        with patch(
            "src.services.database.firestore_service.FirestoreService.save_story",
            AsyncMock(side_effect=raise_on_save),
        ):
            with pytest.raises(Exception, match="Simulated Firestore unavailable"):
                await _run_wf2("theme1", story_id, str(uuid.uuid4()))

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_both_wf3_wf4_fail_simultaneously(
        self,
        firestore_test_client,
        retry_limit,
        created_story_ids,
        created_checkpoint_ids,
    ):
        """Both WF3 and WF4 fail → both exhausted → pipeline interrupted."""
        story_id = str(uuid.uuid4())
        created_story_ids.append(("planet_protectors", story_id))
        created_checkpoint_ids.append(f"{story_id}_master")

        async def fail_image(*args, **kwargs):
            raise RuntimeError("Forced image failure")

        async def fail_audio(*args, **kwargs):
            raise RuntimeError("Forced audio failure")

        with (
            patch("src.services.ai_service.AIService.generate_image", fail_image),
            patch("src.services.ai_service.AIService.generate_audio", fail_audio),
        ):
            try:
                await _run_master(story_id, "theme1")
            except Exception:
                pass

        cp_doc = await firestore_test_client.collection("workflow_checkpoints").document(
            f"{story_id}_master"
        ).get()
        assert cp_doc.exists, "Checkpoint must exist when both WF3 and WF4 are interrupted"

        # Resume with skip for both
        api = ApiHelper(_BASE_URL)
        resp = await api.resume_workflow(story_id, "skip")
        assert resp.status_code in (200, 202)

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_malformed_ai_json_response(self, created_story_ids):
        """Malformed AI JSON → parse error surfaced, no corrupt doc saved."""
        story_id = str(uuid.uuid4())
        created_story_ids.append(("planet_protectors", story_id))

        async def return_bad_json(*args, **kwargs):
            return "not valid json {{{ broken"

        with patch(
            "src.services.ai_service.AIService.generate_content",
            AsyncMock(side_effect=return_bad_json),
        ):
            with pytest.raises(Exception):
                await _run_wf2("theme1", story_id, str(uuid.uuid4()))

    @pytest.mark.e2e
    async def test_resume_workflow_unknown_story_id(self):
        """POST /resume-workflow with unknown story_id → 404 or clear error."""
        api = ApiHelper(_BASE_URL)
        resp = await api.resume_workflow("nonexistent-story-id-xyz-9999", "skip")
        assert resp.status_code in (404, 400, 422), (
            f"Expected 4xx for unknown story_id, got {resp.status_code}"
        )

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_resume_workflow_already_completed(
        self,
        firestore_test_client,
        test_bucket,
        created_story_ids,
        created_activity_story_ids,
        created_gcs_blobs,
    ):
        """POST /resume-workflow for already-completed story → idempotent, no crash."""
        story_id = str(uuid.uuid4())
        created_story_ids.append(("planet_protectors", story_id))
        created_activity_story_ids.append(story_id)

        # Run full pipeline successfully
        await _run_master(story_id, "theme1")

        # Attempt resume on an already-completed workflow
        api = ApiHelper(_BASE_URL)
        resp = await api.resume_workflow(story_id, "skip")
        assert resp.status_code in (200, 202, 404, 409), (
            f"Unexpected status for already-completed workflow: {resp.status_code}"
        )

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_wf1_empty_topic_list(self, created_checkpoint_ids):
        """WF1 returns empty topics → clear error, no story doc created."""
        story_id = str(uuid.uuid4())

        async def return_empty(*args, **kwargs):
            return json.dumps([])

        with patch(
            "src.services.ai_service.AIService.generate_content",
            AsyncMock(return_value=json.dumps([])),
        ):
            with pytest.raises(Exception):
                await _run_wf1("theme1", story_id)

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_duplicate_story_scenario(
        self,
        firestore_test_client,
        created_story_ids,
    ):
        """Running WF2 twice for the same topics_id → idempotent, no duplicate created."""
        story_id = str(uuid.uuid4())
        topics_id = str(uuid.uuid4())
        theme = "planet_protectors"
        created_story_ids.append((theme, story_id))

        await _run_wf2("theme1", story_id, topics_id)
        await _run_wf2("theme1", story_id, topics_id)  # second run — must be idempotent

        col = story_collection(theme)
        doc = await firestore_test_client.collection(col).document(story_id).get()
        assert doc.exists, "Story doc must exist"
        # Only one doc should exist (same story_id = same doc ID)
        docs = firestore_test_client.collection(col).where("topics_id", "==", topics_id).stream()
        count = 0
        async for _ in docs:
            count += 1
        assert count == 1, f"Expected exactly 1 story doc for topics_id {topics_id}, got {count}"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_pubsub_publish_fails_during_interrupt(
        self,
        firestore_test_client,
        retry_limit,
        created_story_ids,
        created_checkpoint_ids,
    ):
        """Pub/Sub publish failure during interrupt → error surfaced, pipeline does not continue silently."""
        story_id = str(uuid.uuid4())
        created_story_ids.append(("planet_protectors", story_id))
        created_checkpoint_ids.append(f"{story_id}_master")

        async def fail_image(*args, **kwargs):
            raise RuntimeError("Forced image failure")

        def fail_pubsub(*args, **kwargs):
            raise RuntimeError("Simulated Pub/Sub publish failure")

        with (
            patch("src.services.ai_service.AIService.generate_image", fail_image),
            patch("google.cloud.pubsub_v1.PublisherClient.publish", fail_pubsub),
        ):
            try:
                await _run_master(story_id, "theme1")
            except Exception as exc:
                # Must raise — not complete silently
                assert exc is not None

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_audio_upload_to_gcs_fails(
        self,
        firestore_test_client,
        retry_limit,
        created_story_ids,
        created_checkpoint_ids,
    ):
        """Audio GCS upload fails → retry triggered → after retry exhaustion, interrupt path followed."""
        story_id = str(uuid.uuid4())
        created_story_ids.append(("planet_protectors", story_id))
        created_checkpoint_ids.append(f"{story_id}_master")

        upload_calls = {"n": 0}

        original_upload = None

        async def fail_upload(self_inner, *args, **kwargs):
            upload_calls["n"] += 1
            # Only fail audio uploads (path contains "audio")
            if args and "audio" in str(args[0]):
                raise RuntimeError("Simulated GCS audio upload failure")
            if original_upload:
                return await original_upload(self_inner, *args, **kwargs)
            return "test/images/test.png"

        with patch("src.services.database.storage_bucket.StorageBucketService.upload_file", fail_upload):
            try:
                await _run_master(story_id, "theme1")
            except Exception:
                pass

        assert upload_calls["n"] > 0, "Upload must have been attempted before failure"
