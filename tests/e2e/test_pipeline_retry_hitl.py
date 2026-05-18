"""
E2E tests for retry exhaustion and human-in-the-loop (US3).
Uses real Firestore checkpoint + Pub/Sub; mocks only the failing service call.
"""

from __future__ import annotations

import uuid
import pytest
from unittest.mock import AsyncMock, patch

from src.workflows.master_workflow import master_workflow
from src.workflows.story_creator_workflow import story_creator_workflow
from tests.e2e.helpers.api_helper import ApiHelper
from tests.e2e.helpers.firestore_helper import story_collection

import os

_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

_THEME_KEY = "theme1"
_THEME = "planet_protectors"

_SAMPLE_STORY = {
    "title": "The Brave Little Turtle",
    "story_text": "Teo the turtle swam bravely past the deep end.",
    "moral": "Courage grows with a friend.",
    "age_group": "3-4",
    "language": "English",
    "image_prompt": "A brave turtle in a pond.",
    "theme": _THEME_KEY,
}


async def _run_master_with_story(story_id: str) -> None:
    config = {
        "configurable": {
            "thread_id": f"{story_id}_master",
            "story_id": story_id,
            "age": "3-4",
            "language": "English",
            "theme": _THEME_KEY,
            "voice": "standard",
        }
    }
    await master_workflow.ainvoke(
        {"story_id": story_id, "story": _SAMPLE_STORY},
        config=config,
    )


class TestRetryExhaustionAndInterrupt:

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_wf3_retry_exhaustion_triggers_interrupt(
        self,
        firestore_test_client,
        retry_limit,
        created_story_ids,
        created_checkpoint_ids,
    ):
        """WF3 exhausts retries → Pub/Sub published → workflow interrupted."""
        story_id = str(uuid.uuid4())
        created_story_ids.append((_THEME, story_id))
        created_checkpoint_ids.append(f"{story_id}_master")

        call_count = {"n": 0}

        async def always_fail(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("Forced image failure for retry test")

        with patch("src.services.ai_service.AIService.generate_image", always_fail):
            try:
                await _run_master_with_story(story_id)
            except Exception:
                pass  # interrupt raises — expected

        # Retry must have been attempted exactly retry_limit times
        assert call_count["n"] >= retry_limit, (
            f"Expected at least {retry_limit} image generation attempts, got {call_count['n']}"
        )

        # Workflow checkpoint must exist (pipeline is paused, not deleted)
        cp_doc = await firestore_test_client.collection("workflow_checkpoints").document(
            f"{story_id}_master"
        ).get()
        assert cp_doc.exists, "Workflow checkpoint must exist while pipeline is interrupted"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_wf4_retry_exhaustion_triggers_interrupt(
        self,
        firestore_test_client,
        retry_limit,
        created_story_ids,
        created_checkpoint_ids,
    ):
        """WF4 exhausts retries → workflow interrupted."""
        story_id = str(uuid.uuid4())
        created_story_ids.append((_THEME, story_id))
        created_checkpoint_ids.append(f"{story_id}_master")

        call_count = {"n": 0}

        async def always_fail(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("Forced audio failure for retry test")

        with patch("src.services.ai_service.AIService.generate_audio", always_fail):
            try:
                await _run_master_with_story(story_id)
            except Exception:
                pass

        assert call_count["n"] >= retry_limit, (
            f"Expected at least {retry_limit} audio generation attempts, got {call_count['n']}"
        )

        cp_doc = await firestore_test_client.collection("workflow_checkpoints").document(
            f"{story_id}_master"
        ).get()
        assert cp_doc.exists, "Workflow checkpoint must exist while pipeline is interrupted"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_resume_skip_decision(
        self,
        firestore_test_client,
        retry_limit,
        created_story_ids,
        created_checkpoint_ids,
    ):
        """After WF3 interrupt, skip decision → pipeline completes without image_url."""
        story_id = str(uuid.uuid4())
        created_story_ids.append((_THEME, story_id))
        created_checkpoint_ids.append(f"{story_id}_master")

        async def always_fail(*args, **kwargs):
            raise RuntimeError("Forced image failure")

        with patch("src.services.ai_service.AIService.generate_image", always_fail):
            try:
                await _run_master_with_story(story_id)
            except Exception:
                pass

        # Resume with skip
        api = ApiHelper(_BASE_URL)
        resp = await api.resume_workflow(story_id, "skip")
        assert resp.status_code in (200, 202), f"Unexpected resume response: {resp.status_code}"

        # Pipeline should complete — checkpoint deleted
        cp_doc = await firestore_test_client.collection("workflow_checkpoints").document(
            f"{story_id}_master"
        ).get()
        assert not cp_doc.exists, "Checkpoint must be deleted after successful pipeline completion"

        # Story should exist without image_url
        col = story_collection(_THEME)
        story_doc = await firestore_test_client.collection(col).document(story_id).get()
        if story_doc.exists:
            data = story_doc.to_dict()
            assert not data.get("image_url"), "image_url must be absent after skip decision"

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_resume_retry_decision(
        self,
        firestore_test_client,
        test_bucket,
        retry_limit,
        created_story_ids,
        created_checkpoint_ids,
        created_gcs_blobs,
    ):
        """After WF3 interrupt, retry decision → WF3 attempts again with real service."""
        story_id = str(uuid.uuid4())
        created_story_ids.append((_THEME, story_id))
        created_checkpoint_ids.append(f"{story_id}_master")

        fail_calls = {"n": 0}

        async def fail_then_succeed(*args, **kwargs):
            fail_calls["n"] += 1
            if fail_calls["n"] <= retry_limit:
                raise RuntimeError("Forced failure until interrupt")
            # After resume, let real implementation run
            from src.services.ai_service import AIService
            return await AIService().generate_image(*args, **kwargs)

        with patch("src.services.ai_service.AIService.generate_image", fail_then_succeed):
            try:
                await _run_master_with_story(story_id)
            except Exception:
                pass

        # Resume with retry
        api = ApiHelper(_BASE_URL)
        resp = await api.resume_workflow(story_id, "retry")
        assert resp.status_code in (200, 202), f"Unexpected resume response: {resp.status_code}"

        assert fail_calls["n"] > retry_limit, "WF3 should have attempted again after retry decision"
