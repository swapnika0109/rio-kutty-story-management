"""
E2E test fixtures for the rio-kutty story pipeline.

Fixture chain:
  session: settings_override → firestore_test_client, storage_bucket, retry_limit
  function (autouse): cleanup_firestore, cleanup_storage
  session: report_writer plugin registration
"""

from __future__ import annotations

import os
import pytest

from google.cloud import firestore as _firestore
from google.cloud import storage as _storage

from tests.e2e.helpers.firestore_helper import bulk_cleanup
from tests.e2e.helpers.storage_helper import delete_test_blobs, get_test_bucket


# ---------------------------------------------------------------------------
# pytest plugin registration
# ---------------------------------------------------------------------------

pytest_plugins = ["tests.e2e.helpers.report_writer"]


# ---------------------------------------------------------------------------
# Session-scoped: override settings to point at rio-test database
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def settings_override():
    """Force all Firestore writes to the `rio-test` named database and use test storage prefix."""
    os.environ.setdefault("FIRESTORE_DATABASE", "rio-test")
    os.environ.setdefault("TEST_STORAGE_PREFIX", "test")
    yield
    # env vars are process-scoped; no teardown needed for CI runs


# ---------------------------------------------------------------------------
# Session-scoped: Firestore async client targeting rio-test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def firestore_test_client(settings_override):
    """Async Firestore client connected to the `rio-test` named database."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "riokutty")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        client = _firestore.AsyncClient.from_service_account_json(
            credentials_path, project=project, database="rio-test"
        )
    else:
        client = _firestore.AsyncClient(project=project, database="rio-test")
    return client


# ---------------------------------------------------------------------------
# Session-scoped: GCS bucket handle
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_bucket(settings_override):
    """GCS bucket handle for kutty_bucket."""
    return get_test_bucket()


# ---------------------------------------------------------------------------
# Session-scoped: configurable retry limit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def retry_limit(settings_override):
    """Returns the configured PARALLEL_WORKFLOW_MAX_RETRIES (default 4)."""
    from src.utils.config import get_settings
    return get_settings().PARALLEL_WORKFLOW_MAX_RETRIES


# ---------------------------------------------------------------------------
# Function-scoped: resource tracking + autouse cleanup
# ---------------------------------------------------------------------------

@pytest.fixture
def created_topic_ids() -> list[tuple[str, str]]:
    """Tracks (theme, topics_id) tuples created during a test."""
    return []


@pytest.fixture
def created_story_ids() -> list[tuple[str, str]]:
    """Tracks (theme, story_id) tuples created during a test."""
    return []


@pytest.fixture
def created_activity_story_ids() -> list[str]:
    """Tracks story_ids whose activities_v1 docs should be deleted."""
    return []


@pytest.fixture
def created_checkpoint_ids() -> list[str]:
    """Tracks LangGraph thread_ids to delete from workflow_checkpoints."""
    return []


@pytest.fixture
def created_gcs_blobs() -> list[str]:
    """Tracks GCS blob paths (under test/) created during a test."""
    return []


@pytest.fixture(autouse=True)
async def cleanup_firestore(
    firestore_test_client,
    created_topic_ids,
    created_story_ids,
    created_activity_story_ids,
    created_checkpoint_ids,
):
    """Delete all Firestore docs created during the test after it completes."""
    yield
    await bulk_cleanup(
        firestore_test_client,
        topic_ids=created_topic_ids,
        story_ids=created_story_ids,
        activity_story_ids=created_activity_story_ids,
        checkpoint_thread_ids=created_checkpoint_ids,
    )


@pytest.fixture(autouse=True)
async def cleanup_storage(test_bucket, created_gcs_blobs):
    """Delete all GCS blobs created under test/images/ and test/audio/ during the test."""
    yield
    if created_gcs_blobs:
        for blob_path in created_gcs_blobs:
            blob = test_bucket.blob(blob_path)
            if blob.exists():
                blob.delete()
    else:
        # Fallback: sweep all test/ prefixes in case tracking was not used
        await delete_test_blobs(test_bucket)
