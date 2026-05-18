"""
GCS storage helpers for E2E tests.
Deletes blobs under test/images/ and test/audio/ prefixes in kutty_bucket.
"""

from __future__ import annotations

from google.cloud import storage as _storage


def get_test_bucket(project: str = "riokutty") -> _storage.Bucket:
    client = _storage.Client(project=project)
    return client.bucket("kutty_bucket")


async def delete_test_blobs(bucket: _storage.Bucket, prefixes: list[str] | None = None) -> int:
    """
    Delete all blobs matching any of the given prefixes.
    Defaults to ['test/images/', 'test/audio/'] if prefixes is None.
    Returns the count of deleted blobs.
    """
    if prefixes is None:
        prefixes = ["test/images/", "test/audio/"]

    deleted = 0
    for prefix in prefixes:
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            blob.delete()
            deleted += 1
    return deleted


def assert_blob_exists(bucket: _storage.Bucket, blob_path: str) -> None:
    blob = bucket.blob(blob_path)
    assert blob.exists(), f"Expected GCS blob to exist: gs://kutty_bucket/{blob_path}"


def gcs_path_to_blob_name(gcs_url: str) -> str:
    """Convert gs://kutty_bucket/test/images/foo.png → test/images/foo.png"""
    assert gcs_url.startswith("gs://kutty_bucket/"), f"Unexpected GCS URL: {gcs_url}"
    return gcs_url[len("gs://kutty_bucket/"):]
