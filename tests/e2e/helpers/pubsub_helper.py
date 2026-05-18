"""
Pub/Sub assertion helpers for E2E tests.
Uses pull subscription to verify a message was published to the HITL topic.
"""

from __future__ import annotations

import json
import time
from typing import Any

from google.cloud import pubsub_v1


def assert_pubsub_message_published(
    topic: str,
    subscription: str,
    expected_attrs: dict[str, str] | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """
    Pull messages from `subscription` and assert at least one was published.
    Optionally check that message attributes match `expected_attrs`.
    Returns the first matched message data.

    Args:
        topic: Full topic path, e.g. 'projects/riokutty/topics/hitl-notifications'
        subscription: Full subscription path for the test subscriber
        expected_attrs: If provided, the pulled message must contain these attributes
        timeout_seconds: How long to wait for the message before failing

    Note: The test subscription must exist and be associated with the topic before the test runs.
    """
    subscriber = pubsub_v1.SubscriberClient()
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        response = subscriber.pull(
            request={"subscription": subscription, "max_messages": 10},
            timeout=5,
        )
        for msg in response.received_messages:
            data = json.loads(msg.message.data.decode("utf-8")) if msg.message.data else {}
            attrs = dict(msg.message.attributes)

            if expected_attrs and not all(attrs.get(k) == v for k, v in expected_attrs.items()):
                continue

            subscriber.acknowledge(
                request={"subscription": subscription, "ack_ids": [msg.ack_id]}
            )
            return data

    raise AssertionError(
        f"No Pub/Sub message received on {subscription} within {timeout_seconds}s"
    )
