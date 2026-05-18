"""
HTTP client helpers for E2E tests.
Wraps /resume-workflow and /workflow-status API endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx


class ApiHelper:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def resume_workflow(
        self,
        story_id: str,
        decision: Literal["skip", "retry", "override"],
    ) -> httpx.Response:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.post(
                "/resume-workflow",
                json={"story_id": story_id, "decision": decision},
            )
        return resp

    async def get_workflow_status(self, story_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.get(f"/workflow-status/{story_id}")
        resp.raise_for_status()
        return resp.json()

    async def assert_workflow_status(self, story_id: str, expected_status: str) -> None:
        data = await self.get_workflow_status(story_id)
        actual = data.get("status")
        assert actual == expected_status, (
            f"Expected workflow status {expected_status!r}, got {actual!r} for story {story_id}"
        )
