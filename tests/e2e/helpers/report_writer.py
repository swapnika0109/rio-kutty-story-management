"""
pytest plugin: writes a structured JSON report per test session.
Output: test/reports/run_{YYYYMMDD_HHMMSS}.json

Register via conftest.py: pytest_plugins = ["tests.e2e.helpers.report_writer"]
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPORTS_DIR = Path("test/reports")
_results: list[dict[str, Any]] = []
_session_start: float = 0.0
_run_id: str = ""


def pytest_sessionstart(session) -> None:
    global _session_start, _run_id, _results
    _session_start = time.monotonic()
    _run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _results = []
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def pytest_runtest_logreport(report) -> None:
    if report.when != "call":
        return

    node_id = report.nodeid
    workflow = _infer_workflow(node_id)
    theme = _infer_theme(node_id)

    entry: dict[str, Any] = {
        "test_id": node_id,
        "workflow": workflow,
        "theme": theme,
        "status": "PASS" if report.passed else ("FAIL" if report.failed else "SKIP"),
        "duration_seconds": round(report.duration, 3),
        "failure_node": _extract_failure_node(report),
        "error": str(report.longrepr) if report.failed else None,
    }
    _results.append(entry)


def pytest_sessionfinish(session, exitstatus) -> None:
    duration = round(time.monotonic() - _session_start, 3)
    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    skipped = sum(1 for r in _results if r["status"] == "SKIP")

    report = {
        "run_id": _run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "results": _results,
        "summary": {
            "total": len(_results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
    }

    output_path = _REPORTS_DIR / f"run_{_run_id}.json"
    output_path.write_text(json.dumps(report, indent=2))


def _infer_workflow(node_id: str) -> str | None:
    for wf in ["wf1", "wf2", "wf3", "wf4", "wf5"]:
        if wf in node_id.lower():
            return wf.upper()
    if "pipeline" in node_id.lower():
        return "PIPELINE"
    return None


def _infer_theme(node_id: str) -> str | None:
    for theme in ["planet_protectors", "mindful", "chill"]:
        if theme in node_id:
            return theme
    return None


def _extract_failure_node(report) -> str | None:
    if not report.failed:
        return None
    longrepr = str(report.longrepr)
    for keyword in ["generate_image", "generate_audio", "generate_content", "save_story", "save_activity"]:
        if keyword in longrepr:
            return keyword
    return None
