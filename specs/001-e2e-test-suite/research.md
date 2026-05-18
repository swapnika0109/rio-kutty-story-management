# Research: End-to-End Test Suite

**Date**: 2026-04-26 | **Feature**: 001-e2e-test-suite

All NEEDS CLARIFICATION items from Technical Context have been resolved. This document records the key decisions and rationale.

---

## 1. Firestore Multi-Database Isolation

**Decision**: Use Firestore named database `rio-test` within the same GCP project.

**How**: Set `FIRESTORE_DATABASE=rio-test` in the test environment. `src/utils/config.py` already reads this as `Settings.FIRESTORE_DATABASE`. `FirestoreService` passes it to the Firestore client constructor. No production code changes needed.

**Rationale**: Named databases are a native Firestore feature (GA since 2023). The default `(default)` database remains untouched. Same credentials, same project — zero additional IAM setup. Cleanup is straightforward (delete all docs in known collections).

**Alternatives rejected**:
- Separate GCP project: requires separate service account, quota, and billing — overkill for test isolation.
- Firestore emulator: doesn't validate real network/auth behavior (clarification Q2 answer).
- Namespace prefix on collection names: violates the "same collection names as production" requirement.

---

## 2. Test Storage Path Isolation (GCS)

**Decision**: Use GCS path prefixes `test/images/` and `test/audio/` within the existing `kutty_bucket`.

**How**: A `TEST_STORAGE_PREFIX=test` env var is read by E2E fixtures. When set, `StorageBucketService` upload paths are prefixed with `test/` → files land at `gs://kutty_bucket/test/images/...` and `gs://kutty_bucket/test/audio/...`. Cleanup deletes all blobs under `test/` after each test.

**Rationale**: Reuses existing bucket, credentials, and upload code path. Tests validate real GCS uploads. Cleanup is a single `bucket.list_blobs(prefix="test/")` + delete loop.

**Alternatives rejected**:
- Separate test bucket: requires separate IAM + bucket creation; unnecessary overhead.
- Local filesystem mock: doesn't validate real GCS upload behavior.

---

## 3. Retry Count Configurability

**Decision**: Tests read `settings.PARALLEL_WORKFLOW_MAX_RETRIES` at runtime; never assert on literal `4`.

**How**: E2E conftest exposes a `retry_limit` fixture that returns `get_settings().PARALLEL_WORKFLOW_MAX_RETRIES`. All retry-related assertions use this fixture value. To test with a different retry count, set `PARALLEL_WORKFLOW_MAX_RETRIES=2` before the test run.

**Rationale**: Clarification Q5 confirmed 4 is the default but configurable per workflow. Hardcoding `4` in assertions would break whenever the config default changes.

---

## 4. Human-in-the-Loop Test Strategy

**Decision**: Mock the service layer (not the agent) for failure simulation; let the checkpoint/interrupt/Pub/Sub path run against real infrastructure.

**How**:
- `patch("src.services.ai_service.AIService.generate_image", side_effect=Exception("forced failure"))` for WF3 tests.
- `patch("src.services.ai_service.AIService.generate_audio", side_effect=Exception("forced failure"))` for WF4 tests.
- The LangGraph master workflow, checkpointer (writing to `rio-test`), `interrupt()`, and Pub/Sub publish all run for real.
- After interrupt, test calls `POST /resume-workflow` via `httpx.AsyncClient`.

**Rationale**: Tests the actual interrupt/resume code path end-to-end. Avoids wasting real API quota for intentional failure scenarios. Aligns with constitution §VI (mock `AIService` at call site).

---

## 5. JSON Test Report Schema

**Decision**: One JSON file per run at `test/reports/run_{YYYYMMDD_HHMMSS}.json`.

**Schema**:
```json
{
  "run_id": "20260426_143022",
  "timestamp": "2026-04-26T14:30:22Z",
  "duration_seconds": 187.4,
  "results": [
    {
      "test_id": "test_wf1_happy_path[planet_protectors]",
      "workflow": "WF1",
      "theme": "planet_protectors",
      "status": "PASS",
      "duration_seconds": 12.1,
      "failure_node": null,
      "error": null
    },
    {
      "test_id": "test_wf3_retry_exhaustion",
      "workflow": "WF3",
      "theme": "planet_protectors",
      "status": "FAIL",
      "duration_seconds": 45.2,
      "failure_node": "generate_image",
      "error": "RetryExhausted: 4 attempts failed"
    }
  ],
  "summary": {
    "total": 42,
    "passed": 40,
    "failed": 2,
    "skipped": 0
  }
}
```

**How**: A pytest plugin (`tests/e2e/helpers/report_writer.py`) hooks into `pytest_runtest_logreport` to accumulate results, then writes the file in `pytest_sessionfinish`.

---

## 6. DeepEval GEval Integration

**Decision**: Use `deepeval.metrics.GEvalMetric` with `gemini-2.0-flash-lite` as the judge model. Threshold = 0.7.

**How**:
```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

metric = GEval(
    name="ActivityQuality",
    criteria="Is the activity age-appropriate, educational, and well-structured?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model="gemini/gemini-2.0-flash-lite"  # via deepeval's LiteLLM integration
)
```

**Rationale**: DeepEval is already in the project dependencies. GEval is the appropriate metric for open-ended quality evaluation. `gemini-2.0-flash-lite` is the cheapest model per constitution §VII.

---

## 7. Parallel Execution Validation

**Decision**: Time-based assertion using `asyncio` event loop timestamps.

**How**:
```python
t_start = asyncio.get_event_loop().time()
await master_workflow.run(...)  # internally uses asyncio.gather for WF3+WF4
t_end = asyncio.get_event_loop().time()
total = t_end - t_start
# WF3 and WF4 durations are captured via state timestamps
assert total < wf3_duration + wf4_duration  # proves parallel, not serial
```

**Rationale**: Simple, no additional dependencies. Captures real wall-clock concurrency.
