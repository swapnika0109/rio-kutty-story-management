# Implementation Plan: End-to-End Test Suite

**Branch**: `001-e2e-test-suite` | **Date**: 2026-04-26 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `specs/001-e2e-test-suite/spec.md`

## Summary

Build a comprehensive end-to-end test suite covering all five story generation workflows (WF1–WF5) individually and as a complete pipeline. Tests run against a named `rio-test` Firestore database (same GCP project, same collection names, isolated by database name) and dedicated `test/images/` and `test/audio/` storage folders. The suite validates happy paths, retry exhaustion + human-in-the-loop interrupts, topic deduplication, DeepEval GEval quality gates (threshold >= 0.7), parallel WF3+WF4 execution, and all edge cases. A structured JSON report is emitted per run. Full pipeline must complete within 5 minutes.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: pytest, pytest-asyncio, deepeval, langgraph, google-cloud-firestore, google-cloud-pubsub, httpx (for API calls), asyncio  
**Storage**: Firestore named database `rio-test` (same GCP project as production); GCS `kutty_bucket` under `test/images/` and `test/audio/` prefixes  
**Testing**: pytest with pytest-asyncio; existing test structure under `tests/integration/` and `tests/unit/`  
**Target Platform**: Linux CI (GitHub Actions compatible) + local dev  
**Performance Goals**: Full pipeline E2E test completes in < 5 minutes  
**Constraints**: Real AI API calls in E2E tests; `FIRESTORE_DATABASE=rio-test` env var switches database; retry count driven by `PARALLEL_WORKFLOW_MAX_RETRIES` config (default 4, configurable)  
**Scale/Scope**: 5 workflows × 3 themes + edge cases + retry/interrupt paths ≈ ~40 test cases

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Workflow-First Architecture | PASS | Tests invoke compiled workflow graphs, not individual nodes directly |
| II. Agent Responsibility Boundary | PASS | Unit tests mock `AIService`; E2E tests let agents call through normally |
| III. Versioned Prompts | PASS | Tests use existing prompt files; no prompt hardcoding in test code |
| IV. Resilience at Service Layer | PASS | Retry tests validate `PARALLEL_WORKFLOW_MAX_RETRIES` config value, not hardcoded 4 |
| V. Human-in-Loop on Failure | PASS | FR-005/FR-006 explicitly test interrupt + resume paths |
| VI. Test-First for Agents | PASS | New conftest fixtures and helpers added alongside tests |
| VII. Cost-Conscious Model Selection | PASS | E2E tests use real models per config; no model overrides in test code |
| VIII. Idempotent Pipelines | PASS | FR-010 cleanup + idempotency edge case (duplicate story) both tested |
| IX. Async-First | PASS | All test functions are `async`; pytest-asyncio mode=auto |
| X. Observability | PASS | JSON report per run; no `print()` in test helpers |

**No violations. Gates pass.**

## Project Structure

### Documentation (this feature)

```text
specs/001-e2e-test-suite/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── clarify.md           # Clarification record
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
tests/
├── conftest.py                          # Existing — extend with E2E fixtures
├── e2e/                                 # NEW — end-to-end test suite
│   ├── __init__.py
│   ├── conftest.py                      # E2E fixtures: rio-test DB client, cleanup, report writer
│   ├── helpers/
│   │   ├── __init__.py
│   │   ├── firestore_helper.py          # Assert doc structure, cleanup collections
│   │   ├── storage_helper.py            # Upload/delete test/images/ and test/audio/
│   │   ├── pubsub_helper.py             # Assert Pub/Sub message published
│   │   ├── api_helper.py                # HTTP client for /resume-workflow, /workflow-status
│   │   └── report_writer.py             # Emit structured JSON report per run
│   ├── test_wf1_topics.py               # WF1: topic generation, deduplication, caching
│   ├── test_wf2_story.py                # WF2: story creation, Firestore persistence
│   ├── test_wf3_image.py                # WF3: image generation, test/images/ storage, retry
│   ├── test_wf4_audio.py                # WF4: audio generation, test/audio/ storage, retry
│   ├── test_wf5_activities.py           # WF5: activity generation, DeepEval GEval >= 0.7
│   ├── test_pipeline_happy_path.py      # Full WF1→WF2→WF3+WF4→WF5 pipeline, all 3 themes
│   ├── test_pipeline_retry_hitl.py      # Retry exhaustion + human-in-the-loop + resume
│   └── test_pipeline_edge_cases.py      # All edge cases from spec
├── integration/                         # Existing — keep; extend if needed
│   ├── test_workflow.py
│   ├── test_api.py
│   └── test_deepeval_activity_llms.py
└── unit/                                # Existing — keep unchanged
    ├── test_agents/
    └── test_utils/

test/                                    # Test artifact output (gitignored)
├── images/                              # WF3 test image files (cleaned up post-run)
├── audio/                               # WF4 test audio files (cleaned up post-run)
└── reports/                             # JSON run reports (kept for CI artifact upload)
```

**Structure Decision**: Single project layout extending the existing `tests/` tree. A new `tests/e2e/` directory is added alongside the existing `unit/` and `integration/` directories. Test output artifacts (images, audio, reports) go under a top-level `test/` directory (different from `tests/`) and are gitignored except for reports.

## Phase 0: Research

### Decision Log

**Decision 1: Firestore database switching mechanism**
- **Chosen**: `FIRESTORE_DATABASE=rio-test` environment variable, read by `src/utils/config.py` `FIRESTORE_DATABASE` field. `FirestoreService` already accepts this config value — no code changes needed, only test env setup.
- **Rationale**: Least invasive; uses existing config pattern; same GCP credentials work across both databases.
- **Alternatives considered**: Separate GCP project (rejected — requires separate credentials + quota setup); Firestore emulator (rejected — misses real auth/network behavior per clarification Q2).

**Decision 2: Test storage path isolation**
- **Chosen**: GCS path prefix `test/images/` and `test/audio/` within `kutty_bucket`. Tests pass these prefixes via a `TEST_STORAGE_PREFIX` env var or fixture override on `StorageBucketService`.
- **Rationale**: Reuses existing bucket and credentials; cleanup is a simple prefix delete; no bucket IAM changes needed.
- **Alternatives considered**: Separate test bucket (rejected — adds setup overhead); local filesystem (rejected — doesn't validate real GCS upload behavior).

**Decision 3: Retry count configurability**
- **Chosen**: Tests read `settings.PARALLEL_WORKFLOW_MAX_RETRIES` (currently defaults to 4) at fixture setup time. Test assertions use this value, not a hardcoded literal. A `pytest.ini` or env var can override `PARALLEL_WORKFLOW_MAX_RETRIES` for specific test scenarios.
- **Rationale**: Matches clarification Q5 (configurable, default 4); prevents tests from being brittle when the default changes.

**Decision 4: Human-in-the-loop test strategy**
- **Chosen**: WF3/WF4 service calls are replaced with a failing mock using `unittest.mock.patch` at the service layer (not agent layer) for the retry/interrupt tests. The rest of the pipeline (checkpointer, Pub/Sub, interrupt) runs against real infrastructure (`rio-test` DB, real Pub/Sub topic).
- **Rationale**: Avoids wasting real API quota for failure simulation; still validates the real interrupt + checkpoint + resume path end-to-end.

**Decision 5: JSON report structure**
- **Chosen**: One JSON file per run at `test/reports/run_{timestamp}.json` with schema:
  ```json
  {
    "run_id": "...",
    "timestamp": "...",
    "duration_seconds": 0.0,
    "results": [
      {
        "test_id": "test_wf1_happy_path[planet_protectors]",
        "workflow": "WF1",
        "theme": "planet_protectors",
        "status": "PASS|FAIL|SKIP",
        "duration_seconds": 0.0,
        "failure_node": null,
        "error": null
      }
    ],
    "summary": { "total": 0, "passed": 0, "failed": 0, "skipped": 0 }
  }
  ```
- **Rationale**: Machine-readable for CI; maps directly to FR-011 requirement; aligns with pytest's existing result model.

**Decision 6: DeepEval GEval integration**
- **Chosen**: Use DeepEval `GEval` metric with `gemini-2.0-flash-lite` as the evaluation model (per constitution §VII). Minimum score threshold = 0.7. Evaluated per activity type (MCQ, art, science, moral) after WF5 generates them.
- **Alternatives considered**: Raw LLM prompt scoring (rejected per project preference for DeepEval).

## Phase 1: Design & Contracts

See [data-model.md](./data-model.md) for full entity definitions.

### Key Design Decisions

**E2E Conftest fixture chain**:
```
session-scoped: settings_override (FIRESTORE_DATABASE=rio-test)
  → session-scoped: firestore_test_client (google.cloud.firestore.AsyncClient with database='rio-test')
  → function-scoped: cleanup_firestore (deletes all docs created during test)
  → function-scoped: cleanup_storage (deletes test/images/* and test/audio/* created during test)
  → session-scoped: report_writer (accumulates results, writes JSON at session end)
```

**Parallelism validation** (SC-006):
- WF3+WF4 are invoked via `asyncio.gather()` in the master workflow.
- Test records `start_time` before `gather`, records individual completion times after.
- Asserts `max(wf3_end, wf4_end) - start < wf3_duration + wf4_duration` (parallel, not serial).

**Edge case test pattern**:
- Each edge case in spec maps to exactly one `test_pipeline_edge_cases.py::test_*` function.
- Edge cases that require service failure use `pytest.raises` or assert on the JSON report's `failure_node` field.

### Interface Contracts

The E2E test suite calls two existing API endpoints (no new endpoints):

**POST /resume-workflow**
```
Body: { "story_id": str, "decision": "skip" | "retry" | "override" }
Response: 202 Accepted
```

**GET /workflow-status/{story_id}**
```
Response: { "status": "pending" | "running" | "completed" | "failed" | "interrupted", "story_id": str }
```

These are read-only from the test perspective — no contract changes required.

### Environment Variables for Tests

| Variable | Value for E2E | Purpose |
|----------|--------------|---------|
| `FIRESTORE_DATABASE` | `rio-test` | Routes all Firestore writes to test database |
| `GOOGLE_API_KEY` | real key | Gemini + TTS access |
| `HF_TOKEN` | real token | FLUX.1-schnell access |
| `PARALLEL_WORKFLOW_MAX_RETRIES` | 4 (or override) | Configurable retry limit |
| `HUMAN_LOOP_NOTIFICATION_TOPIC` | test Pub/Sub topic | Interrupt notifications |
| `TEST_STORAGE_PREFIX` | `test` | Routes image/audio to `test/images/` and `test/audio/` |

### pytest Configuration additions (`pytest.ini` or `pyproject.toml`)

```ini
[pytest]
asyncio_mode = auto
markers =
    e2e: end-to-end tests requiring real API keys and rio-test Firestore
    slow: tests that may take > 60 seconds
    integration: existing integration tests
```

Run commands:
```bash
# All E2E tests
FIRESTORE_DATABASE=rio-test pytest tests/e2e/ -m e2e --timeout=300

# Single workflow
FIRESTORE_DATABASE=rio-test pytest tests/e2e/test_wf1_topics.py -v

# Skip slow (no full pipeline)
pytest tests/e2e/ -m "e2e and not slow"
```
