# Tasks: End-to-End Test Suite

**Input**: Design documents from `specs/001-e2e-test-suite/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅

**Organization**: Tasks grouped by user story for independent implementation and testing.
**Tests**: E2E test files ARE the deliverable — every task produces test code.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the `tests/e2e/` directory structure, pytest configuration, output artifact directories, and gitignore entries. No implementation logic yet.

- [x] T001 Create `tests/e2e/` directory with `__init__.py` and `tests/e2e/helpers/__init__.py`
- [x] T002 Create `test/images/`, `test/audio/`, and `test/reports/` output directories with `.gitkeep`
- [x] T003 Add `test/images/`, `test/audio/` to `.gitignore` (keep `test/reports/` tracked)
- [x] T004 Add `e2e` and `slow` pytest markers and `asyncio_mode = auto` to `pytest.ini` (or `pyproject.toml`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the shared helpers and E2E conftest that ALL user story tests depend on. Nothing in Phase 3+ can work without these.

**⚠️ CRITICAL**: No E2E test can run until this phase is complete.

- [x] T005 Implement `tests/e2e/helpers/firestore_helper.py` — async helpers to assert Firestore doc structure (field presence, types, values) and bulk-delete docs from `{theme}_topics`, `{theme}_stories`, `activities_v1`, `workflow_checkpoints` in the `rio-test` database
- [x] T006 [P] Implement `tests/e2e/helpers/storage_helper.py` — helpers to delete all GCS blobs under `test/images/` and `test/audio/` prefixes in `kutty_bucket`
- [x] T007 [P] Implement `tests/e2e/helpers/pubsub_helper.py` — helper to assert that a Pub/Sub message was published to the configured `HUMAN_LOOP_NOTIFICATION_TOPIC` (pull-based assertion with timeout)
- [x] T008 [P] Implement `tests/e2e/helpers/api_helper.py` — async HTTP client wrapper around `httpx.AsyncClient` for `POST /resume-workflow` and `GET /workflow-status/{story_id}`
- [x] T009 [P] Implement `tests/e2e/helpers/report_writer.py` — pytest plugin that hooks `pytest_runtest_logreport` and `pytest_sessionfinish` to write `test/reports/run_{YYYYMMDD_HHMMSS}.json` with the schema defined in `research.md §5`
- [x] T010 Implement `tests/e2e/conftest.py` — session-scoped `settings_override` fixture (`FIRESTORE_DATABASE=rio-test`, `TEST_STORAGE_PREFIX=test`), session-scoped `firestore_test_client` (`AsyncClient(project="riokutty", database="rio-test")`), function-scoped `cleanup_firestore` and `cleanup_storage` autouse fixtures, session-scoped `report_writer` registration, and `retry_limit` fixture reading `settings.PARALLEL_WORKFLOW_MAX_RETRIES`

**Checkpoint**: All helpers and conftest ready — E2E tests can now be written and run.

---

## Phase 3: User Story 1 — Full Happy Path Pipeline (Priority: P1) 🎯 MVP

**Goal**: Verify the complete WF1→WF2→WF3+WF4→WF5 pipeline runs successfully for all 3 themes, persists all fields to `rio-test`, and completes within 5 minutes.

**Independent Test**: `FIRESTORE_DATABASE=rio-test pytest tests/e2e/test_pipeline_happy_path.py -m e2e -v`

- [x] T011 [US1] Implement `tests/e2e/test_wf1_topics.py` — `test_wf1_happy_path[planet_protectors/mindful/chill]` (parametrized): submit theme to WF1, assert `{theme}_topics` doc exists in `rio-test` with valid `topics_id` and non-empty `topics` array; cleanup via `cleanup_firestore`
- [x] T012 [P] [US1] Implement `tests/e2e/test_wf2_story.py` — `test_wf2_story_persisted[planet_protectors/mindful/chill]`: given a `topics_id`, run WF2, assert `{theme}_stories` doc has `title`, `description`, `moral`, `topics_id` as doc ID; cleanup via `cleanup_firestore`
- [x] T013 [P] [US1] Implement `tests/e2e/test_wf3_image.py` — `test_wf3_image_generated[planet_protectors/mindful/chill]`: run WF3, assert `image_url` on story doc starts with `gs://kutty_bucket/test/images/`; assert blob exists in GCS; cleanup via `cleanup_storage`
- [x] T014 [P] [US1] Implement `tests/e2e/test_wf4_audio.py` — `test_wf4_audio_generated[planet_protectors/mindful/chill]`: run WF4, assert `audio_url` on story doc starts with `gs://kutty_bucket/test/audio/`; assert blob exists in GCS; cleanup via `cleanup_storage`
- [x] T015 [P] [US1] Implement `tests/e2e/test_wf5_activities.py` (happy path section) — `test_wf5_activities_generated[planet_protectors/mindful/chill]`: run WF5, assert 4 activity docs exist in `activities_v1` tagged with `story_id`, one per type (`mcq`, `art`, `science`, `moral`); assert `activity_image_url` present for art/science/moral activities (starts with `gs://kutty_bucket/test/images/`); cleanup via `cleanup_firestore` + `cleanup_storage`
- [x] T016 [US1] Implement `tests/e2e/test_pipeline_happy_path.py` — `test_full_pipeline[planet_protectors/mindful/chill]` (parametrized, marked `@pytest.mark.slow`): run WF1→select topic→WF2→parallel WF3+WF4→WF5 end-to-end; assert all fields on story doc and all 4 activities; assert total duration < 300 seconds (5 min); cleanup all created resources

**Checkpoint**: Full happy path passing for all 3 themes — User Story 1 complete.

---

## Phase 4: User Story 2 — Topic Deduplication and Caching (Priority: P2)

**Goal**: Verify WF1 excludes previously generated titles on successive calls and correctly caches to `{theme}_topics` in `rio-test`.

**Independent Test**: `FIRESTORE_DATABASE=rio-test pytest tests/e2e/test_wf1_topics.py::TestDeduplication -m e2e -v`

- [x] T017 [US2] Implement deduplication tests in `tests/e2e/test_wf1_topics.py` (new `TestDeduplication` class) — `test_wf1_no_duplicate_titles[planet_protectors/mindful/chill]`: call WF1 twice for same theme; collect both `topics` arrays; assert no title string appears in both; cleanup both topic docs
- [x] T018 [P] [US2] Implement caching tests in `tests/e2e/test_wf1_topics.py` — `test_wf1_topics_cached_in_firestore[planet_protectors/mindful/chill]`: after WF1 call, assert `{theme}_topics` doc exists with `topics_id` and `topics` array; call WF1 again; assert second call reads from existing doc (no new LLM call — verify by checking doc `created_at` unchanged); cleanup

**Checkpoint**: Deduplication and caching verified for all 3 themes — User Story 2 complete.

---

## Phase 5: User Story 3 — Retry and Human-in-the-Loop on Failure (Priority: P2)

**Goal**: Verify retry exhaustion triggers Pub/Sub notification + LangGraph interrupt, and that `/resume-workflow` with `skip` or `retry` decision resumes correctly.

**Independent Test**: `FIRESTORE_DATABASE=rio-test pytest tests/e2e/test_pipeline_retry_hitl.py -m e2e -v`

- [x] T019 [US3] Implement `tests/e2e/test_pipeline_retry_hitl.py` — `test_wf3_retry_exhaustion_triggers_interrupt`: patch `AIService.generate_image` to always raise; run pipeline; assert retry counter reaches `retry_limit` fixture value; assert Pub/Sub notification published (via `pubsub_helper`); assert workflow checkpoint exists in `rio-test`; assert `GET /workflow-status` returns `"interrupted"`
- [x] T020 [P] [US3] Add `test_wf4_retry_exhaustion_triggers_interrupt` in `tests/e2e/test_pipeline_retry_hitl.py` — same pattern as T019 but patches `AIService.generate_audio`
- [x] T021 [US3] Add `test_resume_skip_decision` in `tests/e2e/test_pipeline_retry_hitl.py` — after interrupt (WF3 mock), POST `{"decision": "skip"}` to `/resume-workflow`; assert pipeline continues; assert story doc has no `image_url`; assert checkpoint deleted; assert `GET /workflow-status` returns `"completed"`
- [x] T022 [P] [US3] Add `test_resume_retry_decision` in `tests/e2e/test_pipeline_retry_hitl.py` — after interrupt (WF3 mock), restore real `AIService.generate_image`, POST `{"decision": "retry"}`; assert WF3 attempts again; assert `image_url` populated if retry succeeds

**Checkpoint**: Full retry + interrupt + resume paths verified — User Story 3 complete.

---

## Phase 6: User Story 4 — Activity Generation Quality / DeepEval (Priority: P3)

**Goal**: Verify all 4 activity types pass DeepEval GEval evaluation with score >= 0.7, and that retry-on-low-score behavior works.

**Independent Test**: `FIRESTORE_DATABASE=rio-test pytest tests/e2e/test_wf5_activities.py -m e2e -v`

- [x] T023 [US4] Implement DeepEval GEval tests in `tests/e2e/test_wf5_activities.py` (new `TestDeepEval` class) — `test_mcq_geval_score_above_threshold`: run WF5 for a fixed story; retrieve MCQ activity from `rio-test`; evaluate with `GEval(criteria="...", threshold=0.7, model="gemini/gemini-2.0-flash-lite")`; assert score >= 0.7; cleanup
- [x] T024 [P] [US4] Add `test_art_activity_geval_score_above_threshold` in `tests/e2e/test_wf5_activities.py` — same as T023 for art activity; also assert `activity_image_url` present and GCS blob exists; cleanup
- [x] T025 [P] [US4] Add `test_science_activity_geval_score_above_threshold` in `tests/e2e/test_wf5_activities.py` — same pattern for science activity; assert `activity_image_url` present; cleanup
- [x] T026 [P] [US4] Add `test_moral_activity_geval_score_above_threshold` in `tests/e2e/test_wf5_activities.py` — same pattern for moral activity; assert `activity_image_url` present; cleanup
- [x] T027 [US4] Add `test_wf5_retries_on_low_geval_score` in `tests/e2e/test_wf5_activities.py` — patch GEval to return score 0.5 on first call, 0.8 on retry; run WF5; assert activity doc has final score >= 0.7; assert retry count > 1 in workflow state; cleanup

**Checkpoint**: All activity types pass DeepEval quality gate — User Story 4 complete.

---

## Phase 7: Edge Cases (Cross-Cutting)

**Goal**: Every edge case listed in the spec has a dedicated, deterministic test. No silent failures.

**Independent Test**: `FIRESTORE_DATABASE=rio-test pytest tests/e2e/test_pipeline_edge_cases.py -m e2e -v`

- [x] T028 [P] Implement `tests/e2e/test_pipeline_edge_cases.py` — `test_invalid_theme_rejected`: submit unsupported theme (e.g., `"dragons"`); assert pipeline raises or returns a clear error; assert no Firestore docs created
- [x] T029 [P] Add `test_firestore_unavailable_on_story_save`: patch `FirestoreService.save_story` to raise; run WF2; assert error is surfaced clearly (not swallowed); assert checkpoint exists for recovery
- [x] T030 [P] Add `test_both_wf3_wf4_fail_simultaneously`: patch both `generate_image` and `generate_audio` to always raise; exhaust retries on both; assert two Pub/Sub notifications published; assert workflow interrupted; resume with `skip` for both; assert pipeline completes without image or audio
- [x] T031 [P] Add `test_malformed_ai_json_response`: patch `AIService.generate_content` to return `"not valid json {{{"`; run WF2; assert error is raised with a parse failure message; assert no corrupt doc saved to Firestore
- [x] T032 [P] Add `test_resume_workflow_unknown_story_id`: POST to `/resume-workflow` with `story_id="nonexistent-id"`; assert 404 or clear error response
- [x] T033 [P] Add `test_resume_workflow_already_completed`: complete a full pipeline; then POST to `/resume-workflow` for the same story_id; assert idempotent response (no crash, no duplicate processing)
- [x] T034 [P] Add `test_wf1_empty_topic_list`: patch WF1 to return empty `topics` array; assert pipeline raises or returns a clear empty-topics error; assert no story doc created
- [x] T035 [P] Add `test_duplicate_story_scenario`: run WF2 for the same `topics_id` twice; assert second run is idempotent (existing story doc unchanged, no duplicate created — per constitution §VIII)
- [x] T036 [P] Add `test_pubsub_publish_fails_during_interrupt`: patch `pubsub_v1.PublisherClient.publish` to raise; exhaust WF3 retries; assert error is surfaced (not silently swallowed); assert workflow does not continue silently
- [x] T037 [P] Add `test_audio_upload_to_gcs_fails`: patch `StorageBucketService.upload_file` to raise for audio only; run WF4; assert retry triggered; after retries exhausted, assert interrupt/Pub/Sub path followed

**Checkpoint**: All 10 edge cases covered with deterministic outcomes.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Parallel execution validation, cleanup verification, and CI configuration.

- [x] T038 Add `test_wf3_wf4_run_in_parallel` in `tests/e2e/test_pipeline_happy_path.py` — record `start_time`; run full pipeline; record individual WF3/WF4 completion timestamps from state; assert `total_elapsed < wf3_duration + wf4_duration` (proves `asyncio.gather`, not serial)
- [x] T039 [P] Add `test_firestore_cleanup_verified` in `tests/e2e/test_pipeline_happy_path.py` — run full pipeline; teardown; assert `{theme}_stories`, `{theme}_topics`, `activities_v1`, `workflow_checkpoints` contain no docs created during this test run
- [x] T040 [P] Add `test_gcs_cleanup_verified` in `tests/e2e/test_pipeline_happy_path.py` — run full pipeline; teardown; assert no blobs exist under `test/images/` or `test/audio/` created during this test run
- [x] T041 [P] Add `test_json_report_written` in `tests/e2e/test_pipeline_happy_path.py` — after a test session, assert `test/reports/run_*.json` exists, is valid JSON, has required fields (`run_id`, `timestamp`, `results`, `summary`), and `summary.total > 0`
- [x] T042 Add GitHub Actions CI workflow at `.github/workflows/e2e.yml` — triggers on push to `001-e2e-test-suite` branch; sets `FIRESTORE_DATABASE=rio-test`, `GOOGLE_API_KEY`, `HF_TOKEN` from secrets; runs `pytest tests/e2e/ -m e2e --timeout=300`; uploads `test/reports/` as CI artifact

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user story phases
- **Phase 3 (US1 Happy Path)**: Depends on Phase 2
- **Phase 4 (US2 Deduplication)**: Depends on Phase 2; independent of Phase 3
- **Phase 5 (US3 Retry/HITL)**: Depends on Phase 2; independent of Phases 3–4
- **Phase 6 (US4 DeepEval)**: Depends on Phase 2; independent of Phases 3–5
- **Phase 7 (Edge Cases)**: Depends on Phase 2; best started after Phase 3 (reuses happy path helpers)
- **Phase 8 (Polish)**: Depends on Phases 3–7 being substantially complete

### User Story Dependencies

- **US1 (P1)**: Can start after Phase 2 — foundational E2E infrastructure complete
- **US2 (P2)**: Can start after Phase 2 — adds to `test_wf1_topics.py` (independent of US1)
- **US3 (P2)**: Can start after Phase 2 — new file `test_pipeline_retry_hitl.py`
- **US4 (P3)**: Can start after Phase 2 — adds to `test_wf5_activities.py` (independent)

### Parallel Opportunities

- T005–T009 (helpers): All can run in parallel — different files
- T011–T015 (US1 per-workflow tests): All can run in parallel — different files
- T023–T026 (US4 DeepEval per activity type): All can run in parallel — different test methods
- T028–T037 (Edge cases): All can run in parallel — different test methods in same file
- T038–T041 (Polish): T039–T041 can run in parallel after T038

---

## Parallel Example: Phase 2 Foundational Helpers

```bash
# All 5 helper files can be implemented simultaneously:
Task: "Implement tests/e2e/helpers/firestore_helper.py"   # T005
Task: "Implement tests/e2e/helpers/storage_helper.py"     # T006
Task: "Implement tests/e2e/helpers/pubsub_helper.py"      # T007
Task: "Implement tests/e2e/helpers/api_helper.py"         # T008
Task: "Implement tests/e2e/helpers/report_writer.py"      # T009
# Then T010 (conftest.py) once all helpers exist
```

## Parallel Example: Phase 3 User Story 1

```bash
# Per-workflow tests can be built simultaneously after T010:
Task: "tests/e2e/test_wf1_topics.py happy path"   # T011
Task: "tests/e2e/test_wf2_story.py"               # T012
Task: "tests/e2e/test_wf3_image.py"               # T013
Task: "tests/e2e/test_wf4_audio.py"               # T014
Task: "tests/e2e/test_wf5_activities.py"          # T015
# Then T016 (full pipeline test) after T011–T015
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T004)
2. Complete Phase 2: Foundational helpers + conftest (T005–T010)
3. Complete Phase 3: Happy path per-workflow + full pipeline (T011–T016)
4. **STOP and VALIDATE**: `FIRESTORE_DATABASE=rio-test pytest tests/e2e/test_pipeline_happy_path.py -m "e2e and slow"`
5. Full pipeline green for all 3 themes → MVP achieved

### Incremental Delivery

1. Setup + Foundational → helpers and conftest ready
2. Phase 3 (US1) → happy path E2E green → MVP
3. Phase 4 (US2) → deduplication verified
4. Phase 5 (US3) → retry + HITL verified
5. Phase 6 (US4) → DeepEval quality gate verified
6. Phase 7 → all edge cases covered
7. Phase 8 → CI workflow live, parallel execution confirmed

### Parallel Team Strategy

With two developers after Phase 2 is complete:
- Developer A: Phase 3 (US1) + Phase 7 (Edge Cases)
- Developer B: Phases 4 + 5 + 6 (US2, US3, US4)
- Both merge into Phase 8 (Polish + CI)

---

## Notes

- `[P]` tasks touch different files — safe to implement concurrently
- All E2E test functions MUST be `async def` (pytest-asyncio mode=auto)
- All fixtures that create Firestore/GCS resources MUST register IDs for cleanup
- Never hardcode the retry count — always use `retry_limit` fixture
- `test/reports/` is kept (not gitignored) so CI can upload as artifact
- Run with `--timeout=300` in CI to enforce the 5-minute wall-clock limit
