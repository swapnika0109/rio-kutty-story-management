# Feature Specification: End-to-End Test Suite

**Feature Branch**: `001-e2e-test-suite`  
**Created**: 2026-04-26  
**Status**: Draft  
**Input**: User description: "Implement test cases that can run end to end test for all cases including edge cases"

## Clarifications

### Session 2026-04-26

- Q: What is the maximum allowed wall-clock time for the full end-to-end pipeline test in CI? → A: 5 minutes
- Q: How should the test environment isolate Firestore data from production? → A: Same GCP project, separate named Firestore database (e.g., `rio-test`), same collection names as production
- Q: What is the minimum DeepEval GEval score that constitutes a passing activity? → A: 0.7 (on a 0.0–1.0 scale)
- Q: What format should the test suite use to report workflow pass/fail results? → A: Structured JSON report file per run (machine-readable, CI-compatible)
- Q: Is the retry count always exactly 4? → A: 4 is the default but configurable per workflow
- Q: Where should generated images and audio files be stored during tests? → A: Separate test folders (`test/images/` and `test/audio/`), distinct from production storage paths

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Full Happy Path Pipeline (Priority: P1)

A developer triggers the complete story generation pipeline from topic selection through to a fully assembled story with image, audio, and activities — and verifies that all output is persisted correctly in Firestore.

**Why this priority**: This is the core product flow. All other workflows depend on this path working correctly. Verifying it end-to-end gives the highest confidence in system correctness.

**Independent Test**: Can be fully tested by submitting a valid topic request, confirming topic list is returned, selecting a topic, and verifying that a story document with `title`, `description`, `image_url`, `audio_url`, and linked activities appears in the appropriate collection of the `rio-test` named Firestore database.

**Acceptance Scenarios**:

1. **Given** the pipeline is idle, **When** a valid theme (e.g., `planet_protectors`) and user preferences are submitted to WF1, **Then** a list of unique, non-duplicate story topics is returned within 5 minutes total pipeline time.
2. **Given** a topic list is returned, **When** the human selects one topic, **Then** WF2 produces a complete story (title, description, moral) in the correct format and persists it to the `rio-test` Firestore database with the `topics_id` as the document ID.
3. **Given** a story is saved to Firestore, **When** WF3 and WF4 run in parallel, **Then** both `image_url` and `audio_url` are populated on the story document.
4. **Given** image and audio generation succeed, **When** WF5 runs, **Then** MCQ, art, science, and moral activities are generated and saved to `activities_v1` tagged with the story ID; art, science, and moral activities also have an `activity_image_url` stored in the `test/images/` folder.

---

### User Story 2 - Topic Deduplication and Caching (Priority: P2)

A developer verifies that the topics workflow correctly avoids generating duplicate titles across calls for the same theme, and that topics are cached in Firestore.

**Why this priority**: Duplicate topics degrade content quality and waste API quota. This ensures cache correctness and generation uniqueness.

**Independent Test**: Can be tested by calling WF1 twice for the same theme against the `rio-test` database and confirming no title appears in both responses, and that the second call reads from the topics cache collection.

**Acceptance Scenarios**:

1. **Given** topics already exist in the `{theme}_topics` collection, **When** WF1 is called again for the same theme, **Then** previously generated titles are excluded from new suggestions.
2. **Given** a topic generation call succeeds, **When** inspecting the `{theme}_topics` collection in the `rio-test` database, **Then** the new topics document is present with a valid `topics_id` and a non-empty `topics` array.

---

### User Story 3 - Retry and Human-in-the-Loop on Failure (Priority: P2)

A developer simulates repeated failures in WF3 (image) or WF4 (audio) and verifies the pipeline retries up to the configured limit (default: 4), then pauses for human intervention via Pub/Sub notification and a resume API call.

**Why this priority**: Resilience is critical — silent failures would result in incomplete stories reaching end users.

**Independent Test**: Can be tested by mocking the image or audio service to always fail, confirming the retry counter reaches the configured limit, a Pub/Sub message is published, and the workflow enters an interrupted state. Then POST to `/resume-workflow` and confirm the pipeline either resumes or marks the story failed cleanly.

**Acceptance Scenarios**:

1. **Given** the image generator fails on every attempt, **When** WF3 has exhausted the configured retry limit (default: 4), **Then** a Pub/Sub notification is published and the workflow suspends at an interrupt checkpoint.
2. **Given** the workflow is suspended, **When** a resume decision of `skip` is POSTed to `/resume-workflow`, **Then** the pipeline continues without an image URL and completes the remaining steps.
3. **Given** the workflow is suspended, **When** a resume decision of `retry` is POSTed to `/resume-workflow`, **Then** WF3 attempts image generation one more time.

---

### User Story 4 - Activity Generation Quality (Priority: P3)

A developer verifies that DeepEval GEval evaluation scores meet the minimum threshold of 0.7 for generated activities (MCQ, art, science, moral).

**Why this priority**: Activity quality directly affects the child's learning experience. DeepEval provides an objective quality gate.

**Independent Test**: Can be tested by running WF5 against a known story and asserting all GEval scores are >= 0.7 on a 0.0–1.0 scale.

**Acceptance Scenarios**:

1. **Given** a valid story, **When** WF5 generates all activity types, **Then** each activity passes DeepEval GEval evaluation with a score >= 0.7.
2. **Given** an activity fails GEval (score < 0.7), **When** WF5 retries up to the configured limit (default: 4), **Then** either a passing activity (score >= 0.7) is produced or the failure is recorded with a clear reason.

---

### Edge Cases

- What happens when the topic theme is invalid or unsupported (not one of `planet_protectors`, `mindful`, `chill`)?
- What happens when the `rio-test` Firestore database is unavailable during story save?
- What happens when both WF3 and WF4 fail simultaneously after exhausting all retries?
- What happens when the AI model returns malformed JSON that cannot be parsed?
- What happens when the `resume-workflow` API is called with an unknown or already-completed workflow ID?
- What happens when the Pub/Sub publish itself fails during the human-in-the-loop interrupt?
- What happens when WF1 returns zero topics (empty list)?
- What happens when the selected topic already has a story (duplicate story scenario)?
- What happens when story content exceeds expected length limits?
- What happens when audio generation produces a file but upload to storage fails?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The test suite MUST cover WF1 (Topic Generation), WF2 (Story Creator), WF3 (Image Generator), WF4 (Audio Generator), and WF5 (Activities) individually and as a complete pipeline.
- **FR-002**: Each workflow test MUST assert correct Firestore document structure after execution, using the `rio-test` named Firestore database within the same GCP project (same collection names as production).
- **FR-003**: The test suite MUST include tests for all three theme variants: `planet_protectors`, `mindful`, and `chill`.
- **FR-004**: Retry logic MUST be tested by simulating consecutive failures up to the configured retry limit (default: 4) before triggering the human-in-the-loop interrupt; the retry count MUST be configurable per workflow.
- **FR-005**: The human-in-the-loop interrupt MUST be tested — confirming Pub/Sub notification is published and the workflow pauses after the configured retry limit is exhausted.
- **FR-006**: The resume API endpoint MUST be tested with both `skip` and `retry` decisions after a workflow interrupt.
- **FR-007**: The test suite MUST validate that duplicate topic titles are not generated across successive WF1 calls for the same theme.
- **FR-008**: DeepEval GEval evaluation MUST be exercised in WF5 tests, asserting all activity scores are >= 0.7 on a 0.0–1.0 scale.
- **FR-009**: All edge case scenarios (invalid theme, malformed AI response, Firestore unavailable, empty topic list) MUST have dedicated tests.
- **FR-010**: Tests MUST clean up all data in the `rio-test` Firestore database (stories, topics, activities, checkpoints) after each run.
- **FR-013**: Image and audio files generated during tests MUST be stored in dedicated test folders (`test/images/` and `test/audio/`), separate from production storage paths, and cleaned up after each run. This includes both the story image (WF3) and the per-activity images generated by WF5 for art, science, and moral activities.
- **FR-011**: The test suite MUST produce a structured JSON report file per run identifying which workflows passed and failed, with enough detail to identify the failing node.
- **FR-012**: Parallel execution of WF3 and WF4 MUST be validated — confirming both run concurrently and both results are awaited before WF5 starts.

### Key Entities

- **Story**: The primary output document; has `title`, `description`, `image_url`, `audio_url`, `topics_id` (used as doc ID), stored per theme collection in the `rio-test` database.
- **Topic**: A candidate story idea with a title; stored in `{theme}_topics` collection, referenced by `topics_id`.
- **Activity**: MCQ/art/science/moral item linked to a `story_id`; stored in `activities_v1`. Art, science, and moral activities each include an `activity_image_url` (GCS path under `test/images/` during tests); MCQ activities do not generate images.
- **Workflow Checkpoint**: LangGraph state persisted during pipeline execution; deleted on success.
- **Resume Decision**: Human input (`skip` / `retry`) submitted to `/resume-workflow` after a workflow interrupt.
- **Test Report**: Structured JSON file produced per test run; records pass/fail status per workflow node with failure details.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All happy-path workflow tests pass end-to-end without manual intervention in the `rio-test` database environment.
- **SC-002**: Retry logic tests confirm the system retries up to the configured limit (default: 4) before triggering the human-in-the-loop interrupt.
- **SC-003**: DeepEval GEval scores for all activity types are >= 0.7 on at least 90% of test runs.
- **SC-004**: All edge case scenarios produce a clear, deterministic outcome (no silent failures or unhandled exceptions).
- **SC-005**: The `rio-test` Firestore database contains no test-run pollution after the test suite completes (cleanup verified).
- **SC-006**: Parallel WF3+WF4 execution is confirmed by timing — both complete before WF5 begins and total time is less than the sum of sequential execution.
- **SC-007**: The full pipeline test (WF1→WF2→WF3+WF4→WF5) completes within 5 minutes in a CI environment.

## Assumptions

- Tests run against a named Firestore database called `rio-test` within the same GCP project — using the same collection names as production, but fully isolated by database name.
- The AI models (`gemini-2.0-flash-lite`, `gemini-2.5-flash-preview`, FLUX.1-schnell) are accessible during test runs via valid API keys.
- Environment variables `GOOGLE_API_KEY` and `HF_TOKEN` are available in the test environment.
- The Pub/Sub topic for human-in-the-loop notifications exists and is accessible in the test environment.
- The `/resume-workflow` API endpoint is running and reachable during integration tests.
- DeepEval GEval minimum passing score is 0.7 on a 0.0–1.0 scale, pre-configured and consistent across test runs.
- Retry limits default to 4 per workflow but are configurable; tests must respect the configured value, not hardcode 4.
- Test results are written as a structured JSON report file per run for CI consumption.
- Image files generated during tests are stored under `test/images/` and audio files under `test/audio/`, both separate from production storage and deleted after each test run.
- Mobile support and frontend UI testing are out of scope — only backend pipeline and API behavior is tested.
