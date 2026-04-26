# Rio Kutty Story Management Constitution

## Core Principles

### I. Workflow-First Architecture
Every feature belongs to a clearly scoped workflow (WF1–WF5 or Master). Each workflow is an independently compiled LangGraph state machine with its own state TypedDict, nodes, edges, and checkpointer. No cross-workflow state mutation. Workflows communicate only through Firestore (persisted output) or explicit state handoff at API boundaries.

### II. Agent Responsibility Boundary
Each agent does exactly one thing: load a versioned prompt → call AIService → parse JSON → return a partial state update. Agents never call Firestore directly. Agents never own retry logic — that belongs to the workflow graph. One agent = one LLM call.

### III. Versioned Prompts (NON-NEGOTIABLE)
All prompts live in `src/prompts/{agent}/{theme}/v{N}.txt` or `src/prompts/{agent}/v{N}.txt`. Prompts are never hardcoded in Python. The `PromptRegistry` resolves version via config. When changing prompt behavior, create a new version file — never edit existing ones in production.

### IV. Resilience at the Service Layer
External API calls (Gemini, FLUX, TTS, Firestore, GCS) are protected by circuit breaker + retry + rate limiter in `src/utils/resilience.py`. Retry logic lives in workflows (max 4 for WF3/WF4/WF5; max 2 evaluation loops for WF1/WF2). Agents and services do not implement their own retry — they raise, the graph catches.

### V. Human-in-Loop on Parallel Workflow Failure
When WF3 (image), WF4 (audio), or WF5 (activities) exhausts retries, the Master workflow calls `interrupt()`, pauses the graph in Firestore checkpoint state, and sends a Pub/Sub notification. No silent failure. Admin reviews via `POST /resume-workflow` with decision: `retry | skip | override`. This path must remain exercisable in production at all times.

### VI. Test-First for Agents and Validators
Unit tests are written before or alongside each new agent and validator. Tests mock `AIService` at the call site (`src.agents.<module>.AIService`). Tests must not require real API keys — use `GOOGLE_API_KEY=test_key HF_TOKEN=test_token`. Integration tests (tests/integration/) may hit real services but must be explicitly opted into.

### VII. Cost-Conscious Model Selection
- WF1 (topics), WF5 (activities), evaluation: `gemini-2.0-flash-lite` (cheapest)
- WF2 (story creation): `gemini-2.5-flash-lite` (quality matters)
- Image: `FLUX.1-schnell` (4 steps, ~10× cheaper than FLUX.1-dev)
- Audio: Google Cloud TTS Standard voices (not WaveNet/Chirp unless quality justified)
- Never use `gemini-2.0-flash` (no suffix) — it returns 404. Use `-lite` or `-001` suffixed variants.

### VIII. Idempotent Pipelines
Every workflow node checks before writing. `check_if_activity_exists()` before generating activities. `get_story_by_title()` before creating stories. `get_library_topics()` before calling LLM for topics. Duplicate runs must be safe. Checkpoints are deleted only after successful pipeline completion (`delete_workflow_checkpoints()`).

### IX. Async-First, No Blocking
All service calls, LLM invocations, and database operations are `async/await`. Background work is dispatched via FastAPI `BackgroundTasks`. API endpoints return `202 Accepted` immediately — never block on AI generation. Parallel tasks (WF3+WF4, all four activities in WF5) use `asyncio.gather()`.

### X. Observability is Optional but Structured
Langfuse tracing is always wired in but defaults to disabled (`LANGFUSE_ENABLED=false`). Structured logging via `setup_logger(__name__)` is mandatory in every module. No `print()` statements in production code. Tracing callbacks are passed through to LangGraph nodes — never hardcoded inside agents.

---

## Constraints

### Data & Storage
- Stories, topics, activities live in theme-specific Firestore collections (`planet_protectors_*`, `mindful_*`, `chill_stories_*`)
- Story doc ID = `story_id` (UUID); activities tagged with `story_id` in `activities_v1`
- Media (images, audio) stored in GCS at `gs://kutty_bucket/` — URLs saved back to story doc
- `workflow_checkpoints` (LangGraph) are ephemeral — deleted post-finalization; never used as a data store

### API Design
- All generation endpoints return `202 Accepted` with no body (background processing)
- Pub/Sub endpoints return `204 No Content` (immediate ACK to Cloud Pub/Sub)
- Status polling via `GET /workflow-status/{story_id}`
- No streaming endpoints — Go client polls Firestore

### Prompt Authorship
- Engineers write placeholder prompt files only
- Product/content owner writes all actual prompt content
- Prompts use `{identifier}` template variables only — `PromptRegistry._safe_format()` handles substitution safely (no `str.format()` — prevents KeyError on JSON examples inside prompts)

### Configuration
- All config via Pydantic `BaseSettings` in `src/utils/config.py` (singleton via `lru_cache`)
- No hardcoded model names, collection names, or bucket names outside config
- `USE_MEMORY_CHECKPOINTER=true` for local dev/testing; Firestore checkpointer in production

---

## Development Workflow

### Adding a New Workflow
1. Define state TypedDict in `src/models/state.py` with Annotated reducers
2. Create agents in `src/agents/{domain}/`
3. Create workflow in `src/workflows/` (compile with appropriate checkpointer)
4. Add API endpoint in `src/api/`
5. Add versioned prompt files (placeholder content)
6. Write unit tests for agents before wiring workflow

### Adding a New Agent
1. Create file in `src/agents/{domain}/{name}_agent.py`
2. Load versioned prompt via `PromptRegistry`
3. Call `AIService.generate_content(model_override=X)` — never instantiate Gemini directly
4. Parse JSON, validate required fields, return partial state dict
5. Write unit test mocking `AIService` at `src.agents.{domain}.{name}_agent.AIService`

### Changing a Prompt
1. Create new version file: `src/prompts/{agent}/v{N+1}.txt`
2. Update version env var or config default
3. Never modify existing version files that have been used in production

### Firestore Collection Changes
- New collections require updates to `firestore_service.py` (get/save methods)
- Collection names are constants in `firestore_service.py` — never scattered across the codebase
- Schema changes must be backward compatible (add fields, never remove)

---

## Governance

This constitution supersedes all other development practices for this project. Deviations require explicit documented justification.

**All PRs must verify:**
- No hardcoded model names, prompts, or collection names
- New agents have unit tests
- New external calls are wrapped with resilience patterns
- No blocking I/O in async context

**Version**: 1.0.0 | **Ratified**: 2026-04-26 | **Last Amended**: 2026-04-26
