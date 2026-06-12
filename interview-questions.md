# Rio Kutty — Interview Questions
### From Basic to Advanced · Python · AI · Prompt Engineering · RAG · Agents · LangGraph

---

## SECTION 1 — PYTHON FUNDAMENTALS

### Basic

**Q1. What is the difference between `async def` and `def` in Python?**
> `async def` defines a coroutine function that can use `await` to pause execution without blocking the thread. `def` is synchronous — it blocks until complete. In this project, all agent and workflow functions are `async` so the event loop can run many LLM calls concurrently.

**Q2. What is `asyncio.gather()` and why did you use it?**
> `gather()` runs multiple coroutines concurrently in the same event loop and collects their results. Used in the master workflow to run WF3 (image), WF4 (audio), WF5 (activities) in parallel. Also used in `TopicsCreatorAgent` to generate all religion/lifestyle topics simultaneously.
> `return_exceptions=True` means one failure doesn't cancel the others.

**Q3. What is a `Semaphore` and why was it needed?**
> A semaphore limits the number of coroutines running concurrently. `asyncio.Semaphore(10)` ensures at most 10 story pipelines run at the same time — prevents hammering the Gemini API and exceeding rate limits.

**Q4. Explain `@lru_cache` in `get_settings()`.**
> `@lru_cache()` makes `get_settings()` a singleton — the `Settings` object is created once and reused. This avoids re-reading `.env` and re-validating Pydantic settings on every call. Important for performance since settings are accessed in every workflow node.

**Q5. What is `TypedDict` and how is it used in this project?**
> `TypedDict` defines a dict with typed keys. Used in `src/models/state.py` to define workflow states like `MasterWorkflowState`, `StoryTopicsState` etc. LangGraph reads these to know what keys each workflow carries and how to merge state updates between nodes.

**Q6. What is the difference between `pickle` and `json` serialisation?**
> JSON is human-readable, language-agnostic, but only handles basic types. `pickle` is Python-specific, handles arbitrary objects (class instances, custom types), but is not secure with untrusted data. The `FirestoreCheckpointer` uses `pickle` + base64 to store LangGraph checkpoint objects (which contain Python-specific types) in Firestore.

---

### Intermediate

**Q7. How does Python's `str.format()` differ from regex-based substitution, and why did you replace it?**
> `str.format()` scans the entire string for `{...}` patterns — including multi-line JSON objects in prompt templates. A JSON example block like `{\n  "story": "..."\n}` causes `KeyError: '\n  "story"'` because Python treats the multi-line brace as a format field.
> We replaced it with `_safe_format()` using `re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", ...)` — only substitutes `{single_word}` identifiers, leaves JSON braces untouched.

**Q8. What is Pydantic `BaseSettings` and how does it differ from a plain dataclass?**
> `BaseSettings` reads values from environment variables and `.env` files automatically, with type coercion and validation. A plain dataclass has no env-reading capability. In this project, `Settings` in `config.py` gets all model names, API keys, limits from env — zero manual `os.getenv()` calls.

**Q9. How does the `FirestoreCheckpointer` use the lazy property pattern?**
> The `client` property creates `firestore.AsyncClient` only on first access:
> ```python
> @property
> def client(self):
>     if self._client is None:
>         self._client = firestore.AsyncClient(...)
>     return self._client
> ```
> This avoids creating a DB connection at import time (which would fail in test environments or before env vars are loaded).

**Q10. Explain `return_exceptions=True` in `asyncio.gather()`.**
> Without it, the first exception cancels all remaining coroutines and raises immediately. With `return_exceptions=True`, exceptions are returned as values in the results list. This lets the master workflow inspect which of WF3/WF4/WF5 failed, report them all, and route to HITL — instead of crashing the whole batch on one failure.

**Q11. What are relative imports (e.g. `from ...services`) and why are they needed here?**
> Relative imports navigate the package hierarchy without hardcoding the package name. `...services` means "go up 3 levels from current package". Used because activity agents are at `src/agents/activities/mcq_agent.py` — three levels deep — and need to import `src/services/ai_service.py`. If the package is renamed, relative imports still work.

---

### Advanced

**Q12. Why does the `PromptRegistry` use a module-level singleton (`_registry`) instead of creating an instance per request?**
> Prompt files are small and static. Loading them from disk on every request would be wasteful I/O. The singleton loads the registry once; subsequent calls just read from the in-memory object. The `lru_cache` pattern on `get_settings()` is the same idea.

**Q13. What is a `circular import` and how was it avoided in the batch workflow?**
> A circular import happens when module A imports from B, and B imports from A — Python can't resolve the order. In `story_topics_workflow.py`, the batch node needs `story_creator_workflow` and `master_workflow`, but those modules also import from `story_topics_workflow` indirectly. Solved with **local imports inside the function**:
> ```python
> async def batch_create_stories_node(...):
>     from ..workflows.story_creator_workflow import story_creator_workflow
>     from ..workflows.master_workflow import master_workflow
> ```
> The import runs at call time, not at module load time, breaking the cycle.

---

## SECTION 2 — FASTAPI & API DESIGN

**Q14. What is `BackgroundTasks` in FastAPI and why use it?**
> `BackgroundTasks` runs a function after the HTTP response has been sent. The API returns `202 Accepted` immediately, and the actual AI workflow runs asynchronously. This is essential here — generating a story takes 30–120 seconds; the client would time out if it had to wait synchronously.

**Q15. What is the difference between `202 Accepted` and `200 OK`?**
> `200 OK` means the request was completed. `202 Accepted` means the request was received and will be processed — but isn't done yet. All write endpoints (`/generate-topics`, `/generate-media`, etc.) return `202` because the work happens in the background.

**Q16. How does the Go API layer complement the Python FastAPI backend?**
> Go handles high-concurrency reads (1000s of users loading story libraries) with very low memory per goroutine. Python handles AI-heavy, CPU/IO-bound background jobs. The split means: Go serves the data, Python generates the data. Each language does what it's best at.

---

## SECTION 3 — AI & LLM BASICS

**Q17. What is a language model (LLM) and what does "temperature" control?**
> An LLM predicts the next token given a context. Temperature controls randomness: 0 = deterministic (always picks highest probability token), 1+ = more creative/random. For children's stories, a moderate temperature (0.7–0.9) produces varied, creative output without going off-topic.

**Q18. What is prompt engineering?**
> Crafting the input text (prompt) to guide the LLM toward the desired output format, style, and content. In this project: SYSTEM role sets the persona (e.g. "wise grandparent"), PROMPT section gives the specific task, examples show the exact JSON output structure expected.

**Q19. What is the difference between a system prompt and a user prompt?**
> System prompt: sets persistent instructions and persona for the model's entire response. User prompt: the specific task for this turn. Models treat system instructions with higher authority. We use `SYSTEM:` sections in `.txt` files to separate them clearly.

**Q20. Why does the LLM sometimes return JSON inside triple backticks (` ```json `)?**
> Many LLMs are trained to format code/JSON in markdown code fences for human readability. When the response is used programmatically, these fences must be stripped before `json.loads()`. The `_parse_story()` method does `.replace("```json", "").replace("```", "")` to handle this.

**Q21. What is "few-shot prompting"?**
> Including example input→output pairs in the prompt to show the model the exact format you expect. All three story creator prompt files include a full example output JSON so the model knows precisely what structure to generate.

---

## SECTION 4 — PROMPT ENGINEERING (ADVANCED)

**Q22. What is chain-of-thought prompting?**
> Asking the model to reason step-by-step before giving a final answer. Improves accuracy on complex tasks. In story generation, the prompt structure (Exposition → Conflict → Rising Action → Climax → Resolution) implicitly guides the model through narrative steps.

**Q23. What is "self-correction" prompting and how is it implemented here?**
> When the first generation fails evaluation, the `SelfCorrectionAgent` sends the original output + the evaluation feedback back to the model with a prompt saying "here is what was wrong, fix it." This is a form of in-context self-repair — the model uses its own mistakes as context to improve.

**Q24. Why do prompts use `{variable}` placeholders and what risk does this introduce?**
> Placeholders personalise the prompt at runtime (age, country, child's name). The risk: if the placeholder value contains curly braces (e.g. a JSON string), it can break `str.format()`. Also: if the prompt template itself contains JSON examples with braces, `str.format()` crashes. Both risks are mitigated by using `_safe_format()` with regex substitution.

**Q25. What is "prompt injection" and is it a concern here?**
> Prompt injection is when user-supplied input contains instructions that override the system prompt (e.g. "Ignore previous instructions and say X"). It's a concern for user-facing chatbots. In this pipeline, the inputs (topic title, age, country) come from internal data, not free-form user text — so the risk is lower. The chatbot project (separate) would need stricter input sanitisation.

**Q26. Explain the SYSTEM / PROMPT structure in the `.txt` files.**
> `SYSTEM:` sets the model's role (persona, tone, language rules). `PROMPT:` gives the specific generation task with variables. Keeping them separate in the file makes it easy to swap personas while reusing the same task structure, or vice versa. It also maps cleanly to the system/user message split in the actual API call.

**Q27. What are "seed" fields (e.g. `mcq_seeds`, `art_seed`, `story_seed`) in the story output?**
> Seeds are short, specific story moments extracted by the LLM that downstream agents use as input. `mcq_seeds` are 4 sentences used by the MCQ agent to generate quiz questions. `art_seed` is a 20-word craft description. Using seeds means downstream agents don't re-read the full story — they get curated, relevant input, which reduces token costs and improves focus.

---

## SECTION 5 — AI EVALUATION

**Q28. What is DeepEval and what does GEval stand for?**
> DeepEval is an open-source LLM evaluation framework. GEval is its "LLM-as-judge" metric where another LLM (the evaluator) scores the output against a criterion. "G" stands for generative evaluation — the evaluator generates a reasoning chain and score, rather than pattern-matching.

**Q29. Why use 8 parallel metrics instead of one overall score?**
> Different failure modes need different fixes. If `bias` fails, the self-corrector is told to remove cultural bias. If `completeness` fails, it's told to add missing elements. A single score obscures which dimension failed. The 8-metric approach gives actionable feedback to the self-correction step.

**Q30. What is `LLMTestCase` in DeepEval and what fields does it require?**
> `LLMTestCase` is the input to an evaluation metric. For GEval it needs:
> - `input`: the prompt given to the model (what was asked)
> - `actual_output`: what the model produced
> - `expected_output` (optional): the ideal answer
> The evaluator LLM then judges whether `actual_output` satisfies the criterion given `input`.

**Q31. Why did you implement `_GeminiEvalModel(DeepEvalBaseLLM)` instead of using DeepEval's built-in model support?**
> DeepEval defaults to OpenAI. It treats any string `model=` as an OpenAI model name and requires `OPENAI_API_KEY`. Since this project uses Gemini exclusively, a custom `DeepEvalBaseLLM` subclass wraps `google.genai.Client`. DeepEval calls `.generate()` / `.a_generate()` on our class, which routes to Gemini internally.

**Q32. Why use `await metric.a_measure()` instead of `run_in_executor(metric.measure)`?**
> `run_in_executor` runs `metric.measure` in a thread pool. DeepEval's `async_mode=True` (default) schedules internal coroutines on the *main event loop* from outside that thread — causing "Future attached to a different event loop" error. `await metric.a_measure()` runs the async path natively on the current loop, no thread involved.

---

## SECTION 6 — RAG (RETRIEVAL-AUGMENTED GENERATION)

**Q33. What is RAG and why isn't it used in this project?**
> RAG augments LLM prompts with retrieved documents from a knowledge base (vector DB). This project doesn't need RAG because:
> 1. Topic libraries (PlanetProtector, MindfullTopics, ChillStories) are small, structured JSON files — loaded directly, not searched
> 2. Story generation is creative, not factual Q&A — RAG would constrain creativity
> 3. The "cache" pattern (Firestore title library) serves a similar purpose to RAG for topic reuse, without the complexity

**Q34. Where would RAG be a good addition to this project in future?**
> For the **chatbot story creator** (separate project): if a child says "tell me about dinosaurs", RAG could retrieve verified paleontology facts from a curated knowledge base, grounding the story in accurate science. Also useful for a **parent dashboard** to answer questions about child development milestones, pulling from expert-curated articles.

**Q35. What is a vector database and how does it differ from Firestore?**
> A vector database (Pinecone, Weaviate, pgvector) stores embeddings (numerical representations of text) and supports semantic similarity search — "find text similar to this query." Firestore is a document database supporting exact-match and range queries, not semantic search. For RAG, you'd use a vector DB; for structured data like stories and activities, Firestore is the right choice.

**Q36. What is an "embedding" and why is it used in RAG?**
> An embedding is a fixed-length vector (e.g. 768 numbers) that captures the semantic meaning of text. Two pieces of text with similar meaning have similar vectors (close in cosine distance). RAG uses embeddings to retrieve the most semantically relevant documents for a query, not just keyword matches.

---

## SECTION 7 — AI AGENTS

**Q37. What is an AI agent vs a simple LLM call?**
> A simple LLM call is stateless: prompt in → text out. An agent has a loop: it can decide to call tools, observe results, then decide next steps — repeating until the task is done. In this project, the agents (`StoryCreatorAgent`, `TopicsCreatorAgent`) are "simple" — one LLM call per invocation. The "agentic" behaviour comes from the LangGraph workflow orchestrating retries, validation, and self-correction across multiple nodes.

**Q38. What is the difference between a tool-using agent and a workflow agent?**
> A **tool-using agent** (like ReAct, OpenAI function calling) dynamically decides at runtime which tool to call and how many times. A **workflow agent** (like LangGraph) has a pre-defined graph of steps with conditional routing. This project uses the workflow pattern — predictable, auditable, easier to test and debug for production content generation.

**Q39. Why is a LangGraph `StateGraph` better than a simple retry loop for this use case?**
> A `StateGraph` provides:
> - **Persistent state**: every node update is checkpointed to Firestore — survives crashes
> - **Resume**: a failed workflow can restart from the last checkpoint, not from scratch
> - **HITL**: `interrupt()` pauses execution, preserves state, and resumes on admin decision
> - **Conditional routing**: clean branching logic (pass → save, fail → correct → retry)
> A `while retry < MAX` loop has none of these properties.

**Q40. What is LangGraph's `interrupt()` and how does it implement human-in-the-loop?**
> `interrupt(payload)` suspends the graph mid-execution. The payload (failed workflows, instructions) is stored in the checkpoint. The graph is frozen — it won't proceed until resumed. An admin calls `graph.ainvoke(None, config, command=Command(resume=decision))` which provides the "human decision" value that `interrupt()` returns. The graph then continues from exactly where it paused.

**Q41. What is a "compiled subgraph" in LangGraph and why use it?**
> A compiled subgraph is a `.compile(checkpointer=...)` LangGraph graph that can be called like a function (`await subgraph.ainvoke(...)`). Each subgraph (WF3, WF4, WF5) manages its own state, retries, and checkpoints under its own `thread_id`. The master workflow doesn't need to model their internal state — it just calls them, collects results, and handles failures. Clean encapsulation.

**Q42. How does `config.configurable` differ from workflow state?**
> `config.configurable` carries **read-only inputs** that don't change during the workflow (story_id, age, language, theme). Workflow state carries **mutable data** that nodes produce and consume (story text, evaluation result, retry count). Separating them prevents nodes from accidentally overwriting input parameters and keeps the state schema clean.

---

## SECTION 8 — LANGGRAPH & WORKFLOW ORCHESTRATION

**Q43. Explain the `merge_dicts` reducer in LangGraph state.**
> In LangGraph, when multiple nodes update the same state key, a reducer decides how to merge the updates. `merge_dicts` (Annotated type) deep-merges dictionaries instead of replacing. Used for `errors` dict — if WF3 adds `{"wf3": "error"}` and WF4 adds `{"wf4": "error"}`, merge_dicts produces `{"wf3": "error", "wf4": "error"}` rather than the second overwriting the first.

**Q44. What does `set_entry_point()` do vs `add_edge()`?**
> `set_entry_point("node_name")` specifies which node runs first when the graph is invoked. `add_edge("A", "B")` creates an unconditional transition from A to B after A completes. `add_conditional_edges("A", routing_fn, {...})` routes to different nodes based on the return value of `routing_fn`.

**Q45. What is a `CheckpointTuple` and what does it contain?**
> A `CheckpointTuple` is what LangGraph's checkpointer returns. It contains:
> - `config`: the thread config (thread_id, checkpoint_id) for resuming
> - `checkpoint`: the serialised graph state at that point
> - `metadata`: step number, source, writes
> - `parent_config`: the previous checkpoint's config (for history traversal)
> The `FirestoreCheckpointer.aget_tuple()` reconstructs this from a Firestore document.

**Q46. Why does `aget_tuple()` query by `thread_id` + `order_by created_at DESC`?**
> LangGraph calls `aget_tuple()` to get the **latest checkpoint** for a thread when resuming. The thread may have multiple checkpoints (one per node). Ordering by `created_at DESC` and taking `limit(1)` gives the most recent — the point where the graph was last paused. Without the composite index, Firestore can't do this query efficiently.

---

## SECTION 9 — GOOGLE CLOUD & INFRASTRUCTURE

**Q47. What is Pub/Sub and how is it used for HITL notifications?**
> Google Cloud Pub/Sub is a managed message queue. Publishers send messages to a topic; subscribers receive them. When a workflow fails after 4 retries, `_publish_hitl_notification()` publishes a JSON message with the failed workflow details. The admin's monitoring system (subscriber) receives it and alerts them to review. Decoupled — the workflow doesn't know or care who reads the notification.

**Q48. Why use Firestore over a relational database (PostgreSQL)?**
> - **Schemaless**: story structure evolves without migrations
> - **Real-time**: Go API can subscribe to story changes and push updates to mobile
> - **Globally distributed**: low latency reads from any region
> - **Document-based**: a story (with all its fields) is one document — one read, no joins
> Relational DBs excel at complex queries and transactions across tables — not needed here.

**Q49. What is a service account and why is it used instead of user credentials?**
> A service account is a non-human identity for applications. It has specific IAM roles (Firestore read/write, GCS admin) without having any user-level access. Using `GOOGLE_APPLICATION_CREDENTIALS` pointing to the service account JSON means the app authenticates automatically — no OAuth flow, no user login required for background jobs.

**Q50. What is Google Cloud Storage and how do images/audio URLs work?**
> GCS is an object store (like S3). `StorageBucketService.upload_file()` uploads bytes and returns the public URL: `https://storage.googleapis.com/{bucket}/{filename}`. This URL is stored in the Firestore story document as `image_url`/`audio_url`. The mobile app loads images/audio directly from GCS CDN — the Python backend is not in the read path.

---

## SECTION 10 — SYSTEM DESIGN & ARCHITECTURE

**Q51. Why does the Go layer exist when Python already has FastAPI?**
> Python with asyncio handles concurrent I/O well but still has the GIL limiting CPU parallelism, and per-coroutine overhead. Go goroutines are extremely lightweight (~2KB each vs ~64KB thread). For serving thousands of simultaneous story reads from Firestore, Go is 5–10x more efficient. Python is reserved for AI-heavy background jobs where the bottleneck is LLM latency, not goroutine overhead.

**Q52. What is the "cache-first" pattern used in `TopicsCreatorAgent`?**
> Before calling the LLM, check Firestore for an existing entry (cache hit). On miss: call LLM, save result to Firestore, return. On subsequent calls with the same (theme, age, language, filter_value): return cached result instantly with no LLM call. This makes re-generation free and reduces API costs dramatically once topics are populated.

**Q53. What is the "fan-out / fan-in" pattern in `dispatch_parallel_node`?**
> Fan-out: one task splits into multiple concurrent subtasks (master → WF3, WF4, WF5 via `asyncio.gather`). Fan-in: wait for all to complete and collect results atomically. `collect_results_node` is the fan-in point — it checks all statuses and either finalises or triggers HITL.

**Q54. How would you scale this system to 1 million users?**
> - **Go API**: horizontally scalable, add instances behind load balancer
> - **Python AI pipeline**: run as Cloud Run jobs (auto-scale to zero, scale up on demand)
> - **Firestore**: natively scales to millions of reads/writes per second
> - **GCS**: global CDN, scales infinitely for static asset delivery
> - **LLM rate limits**: increase `MAX_CONCURRENCY`, add exponential backoff, use multiple API keys
> - **Cache**: topic library covers most users — O(users) reads, O(themes × filters) LLM calls

**Q55. What is a circuit breaker pattern and why is it used?**
> A circuit breaker tracks failure rate for a dependency (e.g. Gemini API). If failures exceed a threshold, it "opens" — subsequent calls fail immediately without trying, giving the dependency time to recover. After a timeout, it "half-opens" — tries one request; if it succeeds, closes again. Prevents cascade failures: one slow/down API doesn't exhaust your retry budget and slow everything else.

---

## SECTION 11 — EDGE CASES & DEBUGGING

**Q56. The LLM sometimes returns JSON, sometimes pipe-separated text. How do you handle this?**
> `_parse_pipe_response()` in `TopicsCreatorAgent` tries JSON first (with code-fence stripping via regex), falls through to pipe-separated parsing if JSON fails. This handles model non-determinism: the same prompt may return different formats on different runs.

**Q57. What happens if a story is generated but the image workflow fails permanently?**
> After 4 retries, `image_workflow` returns `status: "needs_human"`. The master workflow's `collect_results_node` detects this, publishes a Pub/Sub notification, and calls `interrupt()`. The story is already saved to Firestore (WF2 completed). Admin can resume with `"skip"` (story published without image) or `"override"` (admin manually uploads an image).

**Q58. What if the Firestore composite index doesn't exist in production?**
> The `FirestoreCheckpointer` catches the index-missing error, logs a warning, and falls back to fetching all documents for that `thread_id` and sorting client-side. This is correct but slower — O(n) reads instead of O(1). The fallback ensures the system still works in dev; the composite index should be created before production load.

**Q59. Why is `story_id` used as the Firestore document ID instead of an auto-generated ID?**
> Using `story_id` as the document ID enables O(1) direct lookups: `collection.document(story_id).get()`. Auto-generated IDs would require a query (`where story_id == X`) which needs an index and is slower. Since `story_id` is a UUID generated once per story, it's guaranteed unique and collision-free.

**Q60. How does the self-correction loop prevent infinite retries?**
> `MAX_CORRECTION_ATTEMPTS = 2` in each workflow. The state tracks `correction_attempts`. The router `route_after_evaluate()` checks:
> ```python
> if attempts < MAX_CORRECTION_ATTEMPTS:
>     return "self_correct"
> return END  # fail gracefully
> ```
> After 2 failed corrections, the workflow ends with an error state rather than looping forever. The master workflow then sees `status: "needs_human"` and escalates to HITL.

---

## SECTION 12 — BONUS — SYSTEM THINKING

**Q61. How would you add a new language (e.g. Hindi) to this system?**
> 1. Add `"Hindi": "hi"` to `_LANG_CODE` maps in agents
> 2. Create prompt files: `src/prompts/story_creator/theme1/v1_hi.txt` etc.
> 3. Create topic prompt files: `src/prompts/story_topics/theme1/v1_hi.txt`
> 4. Test TTS: check if Google TTS has a Hindi voice, add to `TTS_LANGUAGE_CODE`
> 5. Add `hi` to the `_LANG_CODE_MAP` in `story_topics_workflow.py`
> No code changes needed in workflow logic — the prompt loading is language-agnostic by design.

**Q62. How would you A/B test two versions of the story prompt?**
> 1. Create `v2_en.txt` alongside `v1_en.txt` in the prompt directory
> 2. Add a `STORY_CREATOR_PROMPT_VERSION` env var (already in config)
> 3. Route 50% of requests to v1, 50% to v2 by passing version in config
> 4. Tag Firestore story docs with `prompt_version`
> 5. Analyse evaluation scores, engagement metrics by version
> 6. Promote winning version by updating the env var

**Q63. What is the biggest technical risk in this architecture?**
> **LLM API rate limits and latency variance.** 45 parallel stories × 3 LLM calls each = 135 concurrent API calls. Gemini's rate limits could throttle or fail some. Mitigation: semaphore limits concurrency, circuit breaker prevents flood retries, exponential backoff spaces retries. Long-term: caching reduces repeat calls; pre-generation during off-peak hours avoids real-time pressure.

**Q64. How would you add Telugu support for the chatbot project?**
> The chatbot is a separate Python project but likely uses the same `AIService` and `PromptRegistry` patterns. Steps:
> 1. Telugu prompt files (story + evaluation in Telugu)
> 2. Telugu TTS voice in Google Cloud TTS (`te-IN-Standard-A`)
> 3. Input normalisation (Telugu Unicode characters)
> 4. Test evaluation: DeepEval metrics may need Telugu-capable judge model
> The `_LANG_CODE` map and prompt versioning system make this straightforward.

---

*Interview prep document generated: March 2026*
*Covers: Python async · FastAPI · Prompt Engineering · DeepEval · RAG · LangGraph Agents · Google Cloud · System Design*
