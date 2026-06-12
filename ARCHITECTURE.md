# Rio Kutty Story Management — Architecture

## Overview

A FastAPI backend that generates children's stories with images, audio, and learning activities using a multi-workflow LLM pipeline. Stories are generated in batches of up to 10 concurrently, with quality evaluation at every step and human-in-the-loop escalation when AI generation fails after 4 retries.

**Stack:** Python 3.11 · FastAPI · LangGraph · Google Gemini · FLUX.1-schnell · Google TTS · Firestore · GCS · Langfuse

---

## System Architecture Diagram

```
Client (Go)
    │
    ├── POST /generate-topics          ──► WF1: Topic Generation
    │                                         │
    │                                         ▼
    │                                  batch_create_stories
    │                               ┌──────────────────────────┐
    │                               │  semaphore(10 max)       │
    │                               │  Per topic:              │
    │                               │  ┌────────────────────┐  │
    │                               │  │ Resume check       │  │
    │                               │  │ story_id in topic? │  │
    │                               │  │  yes → skip/resume │  │
    │                               │  │  no  → run WF2     │  │
    │                               │  └────────────────────┘  │
    │                               └──────────────────────────┘
    │                                         │
    ├── POST /select-topic    ──► WF2         ▼
    │                            Story   Master Workflow (per story)
    │                                    ┌───────────────────────────┐
    │                                    │  Phase 1: parallel        │
    ├── POST /generate-media/{id} ──►    │  WF3 Image ─┐             │
    │                                    │  WF4 Audio ─┴─► collect   │
    │                                    │              (HITL if any │
    │                                    │               fails)      │
    │                                    │                           │
    │                                    │  Phase 2: sequential      │
    │                                    │  WF5 Activities (seeds)   │
    │                                    │     MCQ / Art / Science / │
    │                                    │     Moral                 │
    │                                    │                           │
    │                                    │  Finalize: cleanup        │
    │                                    │  checkpoints              │
    │                                    └───────────────────────────┘
    │
    ├── POST /generate-image/{id}  ──► WF3 standalone (HITL retry)
    ├── POST /generate-audio/{id}  ──► WF4 standalone (HITL retry)
    ├── POST /resume-workflow      ──► Resume interrupted master (skip/override)
    ├── POST /generate-activities  ──► WF5 standalone (backward compat)
    └── GET  /workflow-status/{id} ──► Poll story completion state
```

---

## Workflow Details

### WF1 — Story Topics

**Trigger:** `POST /generate-topics` (background task)
**Model:** `gemini-2.0-flash-lite`

```
generate_topics → validate_topics → evaluate_topics
                                         │
                              pass ──► save_topics → batch_create_stories → END
                              fail ──► correction_attempts < 2?
                                            yes → self_correct_topics → generate_topics
                                            no  → END (error)
```

- Generates topics for all 3 themes in parallel (`asyncio.gather`)
- **Theme 1 (PlanetProtector):** 1 LLM call, filtered by `country`
- **Theme 2 (MindfullTopics):** 1 LLM call per religion, all religions in parallel
- **Theme 3 (ChillStories):** 1 LLM call per lifestyle area, all 7 areas in parallel
- Results cached in Firestore topic collections on first LLM call
- **Partial cache:** if `len(cached) < n` (e.g. scaling 5→10), generates only the missing `n - len(cached)` topics, deduplicates by title, merges and saves combined list
- `batch_create_stories` spawns WF2→Master pipeline for each topic (semaphore=10)
- Evaluation: DeepEval GEval

**Output:** Topics saved to `planet_protectors_topics` / `mindful_topics` / `chill_stories_topics`

#### batch_create_stories Resume Logic

On every run (including restarts), each topic pipeline checks before generating:

```
topic.story_id present?
    │
    ├── yes → fetch story from Firestore
    │          story_text exists?
    │              │
    │              ├── yes → check what's missing:
    │              │          image_url missing AND/OR audio_url missing → run Master
    │              │          only activities missing → run WF5 directly
    │              │          nothing missing → skip entirely ("Already complete")
    │              │
    │              └── no  → story_id in library but story lost → fresh run
    │
    └── no  → fresh run: WF2 → Master → patch story_id into topic library
```

This means **WF2 (story creator) is never re-run** for a topic that already has a saved story. Only the missing media or activities are generated.

---

### WF2 — Story Creator

**Trigger:** Human picks topic via `POST /select-topic` OR called by WF1 batch
**Model:** `gemini-2.5-flash-lite` (higher quality for narrative)

```
generate_story → validate_story → evaluate_story
                                       │
                            pass ──► save_story → END
                            fail ──► correction_attempts < 2?
                                          yes → self_correct_story → generate_story
                                          no  → END (error)
```

- Loads versioned prompt from `src/prompts/story_creator/{theme}/v{N}_{lang}.txt`
- Story JSON from LLM uses key `"story"` — normalised to `"story_text"` by the agent
- Raw control characters in LLM response (e.g. literal newlines inside JSON strings) are sanitised by `_escape_control_chars()` before JSON parsing
- `title` falls back to `selected_topic["title"]` if LLM omits it
- `age_group` and `language` are injected from runtime state (LLM never returns these)
- Validation: checks `{title, story_text, moral, age_group, language}` are present and non-empty
- Story JSON contains activity seeds: `mcq_seeds`, `art_seed`, `science_concepts`, `moral`, `image_prompt`
- Saves to Firestore with `topics_id` for lineage tracing

**Output:** Story saved to `planet_protectors_stories` / `mindful_stories` / `chill_stories`

---

### Master Workflow — Media + Activities Orchestrator

**Trigger:** `POST /generate-media/{story_id}` OR called automatically by WF1 batch

```
Phase 1 ─ Media (parallel)
  dispatch_media
    ├── WF3 image_workflow.ainvoke(...)  ─┐
    └── WF4 audio_workflow.ainvoke(...)  ─┴─► collect_media
                                                  │
                                    all ok ──► dispatch_activities
                                    fail   ──► HITL interrupt()
                                                  │
                                         admin resumes via POST /resume-workflow
                                                  │
                                    handle_media_decision ──► dispatch_activities

Phase 2 ─ Activities (sequential, after Phase 1)
  dispatch_activities
    └── WF5 activity_workflow.ainvoke(with seeds) ──► collect_activities
                                                            │
                                             ok  ──► finalize
                                             fail ──► HITL interrupt()
                                                            │
                                                  handle_activities_decision ──► finalize

finalize ──► delete workflow_checkpoints ──► END
```

---

### WF3 — Image Generator (compiled subgraph)

**Model:** FLUX.1-schnell via HuggingFace InferenceClient (4 inference steps, ~10× cheaper than FLUX.1-dev)
**Max retries:** 4 before `needs_human`

```
generate_image → validate_image → evaluate_image
                                       │
                            pass ──► save_image → END (status=completed)
                            fail ──► retry_count < 4 → generate_image
                                     retry_count >= 4 → mark_needs_human → END
```

- Uses `image_prompt` field from story JSON (not raw story text)
- Uploads PNG to GCS, saves URL on story doc
- Standalone retrigger: `POST /generate-image/{story_id}`

---

### WF4 — Audio Generator (compiled subgraph)

**Service:** Google Cloud Text-to-Speech
**Max retries:** 4 before `needs_human`

```
generate_audio → validate_audio → evaluate_audio
                                       │
                            pass ──► save_audio → END (status=completed)
                            fail ──► retry_count < 4 → generate_audio
                                     retry_count >= 4 → mark_needs_human → END
```

- Single language per story (from story doc)
- **Voice selection:** random per story from pool — each story gets a different voice
- Uploads MP3 to GCS, saves URL on story doc
- Standalone retrigger: `POST /generate-audio/{story_id}`

#### Voice Selection

Request JSON sends `voice_type = "chirp" | "standard"` (default: `"standard"`).

Full voice name is built as: `{lang_prefix}-{country}{suffix}`
e.g. `en-US-Standard-A` or `en-US-Chirp3-HD-Gacrux`

Language → BCP-47 mapping:

| Language | Prefix | Country |
|---|---|---|
| English / en | `en` | `US` |
| Telugu / te | `te` | `IN` |

**Standard suffixes** (4 voices, random pick per story):
```
-Standard-A  -Standard-B  -Standard-C  -Standard-D
```

**Chirp suffixes** (17 voices, random pick per story):
```
-Chirp3-HD-Gacrux      -Chirp3-HD-Callirrhoe  -Chirp3-HD-Despina
-Chirp3-HD-Iapetus     -Chirp3-HD-Leda         -Chirp3-HD-Zephyr
-Chirp3-HD-Schedar     -Chirp3-HD-Sadaltager   -Chirp3-HD-Rasalgethi
-Chirp3-HD-Umbriel     -Chirp3-HD-Pulcherrima  -Chirp3-HD-Charon
-Chirp3-HD-Zubenelgenubi -Chirp3-HD-Achird     -Chirp3-HD-Algenib
-Chirp3-HD-Algieba     -Chirp3-HD-Erinome
```

`random.choice` is called independently per story so consecutive stories are unlikely to share a voice (probability 1/4 for Standard, 1/17 for Chirp).

---

### WF5 — Activities (compiled subgraph)

**Models:** `gemini-2.0-flash-lite` · `gemini-2.0-flash-lite` (fallback)
**Max retries:** 4 per activity type

```
start ──► route_start (checks DB, skips existing)
    │
    ├── gen_mcq → val_mcq → (retry / save_mcq → END)
    ├── gen_art → val_art → (retry / save_art → END)
    ├── gen_mor → val_mor → (retry / save_mor → END)
    └── gen_sci → val_sci → (retry / save_sci → END)
```

- **MCQ:** uses `mcq_seeds` (key story points list, not full text)
- **Art:** uses `art_seed` (concise visual direction string)
- **Science:** uses `science_concepts` (list of `{concept, explanation}` dicts)
- **Moral:** uses `moral` (moral lesson string)
- Art/Science/Moral activities include image generation (FLUX.1-schnell)
- Images uploaded to GCS; activity records saved to `activities_v1`

---

## Human-in-the-Loop (HITL)

When a subgraph fails after 4 retries, the master workflow:

1. Publishes a Cloud Pub/Sub notification to `HUMAN_LOOP_NOTIFICATION_TOPIC`
2. Calls LangGraph `interrupt()` — graph suspends, state persisted in `workflow_checkpoints`
3. Admin resolves via:

| Option | How |
|--------|-----|
| **Retry image directly** | `POST /generate-image/{story_id}` → then `POST /resume-workflow` with `decision=override` |
| **Retry audio directly** | `POST /generate-audio/{story_id}` → then `POST /resume-workflow` with `decision=override` |
| **Skip** | `POST /resume-workflow` `{"thread_id": "..._master", "decision": "skip"}` |
| **Override (mark done)** | `POST /resume-workflow` `{"thread_id": "..._master", "decision": "override"}` |

Checkpoints are automatically deleted from `workflow_checkpoints` after the full pipeline finishes successfully.

---

## Firestore Data Model

### Topic Collections

```
planet_protectors_topics   (theme1 — environment/nature)
mindful_topics             (theme2 — religion/spirituality)
chill_stories_topics       (theme3 — lifestyle/wellness)

Document ID: {age}__{lang}__{filter_value}   e.g. "3_4__en__india"

Fields:
  topics_id:    string (UUID for this batch)
  theme:        "theme1" | "theme2" | "theme3"
  age:          "3-4" | "5-6" | ...
  language:     "en" | "te"
  filter_type:  "country" | "religion" | "lifestyle_area"
  filter_value: "India" | "hindu" | "Breathwork & Body Awareness"
  topics:       [{title, description, moral, theme, story_id?}]
                 story_id is patched in after each story is successfully created.
                 On re-run, story_id presence triggers resume logic (skip/partial).
  created_at:   timestamp
  updated_at:   timestamp (set when story_id is patched in)
```

### Story Collections

```
planet_protectors_stories   (theme1)
mindful_stories             (theme2)
chill_stories               (theme3)

Document ID: story_id (UUID — O(1) direct lookup)

Fields:
  story_id:         string (UUID)
  topics_id:        string (links back to topic batch)
  title:            string
  description:      string
  story_text:       string (normalised from "story" key in LLM output)
  moral:            string
  age_group:        string (injected from runtime, not from LLM)
  language:         "en" | "te" (injected from runtime, not from LLM)
  theme:            "theme1" | "theme2" | "theme3"
  image_url:        string (GCS public URL, set after WF3)
  audio_url:        string (GCS public URL, set after WF4)
  image_prompt:     string (FLUX prompt from story JSON)
  mcq_seeds:        string[] (activity seeds)
  art_seed:         string
  science_concepts: [{concept, explanation}]
  activities:       {mcq: "ready", art: "ready", science: "ready", moral: "ready"}
  updated_at:       timestamp
```

### Activity Collection

```
activities_v1

Document ID: auto-generated

Fields:
  story_id:   string (links to story doc)
  type:       "mcq" | "art" | "moral" | "science"
  items:      [...] | (activity-specific payload)
  image:      string (GCS path for art/moral/science)
  created_at: timestamp
```

### Supporting Collections

| Collection | Purpose |
|---|---|
| `workflow_checkpoints` | LangGraph state persistence for HITL resumption. Auto-deleted after successful pipeline completion. |
| `story_topics_v1` | Session log of topic generation runs (topic list + human selection). |
| `story_images_v1` | Image generation metadata log (prompt, URL, theme). |
| `story_audio_v1` | Audio generation metadata log (language, voice name, URL, theme). |

---

## AI Models

| Workflow | Model | Reason |
|---|---|---|
| WF1 Topics | `gemini-2.0-flash-lite` | Short/simple output, cost-optimised |
| WF1 Self-correction | `gemini-2.0-flash-lite` | Same as generator |
| WF2 Story | `gemini-2.5-flash-lite` | Higher quality for narrative |
| WF2 Self-correction | `gemini-2.5-flash-lite` | Match generator quality |
| WF3 Image | FLUX.1-schnell (HF) | 4 steps, ~10× cheaper than FLUX.1-dev |
| WF4 Audio | Google Cloud TTS | Managed, reliable, multi-language |
| WF5 Activities | `gemini-2.0-flash-lite` | Simple structured output |
| Evaluation (all) | `gemini-2.0-flash-lite` | GEval scoring is a simpler task |
| Fallback (all) | `gemini-2.0-flash-lite` | Note: `gemini-2.0-flash` (no suffix) returns 404 |

---

## Resilience Patterns

```
src/utils/resilience.py
```

| Pattern | Where used |
|---|---|
| `@circuit_breaker` (threshold=5, recovery=60s) | AIService primary/fallback LLM calls |
| `@retry_with_backoff` (max=3, base_delay=2s, exponential+jitter) | AIService LLM calls |
| `RateLimiter` (token bucket, 3 req/s, burst=6) | AIService all API calls |
| Subgraph 4-retry loop | WF3 image, WF4 audio, WF5 each activity |
| LangGraph `interrupt()` + Firestore checkpoint | HITL after 4 subgraph failures |
| `asyncio.Semaphore(10)` | Batch story creation concurrency cap |
| `asyncio.gather(return_exceptions=True)` | WF3+WF4 parallel — one failure doesn't cancel the other |
| Restart resume check | `batch_create_stories` checks `topic.story_id` before running WF2 |
| Partial cache merge | `_generate_one` generates only `n - len(cached)` new topics when count increases |

---

## Story JSON Normalisation

The LLM story response uses non-standard field names. The `StoryCreatorAgent` normalises these before validation:

| LLM returns | Pipeline expects | Fix |
|---|---|---|
| `"story"` | `"story_text"` | `_parse_story` remaps key |
| no `"title"` | `"title"` | falls back to `selected_topic["title"]` |
| no `"age_group"` | `"age_group"` | injected from `state["age"]` |
| no `"language"` | `"language"` | injected from `state["language"]` |
| raw `\n` inside JSON string | valid JSON | `_escape_control_chars()` state-machine sanitiser |

The `_escape_control_chars()` method walks the response character by character, tracking whether it's inside a JSON string, and escapes any raw control characters (`\n`, `\r`, `\t`, etc.) before the second parse attempt.

---

## Prompt System

```
src/prompts/
  __init__.py                    PromptRegistry (singleton, versioned loading)
  story_topics/
    theme1/v1_en.txt             PlanetProtector topic generation (English)
    theme1/v1_te.txt             (Telugu)
    theme2/v1_en.txt             MindfullTopics
    theme3/v1_en.txt             ChillStories
  story_creator/
    theme1/v1_en.txt
    theme2/v1_en.txt
    theme3/v1_en.txt
  mcq/v1_en.txt
  art/v1_en.txt
  moral/v1_en.txt
  science/v1_en.txt
  image_generator/v1_en.txt
```

- `get_prompt(agent, version="latest", **kwargs)` loads and interpolates
- `_safe_format()` uses regex substitution (not `str.format`) — safe with JSON examples in prompts that contain `{...}` literals
- Version pins via config: `MCQ_PROMPT_VERSION`, `ART_PROMPT_VERSION`, etc.
- Prompt files are user-managed — the system creates placeholder paths only

---

## Topic Taxonomy

```
src/topics/
  pp_topics.py         PlanetProtector — nature/environment topics, age-filtered
  mindfull_topics.py   MindfullTopics  — religion_sources map (hindu, christian, islam, ...)
  chill_stories.py     ChillStories    — lifestyle_areas (Breathwork, Movement, Sleep, ...)
```

These are hardcoded taxonomy files used to build LLM prompts (`promptText` variable). The LLM receives topic names as inspiration seeds, not as exact output titles.

---

## Observability

**Langfuse** — open-source LLM tracing (MIT license, free cloud tier at cloud.langfuse.com)

```
src/utils/tracing.py    get_trace_callbacks(name, metadata, tags, session_id)
```

| Trace | Tags | Session |
|---|---|---|
| `WF1-topics` | `wf1, topics` | session UUID |
| `WF2-story` | `wf2, story` | `story_id` |
| `master-pipeline` | `master, image, audio, activities` | `story_id` |
| `WF3-image` | `wf3, image, manual-retry` | `story_id` |
| `WF4-audio` | `wf4, audio, manual-retry` | `story_id` |

All traces carry `story_id`, `theme`, `age`, `title` as metadata. Batch runs share `session_id = topics_id` so all concurrent stories are grouped under one trace session.

**Activation** — add to `.env`:
```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

When `LANGFUSE_ENABLED=false` (default), `get_trace_callbacks()` returns `[]` — zero overhead.

---

## API Reference

### Stories

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/generate-topics` | `{country, religion[], age, language, theme?, voice_type?, preferences[]}` | WF1: Generate topic options (background). Returns 202 immediately. |
| `POST` | `/select-topic` | `{story_id, selected_topic}` | Human picks a topic → triggers WF2 story generation (background). |

### Media

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/generate-media/{story_id}` | `{story_id, age, language, voice_type?}` | Full pipeline: WF3+WF4 then WF5 (background). `voice_type`: `"chirp"` or `"standard"` (default `"standard"`). |
| `POST` | `/generate-image/{story_id}` | `{age?, language?}` | Retrigger WF3 image only (HITL bypass). |
| `POST` | `/generate-audio/{story_id}` | `{language?, voice?}` | Retrigger WF4 audio only (HITL bypass). |
| `POST` | `/resume-workflow` | `{thread_id, decision}` | Resume interrupted master. `decision`: `retry` / `skip` / `override`. |
| `GET`  | `/workflow-status/{story_id}` | — | Poll completion state (`wf2_story`, `wf3_image`, `wf4_audio`, `activities`). |

### Activities

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate-activities` | WF5 standalone — backward-compatible direct trigger. |
| `POST` | `/pubsub-handler` | Cloud Pub/Sub push handler (triggers WF5). |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check. |

---

## Infrastructure

```
Google Cloud Run       FastAPI app (container, PORT=8080)
Firestore              State, stories, topics, activities, checkpoints
Cloud Storage (GCS)    story-images/{uuid}.png  ·  story-audio/{uuid}.mp3
Cloud Pub/Sub          HITL notifications (HUMAN_LOOP_NOTIFICATION_TOPIC)
HuggingFace Inference  FLUX.1-schnell image generation (HF_TOKEN)
Google TTS             Audio narration (Standard and Chirp3-HD voices)
Langfuse               LLM tracing (optional, free cloud tier)
```

---

## Key File Index

```
src/
  main.py                              FastAPI app, startup/shutdown hooks
  api/
    stories.py                         /generate-topics, /select-topic
    media.py                           /generate-media, /generate-image, /generate-audio, /resume-workflow
    activities.py                      /generate-activities, /pubsub-handler
    health.py                          /health
  workflows/
    story_topics_workflow.py           WF1 graph + batch_create_stories_node (resume logic)
    story_creator_workflow.py          WF2 graph
    master_workflow.py                 Master: Phase1(WF3+WF4) → Phase2(WF5) → finalize
                                       Voice pool: _pick_voice(language, voice_type)
    image_workflow.py                  WF3 compiled subgraph
    audio_workflow.py                  WF4 compiled subgraph
    activity_workflow.py               WF5 compiled subgraph
  agents/
    story/
      topics_creator_agent.py          Theme 1/2/3 topic generation + Firestore cache
                                       Partial cache: generates missing topics when n increases
      story_creator_agent.py           Full story JSON from selected topic
                                       Normalises "story"→"story_text", injects age_group/language
                                       _escape_control_chars() for JSON parse resilience
      self_correction_agent.py         LLM-based content correction from evaluation feedback
    media/
      image_generator_agent.py         FLUX.1-schnell image bytes
      audio_generator_agent.py         Google TTS audio bytes
    activities/
      mcq_agent.py                     Multiple-choice questions (uses mcq_seeds)
      art_agent.py                     Art activity + image (uses art_seed)
      science_agent.py                 Science activity + image (uses science_concepts)
      moral_agent.py                   Moral activity + image (uses moral)
    validators/
      validator_agent.py               Structural validation (required fields)
      evaluation_agent.py              DeepEval GEval scoring
  models/
    state.py                           LangGraph TypedDicts + merge_dicts reducer
  services/
    ai_service.py                      Gemini API + FLUX + caching + circuit breaker + rate limiter
    audio_service.py                   Google Cloud TTS wrapper
    database/
      firestore_service.py             All Firestore CRUD (stories, topics, activities, checkpoints)
      storage_bucket.py                GCS upload (images, audio)
      checkpoint_service.py            LangGraph Firestore checkpointer
  prompts/
    __init__.py                        PromptRegistry, versioned loading, _safe_format()
    story_topics/{theme}/v{N}_{lang}.txt
    story_creator/{theme}/v{N}_{lang}.txt
    mcq|art|moral|science/v{N}_{lang}.txt
  topics/
    pp_topics.py                       PlanetProtector taxonomy (age-filtered)
    mindfull_topics.py                 Religion sources map
    chill_stories.py                   Lifestyle areas taxonomy
  utils/
    config.py                          Pydantic BaseSettings (env vars, models, limits)
    logger.py                          Structured logging setup
    resilience.py                      CircuitBreaker, retry_with_backoff, RateLimiter
    tracing.py                         Langfuse callbacks (no-op when disabled)
```

---

## Environment Variables

```
# Google Cloud
GOOGLE_CLOUD_PROJECT=riokutty
GOOGLE_CLOUD_BUCKET=kutty_bucket
GOOGLE_API_KEY=...
FIRESTORE_DATABASE=(default)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json   # optional

# AI Models
STORY_TOPICS_MODEL=gemini-2.0-flash-lite
STORY_CREATOR_MODEL=gemini-2.5-flash-lite
EVALUATION_MODEL=gemini-2.0-flash-lite
FLUX_IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
HF_TOKEN=...

# TTS defaults (overridden by voice pool selection at runtime)
TTS_LANGUAGE_CODE=en-US
TTS_VOICE_NAME=en-US-Standard-A
TTS_AUDIO_ENCODING=MP3

# Resilience
MAX_RETRIES=3
RETRY_DELAY_SECONDS=2
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=60
PARALLEL_WORKFLOW_MAX_RETRIES=4
MAX_CONCURRENCY=10

# Batch
TOPICS_PER_THEME=5

# HITL
HUMAN_LOOP_NOTIFICATION_TOPIC=projects/.../topics/...

# Langfuse (optional)
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```
