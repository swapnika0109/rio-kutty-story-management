# Rio Kutty — Stakeholder Pitch Deck
---

## SLIDE 1 — TITLE

# Rio Kutty
### AI-Powered Personalised Story Platform for Children

> Every child deserves a story written just for them.

---

## SLIDE 2 — THE PROBLEM

### Children's content today is generic, passive, and one-size-fits-all.

| Problem | Impact |
|---|---|
| Same stories for every child | No cultural or personal connection |
| No educational layer | Entertainment without learning |
| Static content libraries | No freshness, no replayability |
| Language barriers | Non-English kids left behind |

> A 4-year-old in rural Andhra Pradesh and a 7-year-old in New York shouldn't get the same story.

---

## SLIDE 3 — THE SOLUTION

### Rio Kutty generates a **complete, personalised story package** for every child — on demand.

**One story = Story + Images + Audio + 4 Learning Activities**

```
Child's Profile  →  AI Engine  →  Personalised Package
─────────────────────────────────────────────────────
Age: 5            Story        Rich narrative (480–520 words)
Country: India    Images       4 scene illustrations (FLUX AI)
Language: Telugu  Audio        Full narration (Google TTS)
Theme: Nature     Activities   MCQ · Art · Science · Moral
```

**3 Content Themes:**
- 🌿 **PlanetProtector** — nature, environment, science
- 🕌 **MindfullTopics** — wisdom traditions, ancient stories
- ☀️ **ChillStories** — mindfulness, slow living, wellbeing

---

## SLIDE 4 — PRODUCT FLOW (USER PERSPECTIVE)

```
┌─────────────────────────────────────────────────────────┐
│                     USER JOURNEY                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Child opens app                                     │
│     → Profile: age, language, country, preferences      │
│                                                         │
│  2. Story library loads instantly                       │
│     → Pre-generated stories by theme                   │
│     → Already personalised to their profile            │
│                                                         │
│  3. Child listens / reads                               │
│     → Full narration in their language                  │
│     → Scene illustrations per story moment              │
│                                                         │
│  4. Learning activities unlock                          │
│     → Quiz (MCQ) · Drawing prompt · Science fact        │
│     → Moral discussion guide                           │
│                                                         │
│  5. Parent / Child creates custom story (Chatbot)       │
│     → Conversational story builder                     │
│     → Same AI quality, child-directed                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## SLIDE 5 — HIGH-LEVEL SYSTEM ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE                          │
└──────────────────────────────────────────────────────────────┘

   ┌──────────┐
   │  Mobile  │
   │  / Web   │         (Flutter / React Native)
   │    UI    │
   └────┬─────┘
        │  REST
        ▼
   ┌──────────────────────────────────┐
   │          GO API GATEWAY           │  ← Concurrent · Fast · Low latency
   │                                  │    Handles all user-facing requests
   │  GET  /stories          (reads)  │    Reads directly from Firestore
   │  POST /generate-topics  (write)  │    Delegates heavy work to Python
   │  POST /select-topic     (write)  │
   │  POST /generate-media   (write)  │
   └───────────┬──────────────────────┘
               │  Internal HTTP
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────────┐
│   PYTHON    │  │     PYTHON       │
│  Story Mgmt │  │  Chatbot Creator │  ← Separate project
│  (This      │  │  (User-directed  │    Conversational story
│  project)   │  │  story builder)  │    builder via chat
└──────┬──────┘  └──────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│         GOOGLE CLOUD            │
│  Firestore  ·  GCS  ·  Pub/Sub  │
└─────────────────────────────────┘
```

---

## SLIDE 6 — TWO PATHS: LIBRARY vs CHATBOT

```
┌─────────────────────────────────────────────────────────────┐
│                    TWO CONTENT PATHS                        │
├────────────────────────────┬────────────────────────────────┤
│   PRE-GENERATED LIBRARY    │     CHATBOT STORY CREATOR      │
├────────────────────────────┼────────────────────────────────┤
│ Batch AI pipeline          │ Conversational AI              │
│ Run at content refresh     │ Run on user request            │
│ Cached in Firestore        │ Created fresh                  │
│ Personalised by profile    │ Child / parent directed        │
│ 45+ stories per refresh    │ Unlimited, unique              │
│ Instant load for user      │ Minutes to generate            │
├────────────────────────────┼────────────────────────────────┤
│ Story Mgmt Python project  │ Chatbot Python project         │
│ (this system)              │ (separate)                     │
└────────────────────────────┴────────────────────────────────┘
```

Both paths produce the **same full package**: story + images + audio + activities.

---

## SLIDE 7 — TECHNICAL ARCHITECTURE (PYTHON AI PIPELINE)

```
POST /generate-topics
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  WF1 — STORY TOPICS WORKFLOW                             │
│                                                          │
│  [Generate Topics] → [Validate] → [Evaluate (DeepEval)] │
│       ↑ retry             │ pass                         │
│  [Self-Correct] ←─────────┘                             │
│                                                          │
│  Per theme, per filter (country / religion / lifestyle): │
│  Theme 1: PlanetProtector  (1 LLM call per country)      │
│  Theme 2: MindfullTopics   (1 call per religion — ×8)    │
│  Theme 3: ChillStories     (1 call per lifestyle — ×7)   │
│                                                          │
│  Cache: rio_titles_theme1/2/3 in Firestore               │
└──────────────────────┬───────────────────────────────────┘
                       │ for each topic (up to 10 parallel)
                       ▼
┌──────────────────────────────────────────────────────────┐
│  WF2 — STORY CREATOR WORKFLOW                            │
│                                                          │
│  [Generate Story] → [Validate] → [Evaluate (DeepEval)]  │
│       ↑ retry              │ pass                        │
│  [Self-Correct] ←──────────┘                            │
│                       │                                  │
│               [Save to Firestore]                        │
│          rio_stories_theme1/2/3 (by story_id)            │
└──────────────────────┬───────────────────────────────────┘
                       │ asyncio.gather (parallel)
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌─────────┐ ┌──────────────────────────┐
   │ WF3 IMAGE  │ │WF4 AUDIO│ │    WF5 ACTIVITIES         │
   │            │ │         │ │                           │
   │ FLUX.1-    │ │Google   │ │ MCQ · Art                 │
   │ schnell    │ │   TTS   │ │ Science · Moral           │
   │ (4 retries)│ │(4 retry)│ │ (4 retries each, parallel)│
   │            │ │         │ │                           │
   │ → GCS      │ │ → GCS   │ │ → Firestore activities_v1 │
   │ → story    │ │ → story │ │   tagged with story_id    │
   │   image_url│ │ audio_  │ │                           │
   └────────────┘ │   url   │ └──────────────────────────┘
                  └─────────┘
```

---

## SLIDE 8 — AI QUALITY PIPELINE

### Every piece of content passes through a 3-stage quality gate:

```
STAGE 1 — STRUCTURAL VALIDATION (instant)
  ✓ Required fields present
  ✓ Non-empty content
  ✓ Correct format

STAGE 2 — AI EVALUATION (DeepEval GEval — 8 metrics in parallel)
  ✓ Non-toxicity      Score: 0–1
  ✓ Bias              Age-appropriate, culture-neutral
  ✓ Completeness      All story elements present
  ✓ Engagability      Child interest retention
  ✓ Trustworthiness   Factual accuracy
  ✓ Precision         On-topic relevance
  ✓ Recall            Theme coverage
  ✓ Latency           Response speed

  Pass threshold: avg ≥ 0.75 across all 8 metrics

STAGE 3 — SELF-CORRECTION (up to 2 attempts)
  If evaluation fails → AI corrects its own output
  → Regenerates and re-evaluates
  → Escalates to human review if still failing
```

**Evaluator model:** Gemini 2.0 Flash Lite (cost-effective, dedicated)
**Story model:** Gemini 2.5 Flash (high quality narrative generation)

---

## SLIDE 9 — HUMAN-IN-THE-LOOP (HITL) SAFETY NET

```
Parallel workflow fails after 4 retries
              │
              ▼
   ┌─────────────────────┐
   │  Pub/Sub Notification│  → Admin notified instantly
   │  to admin channel    │
   └──────────┬──────────┘
              │
              ▼
   LangGraph interrupt()    ← Workflow PAUSED here
   State persisted to         Full state saved in Firestore
   Firestore checkpoint
              │
   Admin reviews & decides
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 "retry"   "skip"    "override"
 Re-trigger  Mark as   Admin-approved
 workflow    skipped   as complete
```

**No content is published to users without passing quality gates.**

---

## SLIDE 10 — PERSONALISATION ENGINE

### How we personalise every story:

```
User Profile Input:
┌─────────────────────────────────────────────────┐
│  age:      3-4 / 5-6 / 7-8 / 9-10              │
│  language: English / Telugu (+ more)            │
│  country:  India / USA / UK / Any              │
│  religion: Hindu / Christian / Islamic / ...    │
│            (8 traditions + universal)           │
│  theme:    PlanetProtector / Mindful / Chill    │
│  prefs:    Nature / Animals / Space / ...       │
└─────────────────────────────────────────────────┘

Generates → Unique content per combination
           Story vocab adapted to age
           Cultural references from country
           Wisdom traditions from religion
           Voice & pacing in native language
```

**Cache-first:** Same profile combination → instant return from Firestore
**LLM only on miss:** New combination → generate → cache → return

---

## SLIDE 11 — DATA ARCHITECTURE

```
FIRESTORE COLLECTIONS
─────────────────────

rio_titles_theme1         Topic title library — PlanetProtector
rio_titles_theme2         Topic title library — MindfullTopics
rio_titles_theme3         Topic title library — ChillStories
  └─ doc: {age}__{lang}__{filter_value}
     fields: titles[], cached LLM output

rio_stories_theme1        Full stories — PlanetProtector
rio_stories_theme2        Full stories — MindfullTopics
rio_stories_theme3        Full stories — ChillStories
  └─ doc: {story_id}   ← direct O(1) lookup
     fields: story_text, moral, image_url, audio_url,
             character_names, setting, age_group, language

activities_v1             All learning activities
  └─ doc: {story_id}_{activity_type}
     tagged with story_id for cross-reference

workflow_checkpoints      LangGraph state persistence
  └─ doc: {thread_id}_{checkpoint_id}
     enables crash recovery & HITL resume

GOOGLE CLOUD STORAGE
────────────────────
kutty_bucket/
  images/{story_id}.png    FLUX.1 generated illustrations
  audio/{story_id}.mp3     Google TTS narrations
```

---

## SLIDE 12 — TECHNOLOGY STACK

```
LAYER              TECHNOLOGY              WHY
─────────────────────────────────────────────────────────────
Mobile / Web       Flutter / React Native  Cross-platform

API Gateway        Go                      Concurrent, low latency
                                           Handles 1000s of reads/s

AI Orchestration   Python + LangGraph      Stateful AI workflow graphs
                   FastAPI                 Async HTTP, production-ready

AI Models          Gemini 2.5 Flash        Story generation (quality)
                   Gemini 2.0 Flash Lite   Topics, eval (cost-efficient)
                   FLUX.1-schnell          Image generation (4-step, fast)
                   Google Cloud TTS        Audio narration

Evaluation         DeepEval (GEval)        8 parallel LLM-as-judge metrics

Database           Google Firestore        Real-time, globally distributed
Storage            Google Cloud Storage    Images + audio CDN delivery
Messaging          Google Cloud Pub/Sub    HITL admin notifications
Checkpointing      Firestore              Crash recovery, workflow state

Infrastructure     Google Cloud Platform   Single vendor, managed
```

---

## SLIDE 13 — PERFORMANCE & SCALE

```
CONCURRENCY
  • 10 stories generated in parallel (configurable)
  • Within each story: image + audio + 4 activities run in parallel
  • Go API handles concurrent reads with zero blocking

GENERATION THROUGHPUT (estimate)
  • 45 topics → 45 full story packages
  • Wall-clock time: ~10–15 minutes (parallel)
  • vs sequential: ~3–4 hours

CACHE EFFICIENCY
  • Topic titles: generated once, reused across all users
  • Story lookup: O(1) — story_id is the Firestore document ID
  • No queries, no indexes needed for reads

RESILIENCE
  • 4-retry with exponential backoff per workflow step
  • Circuit breaker prevents cascade failures
  • Rate limiter: 3 req/s burst 6 (Gemini API limits)
  • Full state checkpointed after every graph node
  • HITL escalation as final safety net
```

---

## SLIDE 14 — CONTENT PIPELINE SUMMARY

```
1 API call  →  POST /generate-topics (age, language, country)

               45 topic titles generated across 3 themes
               ↓
               45 full stories created (WF2, 10 parallel)
               ↓
               45 × {4 images + 1 audio + 4 activities}
               generated (WF3+WF4+WF5, parallel per story)
               ↓
               All persisted to Firestore + GCS
               ↓
               Go API serves reads instantly to millions of users

Total content per run: 45 stories × full package =
  45 stories · 180 images · 45 audio files · 180 activities
```

---

## SLIDE 15 — CURRENT STATUS & ROADMAP

### Status: Backend AI Pipeline — Production Ready ✓

| Component | Status |
|---|---|
| WF1 Story Topics | ✅ Complete |
| WF2 Story Creator | ✅ Complete |
| WF3 Image Generator | ✅ Complete |
| WF4 Audio Generator | ✅ Complete |
| WF5 Activities (MCQ/Art/Science/Moral) | ✅ Complete |
| AI Evaluation (DeepEval 8 metrics) | ✅ Complete |
| Human-in-loop + Admin Resume | ✅ Complete |
| Firestore theme collections | ✅ Complete |
| GCS image + audio storage | ✅ Complete |
| Chatbot story creator | ✅ Separate project |

### Next Milestones

```
Phase 7:  Go API integration + mobile UI connects to live pipeline
Phase 8:  Telugu language full prompt set (WF1+WF2)
Phase 9:  More languages (Hindi, Tamil, Kannada)
Phase 10: User analytics + personalisation feedback loop
Phase 11: Teacher / parent dashboard
Phase 12: Subscription & monetisation layer
```

---

## SLIDE 16 — WHY NOW, WHY US

```
MARKET OPPORTUNITY
  • 1.2B children under 14 worldwide
  • EdTech market: $400B by 2028
  • Regional language content: massively underserved
  • Parents willing to pay for quality, safe, educational content

OUR EDGE
  ✓ AI-generated = unlimited fresh content (vs static libraries)
  ✓ Personalised to culture, language, age, values
  ✓ Full package: story + images + audio + activities
  ✓ Quality-gated: every piece of content evaluated by AI
  ✓ Production-ready AI pipeline running today
  ✓ Two content modes: library + chatbot creator
  ✓ Built on Google Cloud — scales globally from day 1

MOAT
  The more profiles we serve, the more personalisation data
  The richer our topic library (cached) → faster, cheaper
  Network effect: content reuse across similar profiles
```

---

## SLIDE 17 — CLOSING

# Rio Kutty

### Every child, every language, every story.

**What we're asking for:**
- Feedback on content themes and personalisation depth
- Partnerships for regional language content experts
- Go-to-market strategy for first 3 markets

**Contact:**
- Demo available: live API running today
- Firestore data: stories already generated and stored

> _"The stories children hear become the stories they live."_

---
*Deck prepared: March 2026*
