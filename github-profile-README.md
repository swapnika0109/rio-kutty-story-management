[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/[your-linkedin-id])

# Satyaraj [Your Last Name]

---

### Professional Summary:

**AI Engineer & Backend Architect** building production-grade AI systems with a focus on agentic pipelines, LLM orchestration, and cloud-native infrastructure. Specialised in end-to-end AI product development — from prompt engineering and model evaluation to concurrent API design and scalable deployment on Google Cloud.

Currently building **Rio Kutty** — an AI-powered personalised story platform for children, generating complete story packages (narrative + images + audio + activities) using a 5-workflow LangGraph pipeline with DeepEval quality gates.

---

### Engineering Highlights:

- Designed and shipped a **5-stage AI pipeline** (LangGraph) orchestrating story generation, image synthesis, TTS audio, and educational activities — all with automated quality evaluation and human-in-loop fallback
- Built **concurrent batch story generation** processing 45 stories in parallel using `asyncio.gather` with semaphore-bounded concurrency, reducing wall-clock time from hours to minutes
- Implemented **AI self-correction loops** using DeepEval GEval (8 parallel metrics) — content that fails quality thresholds is automatically corrected before reaching users
- Architected **theme-based Firestore collections** with `story_id` as document ID for O(1) direct lookups at any scale
- Integrated **FLUX.1-schnell** (HuggingFace) for cost-effective image generation and **Google Cloud TTS** for multilingual audio narration
- Built a **human-in-loop system** using LangGraph `interrupt()` + Pub/Sub — failed workflows pause, notify admins, and resume on decision without losing state
- Designed a **Go API gateway** for high-concurrency reads (1000s of users) with Python handling AI-heavy background jobs — each layer doing what it does best

---

### Technical Proficiency:

| Domain | Technologies |
|---|---|
| **AI Orchestration** | LangGraph · LangChain · Agentic Pipelines |
| **LLM / Generative AI** | Google Gemini 2.5 / 2.0 · Prompt Engineering · Self-Correction |
| **Image Generation** | FLUX.1-schnell · HuggingFace Inference API |
| **Evaluation** | DeepEval · GEval · LLM-as-Judge |
| **Backend** | Python · FastAPI · Go · asyncio |
| **Cloud & Infra** | Google Cloud Platform · Firestore · Cloud Storage · Pub/Sub |
| **Vector / RAG** | [Pinecone / Weaviate / pgvector if applicable] |
| **DevOps** | Docker · Cloud Run · GitHub Actions |

---

### Featured Projects:

**🧒 Rio Kutty — AI Story Platform for Children**
> End-to-end AI pipeline generating personalised stories for children, personalised by age, language, country, and cultural tradition.
> `Python` `LangGraph` `FastAPI` `Go` `Gemini` `FLUX.1` `DeepEval` `Firestore` `GCP`
- 5 LangGraph workflows: topics → story → image → audio → activities
- 3 content themes: PlanetProtector · MindfullTopics · ChillStories
- Automated quality evaluation with 8 GEval metrics before any content reaches users

---

**[Project 2 Name]**
> [One line description]
> `Tech1` `Tech2` `Tech3`

---

**[Project 3 Name]**
> [One line description]
> `Tech1` `Tech2` `Tech3`

---

### Currently:

- 🏗️ Building Rio Kutty — AI personalised story platform
- 📍 [City], India
- 💼 [Current Company / Open to opportunities]

---

*[your.email@example.com] · [linkedin.com/in/your-id] · [portfolio-site if any]*
