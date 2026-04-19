"""
TopicsCreatorAgent — generates story topic titles from 3 themes, cached in Firestore.

Per-run flow:
  Theme 1 (PlanetProtector):  N titles for the requested COUNTRY (1 LLM call)
  Theme 2 (MindfullTopics):   N titles for EACH RELIGION in the taxonomy (parallel)
  Theme 3 (ChillStories):     N titles for EACH LIFESTYLE AREA — all 7 (parallel)

Cache: story_title_library_v1/{theme}__{age}__{lang}__{filter_value_normalized}
  - Cache hit  → return stored titles, no LLM call
  - Cache miss → call LLM → save to cache → return titles

Output state["topics"] — flat list of dicts:
  {title, description, theme, moral, filter_type, filter_value}
"""

import asyncio
import json
import random
import re
import uuid

from ...services.ai_service import AIService
from ...services.database.firestore_service import FirestoreService
from ...utils.logger import setup_logger
from ...utils.config import get_settings
from ...prompts import get_registry
from ...topics.pp_topics import PlanetProtector
from ...topics.mindfull_topics import MindfullTopics
from ...topics.chill_stories import ChillStoriesTopics

logger   = setup_logger(__name__)
settings = get_settings()

_LANG_CODE = {
    "English": "en",
    "Telugu":  "te",
    "en":      "en",
    "te":      "te",
}


# ---------------------------------------------------------------------------
# Prompt-text builders  (comma-separated high-level topic names)
# ---------------------------------------------------------------------------

def _parse_age_range(age: str) -> tuple[int, int]:
    try:
        parts = age.split("-")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        v = int(parts[0])
        return v, v
    except (ValueError, IndexError):
        return 3, 9


def _pp_prompt_text(age: str, n: int) -> str:
    """Age-filtered PlanetProtector topic names, comma-separated."""
    age_min, age_max = _parse_age_range(age)
    topics = PlanetProtector().topics.get("topics", [])
    filtered = [
        t for t in topics
        if t.get("is_active", True)
        and t.get("age_min", 0) <= age_max
        and t.get("age_max", 99) >= age_min
    ] or topics
    return ", ".join(t["name"] for t in filtered[:n])


def _mindful_prompt_text(religion_key: str, n: int) -> str:
    """Religion source list for one religion, comma-separated."""
    sources_map = MindfullTopics().topics.get("religion_sources", {})
    sources = sources_map.get(religion_key) or sources_map.get("universal_wisdom", [])
    return ", ".join(sources[:n])


def _chill_prompt_text(lifestyle_area: str, age: str, n: int) -> str:
    """ChillStories topic names filtered by lifestyle area AND age, comma-separated."""
    age_min, age_max = _parse_age_range(age)
    topics = ChillStoriesTopics().topics.get("topics", [])
    filtered = [
        t for t in topics
        if t.get("is_active", True)
        and t.get("lifestyle_area") == lifestyle_area
        and t.get("age_min", 0) <= age_max
        and t.get("age_max", 99) >= age_min
    ] or [t for t in topics if t.get("lifestyle_area") == lifestyle_area]
    return ", ".join(t["name"] for t in filtered[:n])


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_pipe_response(
    response: str, theme: str, filter_type: str, filter_value: str
) -> list:
    """
    Parses LLM response into topic dicts.
    Handles two formats the model may return:

    1. JSON array (actual):
       [{"title": "...", "description": "..."}, ...]
       Also handles ```json ... ``` code fences.

    2. Pipe-separated lines (prompt asks for):
       Title text|Description text
    """
    def _make_topic(title: str, description: str) -> dict:
        return {
            "topic_id":     str(uuid.uuid4()),
            "title":        title.strip(),
            "description":  description.strip(),
            "theme":        theme,
            "moral":        "",
            "filter_type":  filter_type,
            "filter_value": filter_value,
        }

    text = response.strip()

    # --- Try JSON first (strip optional ```json ... ``` fences) ---
    json_text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    json_text = re.sub(r"\s*```$", "", json_text).strip()

    if json_text.startswith("["):
        try:
            items = json.loads(json_text)
            topics = []
            for item in items:
                title = item.get("title", "").strip()
                desc  = item.get("description", "").strip()
                if title and desc:
                    topics.append(_make_topic(title, desc))
            if topics:
                return topics
        except (json.JSONDecodeError, AttributeError):
            pass  # fall through to pipe parser

    # --- Fallback: pipe-separated lines ---
    topics = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            parts = line.split("|", 1)
            if parts[0].strip() and parts[1].strip():
                topics.append(_make_topic(parts[0], parts[1]))
    return topics


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TopicsCreatorAgent:
    def __init__(self, prompt_version: str = "1"):
        self.ai_service = AIService()
        self.db         = FirestoreService()
        self.prompt_version = prompt_version

    async def generate(self, state: dict) -> dict:
        """
        Generates / retrieves cached story topic titles.

        State fields used (from config.configurable via _unpack_config):
            age, language, religion, country, theme (optional — if set, only that theme runs)
        """
        age      = state.get("age", "3-4")
        language = state.get("language", "English")
        religion = state.get("religion", "universal_wisdom")
        country  = state.get("country", "Any")
        theme    = state.get("theme", "")   # e.g. "theme1", "theme2", "theme3", or "" for all

        lang_code = _LANG_CODE.get(language, "en")
        version   = f"v{self.prompt_version}_{lang_code}"
        n         = settings.TOPICS_PER_THEME
        registry  = get_registry()

        # Normalise: accept "1"/"2"/"3" as well as "theme1"/"theme2"/"theme3"
        requested = theme.lower().strip() if theme else ""
        if requested and not requested.startswith("theme"):
            requested = f"theme{requested}"

        run_theme1 = not requested or requested == "theme1"
        run_theme2 = not requested or requested == "theme2"
        run_theme3 = not requested or requested == "theme3"

        logger.info(
            f"[TopicsCreator] age={age} lang={lang_code} religion={religion} "
            f"country={country} n={n} theme_filter={requested or 'all'}"
        )

        all_topics: list[dict] = []

        # ------------------------------------------------------------------
        # Theme 1 — PlanetProtector, one record per COUNTRY
        # ------------------------------------------------------------------
        if run_theme1:
            t1 = await self._generate_one(
                theme_name   = "theme1",
                version      = version,
                filter_type  = "country",
                filter_value = country,
                prompt_kwargs= {
                    "age":        age,
                    "length":     n,
                    "promptText": _pp_prompt_text(age, n),
                    "country":    country,
                },
                age=age, lang=lang_code, registry=registry,
            )
            all_topics.extend(t1)

        # ------------------------------------------------------------------
        # Theme 2 — MindfullTopics, filtered by religion (or all if "any")
        # ------------------------------------------------------------------
        if run_theme2:
            all_religions = list(MindfullTopics().topics.get("religion_sources", {}).keys())
            _skip = {"any", "universal_wisdom", ""}
            if isinstance(religion, list):
                requested_religions = [r.lower().strip() for r in religion if r.lower().strip() not in _skip]
            else:
                requested_religions = [religion.lower().strip()] if religion and religion.lower().strip() not in _skip else []

            if requested_religions:
                religions_to_run = [r for r in all_religions if r.lower() in requested_religions]
                if not religions_to_run:
                    logger.warning(f"[TopicsCreator] Religion(s) '{religion}' not found in taxonomy, using all")
                    religions_to_run = all_religions
                else:
                    logger.info(f"[TopicsCreator] Religion filter: {religions_to_run}")
            else:
                religions_to_run = all_religions

            t2_results = await asyncio.gather(*[
                self._generate_one(
                    theme_name   = "theme2",
                    version      = version,
                    filter_type  = "religion",
                    filter_value = rel,
                    prompt_kwargs= {
                        "age":        age,
                        "length":     n,
                        "promptText": _mindful_prompt_text(rel, n),
                        "religion":   rel,
                    },
                    age=age, lang=lang_code, registry=registry,
                )
                for rel in religions_to_run
            ], return_exceptions=True)

            for result in t2_results:
                if isinstance(result, list):
                    all_topics.extend(result)

        # ------------------------------------------------------------------
        # Theme 3 — ChillStories, one randomly selected lifestyle area
        # ------------------------------------------------------------------
        if run_theme3:
            lifestyle_areas = (
                ChillStoriesTopics().topics.get("meta", {}).get("lifestyle_areas", [])
            )
            if lifestyle_areas:
                area = random.choice(lifestyle_areas)
                logger.info(f"[TopicsCreator] theme3 selected lifestyle area: {area}")
                t3 = await self._generate_one(
                    theme_name   = "theme3",
                    version      = version,
                    filter_type  = "lifestyle_area",
                    filter_value = area,
                    prompt_kwargs= {
                        "age":        age,
                        "length":     n,
                        "promptText": _chill_prompt_text(area, age, n),
                    },
                    age=age, lang=lang_code, registry=registry,
                )
                all_topics.extend(t3)

        logger.info(f"[TopicsCreator] Total topics collected: {len(all_topics)}")

        if not all_topics:
            return {"errors": {"topics_creator": "All themes failed to generate topics"}}

        return {
            "topics":     all_topics,
            "validated":  False,
            "evaluation": None,
            "completed":  [],
            "errors":     {},
        }

    # ------------------------------------------------------------------
    # check cache → generate on miss → save → return titles
    # ------------------------------------------------------------------

    async def _generate_one(
        self,
        theme_name: str,
        version: str,
        filter_type: str,
        filter_value: str,
        prompt_kwargs: dict,
        age: str,
        lang: str,
        registry,
    ) -> list:
        """
        Returns titles for one (theme, filter_value) slot.
        Reads from Firestore cache; calls LLM only on cache miss.
        """
        # 1. Cache check
        n = settings.TOPICS_PER_THEME
        cached = await self.db.get_title_library_entry(theme_name, age, lang, filter_value)
        if cached:
            if len(cached) >= n:
                logger.info(f"[TopicsCreator] Cache hit: {theme_name}/{filter_value} ({len(cached)} topics)")
                return cached
            # Cache exists but has fewer than requested — generate the remaining ones below
            logger.info(
                f"[TopicsCreator] Partial cache hit: {theme_name}/{filter_value} "
                f"({len(cached)}/{n}) — generating {n - len(cached)} more"
            )

        # 2. Inject duplicate-prevention context when we have partial cache
        need = n - len(cached) if cached else n
        if cached:
            existing_titles_str = ", ".join(t["title"] for t in cached)
            prompt_kwargs = {
                **prompt_kwargs,
                "length":          need,
                "existing_titles": existing_titles_str,
            }
        else:
            prompt_kwargs = {**prompt_kwargs, "existing_titles": ""}

        # 3. Load prompt (skip silently if file not written yet)
        try:
            prompt = registry.get_prompt(
                f"story_topics/{theme_name}",
                version=version,
                **prompt_kwargs,
            )
        except FileNotFoundError:
            logger.warning(
                f"[TopicsCreator] {theme_name}/{version}.txt not found — skipping {filter_value}"
            )
            return []

        # 4. LLM call
        try:
            response = await self.ai_service.generate_content(
                prompt,
                model_override=settings.STORY_TOPICS_MODEL,
                fallback_override=settings.STORY_TOPICS_FALLBACK_MODEL,
                use_cache=False,
            )
        except Exception as e:
            logger.error(f"[TopicsCreator] LLM failed for {theme_name}/{filter_value}: {e}")
            return []

        # 5. Parse
        new_titles = _parse_pipe_response(response, theme_name, filter_type, filter_value)
        logger.info(f"[TopicsCreator] {theme_name}/{filter_value}: {len(new_titles)} new titles")

        # 6. Merge with existing cached titles (dedup by title text)
        if cached:
            existing_set = {t["title"] for t in cached}
            deduped = [t for t in new_titles if t["title"] not in existing_set]
            titles = cached + deduped
            logger.info(
                f"[TopicsCreator] Merged: {len(cached)} cached + {len(deduped)} new = {len(titles)} total"
            )
        else:
            titles = new_titles

        # 7. Save to cache (non-fatal if it fails)
        if titles:
            try:
                await self.db.save_title_library_entry(
                    theme_name, age, lang, filter_type, filter_value, titles
                )
            except Exception as e:
                logger.warning(f"[TopicsCreator] Cache save failed for {theme_name}/{filter_value}: {e}")

        return titles
