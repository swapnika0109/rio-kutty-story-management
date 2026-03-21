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
import re

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
        Generates / retrieves cached story topic titles from all 3 themes.

        State fields used (from config.configurable via _unpack_config):
            age, language, religion, country
        """
        age      = state.get("age", "3-4")
        language = state.get("language", "English")
        religion = state.get("religion", "universal_wisdom")
        country  = state.get("country", "Any")

        lang_code = _LANG_CODE.get(language, "en")
        version   = f"v{self.prompt_version}_{lang_code}"
        n         = settings.TOPICS_PER_THEME
        registry  = get_registry()

        logger.info(
            f"[TopicsCreator] age={age} lang={lang_code} "
            f"religion={religion} country={country} n={n}"
        )

        all_topics: list[dict] = []

        # ------------------------------------------------------------------
        # Theme 1 — PlanetProtector, one record per COUNTRY
        # ------------------------------------------------------------------
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
        # Theme 2 — MindfullTopics, one record per RELIGION (all religions)
        # ------------------------------------------------------------------
        all_religions = list(MindfullTopics().topics.get("religion_sources", {}).keys())

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
            for rel in all_religions
        ], return_exceptions=True)

        for result in t2_results:
            if isinstance(result, list):
                all_topics.extend(result)

        # ------------------------------------------------------------------
        # Theme 3 — ChillStories, one record per LIFESTYLE AREA (all 7)
        # ------------------------------------------------------------------
        lifestyle_areas = (
            ChillStoriesTopics().topics.get("meta", {}).get("lifestyle_areas", [])
        )

        t3_results = await asyncio.gather(*[
            self._generate_one(
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
            for area in lifestyle_areas
        ], return_exceptions=True)

        for result in t3_results:
            if isinstance(result, list):
                all_topics.extend(result)

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
