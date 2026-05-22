"""
Langfuse tracing utility for LangGraph workflows (langfuse v3/v4 SDK).

Langfuse is free and open-source (MIT). Free cloud tier at cloud.langfuse.com.
Self-hostable via Docker if needed.

Setup:
1. Sign up at https://cloud.langfuse.com  (free)
2. Create a project and copy your Public Key + Secret Key
3. Add to .env:
       LANGFUSE_ENABLED=true
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...
       LANGFUSE_HOST=https://cloud.langfuse.com   # or self-hosted URL

Usage in workflow invocations:
    from src.utils.tracing import build_trace_config

    config = {
        "configurable": {...},
        **build_trace_config(
            name="WF2-story",
            metadata={"story_id": sid},
            tags=["wf2", "story"],
            session_id=sid,
        ),
    }
    await workflow.ainvoke(state, config=config)

What you see in the Langfuse dashboard:
- Each workflow run as a trace with a name and timeline
- Every LLM call (Gemini, FLUX) as a nested span with input/output/latency/cost
- Every LangGraph node as a named step
- Errors highlighted inline
- Full metadata (story_id, theme, age, language) on every trace
"""

import os
from typing import Optional
from .config import get_settings
from .logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

# Cached Langfuse client instance (None when disabled or keys missing)
_langfuse_client = None


def _get_client():
    """Lazy-init the Langfuse client. Returns None if disabled or misconfigured."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if not settings.LANGFUSE_ENABLED:
        return None
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning("[Tracing] LANGFUSE_ENABLED=true but keys are missing — tracing disabled")
        return None

    try:
        # v3/v4 SDK reads credentials from env vars when the global client is
        # constructed; setting them here lets the CallbackHandler (which takes
        # no constructor args in v3+) find the same configuration.
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
        os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_HOST)

        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info(f"[Tracing] Langfuse connected: {settings.LANGFUSE_HOST}")
        return _langfuse_client
    except ImportError:
        logger.warning("[Tracing] langfuse package not installed — pip install langfuse")
        return None
    except Exception as e:
        logger.error(f"[Tracing] Langfuse init failed: {e}")
        return None


def _get_callback_handler():
    """Build a single Langfuse CallbackHandler for the current process."""
    client = _get_client()
    if client is None:
        return None

    # v3/v4 SDK — handler lives in langfuse.langchain and requires the
    # `langchain` package (not just langchain-core) to be installed.
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except ModuleNotFoundError as e:
        if "langchain" in str(e):
            logger.error(
                "[Tracing] langfuse langchain integration needs the `langchain` "
                "package: pip install langchain"
            )
            return None
        # Different missing module — try the v2 fallback below.
    except Exception as e:
        logger.error(f"[Tracing] Failed to create v3/v4 callback handler: {e}")
        return None

    # v2.x fallback
    try:
        from langfuse.callback import CallbackHandler  # type: ignore
        return CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as e:
        logger.error(f"[Tracing] Failed to import CallbackHandler (v2 fallback): {e}")
        return None


def build_trace_config(
    name: str,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Build the trace-related portion of a LangChain RunnableConfig.

    In langfuse v3/v4, the CallbackHandler reads these special keys from the
    LangChain RunnableConfig:
        run_name                  → trace name
        metadata.langfuse_session_id → groups traces into a session
        metadata.langfuse_tags    → tags shown in the dashboard
        metadata (other keys)     → free-form trace metadata

    Returns an empty dict when tracing is disabled, so callers can always
    spread the result into their config without branching:

        config = {"configurable": {...}, **build_trace_config(name="WF2-story", ...)}
    """
    handler = _get_callback_handler()
    if handler is None:
        return {}

    md = dict(metadata or {})
    if session_id:
        md["langfuse_session_id"] = session_id
    if tags:
        md["langfuse_tags"] = list(tags)

    return {
        "callbacks": [handler],
        "metadata":  md,
        "tags":      list(tags) if tags else [],
        "run_name":  name,
    }


def get_trace_callbacks(
    name: str,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> list:
    """
    Back-compat: returns just the callbacks list.

    NOTE: In langfuse v3/v4 the trace name, tags, and session_id are read from
    the LangChain RunnableConfig (run_name / metadata), not the handler. If you
    want those to appear in the dashboard, use build_trace_config() instead and
    spread it into your config dict.
    """
    cfg = build_trace_config(name=name, metadata=metadata, tags=tags, session_id=session_id)
    return cfg.get("callbacks", [])


def flush():
    """
    Flushes any pending Langfuse events to the server.
    Call this at app shutdown to avoid losing the last few traces.
    """
    client = _get_client()
    if client:
        try:
            client.flush()
        except Exception as e:
            logger.error(f"[Tracing] Flush failed: {e}")
