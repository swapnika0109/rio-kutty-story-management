"""
Unit tests for AIService fallback and resilience wiring.
"""

import pytest
from unittest.mock import AsyncMock

from src.services.ai_service import AIService


class TestAIServiceFallback:
    @pytest.mark.asyncio
    async def test_returns_primary_response_when_primary_succeeds(self):
        service = AIService()
        service._generate_with_primary = AsyncMock(return_value='{"ok": true, "source": "primary"}')
        service._generate_with_fallback = AsyncMock(return_value='{"ok": true, "source": "fallback"}')

        result = await service.generate_content("test prompt")

        assert '"source": "primary"' in result
        service._generate_with_primary.assert_awaited_once()
        service._generate_with_fallback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_fallback_when_primary_fails(self):
        service = AIService()
        # Ensure models are distinct so the fallback path is reachable
        service.fallback_model_name = "gemini-2.0-flash-001"
        service._generate_with_primary = AsyncMock(side_effect=RuntimeError("primary failed"))
        service._generate_with_fallback = AsyncMock(return_value='{"ok": true, "source": "fallback"}')

        result = await service.generate_content("test prompt")

        assert '"source": "fallback"' in result
        service._generate_with_primary.assert_awaited_once()
        service._generate_with_fallback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_fallback_same_as_primary_and_primary_fails(self):
        service = AIService()
        service.fallback_model_name = service.model_name
        service._generate_with_primary = AsyncMock(side_effect=RuntimeError("primary failed"))

        with pytest.raises(RuntimeError, match="No distinct fallback model available"):
            await service.generate_content("test prompt")
