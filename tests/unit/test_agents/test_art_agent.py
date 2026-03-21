"""
Unit tests for Art Agent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch

from src.agents.activities.art_agent import ArtAgent


class TestArtAgent:
    """Tests for ArtAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create ArtAgent instance with mocked AI service."""
        with patch("src.agents.activities.art_agent.AIService") as MockAI:
            instance = MockAI.return_value
            instance.generate_content = AsyncMock()
            instance.generate_image = AsyncMock(return_value=b"fake_image")
            agent = ArtAgent()
            agent.ai_service = instance
            yield agent
    
    @pytest.fixture
    def sample_state(self):
        return {
            "story_text": "A colorful butterfly visited the garden.",
            "age": 6,
            "language": "English",
            "activities": {},
            "completed": [],
            "errors": {}
        }
    
    @pytest.mark.asyncio
    async def test_generate_returns_art_activity_with_image(self, agent, sample_state):
        """Test that generate() returns art activity with generated image."""
        mock_response = json.dumps({
            "title": "Butterfly Craft",
            "age_appropriateness": "Great for 6-year-olds",
            "materials": ["paper", "paint"],
            "steps": ["Fold paper", "Paint wings", "Add antennae"],
            "image_generation_prompt": "A colorful butterfly craft"
        })
        agent.ai_service.generate_content.return_value = mock_response
        
        result = await agent.generate(sample_state)
        
        assert "activities" in result
        assert "art" in result["activities"]
        assert result["activities"]["art"]["image"] == b"fake_image"
        assert "completed" in result
        assert "art" in result["completed"]
    
    @pytest.mark.asyncio
    async def test_generate_continues_if_image_fails(self, agent, sample_state):
        """Test that art activity is still saved even if image generation fails."""
        mock_response = json.dumps({
            "title": "Paper Craft",
            "materials": ["paper"],
            "steps": ["Cut", "Fold"],
            "image_generation_prompt": "A craft"
        })
        agent.ai_service.generate_content.return_value = mock_response
        agent.ai_service.generate_image.return_value = None  # Image failed
        
        result = await agent.generate(sample_state)
        
        assert "activities" in result
        assert "art" in result["activities"]
        assert result["activities"]["art"]["image"] is None
    
    @pytest.mark.asyncio
    async def test_generate_returns_error_on_ai_failure(self, agent, sample_state):
        """Test that generate() returns error on AI failure."""
        agent.ai_service.generate_content.side_effect = Exception("AI Error")
        
        result = await agent.generate(sample_state)
        
        assert "errors" in result
        assert "art" in result["errors"]
