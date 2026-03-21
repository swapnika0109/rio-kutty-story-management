"""
Unit tests for MCQ Agent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.activities.mcq_agent import MCQAgent


class TestMCQAgent:
    """Tests for MCQAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create MCQAgent instance with mocked AI service."""
        with patch("src.agents.activities.mcq_agent.AIService") as MockAI:
            instance = MockAI.return_value
            instance.generate_content = AsyncMock()
            agent = MCQAgent()
            agent.ai_service = instance
            yield agent
    
    @pytest.fixture
    def sample_state(self):
        """Sample state for testing."""
        return {
            "story_text": "Once upon a time, a brave rabbit saved the forest.",
            "age": 5,
            "language": "English",
            "activities": {},
            "completed": [],
            "errors": {}
        }
    
    @pytest.mark.asyncio
    async def test_generate_returns_mcq_activities(self, agent, sample_state):
        """Test that generate() returns properly structured MCQ data."""
        mock_response = json.dumps([
            {"question": "Who was brave?", "options": ["Rabbit", "Fox", "Bear"], "correct": "Rabbit"},
            {"question": "What did the rabbit save?", "options": ["Forest", "River", "Mountain"], "correct": "Forest"},
            {"question": "What happened?", "options": ["Adventure", "Sleep", "Eat"], "correct": "Adventure"}
        ])
        agent.ai_service.generate_content.return_value = mock_response
        
        result = await agent.generate(sample_state)
        
        assert "activities" in result
        assert "mcq" in result["activities"]
        assert "completed" in result
        assert "mcq" in result["completed"]
        assert len(result["activities"]["mcq"]) == 3
    
    @pytest.mark.asyncio
    async def test_generate_handles_json_with_markdown(self, agent, sample_state):
        """Test that generate() handles JSON wrapped in markdown code blocks."""
        mock_response = """```json
        [{"question": "Test?", "options": ["A", "B", "C"], "correct": "A"}]
        ```"""
        agent.ai_service.generate_content.return_value = mock_response
        
        result = await agent.generate(sample_state)
        
        assert "activities" in result
        assert "mcq" in result["activities"]
    
    @pytest.mark.asyncio
    async def test_generate_returns_error_on_failure(self, agent, sample_state):
        """Test that generate() returns error dict on exception."""
        agent.ai_service.generate_content.side_effect = Exception("API Error")
        
        result = await agent.generate(sample_state)
        
        assert "errors" in result
        assert "mcq" in result["errors"]
        assert "API Error" in result["errors"]["mcq"]
    
    @pytest.mark.asyncio
    async def test_generate_handles_invalid_json(self, agent, sample_state):
        """Test that generate() handles invalid JSON responses."""
        agent.ai_service.generate_content.return_value = "not valid json at all"
        
        result = await agent.generate(sample_state)
        
        assert "errors" in result
        assert "mcq" in result["errors"]
    
    def test_uses_prompt_registry(self):
        """Test that MCQAgent uses the prompt registry."""
        with patch("src.agents.activities.mcq_agent.get_registry") as mock_registry:
            mock_registry.return_value.get_prompt.return_value = "test prompt"
            
            with patch("src.agents.activities.mcq_agent.AIService"):
                agent = MCQAgent(prompt_version="v1")
                assert agent.prompt_version == "v1"
