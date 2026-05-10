"""
Integration tests for the activity workflow.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json


class TestActivityWorkflow:
    """Integration tests for the full activity workflow."""
    
    @pytest.fixture
    def mock_all_services(self):
        """Mock all external services."""
        import src.workflows.activity_workflow as _wf_module

        mock_firestore = MagicMock()
        mock_firestore.check_if_activity_exists = AsyncMock(return_value=False)
        mock_firestore.save_activity = AsyncMock()

        mock_storage = MagicMock()
        mock_storage.upload_file = AsyncMock(return_value="images/test.png")

        with patch.object(_wf_module, "firestore_service", mock_firestore), \
             patch.object(_wf_module, "storage_service", mock_storage) if hasattr(_wf_module, "storage_service") else patch("src.workflows.activity_workflow.StorageBucketService"), \
             patch("src.agents.activities.mcq_agent.AIService") as MockAI1, \
             patch("src.agents.activities.art_agent.AIService") as MockAI2, \
             patch("src.agents.activities.moral_agent.AIService") as MockAI3, \
             patch("src.agents.activities.science_agent.AIService") as MockAI4:

            # Configure AI mocks
            for mock_ai in [MockAI1, MockAI2, MockAI3, MockAI4]:
                instance = mock_ai.return_value
                instance.generate_content = AsyncMock(return_value=json.dumps([
                    {"question": "Test?", "options": ["A", "B"], "correct": "A"}
                ]))
                instance.generate_image = AsyncMock(return_value=b"image_bytes")

            yield {
                "firestore": mock_firestore,
                "storage": mock_storage
            }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_skips_existing_activities(self, mock_all_services):
        """Test that workflow skips activities that already exist."""
        # Make MCQ already exist
        async def check_exists(story_id, activity_type):
            return activity_type == 'mcq'
        
        mock_all_services["firestore"].check_if_activity_exists = AsyncMock(
            side_effect=check_exists
        )
        
        # Import here to use mocked services
        from src.workflows.activity_workflow import route_start
        
        state = {"activities": {}, "completed": [], "errors": {}, "retry_count": {}}
        config = {
            "configurable": {
                "thread_id": "test",
                "story_id": "test_123"
            }
        }
        
        nodes_to_run = await route_start(state, config)
        
        # MCQ should be skipped
        assert "gen_mcq" not in nodes_to_run
        # Others should run
        assert "gen_art" in nodes_to_run
        assert "gen_mor" in nodes_to_run
        assert "gen_sci" in nodes_to_run
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_retry_logic_retries_on_validation_failure(self):
        """Test that retry logic correctly determines retry/next/fail."""
        from src.workflows.activity_workflow import create_retry_logic
        
        retry_fn = create_retry_logic("mcq")
        
        # Case 1: Activity completed - should proceed
        state_completed = {"completed": ["mcq"], "errors": {}, "retry_count": {"mcq": 1}}
        assert retry_fn(state_completed) == "next"
        
        # Case 2: Activity not completed, retries available - should retry
        state_retry = {"completed": [], "errors": {}, "retry_count": {"mcq": 1}}
        assert retry_fn(state_retry) == "retry"
        
        # Case 3: Activity not completed, max retries reached - should fail
        state_fail = {"completed": [], "errors": {}, "retry_count": {"mcq": 4}}
        assert retry_fn(state_fail) == "fail"
        
        # Case 4: Activity has error - should fail immediately
        state_error = {"completed": [], "errors": {"mcq": "Error"}, "retry_count": {"mcq": 1}}
        assert retry_fn(state_error) == "fail"
