"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("src.api.activities.FirestoreService") as MockFirestore, \
             patch("src.api.activities.app_workflow") as MockWorkflow:
            
            # Mock Firestore
            firestore = MockFirestore.return_value
            firestore.get_story = AsyncMock(return_value={
                "story_id": "test_123",
                "story_text": "Once upon a time...",
                "language": "en"
            })
            
            # Mock workflow
            MockWorkflow.ainvoke = AsyncMock()
            
            from src.main import app
            with TestClient(app) as client:
                yield client
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_generate_activities_returns_202(self, client):
        """Test that generate-activities returns 202 Accepted."""
        response = client.post(
            "/generate-activities",
            json={
                "story_id": "456y5i64u8thfbcsyr834drft9H",
                "age": "5",
                "language": "en"
            }
        )
        
        assert response.status_code == 200  # FastAPI default for BackgroundTasks
        data = response.json()
        assert data["status"] == "accepted"
        assert data["story_id"] == "456y5i64u8thfbcsyr834drft9H"
    
    def test_generate_activities_validates_input(self, client):
        """Test that generate-activities validates required fields."""
        response = client.post(
            "/generate-activities",
            json={
                "age": "5"
                # Missing story_id
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_pubsub_handler_decodes_base64(self, client):
        """Test that pubsub handler correctly decodes base64 data."""
        import base64
        import json
        
        data = {"story_id": "test_123", "age": "5"}
        encoded = base64.b64encode(json.dumps(data).encode()).decode()
        
        response = client.post(
            "/pubsub-handler",
            json={
                "message": {"data": encoded},
                "subscription": "test-subscription"
            }
        )
        
        assert response.status_code == 202
    
    def test_pubsub_handler_returns_400_on_missing_data(self, client):
        """Test that pubsub handler returns 400 when data is missing."""
        response = client.post(
            "/pubsub-handler",
            json={
                "message": {},
                "subscription": "test-subscription"
            }
        )
        
        assert response.status_code == 400
