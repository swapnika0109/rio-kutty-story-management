"""
Unit tests for Validator Agent.
"""

import pytest
from src.agents.validators.validator_agent import ValidatorAgent


class TestValidatorAgent:
    """Tests for ValidatorAgent class."""

    @pytest.fixture
    def validator(self):
        return ValidatorAgent()

    # MCQ Validation Tests

    def test_validate_mcq_success(self, validator):
        """Test MCQ validation passes with valid data."""
        state = {
            "activities": {
                "mcq": [
                    {"question": "Q1?", "options": ["A", "B", "C"], "correct": "A"},
                    {"question": "Q2?", "options": ["A", "B", "C"], "correct": "B"},
                ]
            },
            "completed": [],
            "errors": {}
        }

        result = validator.validate_mcq(state)

        assert "completed" in result
        assert "mcq" in result["completed"]
        assert "errors" not in result or "mcq" not in result.get("errors", {})

    def test_validate_mcq_missing_activity(self, validator):
        """Test MCQ validation increments retry_count when activity is missing."""
        state = {
            "activities": {},
            "completed": [],
            "errors": {},
            "retry_count": {}
        }

        result = validator.validate_mcq(state)

        assert "retry_count" in result
        assert result["retry_count"]["mcq"] == 1

    # Art Validation Tests

    def test_validate_art_success(self, validator):
        """Test Art validation passes with valid data including all required fields."""
        state = {
            "activities": {
                "art": {
                    "title": "Paper Butterfly",
                    "age_appropriateness": "Great for 5-year-olds",
                    "materials": ["paper", "scissors"],
                    "steps": ["Cut", "Fold", "Decorate"],
                    "image_generation_prompt": "A paper butterfly craft",
                    "image": b"fake_image_bytes"
                }
            },
            "completed": [],
            "errors": {}
        }

        result = validator.validate_art(state)

        assert "completed" in result
        assert "art" in result["completed"]

    def test_validate_art_missing_required_fields(self, validator):
        """Test Art validation increments retry_count when required fields are missing."""
        state = {
            "activities": {
                "art": {
                    "title": "Incomplete Activity"
                    # Missing materials, steps, image_generation_prompt, image
                }
            },
            "completed": [],
            "errors": {},
            "retry_count": {}
        }

        result = validator.validate_art(state)

        assert "retry_count" in result
        assert result["retry_count"]["art"] == 1

    # Science Validation Tests

    def test_validate_science_success(self, validator):
        """Test Science validation passes with valid data including all required fields."""
        state = {
            "activities": {
                "science": [{
                    "title": "Water Experiment",
                    "age_appropriateness": "Suitable for 5-year-olds",
                    "What it Teaches": "States of matter",
                    "materials": ["water", "cup"],
                    "Instructions": ["Pour", "Observe"],
                    "image_generation_prompt": "A science experiment",
                    "image": b"fake_image_bytes"
                }]
            },
            "completed": [],
            "errors": {}
        }

        result = validator.validate_science(state)

        assert "completed" in result
        assert "science" in result["completed"]

    # Moral Validation Tests

    def test_validate_moral_success(self, validator):
        """Test Moral validation passes with valid data including all required fields."""
        state = {
            "activities": {
                "moral": [{
                    "title": "Sharing Activity",
                    "age_appropriateness": "Suitable for 5-year-olds",
                    "What it Teaches": "Empathy and sharing",
                    "materials": ["toys"],
                    "Instructions": ["Share", "Take turns"],
                    "image_generation_prompt": "Children sharing toys",
                    "image": b"fake_image_bytes"
                }]
            },
            "completed": [],
            "errors": {}
        }

        result = validator.validate_moral(state)

        assert "completed" in result
        assert "moral" in result["completed"]
