"""Tests for active learning / Label Studio integration."""

import json
from unittest.mock import patch, MagicMock

from src.data.active_learning import LabelStudioClient, LABEL_CONFIG


def test_label_config_is_valid_xml():
    assert "<View>" in LABEL_CONFIG
    assert "REAL" in LABEL_CONFIG
    assert "FAKE" in LABEL_CONFIG
    assert "UNSURE" in LABEL_CONFIG


def test_client_initialization():
    client = LabelStudioClient(url="http://test:8080", api_token="abc123")
    assert client.url == "http://test:8080"
    assert client.api_token == "abc123"
    assert "Authorization" in client.headers


@patch("src.data.active_learning.requests.request")
def test_create_project(mock_request):
    mock_request.return_value.json.return_value = {"id": 42, "title": "Test"}
    mock_request.return_value.raise_for_status.return_value = None

    client = LabelStudioClient(url="http://test:8080", api_token="token")
    pid = client.create_project()

    assert pid == 42
    mock_request.assert_called_once()


@patch("src.data.active_learning.requests.request")
def test_get_or_create_existing(mock_request):
    mock_request.return_value.json.return_value = {
        "results": [
            {"id": 1, "title": "Other"},
            {"id": 99, "title": "AISlopDetector Review Queue"},
        ]
    }
    mock_request.return_value.raise_for_status.return_value = None

    client = LabelStudioClient(url="http://test:8080", api_token="token")
    pid = client.get_or_create_project()

    assert pid == 99


@patch("src.data.active_learning.requests.request")
def test_get_completed_annotations(mock_request):
    mock_request.return_value.json.return_value = [
        {
            "id": 1,
            "data": {"image": "/data/test.jpg"},
            "annotations": [{
                "id": 10,
                "result": [{"type": "choices", "value": {"choices": ["REAL"]}}],
                "completed_by": {"id": 5},
                "created_at": "2025-01-01",
            }],
        },
        {
            "id": 2,
            "data": {"image": "/data/test2.jpg"},
            "annotations": [],  # no annotations yet
        },
    ]
    mock_request.return_value.raise_for_status.return_value = None

    client = LabelStudioClient(url="http://test:8080", api_token="token")
    annotations = client.get_completed_annotations(1)

    assert len(annotations) == 1
    assert annotations[0]["label"] == "REAL"
    assert annotations[0]["task_id"] == 1


def test_export_labels_structure():
    """Test export_labels returns the correct structure without API calls."""
    client = LabelStudioClient(url="http://test:8080", api_token="token")

    with patch.object(client, "get_completed_annotations") as mock_get:
        mock_get.return_value = []
        result = client.export_labels(1, "/tmp/test_output")
        assert result["total"] == 0
        assert "counts" in result
