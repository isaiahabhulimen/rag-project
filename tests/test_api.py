import sys

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


# Prevent the real ML models from loading during API tests.
mock_models = MagicMock()
mock_models.model = MagicMock()
mock_models.cross_encoder = MagicMock()
sys.modules["models"] = mock_models


# Prevent API tests from requiring real object storage.
mock_storage = MagicMock()
mock_storage.ObjectStorage = MagicMock
sys.modules["storage"] = mock_storage


from fastapi.testclient import TestClient

from api import app

from config import rag_api_key


client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == "RAG API is running."


def test_ask_without_authentication():

    response = client.post(
        "/ask",
        json={"question": "Who is Arkad?"}
    )

    assert response.status_code == 401


def test_ask_with_invalid_authentication():

    response = client.post(
        "/ask",
        headers={"Authorization": "Bearer wrong-key"},
        json={"question": "Who is Arkad?"}
    )

    assert response.status_code == 401


def test_ask_with_invalid_question():

    response = client.post(
        "/ask",
        headers={"Authorization": f"Bearer {rag_api_key}"},
        json={"question": ""}
    )

    assert response.status_code == 422


def test_ask_with_valid_authentication():

    with patch(
        "api.ask_question",
        return_value="Arkad is the richest man in Babylon."
    ) as mock_ask:

        response = client.post(
            "/ask",
            headers={"Authorization": f"Bearer {rag_api_key}"},
            json={"question": "Who is Arkad?"}
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "Who is Arkad?"

    assert data["answer"] == "Arkad is the richest man in Babylon."

    mock_ask.assert_called_once()


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["database"] == "connected"

    assert data["embedding_model"] == "loaded"

    assert data["cross_encoder"] == "loaded"