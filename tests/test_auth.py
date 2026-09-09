import hashlib

import pytest

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from app.auth import verify_api_key
from api import app


client = TestClient(app)


def test_missing_authorization():

    response = client.post(
        "/ask",
        json={"question": "Who is Arkad?"}
    )

    assert response.status_code == 401


def test_invalid_api_key():

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="wrong-key"
    )

    with pytest.raises(HTTPException) as exc_info:

        verify_api_key(credentials)

    assert exc_info.value.status_code == 401


def test_invalid_authentication_scheme():

    credentials = HTTPAuthorizationCredentials(
        scheme="Basic",
        credentials="some-key"
    )

    with pytest.raises(HTTPException) as exc_info:

        verify_api_key(credentials)

    assert exc_info.value.status_code == 401


def test_valid_api_key():

    from config import rag_api_key

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=rag_api_key
    )

    result = verify_api_key(credentials)

    expected_client_id = hashlib.sha256(
        rag_api_key.encode()
    ).hexdigest()

    assert result == expected_client_id