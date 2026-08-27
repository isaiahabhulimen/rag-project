import hashlib

import pytest

from app.auth import verify_api_key


def test_missing_authorization():
    with pytest.raises(Exception) as exc_info:
        verify_api_key(None)

    assert exc_info.value.status_code == 401


def test_invalid_api_key():
    with pytest.raises(Exception) as exc_info:
        verify_api_key("Bearer wrong-key")

    assert exc_info.value.status_code == 401


def test_invalid_authentication_scheme():
    with pytest.raises(Exception) as exc_info:
        verify_api_key("Basic some-key")

    assert exc_info.value.status_code == 401


def test_valid_api_key():
    from config import rag_api_key

    result = verify_api_key(
        f"Bearer {rag_api_key}"
    )

    expected_client_id = hashlib.sha256(
        rag_api_key.encode()
    ).hexdigest()

    assert result == expected_client_id