from app.rate_limiter import (
    check_rate_limit,
    request_history,
    REQUEST_LIMIT,
)


def test_requests_within_limit_are_allowed():
    client_id = "test-client-within-limit"

    for _ in range(REQUEST_LIMIT):
        assert check_rate_limit(client_id) is True


def test_request_above_limit_is_blocked():
    client_id = "test-client-above-limit"

    for _ in range(REQUEST_LIMIT):
        assert check_rate_limit(client_id) is True

    assert check_rate_limit(client_id) is False
