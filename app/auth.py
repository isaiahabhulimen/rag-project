import hashlib
import secrets

from fastapi import Header, HTTPException

from config import rag_api_key


def verify_api_key(
    authorization: str = Header(default=None)
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme"
        )

    provided_key = authorization.split(" ", 1)[1]

    if not secrets.compare_digest(
        provided_key,
        rag_api_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )

    client_id = hashlib.sha256(
        provided_key.encode()
    ).hexdigest()

    return client_id