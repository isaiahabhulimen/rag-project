import hashlib
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import rag_api_key

security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    provided_key = credentials.credentials

    if not secrets.compare_digest(provided_key, rag_api_key):
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )

    client_id = hashlib.sha256(provided_key.encode()).hexdigest()

    return client_id
