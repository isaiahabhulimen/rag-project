import uuid
from datetime import datetime, timezone

from storage import ObjectStorage


storage = ObjectStorage()


def _job_key(job_id):
    return f"jobs/{job_id}.json"


def _timestamp():
    return datetime.now(
        timezone.utc
    ).isoformat()


def create_job(
    filename,
    object_key,
    job_id=None
):
    if job_id is None:
        job_id = str(uuid.uuid4())

    job = {
        "job_id": job_id,
        "filename": filename,
        "object_key": object_key,
        "status": "queued",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "error": None
    }

    storage.put_json(
        _job_key(job_id),
        job
    )

    return job


def get_job(job_id):
    try:
        return storage.get_json(
            _job_key(job_id)
        )
    except Exception:
        return None


def update_job(
    job_id,
    status,
    error=None
):
    job = get_job(job_id)

    if not job:
        return None

    job["status"] = status
    job["error"] = error
    job["updated_at"] = _timestamp()

    storage.put_json(
        _job_key(job_id),
        job
    )

    return job


def list_jobs():
    keys = storage.list_objects(
        "jobs/"
    )

    jobs = []

    for key in keys:
        if not key.endswith(".json"):
            continue

        try:
            jobs.append(
                storage.get_json(key)
            )
        except Exception:
            continue

    return jobs