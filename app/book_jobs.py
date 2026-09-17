import uuid

from datetime import datetime, timezone

from storage import ObjectStorage

storage = ObjectStorage()


def _job_key(job_id):
    return f"jobs/{job_id}.json"


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def create_job(filename, object_key, job_id=None, file_hash=None):
    if job_id is None:
        job_id = str(uuid.uuid4())

    job = {
        "job_id": job_id,
        "filename": filename,
        "object_key": object_key,
        "file_hash": file_hash,
        "status": "queued",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "error": None,
        "last_completed_page": 0,
        "text_index": 0,
        "image_index": 0,
    }

    storage.put_json(_job_key(job_id), job)

    return job


def get_job(job_id):
    try:
        return storage.get_json(_job_key(job_id))
    except Exception:
        return None


def update_job(job_id, status, error=None):
    job = get_job(job_id)

    if not job:
        return None

    job["status"] = status
    job["error"] = error
    job["updated_at"] = _timestamp()

    storage.put_json(_job_key(job_id), job)

    return job


def update_checkpoint(job_id, last_completed_page, text_index, image_index):
    job = get_job(job_id)

    if not job:
        return None

    job["last_completed_page"] = last_completed_page

    job["text_index"] = text_index
    job["image_index"] = image_index
    job["updated_at"] = _timestamp()

    storage.put_json(_job_key(job_id), job)

    return job


def find_job_by_hash(file_hash):
    if not file_hash:
        return None

    jobs = list_jobs()

    matching_jobs = [job for job in jobs if job.get("file_hash") == file_hash]

    if not matching_jobs:
        return None

    matching_jobs.sort(key=lambda job: job.get("updated_at", ""), reverse=True)

    return matching_jobs[0]


def list_jobs():
    keys = storage.list_objects("jobs/")

    jobs = []

    for key in keys:
        if not key.endswith(".json"):
            continue

        try:
            jobs.append(storage.get_json(key))
        except Exception:
            continue

    return jobs
