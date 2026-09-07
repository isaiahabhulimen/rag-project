import json
import os
import tempfile
import time
import urllib.parse
import urllib.request

from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer

from app.book_jobs import (
    list_jobs,
    update_job,
)

from config import (
    model_name,
    worker_api_url,
    worker_poll_seconds,
    worker_stale_minutes,
)

from indexer import index_book
from storage import ObjectStorage


storage = ObjectStorage()

worker_token = os.getenv(
    "WORKER_TOKEN"
)

if not worker_token:
    raise RuntimeError(
        "WORKER_TOKEN environment variable "
        "is not configured"
    )


model = SentenceTransformer(
    model_name,
    device="cpu"
)


def api_request(
    method,
    path,
    payload
):
    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{worker_api_url}{path}",
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method=method
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            response_data = (
                response.read()
                .decode("utf-8")
            )

            return json.loads(
                response_data
            )

    except Exception as e:
        raise RuntimeError(
            f"Worker API request failed: "
            f"{method} {path} | {str(e)}"
        ) from e


def delete_existing_index(
    book_name
):
    encoded_name = urllib.parse.quote(
        book_name,
        safe=""
    )

    api_request(
        "DELETE",
        f"/internal/index/{encoded_name}",
        {
            "worker_token": worker_token
        }
    )


def store_batch(
    batch_type,
    book_name,
    file_hash,
    items
):
    api_request(
        "POST",
        "/internal/index-batch",
        {
            "worker_token": worker_token,
            "batch_type": batch_type,
            "book_name": book_name,
            "file_hash": file_hash,
            "items": items
        }
    )


def is_stale(job):
    if job.get("status") != "processing":
        return False

    updated_at = datetime.fromisoformat(
        job["updated_at"]
    )

    age = (
        datetime.now(timezone.utc)
        - updated_at
    )

    return (
        age.total_seconds()
        > worker_stale_minutes * 60
    )


def find_next_job():
    jobs = list_jobs()

    for job in jobs:

        if job.get("status") == "queued":
            return job

        if is_stale(job):
            print(
                f"Recovering stale job: "
                f"{job['job_id']}"
            )

            update_job(
                job["job_id"],
                "queued"
            )

            return job

    return None


def process_job(job):
    job_id = job["job_id"]
    book_name = job["filename"]
    object_key = job["object_key"]

    print(
        f"\n===== Job: {job_id} ====="
    )

    print(
        f"Book: {book_name}"
    )

    update_job(
        job_id,
        "processing"
    )

    temp_path = None
    index_started = False

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_path = temp_file.name

        print(
            "Downloading book..."
        )

        storage.download_file(
            object_key,
            temp_path
        )

        print(
            "Removing previous index..."
        )

        delete_existing_index(
            book_name
        )

        index_started = True

        print(
            "Starting indexing..."
        )

        result = index_book(
            pdf_path=temp_path,
            book_name=book_name,
            job_id=job_id,
            model=model,
            store_batch=store_batch
        )

        update_job(
            job_id,
            "completed"
        )

        print(
            f"Job completed: {job_id}"
        )

        print(
            f"Text chunks: "
            f"{result['text_chunks']}"
        )

        print(
            f"Image chunks: "
            f"{result['image_chunks']}"
        )

    except Exception as e:

        print(
            f"Job failed: {job_id} | "
            f"{str(e)}"
        )

        if index_started:
            try:
                print(
                    "Removing partial index..."
                )

                delete_existing_index(
                    book_name
                )

                print(
                    "Partial index removed."
                )

            except Exception as cleanup_error:
                print(
                    "Failed to remove partial "
                    f"index: {cleanup_error}"
                )

        update_job(
            job_id,
            "failed",
            str(e)
        )

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)


def main():

    print(
        "Starting book ingestion worker..."
    )

    print(
        f"Polling every "
        f"{worker_poll_seconds} seconds"
    )

    while True:

        try:
            job = find_next_job()

            if job:
                process_job(job)

            else:
                time.sleep(
                    worker_poll_seconds
                )

        except Exception as e:

            print(
                f"Worker loop error: "
                f"{str(e)}"
            )

            time.sleep(
                worker_poll_seconds
            )


if __name__ == "__main__":
    main()