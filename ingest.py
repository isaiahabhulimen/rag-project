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
    get_job,
    update_job,
    update_checkpoint
)

from config import (
    model_name,
    worker_api_url,
    worker_poll_seconds,
    worker_stale_minutes
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

            refreshed_job = get_job(
                job["job_id"]
            )

            if refreshed_job:
                return refreshed_job

            return job

    return None


def process_job(job):

    job_id = job["job_id"]

    current_job = get_job(
        job_id
    )

    if not current_job:
        print(
            f"Job no longer exists: "
            f"{job_id}"
        )
        return

    job = current_job

    book_name = job["filename"]
    object_key = job["object_key"]
    file_hash = job.get("file_hash")

    last_completed_page = job.get(
        "last_completed_page",
        0
    )

    text_index = job.get(
        "text_index",
        0
    )

    image_index = job.get(
        "image_index",
        0
    )

    print(
        f"\n===== Job: {job_id} ====="
    )

    print(
        f"Book: {book_name}"
    )

    print(
        f"File hash: {file_hash}"
    )

    print(
        f"Last completed page: "
        f"{last_completed_page}"
    )

    print(
        f"Text index: {text_index}"
    )

    print(
        f"Image index: {image_index}"
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

        if not file_hash:
            print(
                "No stored file hash found. "
                "Using PDF hash calculated during indexing."
            )

        if last_completed_page == 0:

            print(
                "New indexing job."
            )

            print(
                "Removing previous index..."
            )

            delete_existing_index(
                book_name
            )

            index_started = True
            start_page = 1

        else:

            print(
                f"Resuming indexing from page "
                f"{last_completed_page + 1}..."
            )

            start_page = (
                last_completed_page + 1
            )

            index_started = True

        def checkpoint_callback(
            page_number,
            current_text_index,
            current_image_index
        ):

            nonlocal last_completed_page
            nonlocal text_index
            nonlocal image_index

            updated_job = update_checkpoint(
                job_id,
                page_number,
                current_text_index,
                current_image_index
            )

            if updated_job:
                last_completed_page = (
                    updated_job[
                        "last_completed_page"
                    ]
                )

                text_index = (
                    updated_job[
                        "text_index"
                    ]
                )

                image_index = (
                    updated_job[
                        "image_index"
                    ]
                )

            print(
                f"Checkpoint saved | "
                f"page={page_number} | "
                f"text_index={current_text_index} | "
                f"image_index={current_image_index}"
            )

        print(
            "Starting indexing..."
        )

        result = index_book(
            pdf_path=temp_path,
            book_name=book_name,
            job_id=job_id,
            model=model,
            store_batch=store_batch,
            start_page=start_page,
            text_index=text_index,
            image_index=image_index,
            checkpoint_callback=checkpoint_callback
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

        if (
            last_completed_page == 0
            and index_started
        ):

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

        else:

            print(
                "Checkpoint exists. "
                "Keeping partial index for resume."
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
            os.remove(
                temp_path
            )


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