import hashlib
import os
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse

from app.exceptions import LLMError, RetrievalError
from app.schemas import (
    QuestionRequest,
    QuestionResponse,
    RootResponse,
    HealthResponse,
)
from app.rag_service import ask_question
from app.state import app_context
from logger import logger
from app.auth import verify_api_key
from app.rate_limiter import check_rate_limit
from storage import ObjectStorage
from app.book_jobs import (
    create_job,
    get_job,
    update_job,
    find_job_by_hash,
)
from config import worker_token

app = FastAPI()
storage = ObjectStorage()


@app.exception_handler(LLMError)
def llm_error_handler(request, exc):
    logger.error(f"LLM error: {str(exc)}")
    return JSONResponse(
        status_code=502,
        content={"detail": "LLM service unavailable"},
    )


@app.exception_handler(RetrievalError)
def retrieval_error_handler(request, exc):
    logger.error(f"Retrieval error: {str(exc)} | " f"Cause: {repr(exc.__cause__)}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Retrieval service unavailable"},
    )


@app.get("/", response_model=RootResponse)
def root():
    return {"message": "RAG API is running."}


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        document_count = app_context.text_collection.count()

        return {
            "status": "healthy",
            "database": "connected",
            "documents": document_count,
            "embedding_model": "loaded",
            "cross_encoder": "loaded",
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable",
        )


@app.post("/ask", response_model=QuestionResponse)
def ask(
    request: QuestionRequest,
    authenticated: str = Depends(verify_api_key),
):
    logger.info(f"Question received | " f"length={len(request.question)}")

    if not check_rate_limit(authenticated):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )

    answer = ask_question(
        question=request.question,
        search_all="yes",
        selected_book=None,
        context=app_context,
    )

    logger.info("Answer generated successfully")

    return {
        "question": request.question,
        "answer": answer,
    }


@app.post("/books/upload")
def upload_book(
    file: UploadFile = File(...),
    authenticated: str = Depends(verify_api_key),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDFs are supported",
        )

    job_id = str(uuid.uuid4())
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_path = temp_file.name
            hasher = hashlib.sha256()

            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                hasher.update(chunk)
                temp_file.write(chunk)

        file_hash = hasher.hexdigest()

        existing_job = find_job_by_hash(file_hash)

        if existing_job:
            existing_status = existing_job.get("status")

            # IMPORTANT:
            # A completed job in object storage does not automatically
            # mean this current API/Chroma instance has the book indexed.
            #
            # Verify the current Chroma database before returning
            # "already_indexed".

            if existing_status == "completed":
                indexed_book = app_context.text_collection.get(
                    where={"source": existing_job["filename"]},
                    limit=1,
                )

                indexed_ids = indexed_book.get("ids", [])

                if indexed_ids:
                    logger.info(
                        f"Duplicate book upload skipped | "
                        f"filename={file.filename} | "
                        f"file_hash={file_hash} | "
                        f"existing_job={existing_job['job_id']} | "
                        f"current_index=present"
                    )

                    return {
                        "job_id": existing_job["job_id"],
                        "filename": existing_job["filename"],
                        "status": "completed",
                        "already_indexed": True,
                    }

                # The job is completed in object storage, but the
                # current Chroma database does not contain the book.
                #
                # This can happen after moving the API to a new
                # deployment with a new/empty persistent database.

                logger.info(
                    f"Completed job found but book is not indexed "
                    f"in current database | "
                    f"filename={file.filename} | "
                    f"file_hash={file_hash} | "
                    f"old_job={existing_job['job_id']} | "
                    f"creating_new_job=True"
                )

            elif existing_status in {"queued", "processing"}:
                logger.info(
                    f"Existing book job found | "
                    f"filename={file.filename} | "
                    f"file_hash={file_hash} | "
                    f"existing_job={existing_job['job_id']} | "
                    f"status={existing_status}"
                )

                return {
                    "job_id": existing_job["job_id"],
                    "filename": existing_job["filename"],
                    "status": existing_status,
                    "already_indexed": False,
                }

            elif existing_status == "failed":
                logger.info(
                    f"Previous book job failed | "
                    f"filename={file.filename} | "
                    f"file_hash={file_hash} | "
                    f"old_job={existing_job['job_id']} | "
                    f"creating_new_job=True"
                )

        object_key = f"books/{file_hash}/{file.filename}"

        storage.upload_file(
            temp_path,
            object_key,
        )

        job = create_job(
            filename=file.filename,
            object_key=object_key,
            job_id=job_id,
            file_hash=file_hash,
        )

        logger.info(
            f"Book queued | "
            f"job_id={job_id} | "
            f"filename={file.filename} | "
            f"file_hash={file_hash}"
        )

        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "queued",
            "already_indexed": False,
        }

    except Exception as e:
        logger.error(f"Book upload failed | " f"job_id={job_id} | " f"error={str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Book upload failed",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/books/{job_id}")
def book_status(
    job_id: str,
    authenticated: str = Depends(verify_api_key),
):
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return job


@app.post("/internal/index-batch")
def index_batch(payload: dict):
    if not worker_token:
        raise HTTPException(
            status_code=503,
            detail="Worker authentication is not configured",
        )

    token = payload.get("worker_token")

    if token != worker_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid worker credentials",
        )

    book_name = payload.get("book_name")
    file_hash = payload.get("file_hash")
    batch_type = payload.get("batch_type")
    items = payload.get("items", [])

    if not book_name:
        raise HTTPException(
            status_code=400,
            detail="book_name is required",
        )

    if not file_hash:
        raise HTTPException(
            status_code=400,
            detail="file_hash is required",
        )

    if batch_type not in {"text", "image"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid batch_type",
        )

    if not items:
        return {
            "status": "accepted",
            "count": 0,
        }

    collection = (
        app_context.text_collection
        if batch_type == "text"
        else app_context.image_collection
    )

    item_ids = [item["id"] for item in items]

    existing = collection.get(ids=item_ids)

    existing_ids = set(existing.get("ids", []))

    new_items = [item for item in items if item["id"] not in existing_ids]

    if new_items:
        collection.add(
            ids=[item["id"] for item in new_items],
            documents=[item["document"] for item in new_items],
            embeddings=[item["embedding"] for item in new_items],
            metadatas=[item["metadata"] for item in new_items],
        )

    skipped_count = len(items) - len(new_items)

    logger.info(
        f"Index batch stored | "
        f"book={book_name} | "
        f"type={batch_type} | "
        f"stored={len(new_items)} | "
        f"already_exists={skipped_count}"
    )

    return {
        "status": "accepted",
        "count": len(new_items),
        "already_exists": skipped_count,
    }


@app.delete("/internal/index/{book_name}")
def delete_book_index(
    book_name: str,
    payload: dict,
):
    if not worker_token:
        raise HTTPException(
            status_code=503,
            detail="Worker authentication is not configured",
        )

    token = payload.get("worker_token")

    if token != worker_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid worker credentials",
        )

    app_context.text_collection.delete(where={"source": book_name})

    app_context.image_collection.delete(where={"source": book_name})

    logger.info(f"Book index deleted | " f"book={book_name}")

    return {"status": "deleted"}
