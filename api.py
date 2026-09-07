from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends
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

from indexer import index_books
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import chunk_size, chunk_overlap


indexing_status = {
    "status": "not_started",
    "error": None,
}


executor = ThreadPoolExecutor(max_workers=1)


def run_indexing():
    indexing_status["status"] = "indexing"
    indexing_status["error"] = None

    logger.info("Book indexing started")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        index_books(
            app_context.text_collection,
            app_context.image_collection,
            app_context.model,
            splitter
        )

        all_documents = app_context.text_collection.get()

        app_context.ids = all_documents["ids"]
        app_context.documents = all_documents["documents"]
        app_context.metadatas = all_documents["metadatas"]

        indexing_status["status"] = "completed"

        logger.info(
            f"Book indexing completed | "
            f"documents={len(app_context.documents)}"
        )

    except Exception as e:
        indexing_status["status"] = "failed"
        indexing_status["error"] = str(e)

        logger.error(
            f"Book indexing failed: {str(e)}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application starting")

    executor.submit(run_indexing)

    yield

    logger.info("FastAPI application shutting down")

    executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


@app.exception_handler(LLMError)
def llm_error_handler(request, exc):
    logger.error(f"LLM error: {str(exc)}")

    return JSONResponse(
        status_code=502,
        content={"detail": "LLM service unavailable"}
    )


@app.exception_handler(RetrievalError)
def retrieval_error_handler(request, exc):
    logger.error(
        f"Retrieval error: {str(exc)} | "
        f"Cause: {repr(exc.__cause__)}"
    )

    return JSONResponse(
        status_code=503,
        content={"detail": "Retrieval service unavailable"}
    )


@app.get("/", response_model=RootResponse)
def root():
    return {"message": "RAG API is running."}


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        document_count = app_context.text_collection.count()

        if indexing_status["status"] == "failed":
            raise HTTPException(
                status_code=503,
                detail="Book indexing failed"
            )

        return {
            "status": "healthy",
            "database": "connected",
            "documents": document_count,
            "embedding_model": "loaded",
            "cross_encoder": "loaded"
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Health check failed: {str(e)}"
        )

        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )


@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask(
    request: QuestionRequest,
    authenticated: str = Depends(verify_api_key)
):
    logger.info(
        f"Question received | "
        f"length={len(request.question)}"
    )

    if not check_rate_limit(authenticated):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )

    if indexing_status["status"] == "indexing":
        raise HTTPException(
            status_code=503,
            detail="Book indexing is still in progress. Please try again shortly."
        )

    if indexing_status["status"] == "failed":
        raise HTTPException(
            status_code=503,
            detail="Book indexing failed. Please try again later."
        )

    answer = ask_question(
        question=request.question,
        search_all="yes",
        selected_book=None,
        context=app_context
    )

    logger.info("Answer generated successfully")

    return {
        "question": request.question,
        "answer": answer
    }