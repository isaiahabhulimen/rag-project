from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.exceptions import LLMError, RetrievalError
from app.schemas import QuestionRequest, QuestionResponse, RootResponse, HealthResponse, ErrorResponse
from app.rag_service import ask_question
from app.state import app_context
from logger import logger
from app.auth import verify_api_key
from app.rate_limiter import check_rate_limit


app = FastAPI()
@app.exception_handler(LLMError)
def llm_error_handler(request, exc):
    logger.error(f"LLM error: {str(exc)}")

    return JSONResponse(
        status_code=502,
        content={"detail": "LLM service unavailable"}
    )


@app.exception_handler(RetrievalError)
def retrieval_error_handler(request, exc):
    logger.error(f"Retrieval error: {str(exc)}")

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

        return {
            "status": "healthy",
            "database": "connected",
            "documents": document_count,
            "embedding_model": "loaded",
            "cross_encoder": "loaded"
        }

    except Exception as e:

        logger.error(f"Health check failed: {str(e)}")

        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest, authenticated: str = Depends(verify_api_key)):

    logger.info(
    f"Question received | length={len(request.question)}"
)
    if not check_rate_limit(authenticated):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
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