from fastapi import FastAPI, HTTPException

from app.schemas import QuestionRequest
from app.rag_service import ask_question
from app.state import app_context
from logger import logger


app = FastAPI()


@app.get("/")
def root():
    return {"message": "RAG API is running."}

@app.get("/health")
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

@app.post("/ask")
def ask(request: QuestionRequest):

    logger.info(f"Question received: {request.question}")

    try:
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

    except Exception as e:
        logger.error(f"RAG failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process request"
        )