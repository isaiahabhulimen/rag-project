from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class QuestionResponse(BaseModel):
    question: str
    answer: str


class RootResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    database: str
    documents: int
    embedding_model: str
    cross_encoder: str


class ErrorResponse(BaseModel):
    detail: str
