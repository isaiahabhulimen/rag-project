class RAGError(Exception):
    """Base exception for application-level RAG errors."""


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""


class LLMError(RAGError):
    """Raised when LLM generation fails."""
    