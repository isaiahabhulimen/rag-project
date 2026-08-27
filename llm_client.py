from groq import Groq

from app.exceptions import LLMError
from config import groq_api_key, llm_backend
from logger import logger


if llm_backend != "groq":
    raise RuntimeError(
        f"Unsupported LLM_BACKEND: {llm_backend}"
    )


client = Groq(
    api_key=groq_api_key
)


def chat(messages, model, temperature=0.2, max_tokens=300):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    except Exception as e:
        logger.error(
            f"LLM request failed: {type(e).__name__}"
        )

        raise LLMError("LLM request failed") from e

