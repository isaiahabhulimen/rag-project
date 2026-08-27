import os

from dotenv import load_dotenv


load_dotenv()



# Secrets


rag_api_key = os.getenv("RAG_API_KEY")

if not rag_api_key:
    raise RuntimeError(
        "RAG_API_KEY environment variable is not configured"
    )



# LLM configuration


llm_backend = os.getenv(
    "LLM_BACKEND",
    "groq"
)

llm_name = os.getenv(
    "LLM_NAME",
    "openai/gpt-oss-120b"
)

groq_api_key = os.getenv("GROQ_API_KEY")

if llm_backend == "groq" and not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not configured"
    )



# Storage configuration


database_path = os.getenv(
    "DATABASE_PATH",
    "database"
)

book_folder = os.getenv(
    "BOOK_FOLDER",
    "books"
)



# Model configuration


model_name = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

cross_encoder_name = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-TinyBERT-L2-v2"
)



# RAG configuration


collection_name = os.getenv(
    "COLLECTION_NAME",
    "richest_man_babylon"
)

embedding_batch_size = 8

chunk_size = 1000

chunk_overlap = 200

retrieval_results = 10

llm_context_chunks = 5

retrieval_candidate_pool = 50

semantic_weight = 0.7

keyword_weight = 0.3

mmr_lambda = 0.7

phrase_bonus = 2

max_frequency_bonus = 5

benchmark_folder = "benchmark"

semantic_chunk_threshold = 0.75

min_chunk_characters = 300

max_chunk_characters = 1200



# Debug settings


print_image_results = False

debug_keyword_results = 10

debug_semantic_results = 10
