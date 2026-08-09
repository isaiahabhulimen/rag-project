import os
from dotenv import load_dotenv
load_dotenv()

llm_backend = os.getenv(
    "LLM_BACKEND",
    "groq"
)

llm_name = os.getenv(
    "LLM_NAME",
    "llama-3.3-70b-versatile"
)

database_path = os.getenv(
    "DATABASE_PATH",
    "database"
)

book_folder = os.getenv(
    "BOOK_FOLDER",
    "books"
)

model_name = "all-MiniLM-L6-v2"

cross_encoder_name = "cross-encoder/ms-marco-TinyBERT-L2-v2"

collection_name = "richest_man_babylon"

embedding_batch_size = 8

chunk_size = 1000

chunk_overlap = 200

retrieval_results = 10

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