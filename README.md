# Multi-Book RAG Question Answering System.

A production-oriented Retrieval-Augmented Generation (RAG) system for asking questions across multiple books and generating answers grounded in retrieved document content.

The system goes beyond basic vector search by combining semantic retrieval, BM25 keyword search, cross-encoder reranking, query decomposition, incremental indexing, background ingestion, evaluation, and a secured FastAPI API.

---

## What It Does

The application allows users to:

* Upload and index multiple PDF books
* Ask questions about a specific book or across all indexed books
* Retrieve relevant information using semantic and keyword search
* Rerank retrieved candidates using a lightweight TinyBERT cross-encoder
* Break complex questions into smaller queries when appropriate
* Generate answers from retrieved document context
* Add new books without rebuilding the entire database
* Resume interrupted indexing jobs
* Access the system through a secured API

The project started as a basic RAG implementation and was progressively developed into a deployable application with stronger retrieval, reliability, security, testing, and operational capabilities.

---

## RAG Pipeline

The core question-answering pipeline is:

```text
User Question
      ↓
Question Routing
      ↓
Query Decomposition
  (when needed)
      ↓
Hybrid Retrieval
 ┌───────────────┐
 │ Semantic      │
 │ + BM25        │
 └───────────────┘
      ↓
Candidate Pool
      ↓
TinyBERT Cross-Encoder
Reranking
      ↓
Top Context Chunks
      ↓
LLM
      ↓
Grounded Answer
```

### Why the multiple retrieval stages?

The system does not rely on a single similarity search.

Semantic retrieval helps find content based on meaning, while BM25 helps with exact terms and keyword-based matches. A larger candidate pool is then passed to the cross-encoder, which provides a second relevance-ranking stage before the final context reaches the LLM.

This gives the system separate stages for **retrieval and relevance refinement** rather than asking the LLM to compensate for weak retrieval.

---

## Key Features

### Multi-Book Support

Multiple books can be indexed into the same knowledge base.

Questions can be searched across all books or restricted to a selected book.

### Hybrid Retrieval

The retriever combines:

* Sentence Transformer semantic search
* BM25 keyword retrieval

The two retrieval signals are combined using configurable weights.

### Cross-Encoder Reranking

Retrieved candidates are reranked using:

`cross-encoder/ms-marco-TinyBERT-L2-v2`

The reranker provides a second relevance check before the final context is sent to the language model.

### Complex Question Handling

The system can route questions based on complexity.

Complex questions can be decomposed into smaller sub-questions, retrieved independently, combined, deduplicated, and reranked before answer generation.

### Context-Grounded Generation

The LLM receives retrieved document context rather than being asked to answer solely from its general knowledge.

This keeps the generation stage tied to the information retrieved from the indexed books.

### Incremental Indexing

The system uses SHA-256 file hashes to detect whether a document has changed.

Unchanged documents can be skipped, while new or modified books can be indexed without unnecessarily rebuilding the entire database.

### Background Ingestion

Book processing is separated from normal question-answering requests through a background ingestion worker.

Long-running indexing jobs can therefore operate independently of the API.

### Resumable Ingestion

Ingestion jobs maintain checkpoints during processing.

If a longer indexing job is interrupted, processing can resume from stored progress rather than necessarily starting from the beginning.

### Persistent Storage

ChromaDB provides persistent vector storage.

The application also supports S3-compatible object storage for uploaded documents and ingestion job state.

### API Security

The API uses Bearer API-key authentication.

Invalid credentials are rejected, and API keys are compared securely.

### Rate Limiting

The API includes a rolling-window rate limiter to control request frequency.

The current implementation is intentionally simple and in-memory, so the limit is local to an application process rather than shared across multiple instances.

### Health Checks

The `/health` endpoint reports the status of important application components, including:

* Database
* Indexed documents
* Embedding model
* Cross-encoder

### Application Logging

The application uses Python's logging framework to provide timestamped log messages and log levels for operational visibility.

---

## Engineering Decisions

Several parts of the system were shaped by practical constraints and problems encountered during development rather than simply following a basic RAG tutorial.

### Resource-Conscious Model Selection

The system uses:

* `all-MiniLM-L6-v2` for embeddings
* `cross-encoder/ms-marco-TinyBERT-L2-v2` for reranking

The project was developed and tested on a machine with **4 GB of RAM**, so model selection required balancing retrieval quality with realistic hardware requirements.

Using TinyBERT as the reranker allowed the system to benefit from cross-encoder reranking without requiring a much larger model.

### Incremental Indexing

Repeatedly rebuilding the entire knowledge base became inefficient as the project moved toward multi-book support.

SHA-256 file hashing was therefore introduced to identify unchanged files.

The resulting workflow is:

```text
Book
 ↓
SHA-256 Hash
 ↓
Compare With Stored Hash
 ↓
Unchanged → Skip
Changed/New → Index
```

This makes document ingestion more practical as the collection grows.

### Hybrid Retrieval

Vector search alone was not treated as sufficient for every type of question.

BM25 was added alongside semantic retrieval so the system could benefit from both meaning-based matching and exact keyword matching.

### Retrieval Before Reranking

Instead of immediately reranking a very small set of results, the retriever first creates a larger candidate pool.

The cross-encoder then evaluates those candidates and selects the most relevant context for generation.

This separates the responsibilities of:

**finding possible matches → determining the strongest matches → generating the answer**

### Background and Resumable Ingestion

As document processing became more substantial, indexing was separated from the API request path.

Job tracking and checkpoints were introduced so ingestion could be monitored and recovered more reliably.

---

## Document Ingestion

The ingestion pipeline is:

```text
PDF Upload
    ↓
Book Job
    ↓
Background Worker
    ↓
PDF Processing
    ↓
Text Extraction
    ↓
Semantic Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Checkpointing
    ↓
Completed Job
```

New or changed books can be processed independently without rebuilding unrelated documents.

---

## Technology Stack

| Area               | Technology                    |
| ------------------ | ----------------------------- |
| API                | FastAPI                       |
| Server             | Uvicorn                       |
| Vector Database    | ChromaDB                      |
| Embeddings         | Sentence Transformers         |
| Reranking          | TinyBERT Cross-Encoder        |
| Keyword Retrieval  | BM25                          |
| Text Processing    | LangChain Text Splitters      |
| PDF Processing     | pdfplumber                    |
| LLM                | Groq                          |
| Validation         | Pydantic                      |
| Object Storage     | boto3 / S3-compatible storage |
| Configuration      | python-dotenv                 |
| Containerization   | Docker                        |
| CI/CD              | GitHub Actions                |
| Container Registry | GitHub Container Registry     |

### Models

**Embedding model:** `all-MiniLM-L6-v2`

**Reranker:** `cross-encoder/ms-marco-TinyBERT-L2-v2`

The language model is configurable through environment variables.

---

## Project Structure

```text
.
├── api.py
├── config.py
├── main.py
├── indexer.py
├── ingest.py
├── preprocessor.py
├── semantic_chunker.py
├── retriever.py
├── reranker.py
├── generator.py
├── query_processor.py
├── llm_client.py
├── storage.py
├── evaluation.py
├── benchmark_loader.py
├── models.py
├── utils.py
├── logger.py
├── vision.py
│
├── app/
│   ├── auth.py
│   ├── book_jobs.py
│   ├── context.py
│   ├── exceptions.py
│   ├── rag_service.py
│   ├── rate_limiter.py
│   ├── schemas.py
│   └── state.py
│
├── tests/
│   ├── test_api.py
│   ├── test_auth.py
│   └── test_rate_limiter.py
│
├── benchmark/
├── evaluation/
├── reports/
│
├── Dockerfile
├── Dockerfile.worker
├── docker-compose.yml
├── requirements.txt
└── .github/
    └── workflows/
        └── docker-build.yml
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

Configuration includes items such as:

* LLM provider and model
* Groq API key
* RAG API key
* Worker authentication
* Database configuration
* Object storage configuration

**Never commit `.env` or real credentials to source control.**

### 5. Start the API

```bash
uvicorn api:app --reload
```

The API runs on port `8000`.

FastAPI's interactive Swagger documentation can be used to explore the available endpoints.

---

## API

### Health Check

```http
GET /health
```

Returns the current application and model status.

### Ask a Question

```http
POST /ask
```

Requires Bearer authentication.

Example request:

```json
{
  "question": "Who is Arkad?"
}
```

Example response:

```json
{
  "question": "Who is Arkad?",
  "answer": "Arkad is the richest man in Babylon."
}
```

### Upload a Book

```http
POST /books/upload
```

Uploads a PDF and creates an ingestion job.

### Check Ingestion Status

```http
GET /books/{job_id}
```

Returns the current state and progress of an ingestion job.

---

## Testing

The project includes automated tests covering the API, authentication, validation, health checks, and rate limiting.

Run the tests with:

```bash
pytest
```

### API Tests

The API tests verify:

* Root endpoint
* Authentication requirements
* Invalid credentials
* Request validation
* Successful authenticated requests
* Health endpoint

Heavy ML models and object storage are mocked during these tests so the API layer can be tested independently.

### Authentication Tests

Authentication tests cover:

* Missing credentials
* Invalid API keys
* Invalid authentication schemes
* Valid Bearer credentials
* Client identifier generation

### Rate Limiting Tests

The rate limiter is tested to ensure that requests within the configured limit are accepted and requests exceeding the limit are blocked.

---

## RAG Evaluation

The project includes a benchmark-based evaluation workflow.

Each benchmark question is processed through the actual RAG pipeline:

```text
Benchmark Question
       ↓
Retrieval
       ↓
Cross-Encoder Reranking
       ↓
Answer Generation
       ↓
Semantic Answer Evaluation
       ↓
Evaluation Report
```

Generated answers are compared with expected answers for **meaning**, rather than requiring an exact string match.

Reports contain:

* Total questions
* Passed questions
* Failed questions
* Accuracy
* Individual question results

Timestamped JSON reports are stored in:

```text
reports/
```

The benchmark provides a repeatable way to monitor answer quality as the system changes.

---

## Docker

The project provides separate Docker configurations for the API and ingestion worker.

### API Image

```bash
docker build -t rag-api .
```

### Worker Image

```bash
docker build -f Dockerfile.worker -t rag-worker .
```

A `docker-compose.yml` file is also provided for local API deployment with persistent database storage and a container health check.

---

## CI/CD

GitHub Actions automatically builds and publishes the API Docker image to GitHub Container Registry when changes are pushed to the `main` branch.

The workflow:

1. Checks out the repository
2. Authenticates with GitHub Container Registry
3. Builds the Docker image
4. Publishes the image

This provides an automated path from source-code changes to a deployable container image.

---

## Security

The application includes:

* Bearer API-key authentication
* Secure API-key comparison
* SHA-256 client identifier generation
* Pydantic request validation
* Rate limiting
* Protected internal ingestion endpoints
* Environment-based secret configuration
* Secrets excluded from source control

The system is designed so credentials are supplied through environment variables rather than hard-coded into the application.

---

## Reliability

The project includes several mechanisms intended to make it more suitable for deployment:

* Persistent vector storage
* Incremental indexing
* Background ingestion
* Checkpoint-based recovery
* Application logging
* Health checks
* API error handling
* Rate limiting
* Automated tests
* Dockerized deployment
* Automated container builds

These features were added progressively as the system moved beyond the initial RAG prototype.

---

## Current Status

The project has evolved from a local single-book RAG experiment into a deployed, multi-book RAG application with:

* Hybrid retrieval
* TinyBERT cross-encoder reranking
* Complex-question handling
* Incremental indexing
* Background ingestion
* Resumable jobs
* Persistent storage
* API authentication
* Rate limiting
* Health checks
* Automated testing
* Benchmark evaluation
* Docker packaging
* CI/CD
* Cloud deployment

---

## What This Project Demonstrates

This project demonstrates practical experience building a RAG system across both **machine-learning and application-engineering layers**.

The core system combines:

```text
Document Processing
        ↓
Semantic Chunking
        ↓
Embeddings
        ↓
Hybrid Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Context Selection
        ↓
LLM Generation
        ↓
FastAPI
        ↓
Security / Testing / Evaluation
        ↓
Docker / CI/CD / Deployment
```

The emphasis is not on using the largest possible models. The system was designed around practical constraints, retrieval quality, reliability, and the ability to deploy and operate the application.

---

## Future Improvements

Potential future improvements include:

* Distributed rate limiting for multi-instance deployments
* Larger and more diverse evaluation datasets
* More advanced retrieval evaluation metrics
* Improved observability and monitoring
* Additional document formats
* Further ingestion and retrieval optimization

---

## Author

**Isaiah Abhulimen**

This project was developed as a practical RAG engineering project, progressing from an initial prototype toward a production-oriented application.
