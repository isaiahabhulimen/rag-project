
FROM python:3.11

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Python dependencies

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.1

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face model cache

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); CrossEncoder('cross-encoder/ms-marco-TinyBERT-L2-v2')"


# Runtime configuration

ENV HF_HUB_OFFLINE=1

COPY . .



# Application


EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
