
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



# Runtime configuration

ENV HF_HUB_OFFLINE=1

COPY . .



# Application


EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
