# Pinned base: the previous `langchain/langchain` image floated and supplied
# langchain implicitly, so builds were not reproducible.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY utils.py .
COPY chains.py .

HEALTHCHECK CMD curl --fail http://localhost:8504

ENTRYPOINT [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8504" ]
