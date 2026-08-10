# Pinned base: the previous `langchain/langchain` image floated and supplied
# langchain implicitly, so builds were not reproducible.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY loader.py .
COPY code_index.py .
COPY utils.py .
COPY chains.py .

EXPOSE 8502

# The loader serves on 8502; this used to probe 8501 and never matched.
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "loader.py","--server.port=8502", "--server.address=0.0.0.0"]
