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

COPY pdf_bot.py .
COPY utils.py .
COPY chains.py .

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "pdf_bot.py", "--server.port=8503", "--server.address=0.0.0.0"]
