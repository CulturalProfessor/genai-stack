# Pinned base: the previous `langchain/langchain` image floated and supplied
# langchain implicitly, so builds were not reproducible.
FROM python:3.12-slim

WORKDIR /app

# So `import genai_stack` resolves when Streamlit/uvicorn set sys.path[0]
# to the app's own directory.
ENV PYTHONPATH=/app

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY genai_stack ./genai_stack
COPY apps/loader.py ./apps/loader.py

EXPOSE 8502

# The loader serves on 8502; this used to probe 8501 and never matched.
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "apps/loader.py","--server.port=8502", "--server.address=0.0.0.0"]
