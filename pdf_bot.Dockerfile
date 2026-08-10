# Pinned base: the previous `langchain/langchain` image floated and supplied
# langchain implicitly, so builds were not reproducible.
FROM python:3.12-slim

WORKDIR /app

# So `import genai_stack` resolves when Streamlit/uvicorn set sys.path[0]
# to the app's own directory.
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY genai_stack ./genai_stack
COPY apps/pdf_bot.py ./apps/pdf_bot.py

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "apps/pdf_bot.py", "--server.port=8503", "--server.address=0.0.0.0"]
