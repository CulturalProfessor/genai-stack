# One image for every Python service. They previously had four near-identical
# Dockerfiles, which meant four copies of the same multi-GB dependency layer -
# and any difference between them (the loader needed git) forked the layer
# chain and duplicated it. Each service now runs this image with its own
# `command` in docker-compose.yml.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl for the healthchecks, git so the loader can clone the repo you give it.
# No build-essential: every pinned dependency ships a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY genai_stack ./genai_stack
COPY apps ./apps
# Theme options for the Streamlit app. Read from the working directory, which
# is why this sits at /app rather than in a home directory.
COPY .streamlit ./.streamlit
