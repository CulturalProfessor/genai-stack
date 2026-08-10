# GenAI Stack
This GenAI application stack will get you started building your
own GenAI application in no time.
The demo applications can serve as inspiration or as a starting point.
Learn more about the details in the [technical blog post](https://neo4j.com/developer-blog/genai-app-how-to-build/).

# Configure

Create a `.env` file from the environment template file `env.example`

Available variables:
| Variable Name          | Default value                      | Description                                                             |
|------------------------|------------------------------------|-------------------------------------------------------------------------|
| OLLAMA_BASE_URL        | http://host.docker.internal:11434  | REQUIRED - URL to Ollama LLM API                                        |   
| NEO4J_URI              | neo4j://database:7687              | REQUIRED - URL to Neo4j database                                        |
| NEO4J_USERNAME         | neo4j                              | REQUIRED - Username for Neo4j database                                  |
| NEO4J_PASSWORD         | password                           | REQUIRED - Password for Neo4j database                                  |
| LLM                    | llama2                             | REQUIRED - Can be any Ollama model tag, or gpt-4 or gpt-3.5 or claudev2 |
| EMBEDDING_MODEL        | sentence_transformer               | REQUIRED - Can be sentence_transformer, openai, aws or ollama           |
| AWS_ACCESS_KEY_ID      |                                    | REQUIRED - Only if LLM=claudev2 or embedding_model=aws                  |
| AWS_SECRET_ACCESS_KEY  |                                    | REQUIRED - Only if LLM=claudev2 or embedding_model=aws                  |
| AWS_DEFAULT_REGION     |                                    | REQUIRED - Only if LLM=claudev2 or embedding_model=aws                  |
| OPENAI_API_KEY         |                                    | REQUIRED - Only if LLM=gpt-4 or LLM=gpt-3.5 or embedding_model=openai   |
| LANGCHAIN_ENDPOINT     | "https://api.smith.langchain.com"  | OPTIONAL - URL to Langchain Smith API                                   |
| LANGCHAIN_TRACING_V2   | false                              | OPTIONAL - Enable Langchain tracing v2                                  |
| LANGCHAIN_PROJECT      |                                    | OPTIONAL - Langchain project name                                       |
| LANGCHAIN_API_KEY      |                                    | OPTIONAL - Langchain API key                                            |
| CODE_INDEX_PATH        | data/all_code_blocks.json          | OPTIONAL - Index file the loader reads                                  |
| IMPORT_BATCH_SIZE      | 100                                | OPTIONAL - Symbols embedded and written per batch                       |
| API_KEY                |                                    | OPTIONAL - When set, the API requires it in the X-API-Key header        |
| ALLOWED_ORIGINS        | http://localhost:8505,...          | OPTIONAL - Comma separated CORS allowlist for the API                   |
| RATE_LIMIT_REQUESTS    | 30                                 | OPTIONAL - Requests per IP per window (0 disables)                      |
| RATE_LIMIT_WINDOW_SECONDS | 60                              | OPTIONAL - Rate limit window                                            |

## LLM Configuration
MacOS and Linux users can use any LLM that's available via Ollama. Check the "tags" section under the model page you want to use on https://ollama.ai/library and write the tag for the value of the environment variable `LLM=` in th e`.env` file.
All platforms can use GPT-3.5-turbo and GPT-4 (bring your own API keys for OpenAIs models).

**MacOS**
Install [Ollama](https://ollama.ai) on MacOS and start it before running `docker compose up`.

**Linux**
No need to install Ollama manually, it will run in a container as
part of the stack when running with the Linux profile: run `docker compose --profile linux up`.
Make sure to set the `OLLAMA_BASE_URL=http://llm:11434` in the `.env` file when using Ollama docker container.

**Windows**
Not supported by Ollama, so Windows users need to generate a OpenAI API key and configure the stack to use `gpt-3.5` or `gpt-4` in the `.env` file.
# Develop

> [!WARNING]
> There is a performance issue that impacts python applications in the latest release of Docker Desktop. Until a fix is available, please use [version `4.23.0`](https://docs.docker.com/desktop/release-notes/#4230) or earlier.

**1. Index a codebase**

`fs.py` parses a TS/JS repository with tree-sitter and writes one JSON record
per file, listing the functions, methods, classes and types it defines. It runs
on the host and needs the tree-sitter packages from `requirements.txt`:

```bash
pip install tree-sitter tree-sitter-typescript tree-sitter-javascript
```

```bash
python3 fs.py --repo ../Rocket.Chat --output data/all_code_blocks.json
```

Useful flags: `--ignore-dir` / `--ignore-glob` to skip paths, `--max-chars` to
change where long symbols are truncated (the default of 4000 is chosen to fit
the retriever's `k=4` inside the model context), `--min-chars` to drop trivial
one-liners.

The `data/` directory is mounted into the loader container, so re-indexing a
repository never requires a rebuild.

**2. Set the model**

In `.env` use a code model, for example `LLM=codellama` and
`EMBEDDING_MODEL=sentence_transformer`.

**3. Start everything**

```bash
docker compose up
```

Then open the loader at http://localhost:8502 and press **Import** to embed the
symbols and write them to Neo4j. The importer is idempotent: re-importing
updates existing nodes rather than duplicating them, and "Delete imported graph"
under *Danger zone* clears them out.
If changes to build scripts has been made, **rebuild**.
```
docker compose up --build
```

To enter **watch mode** (auto rebuild on file changes).
First start everything, then in new terminal:
```
docker compose alpha watch
```

**Shutdown**
If health check fails or containers don't start up as expected, shutdown
completely to start up again.
```
docker compose down
```

# Applications

Here's what's in this repo:

| Name | Main files | Compose name | URLs | Description |
|---|---|---|---|---|
| Support Bot | `bot.py` | `bot` | http://localhost:8501 | Main usecase. Fullstack Python application. |
| Code Loader | `fs.py`, `loader.py` | `loader` | http://localhost:8502 | Extract code symbols from a repo (`fs.py`) and load them into the database with vector embeddings (`loader.py`). |
| PDF Reader | `pdf_bot.py` | `pdf_bot` | http://localhost:8503 | Read local PDF and ask it questions. Fullstack Python application. |
| Standalone Bot API | `api.py` | `api` | http://localhost:8504 | Standalone HTTP API streaming (SSE) + non-streaming endpoints. Python. |
| Standalone Bot UI | `front-end/` | `front-end` | http://localhost:8505 | Standalone client that uses the Standalone Bot API to interact with the model. JavaScript (Svelte) front-end. |

The database can be explored at http://localhost:7474.

## App 1 - Support Agent Bot

UI: http://localhost:8501
DB client: http://localhost:7474

- answer support question based on recent entries
- provide summarized answers with sources
- demonstrate difference between
    - RAG Disabled (pure LLM response)
    - RAG Enabled (vector + knowledge graph context)
- allow to generate a high quality support ticket for the current conversation based on the style of highly rated questions in the database.

![](.github/media/app1-rag-selector.png)
*(Chat input + RAG mode selector)*

|  |  |
|---|---|
| ![](.github/media/app1-generate.png) | ![](.github/media/app1-ticket.png) |
| *(CTA to auto generate support ticket draft)* | *(UI of the auto generated support ticket draft)* |

---

##  App 2 - Loader

UI: http://localhost:8502
DB client: http://localhost:7474

- import the code index produced by `fs.py` into a knowledge graph of
  `(:File)-[:CONTAINS]->(:Code)` nodes
- embed each symbol and store the vector on the `Code` node, backed by a vector
  index and a full-text index for hybrid search
- UI: pick or upload an index, run the import, watch progress, clear the graph
  to re-import from scratch

The import is idempotent - symbols are keyed by `file path : line : qualified
name`, so re-running it updates nodes in place instead of duplicating them.




|  |  |
|---|---|
| ![](.github/media/app2-ui-1.png) | ![](.github/media/app2-model.png) |

## App 3 Question / Answer with a local PDF
UI: http://localhost:8503  
DB client: http://localhost:7474

This application lets you load a local PDF into text
chunks and embed it into Neo4j so you can ask questions about
its contents and have the LLM answer them using vector similarity
search.

![](.github/media/app3-ui.png)

## App 4 Standalone HTTP API
Endpoints:
  - `POST /query` - non streaming, JSON body `{"text": "...", "rag": false}`
  - `POST /generate-ticket` - JSON body `{"text": "..."}`
  - `GET  /query-stream?text=hello&rag=false` - SSE streaming. This one stays a
    GET with query parameters because `EventSource` cannot issue POSTs.

Example cURL commands:
```bash
curl -X POST http://localhost:8504/query -H 'Content-Type: application/json' -d '{"text": "how is authentication handled", "rag": true}'
```

```bash
curl http://localhost:8504/query-stream\?text\=minimal%20hello%20world%20in%20python\&rag\=false
```

When `API_KEY` is set, send it as `-H "X-API-Key: ..."` (or `&api_key=...` on
the streaming endpoint). Requests are rate limited per client IP; see
`RATE_LIMIT_REQUESTS`.

Exposes the functionality to answer questions in the same way as App 1 above. Uses
same code and prompts.

## App 5 Static front-end
UI: http://localhost:8505

This application has the same features as App 1, but is built separate from
the back-end code using modern best practices (Vite, Svelte, Tailwind).  
The auto-reload on changes are instant using the Docker watch `sync` config.  
![](.github/media/app5-ui.png)
