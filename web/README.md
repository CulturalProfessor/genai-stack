# Code graph web

The React client for the chat, talking to the FastAPI service on :8504.

```bash
docker compose up web        # http://localhost:5173
```

`VITE_API_BASE_URL` points it at the API; `VITE_API_KEY` is only needed when
the API has `API_KEY` set.
