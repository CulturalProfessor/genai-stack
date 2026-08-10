// Base URL of the FastAPI service. Override at build time with VITE_API_BASE_URL
// so the front-end can point at something other than a local container.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8504").replace(/\/$/, "");

// Only set when the API is started with API_KEY. EventSource cannot send
// headers, so the streaming endpoint takes it as a query parameter instead.
export const API_KEY = import.meta.env.VITE_API_KEY ?? "";

export function apiHeaders() {
    const headers = { "Content-Type": "application/json" };
    if (API_KEY) {
        headers["X-API-Key"] = API_KEY;
    }
    return headers;
}
