import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    host: "0.0.0.0",
    port: 5173,
    // The dev server runs in a container; without polling it misses edits made
    // on the host through the bind mount.
    watch: { usePolling: true },
  },
});
