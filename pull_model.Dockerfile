#syntax = docker/dockerfile:1.4

FROM ollama/ollama:latest AS ollama
FROM babashka/babashka:latest

# just using as a client - never as a server
COPY --from=ollama /bin/ollama ./bin/ollama

COPY <<EOF pull_model.clj
(ns pull-model
  (:require [babashka.process :as process]
            [clojure.core.async :as async]))

;; This service gates every app service via service_completed_successfully, so
;; it must exit 0 even when it cannot pull. A missing model degrades answers;
;; it should not stop the database, the loader, or the UIs from starting.
(defn warn [& lines]
  (println)
  (println "  ----------------------------------------------------------------")
  (doseq [line lines] (println (str "  " line)))
  (println "  ----------------------------------------------------------------")
  (println))

(try
  (let [llm (get (System/getenv) "LLM")
        url (get (System/getenv) "OLLAMA_BASE_URL")]
    (if (and llm url (not (#{"gpt-4" "gpt-3.5" "claude" "claudev2"} llm)))
      (do
        (println (format "pulling ollama model %s using %s" llm url))
        (let [done (async/chan)]
          (async/go-loop [n 0]
            (let [[v _] (async/alts! [done (async/timeout 5000)])]
              (if (= :stop v) :stopped (do (println (format "... pulling model (%ss) - will take several minutes" (* n 5))) (recur (inc n))))))
          ;; :extra-env, not :env - :env REPLACES the environment, which wipes
          ;; HOME and makes the ollama client panic before it connects.
          ;; :continue keeps a failed pull from throwing.
          (let [{:keys [exit]} (process/shell {:extra-env {"OLLAMA_HOST" url "HOME" "/root"}
                                               :continue true
                                               :out :inherit
                                               :err :inherit}
                                              (format "./bin/ollama pull %s" llm))]
            (async/>!! done :stop)
            (if (zero? exit)
              (println (format "pulled %s" llm))
              (warn (format "Could not pull '%s' from %s" llm url)
                    ""
                    "The stack will still start, but any question you ask will fail"
                    "until an LLM is reachable. Options:"
                    ""
                    "  - run Ollama in a container:"
                    "      docker compose --profile linux up"
                    "    with OLLAMA_BASE_URL=http://llm:11434 in .env"
                    ""
                    "  - or use OpenAI: set LLM=gpt-3.5 and OPENAI_API_KEY in .env"
                    ""
                    "Indexing a repository in the loader needs no LLM and works now.")))))
      (println (format "no ollama pull needed for LLM=%s" llm))))
  (catch Throwable t
    (warn "pull-model hit an unexpected error:" (str (.getMessage t))
          "Continuing so the rest of the stack can start.")))

;; Always succeed - see the note at the top.
(System/exit 0)
EOF

ENTRYPOINT ["bb", "-f", "pull_model.clj"]
