import { writable } from "svelte/store";
import { API_BASE_URL, apiHeaders } from "./api";

const API_ENDPOINT = `${API_BASE_URL}/generate-ticket`;

export const generationStates = {
    IDLE: "idle",
    SUCCESS: "success",
    ERROR: "error",
    LOADING: "loading",
};

function createGenerationStore() {
    const { subscribe, update } = writable({ state: generationStates.IDLE, data: { title: "", text: "" } });

    return {
        subscribe,
        generate: async (fromQuestion) => {
            update(() => ({ state: generationStates.LOADING, data: { title: "", text: "" } }));
            try {
                const response = await fetch(API_ENDPOINT, {
                    method: "POST",
                    headers: apiHeaders(),
                    body: JSON.stringify({ text: fromQuestion }),
                });
                if (!response.ok) {
                    throw new Error(`Request failed with status ${response.status}`);
                }
                const generation = await response.json();
                update(() => ({ state: generationStates.SUCCESS, data: generation.result }));
            } catch (e) {
                console.log("e: ", e);
                update(() => ({ state: generationStates.ERROR, data: { title: "", text: "" } }));
            }
        },
    };
}

export const generationStore = createGenerationStore();
