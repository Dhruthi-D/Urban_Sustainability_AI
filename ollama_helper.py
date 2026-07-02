import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi"   # or llama3

def generate_explanation(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Ollama error: {response.text}"

    except Exception as e:
        return f"Ollama connection failed: {e}"