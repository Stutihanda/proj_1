import json
import os

import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from main import main

app = FastAPI(title="Climate Guardian AI API")

# Streamlit / n8n run on different ports than FastAPI, so CORS needs to be open
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"


@app.get("/")
def home():
    return {"message": "Climate Guardian AI API is running"}


@app.post("/run-pipeline")
def run_pipeline(
    climate_file: UploadFile = File(...),
    health_file: UploadFile = File(...),
    social_file: UploadFile = File(...),
):
    try:
        os.makedirs("datasets", exist_ok=True)

        climate_path = "datasets/climate.csv"
        health_path = "datasets/health.csv"
        social_path = "datasets/social.csv"

        for uploaded, path in [
            (climate_file, climate_path),
            (health_file, health_path),
            (social_file, social_path),
        ]:
            with open(path, "wb") as f:
                f.write(uploaded.file.read())

        results = main(
            climate_path=climate_path,
            health_path=health_path,
            social_path=social_path,
        )

        return {
            "status": "success",
            "message": "Pipeline executed successfully",
            "results": results,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/ask")
def ask(question: str):
    try:
        if not os.path.exists("results/latest_run.json"):
            return {
                "status": "error",
                "message": "No pipeline results found yet. Run the pipeline first.",
            }

        with open("results/latest_run.json", "r", encoding="utf-8") as f:
            context = json.load(f)

        prompt = f"""You are an assistant analyzing dengue outbreak risk data.
Use ONLY the data below to answer the question. Be concise and specific,
referencing actual region names and numbers from the data where relevant.

DATA:
{json.dumps(context, indent=2)}

QUESTION: {question}

ANSWER:"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()

        answer = response.json().get("response", "").strip()

        return {
            "status": "success",
            "answer": answer,
        }

    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": "Could not reach Ollama. Is it running? (ollama serve)",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
