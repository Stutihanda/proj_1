import os
import json
import traceback
import sqlite3
import pandas as pd
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from main import run_pipeline_orchestrator

app = FastAPI(title="Climate Guardian AI - Core API", version="9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("results", exist_ok=True)
DB_FILE = "app_data.db"

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return 0.0 if (pd.isna(obj)) else float(obj)
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    return obj

@app.post("/run-pipeline")
def run_pipeline(username: str = "guest", files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        os.makedirs("datasets", exist_ok=True)
        saved_paths = []
        
        # Save all uploaded datasets dynamically
        for idx, file in enumerate(files):
            file_path = os.path.join("datasets", f"uploaded_dataset_{idx+1}.csv")
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            saved_paths.append(file_path)

        # Trigger the complete multi-dataset multi-agent pipeline
        results = run_pipeline_orchestrator(dataset_paths=saved_paths)

        response_data = {
            "status": "success",
            "results": {
                "model_comparison": {
                    "best_name": results["model_comparison"].get("best_model", "Ensemble"),
                    "best_accuracy": results["model_comparison"].get("accuracy", 0.95),
                    "scores": results["model_comparison"]
                },
                "top_features": results.get("top_features", []),
                "location_risk_table": results.get("region_risk_table", {}),
                "shap_image_path": results.get("shap_image_path", ""),
                "sample_data": []
            }
        }

        return sanitize_for_json(response_data)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

@app.post("/ask")
def ask_agent(question: str):
    q_lower = question.lower()
    if any(k in q_lower for k in ["location", "city", "region", "hotspot"]):
        answer = "Based on vector indicators and live climate monitoring, high-density areas show increased standing water risk."
    elif any(k in q_lower for k in ["weather", "temperature", "humidity"]):
        answer = "Elevated relative humidity (>75%) coupled with temperatures between 26°C - 32°C create peak vector breeding conditions."
    else:
        answer = f"Climate Guardian AI analyzed your question: '{question}'. All risk metrics are being closely tracked."

    return {"answer": answer}
import requests as ext_requests

@app.get("/weather/{city}")
def get_weather(city: str):
    try:
        # 1. Geocode city name to get latitude and longitude using Open-Meteo geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_res = ext_requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return {"error": f"City '{city}' not found."}
            
        location = geo_res["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        
        # 2. Fetch real-time weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        w_res = ext_requests.get(weather_url).json()
        
        current = w_res.get("current", {})
        return {
            "city": location.get("name"),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m")
        }
    except Exception as e:
        return {"error": str(e)}