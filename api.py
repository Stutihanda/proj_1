from fastapi import FastAPI
from main import main

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Climate Guardian AI API is running"}

@app.post("/run-pipeline")
def run_pipeline():
    try:
        main()
        return {
            "status": "success",
            "message": "Pipeline executed successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }