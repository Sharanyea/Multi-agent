from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Coordinator Agent is running"}

@app.get("/diagnose")
def diagnose():
    try:
        # Step 1: Ask Imaging Agent for preprocessing
        img_result = requests.get("http://imaging:8003/process").json()

        # Step 2: Ask Knowledge Agent for context
        knowledge = requests.get("http://knowledge:8002/query?symptom=Microcalcification").json()

        # Step 3: Ask Diagnosis Agent for prediction
        diagnosis = requests.get("http://diagnosis:8001/predict").json()

        return {
            "image_result": img_result,
            "knowledge_info": knowledge,
            "diagnosis": diagnosis
        }

    except Exception as e:
        return {"error": str(e)}
