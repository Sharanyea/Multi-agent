from fastapi import FastAPI
import requests
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Coordinator Agent is running"}

@app.get("/diagnose")
def diagnose():
    try:
        # Step 1: Ask Imaging Agent for preprocessing
        print( "Requesting imaging agent..." )
        img_result = requests.get("http://localhost:5008/process").json()
        print(img_result)


        # Step 2: Ask Knowledge Agent for context
        knowledge = requests.get("http://localhost:5009/query?symptom=Microcalcification").json()

        # Step 3: Ask Diagnosis Agent for prediction
        diagnosis = requests.get("http://localhost:5007/predict").json()

        return {
            "image_result": img_result,
            "knowledge_info": knowledge,
            "diagnosis": diagnosis
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5010)  # Set your port here