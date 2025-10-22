from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Diagnosis Agent running"}

@app.get("/predict")
def predict():
    print("Processing predict...")
    return {"prediction": "Likely benign", "confidence": 0.85}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5007)  # Set your port here
