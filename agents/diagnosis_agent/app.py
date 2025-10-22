from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Diagnosis Agent running"}

@app.get("/predict")
def predict():
    # Placeholder: return fake prediction for demo
    return {"prediction": "Likely benign", "confidence": 0.85}
