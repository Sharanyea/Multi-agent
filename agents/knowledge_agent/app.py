from fastapi import FastAPI, Query
from knowledge_graph import query_knowledge
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Knowledge Agent running"}

@app.get("/query")
def query(symptom: str = Query(...)):
    results = query_knowledge(symptom)
    return {"symptom": symptom, "knowledge": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5009)  # Set your port here