from fastapi import FastAPI, Query
from knowledge_graph import query_knowledge

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Knowledge Agent running"}

@app.get("/query")
def query(symptom: str = Query(...)):
    results = query_knowledge(symptom)
    return {"symptom": symptom, "knowledge": results}
