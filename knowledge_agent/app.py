from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Dict, Any, List
from agents.knowledge_agent.knowledge_graph import query_knowledge, generate_llm_analysis, DifferentialDiagnosis

app = FastAPI()

class KnowledgeAgentInput(BaseModel):
    symptom: str
    processed_features: Dict[str, Any]
    diagnosis_prediction: Dict[str, Any]
    knowledge_facts: List[str]

@app.get("/")
def ok():
    return {"status": "Knowledge Agent running"}

@app.get("/facts")
async def get_facts(symptom: str = Query(...)):
    try:
        return await query_knowledge(symptom)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_post(data: KnowledgeAgentInput = Body(...)):
    try:
        result = await generate_llm_analysis(
            symptom=data.symptom,
            processed_features=data.processed_features,
            knowledge_facts=data.knowledge_facts,
            diagnosis_prediction=data.diagnosis_prediction
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
