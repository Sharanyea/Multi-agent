import httpx
import json
from pydantic import BaseModel
from typing import Dict, Any, List

# LLM configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct:free"

class DifferentialDiagnosis(BaseModel):
    differential_diagnosis: str
    confidence_score_delta: float
    counter_question: str

async def query_knowledge(symptom: str) -> Dict[str, Any]:
    """Fetch clinical facts for a given symptom from LLM"""
    if not OPENROUTER_API_KEY:
        return {"symptom": symptom, "knowledge_facts": ["LLM API Key Missing."]}
    system_prompt = (
        "You are an expert clinical pathologist. Provide 3-5 concise, verified medical facts "
        f"about the symptom: {symptom}. Respond strictly as a comma-separated list in a single text block."
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": "https://yourprojectdomain.com",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Facts needed for {symptom}"}
        ],
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            fact_string = result['choices'][0]['message']['content']
            facts_list = [f.strip() for f in fact_string.split(',') if f.strip()]
            return {"symptom": symptom, "knowledge_facts": facts_list}
        except Exception as e:
            return {"symptom": symptom, "knowledge_facts": [f"Error fetching facts: {str(e)}"]}

async def generate_llm_analysis(
    symptom: str,
    processed_features: Dict[str, Any],
    knowledge_facts: List[str],
    diagnosis_prediction: Dict[str, Any]
) -> DifferentialDiagnosis:
    system_prompt = (
        "You are an expert 'Imagination Agent' providing a second-opinion differential diagnosis. "
        "Respond ONLY with a JSON object containing: "
        "differential_diagnosis (string), confidence_score_delta (float), counter_question (string)."
    )
    user_query = (
        f"CONTEXT:\n"
        f"1. Primary Diagnosis: {diagnosis_prediction.get('prediction', 'N/A')} with Confidence: {diagnosis_prediction.get('confidence', 'N/A')}.\n"
        f"2. Symptom: {symptom}\n"
        f"3. Image Features: {json.dumps(processed_features)}\n"
        f"4. Relevant Clinical Facts: {json.dumps(knowledge_facts)}\n\n"
        "Critique the primary diagnosis. What is a plausible differential diagnosis (edge case)? "
        "Provide a confidence score delta (e.g., -0.15). Provide a critical counter-question to test the primary diagnosis's validity."
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": "https://yourprojectdomain.com",
    }
    payload_dict = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload_dict)
        response.raise_for_status()
        result = response.json()
        json_content = result['choices'][0]['message']['content']
        if isinstance(json_content, dict):
            dd = DifferentialDiagnosis(**json_content)
        else:
            dd = DifferentialDiagnosis(**json.loads(json_content))
        return dd
