# import os
# import re
# import json
# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Get OpenRouter API key
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# if not OPENROUTER_API_KEY:
#     raise ValueError("❌ OPENROUTER_API_KEY not found in environment variables (.env file)")

# # OpenRouter endpoint
# OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# def reason_with_llm(imaging_result, clinical_text, kg_context):
#     """
#     Uses OpenRouter API to reason over patient data and return diagnosis in clean JSON.
#     """
#     try:
#         # 🧠 Step 1: Construct the prompt
#         prompt = f"""
#         You are a breast cancer diagnostic reasoning agent.
#         Integrate the following data and reason step-by-step.

#         🧩 Imaging Findings: {imaging_result}
#         🩺 Clinical Notes: {clinical_text}
#         🧠 Knowledge Graph Context: {kg_context}

#         Respond **only** in JSON format with:
#         {{
#             "diagnosis": "<Likely malignant / Likely benign / Uncertain>",
#             "confidence": <float between 0 and 1>,
#             "reasoning_text": "<brief reasoning>"
#         }}
#         """

#         # 🧩 Step 2: Set headers and payload
#         headers = {
#             "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#             "Content-Type": "application/json",
#             "HTTP-Referer": "http://localhost",
#             "X-Title": "ReasoningAgent"
#         }

#         data = {
#             "model": "gpt-4o-mini",
#             "messages": [
#                 {"role": "system", "content": "You are a precise, medical reasoning assistant. Always respond in valid JSON only."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.3
#         }

#         # 🧩 Step 3: Call API
#         response = requests.post(OPENROUTER_URL, headers=headers, json=data)

#         # Handle API errors
#         if response.status_code != 200:
#             return {
#                 "diagnosis": "Unknown",
#                 "confidence": 0.0,
#                 "reasoning_text": f"Error: {response.status_code} - {response.text}",
#                 "kg_context_used": [],
#                 "input_used": {"imaging_result": imaging_result, "clinical_text": clinical_text}
#             }

#         # 🧩 Step 4: Extract model reply
#         content = response.json()["choices"][0]["message"]["content"].strip()

#         # 🧩 Step 5: Clean markdown-wrapped JSON (```json ... ```)
#         content = re.sub(r"^```json\s*|\s*```$", "", content.strip())

#         # 🧩 Step 6: Parse JSON safely
#         try:
#             parsed = json.loads(content)
#         except json.JSONDecodeError:
#             # fallback: try to extract JSON substring
#             match = re.search(r"\{.*\}", content, re.DOTALL)
#             if match:
#                 try:
#                     parsed = json.loads(match.group(0))
#                 except json.JSONDecodeError:
#                     parsed = {
#                         "diagnosis": "Unknown",
#                         "confidence": 0.0,
#                         "reasoning_text": content
#                     }
#             else:
#                 parsed = {
#                     "diagnosis": "Unknown",
#                     "confidence": 0.0,
#                     "reasoning_text": content
#                 }

#         # 🧩 Step 7: Append context and inputs
#         parsed["kg_context_used"] = kg_context
#         parsed["input_used"] = {
#             "imaging_result": imaging_result,
#             "clinical_text": clinical_text
#         }

#         return parsed

#     except Exception as e:
#         # Catch all runtime errors
#         return {
#             "diagnosis": "Unknown",
#             "confidence": 0.0,
#             "reasoning_text": f"Exception occurred: {str(e)}",
#             "kg_context_used": kg_context,
#             "input_used": {"imaging_result": imaging_result, "clinical_text": clinical_text}
#         }
# llm_utils.py - Enhanced LLM Reasoning with Structured KG Context

import os
import re
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in environment variables (.env file)")

# OpenRouter endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def format_kg_context_for_llm(kg_context: dict) -> str:
    """
    Format the KG context into a clear, structured prompt for the LLM
    """
    if not kg_context.get("differential_diagnoses"):
        return "No relevant knowledge graph matches found."
    
    formatted = "📊 KNOWLEDGE GRAPH ANALYSIS:\n\n"
    
    # Identified features
    if kg_context.get("identified_findings"):
        formatted += f"🔍 Imaging Findings Detected: {', '.join(kg_context['identified_findings'])}\n"
    if kg_context.get("identified_symptoms"):
        formatted += f"🩺 Clinical Symptoms: {', '.join(kg_context['identified_symptoms'])}\n"
    if kg_context.get("identified_risk_factors"):
        formatted += f"⚠️ Risk Factors: {', '.join(kg_context['identified_risk_factors'])}\n"
    
    formatted += "\n📋 DIFFERENTIAL DIAGNOSES (from Knowledge Graph):\n\n"
    
    # Top differential diagnoses
    for i, diag in enumerate(kg_context["differential_diagnoses"], 1):
        formatted += f"{i}. {diag['disease_name']} ({diag['malignancy'].upper()})\n"
        formatted += f"   • KG Confidence Score: {diag['total_score']:.2f}\n"
        formatted += f"   • Matched Findings: {', '.join(diag['matched_findings'])}\n"
        
        if diag.get('symptom_match', {}).get('matched_symptoms'):
            formatted += f"   • Matched Symptoms: {', '.join(diag['symptom_match']['matched_symptoms'])}\n"
        
        if diag.get('risk_factors', {}).get('matched_risk_factors'):
            risk_factors = diag['risk_factors']['matched_risk_factors']
            risk_mult = diag['risk_factors']['risk_multiplier']
            formatted += f"   • Risk Factors: {', '.join(risk_factors)} (multiplier: {risk_mult:.1f}x)\n"
        
        formatted += "\n"
    
    return formatted


def reason_with_llm(imaging_result, clinical_text, kg_context):
    """
    Uses OpenRouter API to reason over patient data using structured KG evidence
    """
    try:
        # 🧠 Step 1: Format KG context nicely
        kg_formatted = format_kg_context_for_llm(kg_context)
        
        # 🧠 Step 2: Construct enhanced prompt
        prompt = f"""
You are an expert breast cancer diagnostic reasoning agent. Analyze the following patient data and provide a diagnostic assessment.

{kg_formatted}

📋 RAW IMAGING DATA:
{json.dumps(imaging_result, indent=2)}

📋 CLINICAL NOTES:
{clinical_text}

INSTRUCTIONS:
1. Integrate the Knowledge Graph analysis with the raw data
2. Consider the confidence scores and evidence from the KG
3. Reason step-by-step about the most likely diagnosis
4. Provide a final assessment with confidence level

Respond **only** in JSON format:
{{
    "diagnosis": "Likely malignant / Likely benign / Uncertain / Requires further investigation",
    "confidence": <float between 0 and 1>,
    "reasoning_text": "<detailed step-by-step reasoning using KG evidence>",
    "primary_concern": "<specific disease name if malignant suspected, or 'None' if benign>",
    "recommended_action": "<Next steps: biopsy / follow-up imaging / routine screening>"
}}
"""

        # 🧩 Step 3: Set headers and payload
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "ReasoningAgent"
        }

        data = {
            "model": "openai/gpt-4o-mini",  # Updated model string for OpenRouter
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a precise medical reasoning assistant specializing in breast cancer diagnosis. Always respond in valid JSON only. Use the Knowledge Graph evidence to support your reasoning."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Lower temperature for more deterministic medical reasoning
            "max_tokens": 800
        }

        # 🧩 Step 4: Call API
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=30)

        # Handle API errors
        if response.status_code != 200:
            return {
                "diagnosis": "Unknown",
                "confidence": 0.0,
                "reasoning_text": f"API Error: {response.status_code} - {response.text}",
                "primary_concern": "Error",
                "recommended_action": "System error - retry request"
            }

        # 🧩 Step 5: Extract model reply
        content = response.json()["choices"][0]["message"]["content"].strip()

        # 🧩 Step 6: Clean markdown-wrapped JSON (```json ... ```)
        content = re.sub(r"^```json\s*|\s*```$", "", content.strip())

        # 🧩 Step 7: Parse JSON safely
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON substring
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = {
                        "diagnosis": "Unknown",
                        "confidence": 0.0,
                        "reasoning_text": content,
                        "primary_concern": "Parsing Error",
                        "recommended_action": "Manual review required"
                    }
            else:
                parsed = {
                    "diagnosis": "Unknown",
                    "confidence": 0.0,
                    "reasoning_text": content,
                    "primary_concern": "Parsing Error",
                    "recommended_action": "Manual review required"
                }

        # 🧩 Step 8: Ensure all required fields exist
        parsed.setdefault("diagnosis", "Unknown")
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("reasoning_text", "No reasoning provided")
        parsed.setdefault("primary_concern", "None")
        parsed.setdefault("recommended_action", "Consult with radiologist")

        return parsed

    except requests.exceptions.Timeout:
        return {
            "diagnosis": "Unknown",
            "confidence": 0.0,
            "reasoning_text": "Request timeout - API took too long to respond",
            "primary_concern": "Timeout Error",
            "recommended_action": "Retry request"
        }
    except Exception as e:
        # Catch all runtime errors
        return {
            "diagnosis": "Unknown",
            "confidence": 0.0,
            "reasoning_text": f"Exception occurred: {str(e)}",
            "primary_concern": "System Error",
            "recommended_action": "Check logs and retry"
        }


# ===== OPTIONAL: Fallback reasoning without LLM =====
def reason_without_llm(imaging_result, clinical_text, kg_context):
    """
    Fallback reasoning using only KG if LLM fails
    """
    if not kg_context.get("differential_diagnoses"):
        return {
            "diagnosis": "Uncertain",
            "confidence": 0.5,
            "reasoning_text": "No Knowledge Graph matches found. Unable to provide diagnosis.",
            "primary_concern": "Unknown",
            "recommended_action": "Requires manual review"
        }
    
    top_diagnosis = kg_context["differential_diagnoses"][0]
    
    # Simple rule-based classification
    if top_diagnosis["malignancy"] == "malignant":
        diagnosis = "Likely malignant"
        action = "Urgent: Biopsy recommended"
    elif top_diagnosis["malignancy"] == "pre-malignant":
        diagnosis = "Pre-malignant lesion detected"
        action = "Biopsy recommended for confirmation"
    else:
        diagnosis = "Likely benign"
        action = "Follow-up imaging in 6 months"
    
    confidence = min(top_diagnosis["total_score"] / 20.0, 0.95)  # Normalize score
    
    reasoning = f"Based on Knowledge Graph analysis: {top_diagnosis['disease_name']} is the top match. "
    reasoning += f"Evidence: {', '.join(top_diagnosis['matched_findings'])}."
    
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "reasoning_text": reasoning,
        "primary_concern": top_diagnosis["disease_name"],
        "recommended_action": action
    }


if __name__ == "__main__":
    # Test the formatting function
    test_kg_context = {
        "differential_diagnoses": [
            {
                "disease_id": "IDC",
                "disease_name": "Invasive Ductal Carcinoma",
                "malignancy": "malignant",
                "total_score": 17.0,
                "matched_findings": ["Irregular Mass", "High Density Mass"],
                "symptom_match": {"matched_symptoms": ["Palpable Lump"], "symptom_score": 3.5},
                "risk_factors": {"matched_risk_factors": ["Family History"], "risk_multiplier": 2.0}
            }
        ],
        "identified_findings": ["Irregular Mass", "High Density Mass"],
        "identified_symptoms": ["Palpable Lump"],
        "identified_risk_factors": ["Family History"]
    }
    
    formatted = format_kg_context_for_llm(test_kg_context)
    print(formatted)