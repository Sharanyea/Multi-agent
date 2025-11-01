# from fastapi import FastAPI, Request
# import uvicorn
# import random
# from kg_utils import get_kg_context       # ✅ use new dynamic KG context
# from llm_utils import reason_with_llm
# from dummy_inputs import generate_dummy_input

# app = FastAPI(title="Reasoning Agent")

# @app.get("/")
# def root():
#     return {"message": "Reasoning Agent running"}

# # Optional health check
# @app.get("/predict")
# def predict():
#     return {"prediction": "Likely benign", "confidence": 0.85}


# @app.post("/reason")
# async def reason_endpoint(request: Request):
#     data = await request.json()
#     imaging_result = data.get("imaging_result")
#     clinical_text = data.get("clinical_text")

#     # ✅ 1. Use dummy input if none provided
#     if imaging_result is None or clinical_text is None:
#         dummy_data = generate_dummy_input()
#         imaging_result = dummy_data["image_features"]
#         clinical_text = dummy_data["clinical_notes"]

#     # ✅ 2. Fetch REAL KG context dynamically
#     kg_context = get_kg_context(imaging_result, clinical_text)

#     # ✅ 3. Call the Reasoning LLM with real KG context
#     reasoning_output = reason_with_llm(imaging_result, clinical_text, kg_context)

#     # ✅ 4. Add simulated or returned confidence
#     confidence = reasoning_output.get("confidence", round(random.uniform(0.7, 0.95), 2))

#     # ✅ 5. Final structured response
#     response = {
#         "diagnosis": reasoning_output.get("diagnosis", "Unknown"),
#         "confidence": confidence,
#         "reasoning_text": reasoning_output.get(
#             "reasoning_text",
#             "Reasoning not available. Check LLM connection."
#         ),
#         "kg_context_used": kg_context,
#         "input_used": {
#             "imaging_result": imaging_result,
#             "clinical_text": clinical_text
#         }
#     }

#     return response


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5007)
# app.py - Enhanced Reasoning Agent with Knowledge Graph Integration

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import random
from kg_utils import get_kg_context       # ✅ Enhanced KG with NetworkX
from llm_utils import reason_with_llm, reason_without_llm
from dummy_inputs import generate_dummy_input

app = FastAPI(
    title="Breast Cancer Reasoning Agent",
    description="Multi-modal reasoning agent with Knowledge Graph integration",
    version="2.0.0"
)

# ===== HEALTH CHECK ENDPOINTS =====

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Reasoning Agent running",
        "version": "2.0.0",
        "status": "online",
        "features": [
            "Knowledge Graph reasoning",
            "LLM-based diagnosis",
            "Multi-modal data fusion"
        ]
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    try:
        # Test KG initialization
        test_kg = get_kg_context(
            {"mass_size": "small", "density": "low", "calcifications": "none"},
            "test"
        )
        kg_status = "✅ OK" if test_kg else "❌ Failed"
    except Exception as e:
        kg_status = f"❌ Error: {str(e)}"
    
    return {
        "status": "healthy",
        "knowledge_graph": kg_status,
        "endpoints": {
            "/": "Root endpoint",
            "/health": "Health check",
            "/reason": "Main reasoning endpoint (POST)",
            "/predict": "Legacy endpoint",
        }
    }

# ===== LEGACY ENDPOINT (backward compatible) =====

@app.get("/predict")
def predict():
    """Legacy prediction endpoint for backward compatibility"""
    return {
        "prediction": "Likely benign",
        "confidence": 0.85,
        "note": "This is a legacy endpoint. Use POST /reason for full diagnosis."
    }


# ===== MAIN REASONING ENDPOINT =====

@app.post("/reason")
async def reason_endpoint(request: Request):
    """
    Main reasoning endpoint
    
    Expected input:
    {
        "imaging_result": {
            "mass_size": "small|medium|large",
            "density": "low|medium|high",
            "calcifications": "none|micro|macro"
        },
        "clinical_text": "Clinical notes as text"
    }
    
    Returns:
    {
        "diagnosis": "Likely malignant / Likely benign / Uncertain",
        "confidence": 0.0-1.0,
        "reasoning_text": "Detailed reasoning",
        "primary_concern": "Disease name if malignant",
        "recommended_action": "Next steps",
        "kg_context_used": {...},
        "input_used": {...}
    }
    """
    try:
        data = await request.json()
        imaging_result = data.get("imaging_result")
        clinical_text = data.get("clinical_text")

        # ✅ 1. Use dummy input if none provided
        if imaging_result is None or clinical_text is None:
            print("⚠️  No input provided, using dummy data...")
            dummy_data = generate_dummy_input()
            imaging_result = dummy_data["image_features"]
            clinical_text = dummy_data["clinical_notes"]
            used_dummy = True
        else:
            used_dummy = False

        # ✅ 2. Fetch REAL KG context dynamically
        print(f"🔍 Querying Knowledge Graph...")
        kg_context = get_kg_context(imaging_result, clinical_text)
        
        print(f"📊 KG found {len(kg_context.get('differential_diagnoses', []))} differential diagnoses")

        # ✅ 3. Call the Reasoning LLM with real KG context
        print(f"🧠 Calling LLM for reasoning...")
        reasoning_output = reason_with_llm(imaging_result, clinical_text, kg_context)

        # ✅ 4. Add confidence if not provided
        if "confidence" not in reasoning_output or reasoning_output["confidence"] == 0.0:
            # Use KG score as fallback
            if kg_context.get('differential_diagnoses'):
                top_score = kg_context['differential_diagnoses'][0]['total_score']
                reasoning_output["confidence"] = min(top_score / 20.0, 0.95)
            else:
                reasoning_output["confidence"] = 0.5

        # ✅ 5. Final structured response
        response = {
            "diagnosis": reasoning_output.get("diagnosis", "Unknown"),
            "confidence": round(float(reasoning_output.get("confidence", 0.5)), 2),
            "reasoning_text": reasoning_output.get("reasoning_text", "Reasoning not available"),
            "primary_concern": reasoning_output.get("primary_concern", "None"),
            "recommended_action": reasoning_output.get("recommended_action", "Consult radiologist"),
            "kg_context_used": kg_context,
            "input_used": {
                "imaging_result": imaging_result,
                "clinical_text": clinical_text,
                "used_dummy_data": used_dummy
            }
        }

        print(f"✅ Reasoning complete: {response['diagnosis']} (confidence: {response['confidence']})")
        return response

    except Exception as e:
        print(f"❌ Error in reasoning endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "diagnosis": "Unknown",
                "confidence": 0.0
            }
        )


# ===== OPTIONAL: KG INSPECTION ENDPOINT =====

@app.post("/kg-only")
async def kg_only_endpoint(request: Request):
    """
    Test Knowledge Graph reasoning without LLM
    Useful for debugging
    """
    try:
        data = await request.json()
        imaging_result = data.get("imaging_result")
        clinical_text = data.get("clinical_text")

        if not imaging_result or not clinical_text:
            dummy_data = generate_dummy_input()
            imaging_result = dummy_data["image_features"]
            clinical_text = dummy_data["clinical_notes"]

        # Only get KG context
        kg_context = get_kg_context(imaging_result, clinical_text)
        
        # Use fallback reasoning (no LLM)
        fallback_result = reason_without_llm(imaging_result, clinical_text, kg_context)
        
        return {
            "kg_context": kg_context,
            "fallback_diagnosis": fallback_result,
            "note": "This endpoint shows KG reasoning only, without LLM enhancement"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== ERROR HANDLERS =====

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )


# ===== STARTUP EVENT =====

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("=" * 70)
    print("🏥 BREAST CANCER REASONING AGENT - STARTING UP")
    print("=" * 70)
    
    # Test KG initialization
    try:
        test_kg = get_kg_context(
            {"mass_size": "small", "density": "low", "calcifications": "none"},
            "test initialization"
        )
        print("✅ Knowledge Graph initialized successfully")
        print(f"   Found {len(test_kg.get('differential_diagnoses', []))} diagnoses in test query")
    except Exception as e:
        print(f"⚠️  Warning: KG initialization issue: {e}")
    
    print("\n📡 Server ready on http://localhost:5007")
    print("📚 API Documentation: http://localhost:5007/docs")
    print("=" * 70)


# ===== MAIN =====

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     BREAST CANCER REASONING AGENT with Knowledge Graph       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5007,
        log_level="info"
    )