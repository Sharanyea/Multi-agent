from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from pathlib import Path
import traceback
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==================== ImagingAgent Class ====================
class ImagingAgent:
    """AI-powered imaging agent for breast cancer diagnosis"""
    
    def __init__(self, model_path=None):
        """Initialize the imaging agent"""
        self.img_height = 256
        self.img_width = 256
        self.class_names = ['benign', 'malignant', 'normal']
        self.num_classes = len(self.class_names)
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            self.model = load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, img_path):
        """Preprocess image for model input"""
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_width, self.img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    
    def predict(self, img_path):
        """Make prediction on image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        class_probabilities = {
            self.class_names[i]: float(predictions[0][i])
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
    
    def analyze_image(self, img_path):
        """Analyze image and provide detailed diagnosis"""
        prediction = self.predict(img_path)
        predicted_class = prediction['predicted_class']
        probability = prediction['confidence']
        
        if predicted_class == "benign":
            result = {
                "predicted_class": predicted_class,
                "confidence": probability,
                "class_probabilities": prediction['class_probabilities'],
                "findings": {
                    "mass_detected": True,
                    "mass_characteristics": {
                        "size": "small to moderate",
                        "shape": "oval or round",
                        "margin": "circumscribed (well-defined)",
                    },
                    "calcifications": "none or macrocalcifications",
                    "tissue_density": "low to moderate",
                    "recommendation": "Follow-up imaging in 6 months recommended"
                }
            }
        elif predicted_class == "malignant":
            result = {
                "predicted_class": predicted_class,
                "confidence": probability,
                "class_probabilities": prediction['class_probabilities'],
                "findings": {
                    "mass_detected": True,
                    "mass_characteristics": {
                        "size": "variable",
                        "shape": "irregular",
                        "margin": "spiculated or ill-defined",
                    },
                    "calcifications": "microcalcifications possible",
                    "tissue_density": "moderate to high",
                    "recommendation": "Urgent biopsy recommended - consult oncologist"
                }
            }
        else:  # normal
            result = {
                "predicted_class": predicted_class,
                "confidence": probability,
                "class_probabilities": prediction['class_probabilities'],
                "findings": {
                    "mass_detected": False,
                    "mass_characteristics": {
                        "size": "none",
                        "shape": "none",
                        "margin": "none",
                    },
                    "calcifications": "none",
                    "tissue_density": "normal",
                    "recommendation": "No abnormalities detected - routine screening recommended"
                }
            }
        
        return result

# ==================== FastAPI App ====================
app = FastAPI(title="Imaging Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_PATH = "best_model.h5"

# Initialize imaging agent
imaging_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the imaging agent on startup"""
    global imaging_agent
    try:
        if os.path.exists(MODEL_PATH):
            imaging_agent = ImagingAgent(model_path=MODEL_PATH)
            print(f"✓ Loaded existing model from {MODEL_PATH}")
        else:
            imaging_agent = ImagingAgent()
            print("⚠ Model file not found. Please train the model first.")
            print(f"Expected model path: {os.path.abspath(MODEL_PATH)}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        imaging_agent = ImagingAgent()
        print("✓ Created new model instance (needs training)")

@app.get("/")
async def root():
    """Health check endpoint"""
    model_loaded = imaging_agent is not None and imaging_agent.model is not None
    return {
        "status": "active",
        "service": "Imaging Agent API",
        "model_loaded": model_loaded,
        "classes": imaging_agent.class_names if imaging_agent else [],
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an ultrasound image for diagnosis"""
    if not imaging_agent:
        raise HTTPException(status_code=500, detail="Imaging agent not initialized")
    
    if not imaging_agent.model:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please train the model first using train_imaging_agent.py"
        )
    
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        file_path = UPLOAD_FOLDER / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = imaging_agent.analyze_image(str(file_path))
        
        return {
            "success": True,
            "filename": file.filename,
            "diagnosis": result
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        )

@app.get("/diagnose")
async def diagnose():
    """Test endpoint for diagnosis (uses the most recent uploaded image)"""
    if not imaging_agent:
        return {"status": "error", "message": "Imaging agent not initialized"}
    
    if not imaging_agent.model:
        return {"status": "error", "message": "Model not loaded. Please train the model first."}
    
    uploaded_files = list(UPLOAD_FOLDER.glob("*"))
    image_files = [f for f in uploaded_files if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}]
    
    if not image_files:
        return {
            "status": "waiting",
            "message": "No image uploaded yet. Please upload an image first using POST /upload",
            "available_classes": imaging_agent.class_names
        }
    
    latest_image = max(image_files, key=lambda x: x.stat().st_mtime)
    
    try:
        result = imaging_agent.analyze_image(str(latest_image))
        return {
            "status": "success",
            "image_analyzed": latest_image.name,
            "diagnosis": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/model-info")
async def model_info():
    """Get information about the model"""
    if not imaging_agent:
        raise HTTPException(status_code=500, detail="Imaging agent not initialized")
    
    return {
        "model_type": "MobileNetV2 Transfer Learning",
        "classes": imaging_agent.class_names,
        "num_classes": imaging_agent.num_classes,
        "input_size": f"{imaging_agent.img_height}x{imaging_agent.img_width}",
        "model_path": MODEL_PATH,
        "model_loaded": imaging_agent.model is not None,
        "model_file_exists": os.path.exists(MODEL_PATH)
    }

@app.delete("/clear-uploads")
async def clear_uploads():
    """Clear all uploaded images"""
    try:
        count = 0
        for file in UPLOAD_FOLDER.glob("*"):
            if file.is_file():
                os.remove(file)
                count += 1
        return {"success": True, "message": f"Cleared {count} uploaded file(s)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Imaging Agent API on port 5010...")
    print("📊 Classes: Benign, Malignant, Normal")
    print(f"📁 Model path: {os.path.abspath(MODEL_PATH)}")
    uvicorn.run(app, host="0.0.0.0", port=5010)