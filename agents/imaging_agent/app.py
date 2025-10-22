from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Imaging Agent running"}

@app.get("/process")
def process_image():
    # Placeholder preprocessing result
    print("Processing image...")
    return {"preprocessing": "Image normalized and resized"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5008)  # Set your port here