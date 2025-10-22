from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Imaging Agent running"}

@app.get("/process")
def process_image():
    # Placeholder preprocessing result
    return {"preprocessing": "Image normalized and resized"}
