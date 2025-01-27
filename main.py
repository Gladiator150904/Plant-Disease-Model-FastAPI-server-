import os
import requests
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import gdown

# Initialize FastAPI app
app = FastAPI()

# Allow all origins for CORS (adjust as necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Model URL and path
MODEL_URL = "https://drive.google.com/uc?id=1Rcz1OE6AKBoduxvZi5TNqY3iXdoc7ppH"
MODEL_PATH = "./model0001.keras"

# Class names for predictions
CLASS_NAMES = [
    "Algal_Leaf_in_Tea", "Anthracnose_in_Mango", "Anthracnose_in_Tea",
    "Anthracnose_Leaf_Spot_in_Spinach", "Apple_Scab_on_Apple", "Bacterial_Blight_in_Rice",
    # Add more class names as needed
]

# Download the model if not already present
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
    print("Downloading model using gdown...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000:
        print(f"Model downloaded successfully to {MODEL_PATH}.")
    else:
        print("Download failed or file is corrupted.")
        exit(1)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Image size to which the uploaded image will be resized
IMAGE_SIZE = (128, 128)

def read_file_as_image(data) -> np.ndarray:
    """Reads image data, converts it to RGB, resizes, and returns as numpy array."""
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    return image

@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    print("Content type:", file.content_type)
    """Endpoint to predict the class of a plant disease from an image."""
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension
    img_batch = img_batch / 255.0  # Normalize the image

    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
