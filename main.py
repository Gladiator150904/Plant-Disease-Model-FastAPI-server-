import os.path

import requests
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

MODEL_URL = "https://drive.google.com/uc?id=1Rcz1OE6AKBoduxvZi5TNqY3iXdoc7ppH"
MODEL_PATH = "./model0001.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded")

model = tf.keras.models.load_model(MODEL_PATH)
IMAGE_SIZE = (128,128)
app = FastAPI()

origins = [ "http://localhost:5173" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Algal_Leaf_in_Tea", "Anthracnose_in_Mango", "Anthracnose_in_Tea", "Anthracnose_Leaf_Spot_in_Spinach", "Apple_Scab_on_Apple", "Bacterial_Blight_in_Rice", "Bacterial_Blight_in_Soyabean", "Bacterial_Blight_in_Strawberry","Bacterial_Blight_on_Corn","Bacterial_Spot_in_Bell_Pepper","Bacterial_Spot_in_Peach","Bacterial_Spot_on_Tomato", "Bird_Eye_Spot_on_Tea", "Black_Rot_in_Grapes", "Black_Rot_on_Apple", "Blast_in_Rice", "Brownspot_in_Rice", "Brown_Blight_on_Tea", "Caterpillar_on_Soyabean", "Cedar_Apple_Rust_on_Apple", "Cercospora_Gray_Leaf_Spot_on_Corn", "Cercospora_Leaf_Spot_in_Coffee", "Cercospora_Leaf_Spot_on_Pomegranate", "Citrus_Scab_on_Lemon", "Comon_Rust_on_Corn", "CP_Deficiency_in_Cabbage", "Curl_Virus_in_Cotton", "Diabrotica_Speciosa_on_Soyabean", "Diseased_Cucumber", "Diseased_Guava", "Diseased_Jamun", "Downy_Mildew_on_Soyabean", "Early_Blight_on_Potato", "Early_Blight_on_Tomato", "Esca_(Black_Measles)_in_Grapes", "Fussarium_Wilt_in_Cotton", "Haunglongbing_(Citrus_Greening)_in_Orange", "Healthy_Apple", "Healthy_Bell_Pepper", "Healthy_Brinjal", "Healthy_Cabbage", "Healthy_Cherry", "Healthy_Chili", "Healthy_Coffee", "Healthy_Corn", "Healthy_Cotton", "Healthy_Cucumber", "Healthy_Grapes", "Healthy_Guava", "Healthy_Jamun", "Healthy_Lemon", "Healthy_Mango", "Healthy_Okra", "Healthy_Onion", "Healthy_Peach", "Healthy_Pomegranate", "Healthy_Potato", "Healthy_Soyabean", "Healthy_Spinach", "Healthy_Strawberry", "Healthy_Sugarcane", "Healthy_Tea", "Healthy_Tomato", "Healthy_Wheat", "Late_Blight_on_Potato", "Late_Blight_on_Tomato", "Leafsmut_in_Rice", "Leaf_Blight_(Isariopsis_Leaf_Spot)_in_Grapes", "Leaf_Curl_in_Chili", "Leaf_Mold_on_Tomato", "Leaf_Scorch_in_Strawberry", "Leaf_Spot_in_Chili", "Little_Leaf_on_Brinjal", "Mosaic_Virus_in_Tomato", "Mosaic_Virus_on_Soyabean", "Northern_Leaf_Blight_in_Corn", "Powderly_Mildew_on_Okra", "Powderly_Mildew_on_Onion", "Powdery_Mildew_on_Cherry", "Powdery_Mildew_on_Soyabean", "Red_Leaf_Spot_on_Tea", "Red_Rot_Sugarcane", "Red_Spider_Mite_on_Coffee", "Red_Stripe_in_Sugarcane", "Rust_in_Coffee", "Rust_in_Sugarcane", "Rust_on_Soyabean", "Septoria_Leaf_Spot_on_Tomato", "Septoria_on_Wheat", "Southern_Blight_on_Soyabean", "Spider_Mite_on_Tomato", "Straw_Mite_in_Spinach", "Target_Spot_on_Tomato", "Tungro_in_Rice", "White_Fly_in_Chili", "Yellowish_Chili", "Yellow_Leaf_Curl_Virus_in_Tomato", "Yellow_Rust_in_Wheat"]

for i in range(len(CLASS_NAMES)):
    CLASS_NAMES[i] = CLASS_NAMES[i].replace('_', ' ')


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)

    return image
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    img_batch = img_batch/255.0
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")
