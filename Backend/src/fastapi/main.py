# from fastapi import FastAPI, HTTPException, Header
# from pydantic import BaseModel
# import requests
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import joblib
# from fastapi.middleware.cors import CORSMiddleware

# # Load your trained .pkl model
# model = joblib.load("model.pkl")  # Must exist inside fastapi/ or provide full path

# # FastAPI app setup
# app = FastAPI()

# # CORS (for dev/testing)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Input schema
# class ImageInput(BaseModel):
#     image_url: str

# # Prediction route
# @app.post("/predict")
# def predict_fish(input: ImageInput, Authorization: str = Header(...)):
#     try:
#         # Download the image
#         response = requests.get(input.image_url)
#         image = Image.open(BytesIO(response.content)).convert("RGB")

#         # Preprocess image (resize and flatten)
#         image = image.resize((224, 224))
#         features = np.array(image).flatten().reshape(1, -1)

#         # Prediction
#         species = model.predict(features)[0]
#         confidence = model.predict_proba(features).max() * 100

#         return {
#             "species": species,
#             "confidence": round(confidence, 2)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import numpy as np

# Load model architecture (must match how it was saved)
from transformers import ViTForImageClassification, ViTImageProcessor

# Load model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.classifier = torch.nn.Linear(model.classifier.in_features, 6)  # 6 classes
model.load_state_dict(torch.load("best_vit_model.pth", map_location=torch.device("cpu")))
model.eval()

# Class labels
class_names = ['Catla', 'Cyprinus carpio', 'Grass Carp', 'Mori', 'Rohu', 'Silver']

# Preprocessor
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# --- ResNet50 Setup ---

# Load ResNet50 Keras model
resnet_model = tf.keras.models.load_model("fish_classifier_resnet50.h5")

# ResNet50 Class labels (same as above)
resnet_class_names = ['Catla', 'Cyprinus carpio', 'Grass Carp', 'Mori', 'Rohu', 'Silver']

# ResNet50 Preprocessing
def preprocess_resnet_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class ImageInput(BaseModel):
    image_url: str

# Predict route
@app.post("/predict")
def predict(input: ImageInput, Authorization: str = Header(...)):
    try:
        # Load image from URL
        response = requests.get(input.image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess image
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dim

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        return {
             "model": "VitTransformer",
            "species": class_names[predicted.item()],
            "confidence": round(confidence.item(), 4),
            "all_predictions": {class_names[i]: round(p.item(), 4) for i, p in enumerate(probs[0])}
        }
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/model2")
def predict_resnet(input: ImageInput, Authorization: str = Header(...)):
    try:
        response = requests.get(input.image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        input_tensor = preprocess_resnet_image(image)

        prediction = resnet_model.predict(input_tensor)[0]
        predicted_idx = np.argmax(prediction)
        confidence = float(prediction[predicted_idx])

        return {
            "model": "resnet50",
            "species": resnet_class_names[predicted_idx],
            "confidence": round(confidence, 4),
            "all_predictions": {resnet_class_names[i]: round(float(p), 4) for i, p in enumerate(prediction)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ResNet50 prediction failed: {str(e)}")