from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from keras.models import load_model
from fastapi import Request
import base64
import io
from PIL import Image
import numpy as np
import os


app = FastAPI()

# Try to load model from different possible locations
MODEL_PATHS = [
    "my_model.keras",  # Root directory
    "best_model.keras",  # Back-end subdirectory
    "back-end/best_model.keras",  # Back-end subdirectory
    "../my_model.keras"  # Parent directory
]

model = None
for model_path in MODEL_PATHS:
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Model loaded successfully from: {model_path}")
            break
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")

if model is None:
    raise RuntimeError("Could not load the trained model from any location!")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        data_url = data.get("data_url")
        if not data_url:
            return {"error": "No data_url provided."}
        
        # Strip the prefix
        if "," in data_url:
            _, b64data = data_url.split(",", 1)
        else:
            b64data = data_url

        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(b64data)
        except Exception as e:
            return {"error": f"Invalid base64 data: {str(e)}"}

        # Open and process image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        except Exception as e:
            return {"error": f"Invalid image data: {str(e)}"}

        # Resize and preprocess as in training
        image = image.resize((28, 28))
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        #view image for debugging
        #image.show()  # Uncomment this line to view the image during debugging

        # Predict
        preds = model.predict(img_array, verbose=0)  # Suppress verbose output
        
        # Convert logits to probabilities using softmax
        logits = preds[0]
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probs = (exp_logits / np.sum(exp_logits)).tolist()
        
        predicted = int(np.argmax(probs))

        return {"digit": predicted, "probs": probs}
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
