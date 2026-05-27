from pathlib import Path
from PIL import Image
import joblib

import matplotlib.pyplot as plt

from src.vision.feature_extractor import extract_features

MODEL_PATH = Path("models/image_model.joblib")

def load_model():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        raise SystemExit(1)
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")
    return model

def predict_image(model, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return
    with Image.open(path) as image:
        features = extract_features(image)
    prediction = model.predict([features])[0]
    print("=== Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {prediction}")

def predict_image(model, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return
    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()
    prediction = model.predict([features])[0]
    print("=== Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {prediction}")
    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()

def main():
    model = load_model()
    image_path = "data/inference_samples/noise.jpg"    
    predict_image(model, image_path)
if __name__ == "__main__":
    main()