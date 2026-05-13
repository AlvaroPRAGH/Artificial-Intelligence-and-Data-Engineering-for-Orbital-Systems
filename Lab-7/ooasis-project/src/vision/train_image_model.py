from pathlib import Path
from PIL import Image
import numpy as np

from src.vision.feature_extractor import extract_features
from sklearn.metrics import accuracy_score

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import time

import matplotlib.pyplot as plt
import numpy as np

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=3
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "SVM": SVC()
}

MODEL_PATH = Path("models/image_model.joblib")
DATASET_DIR = Path("data/processed/images")

def load_image_split(split_dir):
    X = []
    y = []
    class_dirs = sorted([
        path for path in split_dir.iterdir()
        if path.is_dir()
    ])
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([
            path for path in class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)
            X.append(features)
            y.append(class_name)
    X = np.array(X)
    y = np.array(y)
    return X, y

def load_training_and_test_data():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"

    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)

    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}\n")
    return X_train, X_test, y_train, y_test

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\n=== Saving Best Model ===")
    print(f"Saved model to: {MODEL_PATH}")

def plot_accuracy_vs_time(results):
    """Task 11.4: Create Accuracy vs Training Time Plot"""
    plt.figure(figsize=(10, 6))
    
    for result in results:
        plt.scatter(
            result["training_time"], 
            result["accuracy"], 
            label=result["model_name"], 
            s=100 # marker size
        )
        # Add labels to the points
        plt.annotate(
            result["model_name"], 
            (result["training_time"], result["accuracy"]), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center'
        )
        
    plt.title('Model Accuracy vs. Training Time')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save and display the plot
    plt.savefig("accuracy_vs_time.png")
    plt.show()

def visualize_predictions(image_path, results):
    """Task 11.5: Visualize Predictions from All Models"""
    print(f"\nImage: {image_path}")
    
    # Extract just the models from the results list
    trained_models = {res["model_name"]: res["model"] for res in results}
    
    with Image.open(image_path) as img:
        # Extract features (using the same method as training)
        features = extract_features(img)
        # Reshape for a single prediction: expected shape is (1, n_features)
        features = np.array(features).reshape(1, -1)
        
        title_parts = []
        
        # Print to console and gather title parts
        for model_name, model in trained_models.items():
            prediction = model.predict(features)[0]
            print(f"{model_name}: {prediction}")
            title_parts.append(f"{model_name}: {prediction}")
            
        # Create visual output
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off') # Hide axes
        
        # Format title exactly as requested
        title = " | ".join(title_parts)
        plt.title(title, fontsize=12, pad=15)
        
        plt.tight_layout()
        plt.show()

def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()
    
    results = []
    
    print("=== Training and Evaluating Models ===")
    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            "model_name": model_name,
            "accuracy": accuracy,
            "training_time": training_time,
            "model": model
        })
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.4f} seconds")

    # Save the best model
    best_result = max(results, key=lambda x: x["accuracy"])
    save_model(best_result["model"])
    
    # --- NEW CALLS ---
    
    # Run Task 11.4
    plot_accuracy_vs_time(results)
    
    # Run Task 11.5 (Select an image from your dataset)
    sample_image_path = DATASET_DIR / "test" / "forest" / "forest_0000.jpg"
    if sample_image_path.exists():
        visualize_predictions(sample_image_path, results)
    else:
        print(f"\nWarning: Could not find image at {sample_image_path} for visualization.")

if __name__ == "__main__":
    main()