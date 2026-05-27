import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def main():
    features_path = "data/processed/model_features.csv"
    labels_path = "data/processed/model_labels.csv"

    print("=== Machine Learning: Loading Feature Dataset ===")
    
    if not os.path.exists(features_path):
        print(f"Error: File not found -> {features_path}")
        return

    print(f"Input file: {features_path}")
    
    X = []
    feature_names = []
    
    with open(features_path, 'r') as f:
        reader = csv.reader(f)
        feature_names = next(reader)
        for row in reader:
            X.append([float(val) for val in row])
            
    print(f"Records loaded: {len(X)}")
    print(f"Columns: {feature_names}")
    print()

    print("=== Machine Learning: Preparing Features and Target ===")
    
    y = []
    
    if not os.path.exists(labels_path):
        print(f"Error: File not found -> {labels_path}")
        return

    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            y.append(int(float(row[0])))

    print(f"Number of samples in X: {len(X)}")
    print(f"Number of labels in y: {len(y)}")
    
    unique_targets = list(set(y))
    print(f"Target values detected: {unique_targets}")
    print()

    print("=== Machine Learning: Train/Test Split ===")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    print("=== Machine Learning: Model Training ===")
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    print("Model: DecisionTreeClassifier")
    print("Training completed successfully.")
    print()

    print("=== Machine Learning: Prediction ===")
    
    predictions = model.predict(X_test)
    
    print("Predictions generated for test set.")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Example predictions: {list(predictions[:5])}")
    print()

    print("=== Machine Learning: Evaluation ===")
    
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()

    print("=== Machine Learning: Saving and Inspecting Model ===")
    
    os.makedirs("results", exist_ok=True)
    
    model_path = "results/decision_tree_model.joblib"
    joblib.dump(model, model_path)
    
    tree_depth = model.get_depth()
    tree_leaves = model.get_n_leaves()
    tree_rules = export_text(model, feature_names=feature_names)
    
    print(f"Saved model: {model_path}")
    print("Model type: DecisionTreeClassifier")
    print(f"Tree depth: {tree_depth}")
    print(f"Number of leaves: {tree_leaves}")
    print("\nDecision Tree Rules:")
    print(tree_rules)
    print()

    print("=== Machine Learning: Saving Evaluation Results ===")
    
    eval_path = "results/model_evaluation.txt"
    
    with open(eval_path, "w") as f:
        f.write("OOAIS Model Evaluation\n")
        f.write("========================\n")
        f.write("Model: DecisionTreeClassifier\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")
        
    print(f"Saved file: {eval_path}")
    print()

    print("=== Machine Learning: Saving Training Report ===")
    
    os.makedirs("reports", exist_ok=True)
    
    report_path = "reports/model_training_summary.txt"
    
    with open(report_path, "w") as f:
        f.write("OOAIS Model Training Summary\n")
        f.write("==============================\n\n")
        f.write("Input datasets\n")
        f.write("--------------\n")
        f.write(f"{features_path}\n")
        f.write(f"{labels_path}\n\n")
        f.write("Dataset statistics\n")
        f.write("------------------\n")
        f.write(f"Number of samples: {len(X)}\n")
        f.write(f"Number of features: {len(feature_names)}\n\n")
        f.write("Model\n")
        f.write("-----\n")
        f.write("DecisionTreeClassifier\n\n")
        f.write("Train/Test split\n")
        f.write("----------------\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write("Evaluation summary\n")
        f.write("------------------\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")
        
    print(f"Saved file: {report_path}")

if __name__ == "__main__":
    main()