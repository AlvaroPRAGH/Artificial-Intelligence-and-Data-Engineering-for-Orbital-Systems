import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

#task1
def validate_input_files():
    features_file = Path("data/processed/model_features.csv")
    labels_file = Path("data/processed/model_labels.csv")

    if not features_file.exists() or not labels_file.exists():
        print("Error: missing required input file(s):")
        print("- data/processed/model_features.csv")
        print("- data/processed/model_labels.csv")
        exit(1)

#task2
def load_data():
    features_df = pd.read_csv("data/processed/model_features.csv")
    labels_df = pd.read_csv("data/processed/model_labels.csv")

    print("=== Model Playground: Loading Data ===")
    print("Feature file: data/processed/model_features.csv")
    print("Label file: data/processed/model_labels.csv")

    return features_df, labels_df

#task3
def inspect_data(features_df, labels_df):
    if features_df.empty or labels_df.empty:
        print("Error: One or both datasets are empty.")
        sys.exit(1)
    
    if len(features_df) != len(labels_df):
        print("Error: Datasets do not contain the same number of rows.")
        sys.exit(1)

    if "anomaly_flag" not in labels_df.columns:
        print("Error: Label dataset is missing 'anomaly_flag' column.")
        sys.exit(1)

    print("=== Model Playground: Data Inspection ===")
    print("Number of samples:", len(features_df))
    print("Number of features: ", features_df.shape[1])
    print("Feature columns: ", list(features_df.columns))
    print("Target values detected: ", labels_df["anomaly_flag"].unique())

#task4
def prepare_features_and_labels(features_df, labels_df):
    X = features_df.values
    y = labels_df["anomaly_flag"].astype(int).values

    print("=== Model Playground: Preparing Features and Labels ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y

#task 5
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("=== Model Playground: Train/Test Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test

#task6
def define_models():
    models = {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    return models

#task7
def train_models(models, X_train, y_train):
    print("=== Model Playground: Training Models ===")
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{model_name}: trained")
        trained_models[model_name] = model
    print()
    return trained_models

#task8
def generate_predictions(trained_models, X_test):
    results = []
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        result = {
            "name": model_name,
            "model": model,
            "y_pred": y_pred
        }
        results.append(result)
    return results

#task9
def print_example_predictions(prediction_results, y_test, num_examples=5):
    print("=== Model Playground: Example Predictions ===")
    for i in range(num_examples):
        line = f"True: {y_test[i]}"
        for result in prediction_results:
            model_name = result["name"]
            y_pred = result["y_pred"]
            line += f" | {model_name}: {y_pred[i]}"
        print(line)

#task10
def compute_accuracy(prediction_results, y_test):
    print("=== Model Playground: Accuracy Comparison ===")
    for result in prediction_results:
        y_pred = result["y_pred"]
        accuracy = accuracy_score(y_test, y_pred)
        result["accuracy"] = accuracy
        print(f"{result['name']}: {accuracy:.4f}")
    return prediction_results

#task11
def compute_detailed_metrics(prediction_results, y_test):
    print("=== Model Playground: Detailed Evaluation ===")
    for result in prediction_results:
        y_pred = result["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        result["confusion_matrix"] = cm
        result["classification_report"] = report
        
        print(f"\nModel: {result['name']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClass labels:")
        print("0 -> normal observation")
        print("1 -> anomaly")
        print("\nClassification Report:")
        print("--------------------------------------------------")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support'}")
        print("--------------------------------------------------")
        
        for class_label in ['0', '1']:
            name = "0 (normal)" if class_label == '0' else "1 (anomaly)"
            if class_label in report:
                print(f"{name:<15} {report[class_label]['precision']:.2f}       {report[class_label]['recall']:.2f}       {report[class_label]['f1-score']:.2f}       {int(report[class_label]['support'])}")
        
        print("--------------------------------------------------")
        print(f"{'Macro average':<15} {report['macro avg']['precision']:.2f}       {report['macro avg']['recall']:.2f}       {report['macro avg']['f1-score']:.2f}       {int(report['macro avg']['support'])}")
        print(f"{'Weighted avg':<15} {report['weighted avg']['precision']:.2f}       {report['weighted avg']['recall']:.2f}       {report['weighted avg']['f1-score']:.2f}       {int(report['weighted avg']['support'])}")
    print()
    return prediction_results

#task12
def rank_models(evaluation_results):
    print("=== Model Playground: Ranking ===")
    sorted_results = sorted(
        evaluation_results,
        key=lambda result: result["accuracy"],
        reverse=True
    )
    for index, result in enumerate(sorted_results, start=1):
        print(f"{index}. {result['name']} - {result['accuracy']:.4f}")
    print()
    return sorted_results

#task13
def run_experiments(X_train, y_train, X_test, y_test):
    print("=== Model Playground: Controlled Experiments ===")
    experiment_results = []
    
    # Experiment 1: Decision Tree Depth
    depths = [2, 3, 5]
    dt_accuracies = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        dt_accuracies.append(acc)
        name = f"Decision Tree (max_depth={depth})"
        experiment_results.append({"name": name, "accuracy": acc})
        print(f"{name}: {acc:.4f}")
        
    # Experiment 2: Random Forest Size
    n_estimators_list = [5, 10, 50]
    rf_accuracies = []
    for n in n_estimators_list:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rf_accuracies.append(acc)
        name = f"Random Forest (n_estimators={n})"
        experiment_results.append({"name": name, "accuracy": acc})
        print(f"{name}: {acc:.4f}")
        
    # Plotting for Exploratory Analysis
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(depths, dt_accuracies, marker='o', color='blue')
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Performance")
    
    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_list, rf_accuracies, marker='o', color='green')
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Performance")
    
    plt.tight_layout()
    plt.show()
    
    return experiment_results

#tassk14
def save_experiment_summary(features_path, labels_path, X, X_train, X_test, ranked_models, experiment_results):
    Path("reports").mkdir(exist_ok=True)
    summary_path = "reports/model_playground_summary.txt"
    
    with open(summary_path, "w") as f:
        f.write("OOAIS Model Playground Summary\n")
        f.write("=============================\n\n")
        f.write("Input datasets\n")
        f.write(f"- {features_path}\n")
        f.write(f"- {labels_path}\n\n")
        
        f.write("Dataset statistics\n")
        f.write(f"Number of samples: {X.shape[0]}\n")
        f.write(f"Number of features: {X.shape[1]}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n\n")
        
        f.write("Compared models\n")
        for res in ranked_models:
            f.write(f"{res['name']}: {res['accuracy']:.4f}\n")
            
        best_model = ranked_models[0]
        f.write("\nBest model\n")
        f.write(f"{best_model['name']} achieved the highest accuracy: {best_model['accuracy']:.4f}\n\n")
        
        f.write("Additional experiments\n")
        for exp in experiment_results:
            f.write(f"{exp['name']}: {exp['accuracy']:.4f}\n")
            
        f.write("\nConclusion\n")
        f.write(f"The best candidate for further experiments is {best_model['name']}, "
                "because it achieved the highest accuracy on the current test set.\n")
                
    print("\n=== Model Playground: Saving Summary ===")
    print(f"Saved file: {summary_path}")

def create_metric_plots(ranked_models):
    Path("reports").mkdir(exist_ok=True)
    model_names = [result["name"] for result in ranked_models]
    accuracies = [result["accuracy"] for result in ranked_models]
    
    # Safe extraction in case class '1' is missing in some poorly performing model
    precisions = [result["classification_report"].get("1", {}).get("precision", 0) for result in ranked_models]
    recalls = [result["classification_report"].get("1", {}).get("recall", 0) for result in ranked_models]
    f1_scores = [result["classification_report"].get("1", {}).get("f1-score", 0) for result in ranked_models]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title("Accuracy")
    
    axes[0, 1].bar(model_names, precisions, color='lightgreen')
    axes[0, 1].set_title("Precision (Anomaly)")
    
    axes[1, 0].bar(model_names, recalls, color='salmon')
    axes[1, 0].set_title("Recall (Anomaly)")
    
    axes[1, 1].bar(model_names, f1_scores, color='gold')
    axes[1, 1].set_title("F1-score (Anomaly)")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = "reports/model_comparison_panel.png"
    plt.savefig(plot_path)
    plt.close()
    
    print("=== Model Playground: Saving Visualizations ===")
    print(f"Saved file: {plot_path}\n")



if __name__ == "__main__":
    validate_input_files()

    features_df, labels_df = load_data()

    inspect_data(features_df, labels_df)

    X, y = prepare_features_and_labels(features_df, labels_df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = define_models()

    trained_models = train_models(models, X_train, y_train)

    prediction_results = generate_predictions(trained_models, X_test)

    print_example_predictions(prediction_results, y_test, num_examples=5)

    prediction_results = compute_accuracy(prediction_results, y_test)

    prediction_results = compute_detailed_metrics(prediction_results, y_test)

    ranked_models = rank_models(prediction_results)

    experiment_results = run_experiments(X_train, y_train, X_test, y_test)

    save_experiment_summary(
        "data/processed/model_features.csv", 
        "data/processed/model_labels.csv", 
        X, X_train, X_test, ranked_models, experiment_results
    )
    create_metric_plots(ranked_models)