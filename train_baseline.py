import os
import sys
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.metrics import compute_all_metrics, print_metrics_table
from utils.visualization import plot_confusion_matrix, plot_roc_curves

def load_features():
    """
    Loads extracted features and labels from disk.
    Concatenates ResNet50 and EfficientNet-B0 features to create a hybrid feature vector.
    """
    try:
        x_train_res = np.load(os.path.join(config.OUTPUT_FEATURES, "resnet_train_features.npy"))
        x_test_res = np.load(os.path.join(config.OUTPUT_FEATURES, "resnet_test_features.npy"))
        
        x_train_eff = np.load(os.path.join(config.OUTPUT_FEATURES, "effnet_train_features.npy"))
        x_test_eff = np.load(os.path.join(config.OUTPUT_FEATURES, "effnet_test_features.npy"))
        
        y_train = np.load(os.path.join(config.OUTPUT_FEATURES, "train_labels.npy"))
        y_test = np.load(os.path.join(config.OUTPUT_FEATURES, "test_labels.npy"))
        
        # Concatenate features along feature dimension
        x_train = np.concatenate([x_train_res, x_train_eff], axis=1)
        x_test = np.concatenate([x_test_res, x_test_eff], axis=1)
        
        print(f"Loaded features successfully:")
        print(f"  Train features shape: {x_train.shape}")
        print(f"  Test features shape: {x_test.shape}")
        return x_train, y_train, x_test, y_test
    except FileNotFoundError:
        print("Feature files not found. Running feature extractor first...")
        from feature_extractor import extract_and_save_features
        extract_and_save_features()
        return load_features()

def train_and_eval_baselines():
    """
    Trains and evaluates Random Forest, SVM, and XGBoost baseline classifiers.
    """
    config.set_seed()
    x_train, y_train, x_test, y_test = load_features()
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config.SEED, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=config.SEED),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=config.SEED, eval_metric='mlogloss', n_jobs=-1)
    }
    
    baseline_results = {}
    
    for name, model in models.items():
        print(f"\nTraining classical baseline: {name}...")
        model.fit(x_train, y_train)
        
        print(f"Evaluating {name}...")
        # Get predictions
        y_pred = model.predict(x_test)
        y_probs = model.predict_proba(x_test)
        
        # Compute metrics
        metrics = compute_all_metrics(y_test, y_pred, y_probs, num_classes=config.NUM_CLASSES)
        print_metrics_table(metrics, name)
        
        # Save metrics (remove numpy array confusion matrix for JSON saving)
        baseline_results[name] = {
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "auc": metrics["auc"]
        }
        
        # Plot Confusion Matrix
        cm_path = os.path.join(config.OUTPUT_CM, f"{name.lower().replace(' ', '_')}_cm.png")
        plot_confusion_matrix(metrics["confusion_matrix"], config.CLASS_NAMES, cm_path, title=f"Confusion Matrix - {name}")
        
        # Plot ROC Curves
        roc_path = os.path.join(config.OUTPUT_ROC, f"{name.lower().replace(' ', '_')}_roc.png")
        plot_roc_curves(y_test, y_probs, config.CLASS_NAMES, roc_path, title=f"ROC Curves - {name}")
        
    # Save baseline results to JSON
    out_path = os.path.join(config.OUTPUT_REPORTS, "baseline_metrics.json")
    with open(out_path, 'w') as f:
        json.dump(baseline_results, f, indent=4)
    print(f"Baseline metrics saved to: {out_path}")
    
    return baseline_results

if __name__ == "__main__":
    train_and_eval_baselines()
