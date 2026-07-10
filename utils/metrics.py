import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torch

def compute_all_metrics(y_true, y_pred, y_probs=None, num_classes=9):
    """
    Computes accuracy, precision, recall, F1, sensitivity, specificity, and AUC-ROC.
    y_true: 1D numpy array of ground truth labels
    y_pred: 1D numpy array of predicted labels
    y_probs: 2D numpy array of class probabilities (shape: N x num_classes)
    """
    # Standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # In medical imaging:
    # Recall = Sensitivity
    sensitivity = recall 
    
    # Specificity calculation per class (one-vs-rest)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specificities = []
    for i in range(num_classes):
        # True Negative (TN), False Positive (FP), False Negative (FN), True Positive (TP)
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    specificity = np.mean(specificities)
    
    # AUC-ROC
    auc = 0.5
    if y_probs is not None:
        try:
            # Handle cases where not all classes are present in y_true during testing (e.g. in small demo mode splits)
            unique_classes = np.unique(y_true)
            if len(unique_classes) == num_classes:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            else:
                # If some classes are missing in ground truth (can happen in small samples), compute AUC only on present classes
                present_probs = y_probs[:, unique_classes]
                # Normalize probabilities
                present_probs = present_probs / (present_probs.sum(axis=1, keepdims=True) + 1e-7)
                auc = roc_auc_score(y_true, present_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning computing AUC: {e}")
            auc = 0.5
            
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc": float(auc),
        "per_class_specificity": [float(s) for s in specificities],
        "confusion_matrix": cm.tolist()
    }

def print_metrics_table(metrics_dict, model_name="Model"):
    """
    Nicely prints evaluation metrics in a table.
    """
    print(f"\n=================== {model_name} Evaluation Metrics ===================")
    print(f"Accuracy:    {metrics_dict['accuracy']:.4f}")
    print(f"Sensitivity: {metrics_dict['sensitivity']:.4f} (Recall)")
    print(f"Specificity: {metrics_dict['specificity']:.4f}")
    print(f"Precision:   {metrics_dict['precision']:.4f}")
    print(f"F1 Score:    {metrics_dict['f1']:.4f}")
    print(f"AUC ROC:     {metrics_dict['auc']:.4f}")
    print("========================================================================\n")
