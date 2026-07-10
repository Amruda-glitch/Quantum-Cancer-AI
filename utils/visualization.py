import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """
    Plots and saves a styled, normalized confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    cm_normalized = np.array(cm).astype('float') / (np.sum(cm, axis=1)[:, np.newaxis] + 1e-7)
    
    # Custom vibrant blue/purple colormap for premium look
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_roc_curves(y_true, y_probs, class_names, save_path, title="Receiver Operating Characteristic (ROC)"):
    """
    Plots and saves multi-class ROC curves with micro/macro averages.
    """
    plt.figure(figsize=(10, 8))
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Handle binary edge case in stratification
    if y_true_bin.shape[1] == 1:
        # Binarize creates a 1D array if classes = 2, we make it 2D
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        # Check if class has samples in y_true
        if i in y_true or np.sum(y_true_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=1.5, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        else:
            # Fallback for empty classes in validation subsets
            pass
            
    # Compute micro-average ROC curve
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=3)
    except Exception:
        pass
        
    # Plot diagonal reference
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC Curves saved to: {save_path}")

def plot_training_history(history, save_path, title="Training History"):
    """
    Plots and saves training & validation loss and accuracy curves.
    history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', lw=1.5)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', lw=1.5)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', lw=1.5)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', lw=1.5)
    ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history saved to: {save_path}")

def plot_feature_importance(importances, feature_names, save_path, title="Feature Importance Analysis"):
    """
    Plots and saves feature importance ranking.
    """
    # Sort top 20 features
    indices = np.argsort(importances)[::-1][:20]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_names, palette="rocket")
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature ID', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Feature importance saved to: {save_path}")
