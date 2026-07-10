import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import get_dataloaders
from models.resnet_model import get_resnet_model
from models.efficientnet_model import get_efficientnet_model
from models.hybrid_model import HybridCancerEnsembleModel
from utils.metrics import compute_all_metrics, print_metrics_table
from utils.visualization import plot_confusion_matrix, plot_roc_curves, plot_training_history

class EarlyStopping:
    """
    Early stopping helper class to terminate training when validation loss stops improving.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(model_name, model, train_loader, val_loader, checkpoint_name):
    print(f"\n=================== Training {model_name} ===================")
    config.set_seed()
    
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Warm up classifier head (backbone frozen)
    print("Phase 1: Warmup classifier head (backbone frozen)...")
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone(True)
        
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE * 5, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    early_stopping = EarlyStopping(patience=5)
    
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(config.OUTPUT_MODELS, checkpoint_name)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    warmup_epochs = max(1, config.EPOCHS // 5)
    
    for epoch in range(1, config.EPOCHS + 1):
        # Transition to Phase 2: Unfreeze top layers and do fine-tuning
        if epoch == warmup_epochs + 1:
            print("\nPhase 2: Fine-tuning backbone (unfreezing top blocks)...")
            if hasattr(model, "unfreeze_last_layers"):
                model.unfreeze_last_layers(unfreeze_blocks=1)
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                    
            # Re-initialize optimizer with differential learning rates
            # Lower learning rate for backbone parameters, higher for classifier
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": config.LEARNING_RATE / 10},
                {"params": model.classifier.parameters(), "lr": config.LEARNING_RATE}
            ], weight_decay=config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - warmup_epochs)
            
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        # Get active learning rate
        active_lr = optimizer.param_groups[-1]['lr']
        print(f"Epoch {epoch}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Classifier LR: {active_lr:.6f}")
              
        # Checkpoint Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path}")
            
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("  --> Early stopping triggered.")
            break
            
    # Load best checkpoint before returning
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Save training curves
    plot_training_history(history, os.path.join(config.OUTPUT_REPORTS, f"{model_name.lower()}_history.png"), title=f"Training curves - {model_name}")
    
    return model, history

def evaluate_model_on_test(model_name, model, test_loader, is_ensemble=False):
    print(f"\nEvaluating {model_name} on test set...")
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            
            if is_ensemble:
                probs = model(imgs) # Ensemble directly returns probabilities
                preds = torch.argmax(probs, dim=1)
            else:
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    metrics = compute_all_metrics(y_true, y_pred, y_probs, num_classes=config.NUM_CLASSES)
    print_metrics_table(metrics, model_name)
    
    # Save Confusion Matrix
    cm_path = os.path.join(config.OUTPUT_CM, f"{model_name.lower().replace(' ', '_')}_cm.png")
    plot_confusion_matrix(metrics["confusion_matrix"], config.CLASS_NAMES, cm_path, title=f"Confusion Matrix - {model_name}")
    
    # Save ROC Curves
    roc_path = os.path.join(config.OUTPUT_ROC, f"{model_name.lower().replace(' ', '_')}_roc.png")
    plot_roc_curves(y_true, y_probs, config.CLASS_NAMES, roc_path, title=f"ROC Curves - {model_name}")
    
    return metrics

def run_deep_training():
    config.set_seed()
    
    # 1. Load Data loaders
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 2. Train Model A: ResNet50
    resnet = get_resnet_model(pretrained=True).to(config.DEVICE)
    resnet, resnet_hist = train_model("ResNet50", resnet, train_loader, val_loader, "best_resnet.pth")
    resnet_metrics = evaluate_model_on_test("ResNet50", resnet, test_loader)
    
    # 3. Train Model B: EfficientNet-B0
    effnet = get_efficientnet_model(pretrained=True).to(config.DEVICE)
    effnet, effnet_hist = train_model("EfficientNet", effnet, train_loader, val_loader, "best_efficientnet.pth")
    effnet_metrics = evaluate_model_on_test("EfficientNet", effnet, test_loader)
    
    # 4. Evaluate Model C: Hybrid Ensemble Model (combining ResNet50 and EfficientNet-B0)
    resnet_path = os.path.join(config.OUTPUT_MODELS, "best_resnet.pth")
    effnet_path = os.path.join(config.OUTPUT_MODELS, "best_efficientnet.pth")
    hybrid = HybridCancerEnsembleModel(resnet_path=resnet_path, effnet_path=effnet_path, device=config.DEVICE)
    hybrid_metrics = evaluate_model_on_test("Hybrid Ensemble", hybrid, test_loader, is_ensemble=True)
    
    # Save deep learning metrics to file
    deep_metrics = {
        "ResNet50": {
            "accuracy": resnet_metrics["accuracy"],
            "sensitivity": resnet_metrics["sensitivity"],
            "specificity": resnet_metrics["specificity"],
            "precision": resnet_metrics["precision"],
            "f1": resnet_metrics["f1"],
            "auc": resnet_metrics["auc"]
        },
        "EfficientNet": {
            "accuracy": effnet_metrics["accuracy"],
            "sensitivity": effnet_metrics["sensitivity"],
            "specificity": effnet_metrics["specificity"],
            "precision": effnet_metrics["precision"],
            "f1": effnet_metrics["f1"],
            "auc": effnet_metrics["auc"]
        },
        "Hybrid Ensemble": {
            "accuracy": hybrid_metrics["accuracy"],
            "sensitivity": hybrid_metrics["sensitivity"],
            "specificity": hybrid_metrics["specificity"],
            "precision": hybrid_metrics["precision"],
            "f1": hybrid_metrics["f1"],
            "auc": hybrid_metrics["auc"]
        }
    }
    
    out_path = os.path.join(config.OUTPUT_REPORTS, "deep_learning_metrics.json")
    with open(out_path, 'w') as f:
        json.dump(deep_metrics, f, indent=4)
    print(f"Deep learning metrics saved to: {out_path}")
    
    return deep_metrics

if __name__ == "__main__":
    # Ensure splits are created
    if not os.path.exists(config.SPLIT_CSV):
        from utils.dataset_split import perform_dataset_split
        perform_dataset_split()
        
    run_deep_training()
