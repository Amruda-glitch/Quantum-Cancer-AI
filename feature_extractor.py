import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import get_dataloaders
from models.resnet_model import get_resnet_model
from models.efficientnet_model import get_efficientnet_model

def extract_and_save_features():
    """
    Extracts deep features from ResNet50 and EfficientNet-B0 penultimate layers
    and saves them to outputs/feature_analysis/
    """
    print("Starting deep feature extraction...")
    config.set_seed()
    
    # Ensure splits are created
    if not os.path.exists(config.SPLIT_CSV):
        print("Split CSV not found. Generating train/val/test splits...")
        from utils.dataset_split import perform_dataset_split
        perform_dataset_split()
        
    # 1. Load Data loaders
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 2. Instantiate models
    # Check if we have trained model weights available, otherwise use ImageNet weights
    resnet_weights_path = os.path.join(config.OUTPUT_MODELS, "best_resnet.pth")
    effnet_weights_path = os.path.join(config.OUTPUT_MODELS, "best_efficientnet.pth")
    
    resnet = get_resnet_model(pretrained=True).to(config.DEVICE)
    if os.path.exists(resnet_weights_path):
        resnet.load_state_dict(torch.load(resnet_weights_path, map_location=config.DEVICE))
        print(f"Loaded trained ResNet50 weights from {resnet_weights_path}")
    else:
        print("Trained ResNet50 weights not found. Using pretrained ImageNet feature extractor.")
        
    effnet = get_efficientnet_model(pretrained=True).to(config.DEVICE)
    if os.path.exists(effnet_weights_path):
        effnet.load_state_dict(torch.load(effnet_weights_path, map_location=config.DEVICE))
        print(f"Loaded trained EfficientNet-B0 weights from {effnet_weights_path}")
    else:
        print("Trained EfficientNet-B0 weights not found. Using pretrained ImageNet feature extractor.")
        
    resnet.eval()
    effnet.eval()
    
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    # Extract features for each split
    for split_name, loader in loaders.items():
        print(f"Extracting features for '{split_name}' split...")
        
        resnet_feats = []
        effnet_feats = []
        labels_list = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"{split_name} Split"):
                imgs = imgs.to(config.DEVICE)
                
                # Extract features from penultimate layers
                res_f = resnet.extract_features(imgs)
                eff_f = effnet.extract_features(imgs)
                
                # Move to CPU and numpy
                resnet_feats.append(res_f.cpu().numpy())
                effnet_feats.append(eff_f.cpu().numpy())
                labels_list.append(labels.numpy())
                
        # Concatenate lists into arrays
        resnet_feats = np.concatenate(resnet_feats, axis=0)
        effnet_feats = np.concatenate(effnet_feats, axis=0)
        labels_arr = np.concatenate(labels_list, axis=0)
        
        print(f"[{split_name}] ResNet50 feature shape: {resnet_feats.shape}")
        print(f"[{split_name}] EfficientNet-B0 feature shape: {effnet_feats.shape}")
        
        # Save features
        res_feat_path = os.path.join(config.OUTPUT_FEATURES, f"resnet_{split_name}_features.npy")
        eff_feat_path = os.path.join(config.OUTPUT_FEATURES, f"effnet_{split_name}_features.npy")
        labels_path = os.path.join(config.OUTPUT_FEATURES, f"{split_name}_labels.npy")
        
        np.save(res_feat_path, resnet_feats)
        np.save(eff_feat_path, effnet_feats)
        np.save(labels_path, labels_arr)
        
        print(f"Saved {split_name} features successfully.")
        
    print("Feature extraction completed successfully!")

if __name__ == "__main__":
    # Ensure splits are created
    if not os.path.exists(config.SPLIT_CSV):
        from utils.dataset_split import perform_dataset_split
        perform_dataset_split()
        
    extract_and_save_features()
