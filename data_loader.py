import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from preprocess import preprocess_pipeline, apply_augmentations

class CancerDataset(Dataset):
    """
    Custom PyTorch Dataset for Multi-Class Cancer Image Classification.
    """
    def __init__(self, split_df, transform_aug=False):
        """
        split_df: pandas DataFrame containing the split entries
        transform_aug: bool, whether to apply random augmentations (True for training)
        """
        self.df = split_df.reset_index(drop=True)
        self.transform_aug = transform_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        class_id = int(row["class_id"])
        
        try:
            # 1. Apply image preprocessing pipeline
            img_np = preprocess_pipeline(file_path)
            
            # 2. Apply random data augmentations for training data
            if self.transform_aug:
                img_np = apply_augmentations(img_np)
                
            # 3. Convert from HWC to CHW format expected by PyTorch
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            
            # Normalize to ImageNet statistics for transfer learning models
            # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            return img_tensor, class_id
            
        except Exception as e:
            # Robust error handling for any issues loading individual files
            print(f"Error loading image {file_path}: {e}")
            # Return a dummy tensor and label to avoid crashing the training loop
            dummy_img = torch.zeros((3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=torch.float32)
            return dummy_img, class_id

def get_dataloaders():
    """
    Reads the dataset split CSV, creates the CancerDataset objects,
    and returns Train, Val, and Test DataLoaders.
    """
    if not os.path.exists(config.SPLIT_CSV):
        raise FileNotFoundError(f"Split CSV not found at {config.SPLIT_CSV}. Please run utils/dataset_split.py first.")
        
    df = pd.read_csv(config.SPLIT_CSV)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    print(f"Loading datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    train_dataset = CancerDataset(train_df, transform_aug=True)
    val_dataset = CancerDataset(val_df, transform_aug=False)
    test_dataset = CancerDataset(test_df, transform_aug=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, # set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test data loader
    if os.path.exists(config.SPLIT_CSV):
        tr, vl, ts = get_dataloaders()
        for x, y in tr:
            print(f"Batch shape: images={x.shape}, labels={y.shape}")
            break
    else:
        print("Please run utils/dataset_split.py first to create the split index.")
