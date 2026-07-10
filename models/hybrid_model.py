import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.resnet_model import get_resnet_model
from models.efficientnet_model import get_efficientnet_model

class HybridCancerEnsembleModel(nn.Module):
    """
    Hybrid Ensemble model combining ResNet50 and EfficientNet-B0.
    Averages predictions (probabilities) of both models.
    """
    def __init__(self, resnet_path=None, effnet_path=None, device=config.DEVICE):
        super(HybridCancerEnsembleModel, self).__init__()
        
        self.device = device
        self.resnet = get_resnet_model(pretrained=True).to(device)
        self.efficientnet = get_efficientnet_model(pretrained=True).to(device)
        
        # Load weights if paths are provided and exist
        if resnet_path and os.path.exists(resnet_path):
            try:
                state_dict = torch.load(resnet_path, map_location=device)
                self.resnet.load_state_dict(state_dict)
                print(f"Hybrid Ensemble loaded ResNet50 weights from {resnet_path}")
            except Exception as e:
                print(f"Warning loading ResNet50 weights: {e}")
                
        if effnet_path and os.path.exists(effnet_path):
            try:
                state_dict = torch.load(effnet_path, map_location=device)
                self.efficientnet.load_state_dict(state_dict)
                print(f"Hybrid Ensemble loaded EfficientNet-B0 weights from {effnet_path}")
            except Exception as e:
                print(f"Warning loading EfficientNet-B0 weights: {e}")
                
        # Set to evaluation mode by default
        self.resnet.eval()
        self.efficientnet.eval()

    def forward(self, x):
        """
        Calculates ensemble output by averaging the softmax probabilities.
        """
        with torch.no_grad():
            # Get logits from both models
            logits_res = self.resnet(x)
            logits_eff = self.efficientnet(x)
            
            # Apply Softmax to get probabilities
            probs_res = F.softmax(logits_res, dim=1)
            probs_eff = F.softmax(logits_eff, dim=1)
            
            # Average probabilities
            avg_probs = (probs_res + probs_eff) / 2.0
            
            # Convert back to log-probabilities (useful for loss functions if needed)
            # or just return average probabilities
            return avg_probs

    def predict(self, x):
        """
        Returns class probabilities and predicted class IDs.
        """
        probs = self.forward(x)
        preds = torch.argmax(probs, dim=1)
        return probs, preds

if __name__ == "__main__":
    device = torch.device("cpu")
    model = HybridCancerEnsembleModel(device=device)
    x = torch.randn(2, 3, 224, 224)
    probs = model(x)
    print(f"Hybrid Ensemble model outputs shape (probabilities): {probs.shape}")
