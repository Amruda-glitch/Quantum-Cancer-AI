import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class EfficientNetB0CancerModel(nn.Module):
    """
    EfficientNet-B0 model with custom classifier head for 9 cancer classes transfer learning.
    """
    def __init__(self, num_classes=config.NUM_CLASSES, hidden_units=512, dropout=0.3, pretrained=True):
        super(EfficientNetB0CancerModel, self).__init__()
        
        # Load backbone model
        if pretrained:
            try:
                # Modern torchvision syntax
                from torchvision.models import EfficientNet_B0_Weights
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                print("Loaded EfficientNet-B0 with default ImageNet weights.")
            except (ImportError, AttributeError):
                # Legacy torchvision syntax fallback
                self.backbone = models.efficientnet_b0(pretrained=True)
                print("Loaded EfficientNet-B0 with pretrained=True fallback.")
        else:
            self.backbone = models.efficientnet_b0(pretrained=False)
            print("Loaded randomly initialized EfficientNet-B0.")
            
        # Get the number of input features for the classifier layer
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier head
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone weights by default
        self.freeze_backbone(True)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, num_classes)
        )

    def freeze_backbone(self, freeze=True):
        """
        Freezes or unfreezes all backbone weights.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def unfreeze_last_layers(self, unfreeze_blocks=1):
        """
        Unfreezes the final layers of EfficientNet-B0 for selective fine-tuning.
        unfreeze_blocks = 1: Unfreezes the last two feature block groups (features[-2:])
        unfreeze_blocks = 2: Unfreezes the last four feature block groups (features[-4:])
        """
        # First freeze everything in the backbone
        self.freeze_backbone(True)
        
        # Unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        if unfreeze_blocks >= 1:
            print("  Unfreezing EfficientNet-B0 last blocks features[-2:]...")
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True
        if unfreeze_blocks >= 2:
            print("  Unfreezing EfficientNet-B0 blocks features[-4:]...")
            for param in self.backbone.features[-4:].parameters():
                param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        """
        Extracts features from the penultimate layer (before classifier).
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

def get_efficientnet_model(hidden_units=512, dropout=0.3, pretrained=True):
    return EfficientNetB0CancerModel(hidden_units=hidden_units, dropout=dropout, pretrained=pretrained)

if __name__ == "__main__":
    model = get_efficientnet_model()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"EfficientNet-B0 model outputs shape: {y.shape}")
