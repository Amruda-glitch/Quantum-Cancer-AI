import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ResNet50CancerModel(nn.Module):
    """
    ResNet50 model with custom classifier head for 9 cancer classes transfer learning.
    """
    def __init__(self, num_classes=config.NUM_CLASSES, hidden_units=512, dropout=0.3, pretrained=True):
        super(ResNet50CancerModel, self).__init__()
        
        # Load backbone model
        if pretrained:
            try:
                # Modern torchvision syntax
                from torchvision.models import ResNet50_Weights
                self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                print("Loaded ResNet50 with default ImageNet weights.")
            except (ImportError, AttributeError):
                # Legacy torchvision syntax fallback
                self.backbone = models.resnet50(pretrained=True)
                print("Loaded ResNet50 with pretrained=True fallback.")
        else:
            self.backbone = models.resnet50(pretrained=False)
            print("Loaded randomly initialized ResNet50.")
            
        # Get the number of input features for the final linear layer
        in_features = self.backbone.fc.in_features
        
        # Remove the original classification head
        self.backbone.fc = nn.Identity()
        
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
        Unfreezes the final layers of ResNet50 for selective fine-tuning.
        unfreeze_blocks = 1: Unfreezes layer4
        unfreeze_blocks = 2: Unfreezes layer4 and layer3
        """
        # First freeze everything in the backbone
        self.freeze_backbone(True)
        
        # Unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        if unfreeze_blocks >= 1:
            print("  Unfreezing ResNet50 layer4...")
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        if unfreeze_blocks >= 2:
            print("  Unfreezing ResNet50 layer3...")
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        """
        Extracts features from the penultimate layer (before classifier).
        Useful for downstream classical models and quantum feature selection.
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features

def get_resnet_model(hidden_units=512, dropout=0.3, pretrained=True):
    return ResNet50CancerModel(hidden_units=hidden_units, dropout=dropout, pretrained=pretrained)

if __name__ == "__main__":
    model = get_resnet_model()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"ResNet50 model outputs shape: {y.shape}")
