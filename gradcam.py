import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from preprocess import preprocess_pipeline
from models.resnet_model import get_resnet_model
from models.efficientnet_model import get_efficientnet_model

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for PyTorch CNNs.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register forward and backward hooks on the target layer
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        # Handle older PyTorch version backward hook differences
        try:
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        except AttributeError:
            self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Temporarily enable requires_grad for all parameters so gradients propagate during backward hook
        original_states = {}
        for name, param in self.model.named_parameters():
            original_states[name] = param.requires_grad
            param.requires_grad = True
            
        self.gradients = None
        self.activations = None
        
        try:
            # Force gradient tracking to be enabled
            with torch.enable_grad():
                # Forward pass
                output = self.model(input_tensor)
                
                if class_idx is None:
                    class_idx = torch.argmax(output, dim=1).item()
                    
                # Backward pass
                score = output[0, class_idx]
                score.backward()
        except Exception as e:
            print("Error during backward pass in GradCAM:", e)
            
        # Restore original requires_grad states
        for name, param in self.model.named_parameters():
            param.requires_grad = original_states[name]
            
        # Check if hook successfully captured gradients and activations
        if self.gradients is None or self.activations is None:
            print("GradCAM warning: gradients or activations are None. Returning blank fallback heatmap.")
            return np.zeros((224, 224), dtype=np.float32)
            
        # Pull gradients and activations
        gradients = self.gradients[0] # Shape: [C, H, W]
        activations = self.activations[0] # Shape: [C, H, W]
        
        # Global average pooling of gradients to get weights
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # Shape: [C, 1, 1]
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=0) # Shape: [H, W]
        
        # Apply ReLU to keep only positive features contributing to the class
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam - cam_min
            
        cam_np = cam.cpu().numpy()
        return cam_np

def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    """
    Loads original image, overlays heatmap on top of it, and draws a bounding circle
    around the suspicious region of highest activation.
    """
    # Load original image in BGR
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    # Scale heatmap to [0, 255]
    heatmap_scaled = np.uint8(255 * heatmap)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap_scaled, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    # Apply JET colormap for visualization
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(heatmap_colored, alpha, img_resized, 1 - alpha, 0)
    
    # Highlight suspicious regions: find coordinates of maximum activation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)
    
    # If activation is significant, draw circle highlight around peak
    if max_val > 150:
        cv2.circle(overlay, max_loc, radius=30, color=(0, 255, 0), thickness=2)
        cv2.putText(overlay, "Suspicious Region", (max_loc[0] - 50, max_loc[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    
    return overlay, img_resized

def run_gradcam_on_samples():
    """
    Generates explainable AI Grad-CAM visualizations for one sample of each of the 9 classes.
    """
    print("\n--- Starting Explainable AI (Grad-CAM) Visualizations ---")
    config.set_seed()
    
    # Load ResNet50 model
    resnet_path = os.path.join(config.OUTPUT_MODELS, "best_resnet.pth")
    resnet = get_resnet_model(pretrained=True).to(config.DEVICE)
    if os.path.exists(resnet_path):
        resnet.load_state_dict(torch.load(resnet_path, map_location=config.DEVICE))
        print("Loaded best ResNet50 weights for Grad-CAM.")
    else:
        print("ResNet50 weights not found. Using default weights for demo.")
        
    # Hook to target layer: layer4[-1] for ResNet50 (the last residual block)
    target_layer = resnet.backbone.layer4[-1]
    gcam = GradCAM(resnet, target_layer)
    
    # Select one sample per class from the dataset
    samples = {}
    for parent in os.listdir(config.DATASET_DIR):
        parent_path = os.path.join(config.DATASET_DIR, parent)
        if not os.path.isdir(parent_path):
            continue
        for subclass in os.listdir(parent_path):
            subclass_path = os.path.join(parent_path, subclass)
            if not os.path.isdir(subclass_path):
                continue
            class_name = config.DATASET_MAPPING.get(parent, {}).get(subclass)
            if class_name and class_name not in samples:
                files = os.listdir(subclass_path)
                if files:
                    samples[class_name] = os.path.join(subclass_path, files[0])
                    
    print(f"Generating Grad-CAM overlays for {len(samples)} cancer classes...")
    
    for class_name, img_path in samples.items():
        try:
            # 1. Preprocess image
            img_np = preprocess_pipeline(img_path)
            
            # Convert to PyTorch tensor [1, 3, 224, 224]
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
            
            # Normalize for ResNet50
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.to(config.DEVICE)
            
            # 2. Get target class index
            class_idx = config.CLASS_NAMES.index(class_name)
            
            # 3. Generate heatmap
            heatmap = gcam.generate_heatmap(img_tensor, class_idx=class_idx)
            
            # 4. Create overlay and highlight regions
            overlay, original_resized = overlay_heatmap_on_image(img_path, heatmap)
            
            # 5. Save the side-by-side or overlaid image
            save_name = f"{class_name.lower().replace(' ', '_')}_gradcam.png"
            save_path = os.path.join(config.OUTPUT_GRADCAM, save_name)
            
            # Stack horizontally (Original vs. Grad-CAM Overlay)
            vis_stacked = np.hstack((original_resized, overlay))
            cv2.imwrite(save_path, vis_stacked)
            print(f"Saved Grad-CAM for {class_name} to: {save_path}")
            
        except Exception as e:
            print(f"Error processing Grad-CAM for {class_name}: {e}")
            
    # Cleanup hooks
    gcam.remove_hooks()
    print("Explainable AI visualizations finished successfully!")

if __name__ == "__main__":
    run_gradcam_on_samples()
