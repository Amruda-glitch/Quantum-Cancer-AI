import os
import sys
import json
import base64
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from http.server import SimpleHTTPRequestHandler, HTTPServer

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from preprocess import read_image_medical, denoise_image, clahe_enhancement, contrast_enhancement, resize_image, normalize_intensity
from models.resnet_model import get_resnet_model
from models.efficientnet_model import get_efficientnet_model
from gradcam import GradCAM

# Set directory to serve static files
DIRECTORY = os.path.join(config.PROJECT_DIR, "dashboard")

# Global models
device = torch.device("cpu") # Server runs on CPU
resnet_model = None
effnet_model = None
gradcam_extractor = None

def load_models():
    global resnet_model, effnet_model, gradcam_extractor
    if resnet_model is not None:
        return
    
    print("Loading models for server inference...")
    # ResNet50
    resnet_path = os.path.join(config.OUTPUT_MODELS, "best_resnet.pth")
    resnet_model = get_resnet_model(pretrained=False).to(device)
    if os.path.exists(resnet_path):
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model.eval()
    
    # Target layer for Grad-CAM is resnet_model.backbone.layer4[-1]
    gradcam_extractor = GradCAM(resnet_model, resnet_model.backbone.layer4[-1])
    
    # EfficientNet
    effnet_path = os.path.join(config.OUTPUT_MODELS, "best_efficientnet.pth")
    effnet_model = get_efficientnet_model(pretrained=False).to(device)
    if os.path.exists(effnet_path):
        effnet_model.load_state_dict(torch.load(effnet_path, map_location=device))
    effnet_model.eval()
    print("Models loaded successfully.")

# Preprocessing steps helper to capture each stage
def run_stages_pipeline(img):
    stages = []
    
    # Ensure 3-channel RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    stages.append(img.copy()) # 1. Raw
    
    # 2. Denoise
    img = denoise_image(img)
    stages.append(img.copy())
    
    # 3. CLAHE
    img = clahe_enhancement(img)
    stages.append(img.copy())
    
    # 4. Contrast
    img = contrast_enhancement(img, alpha=1.1, beta=5)
    stages.append(img.copy())
    
    # 5. Resize
    img = resize_image(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    stages.append(img.copy())
    
    # 6. Normalize
    img_norm = normalize_intensity(img)
    # Scaled back to 0-255 for visualization
    stages.append((img_norm * 255.0).astype(np.uint8))
    
    return img_norm, stages

def encode_cv2_to_base64(img):
    # Convert RGB to BGR for cv2.imencode
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

class DualHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
        
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                load_models()
                req_json = json.loads(post_data.decode('utf-8'))
                img_data = req_json['image']
                
                # Decode image
                header, encoded = img_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                
                # Run preprocessing stages
                img_norm, stages_images = run_stages_pipeline(img_rgb)
                
                # Convert preprocessed to tensor
                img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                img_tensor = img_tensor.unsqueeze(0).to(device) # Batch size 1
                
                # ResNet prediction
                with torch.no_grad():
                    res_logits = resnet_model(img_tensor)
                    res_probs = torch.softmax(res_logits, dim=1).squeeze(0)
                    
                    # EfficientNet prediction
                    eff_logits = effnet_model(img_tensor)
                    eff_probs = torch.softmax(eff_logits, dim=1).squeeze(0)
                    
                    # Ensemble average
                    ens_probs = (res_probs + eff_probs) / 2.0
                    
                # Compute Grad-CAM using ResNet50
                heatmap = gradcam_extractor.generate_heatmap(img_tensor)
                
                # Generate overlay image
                overlay_colored = cv2.resize(stages_images[4], (config.IMAGE_SIZE, config.IMAGE_SIZE))
                heatmap_scaled = np.uint8(255 * heatmap)
                heatmap_resized = cv2.resize(heatmap_scaled, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                alpha = 0.4
                overlay_img = cv2.addWeighted(heatmap_colored_rgb, alpha, overlay_colored, 1 - alpha, 0)
                
                # Draw suspicious region circle
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)
                if max_val > 150:
                    cv2.circle(overlay_img, max_loc, radius=30, color=(0, 255, 0), thickness=2)
                    cv2.putText(overlay_img, "Suspicious Region", (max_loc[0] - 50, max_loc[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Build probabilities response
                probs_list = ens_probs.cpu().numpy().tolist()
                classes = config.CLASS_NAMES
                colors = ["#ff4757", "#a29bfe", "#fd79a8", "#ffeaa7", "#55efc4", "#00b894", "#74b9ff", "#6c5ce7", "#fdcb6e"]
                
                sorted_preds = []
                for i in range(len(classes)):
                    sorted_preds.append({
                        "name": classes[i],
                        "prob": probs_list[i],
                        "color": colors[i]
                    })
                sorted_preds.sort(key=lambda x: x["prob"], reverse=True)
                
                # Encode stages and Gradcam
                base64_stages = [encode_cv2_to_base64(s) for s in stages_images]
                base64_gradcam = encode_cv2_to_base64(overlay_img)
                
                response_data = {
                    "status": "success",
                    "predictions": sorted_preds,
                    "stages": base64_stages,
                    "gradcam": base64_gradcam
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                print("Error during predict endpoint:", e)
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            super().do_POST()

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, DualHandler)
    print("Serving clinical dashboard on http://localhost:8000")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
