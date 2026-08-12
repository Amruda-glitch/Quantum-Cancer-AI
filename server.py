import os
import sys
import base64
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# ============================================================
# QUONCO - QUANTUM CANCER AI
# ResNet50 Inference API
# ============================================================

app = FastAPI(
    title="Quantum Cancer Detection API",
    version="1.0.0"
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config

from preprocess import (
    read_image_medical,
    denoise_image,
    clahe_enhancement,
    contrast_enhancement,
    resize_image,
    normalize_intensity,
)

from models.resnet_model import get_resnet_model
from gradcam import GradCAM


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

# ============================================================
# GLOBAL MODEL
# ============================================================

resnet_model = None
gradcam_extractor = None


# ============================================================
# HUGGING FACE MODEL
# ============================================================

RESNET_MODEL_URL = os.getenv(
    "RESNET_MODEL_URL",
    "https://huggingface.co/Arul-bestower/quantum-cancer-models/resolve/main/best_resnet.pth"
)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_file_from_url(url, destination_path):

    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
        print(f"Model already exists: {destination_path}")
        return

    print("Downloading ResNet50 model...")
    print(f"URL: {url}")

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=300,
        )

        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(destination_path, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percent = downloaded * 100 / total_size

                    print(
                        f"\rDownloading: {percent:.1f}%",
                        end="",
                        flush=True
                    )

        print()
        print("Model download completed.")

    except Exception as e:

        if os.path.exists(destination_path):
            try:
                os.remove(destination_path)
            except Exception:
                pass

        raise RuntimeError(
            f"Failed to download model: {e}"
        )


# ============================================================
# LOAD RESNET MODEL
# ============================================================

def load_models():

    global resnet_model
    global gradcam_extractor

    if resnet_model is not None:
        return

    print()
    print("==============================================")
    print("Preparing QuOnco model...")
    print("==============================================")

    # --------------------------------------------------------
    # Choose writable model directory
    # --------------------------------------------------------

    models_dir = config.OUTPUT_MODELS

    try:

        os.makedirs(models_dir, exist_ok=True)

        test_file = os.path.join(
            models_dir,
            ".write_test"
        )

        with open(test_file, "w") as f:
            f.write("test")

        os.remove(test_file)

    except Exception:

        models_dir = "/tmp/models"
        os.makedirs(models_dir, exist_ok=True)

    # --------------------------------------------------------
    # Model path
    # --------------------------------------------------------

    resnet_path = os.path.join(
        models_dir,
        "best_resnet.pth"
    )

    # --------------------------------------------------------
    # Download model if necessary
    # --------------------------------------------------------

    if (
        not os.path.exists(resnet_path)
        or os.path.getsize(resnet_path) == 0
    ):

        print("Downloading QuOnco ResNet50 model...")

        download_file_from_url(
            RESNET_MODEL_URL,
            resnet_path
        )

    # --------------------------------------------------------
    # Create model architecture
    # --------------------------------------------------------

    print("Creating ResNet50 architecture...")

    resnet_model = get_resnet_model(
        pretrained=False
    ).to(device)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    print("Loading trained ResNet50 weights...")

    try:

        checkpoint = torch.load(
            resnet_path,
            map_location=device,
            weights_only=False
        )

        # Handle normal state_dict
        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        # Remove DataParallel prefix if present
        cleaned_state_dict = {}

        for key, value in checkpoint.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[7:]

            cleaned_state_dict[new_key] = value

        resnet_model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        print("ResNet50 weights loaded successfully.")

    except Exception as e:

        resnet_model = None

        raise RuntimeError(
            f"Failed to load ResNet50 weights: {e}"
        )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    resnet_model.eval()

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    print("Preparing Grad-CAM...")

    gradcam_extractor = GradCAM(
        resnet_model,
        resnet_model.backbone.layer4[-1]
    )

    print("QuOnco ResNet50 loaded successfully.")
    print()


# ============================================================
# IMAGE PIPELINE
# ============================================================

def run_stages_pipeline(img):

    stages = []

    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    if img.ndim == 2:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    elif img.ndim == 3 and img.shape[2] == 4:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2RGB
        )

    stages.append(img.copy())

    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    img = denoise_image(img)

    stages.append(img.copy())

    # --------------------------------------------------------
    # CLAHE enhancement
    # --------------------------------------------------------

    img = clahe_enhancement(img)

    stages.append(img.copy())

    # --------------------------------------------------------
    # Contrast enhancement
    # --------------------------------------------------------

    img = contrast_enhancement(
        img,
        alpha=1.1,
        beta=5
    )

    stages.append(img.copy())

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    img = resize_image(
        img,
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        )
    )

    stages.append(img.copy())

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    img_norm = normalize_intensity(img)

    stages.append(
        (img_norm * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )

    return img_norm, stages


# ============================================================
# CV2 -> BASE64
# ============================================================

def encode_cv2_to_base64(img):

    if img.ndim == 2:

        img_bgr = img

    else:

        img_bgr = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

    success, buffer = cv2.imencode(
        ".png",
        img_bgr
    )

    if not success:
        raise ValueError(
            "Failed to encode image."
        )

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    image: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def read_root():

    return {
        "status": "online",
        "message": "Quantum Cancer Detection API is running",
        "model": "QuOnco ResNet50",
        "device": str(device),
        "classes": config.CLASS_NAMES,
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": resnet_model is not None,
        "model": "ResNet50",
        "device": str(device),
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(req: PredictionRequest):

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        load_models()

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if not req.image:

            raise HTTPException(
                status_code=400,
                detail="Image is required."
            )

        if "," not in req.image:

            raise HTTPException(
                status_code=400,
                detail="Invalid image format."
            )

        header, encoded = req.image.split(
            ",",
            1
        )

        # ----------------------------------------------------
        # Decode Base64
        # ----------------------------------------------------

        try:

            img_bytes = base64.b64decode(
                encoded,
                validate=True
            )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Invalid Base64 image."
            )

        # ----------------------------------------------------
        # Convert bytes -> OpenCV image
        # ----------------------------------------------------

        nparr = np.frombuffer(
            img_bytes,
            np.uint8
        )

        img_cv2 = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img_cv2 is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid image data."
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        img_rgb = cv2.cvtColor(
            img_cv2,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        img_norm, stages_images = (
            run_stages_pipeline(img_rgb)
        )

        # ----------------------------------------------------
        # Convert image -> tensor
        # ----------------------------------------------------

        img_tensor = torch.from_numpy(
            img_norm
        ).permute(
            2,
            0,
            1
        ).float()

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        img_tensor = (
            img_tensor - mean
        ) / std

        img_tensor = (
            img_tensor
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # ResNet prediction
        # ----------------------------------------------------

        with torch.no_grad():

            res_logits = resnet_model(
                img_tensor
            )

            res_probs = torch.softmax(
                res_logits,
                dim=1
            ).squeeze(0)

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        heatmap = (
            gradcam_extractor
            .generate_heatmap(img_tensor)
        )

        # ----------------------------------------------------
        # Generate Grad-CAM overlay
        # ----------------------------------------------------

        overlay_colored = cv2.resize(
            stages_images[4],
            (
                config.IMAGE_SIZE,
                config.IMAGE_SIZE
            )
        )

        heatmap_scaled = np.uint8(
            255 * np.clip(
                heatmap,
                0,
                1
            )
        )

        heatmap_resized = cv2.resize(
            heatmap_scaled,
            (
                config.IMAGE_SIZE,
                config.IMAGE_SIZE
            )
        )

        heatmap_colored = cv2.applyColorMap(
            heatmap_resized,
            cv2.COLORMAP_JET
        )

        heatmap_colored_rgb = cv2.cvtColor(
            heatmap_colored,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Blend
        # ----------------------------------------------------

        alpha = 0.4

        overlay_img = cv2.addWeighted(
            heatmap_colored_rgb,
            alpha,
            overlay_colored,
            1 - alpha,
            0
        )

        # ----------------------------------------------------
        # Find suspicious region
        # ----------------------------------------------------

        min_val, max_val, min_loc, max_loc = (
            cv2.minMaxLoc(
                heatmap_resized
            )
        )

        if max_val > 150:

            cv2.circle(
                overlay_img,
                max_loc,
                radius=30,
                color=(0, 255, 0),
                thickness=2
            )

            cv2.putText(
                overlay_img,
                "Suspicious Region",
                (
                    max_loc[0] - 50,
                    max_loc[1] - 40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        probs_list = (
            res_probs
            .cpu()
            .numpy()
            .tolist()
        )

        classes = config.CLASS_NAMES

        colors = [
            "#ff4757",
            "#a29bfe",
            "#fd79a8",
            "#ffeaa7",
            "#55efc4",
            "#00b894",
            "#74b9ff",
            "#6c5ce7",
            "#fdcb6e",
        ]

        sorted_preds = []

        for i in range(
            min(
                len(classes),
                len(probs_list)
            )
        ):

            sorted_preds.append(
                {
                    "name": classes[i],
                    "prob": float(
                        probs_list[i]
                    ),
                    "color": colors[i],
                }
            )

        # ----------------------------------------------------
        # Sort highest probability first
        # ----------------------------------------------------

        sorted_preds.sort(
            key=lambda x: x["prob"],
            reverse=True
        )

        # ----------------------------------------------------
        # Generate Base64 images
        # ----------------------------------------------------

        base64_stages = [
            encode_cv2_to_base64(stage)
            for stage in stages_images
        ]

        base64_gradcam = (
            encode_cv2_to_base64(
                overlay_img
            )
        )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        top_prediction = (
            sorted_preds[0]
            if sorted_preds
            else None
        )

        return {
            "status": "success",
            "model": "QuOnco ResNet50",
            "prediction": top_prediction,
            "predictions": sorted_preds,
            "stages": base64_stages,
            "gradcam": base64_gradcam,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "Error during predict endpoint:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print(
        "Starting QuOnco API..."
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "7860"
            )
        ),
    )