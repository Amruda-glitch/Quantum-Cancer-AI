import cv2
import numpy as np
import os
import sys
import random
import torch
import torchvision.transforms as T
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Try to import SimpleITK and MONAI, set flags if unavailable
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import monai.transforms as mt
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

print(f"Libraries available: SimpleITK={HAS_SITK}, MONAI={HAS_MONAI}")

def read_image_medical(path):
    """
    Reads an image. Uses SimpleITK if available, falling back to OpenCV.
    """
    if HAS_SITK:
        try:
            sitk_img = sitk.ReadImage(path)
            # Convert to numpy
            img_np = sitk.GetArrayFromImage(sitk_img)
            # If 3D, take middle slice
            if len(img_np.shape) == 3 and img_np.shape[0] < img_np.shape[1]:
                img_np = img_np[img_np.shape[0] // 2, :, :]
            return img_np
        except Exception:
            pass
    # Fallback to OpenCV
    img = cv2.imread(path)
    if img is not None:
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def denoise_image(img):
    """
    Applies Gaussian/Bilateral denoising to clear image artifacts.
    """
    # Expects numpy array
    if img.ndim == 3:
        # Denoise color image
        return cv2.fastNlMeansDenoisingColored(img, None, config.DENOISE_H, config.DENOISE_H, 7, 21)
    else:
        # Denoise grayscale image
        return cv2.fastNlMeansDenoising(img, None, config.DENOISE_H, 7, 21)

def clahe_enhancement(img):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    # If 3-channel RGB, apply CLAHE to LAB color space L-channel
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_GRID_SIZE)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_GRID_SIZE)
        return clahe.apply(img)

def histogram_equalization(img):
    """
    Applies standard global histogram equalization.
    """
    if img.ndim == 3:
        # Convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    else:
        return cv2.equalizeHist(img)

def contrast_enhancement(img, alpha=1.2, beta=0):
    """
    Applies linear contrast scaling: out = alpha * img + beta
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def normalize_intensity(img):
    """
    Normalizes pixel intensities to [0, 1].
    """
    img_float = img.astype(np.float32)
    # Min-max normalization
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    if max_val > min_val:
        normalized = (img_float - min_val) / (max_val - min_val)
    else:
        normalized = img_float - min_val
    return normalized

def resize_image(img, size=(224, 224)):
    """
    Resizes image using bilinear interpolation.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def preprocess_pipeline(image_path):
    """
    Complete medical image preprocessing pipeline:
    Reads -> Denoise -> CLAHE -> Contrast -> Resize -> Normalize
    Returns preprocessed RGB image as a numpy float32 array in [0, 1] with shape (224, 224, 3)
    """
    # 1. Read
    img = read_image_medical(image_path)
    if img is None:
        raise ValueError(f"Could not load image at path: {image_path}")
        
    # Ensure it is a 3-channel RGB image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    # 2. Denoise
    img = denoise_image(img)
    
    # 3. Apply CLAHE for local medical detail enhancement
    img = clahe_enhancement(img)
    
    # 4. Contrast enhancement
    img = contrast_enhancement(img, alpha=1.1, beta=5)
    
    # 5. Resize to 224x224
    img = resize_image(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    # 6. Normalize Intensity to [0, 1]
    img_normalized = normalize_intensity(img)
    
    return img_normalized

def apply_augmentations(img_np):
    """
    Applies random augmentation to a preprocessed image (numpy float32 shape 224x224x3).
    Augmentations: rotation, flips, zoom, brightness shift, random crop.
    """
    # Convert numpy [0,1] to PIL for torchvision transforms
    img_pil = Image.fromarray((img_np * 255.0).astype(np.uint8))
    
    # Data Augmentation transforms using torchvision
    aug_transform = T.Compose([
        T.RandomRotation(degrees=10), # Rotation up to 10 deg
        T.RandomHorizontalFlip(p=0.5), # Horizontal flip
        T.RandomVerticalFlip(p=0.5), # Vertical flip
        T.ColorJitter(brightness=0.1), # Brightness shift
        T.RandomResizedCrop(size=(config.IMAGE_SIZE, config.IMAGE_SIZE), scale=(0.9, 1.0)), # Zoom and Random crop
    ])
    
    augmented_pil = aug_transform(img_pil)
    # Convert back to numpy float32 [0, 1]
    augmented_np = np.array(augmented_pil, dtype=np.float32) / 255.0
    return augmented_np

if __name__ == "__main__":
    # Test pipeline on a random file
    test_class = os.path.join(config.DATASET_DIR, "ALL", "all_benign")
    if os.path.exists(test_class):
        files = os.listdir(test_class)
        if files:
            sample_path = os.path.join(test_class, files[0])
            print(f"Testing preprocessing on: {sample_path}")
            proc_img = preprocess_pipeline(sample_path)
            print(f"Preprocessed image shape: {proc_img.shape}, min={proc_img.min():.2f}, max={proc_img.max():.2f}")
            aug_img = apply_augmentations(proc_img)
            print(f"Augmented image shape: {aug_img.shape}, min={aug_img.min():.2f}, max={aug_img.max():.2f}")
