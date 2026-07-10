import os
import random
import numpy as np
import torch

# Base Paths
BASE_DIR = r"c:\IBM Quantathon"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "Multi Cancer")
PROJECT_DIR = os.path.join(BASE_DIR, "quantum_cancer_detection")

# Outputs Subdirectories
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
OUTPUT_MODELS = os.path.join(OUTPUT_DIR, "models")
OUTPUT_REPORTS = os.path.join(OUTPUT_DIR, "reports")
OUTPUT_CM = os.path.join(OUTPUT_DIR, "confusion_matrix")
OUTPUT_ROC = os.path.join(OUTPUT_DIR, "roc_curves")
OUTPUT_GRADCAM = os.path.join(OUTPUT_DIR, "gradcam")
OUTPUT_FEATURES = os.path.join(OUTPUT_DIR, "feature_analysis")

# Ensure all directories exist
for d in [PROJECT_DIR, OUTPUT_DIR, OUTPUT_MODELS, OUTPUT_REPORTS, OUTPUT_CM, OUTPUT_ROC, OUTPUT_GRADCAM, OUTPUT_FEATURES]:
    os.makedirs(d, exist_ok=True)

# Random Seed for Reproducibility
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Class Configuration
# The 9 Cancer Classes mapped from the dataset folder structure
CLASS_NAMES = [
    "Leukemia",      # ALL (all_benign, all_early, all_pre, all_pro)
    "Brain Cancer",  # Brain Cancer (brain_glioma, brain_menin, brain_tumor)
    "Breast Cancer", # Breast Cancer (breast_benign, breast_malignant)
    "Cervical Cancer", # Cervical Cancer (cervix_dyk, cervix_koc, cervix_mep, cervix_pab, cervix_sfi)
    "Kidney Cancer", # Kidney Cancer (kidney_normal, kidney_tumor)
    "Colon Cancer",  # Lung and Colon Cancer (colon_aca, colon_bnt)
    "Lung Cancer",   # Lung and Colon Cancer (lung_aca, lung_bnt, lung_scc)
    "Lymphoma",      # Lymphoma (lymph_cll, lymph_fl, lymph_mcl)
    "Oral Cancer"    # Oral Cancer (oral_normal, oral_scc)
]
NUM_CLASSES = len(CLASS_NAMES)

# Mapping of dataset folder structure to the 9 cancer classes
DATASET_MAPPING = {
    "ALL": {
        "all_benign": "Leukemia",
        "all_early": "Leukemia",
        "all_pre": "Leukemia",
        "all_pro": "Leukemia"
    },
    "Brain Cancer": {
        "brain_glioma": "Brain Cancer",
        "brain_menin": "Brain Cancer",
        "brain_tumor": "Brain Cancer"
    },
    "Breast Cancer": {
        "breast_benign": "Breast Cancer",
        "breast_malignant": "Breast Cancer"
    },
    "Cervical Cancer": {
        "cervix_dyk": "Cervical Cancer",
        "cervix_koc": "Cervical Cancer",
        "cervix_mep": "Cervical Cancer",
        "cervix_pab": "Cervical Cancer",
        "cervix_sfi": "Cervical Cancer"
    },
    "Kidney Cancer": {
        "kidney_normal": "Kidney Cancer",
        "kidney_tumor": "Kidney Cancer"
    },
    "Lung and Colon Cancer": {
        "colon_aca": "Colon Cancer",
        "colon_bnt": "Colon Cancer",
        "lung_aca": "Lung Cancer",
        "lung_bnt": "Lung Cancer",
        "lung_scc": "Lung Cancer"
    },
    "Lymphoma": {
        "lymph_cll": "Lymphoma",
        "lymph_fl": "Lymphoma",
        "lymph_mcl": "Lymphoma"
    },
    "Oral Cancer": {
        "oral_normal": "Oral Cancer",
        "oral_scc": "Oral Cancer"
    }
}

# Execution Modes
DEMO_MODE = True  # Enable for quick validation (uses sample subset)
DEMO_SAMPLE_SIZE = 50  # Number of images to load per subclass in demo mode (total ~1300 images)

# Hyperparameters (Baseline Defaults)
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 2 if DEMO_MODE else 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
DENOISE_H = 3

# Paths to Split Files
SPLIT_CSV = os.path.join(OUTPUT_REPORTS, "dataset_split.csv")
QUALITY_REPORT_JSON = os.path.join(OUTPUT_REPORTS, "quality_report.json")
CLEANING_REPORT_JSON = os.path.join(OUTPUT_REPORTS, "cleaning_report.json")
METRICS_REPORT_JSON = os.path.join(OUTPUT_REPORTS, "model_metrics.json")
COMPARATIVE_REPORT_JSON = os.path.join(OUTPUT_REPORTS, "comparative_metrics.json")
FINAL_REPORT_MD = os.path.join(PROJECT_DIR, "report.md")
