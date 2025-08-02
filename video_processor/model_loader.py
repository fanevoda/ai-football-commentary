import os
import torch
from transformers import AutoProcessor, SiglipVisionModel
from dotenv import load_dotenv
from inference import get_model

# Set ONNX runtime GPU preference
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

# Load environment variables
load_dotenv()

def check_gpu():
    if torch.cuda.is_available():
        print("✅ CUDA is available:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ CUDA not available. Using CPU.")

def load_detection_model():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable ROBOFLOW_API_KEY is not set.")

    model_id = "football-stat-tracker/1"

    model = get_model(model_id=model_id, api_key=api_key)
    return model

def load_embedding_model():
    model_path = 'google/siglip-base-patch16-224'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiglipVisionModel.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor, device

def load_team_classifier(path="models/team_classifier.joblib"):
    import joblib
    return joblib.load(path)

def load_field_detection_model():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable ROBOFLOW_API_KEY is not set.")

    #FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi-rulrg/2"
    model_id = "football-field-detection-f07vi/15"
    return get_model(model_id=model_id, api_key=api_key)

