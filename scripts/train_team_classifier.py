# scripts/train_team_classifier.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from more_itertools import chunked
from video_processor.model_loader import load_detection_model
from video_processor.video_pipeline import collect_player_crops
from sports.common.team import TeamClassifier
import joblib
import torch
from transformers import AutoProcessor, SiglipVisionModel

print("🔍 sys.path =", sys.path)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to training video")
args = parser.parse_args()

VIDEO_PATH = args.video

MODEL_SAVE_PATH = "models/team_classifier.joblib"
PLAYER_ID = 2
STRIDE = 30
DEVICE = "cuda"  # or "cpu" if needed

# ----- Ensure models folder exists -----
os.makedirs("models", exist_ok=True)

# ----- Load detection model -----
print("[INFO] Loading player detection model...")
detection_model = load_detection_model()
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE)


# ----- Collect player crops -----
print(f"[INFO] Collecting player crops from {VIDEO_PATH}...")
crops = collect_player_crops(VIDEO_PATH, detection_model, player_id=PLAYER_ID, stride=STRIDE)
print(f"[INFO] Collected {len(crops)} crops.")

BATCH_SIZE = 32

batches = chunked(crops, BATCH_SIZE)
data = []
with torch.no_grad():
    for batch in tqdm(batches, desc='embedding extraction'):
        inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
        outputs = EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        data.append(embeddings)

data = np.concatenate(data)



REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters=2)

# ----- Train team classifier -----
print("[INFO] Training team classifier...")
classifier = TeamClassifier(device=DEVICE)
classifier.fit(crops)

# ----- Save classifier -----
joblib.dump(classifier, MODEL_SAVE_PATH)
print(f"[INFO] Team classifier saved to {MODEL_SAVE_PATH}")
