# download_mert.py
import os
from transformers import Wav2Vec2Model

# 저장 경로 설정 (/workspace 내부)
SAVE_DIR = "/workspace/n2n-flow/pretrained_models"
MODEL_ID = "m-a-p/MERT-v1-330M"

print(f"Downloading {MODEL_ID} to {SAVE_DIR}...")
model = Wav2Vec2Model.from_pretrained(MODEL_ID, cache_dir=SAVE_DIR)
print("Download Complete!")