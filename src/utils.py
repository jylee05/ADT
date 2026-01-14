# src/utils.py
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset의 매핑을 역으로 이용하여 MIDI 생성 시 사용
REVERSE_DRUM_MAPPING = {
    0: 36,  # Kick
    1: 38,  # Snare
    2: 42,  # Hi-hat (Closed)
    3: 47,  # Tom (Mid-Tom)
    4: 49,  # Crash
    5: 51,  # Ride
    6: 56   # Bell (Cowbell)
}