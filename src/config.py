# src/config.py
import os

class Config:
    # 1. 경로 설정
    DATA_ROOT = "/public/e-gmd-v1.0.0"
    MERT_PATH = "m-a-p/MERT-v1-330M"
    CACHE_DIR = "./pretrained_models"
    
    # 2. 오디오 파라미터
    AUDIO_SR = 44100
    MERT_SR = 24000
    N_FFT = 2048
    HOP_LENGTH = 441
    N_MELS = 128
    FPS = 100
    SEGMENT_SEC = 5.0
    
    # 3. 모델 아키텍처
    DRUM_CHANNELS = 7
    FEATURE_DIM = 2
    HIDDEN_DIM = 512
    N_LAYERS = 6
    COND_LAYERS = 2
    N_HEADS = 8
    MERT_DIM = 1024
    MERT_LAYER_IDX = 10
    
    # 4. Dropout
    DROP_MERT_PROB = 0.15
    DROP_SPEC_PROB = 0.30
    DROP_PARTIAL_PROB = 0.5
    
    # 5. 학습 파라미터 (멀티 GPU 최적화)
    # GPU가 2개일 때: 배치 8 -> GPU당 4개 할당 (안전)
    BATCH_SIZE = 8       
    # 8 * 2(accum) = 16 (기존 배치 사이즈 효과 유지)
    GRAD_ACCUM_STEPS = 2 
    
    LR = 3e-4
    EPOCHS = 100
    NUM_WORKERS = 8 # GPU 2개면 8도 괜찮음
    DEVICE = "cuda"
    
    C_MAX = 1.0
    C_MIN = 1e-4