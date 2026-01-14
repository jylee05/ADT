# src/config.py
import os

class Config:
    # ==============================
    # 1. 경로 및 모델 설정
    # ==============================
    DATA_ROOT = "/public/e-gmd-v1.0.0"
    MERT_PATH = "m-a-p/MERT-v1-330M"
    CACHE_DIR = "/workspace/n2n-flow/pretrained_models"
    
    # ==============================
    # 2. 오디오 처리 파라미터
    # ==============================
    AUDIO_SR = 44100       # Spectrogram용 SR
    MERT_SR = 24000        # MERT 모델 입력용 SR (필수)
    
    # Log Mel-Spectrogram
    N_FFT = 2048
    HOP_LENGTH = 441       # 약 10ms (44100Hz 기준)
    N_MELS = 128
    
    # Grid & Segmentation
    FPS = 100              # Target FPS (10ms 단위)
    SEGMENT_SEC = 5.0      # 학습 시 5초 단위로 Crop
    
    # ==============================
    # 3. 모델 아키텍처 (GTX 1080 최적화)
    # ==============================
    DRUM_CHANNELS = 7      # Kick, Snare, HH, Tom, Crash, Ride, Bell
    FEATURE_DIM = 2        # [Onset, Velocity]
    
    HIDDEN_DIM = 512
    N_LAYERS = 6           # Main Decoder Layer 수 (가볍게 유지)
    COND_LAYERS = 2        # [NEW] 조건 인코더(MERT/Spec) Layer 수
    N_HEADS = 8
    
    MERT_DIM = 1024
    MERT_LAYER_IDX = 10    # 논문 권장 Layer
    
    # ==============================
    # 4. Dropout & Noise
    # ==============================
    DROP_MERT_PROB = 0.15  # MERT 전체 Dropout 확률
    DROP_SPEC_PROB = 0.30  # Spectrogram 전체 Dropout 확률
    DROP_PARTIAL_PROB = 0.5 # 구간 마스킹(Inpainting) 확률
    
    # ==============================
    # 5. 학습 하이퍼파라미터
    # ==============================
    BATCH_SIZE = 16        # 메모리 부족 시 8로 조절
    LR = 3e-4
    EPOCHS = 100
    NUM_WORKERS = 4
    DEVICE = "cuda"
    
    # Loss Annealing (MSE -> MAE)
    C_MAX = 1.0
    C_MIN = 1e-4