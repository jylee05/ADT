# src/config.py
import os

class Config:
    # 경로 설정
    DATA_ROOT = "/public/e-gmd-v1.0.0"
    # MERT 모델 경로 (로컬 캐시 혹은 HuggingFace ID)
    MERT_PATH = "m-a-p/MERT-v1-330M"
    CACHE_DIR = "/workspace/n2n-flow/pretrained_models"
    
    # 오디오 및 스펙트로그램 설정 (논문 설정 준수)
    AUDIO_SR = 44100       # 스펙트로그램용 기본 SR
    MERT_SR = 24000        # MERT 모델용 SR
    
    # Log Mel-Spectrogram 파라미터
    N_FFT = 2048
    HOP_LENGTH = 441       # 10ms at 44.1kHz
    N_MELS = 128
    
    FPS = 100              # Target Grid (10ms 단위)
    SEGMENT_SEC = 5.0      # 학습 시 자를 길이 (5초)
    
    # 모델 파라미터
    DRUM_CHANNELS = 7
    FEATURE_DIM = 2        # Onset, Velocity
    HIDDEN_DIM = 512
    N_LAYERS = 6
    N_HEADS = 8
    MERT_DIM = 1024
    MERT_LAYER_IDX = 10    # 논문 권장: Layer 10 사용
    
    # Dropout 설정 (Conditioning)
    DROP_MERT_PROB = 0.15  # Complete Dropout probability for MERT
    DROP_SPEC_PROB = 0.30  # Complete Dropout probability for Spectrogram
    DROP_PARTIAL_PROB = 0.5 # Partial Dropout probability
    
    # 학습 설정
    BATCH_SIZE = 16
    LR = 3e-4              # 논문 설정: 3e-4
    EPOCHS = 100
    NUM_WORKERS = 4
    DEVICE = "cuda"
    
    # Loss Annealing 설정
    C_MAX = 1.0
    C_MIN = 1e-4