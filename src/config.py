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
    # [수정] FPS를 HOP_LENGTH 기반으로 계산하여 일관성 유지
    # AUDIO_SR / HOP_LENGTH = 100
    FPS = AUDIO_SR // HOP_LENGTH  # = 100
    SEGMENT_SEC = 5.0
    
    # [추가] 시퀀스 길이 계산 (명시적으로)
    # Spectrogram frames: SEGMENT_SEC * FPS = 500
    # MERT frames: 약 SEGMENT_SEC * MERT_SR / 320 ≈ 375 (MERT의 hop은 약 320)
    MERT_HOP = 320  # MERT의 내부 hop length (대략적)
    
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
    
    # 5. 학습 파라미터 (1080 Ti 2개 최적화)
    # 1080 Ti: 11GB VRAM each
    # MERT-330M: ~1.3GB (frozen, no gradient)
    # 모델 + 그래디언트 + 옵티마이저: ~4-5GB per GPU
    # 남은 메모리로 배치 처리
    # GPU 2개: 배치 12 -> GPU당 6개 (11GB에서 여유 있음)
    # OOM 발생 시 8로 낮추기
    BATCH_SIZE = 14

    GRAD_ACCUM_STEPS = 6
    
    LR = 1e-4  # [수정] 더 보수적인 learning rate
    EPOCHS = 100
    NUM_WORKERS = 8 # 8로 늘림  # 1080 Ti는 CPU 병목 방지를 위해 4 권장
    DEVICE = "cuda"
    
    C_MAX = 1.0
    C_MIN = 1e-4