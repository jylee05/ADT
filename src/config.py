# src/config.py
class Config:
    # 경로 설정
    DATA_ROOT = "/public/e-gmd-v1.0.0"
    MERT_PATH = "/workspace/n2n-flow/pretrained_models/models--m-a-p--MERT-v1-330M/snapshots" 
    # 주의: huggingface cache 구조상 실제 스냅샷 폴더 경로를 확인해야 할 수도 있습니다. 
    # 위 download_mert.py에서 cache_dir을 지정했으므로 로딩 시 cache_dir을 동일하게 주면 됩니다.
    CACHE_DIR = "/workspace/n2n-flow/pretrained_models"
    
    # 오디오 설정
    SAMPLE_RATE = 24000  # MERT 학습 SR
    FPS = 100            # Target Grid (10ms)
    SEGMENT_SEC = 5.0    # 학습 시 자를 길이 (5초)
    
    # 모델 파라미터
    DRUM_CHANNELS = 7    # KD, SD, HH, Tom, Crash, Ride, Bell
    FEATURE_DIM = 2      # Onset, Velocity
    HIDDEN_DIM = 512
    N_LAYERS = 6
    N_HEADS = 8
    MERT_DIM = 1024
    
    # 학습 설정
    BATCH_SIZE = 16      # VRAM에 맞춰 조절 (16~32 추천)
    LR = 1e-4
    EPOCHS = 100
    NUM_WORKERS = 4
    DEVICE = "cuda"