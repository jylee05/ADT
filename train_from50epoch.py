# train_from50epoch.py
import os
# [중요] 0번, 1번 GPU만 보이게 설정 (코드 최상단 위치)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset
from src.utils import seed_everything
from tqdm import tqdm
import math

# Gradient Clipping을 위한 max norm (Flow Matching에 적합하게 조정)
MAX_GRAD_NORM = 1.0  # [수정] 0.5 -> 1.0으로 증가 (생성모델에 적합)

def check_for_nan(tensor, name):
    """NaN 체크 유틸리티"""
    if torch.isnan(tensor).any():
        print(f"[WARNING] NaN detected in {name}!")
        return True
    if torch.isinf(tensor).any():
        print(f"[WARNING] Inf detected in {name}!")
        return True
    return False

def create_lr_scheduler(optimizer, total_epochs, warmup_epochs=5):
    """
    Learning Rate Scheduler: Warmup -> Peak -> Cosine Decay
    - Warmup: 5e-5 -> 1e-4 (5 epochs)
    - Peak: 1e-4 (20 epochs)  
    - Cosine Decay: 1e-4 -> 1e-6 (remaining epochs)
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup phase: 0.5 -> 1.0
            return 0.5 + 0.5 * (epoch / warmup_epochs)
        elif epoch < warmup_epochs + 20:
            # Peak phase: maintain 1.0
            return 1.0
        else:
            # Cosine decay phase
            remaining_epochs = total_epochs - warmup_epochs - 20
            progress = (epoch - warmup_epochs - 20) / remaining_epochs
            return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_checkpoint(model, checkpoint_path, device):
    """체크포인트 로딩"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # DataParallel wrapper 고려
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    loaded_epoch = checkpoint.get('epoch', 50)
    print(f"Checkpoint loaded successfully from epoch {loaded_epoch}")
    return loaded_epoch

def main():
    seed_everything(42)
    cfg = Config()
    # [강제 설정] Effective Batch Size 확보 (12 * 6 = 72, 논문 기준 64 이상)
    cfg.GRAD_ACCUM_STEPS = 6
    device = torch.device(cfg.DEVICE)
    
    # [수정] 총 epoch를 150으로 설정
    TOTAL_EPOCHS = 150
    START_EPOCH = 50  # 50 epoch부터 시작
    
    print("Initializing Dataset...")
    dataset = EGMDDataset(is_train=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    print("Initializing Model...")
    backbone = FlowMatchingTransformer(cfg).to(device)
    
    # 체크포인트 로딩
    checkpoint_path = "checkpoints/n2n_ep50.pth"
    if os.path.exists(checkpoint_path):
        loaded_epoch = load_checkpoint(backbone, checkpoint_path, device)
        assert loaded_epoch == 50, f"Expected epoch 50, got {loaded_epoch}"
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # [설정] GPU 2개 모두 사용
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        backbone = nn.DataParallel(backbone)
    
    # 수정된 Loss Wrapper 사용
    loss_wrapper = AnnealedPseudoHuberLoss(backbone, cfg).to(device)
    
    # [수정] Learning Rate를 1e-4로 설정 (peak LR)
    optimizer = torch.optim.AdamW(
        backbone.parameters(), 
        lr=1e-4,  # Peak learning rate
        weight_decay=0.01, 
        betas=(0.9, 0.999)  # [수정] 0.99 -> 0.999 (Flow Matching 안정성 향상)
    )
    
    # Learning Rate Scheduler 설정
    remaining_epochs = TOTAL_EPOCHS - START_EPOCH
    scheduler = create_lr_scheduler(optimizer, remaining_epochs, warmup_epochs=5)
    
    scaler = GradScaler('cuda')  # PyTorch 2.x 호환
    
    print(f"Resume training from epoch {START_EPOCH+1} to {TOTAL_EPOCHS}...")
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # [수정] 첫 epoch LR 충격 방지: 5e-5에서 부드럽게 시작
    optimizer.param_groups[0]['lr'] = 5e-5
    print(f"Force set initial LR to 5e-5 for smooth resume from epoch 50")
    
    # [수정] Progress 계산을 실제 업데이트 횟수 기준으로 수정
    global_update_step = START_EPOCH * len(dataloader) // cfg.GRAD_ACCUM_STEPS
    total_update_steps = TOTAL_EPOCHS * len(dataloader) // cfg.GRAD_ACCUM_STEPS
    print(f"Total update steps: {total_update_steps}, Starting from: {global_update_step}")
    
    optimizer.zero_grad()
    
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        backbone.train()
        total_loss = 0
        current_loss_accum = 0
        nan_count = 0
        
        # LR 업데이트
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} (LR: {current_lr:.2e})")
        for step, (audio_mert, spec, target) in enumerate(pbar):
            audio_mert = audio_mert.to(device)
            spec = spec.to(device)
            target = target.to(device)
            
            # [추가] 입력 데이터 NaN 체크
            if check_for_nan(audio_mert, "audio_mert") or check_for_nan(spec, "spec") or check_for_nan(target, "target"):
                print(f"Skipping batch {step} due to NaN in input")
                nan_count += 1
                continue
            
            # [수정] Progress 계산을 실제 업데이트 기준으로 조정
            progress = global_update_step / total_update_steps
            
            # [수정] Mixed Precision Training - device_type 명시
            with autocast(device_type='cuda'):
                loss = loss_wrapper(audio_mert, spec, target, progress)
                loss = loss / cfg.GRAD_ACCUM_STEPS
            
            # [추가] Loss NaN 체크
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf loss at step {step}, skipping...")
                optimizer.zero_grad()
                scaler.update()  # scaler 상태 업데이트
                nan_count += 1
                continue
            
            scaler.scale(loss).backward()
            current_loss_accum += loss.item()
            
            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                # Gradient Clipping (학습 안정화)
                scaler.unscale_(optimizer)
                
                # [추가] Gradient NaN 체크
                total_norm = torch.nn.utils.clip_grad_norm_(backbone.parameters(), MAX_GRAD_NORM)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"[WARNING] NaN/Inf gradient norm at step {step}, skipping update...")
                    optimizer.zero_grad()
                    scaler.update()
                    current_loss_accum = 0
                    nan_count += 1
                    continue
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # [수정] 실제 업데이트가 일어났을 때만 step 증가
                global_update_step += 1
                
                # Logging (Restore original scale)
                pbar.set_postfix({
                    'loss': current_loss_accum * cfg.GRAD_ACCUM_STEPS, 
                    'lr': f"{current_lr:.2e}",
                    'prog': f"{progress:.3f}"
                })
                total_loss += current_loss_accum * cfg.GRAD_ACCUM_STEPS
                current_loss_accum = 0
            
            # [주석] 매 샘플마다가 아닌 업데이트마다만 증가하므로 이 라인 제거
        
        # Learning Rate Scheduler Step
        scheduler.step()
        
        # [추가] Epoch 종료 시 NaN 발생 횟수 출력
        if nan_count > 0:
            print(f"[WARNING] Epoch {epoch+1}: {nan_count} NaN/Inf occurrences")
            
        avg_loss = total_loss / max((len(dataloader) / cfg.GRAD_ACCUM_STEPS) - nan_count, 1)
        new_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}, LR: {new_lr:.2e}")
        
        # [수정] 매 5 에폭마다 체크포인트 저장 (새로운 naming)
        if (epoch + 1 - START_EPOCH) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/n2n_from50_ep{epoch+1}.pth"
            
            # Unwrap DataParallel
            if isinstance(backbone, nn.DataParallel):
                state_dict = backbone.module.state_dict()
            else:
                state_dict = backbone.state_dict()
            
            # [추가] 저장 전 NaN 체크
            has_nan = False
            for k, v in state_dict.items():
                if torch.isnan(v).any():
                    print(f"[ERROR] NaN in {k}, not saving checkpoint!")
                    has_nan = True
                    break
            
            if not has_nan:
                torch.save({
                    'model_state_dict': state_dict, 
                    'epoch': epoch+1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss
                }, save_path)
                print(f"Saved checkpoint to {save_path}")
            else:
                print(f"[ERROR] Checkpoint at epoch {epoch+1} has NaN, skipping save!")

if __name__ == "__main__":
    main()