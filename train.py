# train.py
import os
# [중요] 0번, 1번 GPU만 보이게 설정 (코드 최상단 위치)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # [수정] torch.cuda.amp deprecated

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset
from src.utils import seed_everything
from tqdm import tqdm

# Gradient Clipping을 위한 max norm (더 강화)
MAX_GRAD_NORM = 0.25  # [수정] 0.5 -> 0.25로 더 강화

def check_for_nan(tensor, name):
    """NaN 체크 유틸리티"""
    if torch.isnan(tensor).any():
        print(f"[WARNING] NaN detected in {name}!")
        return True
    if torch.isinf(tensor).any():
        print(f"[WARNING] Inf detected in {name}!")
        return True
    return False

def main():
    seed_everything(42)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    
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
    
    # [설정] GPU 2개 모두 사용
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        backbone = nn.DataParallel(backbone)
    
    # 수정된 Loss Wrapper 사용
    loss_wrapper = AnnealedPseudoHuberLoss(backbone, cfg).to(device)
    
    # [수정] Learning Rate 더 낮춤 + betas 조정 (논문 준수를 위해)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.99))
    scaler = GradScaler('cuda')  # PyTorch 2.x 호환
    
    print(f"Start Training for {cfg.EPOCHS} epochs...")
    global_step = 0
    total_steps = cfg.EPOCHS * len(dataloader)
    
    optimizer.zero_grad()
    
    for epoch in range(cfg.EPOCHS):
        backbone.train()
        total_loss = 0
        current_loss_accum = 0
        nan_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for step, (audio_mert, spec, target) in enumerate(pbar):
            audio_mert = audio_mert.to(device)
            spec = spec.to(device)
            target = target.to(device)
            
            # [추가] 입력 데이터 NaN 체크
            if check_for_nan(audio_mert, "audio_mert") or check_for_nan(spec, "spec") or check_for_nan(target, "target"):
                print(f"Skipping batch {step} due to NaN in input")
                nan_count += 1
                continue
            
            progress = global_step / total_steps
            
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
                
                # Logging (Restore original scale)
                pbar.set_postfix({'loss': current_loss_accum * cfg.GRAD_ACCUM_STEPS, 'prog': f"{progress:.2f}"})
                total_loss += current_loss_accum * cfg.GRAD_ACCUM_STEPS
                current_loss_accum = 0
            
            global_step += 1
        
        # [추가] Epoch 종료 시 NaN 발생 횟수 출력
        if nan_count > 0:
            print(f"[WARNING] Epoch {epoch+1}: {nan_count} NaN/Inf occurrences")
            
        avg_loss = total_loss / max((len(dataloader) / cfg.GRAD_ACCUM_STEPS) - nan_count, 1)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # [수정] 첫 번째 에폭 또는 매 5 에폭마다 체크포인트 저장
        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/n2n_ep{epoch+1}.pth"
            
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
                torch.save({'model_state_dict': state_dict, 'epoch': epoch+1}, save_path)
                print(f"Saved checkpoint to {save_path}")
            else:
                print(f"[ERROR] Checkpoint at epoch {epoch+1} has NaN, skipping save!")

if __name__ == "__main__":
    main()