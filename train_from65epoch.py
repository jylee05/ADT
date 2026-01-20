# train_from65epoch_fixed.py
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

def load_checkpoint_full(model, optimizer, scheduler, checkpoint_path, device):
    """체크포인트 완전 복원 (모델 + optimizer + scheduler)"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # DataParallel wrapper 고려
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer 상태 복원 (있는 경우)
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer 상태 복원됨")
    else:
        print("⚠️ Optimizer 상태 없음 - 처음부터 시작")
    
    # Scheduler 상태 복원 (있는 경우) 
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✅ Scheduler 상태 복원됨")
    else:
        print("⚠️ Scheduler 상태 없음 - 처음부터 시작")
    
    loaded_epoch = checkpoint.get('epoch', 65)
    loaded_loss = checkpoint.get('loss', 'Unknown')
    
    print(f"✅ Checkpoint 완전 복원 완료!")
    print(f"  - Epoch: {loaded_epoch}")
    print(f"  - Loss: {loaded_loss}")
    
    return loaded_epoch

def main():
    seed_everything(42)
    cfg = Config()
    # [강제 설정] Effective Batch Size 확보 (12 * 6 = 72, 논문 기준 64 이상)
    cfg.GRAD_ACCUM_STEPS = 6
    device = torch.device(cfg.DEVICE)
    
    # [수정] 65 epoch부터 150까지 학습
    TOTAL_EPOCHS = 150
    START_EPOCH = 65  # 65 epoch부터 시작
    
    print(f"🚀 Resume training from epoch {START_EPOCH+1} to {TOTAL_EPOCHS}...")
    
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
    
    # [수정] Learning Rate를 1e-4로 설정 (peak LR)
    optimizer = torch.optim.AdamW(
        backbone.parameters(), 
        lr=1e-4,  # Peak learning rate
        weight_decay=0.01, 
        betas=(0.9, 0.999)  # [수정] 0.99 -> 0.999 (Flow Matching 안정성 향상)
    )
    
    # [핵심] 65 epoch 체크포인트 완전 복원 (scheduler 설정 전에 먼저)
    checkpoint_path = "checkpoints/n2n_from50_ep65.pth"
    if os.path.exists(checkpoint_path):
        # 임시로 전체 스케줄러 생성 (원래 train_from50epoch.py와 동일하게)
        temp_remaining_epochs = TOTAL_EPOCHS - 50  # 50부터 150까지의 원래 스케줄
        temp_scheduler = create_lr_scheduler(optimizer, temp_remaining_epochs, warmup_epochs=5)  # 원래 설정
        loaded_epoch = load_checkpoint_full(backbone, optimizer, temp_scheduler, checkpoint_path, device)
        if loaded_epoch != 66:  # 66부터 시작해야 함 (65 완료 후)
            print(f"⚠️ Warning: Expected epoch 66, got {loaded_epoch}")
        # 실제 시작 epoch 조정
        START_EPOCH = max(loaded_epoch - 1, 65)  # 65 이상에서 시작
        scheduler = temp_scheduler  # 복원된 원래 스케줄러 사용
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    scaler = GradScaler('cuda')  # PyTorch 2.x 호환
    
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # [수정] Progress 계산을 실제 업데이트 횟수 기준으로 수정 (65 epoch부터 시작)
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
            
            # [디버깅] 첫 번째 배치에서 progress 확인
            if epoch == START_EPOCH and step == 0:
                print(f"\n🔍 첫 번째 배치 확인:")
                print(f"   - global_update_step: {global_update_step}")
                print(f"   - total_update_steps: {total_update_steps}")
                print(f"   - progress: {progress:.4f}")
            
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
        
        # Learning Rate Scheduler Step
        scheduler.step()
        
        # [추가] Epoch 종료 시 NaN 발생 횟수 출력
        if nan_count > 0:
            print(f"[WARNING] Epoch {epoch+1}: {nan_count} NaN/Inf occurrences")
            
        avg_loss = total_loss / max((len(dataloader) / cfg.GRAD_ACCUM_STEPS) - nan_count, 1)
        new_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Epoch {epoch+1} Avg Loss: {avg_loss:.4f}, LR: {new_lr:.2e}")
        
        # [수정] 매 5 에폭마다 체크포인트 저장 (65부터 시작하므로 naming 조정)
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/n2n_from65_ep{epoch+1}.pth"
            
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
                print(f"💾 Saved checkpoint to {save_path}")
            else:
                print(f"[ERROR] Checkpoint at epoch {epoch+1} has NaN, skipping save!")
    
    print(f"\n🎉 Training completed! Final epoch: {TOTAL_EPOCHS}")

if __name__ == "__main__":
    main()