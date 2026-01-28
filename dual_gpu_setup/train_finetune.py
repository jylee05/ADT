#!/usr/bin/env python3
"""
Fine-tuning Script for N2N Flow Matching Drum Transcription
- Positive Weighting Loss (10x for drum hits)
- Fresh optimizer/scheduler setup
- 50 epochs with new progress calculation
- Gradient clipping for stability
"""

# [중요] 모든 import 전에 CUDA 설정 (train_from65epoch.py 스타일)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 2. [필수 추가] GPU 간 통신 에러 방지 (시스템 메모리 사용 강제)
# 하드웨어 상태가 불안정할 때 멈춤 현상을 막아줍니다.
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset
from src.utils import seed_everything

# Gradient Clipping을 위한 max norm (Flow Matching에 적합하게 조정)
MAX_GRAD_NORM = 1.0  # Flow Matching에 적합한 값

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, config):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Handle DataParallel
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    # [추가] 저장 전 NaN 체크
    has_nan = False
    for k, v in model_state_dict.items():
        if torch.isnan(v).any():
            print(f"[ERROR] NaN in {k}, not saving checkpoint!")
            has_nan = True
            break
    
    if has_nan:
        print(f"[ERROR] Checkpoint has NaN, skipping save!")
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': {
            'HIDDEN_DIM': config.HIDDEN_DIM,
            'N_LAYERS': config.N_LAYERS,
            'N_HEADS': config.N_HEADS,
            'DRUM_CHANNELS': config.DRUM_CHANNELS,
            'FEATURE_DIM': config.FEATURE_DIM,
            'MERT_DIM': config.MERT_DIM,
            'N_MELS': config.N_MELS,
            'MERT_LAYER_IDX': config.MERT_LAYER_IDX,
            'COND_LAYERS': config.COND_LAYERS
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")

def train_epoch(model, loss_fn, train_loader, optimizer, scheduler, scaler, epoch, total_epochs, device, config):
    """Train one epoch with fine-tuning settings"""
    model.train()
    total_loss = 0.0
    current_loss_accum = 0.0
    nan_count = 0
    num_batches = len(train_loader)
    
    # [Fine-tuning] Progress calculation: 0-1 over 50 epochs (not 150)
    progress = epoch / total_epochs  # Reset progress for fine-tuning
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
    progress_bar.set_postfix({
        'loss': '---',
        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
        'progress': f'{progress:.3f}'
    })
    
    optimizer.zero_grad()
    
    for batch_idx, (audio_mert, spec, target_grid) in enumerate(progress_bar):
        # Move to device
        audio_mert = audio_mert.to(device)
        spec = spec.to(device)
        target_grid = target_grid.to(device)
        
        # [추가] 입력 데이터 NaN 체크
        if torch.isnan(audio_mert).any() or torch.isnan(spec).any() or torch.isnan(target_grid).any():
            print(f"Skipping batch {batch_idx} due to NaN in input")
            nan_count += 1
            continue
        
        # Forward pass with mixed precision (train_from65epoch.py 스타일)
        with autocast('cuda'):
            loss = loss_fn(audio_mert, spec, target_grid, progress)
            loss = loss / config.GRAD_ACCUM_STEPS  # Gradient accumulation
        
        # [추가] Loss NaN 체크
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] NaN/Inf loss at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            scaler.update()
            nan_count += 1
            continue
        
        # Backward pass
        scaler.scale(loss).backward()
        current_loss_accum += loss.item()
        
        if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0:
            # [Fine-tuning] Gradient Clipping for stability (before scaler.step)
            scaler.unscale_(optimizer)
            
            # [추가] Gradient NaN 체크
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(f"[WARNING] NaN/Inf gradient norm at batch {batch_idx}, skipping update...")
                optimizer.zero_grad()
                scaler.update()
                current_loss_accum = 0.0
                nan_count += 1
                continue
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Tracking (restore original scale)
            batch_loss = current_loss_accum * config.GRAD_ACCUM_STEPS
            total_loss += batch_loss
            current_loss_accum = 0.0
            
            # Update progress bar
            avg_loss = total_loss / max((batch_idx + 1) // config.GRAD_ACCUM_STEPS, 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'progress': f'{progress:.3f}'
            })
            
            # 기본 로깅만 유지
            global_step = epoch * (num_batches // config.GRAD_ACCUM_STEPS) + ((batch_idx + 1) // config.GRAD_ACCUM_STEPS)
    
    # Step scheduler (CosineAnnealingLR)
    scheduler.step()
    
    # [추가] Epoch 종료 시 NaN 발생 횟수 출력
    if nan_count > 0:
        print(f"[WARNING] Epoch {epoch+1}: {nan_count} NaN/Inf occurrences")
    
    avg_loss = total_loss / max((num_batches // config.GRAD_ACCUM_STEPS) - nan_count, 1)
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Fine-tune N2N Flow Matching Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint for fine-tuning')
    parser.add_argument('--output_dir', type=str, default='saved_models_finetune',
                        help='Directory to save fine-tuned models')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from fine-tuning checkpoint')
    
    args = parser.parse_args()
    
    # Seed 설정
    seed_everything(42)
    
    # Configuration
    config = Config()
    # [강제 설정] Effective Batch Size 확보 (12 * 6 = 72, 논문 기준 64 이상)
    config.GRAD_ACCUM_STEPS = 6
    device = torch.device(config.DEVICE)
    print(f"🔧 Fine-tuning Configuration:")
    print(f"   Device: {device}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Total epochs: 50 (fine-tuning)")
    print(f"   Learning rate: 1e-4 → 1e-6 (CosineAnnealing)")
    print(f"   Positive weighting: 10x for drum hits")
    print(f"   Gradient clipping: max_norm=1.0")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model (train_from65epoch.py 스타일)
    print("🤖 Initializing Flow Matching Transformer...")
    model = FlowMatchingTransformer(config).to(device)
    
    # Load checkpoint BEFORE DataParallel setup (train_from65epoch.py 스타일)
    print(f"📂 Loading checkpoint for fine-tuning: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Model state dict 로드 (single GPU 모델로 먼저 로드)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle DataParallel naming mismatch
    if any(k.startswith('module.') for k in model_state_dict.keys()):
        # Checkpoint is DataParallel -> remove 'module.' prefix for single model
        new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ Loaded DataParallel checkpoint into single model")
    else:
        model.load_state_dict(model_state_dict)
        print("✅ Loaded single model checkpoint")
    
    # 로드된 체크포인트 정보 출력
    loaded_epoch = checkpoint.get('epoch', 'Unknown')
    loaded_loss = checkpoint.get('loss', 'Unknown')
    print(f"   - Original epoch: {loaded_epoch}")
    print(f"   - Original loss: {loaded_loss}")
    print(f"   - Optimizer/Scheduler: Freshly initialized for fine-tuning")
    
    # Multi-GPU setup AFTER model loading (train_from65epoch.py 스타일)
    if torch.cuda.device_count() > 1:
        print(f"🚀 Setting up DataParallel with {torch.cuda.device_count()} GPUs: {list(range(torch.cuda.device_count()))}")
        model = nn.DataParallel(model)
        print("✅ DataParallel setup completed")
    else:
        print(f"🔧 Using single GPU: {device}")
    
    # Loss function 초기화는 DataParallel 후에 (train_from65epoch.py 스타일)
    loss_fn = AnnealedPseudoHuberLoss(model, config).to(device)
    
    # [Fine-tuning] Fresh optimizer and scheduler setup (train_from65epoch.py 스타일)
    print("⚙️ Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    
    # Mixed precision scaler (train_from65epoch.py 스타일)
    scaler = GradScaler()
    
    # Data loaders (train_from65epoch.py 스타일)
    print("📁 Loading datasets...")
    train_dataset = EGMDDataset(is_train=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"📊 Dataset size:")
    print(f"   Train: {len(train_dataset)} samples")
    
    # Resume from fine-tuning checkpoint if specified (train_from65epoch.py 스타일)
    start_epoch = 0
    if args.resume:
        print(f"📂 Resuming fine-tuning from: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Load everything for resume (different from initial load)
        if hasattr(model, 'module'):
            model.module.load_state_dict(resume_checkpoint['model_state_dict'])
        else:
            # Handle DataParallel mismatch in resume checkpoint
            resume_state_dict = resume_checkpoint['model_state_dict']
            if any(k.startswith('module.') for k in resume_state_dict.keys()):
                new_state_dict = {k.replace('module.', ''): v for k, v in resume_state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(resume_state_dict)
        
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        start_epoch = resume_checkpoint['epoch'] + 1
        
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # Training loop
    total_epochs = 50  # Fixed for fine-tuning
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting fine-tuning...")
    print(f"   Epochs: {start_epoch} → {total_epochs}")
    print(f"   Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    print("="*70)
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        
        # Training (Validation 제거)
        train_loss = train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler, scaler,
            epoch, total_epochs, device, config
        )
        
        # Timing
        epoch_time = time.time() - epoch_start_time
        
        # Logging (TensorBoard 제거)
        print(f"\nEpoch {epoch+1}/{total_epochs} Summary:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        print(f"   Time: {epoch_time:.1f}s")
        
        # Save best model (Training Loss 기준)
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model_path = os.path.join(args.output_dir, 'best_model_finetune.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, best_model_path, config)
            print(f"🏆 New best training loss: {train_loss:.6f}")
        
        # Periodic save
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'finetune_ep{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, save_path, config)
        
        print("-" * 70)
    
    # Final save
    final_path = os.path.join(args.output_dir, 'final_model_finetune.pth')
    save_checkpoint(model, optimizer, scheduler, total_epochs-1, train_loss, final_path, config)
    
    print(f"\n🎉 Fine-tuning completed!")
    print(f"   Best training loss: {best_val_loss:.6f}")
    print(f"   Models saved in: {args.output_dir}")

if __name__ == '__main__':
    main()