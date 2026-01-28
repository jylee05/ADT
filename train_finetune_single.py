#!/usr/bin/env python3
"""
Fine-tuning Script - SINGLE GPU VERSION (NCCL Error Fix)
- GPU 0번 하나만 사용하여 통신 에러 원천 차단
- DataParallel 로직 제거
"""

import os

# [핵심] 무조건 GPU 0번만 보이게 설정 (GPU 3번 및 통신 에러 회피)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast 
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset
from src.utils import seed_everything

MAX_GRAD_NORM = 1.0 

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, config):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Single GPU이므로 module. 처리 불필요
    model_state_dict = model.state_dict()
    
    # NaN 체크
    for k, v in model_state_dict.items():
        if torch.isnan(v).any():
            print(f"[ERROR] NaN in {k}, not saving checkpoint!")
            return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")

def train_epoch(model, loss_fn, train_loader, optimizer, scheduler, scaler, epoch, total_epochs, device, config):
    model.train()
    total_loss = 0.0
    current_loss_accum = 0.0
    nan_count = 0
    num_batches = len(train_loader)
    
    progress = epoch / total_epochs
    
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
        
        # NaN Skip
        if torch.isnan(audio_mert).any() or torch.isnan(spec).any() or torch.isnan(target_grid).any():
            nan_count += 1
            continue
        
        # Forward pass
        with autocast():
            loss = loss_fn(audio_mert, spec, target_grid, progress)
            loss = loss / config.GRAD_ACCUM_STEPS
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            scaler.update()
            nan_count += 1
            continue
        
        # Backward
        scaler.scale(loss).backward()
        current_loss_accum += loss.item()
        
        if (batch_idx + 1) % config.GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                optimizer.zero_grad()
                scaler.update()
                current_loss_accum = 0.0
                nan_count += 1
                continue
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Tracking
            batch_loss = current_loss_accum * config.GRAD_ACCUM_STEPS
            total_loss += batch_loss
            current_loss_accum = 0.0
            
            avg_loss = total_loss / max((batch_idx + 1) // config.GRAD_ACCUM_STEPS, 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'progress': f'{progress:.3f}'
            })
    
    scheduler.step()
    
    if nan_count > 0:
        print(f"[WARNING] Epoch {epoch+1}: {nan_count} NaN/Inf occurrences")
    
    avg_loss = total_loss / max((num_batches // config.GRAD_ACCUM_STEPS) - nan_count, 1)
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='saved_models_finetune')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    seed_everything(42)
    
    # Configuration
    config = Config()
    config.GRAD_ACCUM_STEPS = 6
    
    # [변경] 강제로 0번 GPU 사용
    device = torch.device("cuda:0")
    print(f"🔧 Fine-tuning Configuration (Single GPU Mode)")
    print(f"   Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("🤖 Initializing Flow Matching Transformer...")
    model = FlowMatchingTransformer(config).to(device)
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # [중요] DataParallel로 저장된 모델(module. 접두사)을 Single GPU용으로 변환
    if any(k.startswith('module.') for k in model_state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ Converted DataParallel checkpoint to Single GPU model")
    else:
        model.load_state_dict(model_state_dict)
        print("✅ Loaded Single GPU checkpoint")
    
    # DataParallel 설정 코드 제거됨 (에러 원인 차단)
    
    loss_fn = AnnealedPseudoHuberLoss(model, config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = GradScaler()
    
    print("📁 Loading datasets...")
    train_dataset = EGMDDataset(is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE, # GPU 1개이므로 OOM 나면 줄여야 함
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Resume Logic
    start_epoch = 0
    if args.resume:
        print(f"📂 Resuming from: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device)
        
        # Resume 시에도 키 매핑 확인
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
    
    print(f"\n🚀 Starting fine-tuning (Single GPU)...")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, 50):
        start_time = time.time()
        
        train_loss = train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler, scaler,
            epoch, 50, device, config
        )
        
        print(f"Epoch {epoch+1}/50 | Loss: {train_loss:.6f} | Time: {time.time()-start_time:.1f}s")
        
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, os.path.join(args.output_dir, 'best_model_finetune.pth'), config)
            
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, os.path.join(args.output_dir, f'finetune_ep{epoch+1}.pth'), config)
            
    # Final save
    save_checkpoint(model, optimizer, scheduler, 49, train_loss, os.path.join(args.output_dir, 'final_model_finetune.pth'), config)

if __name__ == '__main__':
    main()