# train.py
import os
# [중요] 0번, 1번 GPU만 보이게 설정 (코드 최상단 위치)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset
from src.utils import seed_everything
from tqdm import tqdm

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
    
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=cfg.LR)
    scaler = GradScaler()
    
    print(f"Start Training for {cfg.EPOCHS} epochs...")
    global_step = 0
    total_steps = cfg.EPOCHS * len(dataloader)
    
    optimizer.zero_grad()
    
    for epoch in range(cfg.EPOCHS):
        backbone.train()
        total_loss = 0
        current_loss_accum = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for step, (audio_mert, spec, target) in enumerate(pbar):
            audio_mert = audio_mert.to(device)
            spec = spec.to(device)
            target = target.to(device)
            
            progress = global_step / total_steps
            
            # Mixed Precision Training
            with autocast():
                loss = loss_wrapper(audio_mert, spec, target, progress)
                loss = loss / cfg.GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            current_loss_accum += loss.item()
            
            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Logging (Restore original scale)
                pbar.set_postfix({'loss': current_loss_accum * cfg.GRAD_ACCUM_STEPS, 'prog': f"{progress:.2f}"})
                total_loss += current_loss_accum * cfg.GRAD_ACCUM_STEPS
                current_loss_accum = 0
            
            global_step += 1
            
        avg_loss = total_loss / (len(dataloader) / cfg.GRAD_ACCUM_STEPS)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/n2n_ep{epoch+1}.pth"
            
            # Unwrap DataParallel
            if isinstance(backbone, nn.DataParallel):
                state_dict = backbone.module.state_dict()
            else:
                state_dict = backbone.state_dict()
                
            torch.save(state_dict, save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()