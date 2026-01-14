# train.py
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset
from src.utils import seed_everything
from tqdm import tqdm

def main():
    seed_everything(42)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    
    # 1. Dataset & Loader
    print("Initializing Dataset...")
    dataset = EGMDDataset(is_train=True)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    
    # 2. Model & Loss
    print("Initializing Model...")
    backbone = FlowMatchingTransformer(cfg).to(device)
    # Loss wrapper requires access to model for MERT extraction
    loss_wrapper = AnnealedPseudoHuberLoss(backbone, cfg).to(device)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=cfg.LR)
    
    # 4. Training Loop
    print(f"Start Training for {cfg.EPOCHS} epochs...")
    global_step = 0
    total_steps = cfg.EPOCHS * len(dataloader)
    
    for epoch in range(cfg.EPOCHS):
        backbone.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for audio_mert, spec, target in pbar:
            audio_mert = audio_mert.to(device)
            spec = spec.to(device)
            target = target.to(device)
            
            # Annealing Progress (0 ~ 1)
            progress = global_step / total_steps
            
            # Calculate Loss
            loss = loss_wrapper(audio_mert, spec, target, progress)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'prog': f"{progress:.2f}"})
            global_step += 1
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/n2n_ep{epoch+1}.pth"
            torch.save(backbone.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    import os
    main()