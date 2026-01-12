# train.py
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.model import FlowMatchingTransformer, RectifiedFlowLoss
from src.dataset import EGMDDataset
from tqdm import tqdm

def main():
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    
    # 1. Dataset & Loader
    print("Initializing Dataset...")
    dataset = EGMDDataset()
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    
    # 2. Model & Loss
    print("Initializing Model...")
    backbone = FlowMatchingTransformer(cfg).to(device)
    rf_loss_fn = RectifiedFlowLoss(backbone).to(device)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=cfg.LR)
    
    # 4. Training Loop
    print(f"Start Training for {cfg.EPOCHS} epochs...")
    for epoch in range(cfg.EPOCHS):
        backbone.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for audio, target in pbar:
            audio = audio.to(device)
            target = target.to(device)
            
            # Calculate Loss
            loss = rf_loss_fn(audio, target)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = f"checkpoints/rf_n2n_ep{epoch+1}.pth"
            torch.save(backbone.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()