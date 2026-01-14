import torch
import torchaudio
import pretty_midi
import argparse
import os
import numpy as np
from src.config import Config
from src.model import FlowMatchingTransformer, RectifiedFlowLoss
from src.utils import REVERSE_DRUM_MAPPING, seed_everything

def save_midi(score_grid, output_path, fps=100):
    n_channels = 7
    score = score_grid.reshape(-1, n_channels, 2)
    
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="N2N Drums")
    
    ONSET_THRESH = 0.5
    
    for t in range(score.shape[0]):
        for ch in range(n_channels):
            onset_val = score[t, ch, 0]
            vel_val = score[t, ch, 1]
            
            if onset_val > ONSET_THRESH:
                start_time = t / fps
                end_time = start_time + 0.1
                vel_midi = int(np.clip((vel_val + 1) * 63.5, 0, 127))
                
                pitch = REVERSE_DRUM_MAPPING[ch]
                note = pretty_midi.Note(
                    velocity=vel_midi,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                inst.notes.append(note)
    
    pm.instruments.append(inst)
    pm.write(output_path)
    print(f"MIDI saved to {output_path}")

def inference(args):
    seed_everything(42)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    
    print(f"Loading model from {args.checkpoint}...")
    model = FlowMatchingTransformer(cfg).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    rf_wrapper = RectifiedFlowLoss(model)
    
    print(f"Processing audio: {args.input}")
    wav, sr = torchaudio.load(args.input)
    
    # [FIX] Force mono and remove channel dim: (Channels, Time) -> (Time,)
    wav = torch.mean(wav, dim=0)
        
    if sr != cfg.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, cfg.SAMPLE_RATE).to(wav.device)
        wav = resampler(wav)
    
    wav = wav.to(device)
    
    print("Generating drum score...")
    with torch.no_grad():
        # [FIX] Add batch dim: (Time,) -> (1, Time)
        wav_batch = wav.unsqueeze(0) 
        generated_score = rf_wrapper.sample(wav_batch, steps=args.steps)
        
    score_np = generated_score[0].cpu().numpy()
    
    output_filename = os.path.splitext(os.path.basename(args.input))[0] + "_pred.mid"
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_midi(score_np, output_path, fps=cfg.FPS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N2N-Flow Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio (.wav)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save MIDI")
    parser.add_argument("--steps", type=int, default=10, help="Number of flow steps (1-10)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    inference(args)