# inference.py
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
    """
    score_grid: (Seq_Len, 14) - Onset/Velocity interleaved or stacked
    output_path: 저장할 .mid 파일 경로
    """
    # Grid shape: (T, 14) -> (T, 7, 2)
    # 0: Onset, 1: Velocity
    n_channels = 7
    score = score_grid.reshape(-1, n_channels, 2)
    
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="N2N Drums")
    
    # Thresholds
    ONSET_THRESH = 0.5  # 0~1 사이 값이므로 0.5 기준 (학습 target이 1.0)
    
    for t in range(score.shape[0]):
        for ch in range(n_channels):
            onset_val = score[t, ch, 0]
            vel_val = score[t, ch, 1]
            
            if onset_val > ONSET_THRESH:
                # Time conversion
                start_time = t / fps
                end_time = start_time + 0.1 # 100ms duration (standard for drums)
                
                # Velocity conversion (-1~1 -> 0~127)
                # 모델이 -1~1 범위를 예측하도록 학습되었다고 가정
                vel_midi = int(np.clip((vel_val + 1) * 63.5, 0, 127))
                
                # Create Note
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
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = FlowMatchingTransformer(cfg).to(device)
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Flow Wrapper for sampling
    rf_wrapper = RectifiedFlowLoss(model)
    
    # 2. Load Audio
    print(f"Processing audio: {args.input}")
    wav, sr = torchaudio.load(args.input)
    
    # Mix down to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    # Resample
    if sr != cfg.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, cfg.SAMPLE_RATE).to(wav.device)
        wav = resampler(wav)
    
    wav = wav.to(device)
    
    # 3. Generate Score (Sampling)
    # steps=10 is recommended for N2N-like quality, but 2-4 might work for Rectified Flow
    print("Generating drum score...")
    with torch.no_grad():
        # Add batch dim: (1, samples)
        wav_batch = wav.unsqueeze(0) 
        generated_score = rf_wrapper.sample(wav_batch, steps=args.steps)
        
    # Remove batch dim & to CPU
    score_np = generated_score[0].cpu().numpy()
    
    # 4. Convert to MIDI
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