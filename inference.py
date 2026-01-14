# inference.py
import torch
import torchaudio
import pretty_midi
import argparse
import os
import numpy as np
from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
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
                vel_midi = int(np.clip((vel_val + 1) * 63.5, 0, 127))
                note = pretty_midi.Note(velocity=vel_midi, pitch=REVERSE_DRUM_MAPPING[ch], start=start_time, end=start_time+0.1)
                inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(output_path)
    print(f"MIDI saved to {output_path}")

def prepare_audio(audio_path, cfg, device):
    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    
    resampler_spec = torchaudio.transforms.Resample(sr, cfg.AUDIO_SR).to(wav.device)
    wav_spec = resampler_spec(wav)
    spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.AUDIO_SR, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, n_mels=cfg.N_MELS, normalized=True).to(wav.device)
    db_transform = torchaudio.transforms.AmplitudeToDB().to(wav.device)
    spec = spec_transform(wav_spec)
    spec = db_transform(spec).transpose(1, 2)
    
    resampler_mert = torchaudio.transforms.Resample(sr, cfg.MERT_SR).to(wav.device)
    wav_mert = resampler_mert(wav)
    
    return wav_mert.to(device), spec.to(device)

def inference(args):
    seed_everything(42)
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    
    print(f"Loading model from {args.checkpoint}...")
    model = FlowMatchingTransformer(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    rf_wrapper = AnnealedPseudoHuberLoss(model, cfg)
    wav_mert, spec = prepare_audio(args.input, cfg, device)
    
    # Refinement Setup (Optional)
    init_score = None
    if args.refine_midi:
        # Load MIDI logic here if needed, or pass purely noise
        pass 
    
    print("Generating drum score...")
    with torch.no_grad():
        # start_t=0.0 -> Generation, start_t > 0 -> Refinement
        generated_score = rf_wrapper.sample(wav_mert, spec, steps=args.steps, init_score=init_score, start_t=args.start_t)
        
    score_np = generated_score[0].cpu().numpy()
    output_filename = os.path.splitext(os.path.basename(args.input))[0] + "_pred.mid"
    output_path = os.path.join(args.output_dir, output_filename)
    save_midi(score_np, output_path, fps=cfg.FPS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input audio (.wav)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--start_t", type=float, default=0.0, help="Start time for flow (0.0=Gen, >0=Refine)")
    parser.add_argument("--refine_midi", type=str, default=None, help="Initial MIDI for refinement (optional)")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    inference(args)