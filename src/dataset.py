# src/dataset.py
import os
import glob
import torch
import torchaudio
import pretty_midi
import numpy as np
from torch.utils.data import Dataset
from .config import Config

# General MIDI Drum Map to 7 Channels
DRUM_MAPPING = {
    35: 0, 36: 0, # Kick
    38: 1, 40: 1, 37: 1, # Snare
    42: 2, 44: 2, 46: 2, # Hi-hat
    41: 3, 43: 3, 45: 3, 47: 3, 48: 3, 50: 3, # Toms
    49: 4, 57: 4, 55: 4, 52: 4, # Crash
    51: 5, 59: 5, 53: 5, # Ride
    56: 6, 54: 6 # Bell
}

class EGMDDataset(Dataset):
    def __init__(self, is_train=True):
        self.config = Config()
        self.files = []
        self.is_train = is_train
        
        # Spectrogram Transform (44.1kHz)
        self.spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.AUDIO_SR,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            normalized=True
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # Resamplers
        self.resample_mert = None # init lazily or per item if SR differs often
        
        # 데이터 폴더 스캔
        search_path = os.path.join(self.config.DATA_ROOT, "drummer*")
        drummer_dirs = glob.glob(search_path)
        
        for d_dir in drummer_dirs:
            audio_files = glob.glob(os.path.join(d_dir, "**", "*.wav"), recursive=True)
            for aud_path in audio_files:
                mid_path = aud_path.replace(".wav", ".mid")
                if not os.path.exists(mid_path):
                    mid_path = aud_path.replace(".wav", ".midi")
                
                if os.path.exists(mid_path):
                    self.files.append((aud_path, mid_path))
        
        print(f"Found {len(self.files)} pairs.")

    def __len__(self):
        return len(self.files)

    def midi_to_grid(self, midi_path, duration):
        n_frames = int(duration * self.config.FPS)
        grid = np.zeros((n_frames, self.config.DRUM_CHANNELS, 2), dtype=np.float32)
        grid[:, :, 1] = -1.0 # Velocity default
        grid[:, :, 0] = -1.0 # Onset default (-1: off, 1: on)
        
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            for instrument in pm.instruments:
                if instrument.is_drum:
                    for note in instrument.notes:
                        if note.pitch in DRUM_MAPPING:
                            idx = DRUM_MAPPING[note.pitch]
                            frame = int(note.start * self.config.FPS)
                            if frame < n_frames:
                                grid[frame, idx, 0] = 1.0
                                # Velocity: 0~127 -> -1 ~ 1
                                norm_vel = (note.velocity / 127.0) * 2 - 1
                                grid[frame, idx, 1] = norm_vel
        except Exception as e:
            print(f"Error parsing MIDI {midi_path}: {e}")
            
        return grid.reshape(n_frames, -1)

    def __getitem__(self, idx):
        aud_path, mid_path = self.files[idx]
        
        # 1. Load Audio (Original SR)
        wav, sr = torchaudio.load(aud_path)
        
        # 2. Resample for Spectrogram (44.1k)
        if sr != self.config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(sr, self.config.AUDIO_SR)
            wav_spec_in = resampler(wav)
        else:
            wav_spec_in = wav

        # 3. Resample for MERT (24k)
        if sr != self.config.MERT_SR:
            resampler_mert = torchaudio.transforms.Resample(sr, self.config.MERT_SR)
            wav_mert_in = resampler_mert(wav)
        else:
            wav_mert_in = wav
            
        # 4. Random Crop
        # 기준: 44.1k 샘플 수
        full_len_spec = wav_spec_in.shape[1]
        seg_len_spec = int(self.config.SEGMENT_SEC * self.config.AUDIO_SR)
        
        # MERT 길이 비율 계산
        ratio_mert = self.config.MERT_SR / self.config.AUDIO_SR
        seg_len_mert = int(self.config.SEGMENT_SEC * self.config.MERT_SR)
        
        if self.is_train and full_len_spec > seg_len_spec:
            start_spec = np.random.randint(0, full_len_spec - seg_len_spec)
            start_mert = int(start_spec * ratio_mert)
            
            wav_crop_spec = wav_spec_in[:, start_spec : start_spec + seg_len_spec]
            wav_crop_mert = wav_mert_in[:, start_mert : start_mert + seg_len_mert]
            
            start_sec = start_spec / self.config.AUDIO_SR
        else:
            # Pad if too short
            pad_len_spec = seg_len_spec - full_len_spec
            wav_crop_spec = torch.nn.functional.pad(wav_spec_in, (0, pad_len_spec))
            
            pad_len_mert = seg_len_mert - wav_mert_in.shape[1]
            wav_crop_mert = torch.nn.functional.pad(wav_mert_in, (0, pad_len_mert))
            
            start_sec = 0
            
        # 5. Generate Spectrogram
        # wav_crop_spec: (C, T) -> Use first channel
        spec = self.spec_transform(wav_crop_spec[0]) # (Freq, Time)
        spec = self.db_transform(spec)
        spec = spec.transpose(0, 1) # (Time, Freq)
        
        # 6. Process MIDI Target
        total_duration = full_len_spec / self.config.AUDIO_SR
        full_grid = self.midi_to_grid(mid_path, total_duration)
        
        start_frame = int(start_sec * self.config.FPS)
        n_frame_seg = int(self.config.SEGMENT_SEC * self.config.FPS)
        
        grid_crop = full_grid[start_frame : start_frame + n_frame_seg]
        
        if grid_crop.shape[0] < n_frame_seg:
            pad_len = n_frame_seg - grid_crop.shape[0]
            padding = np.ones((pad_len, 14)) * -1
            grid_crop = np.vstack([grid_crop, padding])
            
        # Return: Mert Audio, Spectrogram, Target Grid
        return wav_crop_mert[0], spec, torch.from_numpy(grid_crop).float()