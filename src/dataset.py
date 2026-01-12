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
# 0:KD, 1:SD, 2:HH, 3:Tom, 4:Crash, 5:Ride, 6:Bell
DRUM_MAPPING = {
    35: 0, 36: 0, # Kick
    38: 1, 40: 1, 37: 1, # Snare
    42: 2, 44: 2, 46: 2, # Hi-hat (Closed, Pedal, Open)
    41: 3, 43: 3, 45: 3, 47: 3, 48: 3, 50: 3, # Toms
    49: 4, 57: 4, 55: 4, 52: 4, # Crash
    51: 5, 59: 5, 53: 5, # Ride
    56: 6, 54: 6 # Bell (Cowbell etc)
}

class EGMDDataset(Dataset):
    def __init__(self, is_train=True):
        self.config = Config()
        self.files = []
        
        # 데이터 폴더 스캔 (/public/e-gmd-v1.0.0/drummerX/...)
        search_path = os.path.join(self.config.DATA_ROOT, "drummer*")
        drummer_dirs = glob.glob(search_path)
        
        for d_dir in drummer_dirs:
            # 각 드러머 폴더 안의 세션 폴더 탐색 (구조에 따라 조정 필요)
            # 보통 drummer1/session1/audio.wav 이런 식임
            audio_files = glob.glob(os.path.join(d_dir, "**", "*.wav"), recursive=True)
            
            for aud_path in audio_files:
                # 같은 이름의 midi 파일 찾기
                mid_path = aud_path.replace(".wav", ".mid")
                if not os.path.exists(mid_path):
                    # .midi 확장자일 수도 있음
                    mid_path = aud_path.replace(".wav", ".midi")
                
                if os.path.exists(mid_path):
                    self.files.append((aud_path, mid_path))
        
        print(f"Found {len(self.files)} pairs.")

    def __len__(self):
        return len(self.files)

    def midi_to_grid(self, midi_path, duration):
        # 10ms 단위 그리드 생성
        n_frames = int(duration * self.config.FPS)
        # (Seq, 7, 2) -> (Seq, 14)
        grid = np.zeros((n_frames, self.config.DRUM_CHANNELS, 2), dtype=np.float32)
        # Init velocity to -1 (silence)
        grid[:, :, 1] = -1.0 
        grid[:, :, 0] = -1.0 # Onset logic (0 or 1 mapped to -1 or 1 later?) 
        # N2N uses -1~1 range. Let's say -1 is no onset, 1 is onset.
        
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            for instrument in pm.instruments:
                if instrument.is_drum:
                    for note in instrument.notes:
                        if note.pitch in DRUM_MAPPING:
                            idx = DRUM_MAPPING[note.pitch]
                            # Time to Frame
                            frame = int(note.start * self.config.FPS)
                            if frame < n_frames:
                                # Onset: 1.0
                                grid[frame, idx, 0] = 1.0
                                # Velocity: 0~127 -> -1 ~ 1
                                norm_vel = (note.velocity / 127.0) * 2 - 1
                                grid[frame, idx, 1] = norm_vel
        except Exception as e:
            print(f"Error parsing MIDI {midi_path}: {e}")
            
        return grid.reshape(n_frames, -1) # Flatten to 14 dim

    def __getitem__(self, idx):
        aud_path, mid_path = self.files[idx]
        
        # 1. Load Audio
        wav, sr = torchaudio.load(aud_path)
        
        # 2. Resample to 24k (MERT req)
        if sr != self.config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.config.SAMPLE_RATE)
            wav = resampler(wav)
            
        # 3. Random Crop (Training)
        full_len = wav.shape[1]
        seg_len = int(self.config.SEGMENT_SEC * self.config.SAMPLE_RATE)
        
        if full_len > seg_len:
            start = np.random.randint(0, full_len - seg_len)
            wav_crop = wav[:, start:start+seg_len]
            start_sec = start / self.config.SAMPLE_RATE
            dur_sec = self.config.SEGMENT_SEC
        else:
            # Pad if too short
            wav_crop = torch.nn.functional.pad(wav, (0, seg_len - full_len))
            start_sec = 0
            dur_sec = full_len / self.config.SAMPLE_RATE

        # 4. Process MIDI (Crop matching part)
        # 전체 MIDI를 파싱하는 건 비효율적일 수 있으나 E-GMD는 파일이 짧은 편이라 괜찮음.
        # 최적화를 위해선 MIDI도 start_sec 기준으로 잘라야 함.
        # 여기선 전체 변환 후 slicing 방식을 예시로 둠 (구현 편의상)
        total_duration = full_len / self.config.SAMPLE_RATE
        full_grid = self.midi_to_grid(mid_path, total_duration)
        
        start_frame = int(start_sec * self.config.FPS)
        n_frame_seg = int(self.config.SEGMENT_SEC * self.config.FPS)
        
        grid_crop = full_grid[start_frame : start_frame + n_frame_seg]
        
        # Pad grid if needed
        if grid_crop.shape[0] < n_frame_seg:
            pad_len = n_frame_seg - grid_crop.shape[0]
            # Pad with silence (-1)
            padding = np.ones((pad_len, 14)) * -1
            grid_crop = np.vstack([grid_crop, padding])
            
        return wav_crop[0], torch.from_numpy(grid_crop).float() # Mono audio, Float Target