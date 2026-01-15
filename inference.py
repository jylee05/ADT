import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pretty_midi
from scipy.signal import find_peaks
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from src.config import Config
from src.model import FlowMatchingTransformer

# [수정] dataset.py의 7개 클래스 순서에 맞춘 Representative MIDI Note
# 0: Kick, 1: Snare, 2: HH, 3: Toms, 4: Crash, 5: Ride, 6: Bell
DRUM_MAPPING = [36, 38, 42, 48, 49, 51, 56]

class ADTInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        
        # 모델 로드
        self.model = FlowMatchingTransformer(self.config).to(self.device)
        self.load_checkpoint(args.ckpt_path)
        self.model.eval()

        self.init_feature_extractors()

    def load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        # DataParallel로 저장된 경우 module. 접두어 제거
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

    def init_feature_extractors(self):
        # 1. Mel Spectrogram (dataset.py와 동일 설정)
        self.mel_transform = MelSpectrogram(
            sample_rate=self.config.AUDIO_SR,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            normalized=True
        ).to(self.device)
        
        # AmplitudeToDB 사용 (학습과 스케일 통일)
        self.db_transform = AmplitudeToDB().to(self.device)

    def get_features(self, waveform_segment, sr):
        # 1. Mel-Spectrogram
        if sr != self.config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(sr, self.config.AUDIO_SR).to(self.device)
            waveform_mel = resampler(waveform_segment).to(self.device)
        else:
            waveform_mel = waveform_segment.to(self.device)

        melspec = self.mel_transform(waveform_mel)
        melspec = self.db_transform(melspec)
        melspec = melspec.transpose(1, 2) # (B, T, n_mels)

        # 2. MERT Feature
        target_mert_sr = self.config.MERT_SR
        if sr != target_mert_sr:
            resampler_mert = torchaudio.transforms.Resample(sr, target_mert_sr).to(self.device)
            waveform_mert = resampler_mert(waveform_segment.to(self.device))
        else:
            waveform_mert = waveform_segment.to(self.device)

        mert_feat = self.model.extract_mert(waveform_mert)
        # mert_feat shape: (B, T_mert, D_mert)
        
        # [참고] MERT feature의 interpolation은 model.forward() 내부에서 처리됨

        return mert_feat, melspec

    @torch.no_grad()
    def solve_euler(self, x, mert_feat, spec_feat, t_start=0.0, t_end=1.0, steps=50):
        dt = (t_end - t_start) / steps
        times = torch.linspace(t_start, t_end, steps + 1).to(self.device)
        
        for i in range(steps):
            t_curr = torch.ones(x.shape[0], device=self.device) * times[i]
            v_pred = self.model(x, t_curr, mert_feat, spec_feat)
            x = x + v_pred * dt
        return x

    def run(self):
        print(f"Processing audio: {self.args.audio_path}")
        waveform, sr = torchaudio.load(self.args.audio_path)
        
        # [참고] 학습은 Channel 0만 사용하나, 인퍼런스는 Mono Mix 사용 (일반적임)
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        input_filename = os.path.splitext(os.path.basename(self.args.audio_path))[0]
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{input_filename}.mid")
        print(f"Result will be saved to: {save_path}")

        CHUNK_SEC = self.config.SEGMENT_SEC
        OVERLAP_SEC = 1.0 
        
        total_samples = waveform.shape[1]
        chunk_samples = int(CHUNK_SEC * sr)
        stride_samples = int((CHUNK_SEC - OVERLAP_SEC) * sr)
        
        outputs = []
        
        print(f"Total samples: {total_samples}, Chunk: {chunk_samples}, Stride: {stride_samples}")

        # Sliding Window
        for start_idx in range(0, total_samples, stride_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            wav_chunk = waveform[:, start_idx:end_idx]
            original_len = wav_chunk.shape[1]
            
            # 마지막 조각 패딩
            if original_len < chunk_samples:
                wav_chunk = torch.nn.functional.pad(wav_chunk, (0, chunk_samples - original_len))
            
            mert_feat, spec_feat = self.get_features(wav_chunk, sr)
            
            # [수정] Output Dimension 계산 - spec_feat의 시간 길이를 사용
            out_dim = self.config.DRUM_CHANNELS * self.config.FEATURE_DIM 
            seq_len = spec_feat.shape[1]  # Spectrogram의 시간 길이
            x_0 = torch.randn(spec_feat.shape[0], seq_len, out_dim).to(self.device)
            
            generated = self.solve_euler(x_0, mert_feat, spec_feat, steps=self.args.steps)
            
            if self.args.refine_step > 0:
                t_refine = 1.0 - self.args.refine_strength
                noise = torch.randn_like(generated)
                x_refine = (1 - t_refine) * noise + t_refine * generated
                refine_steps = int(self.args.steps * self.args.refine_strength)
                generated = self.solve_euler(x_refine, mert_feat, spec_feat, t_start=t_refine, t_end=1.0, steps=refine_steps)

            gen_np = generated[0].cpu().numpy()
            
            # [수정] Stitching (Overlap 제거) - FPS 기반 계산 통일
            fps = self.config.FPS  # = AUDIO_SR / HOP_LENGTH = 100

            # 앞부분 오버랩 제거
            valid_start = int((OVERLAP_SEC / 2) * fps) if start_idx > 0 else 0
            # 뒷부분 오버랩 제거
            valid_end = int(gen_np.shape[0] - (OVERLAP_SEC / 2) * fps) if end_idx < total_samples else gen_np.shape[0]
            
            # 패딩 부분 제거
            if original_len < chunk_samples:
                # 유효한 오디오 길이 비율로 프레임 수 계산
                ratio = original_len / chunk_samples
                real_end_frame = int(gen_np.shape[0] * ratio)
                valid_end = min(valid_end, real_end_frame)

            # 유효 구간이 존재할 때만 추가
            if valid_end > valid_start:
                outputs.append(gen_np[valid_start:valid_end])

        if len(outputs) == 0:
            print("Error: No audio processed or outputs are empty.")
            return

        full_output = np.concatenate(outputs, axis=0)
        
        # 디버그: 출력값 범위 확인
        print(f"Output Stats - Min: {full_output.min():.3f}, Max: {full_output.max():.3f}, Mean: {full_output.mean():.3f}")
        
        self.save_midi(full_output, save_path)

    def save_midi(self, raw_output, output_path):
        print("Converting to MIDI...")
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        time_per_frame = self.config.HOP_LENGTH / self.config.AUDIO_SR

        num_drums = self.config.DRUM_CHANNELS 
        has_notes = False
        
        for i in range(num_drums):
            if i >= len(DRUM_MAPPING): continue
            drum_note = DRUM_MAPPING[i]
            
            onsets = raw_output[:, i]
            vels = raw_output[:, i + num_drums]
            
            # Peak Picking
            # threshold 0.0은 -1~1 범위의 중간값
            peaks, _ = find_peaks(onsets, height=0.0, distance=3)
            
            for p in peaks:
                vel_val = np.clip(((vels[p] + 1) / 2) * 127, 1, 127)
                
                start = p * time_per_frame
                note = pretty_midi.Note(velocity=int(vel_val), pitch=drum_note, start=start, end=start+0.1)
                inst.notes.append(note)
                has_notes = True

        if not has_notes:
            print("WARNING: No notes detected in the output! The model might need more training or threshold adjustment.")
        
        pm.instruments.append(inst)
        pm.write(output_path)
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--refine_step', type=int, default=0)
    parser.add_argument('--refine_strength', type=float, default=0.3)
    
    args = parser.parse_args()
    ADTInference(args).run()