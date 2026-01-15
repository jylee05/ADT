import os
import argparse
import torch
import torchaudio
import numpy as np
import pretty_midi
from scipy.signal import find_peaks
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from torchaudio.transforms import MelSpectrogram
from src.model import DrumFlowMatching
from src.config import ADTConfig

# MIDI 매핑
DRUM_MAPPING = [36, 38, 42, 46, 43, 47, 50, 49, 51]

class ADTInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ADTConfig()
        
        # 모델 로드
        self.model = DrumFlowMatching(self.config).to(self.device)
        self.load_checkpoint(args.ckpt_path)
        self.model.eval()

        # MERT & Mel 초기화
        self.init_feature_extractors()

    def load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

    def init_feature_extractors(self):
        self.mel_transform = MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=2048,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            normalized=True
        ).to(self.device)

        print("Loading MERT...")
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(self.device)
        self.mert_model.eval()

    def get_features(self, waveform_segment, sr):
        # 1. Mel-Spectrogram
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate).to(self.device)
            waveform_mel = resampler(waveform_segment).to(self.device)
        else:
            waveform_mel = waveform_segment.to(self.device)

        melspec = self.mel_transform(waveform_mel)
        melspec = torch.log(torch.clamp(melspec, min=1e-5))
        melspec = melspec.transpose(1, 2) 

        # 2. MERT Feature
        target_mert_sr = 24000
        if sr != target_mert_sr:
            resampler_mert = torchaudio.transforms.Resample(sr, target_mert_sr)
            waveform_mert = resampler_mert(waveform_segment.cpu()).squeeze().numpy()
        else:
            waveform_mert = waveform_segment.squeeze().cpu().numpy()
        
        if waveform_mert.ndim == 0: waveform_mert = waveform_mert[None] 

        inputs = self.mert_processor(waveform_mert, sampling_rate=target_mert_sr, return_tensors="pt")
        with torch.no_grad():
            outputs = self.mert_model(**inputs.to(self.device), output_hidden_states=True)
            mert_feat = outputs.hidden_states[-1]

        # 3. Time Align
        target_len = melspec.shape[1]
        mert_feat = mert_feat.transpose(1, 2)
        mert_feat = torch.nn.functional.interpolate(mert_feat, size=target_len, mode='linear', align_corners=False)
        mert_feat = mert_feat.transpose(1, 2)

        return torch.cat([melspec, mert_feat], dim=-1)

    @torch.no_grad()
    def solve_euler(self, x, condition, t_start=0.0, t_end=1.0, steps=50):
        dt = (t_end - t_start) / steps
        times = torch.linspace(t_start, t_end, steps + 1).to(self.device)
        
        for i in range(steps):
            t_curr = torch.ones(x.shape[0], device=self.device) * times[i]
            v_pred = self.model(x, t_curr, condition)
            x = x + v_pred * dt
        return x

    def run(self):
        print(f"Processing audio: {self.args.audio_path}")
        waveform, sr = torchaudio.load(self.args.audio_path)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ---------------------------------------------------------
        # Output Path 설정 로직 (수정됨)
        # ---------------------------------------------------------
        # 입력 파일명 추출 (확장자 제거)
        input_filename = os.path.splitext(os.path.basename(self.args.audio_path))[0]
        
        # 저장할 폴더: outputs (없으면 생성)
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 최종 저장 경로: outputs/{곡이름}.mid
        save_path = os.path.join(output_dir, f"{input_filename}.mid")
        print(f"Result will be saved to: {save_path}")

        # Sliding Window Logic
        CHUNK_SEC = 10.0
        OVERLAP_SEC = 2.0
        
        total_samples = waveform.shape[1]
        chunk_samples = int(CHUNK_SEC * sr)
        stride_samples = int((CHUNK_SEC - OVERLAP_SEC) * sr)
        
        outputs = []
        
        for start_idx in range(0, total_samples, stride_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            wav_chunk = waveform[:, start_idx:end_idx]
            original_len = wav_chunk.shape[1]
            if original_len < chunk_samples:
                wav_chunk = torch.nn.functional.pad(wav_chunk, (0, chunk_samples - original_len))
            
            condition = self.get_features(wav_chunk, sr)
            x_0 = torch.randn(condition.shape[0], condition.shape[1], 18).to(self.device)
            generated = self.solve_euler(x_0, condition, steps=self.args.steps)
            
            if self.args.refine_step > 0:
                t_refine = 1.0 - self.args.refine_strength
                noise = torch.randn_like(generated)
                x_refine = (1 - t_refine) * noise + t_refine * generated
                refine_steps = int(self.args.steps * self.args.refine_strength)
                generated = self.solve_euler(x_refine, condition, t_start=t_refine, t_end=1.0, steps=refine_steps)

            gen_np = generated[0].cpu().numpy()
            
            mel_hop = self.config.hop_length * (sr / self.config.sample_rate)
            valid_start = int((OVERLAP_SEC / 2) * (sr / mel_hop)) if start_idx > 0 else 0
            valid_end = int(gen_np.shape[0] - (OVERLAP_SEC / 2) * (sr / mel_hop)) if end_idx < total_samples else gen_np.shape[0]
            
            if original_len < chunk_samples:
                real_end_frame = int(original_len * (sr / mel_hop) / chunk_samples * gen_np.shape[0])
                valid_end = min(valid_end, real_end_frame)

            outputs.append(gen_np[valid_start:valid_end])

        full_output = np.concatenate(outputs, axis=0)
        self.save_midi(full_output, save_path)

    def save_midi(self, raw_output, output_path):
        print("Converting to MIDI...")
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        time_per_frame = self.config.hop_length / self.config.sample_rate

        for i, drum_note in enumerate(DRUM_MAPPING):
            onsets = raw_output[:, i]
            vels = raw_output[:, i+9]
            
            peaks, _ = find_peaks(onsets, height=0.0, distance=3)
            
            for p in peaks:
                vel_val = np.clip(((vels[p] + 1) / 2) * 127, 1, 127)
                start = p * time_per_frame
                note = pretty_midi.Note(velocity=int(vel_val), pitch=drum_note, start=start, end=start+0.1)
                inst.notes.append(note)

        pm.instruments.append(inst)
        pm.write(output_path)
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True, help='Input audio file path')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to trained checkpoint')
    # output_path 인자는 이제 필요 없으므로 제거 (자동 생성됨)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--refine_step', type=int, default=0)
    parser.add_argument('--refine_strength', type=float, default=0.3)
    
    args = parser.parse_args()
    ADTInference(args).run()