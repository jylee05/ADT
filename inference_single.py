import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pretty_midi
from scipy.signal import find_peaks
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm

from src.config import Config
from src.model import FlowMatchingTransformer

# [수정] dataset.py의 7개 클래스 순서에 맞춘 Representative MIDI Note
# 0: Kick, 1: Snare, 2: HH, 3: Toms, 4: Crash, 5: Ride, 6: Bell
DRUM_MAPPING = [36, 38, 42, 48, 49, 51, 56]

class ADTInference:
    def __init__(self, args):
        # [수정] 물리적 GPU 1번만 사용하도록 강제 설정
        # 이렇게 하면 PyTorch는 GPU 1번 하나만 인식하게 되며, 내부적으로는 이를 'cuda:0'으로 취급합니다.
        # 따라서 num_gpus는 1이 되고, DataParallel 로직은 자동으로 건너뛰게 됩니다.
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        
        # GPU 개수 확인 (위 설정으로 인해 1로 감지됨)
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f"Available GPUs: {self.num_gpus} (Targeting Physical GPU 1)")
        
        # 모델 로드
        self.model = FlowMatchingTransformer(self.config).to(self.device)
        self.load_checkpoint(args.ckpt_path)
        
        # Multi-GPU 설정 (GPU가 1개로 인식되므로 이 조건문은 False가 되어 실행되지 않음)
        if self.num_gpus > 1:
            print(f"Using DataParallel on {self.num_gpus} GPUs for inference")
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.eval()
        self.init_feature_extractors()

    def load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # DataParallel로 저장된 경우를 위한 처리
        if any(k.startswith('module.') for k in state_dict.keys()):
            # 저장된 모델이 DataParallel이었다면 module. 접두어 제거
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict
            
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
        """오디오에서 Mel-Spectrogram과 MERT용 waveform 준비"""
        # waveform_segment: (1, samples) - 채널 차원이 1인 mono
        
        # 1. Mel-Spectrogram
        if sr != self.config.AUDIO_SR:
            resampler = torchaudio.transforms.Resample(sr, self.config.AUDIO_SR).to(self.device)
            waveform_mel = resampler(waveform_segment).to(self.device)
        else:
            waveform_mel = waveform_segment.to(self.device)

        melspec = self.mel_transform(waveform_mel)
        melspec = self.db_transform(melspec)
        melspec = melspec.transpose(1, 2) # (1, T, n_mels) - 여기서 1은 배치

        # 2. MERT용 Waveform (MERT 추출은 model.forward() 내부에서 수행)
        target_mert_sr = self.config.MERT_SR
        if sr != target_mert_sr:
            resampler_mert = torchaudio.transforms.Resample(sr, target_mert_sr).to(self.device)
            waveform_mert = resampler_mert(waveform_segment.to(self.device))
        else:
            waveform_mert = waveform_segment.to(self.device)

        # [핵심 수정] MERT는 (B, samples) 형태를 기대함
        # waveform_segment가 (1, samples) 형태인데, 1은 채널 차원임
        # squeeze(0)으로 채널 차원을 제거하고 unsqueeze(0)으로 배치 차원 추가
        # 결과: (1, samples) where 1 is batch dimension
        waveform_mert = waveform_mert.squeeze(0).unsqueeze(0)  # 채널 -> 배치 차원으로 변환
        
        # 디버그 출력
        # print(f"[DEBUG] waveform_mert shape: {waveform_mert.shape}, melspec shape: {melspec.shape}")
        
        return waveform_mert, melspec
    
    def process_batch(self, batch_chunks, sr, batch_start, total_chunks):
        """배치 단위로 오디오 청크들을 병렬 처리"""
        batch_audio_mert = []
        batch_spec_feat = []
        
        # Feature extraction for all chunks in batch
        for wav_chunk in batch_chunks:
            audio_mert, spec_feat = self.get_features(wav_chunk, sr)
            batch_audio_mert.append(audio_mert)
            batch_spec_feat.append(spec_feat)
        
        # Stack batches
        if len(batch_audio_mert) > 1:
            batch_audio_mert = torch.cat(batch_audio_mert, dim=0)  # (batch, samples)
            batch_spec_feat = torch.cat(batch_spec_feat, dim=0)    # (batch, T, n_mels)
        else:
            batch_audio_mert = batch_audio_mert[0]
            batch_spec_feat = batch_spec_feat[0]
            
        # Output Dimension 계산
        out_dim = self.config.DRUM_CHANNELS * self.config.FEATURE_DIM 
        batch_size = batch_spec_feat.shape[0]
        seq_len = batch_spec_feat.shape[1]  # Spectrogram의 시간 길이
        x_0 = torch.randn(batch_size, seq_len, out_dim).to(self.device)
        
        # Batch generation
        desc_start = batch_start + 1
        desc_end = min(batch_start + len(batch_chunks), total_chunks)
        generated = self.solve_euler(x_0, batch_audio_mert, batch_spec_feat, steps=self.args.steps,
                                   desc=f"Batch {desc_start}-{desc_end}/{total_chunks} - Generation")
        
        # Refinement (if enabled)
        if self.args.refine_step > 0:
            t_refine = 1.0 - self.args.refine_strength
            noise = torch.randn_like(generated)
            x_refine = (1 - t_refine) * noise + t_refine * generated
            refine_steps = int(self.args.steps * self.args.refine_strength)
            generated = self.solve_euler(x_refine, batch_audio_mert, batch_spec_feat, 
                                       t_start=t_refine, t_end=1.0, steps=refine_steps,
                                       desc=f"Batch {desc_start}-{desc_end}/{total_chunks} - Refinement")
        
        # Convert to numpy and return list
        batch_outputs = []
        for i in range(batch_size):
            gen_np = generated[i].cpu().numpy()
            batch_outputs.append(gen_np)
            
        return batch_outputs

    @torch.no_grad()
    def solve_euler(self, x, audio_mert, spec_feat, t_start=0.0, t_end=1.0, steps=50, desc="Solving ODE"):
        """Euler ODE Solver for Flow Matching"""
        dt = (t_end - t_start) / steps
        times = torch.linspace(t_start, t_end, steps + 1).to(self.device)
        
        # Progress bar for ODE steps
        for i in tqdm(range(steps), desc=desc, leave=False):
            t_curr = torch.ones(x.shape[0], device=self.device) * times[i]
            # [수정] audio_mert를 직접 전달 (model 내부에서 MERT 추출)
            v_pred = self.model(x, t_curr, audio_mert, spec_feat)
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
        
        # Calculate total number of chunks for progress bar
        chunk_positions = list(range(0, total_samples, stride_samples))
        total_chunks = len(chunk_positions)
        print(f"Processing {total_chunks} audio chunks...")
        
        # [최적화] Multi-GPU 배치 처리를 위한 배치 사이즈
        # GPU 메모리를 고려하여 동시 처리할 청크 수 결정
        # GPU가 1개이므로 num_gpus=1 -> min(2, 4) = 2개의 배치를 동시에 처리함
        batch_size = min(self.num_gpus * 2, 4)  
        print(f"Using batch size: {batch_size} for parallel processing")

        # Batch processing with progress bar
        for batch_start in tqdm(range(0, total_chunks, batch_size), desc="Processing audio batches"):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = []
            batch_info = []  # (start_idx, end_idx, original_len) 정보 저장
            
            # Prepare batch
            for i in range(batch_start, batch_end):
                start_idx = chunk_positions[i]
                end_idx = min(start_idx + chunk_samples, total_samples)
                wav_chunk = waveform[:, start_idx:end_idx]
                original_len = wav_chunk.shape[1]
                
                # 마지막 조각 패딩
                if original_len < chunk_samples:
                    wav_chunk = torch.nn.functional.pad(wav_chunk, (0, chunk_samples - original_len))
                
                batch_chunks.append(wav_chunk)
                batch_info.append((start_idx, end_idx, original_len))
            
            # Process batch
            if batch_chunks:
                batch_outputs = self.process_batch(batch_chunks, sr, batch_start, total_chunks)
                
                # Post-process each output in batch
                for j, gen_np in enumerate(batch_outputs):
                    start_idx, end_idx, original_len = batch_info[j]
                    
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
        
        # Process each drum channel with progress
        for i in tqdm(range(num_drums), desc="Processing drum channels", leave=False):
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