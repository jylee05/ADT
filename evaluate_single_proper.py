#!/usr/bin/env python3
"""
Model Evaluation Script for N2N Flow Matching Drum Transcription
Evaluates checkpoint performance on validation data
- Modified to use ONLY Physical GPU 1
- Fixed Velocity Scaling (Raw -> 1~127)
- Fixed mir_eval logic (Manual Velocity Matching Implementation)
"""

import os

# [설정] 무조건 물리적 GPU 1번만 사용하도록 강제
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pretty_midi
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.signal import find_peaks
from tqdm import tqdm
import json
from pathlib import Path
import mir_eval 

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset

# Drum class names for reporting
DRUM_NAMES = ["Kick", "Snare", "HH", "Toms", "Crash", "Ride", "Bell"]

class ModelEvaluator:
    def __init__(self, args):
        self.args = args 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        
        print(f"🔧 Loading model from: {args.ckpt_path}")
        print(f"   Target Device: {self.device} (Mapped to Physical GPU 1)")
        
        # Load model
        self.model = FlowMatchingTransformer(self.config).to(self.device)
        self.loss_fn = AnnealedPseudoHuberLoss(self.model, self.config).to(self.device)
        self.load_checkpoint(args.ckpt_path)
        
        # GPU 개수가 1개로 제한되었으므로 DataParallel은 건너뜀
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.loss_fn.model = self.model
        else:
            print("🔧 Running in Single GPU Mode")
        
        self.model.eval()
        
        # Load validation dataset
        print("📁 Loading validation dataset...")
        self.val_dataset = EGMDDataset(is_train=False)  
        
        # [빠른 평가] 샘플 수 제한
        if args.quick:
            total_samples = min(200, len(self.val_dataset))
            print(f"⚡ Quick evaluation mode: using {total_samples} samples only")
            indices = np.random.choice(len(self.val_dataset), total_samples, replace=False)
            subset_dataset = torch.utils.data.Subset(self.val_dataset, indices)
            self.val_loader = torch.utils.data.DataLoader(
                subset_dataset,
                batch_size=16, 
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            # Full evaluation
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, 
                batch_size=min(8, len(self.val_dataset)), 
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        dataset_size = total_samples if args.quick else len(self.val_dataset)
        print(f"✅ Evaluation setup complete!")

    def load_checkpoint(self, path):
        # weights_only=False로 설정하여 사용자 정의 객체 로드 허용
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict
            
        self.model.load_state_dict(new_state_dict)
        
        self.epoch = checkpoint.get('epoch', 'Unknown')
        self.train_loss = checkpoint.get('loss', 'Unknown')
        print(f"   - Checkpoint epoch: {self.epoch}")
        print(f"   - Training loss: {self.train_loss}")

    @torch.no_grad()
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n🎯 Starting Model Evaluation...")
        
        # Storage for metrics
        all_losses = []
        frame_preds, frame_targets = [], []
        onset_preds, onset_targets = [], []  
        velocity_preds, velocity_targets = [], [] 
        velocity_errors = []  
        per_drum_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(self.config.DRUM_CHANNELS)}
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch_idx, (audio_mert, spec, target_grid) in enumerate(progress_bar):
            audio_mert = audio_mert.to(self.device)
            spec = spec.to(self.device)
            target_grid = target_grid.to(self.device)
            
            # Loss Calculation
            loss = self.loss_fn(audio_mert, spec, target_grid, progress=0.5)
            all_losses.append(loss.item())
            
            # Generate predictions
            eval_steps = self.args.eval_steps  
            predictions = self.loss_fn.sample(audio_mert, spec, steps=eval_steps)
            
            # Process each sample in batch
            for i in range(predictions.shape[0]):
                pred_sample = predictions[i].cpu().numpy()  
                target_sample = target_grid[i].cpu().numpy() 
                
                pred_onset = pred_sample[:, :self.config.DRUM_CHANNELS]
                pred_velocity = pred_sample[:, self.config.DRUM_CHANNELS:]
                target_onset = target_sample[:, :self.config.DRUM_CHANNELS]
                target_velocity = target_sample[:, self.config.DRUM_CHANNELS:]
                
                # Frame-level evaluation (Binary)
                pred_onset_binary = (pred_onset > 0.0).astype(int)
                target_onset_binary = (target_onset > -0.5).astype(int)
                
                frame_preds.append(pred_onset_binary.flatten())
                frame_targets.append(target_onset_binary.flatten())
                
                # mir_eval용 list
                pred_notes_onset = []
                pred_notes_velocity = []
                target_notes_onset = []
                target_notes_velocity = []
                
                # [Velocity 변환 함수] Raw(-1~1) -> MIDI(1~127)
                def denorm_vel(val):
                    return np.clip(((val + 1) / 2) * 127, 1, 127)

                for drum_idx in range(self.config.DRUM_CHANNELS):
                    # Peak detection
                    pred_peaks, _ = find_peaks(pred_onset[:, drum_idx], height=0.0, distance=3)
                    target_peaks = np.where(target_onset[:, drum_idx] > -0.5)[0]
                    
                    # [기존 유지] per-drum metrics (30ms tolerance)
                    tp = 0
                    matched_targets = set()
                    for pred_peak in pred_peaks:
                        distances = np.abs(target_peaks - pred_peak)
                        if len(distances) > 0:
                            min_idx = np.argmin(distances)
                            if distances[min_idx] <= 3 and min_idx not in matched_targets:
                                tp += 1
                                matched_targets.add(min_idx)
                    
                    fp = len(pred_peaks) - tp
                    fn = len(target_peaks) - tp
                    per_drum_metrics[drum_idx]['tp'] += tp
                    per_drum_metrics[drum_idx]['fp'] += fp
                    per_drum_metrics[drum_idx]['fn'] += fn
                    
                    frame_to_sec = 1.0 / self.config.FPS
                    
                    # [Prediction Note 생성] Velocity 변환 적용
                    for peak_frame in pred_peaks:
                        onset_time = peak_frame * frame_to_sec
                        raw_vel = pred_velocity[peak_frame, drum_idx]
                        real_vel = denorm_vel(raw_vel)  # 변환!
                        
                        pred_notes_onset.append([onset_time, drum_idx + 1])
                        pred_notes_velocity.append([onset_time, drum_idx + 1, real_vel])
                    
                    # [Target Note 생성] Velocity 변환 적용
                    for peak_frame in target_peaks:
                        onset_time = peak_frame * frame_to_sec
                        raw_vel = target_velocity[peak_frame, drum_idx]
                        real_vel = denorm_vel(raw_vel)  # 변환!
                        
                        target_notes_onset.append([onset_time, drum_idx + 1])
                        target_notes_velocity.append([onset_time, drum_idx + 1, real_vel])
                
                onset_preds.append(pred_notes_onset)
                onset_targets.append(target_notes_onset)
                velocity_preds.append(pred_notes_velocity)
                velocity_targets.append(target_notes_velocity)
                
                # [Velocity MAE/MSE 계산]
                active_mask = target_onset_binary > 0
                if np.any(active_mask):
                    active_pred_vel = pred_velocity[active_mask]
                    active_target_vel = target_velocity[active_mask]
                    
                    real_pred = denorm_vel(active_pred_vel)
                    real_target = denorm_vel(active_target_vel)
                    
                    vel_errors = np.abs(real_pred - real_target)
                    velocity_errors.extend(vel_errors.flatten())
            
            avg_loss = np.mean(all_losses)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # mir_eval 평가 준비
        print("\n📊 Computing mir_eval metrics (Core Function)...")
        
        all_pred_onset_intervals, all_pred_onset_pitches = [], []
        all_target_onset_intervals, all_target_onset_pitches = [], []
        all_pred_velocity_intervals, all_pred_velocity_pitches, all_pred_velocities = [], [], []
        all_target_velocity_intervals, all_target_velocity_pitches, all_target_velocities = [], [], []
        
        time_offset = 0.0
        for sample_idx, (pred_onset_notes, target_onset_notes, pred_vel_notes, target_vel_notes) in enumerate(
            zip(onset_preds, onset_targets, velocity_preds, velocity_targets)
        ):
            # Onset Only Data
            if pred_onset_notes:
                pred_notes = np.array(pred_onset_notes)
                pred_times = pred_notes[:, 0] + time_offset
                pred_pitches = pred_notes[:, 1].astype(int)
                pred_intervals = np.column_stack([pred_times, pred_times + 0.1])
                all_pred_onset_intervals.append(pred_intervals)
                all_pred_onset_pitches.append(pred_pitches)
            
            if target_onset_notes:
                target_notes = np.array(target_onset_notes)
                target_times = target_notes[:, 0] + time_offset
                target_pitches = target_notes[:, 1].astype(int)
                target_intervals = np.column_stack([target_times, target_times + 0.1])
                all_target_onset_intervals.append(target_intervals)
                all_target_onset_pitches.append(target_pitches)
            
            # Velocity Data (이미 denorm_vel 적용됨)
            if pred_vel_notes:
                pred_vel_array = np.array(pred_vel_notes)
                pred_times = pred_vel_array[:, 0] + time_offset
                pred_pitches = pred_vel_array[:, 1].astype(int)
                pred_vels = pred_vel_array[:, 2] # 1~127 scale
                pred_intervals = np.column_stack([pred_times, pred_times + 0.1])
                all_pred_velocity_intervals.append(pred_intervals)
                all_pred_velocity_pitches.append(pred_pitches)
                all_pred_velocities.append(pred_vels)
            
            if target_vel_notes:
                target_vel_array = np.array(target_vel_notes)
                target_times = target_vel_array[:, 0] + time_offset
                target_pitches = target_vel_array[:, 1].astype(int)
                target_vels = target_vel_array[:, 2] # 1~127 scale
                target_intervals = np.column_stack([target_times, target_times + 0.1])
                all_target_velocity_intervals.append(target_intervals)
                all_target_velocity_pitches.append(target_pitches)
                all_target_velocities.append(target_vels)
            
            time_offset += 6.0
        
        # Combine arrays
        final_pred_onset_intervals = np.vstack(all_pred_onset_intervals) if all_pred_onset_intervals else np.empty((0, 2))
        final_pred_onset_pitches = np.concatenate(all_pred_onset_pitches) if all_pred_onset_pitches else np.array([], dtype=int)
        final_target_onset_intervals = np.vstack(all_target_onset_intervals) if all_target_onset_intervals else np.empty((0, 2))
        final_target_onset_pitches = np.concatenate(all_target_onset_pitches) if all_target_onset_pitches else np.array([], dtype=int)
        
        final_pred_velocity_intervals = np.vstack(all_pred_velocity_intervals) if all_pred_velocity_intervals else np.empty((0, 2))
        final_pred_velocity_pitches = np.concatenate(all_pred_velocity_pitches) if all_pred_velocity_pitches else np.array([], dtype=int)
        final_pred_velocities = np.concatenate(all_pred_velocities) if all_pred_velocities else np.array([])
        final_target_velocity_intervals = np.vstack(all_target_velocity_intervals) if all_target_velocity_intervals else np.empty((0, 2))
        final_target_velocity_pitches = np.concatenate(all_target_velocity_pitches) if all_target_velocity_pitches else np.array([], dtype=int)
        final_target_velocities = np.concatenate(all_target_velocities) if all_target_velocities else np.array([])
        
        # 데이터 정렬 (mir_eval 안정성 확보)
        if len(final_pred_onset_intervals) > 0:
            sort_idx = np.argsort(final_pred_onset_intervals[:, 0])
            final_pred_onset_intervals = final_pred_onset_intervals[sort_idx]
            final_pred_onset_pitches = final_pred_onset_pitches[sort_idx]
            
        if len(final_target_onset_intervals) > 0:
            sort_idx = np.argsort(final_target_onset_intervals[:, 0])
            final_target_onset_intervals = final_target_onset_intervals[sort_idx]
            final_target_onset_pitches = final_target_onset_pitches[sort_idx]

        if len(final_pred_velocity_intervals) > 0:
            sort_idx = np.argsort(final_pred_velocity_intervals[:, 0])
            final_pred_velocity_intervals = final_pred_velocity_intervals[sort_idx]
            final_pred_velocity_pitches = final_pred_velocity_pitches[sort_idx]
            final_pred_velocities = final_pred_velocities[sort_idx]

        if len(final_target_velocity_intervals) > 0:
            sort_idx = np.argsort(final_target_velocity_intervals[:, 0])
            final_target_velocity_intervals = final_target_velocity_intervals[sort_idx]
            final_target_velocity_pitches = final_target_velocity_pitches[sort_idx]
            final_target_velocities = final_target_velocities[sort_idx]

        # =========================================================================
        # 1. Onset Only Evaluation (velocity 상관 없음)
        # =========================================================================
        try:
            # offset_ratio=None으로 설정하면 offset은 무시함 (Onset만 평가)
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=final_target_onset_intervals,
                ref_pitches=final_target_onset_pitches,
                est_intervals=final_pred_onset_intervals,
                est_pitches=final_pred_onset_pitches,
                onset_tolerance=0.05,
                pitch_tolerance=0.0,
                offset_ratio=None 
            )
            mir_onset_precision, mir_onset_recall, mir_onset_f1 = p, r, f
        except Exception as e:
            print(f"❌ Error in mir_eval onset evaluation: {e}")
            mir_onset_f1 = mir_onset_precision = mir_onset_recall = 0.0

        # =========================================================================
        # 2. Onset + Velocity Evaluation (수동 매칭 구현)
        # =========================================================================
        try:
            # Step 1: 먼저 Onset 기준으로 매칭되는 노트들의 인덱스를 구함
            # match_notes 함수를 사용 (offset_ratio=None으로 offset 무시)
            matches = mir_eval.transcription.match_notes(
                ref_intervals=final_target_velocity_intervals,
                ref_pitches=final_target_velocity_pitches,
                est_intervals=final_pred_velocity_intervals,
                est_pitches=final_pred_velocity_pitches,
                onset_tolerance=0.05,
                pitch_tolerance=0.0,
                offset_ratio=None
            )
            
            # Step 2: 매칭된 노트들 중에서 Velocity 차이가 허용범위 이내인 것만 TP로 인정
            velocity_tp = 0
            
            # Velocity 허용 오차: 127 스케일의 10% = 12.7
            VEL_TOLERANCE = 12.7 
            
            for ref_idx, est_idx in matches:
                ref_vel = final_target_velocities[ref_idx]
                est_vel = final_pred_velocities[est_idx]
                
                if abs(ref_vel - est_vel) <= VEL_TOLERANCE:
                    velocity_tp += 1
            
            # Step 3: Precision, Recall, F1 계산
            n_ref = len(final_target_velocities)
            n_est = len(final_pred_velocities)
            
            mir_velocity_precision = velocity_tp / n_est if n_est > 0 else 0.0
            mir_velocity_recall = velocity_tp / n_ref if n_ref > 0 else 0.0
            
            if (mir_velocity_precision + mir_velocity_recall) > 0:
                mir_velocity_f1 = 2 * mir_velocity_precision * mir_velocity_recall / (mir_velocity_precision + mir_velocity_recall)
            else:
                mir_velocity_f1 = 0.0
                
        except Exception as e:
            print(f"❌ Error in manual velocity evaluation: {e}")
            mir_velocity_f1 = mir_velocity_precision = mir_velocity_recall = 0.0
        
        # Calculate final metrics
        results = self.calculate_metrics(
            all_losses, frame_preds, frame_targets, per_drum_metrics, velocity_errors,
            mir_onset_f1, mir_onset_precision, mir_onset_recall,
            mir_velocity_f1, mir_velocity_precision, mir_velocity_recall
        )
        
        return results

    def calculate_metrics(self, losses, frame_preds, frame_targets, per_drum_metrics, velocity_errors,
                         mir_onset_f1, mir_onset_precision, mir_onset_recall,
                         mir_velocity_f1, mir_velocity_precision, mir_velocity_recall):
        """Calculate and format evaluation metrics"""
        results = {}
        
        # Loss
        results['loss'] = {
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
            'min': float(np.min(losses)),
            'max': float(np.max(losses))
        }
        
        # Frame-level
        frame_preds_all = np.concatenate(frame_preds)
        frame_targets_all = np.concatenate(frame_targets)
        frame_precision, frame_recall, frame_f1, _ = precision_recall_fscore_support(
            frame_targets_all, frame_preds_all, average='binary', zero_division=0
        )
        frame_accuracy = accuracy_score(frame_targets_all, frame_preds_all)
        results['frame_level'] = {
            'precision': float(frame_precision),
            'recall': float(frame_recall),
            'f1_score': float(frame_f1),
            'accuracy': float(frame_accuracy)
        }
        
        # Note-level
        overall_tp = sum(metrics['tp'] for metrics in per_drum_metrics.values())
        overall_fp = sum(metrics['fp'] for metrics in per_drum_metrics.values())
        overall_fn = sum(metrics['fn'] for metrics in per_drum_metrics.values())
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        results['note_level'] = {
            'overall_30ms': { 
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1_score': float(overall_f1),
                'tp': int(overall_tp),
                'fp': int(overall_fp),
                'fn': int(overall_fn),
                'method': 'custom_30ms'
            },
            'mir_eval_onset': { 
                'precision': float(mir_onset_precision),
                'recall': float(mir_onset_recall),
                'f1_score': float(mir_onset_f1),
                'method': 'mir_eval_50ms_onset_only'
            },
            'mir_eval_velocity': { 
                'precision': float(mir_velocity_precision),
                'recall': float(mir_velocity_recall),
                'f1_score': float(mir_velocity_f1),
                'method': 'mir_eval_50ms_onset_velocity_manual'
            }
        }
        
        # Per-drum metrics
        per_drum_results = {}
        for drum_idx, metrics in per_drum_metrics.items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            per_drum_results[DRUM_NAMES[drum_idx]] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
            }
        results['per_drum'] = per_drum_results
        
        # Velocity metrics
        if velocity_errors:
            results['velocity'] = {
                'mae': float(np.mean(velocity_errors)),
                'mse': float(np.mean(np.array(velocity_errors) ** 2)),
                'std': float(np.std(velocity_errors)),
                'samples': len(velocity_errors)
            }
        else:
            results['velocity'] = {'mae': 0.0, 'mse': 0.0, 'std': 0.0, 'samples': 0}
        
        results['model_info'] = {
            'checkpoint_epoch': self.epoch,
            'training_loss': self.train_loss,
            'eval_steps': self.args.eval_steps
        }
        return results

    def print_results(self, results):
        """Print formatted evaluation results"""
        print("\n" + "="*60)
        print(f"🎯 EVALUATION RESULTS - Epoch {results['model_info']['checkpoint_epoch']}")
        print("="*60)
        
        loss_info = results['loss']
        print(f"\n📊 Loss Metrics:")
        print(f"   Mean Loss: {loss_info['mean']:.4f} (±{loss_info['std']:.4f})")
        
        frame = results['frame_level']
        print(f"\n🎵 Frame-Level Performance:")
        print(f"   F1-Score:  {frame['f1_score']:.3f} (Acc: {frame['accuracy']:.3f})")
        
        note_30ms = results['note_level']['overall_30ms']
        note_onset = results['note_level']['mir_eval_onset']
        note_velocity = results['note_level']['mir_eval_velocity']
        
        print(f"\n🎼 Note-Level Performance:")
        print(f"   [기존 30ms] F1: {note_30ms['f1_score']:.3f}")
        print(f"   [논문 50ms Onset] F1: {note_onset['f1_score']:.3f}")
        print(f"   [논문 50ms Onset+Vel] F1: {note_velocity['f1_score']:.3f}")
        
        vel = results['velocity']
        if vel['samples'] > 0:
            print(f"\n🔊 Velocity Estimation (Scale: 1-127):")
            print(f"   MAE: {vel['mae']:.3f}")
            print(f"   RMSE: {np.sqrt(vel['mse']):.3f}")
        
        print(f"\n🥁 Per-Drum F1-Scores:")
        for drum_name, metrics in results['per_drum'].items():
            print(f"   {drum_name:6}: {metrics['f1_score']:.3f}")
            
        paper_f1 = results['note_level']['mir_eval_velocity']['f1_score']
        print(f"\n🎯 Final Assessment (Onset+Vel): F1 {paper_f1:.3f}")

    def save_results(self, results):
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            ckpt_name = Path(self.args.ckpt_path).stem
            output_file = os.path.join(self.args.output_dir, f"eval_{ckpt_name}.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

    def run(self):
        results = self.evaluate_model()
        self.print_results(results)
        if self.args.output_dir:
            self.save_results(results)
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--quick', action='store_true')
    
    args = parser.parse_args()
    if args.quick:
        print("⚡ Quick evaluation mode enabled")
    
    evaluator = ModelEvaluator(args)
    evaluator.run()