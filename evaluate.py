#!/usr/bin/env python3
"""
Model Evaluation Script for N2N Flow Matching Drum Transcription
Evaluates checkpoint performance on validation data
"""

import os
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

from src.config import Config
from src.model import FlowMatchingTransformer, AnnealedPseudoHuberLoss
from src.dataset import EGMDDataset

# Drum class names for reporting
DRUM_NAMES = ["Kick", "Snare", "HH", "Toms", "Crash", "Ride", "Bell"]

class ModelEvaluator:
    def __init__(self, args):
        self.args = args  # args를 instance variable로 저장
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        
        print(f"🔧 Loading model from: {args.ckpt_path}")
        
        # Load model
        self.model = FlowMatchingTransformer(self.config).to(self.device)
        self.loss_fn = AnnealedPseudoHuberLoss(self.model, self.config).to(self.device)
        self.load_checkpoint(args.ckpt_path)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            self.loss_fn.model = self.model
        
        self.model.eval()
        
        # Load validation dataset
        print("📁 Loading validation dataset...")
        self.val_dataset = EGMDDataset(is_train=False)  # Use validation mode
        
        # [빠른 평가] 샘플 수 제한
        if args.quick:
            # Quick evaluation: 200 samples only (5분 내 완료)
            total_samples = min(200, len(self.val_dataset))
            print(f"⚡ Quick evaluation mode: using {total_samples} samples only")
            # Random subset for better representation
            indices = np.random.choice(len(self.val_dataset), total_samples, replace=False)
            subset_dataset = torch.utils.data.Subset(self.val_dataset, indices)
            self.val_loader = torch.utils.data.DataLoader(
                subset_dataset,
                batch_size=16,  # Larger batch for speed
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
        print(f"   - Model: {self.config.HIDDEN_DIM}D, {self.config.N_LAYERS} layers")
        print(f"   - Validation samples: {dataset_size}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle DataParallel state dict
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict
            
        self.model.load_state_dict(new_state_dict)
        
        # Extract training info if available
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
        velocity_errors = []
        per_drum_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(self.config.DRUM_CHANNELS)}
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch_idx, (audio_mert, spec, target_grid) in enumerate(progress_bar):
            audio_mert = audio_mert.to(self.device)
            spec = spec.to(self.device)
            target_grid = target_grid.to(self.device)
            
            # Calculate loss (using progress=0.5 for mid-training evaluation)
            loss = self.loss_fn(audio_mert, spec, target_grid, progress=0.5)
            all_losses.append(loss.item())
            
            # Generate predictions using sampling
            eval_steps = 5 if self.args.quick else self.args.eval_steps  # Quick: 5 steps only
            predictions = self.loss_fn.sample(audio_mert, spec, steps=eval_steps)
            
            # Process each sample in batch
            for i in range(predictions.shape[0]):
                pred_sample = predictions[i].cpu().numpy()  # (T, 14)
                target_sample = target_grid[i].cpu().numpy()  # (T, 14)
                
                # Split into onset and velocity
                pred_onset = pred_sample[:, :self.config.DRUM_CHANNELS]  # (T, 7)
                pred_velocity = pred_sample[:, self.config.DRUM_CHANNELS:]  # (T, 7)
                target_onset = target_sample[:, :self.config.DRUM_CHANNELS]
                target_velocity = target_sample[:, self.config.DRUM_CHANNELS:]
                
                # Frame-level evaluation (threshold-based)
                pred_onset_binary = (pred_onset > 0.0).astype(int)
                target_onset_binary = (target_onset > -0.5).astype(int)  # -1 = silence, >-0.5 = active
                
                frame_preds.append(pred_onset_binary.flatten())
                frame_targets.append(target_onset_binary.flatten())
                
                # Note-level evaluation (peak detection)
                for drum_idx in range(self.config.DRUM_CHANNELS):
                    # Peak detection for predictions
                    pred_peaks, _ = find_peaks(pred_onset[:, drum_idx], height=0.0, distance=3)
                    
                    # Ground truth peaks (where target onset > -0.5)
                    target_peaks = np.where(target_onset[:, drum_idx] > -0.5)[0]
                    
                    # Match predictions to targets (tolerance = 3 frames)
                    tp = 0
                    matched_targets = set()
                    
                    for pred_peak in pred_peaks:
                        # Find closest target within tolerance
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
                
                # Velocity evaluation (only for active frames)
                active_mask = target_onset_binary > 0
                if np.any(active_mask):
                    active_pred_vel = pred_velocity[active_mask]
                    active_target_vel = target_velocity[active_mask]
                    vel_errors = np.abs(active_pred_vel - active_target_vel)
                    velocity_errors.extend(vel_errors.flatten())
            
            # Update progress
            avg_loss = np.mean(all_losses)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Calculate final metrics
        results = self.calculate_metrics(
            all_losses, frame_preds, frame_targets, 
            per_drum_metrics, velocity_errors
        )
        
        return results

    def calculate_metrics(self, losses, frame_preds, frame_targets, per_drum_metrics, velocity_errors):
        """Calculate and format evaluation metrics"""
        results = {}
        
        # Loss metrics
        results['loss'] = {
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
            'min': float(np.min(losses)),
            'max': float(np.max(losses))
        }
        
        # Frame-level metrics
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
        
        # Note-level metrics (per drum and overall)
        overall_tp = sum(metrics['tp'] for metrics in per_drum_metrics.values())
        overall_fp = sum(metrics['fp'] for metrics in per_drum_metrics.values())
        overall_fn = sum(metrics['fn'] for metrics in per_drum_metrics.values())
        
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        results['note_level'] = {
            'overall': {
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1_score': float(overall_f1),
                'tp': int(overall_tp),
                'fp': int(overall_fp),
                'fn': int(overall_fn)
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
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
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
        
        # Training info
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
        
        # Loss
        loss_info = results['loss']
        print(f"\n📊 Loss Metrics:")
        print(f"   Mean Loss: {loss_info['mean']:.4f} (±{loss_info['std']:.4f})")
        print(f"   Range: {loss_info['min']:.4f} - {loss_info['max']:.4f}")
        
        # Frame-level
        frame = results['frame_level']
        print(f"\n🎵 Frame-Level Performance:")
        print(f"   Precision: {frame['precision']:.3f}")
        print(f"   Recall:    {frame['recall']:.3f}")
        print(f"   F1-Score:  {frame['f1_score']:.3f}")
        print(f"   Accuracy:  {frame['accuracy']:.3f}")
        
        # Note-level
        note = results['note_level']['overall']
        print(f"\n🎼 Note-Level Performance (Onset Detection):")
        print(f"   Precision: {note['precision']:.3f}")
        print(f"   Recall:    {note['recall']:.3f}")
        print(f"   F1-Score:  {note['f1_score']:.3f}")
        print(f"   Events: {note['tp']} TP, {note['fp']} FP, {note['fn']} FN")
        
        # Velocity
        vel = results['velocity']
        if vel['samples'] > 0:
            print(f"\n🔊 Velocity Estimation:")
            print(f"   MAE: {vel['mae']:.3f}")
            print(f"   RMSE: {np.sqrt(vel['mse']):.3f}")
            print(f"   Active frames: {vel['samples']}")
        
        # Per-drum breakdown
        print(f"\n🥁 Per-Drum Performance (F1-Scores):")
        for drum_name, metrics in results['per_drum'].items():
            print(f"   {drum_name:6}: {metrics['f1_score']:.3f} ({metrics['tp']:2}TP/{metrics['fp']:2}FP/{metrics['fn']:2}FN)")
        
        # Performance assessment
        overall_f1 = note['f1_score']
        print(f"\n🎯 Performance Assessment:")
        if overall_f1 >= 0.8:
            print("   🔥 EXCELLENT - Near paper-level performance!")
        elif overall_f1 >= 0.6:
            print("   ✅ GOOD - Strong drum transcription capability")
        elif overall_f1 >= 0.4:
            print("   📈 MODERATE - Learning in progress, needs more training")
        elif overall_f1 >= 0.2:
            print("   ⚠️  BASIC - Early learning stage")
        else:
            print("   🔄 EARLY - Model still learning fundamentals")

    def save_results(self, results):
        """Save results to JSON file"""
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            # Create filename with epoch info
            ckpt_name = Path(self.args.ckpt_path).stem
            output_file = os.path.join(self.args.output_dir, f"eval_{ckpt_name}.json")
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")

    def run(self):
        """Run complete evaluation"""
        results = self.evaluate_model()
        self.print_results(results)
        
        if self.args.output_dir:
            self.save_results(results)
        
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate N2N Flow Matching Model')
    parser.add_argument('--ckpt_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--eval_steps', type=int, default=10,
                       help='Number of sampling steps for inference')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation mode (200 samples, 5 steps, ~5min)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("⚡ Quick evaluation mode enabled (200 samples, 5 steps)")
    
    evaluator = ModelEvaluator(args)
    evaluator.run()