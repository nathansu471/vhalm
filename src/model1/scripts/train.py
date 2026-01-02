import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.model import ImageTextModel
from lib.dataset import create_dataloaders
from lib.eval import CaptionEvaluator
from lib.logger import TrainingLogger


class Trainer:
    """Training manager for the image-to-text model."""

    def __init__(self,
                 model: ImageTextModel,
                 train_loader,
                 val_loader,
                 test_loader,
                 optimizer,
                 scheduler,
                 device: torch.device,
                 config: Dict[str, Any],
                 scheduler_step_per_batch: bool = False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.device = device
        self.config = config

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')
        self.best_epoch = -1

        # Directories
        self.checkpoint_dir = config['checkpoint_dir']
        self.log_dir = config['log_dir']
        self.samples_dir = config['samples_dir']

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Enhanced Logger (create first)
        experiment_name = config.get('experiment_name', 'image2haiku')
        self.logger = TrainingLogger(self.log_dir, experiment_name)

        # Tensorboard - AutoDL compatible
        if os.path.exists('/root/tf-logs'):
            # AutoDL environment - use default path for automatic TensorBoard access
            self.tensorboard_dir = '/root/tf-logs'
            self.logger.info("🚀 AutoDL environment detected - using /root/tf-logs/ for automatic TensorBoard access")
        else:
            # Standard environment - use project directory
            self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')

        self.writer = SummaryWriter(self.tensorboard_dir)

        # Evaluator
        self.evaluator = CaptionEvaluator()

        # Save config
        config_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # CSV metrics logging
        self.metrics_csv_path = os.path.join(self.log_dir, 'training_metrics.csv')
        self._initialize_csv_logging()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']}")

        # Local config
        use_amp = bool(self.config.get('use_amp', False) and torch.cuda.is_available())
        grad_accum = max(1, int(self.config.get('gradient_accumulation_steps', 1)))
        scaler = torch.amp.GradScaler(device='cuda') if use_amp else None

        # Rate-limited OOM logging
        self.oom_count = 0
        max_oom_logs = int(self.config.get('max_oom_logs_per_epoch', 3))

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            image_features = batch['image_features'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            target_attention_mask = batch['target_attention_mask'].to(self.device)

            # Zero gradients at the start of an accumulation window
            if (batch_idx % grad_accum) == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional AMP
            try:
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(image_features=image_features,
                                             input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             target_ids=target_ids,
                                             target_attention_mask=target_attention_mask)
                else:
                    outputs = self.model(image_features=image_features,
                                         input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         target_ids=target_ids,
                                         target_attention_mask=target_attention_mask)

                loss = outputs['loss'] / grad_accum
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower():
                    self.oom_count += 1
                    if self.oom_count <= max_oom_logs:
                        self.logger.warning('CUDA OOM during forward; skipping batch after emptying cache')
                    elif self.oom_count == max_oom_logs + 1:
                        self.logger.info(f'Too many OOMs this epoch ({self.oom_count}); further OOM logs suppressed')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            # Backward pass (supports AMP and accumulation)
            try:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower():
                    self.oom_count += 1
                    if self.oom_count <= max_oom_logs:
                        self.logger.warning('CUDA OOM during backward; skipping step after emptying cache')
                    elif self.oom_count == max_oom_logs + 1:
                        self.logger.info(f'Too many OOMs this epoch ({self.oom_count}); further OOM logs suppressed')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                raise

            # When it's time to step (end of accumulation cycle or final batch)
            do_step = ((batch_idx + 1) % grad_accum == 0) or (batch_idx + 1 == num_batches)
            if do_step:
                # Unscale then clip grads if using scaler
                if scaler is not None:
                    scaler.unscale_(self.optimizer)

                if self.config.get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])

                try:
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                except RuntimeError as exc:
                    if 'out of memory' in str(exc).lower():
                        self.oom_count += 1
                        if self.oom_count <= max_oom_logs:
                            self.logger.warning('CUDA OOM during optimizer.step(); skipping update after emptying cache')
                        elif self.oom_count == max_oom_logs + 1:
                            self.logger.info(f'Too many OOMs this epoch ({self.oom_count}); further OOM logs suppressed')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # zero out grads after failed step
                        self.optimizer.zero_grad()
                        continue
                    raise

                if self.scheduler is not None and self.scheduler_step_per_batch:
                    self.scheduler.step()

                # Count an update step
                self.global_step += 1

            # Update metrics
            # loss was scaled for accumulation; multiply back for logging
            total_loss += (loss.item() * grad_accum)

            # Log to tensorboard
            true_loss = loss.item() * grad_accum
            if self.global_step and self.global_step % self.config['log_interval'] == 0:
                self.writer.add_scalar('train/loss', true_loss, self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{true_loss:.4f}",
                                      'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                                      'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"})

        # Update learning rate
        if self.scheduler is not None and not self.scheduler_step_per_batch:
            self.scheduler.step()

        # Summarize OOMs once per epoch to keep logs readable
        last_oom_count = getattr(self, 'oom_count', 0)
        if last_oom_count > 0:
            if last_oom_count > max_oom_logs:
                suppressed = last_oom_count - max_oom_logs
                self.logger.info(f'Suppressed {suppressed} additional OOM warnings this epoch (max_oom_logs_per_epoch={max_oom_logs}).')
            self.logger.info(f'Total OOM occurrences this epoch: {last_oom_count}')

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        num_batches = len(self.val_loader)

        use_amp = bool(self.config.get('use_amp', False) and torch.cuda.is_available())

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                image_features = batch['image_features'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                target_attention_mask = batch['target_attention_mask'].to(self.device)

                # Forward pass (use AMP for validation if enabled)
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(image_features=image_features, input_ids=input_ids, attention_mask=attention_mask, target_ids=target_ids, target_attention_mask=target_attention_mask)
                else:
                    outputs = self.model(image_features=image_features, input_ids=input_ids, attention_mask=attention_mask, target_ids=target_ids, target_attention_mask=target_attention_mask)

                loss = outputs['loss']
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}

    def evaluate_metrics(self, split: str = "val", max_samples: int = None) -> Dict[str, float]:
        """Evaluate the model with caption metrics."""
        dataloader = self.val_loader if split == "val" else self.test_loader

        samples_path = os.path.join(self.samples_dir, f"epoch_{self.current_epoch}_{split}_samples.json")

        metrics, samples = self.evaluator.evaluate_dataset(model=self.model, dataloader=dataloader, device=self.device, save_samples=True, samples_path=samples_path, max_samples=max_samples)

        # Log generation samples to detailed log
        if samples and split == "val":
            self.logger.log_generation_samples(samples, self.current_epoch)

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint.
        
        Only keeps latest_checkpoint.pth and best_checkpoint.pth to save disk space.
        Old epoch-specific checkpoints are automatically removed.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'model_config': {
                'clip_dim': self.config['clip_dim'],
                'hidden_dim': self.config['hidden_dim'],
                't5_model_name': self.config['t5_model_name'],
                'aggregation_method': self.config['aggregation_method'],
                'tokens_per_image': self.config.get('tokens_per_image', 4),
                'attention_layers': self.config.get('attention_layers', 1),
                'attention_heads': self.config.get('attention_heads', 4),
                'use_visual_mlp': self.config.get('use_visual_mlp', False),
                'use_lora': self.config['use_lora'],
                'lora_r': self.config['lora_r'],
                'lora_alpha': self.config['lora_alpha'],
                'lora_dropout': self.config['lora_dropout']
            }
        }

        # Save latest checkpoint (always overwrite)
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # Save best checkpoint if this is the best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved! {self.config['primary_metric']}: {self.best_metric:.4f}")

        # Remove old epoch-specific checkpoints to save space
        # Only keep latest_checkpoint.pth and best_checkpoint.pth
        if self.config.get('keep_only_best_and_latest', True):
            keep_names = {'latest_checkpoint.pth', 'best_checkpoint.pth', 'config.json'}
            for fname in os.listdir(self.checkpoint_dir):
                if fname.endswith('.pth') and fname not in keep_names:
                    old_path = os.path.join(self.checkpoint_dir, fname)
                    try:
                        os.remove(old_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove old checkpoint {fname}: {e}")

    def _initialize_csv_logging(self):
        """Initialize CSV file with headers for metrics logging."""
        import csv

        # Define all possible metrics we want to track
        self.csv_headers = ['epoch', 'train_loss', 'val_loss', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'CIDEr', 'METEOR', 'is_best', 'learning_rate']

        # Create CSV file with headers
        with open(self.metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)

        self.logger.info(f"📊 CSV metrics logging initialized: {self.metrics_csv_path}")

    def _log_metrics_to_csv(self, metrics: Dict[str, float], epoch: int, is_best: bool = False):
        """Log metrics to CSV file for easy analysis."""
        import csv

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0

        # Prepare row data with all metrics (use 0.0 for missing values)
        row_data = [
            epoch,
            metrics.get('train_loss', 0.0),
            metrics.get('val_loss', 0.0),
            metrics.get('BLEU-1', 0.0),
            metrics.get('BLEU-2', 0.0),
            metrics.get('BLEU-3', 0.0),
            metrics.get('BLEU-4', 0.0),
            metrics.get('CIDEr', 0.0),
            metrics.get('METEOR', 0.0), 1 if is_best else 0, current_lr
        ]

        # Write to CSV
        with open(self.metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

    def _log_metrics_to_tensorboard(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to TensorBoard with organized categories."""
        for metric, value in metrics.items():
            if metric == 'train_loss':
                self.writer.add_scalar('Loss/Train', value, epoch)
            elif metric == 'val_loss':
                self.writer.add_scalar('Loss/Validation', value, epoch)
            elif metric.startswith('BLEU'):
                self.writer.add_scalar(f'BLEU/{metric}', value, epoch)
            elif metric in ['CIDEr', 'METEOR']:
                self.writer.add_scalar(f'Caption_Metrics/{metric}', value, epoch)
            else:
                # Fallback for any other metrics
                self.writer.add_scalar(f'Other/{metric}', value, epoch)

    def _log_final_test_to_tensorboard(self, test_metrics: Dict[str, float]):
        """Log final test metrics to TensorBoard."""
        # Use a large epoch number to clearly separate from training curves
        final_epoch = self.current_epoch + 10

        for metric, value in test_metrics.items():
            if metric.startswith('BLEU'):
                self.writer.add_scalar(f'Final_Test/BLEU/{metric}', value, final_epoch)
            elif metric in ['CIDEr', 'METEOR']:
                self.writer.add_scalar(f'Final_Test/Caption_Metrics/{metric}', value, final_epoch)
            else:
                self.writer.add_scalar(f'Final_Test/Other/{metric}', value, final_epoch)

        # Also add a summary scalar for quick reference
        if 'CIDEr' in test_metrics:
            self.writer.add_scalar('Final_Test/Summary/CIDEr_Final', test_metrics['CIDEr'], final_epoch)
        if 'BLEU-4' in test_metrics:
            self.writer.add_scalar('Final_Test/Summary/BLEU-4_Final', test_metrics['BLEU-4'], final_epoch)

    def _log_csv_summary(self):
        """Generate and log CSV summary statistics."""
        import pandas as pd

        try:
            # Read the CSV file
            df = pd.read_csv(self.metrics_csv_path)

            # Generate summary statistics
            summary_stats = {}

            # Find best performance for key metrics
            if 'CIDEr' in df.columns and len(df) > 0:
                best_cider_idx = df['CIDEr'].idxmax()
                summary_stats['best_cider'] = {'epoch': int(df.loc[best_cider_idx, 'epoch']), 'value': float(df.loc[best_cider_idx, 'CIDEr']), 'BLEU-4': float(df.loc[best_cider_idx, 'BLEU-4']) if 'BLEU-4' in df.columns else 0.0}

            if 'BLEU-4' in df.columns and len(df) > 0:
                best_bleu4_idx = df['BLEU-4'].idxmax()
                summary_stats['best_bleu4'] = {'epoch': int(df.loc[best_bleu4_idx, 'epoch']), 'value': float(df.loc[best_bleu4_idx, 'BLEU-4']), 'CIDEr': float(df.loc[best_bleu4_idx, 'CIDEr']) if 'CIDEr' in df.columns else 0.0}

            # Final epoch performance
            if len(df) > 0:
                final_row = df.iloc[-1]
                summary_stats['final_epoch'] = {
                    'epoch': int(final_row['epoch']),
                    'CIDEr': float(final_row['CIDEr']) if 'CIDEr' in df.columns else 0.0,
                    'BLEU-4': float(final_row['BLEU-4']) if 'BLEU-4' in df.columns else 0.0,
                    'METEOR': float(final_row['METEOR']) if 'METEOR' in df.columns else 0.0
                }

            # Save summary to JSON
            summary_path = os.path.join(self.log_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)

            # Log summary
            self.logger.info("📊 TRAINING SUMMARY")
            self.logger.info("-" * 40)
            if 'best_cider' in summary_stats:
                best = summary_stats['best_cider']
                self.logger.info(f"Best CIDEr: {best['value']:.4f} (epoch {best['epoch']})")
            if 'best_bleu4' in summary_stats:
                best = summary_stats['best_bleu4']
                self.logger.info(f"Best BLEU-4: {best['value']:.4f} (epoch {best['epoch']})")
            if 'final_epoch' in summary_stats:
                final = summary_stats['final_epoch']
                self.logger.info(f"Final epoch {final['epoch']}: CIDEr={final['CIDEr']:.4f}, BLEU-4={final['BLEU-4']:.4f}")

        except Exception as e:
            self.logger.warning(f"Failed to generate CSV summary: {e}")

    def cleanup_checkpoints(self):
        """Final cleanup: Remove all checkpoints except best_checkpoint.pth.
        
        This is called at the end of training to save disk space.
        During training, save_checkpoint() already keeps only latest+best.
        This final cleanup removes even latest_checkpoint.pth.
        """
        keep_name = 'best_checkpoint.pth'
        removed = []

        for fname in os.listdir(self.checkpoint_dir):
            # Only remove .pth files
            if not fname.endswith('.pth'):
                continue
            if fname == keep_name:
                continue

            path = os.path.join(self.checkpoint_dir, fname)
            try:
                os.remove(path)
                removed.append(fname)
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {fname}: {e}")

        if removed:
            self.logger.info(f"Final cleanup: removed {len(removed)} checkpoint(s): {removed}")
            self.logger.info(f"Only {keep_name} is retained.")
        else:
            self.logger.info(f"Final cleanup: only {keep_name} exists, no files removed.")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint['best_epoch']

        print(f"Resumed from epoch {self.current_epoch}, best {self.config['primary_metric']}: {self.best_metric:.4f}")

    def train(self):
        """Main training loop."""
        # Log configuration and model info
        self.logger.log_config(self.config)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_model_info(total_params, trainable_params)

        self.logger.info(f"Starting training for {self.config['num_epochs']} epochs")
        self.logger.info(f"Training on {len(self.train_loader.dataset)} samples")
        self.logger.info(f"Validation on {len(self.val_loader.dataset)} samples")

        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch

            # Log epoch start
            self.logger.log_epoch_start(epoch, self.config['num_epochs'])

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Evaluate with caption metrics (less frequently)
            caption_metrics = {}
            if (epoch + 1) % self.config['eval_interval'] == 0 or epoch == self.config['num_epochs'] - 1:
                caption_metrics = self.evaluate_metrics(split="val", max_samples=self.config.get('eval_samples', None))

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics, **caption_metrics}

            # Check if best model
            is_best = False
            if self.config['primary_metric'] in all_metrics:
                current_metric = all_metrics[self.config['primary_metric']]
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    is_best = True

            # Log to tensorboard and CSV (with correct is_best flag)
            self._log_metrics_to_tensorboard(all_metrics, epoch)
            self._log_metrics_to_csv(all_metrics, epoch, is_best=is_best)

            # Save checkpoint
            self.save_checkpoint(all_metrics, is_best)

            # Log epoch end
            self.logger.log_epoch_end(epoch, all_metrics, is_best)

            if is_best:
                best_metrics_dict = {self.config['primary_metric']: self.best_metric}
                self.logger.log_best_metrics(best_metrics_dict, self.best_epoch + 1)

        # Final evaluation on test set
        self.logger.info("Running final evaluation on test set...")
        test_metrics = self.evaluate_metrics(split="test")

        # Log final test results
        self.logger.log_final_test(test_metrics)

        # Log final test metrics to tensorboard
        self._log_final_test_to_tensorboard(test_metrics)

        # Complete training
        self.logger.log_training_complete(self.global_step)

        # Log final CSV summary
        self._log_csv_summary()

        # Log TensorBoard info
        self.logger.info("=" * 60)
        self.logger.info("📊 VISUALIZATION & ANALYSIS")
        self.logger.info("=" * 60)
        self.logger.info(f"📈 CSV metrics: {self.metrics_csv_path}")
        self.logger.info(f"📊 TensorBoard: {self.tensorboard_dir}")
        self.logger.info("")

        if self.tensorboard_dir == '/root/tf-logs':
            # AutoDL environment
            self.logger.info("🚀 AutoDL TensorBoard Access:")
            self.logger.info("  ✅ Files saved to /root/tf-logs/ - TensorBoard auto-available!")
            self.logger.info("  🌐 Access via AutoDL Panel > TensorBoard tab")
        else:
            # Standard environment
            self.logger.info("📊 TensorBoard Access:")
            self.logger.info(f"  tensorboard --logdir {self.tensorboard_dir}")
            self.logger.info("  Open: http://localhost:6006")

        self.logger.info("")
        self.logger.info("📊 CSV Data Analysis:")
        self.logger.info(f"  import pandas as pd")
        self.logger.info(f"  df = pd.read_csv('{self.metrics_csv_path}')")
        self.logger.info("  df.plot(x='epoch', y=['BLEU-4', 'CIDEr'])")
        self.logger.info("=" * 60)  # Final cleanup: ensure only best checkpoint remains if configured
        if self.config.get('delete_checkpoints_after_training', True):
            try:
                self.cleanup_checkpoints()
            except Exception as e:
                self.logger.warning(f"Final checkpoint cleanup failed: {e}")

        self.writer.close()


def create_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        # Model parameters
        'clip_dim': 512,
        'hidden_dim': 512,
        't5_model_name': 't5-base',
        'aggregation_method': 'multi_token',  # 'mean' or 'multi_token'
        'tokens_per_image': 4,
        'attention_layers': 1,
        'attention_heads': 4,
        'use_visual_mlp': False,
        'features_path': None,
        'image_ids_path': None,
        'annotations_path': None,
        'use_tags': True,
        'tag_top_k': 4,
        'max_tags_per_sample': 12,
        'tag_cache_dir': None,
        'clip_text_model_name': 'openai/clip-vit-base-patch32',
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.1,

        # Training parameters
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'min_lr': 0.0,
        'weight_decay': 1e-5,
        'max_grad_norm': 1.0,
        'warmup_steps': 500,
        'scheduler_type': 'cosine',  # 'cosine', 'linear', or None

        # Evaluation parameters
        'eval_interval': 5,  # Evaluate every N epochs
        'eval_samples': 200,  # Max samples for evaluation (None for all)
        'primary_metric': 'BLEU-4',  # Metric for best model selection

        # Data parameters
        'max_length': 128,
        'num_workers': 4,
        # Training memory / perf options
        'use_amp': True,  # enable mixed-precision (autocast + GradScaler) when CUDA is available
        'gradient_accumulation_steps': 1,  # accumulate gradients this many steps before optimizer.step()
        'max_oom_logs_per_epoch': 3,  # cap OOM warnings per epoch to avoid log spam
        'split_ratio': [0.8, 0.1, 0.1],  # train, val, test
        'tag_generation_method': 'blip_noun_phrases',  # 'clip_zero_shot' or 'blip_noun_phrases'
        'blip_model_name': 'Salesforce/blip-image-captioning-base',
        'image_root': None,
        'tag_device': None,

        # Logging parameters
        'log_interval': 100,  # Log every N update steps

        # Checkpoint management
        'keep_only_best_and_latest': True,  # If True, only keep best_checkpoint.pth and latest_checkpoint.pth (saves space)
        'delete_checkpoints_after_training': True,  # If True, remove all checkpoints except best at the end

        # Paths (will be set based on timestamp)
        'checkpoint_dir': '',
        'log_dir': '',
        'samples_dir': ''
    }


def main():
    parser = argparse.ArgumentParser(description='Train Image-to-Text Model')
    parser.add_argument('--features_path', type=str, default=None, help='Path to image_features.npy (overrides config)')
    parser.add_argument('--image_ids_path', type=str, default=None, help='Path to image_ids.json (overrides config)')
    parser.add_argument('--annotations_path', type=str, default=None, help='Path to image_groups_with_1caption.json (overrides config)')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config JSON file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name for this experiment')
    parser.add_argument('--t5_model_name_or_path', type=str, default=None, help='T5 model name or path to local T5 model')
    parser.add_argument('--tag_generation_method', type=str, default=None, choices=['clip_zero_shot', 'blip_noun_phrases'], help='Override tag generation method defined in config')
    parser.add_argument('--tag_cache_dir', type=str, default=None, help='Override tag cache directory')
    parser.add_argument('--blip_model_name', type=str, default=None, help='Override BLIP model name for tag generation')
    parser.add_argument('--image_root', type=str, default=None, help='Override image root directory for BLIP tags')
    parser.add_argument('--tag_device', type=str, default=None, help='Force device for tag generator (e.g. cuda)')

    args = parser.parse_args()

    # Create config and stash CLI overrides to apply after file-based config loads
    config = create_config()
    cli_overrides = {}
    if args.t5_model_name_or_path is not None:
        cli_overrides['t5_model_name'] = args.t5_model_name_or_path
    if args.tag_generation_method is not None:
        cli_overrides['tag_generation_method'] = args.tag_generation_method
    if args.tag_cache_dir is not None:
        cli_overrides['tag_cache_dir'] = args.tag_cache_dir
    if args.blip_model_name is not None:
        cli_overrides['blip_model_name'] = args.blip_model_name
    if args.image_root is not None:
        cli_overrides['image_root'] = args.image_root
    if args.tag_device is not None:
        cli_overrides['tag_device'] = args.tag_device

    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        # Flatten nested config structure
        flattened_config = {}
        for section, params in custom_config.items():
            if isinstance(params, dict):
                flattened_config.update(params)
            else:
                flattened_config[section] = params
        config.update(flattened_config)

    # Apply CLI overrides last so they always win over config.json defaults
    config.update(cli_overrides)

    # Resolve tag cache directory relative to config file if needed
    tag_cache_dir = config.get('tag_cache_dir')
    if not tag_cache_dir:
        tag_cache_dir = '../../data/model1_data/clip_tags'
    if not os.path.isabs(tag_cache_dir):
        base_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()
        tag_cache_dir = os.path.abspath(os.path.join(base_dir, tag_cache_dir))
    config['tag_cache_dir'] = tag_cache_dir

    image_root = config.get('image_root')
    if image_root:
        if not os.path.isabs(image_root):
            base_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()
            image_root = os.path.abspath(os.path.join(base_dir, image_root))
        config['image_root'] = image_root

    if 'tag_generation_method' in config and isinstance(config['tag_generation_method'], str):
        config['tag_generation_method'] = config['tag_generation_method'].lower()
    else:
        config['tag_generation_method'] = 'blip_noun_phrases'
    if config['tag_generation_method'] not in {'clip_zero_shot', 'blip_noun_phrases'}:
        raise ValueError(f"Unsupported tag_generation_method: {config['tag_generation_method']}")
    if config['tag_generation_method'] == 'blip_noun_phrases' and not config.get('image_root'):
        raise ValueError('image_root must be provided when tag_generation_method is blip_noun_phrases.')

    # Set up experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.experiment_name:
        experiment_dir = f"{args.experiment_name}_{timestamp}"
        config['experiment_name'] = args.experiment_name
    else:
        experiment_dir = f"experiment_{timestamp}"
        config['experiment_name'] = 'image2haiku'

    base_output_dir = os.path.join(args.output_dir, experiment_dir)
    config['checkpoint_dir'] = os.path.join(base_output_dir, 'checkpoints')
    config['log_dir'] = os.path.join(base_output_dir, 'logs')
    config['samples_dir'] = os.path.join(base_output_dir, 'samples')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    # Resolve dataset paths (CLI overrides config)
    features_path = args.features_path or config.get('features_path')
    image_ids_path = args.image_ids_path or config.get('image_ids_path')
    annotations_path = args.annotations_path or config.get('annotations_path')

    if not all([features_path, image_ids_path, annotations_path]):
        raise ValueError('Dataset paths must be provided via command line or config (features_path, image_ids_path, annotations_path).')

    # Persist paths in config for logging/checkpointing
    config['features_path'] = features_path
    config['image_ids_path'] = image_ids_path
    config['annotations_path'] = annotations_path

    print("Dataset paths:")
    print(f"  features_path: {features_path}")
    print(f"  image_ids_path: {image_ids_path}")
    print(f"  annotations_path: {annotations_path}")
    print(f"Semantic tags enabled: {config.get('use_tags', False)}")
    if config.get('use_tags', False):
        cache_info = config.get('tag_cache_dir', 'n/a')
        print(f"  tag_cache_dir: {cache_info}")
        print(f"  tag_generation_method: {config.get('tag_generation_method', 'blip_noun_phrases')}")
        print(f"  tag_top_k: {config.get('tag_top_k', 4)} | max_tags_per_sample: {config.get('max_tags_per_sample', 12)}")
        if config.get('tag_generation_method') == 'blip_noun_phrases':
            print(f"  blip_model_name: {config.get('blip_model_name', 'Salesforce/blip-image-captioning-base')}")
            print(f"  image_root: {config.get('image_root', 'n/a')}")
        else:
            print(f"  clip_text_model_name: {config.get('clip_text_model_name', 'openai/clip-vit-base-patch32')}")

    train_loader, val_loader, test_loader = create_dataloaders(features_path=features_path,
                                                               image_ids_path=image_ids_path,
                                                               annotations_path=annotations_path,
                                                               batch_size=config['batch_size'],
                                                               num_workers=config['num_workers'],
                                                               tokenizer_name=config['t5_model_name'],
                                                               max_length=config['max_length'],
                                                               split_ratio=config['split_ratio'],
                                                               use_tags=config.get('use_tags', False),
                                                               tag_top_k=config.get('tag_top_k', 4),
                                                               max_tags_per_sample=config.get('max_tags_per_sample', 12),
                                                               clip_text_model_name=config.get('clip_text_model_name', 'openai/clip-vit-base-patch32'),
                                                               tag_cache_dir=config.get('tag_cache_dir', '../../data/model1_data/clip_tags'),
                                                               tag_generation_method=config.get('tag_generation_method', 'blip_noun_phrases'),
                                                               blip_model_name=config.get('blip_model_name', 'Salesforce/blip-image-captioning-base'),
                                                               image_root=config.get('image_root'),
                                                               tag_device=config.get('tag_device'))

    # Create model
    print("Creating model...")
    model = ImageTextModel(clip_dim=config['clip_dim'],
                           hidden_dim=config['hidden_dim'],
                           t5_model_name=config['t5_model_name'],
                           aggregation_method=config['aggregation_method'],
                           use_visual_mlp=config.get('use_visual_mlp', False),
                           use_lora=config['use_lora'],
                           lora_r=config['lora_r'],
                           lora_alpha=config['lora_alpha'],
                           lora_dropout=config['lora_dropout'],
                           label_smoothing=config.get('label_smoothing', 0.0),
                           tokens_per_image=config.get('tokens_per_image', 4),
                           attention_layers=config.get('attention_layers', 1),
                           attention_heads=config.get('attention_heads', 4)).to(device)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Create scheduler (supports optional warmup和按步更新)
    scheduler = None
    scheduler_step_per_batch = False
    scheduler_type = (config.get('scheduler_type') or '').lower()
    warmup_steps_cfg = max(0, int(config.get('warmup_steps', 0)))
    num_epochs = max(1, int(config.get('num_epochs', 1)))
    base_lr = float(config.get('learning_rate', 0.0))
    min_lr = max(0.0, float(config.get('min_lr', 0.0)))

    steps_per_epoch = len(train_loader)
    total_training_steps = steps_per_epoch * num_epochs

    if total_training_steps > 0 and scheduler_type in {'cosine', 'linear'}:
        warmup_steps = min(warmup_steps_cfg, total_training_steps)
        warmup_scheduler = None
        schedulers = []

        if warmup_steps > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            schedulers.append(warmup_scheduler)

        remaining_steps = max(0, total_training_steps - warmup_steps)

        if scheduler_type == 'cosine' and remaining_steps > 0:
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_steps,
                eta_min=min(min_lr, base_lr) if base_lr > 0 else min_lr,
            )
            schedulers.append(cosine_scheduler)
        elif scheduler_type == 'linear' and remaining_steps > 0:
            if base_lr > 0:
                end_factor = min(1.0, max(0.0, min_lr / base_lr))
            else:
                end_factor = 0.0
            decay_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=remaining_steps,
            )
            schedulers.append(decay_scheduler)

        if schedulers:
            if len(schedulers) == 1:
                scheduler = schedulers[0]
            else:
                # SequentialLR expects milestones for all but last scheduler
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=schedulers,
                    milestones=[warmup_steps] if warmup_steps > 0 else [],
                )
            scheduler_step_per_batch = True

    # Create trainer
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      device=device,
                      config=config,
                      scheduler_step_per_batch=scheduler_step_per_batch)

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
