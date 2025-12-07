import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import time


class TrainingLogger:
    """Enhanced logging system for training with detailed logs and summary."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.start_time = time.time()

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup file paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.detailed_log_path = os.path.join(log_dir, f'{experiment_name}_{timestamp}_detailed.log')
        self.summary_path = os.path.join(log_dir, f'{experiment_name}_{timestamp}_summary.json')

        # Setup detailed logging
        self.logger = logging.getLogger(f'trainer_{experiment_name}')
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(self.detailed_log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Summary data
        self.summary_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'training_progress': [],
            'best_metrics': {},
            'final_test_results': {},
            'training_duration': 0,
            'total_epochs': 0,
            'total_steps': 0
        }

    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.summary_data['config'] = config
        self.logger.info("=" * 60)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 60)
        for key, value in config.items():
            self.logger.info(f"{key:25s}: {value}")
        self.logger.info("=" * 60)

    def log_model_info(self, total_params: int, trainable_params: int):
        """Log model parameter information."""
        self.logger.info("=" * 60)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 60)
        self.logger.info(f"{'Total Parameters':25s}: {total_params:,}")
        self.logger.info(f"{'Trainable Parameters':25s}: {trainable_params:,}")
        self.logger.info(f"{'Trainable Ratio':25s}: {trainable_params/total_params*100:.2f}%")
        self.logger.info("=" * 60)

        self.summary_data['model_info'] = {'total_params': total_params, 'trainable_params': trainable_params, 'trainable_ratio': trainable_params / total_params}

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.logger.info(f"\n{'='*20} EPOCH {epoch+1}/{total_epochs} {'='*20}")

    def log_epoch_end(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Log epoch end with metrics."""
        self.logger.info(f"Epoch {epoch+1} Results:")
        self.logger.info("-" * 40)

        for metric, value in metrics.items():
            self.logger.info(f"{metric:15s}: {value:.6f}")

        if is_best:
            self.logger.info("🏆 NEW BEST MODEL!")

        # Add to summary
        epoch_summary = {'epoch': epoch + 1, 'metrics': metrics, 'is_best': is_best, 'timestamp': datetime.now().isoformat()}
        self.summary_data['training_progress'].append(epoch_summary)

        # Update totals
        self.summary_data['total_epochs'] = epoch + 1

        self.logger.info("-" * 40)

    def log_best_metrics(self, best_metrics: Dict[str, float], best_epoch: int):
        """Log best achieved metrics."""
        self.summary_data['best_metrics'] = {'metrics': best_metrics, 'epoch': best_epoch}

        self.logger.info("🎯 BEST METRICS ACHIEVED:")
        for metric, value in best_metrics.items():
            self.logger.info(f"{metric:15s}: {value:.6f} (epoch {best_epoch})")

    def log_final_test(self, test_metrics: Dict[str, float]):
        """Log final test results."""
        self.summary_data['final_test_results'] = test_metrics

        self.logger.info("=" * 60)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info("=" * 60)
        for metric, value in test_metrics.items():
            self.logger.info(f"{metric:15s}: {value:.6f}")
        self.logger.info("=" * 60)

    def log_training_complete(self, total_steps: int):
        """Log training completion."""
        end_time = time.time()
        duration = end_time - self.start_time

        self.summary_data['training_duration'] = duration
        self.summary_data['total_steps'] = total_steps
        self.summary_data['end_time'] = datetime.now().isoformat()

        # Format duration
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED!")
        self.logger.info("=" * 60)
        self.logger.info(f"{'Total Duration':20s}: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        self.logger.info(f"{'Total Epochs':20s}: {self.summary_data['total_epochs']}")
        self.logger.info(f"{'Total Steps':20s}: {total_steps}")
        self.logger.info(f"{'Detailed Log':20s}: {self.detailed_log_path}")
        self.logger.info(f"{'Summary':20s}: {self.summary_path}")
        self.logger.info("=" * 60)

        # Save summary
        self.save_summary()

    def log_generation_samples(self, samples: List[Dict], epoch: int):
        """Log some generation samples for inspection."""
        self.logger.info(f"\n📝 SAMPLE GENERATIONS (Epoch {epoch+1}):")
        self.logger.info("-" * 50)

        for i, sample in enumerate(samples[:3]):  # Show first 3 samples
            self.logger.info(f"Sample {i+1}:")
            self.logger.info(f"  Reference: {sample.get('reference', 'N/A')}")
            self.logger.info(f"  Prediction: {sample.get('prediction', 'N/A')}")
            if 'metrics' in sample:
                bleu4 = sample['metrics'].get('BLEU-4', 0)
                meteor = sample['metrics'].get('METEOR', 0)
                self.logger.info(f"  BLEU-4: {bleu4:.4f}, METEOR: {meteor:.4f}")
            self.logger.info("")

    def save_summary(self):
        """Save training summary to JSON."""
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary_data, f, indent=2, ensure_ascii=False)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
