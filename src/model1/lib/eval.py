import torch
import json
import os
from typing import Dict, List, Tuple, Any
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
import nltk
import numpy as np
from collections import defaultdict
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, Exception):
    print("Downloading NLTK punkt data...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, Exception):
    print("Downloading NLTK punkt_tab data...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except (LookupError, Exception):
    print("Downloading NLTK wordnet data...")
    nltk.download('wordnet', quiet=True)


def preprocess_text(text: str) -> str:
    """Preprocess text for evaluation."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text or not text.strip():
        return []
    try:
        return nltk.word_tokenize(preprocess_text(text))
    except Exception as e:
        print(f"Warning: Tokenization failed for text '{text}': {e}")
        return preprocess_text(text).split()


class CaptionEvaluator:
    """Evaluator for image captioning using BLEU, METEOR, and CIDEr metrics."""

    def __init__(self):
        self.cider_scorer = Cider()
        self.smoothing_function = SmoothingFunction().method4

    def compute_bleu(self, references: List[List[str]], hypothesis: List[str]) -> Dict[str, float]:
        """Compute BLEU scores."""
        scores = {}

        # BLEU-1 to BLEU-4
        for i in range(1, 5):
            weights = [1.0 / i] * i + [0.0] * (4 - i)
            score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=self.smoothing_function)
            scores[f'BLEU-{i}'] = score

        return scores

    def compute_meteor(self, reference: str, hypothesis: str) -> float:
        """Compute METEOR score."""
        try:
            # METEOR expects tokenized text (list of words)
            ref_tokens = tokenize_text(reference)
            hyp_tokens = tokenize_text(hypothesis)

            # Handle empty tokens
            if not ref_tokens or not hyp_tokens:
                return 0.0

            return meteor_score([ref_tokens], hyp_tokens)
        except Exception as e:
            print(f"Warning: METEOR computation failed: {e}")
            return 0.0

    def compute_cider(self, gts: Dict, res: Dict) -> float:
        """Compute CIDEr score using pycocoevalcap format."""
        score, _ = self.cider_scorer.compute_score(gts, res)
        return score

    def evaluate_batch(self, predictions: List[str], references: List[str], group_ids: List[int] = None) -> Dict[str, float]:
        """
        Evaluate a batch of predictions against references.
        
        Args:
            predictions: List of predicted captions
            references: List of reference captions
            group_ids: Optional list of group IDs
        
        Returns:
            Dictionary of evaluation metrics
        """
        if len(predictions) != len(references):
            raise ValueError(f"Mismatch in number of predictions ({len(predictions)}) and references ({len(references)})")

        bleu_scores = defaultdict(list)
        meteor_scores = []

        # Prepare data for CIDEr (requires specific format)
        gts = {}
        res = {}

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Tokenize
            pred_tokens = tokenize_text(pred)
            ref_tokens = tokenize_text(ref)

            # BLEU scores
            bleu_result = self.compute_bleu([ref_tokens], pred_tokens)
            for key, score in bleu_result.items():
                bleu_scores[key].append(score)

            # METEOR score
            meteor_score_val = self.compute_meteor(ref, pred)
            meteor_scores.append(meteor_score_val)

            # Prepare for CIDEr
            img_id = group_ids[i] if group_ids else i
            gts[img_id] = [ref]  # CIDEr expects list of references
            res[img_id] = [pred]  # CIDEr expects list of predictions

        # Compute CIDEr score
        try:
            cider_score = self.compute_cider(gts, res)
        except Exception as e:
            print(f"Warning: CIDEr computation failed: {e}")
            cider_score = 0.0

        # Aggregate results
        results = {'METEOR': np.mean(meteor_scores), 'CIDEr': cider_score}

        # Add BLEU scores
        for key, scores in bleu_scores.items():
            results[key] = np.mean(scores)

        return results

    def evaluate_dataset(self, model, dataloader, device: torch.device, save_samples: bool = True, samples_path: str = "outputs/samples/evaluation_results.json", max_samples: int = None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Evaluate model on entire dataset.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on
            save_samples: Whether to save sample results
            samples_path: Path to save samples
            max_samples: Maximum number of samples to evaluate (None for all)
        
        Returns:
            Tuple of (metrics_dict, samples_list)
        """
        model.eval()

        all_predictions = []
        all_references = []
        all_group_ids = []
        all_image_ids = []
        samples = []

        total_batches = len(dataloader)
        if max_samples:
            max_batches = (max_samples + dataloader.batch_size - 1) // dataloader.batch_size
            total_batches = min(total_batches, max_batches)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and batch_idx >= max_batches:
                    break

                # Move to device
                image_features = batch['image_features'].to(device)

                # 构造与训练阶段一致的提示模板
                tags_batch = batch.get('tags', [[] for _ in range(len(image_features))])
                prompt_texts: List[str] = []
                for tags in tags_batch:
                    if tags:
                        prompt_texts.append(f"tags: {', '.join(tags)}. generate English caption: ")
                    else:
                        prompt_texts.append("generate English caption: ")

                # Generate predictions with diversity parameters
                generation_results = model.generate(
                    image_features=image_features,
                    max_length=64,  #  更短的最大长度
                    num_beams=4,
                    temperature=0.7,  #  添加一些随机性
                    do_sample=True,  #  启用采样
                    top_p=0.9,  #  nucleus sampling
                    prompt_text=prompt_texts
                    # repetition_penalty和no_repeat_ngram_size已在model.generate()中设置
                )

                predictions = generation_results['generated_texts']
                references = batch['caption']
                group_ids = batch['group_id'].tolist()
                image_ids = batch['image_ids']  # This is a list of lists (batch_size, 3)

                # Store for metrics computation
                all_predictions.extend(predictions)
                all_references.extend(references)
                all_group_ids.extend(group_ids)
                all_image_ids.extend(image_ids)

                # Store samples
                if save_samples:
                    for i in range(len(predictions)):
                        # Ensure we have valid indices for all lists
                        if i < len(group_ids) and i < len(image_ids) and i < len(references):
                            sample = {'group_id': group_ids[i],
                                      'image_ids': image_ids[i],
                                      'reference': references[i],
                                      'prediction': predictions[i],
                                      'tags': tags_batch[i] if i < len(tags_batch) else []}
                            samples.append(sample)

                # Print progress
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    print(f"Evaluated {batch_idx + 1}/{total_batches} batches")

        # Compute metrics
        print("Computing evaluation metrics...")
        metrics = self.evaluate_batch(all_predictions, all_references, all_group_ids)

        # Add sample metrics to each sample
        if save_samples and samples:
            evaluator_single = CaptionEvaluator()
            for sample in samples:
                sample_metrics = evaluator_single.evaluate_batch([sample['prediction']], [sample['reference']], [sample['group_id']])
                sample['metrics'] = sample_metrics

        # Save samples if requested
        if save_samples and samples:
            os.makedirs(os.path.dirname(samples_path), exist_ok=True)
            with open(samples_path, 'w', encoding='utf-8') as f:
                json.dump({'overall_metrics': metrics, 'num_samples': len(samples), 'samples': samples}, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(samples)} samples to {samples_path}")

        return metrics, samples


def evaluate_model_checkpoint(model_path: str,
                              features_path: str,
                              image_ids_path: str,
                              annotations_path: str,
                              device: torch.device = None,
                              batch_size: int = 16,
                              split_type: str = "test",
                              save_samples: bool = True,
                              samples_dir: str = "outputs/samples",
                              max_samples: int = None) -> Dict[str, float]:
    """
    Evaluate a saved model checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint
        features_path: Path to image features
        image_ids_path: Path to image IDs
        annotations_path: Path to annotations
        device: Device to run on
        batch_size: Batch size for evaluation
        split_type: 'test', 'val', or 'train'
        save_samples: Whether to save sample results
        samples_dir: Directory to save samples
        max_samples: Maximum samples to evaluate
    
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import here to avoid circular imports
    from .model import ImageTextModel
    from .dataset import create_dataloaders

    print(f"Loading model from {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    model = ImageTextModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Create dataloader
    _, val_loader, test_loader = create_dataloaders(
        features_path=features_path,
        image_ids_path=image_ids_path,
        annotations_path=annotations_path,
        batch_size=batch_size,
        num_workers=0  # Avoid multiprocessing issues during evaluation
    )

    if split_type == "val":
        dataloader = val_loader
    else:
        dataloader = test_loader

    # Create evaluator and run evaluation
    evaluator = CaptionEvaluator()

    if save_samples:
        checkpoint_name = os.path.splitext(os.path.basename(model_path))[0]
        samples_path = os.path.join(samples_dir, f"{checkpoint_name}_{split_type}_results.json")
    else:
        samples_path = None

    metrics, samples = evaluator.evaluate_dataset(model=model, dataloader=dataloader, device=device, save_samples=save_samples, samples_path=samples_path, max_samples=max_samples)

    # Print results
    print(f"\nEvaluation Results on {split_type} set:")
    print("=" * 40)
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    return metrics


if __name__ == "__main__":
    # Test the evaluator
    evaluator = CaptionEvaluator()

    # Test data
    predictions = ["a cat sitting on a table", "two dogs playing in the park", "a car driving down the street"]

    references = ["a cat is sitting on the table", "two dogs are playing in a park", "a red car driving on the road"]

    group_ids = [1, 2, 3]

    # Compute metrics
    results = evaluator.evaluate_batch(predictions, references, group_ids)

    print("Test Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
