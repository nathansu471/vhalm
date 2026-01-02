import json
import os
import random
from typing import List, Dict, Tuple, Optional, Set

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, CLIPTextModel, CLIPTokenizer,
                          BlipForConditionalGeneration, BlipProcessor)
from nltk.corpus import wordnet as wn

# In-memory cache to avoid repeated large loads
_ZERO_SHOT_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


class ZeroShotTagGenerator:
    """Generates tags via CLIP zero-shot similarity against its full vocabulary."""

    def __init__(self,
                 cache_dir: str,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 top_k: int = 4,
                 max_tags: int = 12,
                 device: Optional[str] = None):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.clip_model_name = clip_model_name
        self.top_k = max(1, top_k)
        self.max_tags = max_tags
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        safe_model_name = clip_model_name.replace('/', '_')
        self.cache_path = os.path.join(self.cache_dir, f"clip_tags_{safe_model_name}.npz")

        self._ensure_cache()
        cached = _ZERO_SHOT_CACHE[self.cache_path]
        self.tag_strings: List[str] = cached['tags']  # type: ignore[assignment]
        self.text_embeddings: torch.Tensor = cached['embeddings']

    def __call__(self, *, image_features: np.ndarray, image_ids: Optional[List[str]] = None) -> List[str]:
        if image_features.ndim == 1:
            features = torch.from_numpy(image_features).float().unsqueeze(0)
        else:
            features = torch.from_numpy(image_features).float()

        with torch.no_grad():
            features = F.normalize(features, dim=-1)
            sims = features @ self.text_embeddings.T
            topk = sims.topk(k=min(self.top_k, sims.size(-1)), dim=-1).indices

        collected: List[str] = []
        for row in topk:
            for idx in row.tolist():
                tag = self.tag_strings[idx]
                if tag not in collected:
                    collected.append(tag)
                    if len(collected) >= self.max_tags:
                        return collected
        return collected

    def _ensure_cache(self) -> None:
        if self.cache_path in _ZERO_SHOT_CACHE:
            return
        if os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=True)
            tags = data['tags'].tolist()
            embeddings = torch.from_numpy(data['embeddings']).float()
            _ZERO_SHOT_CACHE[self.cache_path] = {'tags': tags, 'embeddings': embeddings}
            return

        tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        try:
            text_model = CLIPTextModel.from_pretrained(self.clip_model_name, use_safetensors=True).to(self.device)
        except ValueError as exc:
            raise RuntimeError(
                "Failed to load CLIP text model with safetensors. "
                "Please install torch>=2.6 or ensure safetensor weights are available."
            ) from exc
        text_model.eval()

        vocab = tokenizer.get_vocab()
        id_to_token = {idx: token for token, idx in vocab.items()}
        validity_cache: Dict[str, bool] = {}

        def is_common_noun(word: str) -> bool:
            cached = validity_cache.get(word)
            if cached is not None:
                return cached

            if not word.isascii() or not word.isalpha():
                validity_cache[word] = False
                return False

            if len(word) < 3 or len(word) > 16:
                validity_cache[word] = False
                return False

            try:
                lemma = wn.morphy(word, wn.NOUN) or word
                is_valid = bool(wn.synsets(lemma, pos=wn.NOUN))
            except LookupError as exc:
                raise RuntimeError("WordNet data not found. Please ensure nltk 'wordnet' corpus is downloaded before precomputing tags.") from exc
            validity_cache[word] = is_valid
            return is_valid

        prompts: List[str] = []
        tag_strings: List[str] = []
        seen: Set[str] = set()

        for idx in sorted(id_to_token.keys()):
            decoded = tokenizer.decode([idx]).strip()
            cleaned = decoded.lower().strip()
            if not cleaned:
                continue
            if any(ch in cleaned for ch in '\n\t'):
                continue
            if len(cleaned) > 20:
                continue
            if not any(ch.isalpha() for ch in cleaned):
                continue
            if cleaned in seen:
                continue
            if not is_common_noun(cleaned):
                continue

            prompts.append(f"a photo of {cleaned}")
            tag_strings.append(cleaned)
            seen.add(cleaned)

        embeddings_list: List[torch.Tensor] = []
        batch_size = 256
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start:start + batch_size]
                tokens = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = text_model(**tokens)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    text_embeds = outputs.pooler_output
                else:
                    text_embeds = outputs.last_hidden_state[:, 0, :]
                text_embeds = F.normalize(text_embeds, dim=-1)
                embeddings_list.append(text_embeds.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        np.savez(self.cache_path, tags=np.array(tag_strings, dtype=object), embeddings=embeddings.numpy())
        _ZERO_SHOT_CACHE[self.cache_path] = {'tags': tag_strings, 'embeddings': embeddings}

        # Free large models explicitly
        del text_model
        del tokenizer



class BLIPNounPhraseTagGenerator:
    """Generates tags by running BLIP captioning and extracting noun phrases."""

    def __init__(self,
                 cache_dir: str,
                 image_root: str,
                 blip_model_name: str = "Salesforce/blip-image-captioning-base",
                 max_tags: int = 12,
                 device: Optional[str] = None,
                 allow_generation: bool = True):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.image_root = os.path.abspath(image_root)
        if not os.path.isdir(self.image_root):
            raise ValueError(f"Image root directory not found: {self.image_root}")

        self.blip_model_name = blip_model_name
        self.max_tags = max(1, max_tags)
        self.allow_generation = allow_generation

        resolved_device = device or "cpu"
        try:
            self.device = torch.device(resolved_device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid tag device specified for BLIP: {device}") from exc

        safe_model_name = blip_model_name.replace('/', '_')
        self.store_dir = os.path.join(self.cache_dir, f"blip_np_{safe_model_name}")
        os.makedirs(self.store_dir, exist_ok=True)

        for resource, msg in [
            ("tokenizers/punkt", "nltk 'punkt' tokenizer not found. Please download it via nltk.download('punkt')."),
            ("taggers/averaged_perceptron_tagger", "nltk 'averaged_perceptron_tagger' not found. Download via nltk.download('averaged_perceptron_tagger').")
        ]:
            try:
                nltk.data.find(resource)
            except LookupError as exc:
                raise RuntimeError(msg) from exc

        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None

        if self.allow_generation:
            self._ensure_model_loaded()

    def __call__(self, *, image_features: np.ndarray, image_ids: Optional[List[str]]) -> List[str]:
        del image_features  # Unused but kept for interface compatibility
        if not image_ids:
            return []

        total_limit = max(0, self.max_tags)
        if total_limit == 0:
            return []

        num_images = len(image_ids)
        if num_images == 0:
            return []

        base_quota = total_limit // num_images
        quotas = [base_quota] * num_images
        remainder = total_limit - base_quota * num_images
        if remainder > 0:
            quotas[-1] += remainder  # keep leftover slots for the last image to honor 1/3,1/3,rest

        combined: List[str] = []
        seen = set()

        for idx, image_id in enumerate(image_ids):
            quota = quotas[idx] if idx < len(quotas) else 0
            if quota <= 0:
                continue

            tags = self.ensure_cached(image_id)
            picked = 0
            for tag in tags:
                if tag in seen:
                    continue
                combined.append(tag)
                seen.add(tag)
                picked += 1
                if len(combined) >= total_limit:
                    return combined
                if picked >= quota:
                    break

        return combined

    def ensure_cached(self, image_id: str) -> List[str]:
        cache_path = self._cache_path_for_image(image_id)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cached_tags = data.get('tags', [])
            if isinstance(cached_tags, list):
                return cached_tags[:self.max_tags]

        image_path = os.path.join(self.image_root, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found for BLIP tagging: {image_path}")

        if not self.allow_generation:
            raise RuntimeError(
                "检测到 BLIP 标签缓存缺失。请先运行 scripts/precompute_tags.py 预生成所有标签，"
                "或者在构造 BLIPNounPhraseTagGenerator 时允许在线生成。"
            )

        self._ensure_model_loaded()

        with Image.open(image_path) as img:
            image = img.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_length=50)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        tags = self.extract_noun_tags(caption)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'caption': caption, 'tags': tags}, f, ensure_ascii=False)

        return tags[:self.max_tags]

    @staticmethod
    def extract_noun_tags(caption: str) -> List[str]:
        allowed_tags = {"NN", "NNS", "NNP", "NNPS"}
        nouns: List[str] = []

        for sentence in nltk.sent_tokenize(caption):
            tokens = nltk.word_tokenize(sentence)
            if not tokens:
                continue
            tagged = nltk.pos_tag(tokens)

            current: List[str] = []
            for token, pos in tagged:
                lowered = token.lower()
                cleaned = lowered.strip(".,;:!?\"'")
                if pos in allowed_tags and any(ch.isalpha() for ch in cleaned):
                    current.append(cleaned)
                else:
                    if current:
                        phrase = ' '.join(current)
                        if phrase not in nouns:
                            nouns.append(phrase)
                        current = []
            if current:
                phrase = ' '.join(current)
                if phrase not in nouns:
                    nouns.append(phrase)

        return nouns


    def _cache_path_for_image(self, image_id: str) -> str:
        basename = os.path.basename(image_id)
        safe_name = basename.replace('/', '_')
        return os.path.join(self.store_dir, f"{safe_name}.json")

    def _ensure_model_loaded(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        self.processor = BlipProcessor.from_pretrained(self.blip_model_name)
        try:
            model = BlipForConditionalGeneration.from_pretrained(
                self.blip_model_name,
                use_safetensors=True
            )
        except ValueError as exc:
            raise RuntimeError(
                "Failed to load BLIP model with safetensors. Upgrade torch>=2.6 or ensure safetensor weights are available."
            ) from exc

        self.model = model.to(self.device)
        self.model.eval()



def collate_fn(batch):
    """Custom collate function to properly handle image_ids lists."""
    # Default collate for tensor data
    image_features = torch.stack([item['image_features'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    target_attention_mask = torch.stack([item['target_attention_mask'] for item in batch])
    group_ids = torch.tensor([item['group_id'] for item in batch])

    # Keep these as lists to avoid tensor conversion issues
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_ids'] for item in batch]  # List of lists
    tags = [item.get('tags', []) for item in batch]

    return {'image_features': image_features,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids,
            'target_attention_mask': target_attention_mask,
            'caption': captions,
            'group_id': group_ids,
            'image_ids': image_ids,
            'tags': tags}


class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs using pre-extracted CLIP features."""

    def __init__(self,
                 features_path: str,
                 image_ids_path: str,
                 annotations_path: str,
                 tokenizer_name: str = "t5-base",
                 max_length: int = 128,
                 split_type: str = "train",
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 use_tags: bool = False,
                 tag_top_k: int = 4,
                 max_tags_per_sample: int = 12,
                 clip_text_model_name: str = "openai/clip-vit-base-patch32",
                 tag_cache_dir: str = "../../data/model1_data/clip_tags",
                 tag_generation_method: str = "blip_noun_phrases",
                 blip_model_name: str = "Salesforce/blip-image-captioning-base",
                 image_root: Optional[str] = None,
                 tag_device: Optional[str] = None):
        """
        Args:
            features_path: Path to image_features.npy file
            image_ids_path: Path to image_ids.json file
            annotations_path: Path to image_groups_with_1caption.json file
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for text
            split_type: 'train', 'val', or 'test'
            split_ratio: (train, val, test) split ratios
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.use_tags = use_tags
        self.tag_top_k = tag_top_k
        self.max_tags_per_sample = max_tags_per_sample

        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        self.features = np.load(features_path)  # Shape: (3000, 512)
        with open(image_ids_path, 'r') as f:
            self.image_ids = json.load(f)
        with open(annotations_path, 'r') as f:
            self.groups = json.load(f)

        # Create image_id to index mapping
        self.image_id_to_idx = {img_id: idx for idx, img_id in enumerate(self.image_ids)}

        # Split data
        self.data_split = self._create_split(split_type, split_ratio)

        self.tag_generator = None
        if self.use_tags:
            method = (tag_generation_method or "clip_zero_shot").lower()
            if method == "clip_zero_shot":
                self.tag_generator = ZeroShotTagGenerator(cache_dir=tag_cache_dir,
                                                          clip_model_name=clip_text_model_name,
                                                          top_k=tag_top_k,
                                                          max_tags=max_tags_per_sample,
                                                          device=tag_device)
            elif method == "blip_noun_phrases":
                if not image_root:
                    raise ValueError("image_root must be provided when using BLIP-based tag generation.")
                self.tag_generator = BLIPNounPhraseTagGenerator(cache_dir=tag_cache_dir,
                                                                image_root=image_root,
                                                                blip_model_name=blip_model_name,
                                                                max_tags=max_tags_per_sample,
                                                                device=tag_device,
                                                                allow_generation=False)
            else:
                raise ValueError(f"Unsupported tag_generation_method: {tag_generation_method}")
            self.tag_generation_method = method
        else:
            self.tag_generation_method = "none"

    def _create_split(self, split_type: str, split_ratio: Tuple[float, float, float]) -> List[Dict]:
        """Create train/val/test split."""
        total_groups = len(self.groups)
        train_size = int(total_groups * split_ratio[0])
        val_size = int(total_groups * split_ratio[1])

        # Set random seed for reproducible splits
        random.seed(42)
        shuffled_groups = self.groups.copy()
        random.shuffle(shuffled_groups)

        if split_type == "train":
            return shuffled_groups[:train_size]
        elif split_type == "val":
            return shuffled_groups[train_size:train_size + val_size]
        else:  # test
            return shuffled_groups[train_size + val_size:]

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        group = self.data_split[idx]

        # Get image features for the 3 images in this group
        image_indices = []
        for img_id in group['image_ids']:
            if img_id in self.image_id_to_idx:
                image_indices.append(self.image_id_to_idx[img_id])
            else:
                raise ValueError(f"Image ID {img_id} not found in image_ids.json")

        # Extract features for the 3 images: shape (3, 512)
        image_features = self.features[image_indices]

        tags: List[str] = []
        if self.tag_generator is not None:
            tags = self.tag_generator(image_features=image_features, image_ids=group['image_ids'])
        tag_prefix = f"tags: {', '.join(tags)}. " if tags else ""

        # Tokenize caption
        caption = group['merged_caption']

        # For T5, we need to add prefix for generation tasks
        # Add language specification to force English generation
        input_text = f"{tag_prefix}generate English caption: "
        target_text = caption

        # Tokenize input and target
        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # Replace padding token ids in target with -100 to ignore them in loss computation
        # This is the standard practice in Transformers library
        target_ids = target_encoding['input_ids'].squeeze(0).clone()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            'image_features': torch.tensor(image_features, dtype=torch.float32),  # (3, 512)
            'input_ids': input_encoding['input_ids'].squeeze(0),  # (max_length,)
            'attention_mask': input_encoding['attention_mask'].squeeze(0),  # (max_length,)
            'target_ids': target_ids,  # (max_length,) with padding replaced by -100
            'target_attention_mask': target_encoding['attention_mask'].squeeze(0),  # (max_length,)
            'caption': caption,
            'group_id': group['group_id'],
            'image_ids': group['image_ids'],
            'tags': tags
        }


def create_dataloaders(features_path: str,
                       image_ids_path: str,
                       annotations_path: str,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       tokenizer_name: str = "t5-base",
                       max_length: int = 128,
                       split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                       use_tags: bool = False,
                       tag_top_k: int = 4,
                       max_tags_per_sample: int = 12,
                       clip_text_model_name: str = "openai/clip-vit-base-patch32",
                       tag_cache_dir: str = "../../data/model1_data/clip_tags",
                       tag_generation_method: str = "clip_zero_shot",
                       blip_model_name: str = "Salesforce/blip-image-captioning-base",
                       image_root: Optional[str] = None,
                       tag_device: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    common_kwargs = dict(features_path=features_path,
                         image_ids_path=image_ids_path,
                         annotations_path=annotations_path,
                         tokenizer_name=tokenizer_name,
                         max_length=max_length,
                         split_ratio=split_ratio,
                         use_tags=use_tags,
                         tag_top_k=tag_top_k,
                         max_tags_per_sample=max_tags_per_sample,
                         clip_text_model_name=clip_text_model_name,
                         tag_cache_dir=tag_cache_dir,
                         tag_generation_method=tag_generation_method,
                         blip_model_name=blip_model_name,
                         image_root=image_root,
                         tag_device=tag_device)

    train_dataset = ImageCaptionDataset(split_type="train", **common_kwargs)

    val_dataset = ImageCaptionDataset(split_type="val", **common_kwargs)

    test_dataset = ImageCaptionDataset(split_type="test", **common_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    features_path = "../../data/model1_data/image_features.npy"
    image_ids_path = "../../data/model1_data/image_ids.json"
    annotations_path = "../../data/model1_data/merged_captions_async_augmented.json"

    train_loader, val_loader, test_loader = create_dataloaders(features_path=features_path,
                                                               image_ids_path=image_ids_path,
                                                               annotations_path=annotations_path,
                                                               batch_size=4,
                                                               use_tags=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test a batch
    for batch in train_loader:
        print("Batch shapes:")
        print(f"Image features: {batch['image_features'].shape}")
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Target IDs: {batch['target_ids'].shape}")
        print(f"Sample caption: {batch['caption'][0]}")
        break
