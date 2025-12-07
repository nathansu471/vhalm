import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any, List


class ImageFeatureAggregator(nn.Module):
    """Creates image tokens and mixes cross-image information."""

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 512,
                 aggregation_method: str = "multi_token",
                 tokens_per_image: int = 4,
                 attention_layers: int = 1,
                 attention_heads: int = 4,
                 use_visual_mlp: bool = False):
        """
        Args:
            input_dim: Dimension of input CLIP features
            hidden_dim: Dimension of hidden representation
            aggregation_method: 'mean', 'attention' (alias for multi_token), or 'multi_token'
            tokens_per_image: Number of tokens generated per image
            attention_layers: Transformer encoder depth for cross-image mixing
            attention_heads: Number of attention heads per encoder layer
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Treat legacy "attention" value as the new multi-token path
        self.aggregation_method = "multi_token" if aggregation_method == "attention" else aggregation_method
        self.tokens_per_image = tokens_per_image
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.use_visual_mlp = use_visual_mlp

        if aggregation_method == "mean":
            self.visual_proj = self._build_projector(input_dim, hidden_dim, use_visual_mlp)
        elif self.aggregation_method == "multi_token":
            if tokens_per_image < 1:
                raise ValueError("tokens_per_image must be >= 1")
            if attention_layers < 1:
                raise ValueError("attention_layers must be >= 1")
            if use_visual_mlp:
                self.visual_proj = self._build_projector(input_dim, hidden_dim, True)
                token_input_dim = hidden_dim
            else:
                self.visual_proj = None
                token_input_dim = input_dim
            self.token_proj = nn.Linear(token_input_dim, hidden_dim * tokens_per_image)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                       nhead=attention_heads,
                                                       batch_first=True)
            self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
            self.final_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    @staticmethod
    def _build_projector(in_dim: int, out_dim: int, use_mlp: bool) -> nn.Module:
        if use_mlp:
            mlp_hidden = max(in_dim, out_dim)
            return nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, out_dim)
            )
        return nn.Linear(in_dim, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (batch_size, 3, input_dim)
        Returns:
            aggregated_features: (batch_size, tokens, hidden_dim)
        """
        batch_size, num_images, feature_dim = image_features.shape
        assert num_images == 3, f"Expected 3 images, got {num_images}"

        if self.aggregation_method == "mean":
            mean_features = torch.mean(image_features, dim=1)  # (batch_size, input_dim)
            projected = self.visual_proj(mean_features)
            return projected.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        elif self.aggregation_method == "multi_token":
            if self.visual_proj is not None:
                visual_features = self.visual_proj(image_features)  # (batch_size, 3, hidden_dim)
            else:
                visual_features = image_features  # (batch_size, 3, input_dim)
            tokens = self.token_proj(visual_features)  # (batch_size, 3, hidden_dim * K)
            tokens = tokens.view(batch_size, num_images * self.tokens_per_image, self.hidden_dim)
            tokens = self.token_encoder(tokens)
            return self.final_norm(tokens)


class ImageTextModel(nn.Module):
    """Complete model: Image features -> Aggregation -> Linear -> T5 Decoder -> Text."""

    def __init__(self,
                 clip_dim: int = 512,
                 hidden_dim: int = 512,
                 t5_model_name: str = "t5-base",
                 aggregation_method: str = "multi_token",
                 use_lora: bool = True,
                 lora_r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 label_smoothing: float = 0.0,
                 tokens_per_image: int = 4,
                 attention_layers: int = 1,
                 attention_heads: int = 4,
                 use_visual_mlp: bool = False):
        """
        Args:
            clip_dim: Dimension of CLIP features
            hidden_dim: Hidden dimension for aggregation
                t5_model_name: T5 model to use
                aggregation_method: 'mean', 'multi_token', or legacy 'attention'
            tokens_per_image: Number of tokens generated per image
            attention_layers: Transformer encoder depth for cross-image mixing
            attention_heads: Number of attention heads per encoder layer
            use_lora: Whether to use LoRA for T5 fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            label_smoothing: Label smoothing factor for training
        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.tokens_per_image = tokens_per_image
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.use_visual_mlp = use_visual_mlp

        # Image feature aggregator
        self.aggregator = ImageFeatureAggregator(input_dim=clip_dim,
                                                 hidden_dim=hidden_dim,
                                                 aggregation_method=aggregation_method,
                                                 tokens_per_image=tokens_per_image,
                             attention_layers=attention_layers,
                             attention_heads=attention_heads,
                             use_visual_mlp=use_visual_mlp)

        # Load T5 model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_config = self.t5_model.config

        # Set label smoothing in T5 config if supported
        if hasattr(self.t5_config, 'label_smoothing'):
            self.t5_config.label_smoothing = label_smoothing

        # Linear projection from image features to T5 hidden dimension
        self.image_projection = nn.Linear(hidden_dim, self.t5_config.d_model)

        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v", "k", "o", "wi", "wo"]  # T5 attention and feed-forward layers
            )
            self.t5_model = get_peft_model(self.t5_model, lora_config)

        # Tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _combine_image_and_text(self,
                                projected_features: torch.Tensor,
                                input_ids: Optional[torch.Tensor] = None,
                                attention_mask: Optional[torch.Tensor] = None) -> (torch.Tensor, Optional[torch.Tensor]):
        """Prepends image tokens to text embeddings and builds the joint attention mask."""
        image_tokens = projected_features

        if input_ids is None:
            return image_tokens, None

        text_embeddings = self.t5_model.encoder.embed_tokens(input_ids)
        combined_embeddings = torch.cat([image_tokens, text_embeddings], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], device=input_ids.device, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(input_ids.device)

        image_attention = torch.ones(input_ids.size(0), image_tokens.size(1), device=input_ids.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([image_attention, attention_mask], dim=1)

        return combined_embeddings, combined_attention_mask

    def forward(self,
                image_features: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                target_ids: Optional[torch.Tensor] = None,
                target_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            image_features: (batch_size, 3, clip_dim)
            input_ids: (batch_size, seq_len) - tokenized input text
            attention_mask: (batch_size, seq_len) - attention mask for input
            target_ids: (batch_size, seq_len) - tokenized target text (for training)
            target_attention_mask: (batch_size, seq_len) - attention mask for target
        
        Returns:
            Dictionary containing model outputs
        """
        batch_size = image_features.shape[0]

        # 1. Aggregate image features
        aggregated_features = self.aggregator(image_features)  # (batch_size, tokens, hidden_dim)

        # 2. Project to T5 hidden dimension
        projected_features = self.image_projection(aggregated_features)  # (batch_size, tokens, d_model)

        # 3. Prepare inputs for T5
        # We'll use the projected image features as additional context
        # by prepending them to the input embeddings

        combined_embeddings, combined_attention_mask = self._combine_image_and_text(projected_features,
                                                                                     input_ids,
                                                                                     attention_mask)

        # 4. Forward through T5
        if target_ids is not None:
            # Training mode
            outputs = self.t5_model(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask, labels=target_ids, decoder_attention_mask=target_attention_mask)
            return {'loss': outputs.loss, 'logits': outputs.logits, 'aggregated_features': aggregated_features, 'projected_features': projected_features}
        else:
            # Inference mode - generate text
            encoder_outputs = self.t5_model.encoder(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)

            return {'encoder_outputs': encoder_outputs, 'attention_mask': combined_attention_mask, 'aggregated_features': aggregated_features, 'projected_features': projected_features}

    def generate(self,
                 image_features: torch.Tensor,
                 max_length: int = 128,
                 num_beams: int = 4,
                 temperature: float = 1.0,
                 do_sample: bool = False,
                 input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 prompt_text: Optional[List[str]] = None,
                 **generate_kwargs) -> Dict[str, Any]:
        """
        Generate captions for given image features.
        
        Args:
            image_features: (batch_size, 3, clip_dim)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            input_ids: Optional pre-tokenized prompts (batch, prompt_len)
            attention_mask: Optional mask for the prompts
            prompt_text: Optional raw prompt strings (used when input_ids is None)
        
        Returns:
            Dictionary with generated text and other info
        """
        batch_size = image_features.shape[0]
        device = image_features.device

        # Process image features
        aggregated_features = self.aggregator(image_features)
        projected_features = self.image_projection(aggregated_features)
        if projected_features.dim() == 2:
            projected_features = projected_features.unsqueeze(1)

        # Prepare text prompts
        if input_ids is None:
            if prompt_text is None:
                prompt_text = ["generate English caption: "] * batch_size
            elif isinstance(prompt_text, str):  # type: ignore[arg-type]
                prompt_text = [prompt_text] * batch_size  # type: ignore[assignment]
            encoding = self.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
        else:
            input_ids = input_ids.to(device)
            attention_mask = (attention_mask.to(device) if attention_mask is not None else torch.ones_like(input_ids, device=device))

        combined_embeddings, combined_attention_mask = self._combine_image_and_text(projected_features,
                                                                                     input_ids,
                                                                                     attention_mask)

        # Encode
        encoder_outputs = self.t5_model.encoder(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)

        # Generate with anti-repetition and English-forcing parameters
        generated_ids = self.t5_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_attention_mask,
            max_length=max_length,
            min_length=5,  # 确保最小长度
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=1.5,  # 惩罚重复 token
            length_penalty=1.0,  # 平衡长度
            no_repeat_ngram_size=4,  # 防止4-gram重复
            early_stopping=True,  # 遇到EOS提早停止
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs)

        # Decode generated text
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return {'generated_texts': generated_texts, 'generated_ids': generated_ids, 'aggregated_features': aggregated_features, 'projected_features': projected_features}

    def get_aggregation_method(self) -> str:
        """Return the current aggregation method."""
        return self.aggregator.aggregation_method

    def set_aggregation_method(self, method: str):
        """Change the aggregation method (requires reinitializing the aggregator)."""
        normalized_method = "multi_token" if method == "attention" else method
        if normalized_method != self.aggregator.aggregation_method:
            self.aggregator = ImageFeatureAggregator(input_dim=self.aggregator.input_dim,
                                                     hidden_dim=self.aggregator.hidden_dim,
                                                     aggregation_method=method,
                                                     tokens_per_image=self.tokens_per_image,
                                                     attention_layers=self.attention_layers,
                                                     attention_heads=self.attention_heads,
                                                     use_visual_mlp=self.use_visual_mlp)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = ImageTextModel(clip_dim=512, hidden_dim=512, aggregation_method="mean", use_lora=True).to(device)

    # Test forward pass
    batch_size = 2
    image_features = torch.randn(batch_size, 3, 512).to(device)

    # Test generation
    results = model.generate(image_features, max_length=50, num_beams=2)
    print("Generated texts:")
    for i, text in enumerate(results['generated_texts']):
        print(f"  {i}: {text}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
