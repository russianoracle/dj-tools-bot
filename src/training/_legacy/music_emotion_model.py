"""Music emotion recognition using pretrained DistilHuBERT + regression head."""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor
from pathlib import Path
from typing import Tuple, Optional
from huggingface_hub import PyTorchModelHubMixin

from ..utils import get_logger

logger = get_logger(__name__)


class MusicEmotionRegressor(nn.Module, PyTorchModelHubMixin):
    """
    Music emotion regressor based on DistilHuBERT.

    Predicts arousal and valence values for music.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize model.

        Args:
            config: Model configuration dict with:
                - model_name: Base model (default: ntu-spml/distilhubert)
                - output_size: Number of outputs (default: 2 for arousal/valence)
                - pooling: Pooling strategy (default: max)
                - n_output_layers: Number of regression layers
                - dropout_prob, hidden_dropout_prob, attention_probs_dropout_prob
        """
        super().__init__()

        if config is None:
            config = {}

        self.config = config

        # Base model for feature extraction
        base_model_name = config.get('model_name', 'ntu-spml/distilhubert')
        self.feature_extractor_model = AutoModel.from_pretrained(base_model_name)

        # Freeze base model (only train regression head)
        for param in self.feature_extractor_model.parameters():
            param.requires_grad = False

        # Configuration
        self.pooling = config.get('pooling', 'max')
        self.output_size = config.get('output_size', 2)
        self.n_output_layers = config.get('n_output_layers', 3)

        # Dropout probabilities
        dropout_prob = config.get('dropout_prob', 0.1)
        hidden_dropout = config.get('hidden_dropout_prob', 0.1)

        # Get hidden size from base model
        hidden_size = self.feature_extractor_model.config.hidden_size

        # Build regression head
        layers = []

        # First layer: hidden_size -> hidden_size//2
        layers.extend([
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout)
        ])

        # Middle layers (if n_output_layers > 2)
        for _ in range(self.n_output_layers - 2):
            layers.extend([
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])

        # Final layer: hidden_size//2 -> output_size
        layers.append(nn.Linear(hidden_size // 2, self.output_size))

        self.regression_head = nn.Sequential(*layers)

        logger.info(f"Initialized MusicEmotionRegressor with base model: {base_model_name}")
        logger.info(f"Pooling: {self.pooling}, Output size: {self.output_size}")

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_values: Audio waveform tensor (batch_size, sequence_length)
            attention_mask: Optional attention mask

        Returns:
            Tensor of shape (batch_size, output_size) with arousal/valence predictions
        """
        # Extract features from base model
        outputs = self.feature_extractor_model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get hidden states (last layer)
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_size)

        # Apply pooling
        if self.pooling == 'max':
            pooled = torch.max(hidden_states, dim=1)[0]  # (batch, hidden_size)
        elif self.pooling == 'mean':
            pooled = torch.mean(hidden_states, dim=1)  # (batch, hidden_size)
        else:
            # Default to mean pooling
            pooled = torch.mean(hidden_states, dim=1)

        # Pass through regression head
        output = self.regression_head(pooled)  # (batch, output_size)

        return output


class MusicEmotionFeatureExtractor:
    """
    Feature extractor for music emotion recognition.

    Uses pretrained Rehead/music_emotion_regressor to extract arousal/valence.
    """

    def __init__(self, device: str = 'cpu', use_gpu: bool = True):
        """
        Initialize feature extractor.

        Args:
            device: Device to run model on
            use_gpu: Whether to use GPU if available
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Load pretrained model from HuggingFace
        logger.info("Loading Rehead/music_emotion_regressor...")
        try:
            self.model = MusicEmotionRegressor.from_pretrained("Rehead/music_emotion_regressor")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Load feature extractor for audio preprocessing
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("ntu-spml/distilhubert")
        self.sample_rate = self.feature_extractor.sampling_rate

        logger.info(f"Feature extractor ready (sample_rate: {self.sample_rate})")

    def extract_arousal_valence(self, audio_path: str) -> Tuple[float, float]:
        """
        Extract arousal and valence from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (arousal, valence) values in range [-1, 1]
        """
        import torchaudio

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Preprocess with feature extractor
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        # Move to device
        input_values = inputs.input_values.to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_values)
            arousal, valence = outputs[0].cpu().numpy()

        return float(arousal), float(valence)

    def extract_batch(self, audio_paths: list) -> list:
        """
        Extract arousal/valence from multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            List of (arousal, valence) tuples
        """
        results = []

        for path in audio_paths:
            try:
                arousal, valence = self.extract_arousal_valence(path)
                results.append((arousal, valence))
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append((0.0, 0.0))  # Fallback to neutral

        return results
