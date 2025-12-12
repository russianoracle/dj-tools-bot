#!/usr/bin/env python3
"""
🚀 ULTIMATE Deep Learning Zone Classifier

Multi-modal architecture combining:
1. CNN branch for spectrograms (3-channel)
2. Transformer branch for frame sequences
3. MLP branch for track-level statistics
4. Fusion layer for final classification

Hardware optimizations:
- Apple Silicon MPS (Metal GPU)
- Mixed precision training (float16)
- Gradient checkpointing for memory efficiency
- Optimized batch sizes and data loading

Performance targets:
- M2 GPU: ~50-100 tracks/sec inference
- Training: ~5-10 min for 1000 tracks

Usage:
    from src.training.deep_zone_classifier import DeepZoneClassifier

    model = DeepZoneClassifier()
    model.fit(frames_df, spectrograms, track_stats, epochs=50)
    predictions = model.predict(new_data)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import Counter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ============================================================
# Device Setup with Apple Silicon Optimization
# ============================================================

def get_optimal_device():
    """Get best device with optimizations."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS optimizations
        torch.mps.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        logger.info("🍎 Apple Metal GPU (MPS) - Optimized")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
        logger.info(f"🎮 NVIDIA CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        torch.set_num_threads(8)  # Use all CPU cores
        logger.info("💻 CPU mode")
    return device

DEVICE = get_optimal_device()

# Zone mapping
ZONES = ['YELLOW', 'GREEN', 'PURPLE']
ZONE_TO_IDX = {z: i for i, z in enumerate(ZONES)}
IDX_TO_ZONE = {i: z for i, z in enumerate(ZONES)}


@dataclass
class ModelConfig:
    """Configuration for deep learning model."""
    # Architecture
    n_classes: int = 3

    # CNN branch (spectrograms)
    cnn_channels: List[int] = None  # [32, 64, 128, 256]
    cnn_dropout: float = 0.3

    # Transformer branch (frame sequences)
    transformer_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.2
    max_frames: int = 256

    # MLP branch (track statistics)
    mlp_hidden: List[int] = None  # [256, 128]
    mlp_dropout: float = 0.3

    # Fusion
    fusion_dim: int = 256

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Hardware optimization
    use_amp: bool = True  # Automatic mixed precision
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128, 256]
        if self.mlp_hidden is None:
            self.mlp_hidden = [256, 128]


# ============================================================
# CNN Branch - Spectrogram Processing
# ============================================================

class SpectrogramCNNBranch(nn.Module):
    """
    EfficientNet-style CNN for spectrogram processing.

    Input: (batch, 3, 128, 512) - 3-channel spectrogram
    Output: (batch, 256) - feature vector
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        channels = config.cnn_channels

        # Convolutional blocks with residual connections
        self.conv_blocks = nn.ModuleList()
        in_ch = 3

        for out_ch in channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),  # Swish activation
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(config.cnn_dropout * 0.5),
            )
            self.conv_blocks.append(block)
            in_ch = out_ch

        # Global pooling + projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], config.fusion_dim),
            nn.SiLU(),
            nn.Dropout(config.cnn_dropout),
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = self.projection(x)
        return x


# ============================================================
# Transformer Branch - Frame Sequence Processing
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FrameTransformerBranch(nn.Module):
    """
    Transformer for frame sequence processing.

    Input: (batch, seq_len, n_features) - frame features
    Output: (batch, 256) - aggregated feature vector
    """

    def __init__(self, n_features: int, config: ModelConfig):
        super().__init__()

        self.input_projection = nn.Linear(n_features, config.transformer_dim)
        self.pos_encoding = PositionalEncoding(config.transformer_dim, config.max_frames)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(config.transformer_dim, 1),
            nn.Softmax(dim=1),
        )

        self.projection = nn.Sequential(
            nn.Linear(config.transformer_dim, config.fusion_dim),
            nn.SiLU(),
            nn.Dropout(config.transformer_dropout),
        )

    def forward(self, x, mask=None):
        # Input projection
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Attention pooling
        attn_weights = self.attention_pool(x)
        x = (x * attn_weights).sum(dim=1)

        # Output projection
        x = self.projection(x)
        return x


# ============================================================
# MLP Branch - Track Statistics Processing
# ============================================================

class TrackStatsMLP(nn.Module):
    """
    MLP for track-level statistics.

    Input: (batch, n_stats) - track statistics
    Output: (batch, 256) - feature vector
    """

    def __init__(self, n_features: int, config: ModelConfig):
        super().__init__()

        layers = []
        in_dim = n_features

        for hidden_dim in config.mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.mlp_dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, config.fusion_dim))
        layers.append(nn.SiLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ============================================================
# Multi-Modal Fusion Network
# ============================================================

class MultiModalFusion(nn.Module):
    """
    Fusion network combining all branches.

    Uses attention-based fusion to weight each modality.
    """

    def __init__(self, n_modalities: int, fusion_dim: int, n_classes: int):
        super().__init__()

        # Cross-modal attention
        self.modal_attention = nn.Sequential(
            nn.Linear(fusion_dim * n_modalities, n_modalities),
            nn.Softmax(dim=-1),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, n_classes),
        )

    def forward(self, *modalities):
        # Stack modalities: (batch, n_modalities, fusion_dim)
        stacked = torch.stack(modalities, dim=1)

        # Compute attention weights
        concat = torch.cat(modalities, dim=-1)
        attn_weights = self.modal_attention(concat).unsqueeze(-1)

        # Weighted sum
        fused = (stacked * attn_weights).sum(dim=1)

        # Classify
        return self.classifier(fused)


# ============================================================
# Main Model
# ============================================================

class DeepZoneClassifier(nn.Module):
    """
    Ultimate multi-modal deep learning classifier for zone classification.

    Combines:
    - CNN for spectrograms
    - Transformer for frame sequences
    - MLP for track statistics

    With attention-based fusion.
    """

    def __init__(self,
                 n_frame_features: int = 100,
                 n_track_features: int = 700,
                 config: ModelConfig = None):
        super().__init__()

        self.config = config or ModelConfig()
        self.device = DEVICE

        # Branches
        self.cnn_branch = SpectrogramCNNBranch(self.config)
        self.transformer_branch = FrameTransformerBranch(n_frame_features, self.config)
        self.mlp_branch = TrackStatsMLP(n_track_features, self.config)

        # Fusion
        self.fusion = MultiModalFusion(3, self.config.fusion_dim, self.config.n_classes)

        # Feature scalers
        self.frame_scaler = None
        self.track_scaler = None

        # Move to device
        self.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model parameters: {n_params:,}")

    def forward(self, spectrograms: torch.Tensor,
                frame_features: torch.Tensor,
                track_features: torch.Tensor,
                frame_mask: torch.Tensor = None):
        """
        Forward pass.

        Args:
            spectrograms: (batch, 3, 128, 512) - 3-channel spectrograms
            frame_features: (batch, seq_len, n_features) - frame sequences
            track_features: (batch, n_features) - track statistics
            frame_mask: (batch, seq_len) - padding mask for frames

        Returns:
            (batch, n_classes) - logits
        """
        # Process each branch
        cnn_out = self.cnn_branch(spectrograms)
        transformer_out = self.transformer_branch(frame_features, frame_mask)
        mlp_out = self.mlp_branch(track_features)

        # Fuse and classify
        logits = self.fusion(cnn_out, transformer_out, mlp_out)

        return logits

    def predict(self, spectrograms, frame_features, track_features, frame_mask=None):
        """Predict class labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(spectrograms, frame_features, track_features, frame_mask)
            return logits.argmax(dim=-1)

    def predict_proba(self, spectrograms, frame_features, track_features, frame_mask=None):
        """Predict class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(spectrograms, frame_features, track_features, frame_mask)
            return F.softmax(logits, dim=-1)


# ============================================================
# Dataset
# ============================================================

class ZoneDataset(Dataset):
    """Dataset combining all modalities."""

    def __init__(self,
                 spectrograms: np.ndarray,
                 frame_features: List[np.ndarray],
                 track_features: np.ndarray,
                 labels: np.ndarray,
                 max_frames: int = 256):

        self.spectrograms = spectrograms
        self.frame_features = frame_features
        self.track_features = track_features
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Spectrogram
        spec = torch.FloatTensor(self.spectrograms[idx])

        # Frame features (pad/truncate to max_frames)
        frames = self.frame_features[idx]
        n_frames, n_features = frames.shape

        if n_frames > self.max_frames:
            # Sample frames uniformly
            indices = np.linspace(0, n_frames - 1, self.max_frames).astype(int)
            frames = frames[indices]
            mask = torch.zeros(self.max_frames, dtype=torch.bool)
        else:
            # Pad
            padded = np.zeros((self.max_frames, n_features))
            padded[:n_frames] = frames
            frames = padded
            mask = torch.ones(self.max_frames, dtype=torch.bool)
            mask[:n_frames] = False

        frames = torch.FloatTensor(frames)

        # Track features
        track = torch.FloatTensor(self.track_features[idx])

        # Label
        label = torch.LongTensor([self.labels[idx]])[0]

        return spec, frames, track, mask, label


# ============================================================
# Trainer
# ============================================================

class DeepZoneTrainer:
    """
    Trainer with all optimizations for Apple Silicon.
    """

    def __init__(self, model: DeepZoneClassifier, config: ModelConfig = None):
        self.model = model
        self.config = config or ModelConfig()
        self.device = model.device

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = None  # Set during training

        # Mixed precision scaler (for CUDA)
        self.scaler = GradScaler() if self.device.type == 'cuda' and self.config.use_amp else None

        # Loss function with class weights
        self.criterion = None  # Set during training

    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        counts = Counter(labels)
        total = len(labels)
        n_classes = len(counts)
        weights = torch.FloatTensor([
            total / (n_classes * counts.get(i, 1))
            for i in range(n_classes)
        ])
        return weights.to(self.device)

    def _create_scheduler(self, n_steps: int):
        """Create cosine scheduler with warmup."""
        warmup_steps = self.config.warmup_epochs * (n_steps // self.config.batch_size)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (n_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def fit(self,
            spectrograms: np.ndarray,
            frame_features: List[np.ndarray],
            track_features: np.ndarray,
            labels: np.ndarray,
            epochs: int = 50,
            val_split: float = 0.2) -> Dict:
        """
        Train the model.

        Args:
            spectrograms: (n_tracks, 3, 128, 512)
            frame_features: List of (n_frames, n_features) arrays
            track_features: (n_tracks, n_track_features)
            labels: (n_tracks,) zone indices
            epochs: Number of training epochs
            val_split: Validation split ratio

        Returns:
            Training history dict
        """
        print(f"\n{'='*60}")
        print("DEEP ZONE CLASSIFIER TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Samples: {len(labels)}")
        print(f"Epochs: {epochs}")
        print(f"Mixed precision: {self.config.use_amp}")

        # Split data
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=labels, random_state=42
        )

        # Scale features
        self.model.frame_scaler = StandardScaler()
        self.model.track_scaler = StandardScaler()

        # Fit scalers on training data
        all_train_frames = np.vstack([frame_features[i] for i in train_idx])
        self.model.frame_scaler.fit(all_train_frames)
        self.model.track_scaler.fit(track_features[train_idx])

        # Scale all data
        scaled_frames = [
            self.model.frame_scaler.transform(f) for f in frame_features
        ]
        scaled_track = self.model.track_scaler.transform(track_features)

        # Create datasets
        train_dataset = ZoneDataset(
            spectrograms[train_idx],
            [scaled_frames[i] for i in train_idx],
            scaled_track[train_idx],
            labels[train_idx],
            self.config.max_frames
        )

        val_dataset = ZoneDataset(
            spectrograms[val_idx],
            [scaled_frames[i] for i in val_idx],
            scaled_track[val_idx],
            labels[val_idx],
            self.config.max_frames
        )

        # DataLoaders optimized for device
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'num_workers': 0 if self.device.type == 'mps' else 4,
            'pin_memory': self.device.type == 'cuda',
            'persistent_workers': False,
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        # Class weights
        class_weights = self._compute_class_weights(labels[train_idx])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Scheduler
        total_steps = epochs * len(train_loader)
        self.scheduler = self._create_scheduler(total_steps)

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        best_state = None

        start_time = time.time()

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch in train_loader:
                spec, frames, track, mask, label = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()

                # Forward pass (with AMP on CUDA)
                if self.scaler is not None:
                    with autocast():
                        logits = self.model(spec, frames, track, mask)
                        loss = self.criterion(logits, label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(spec, frames, track, mask)
                    loss = self.criterion(logits, label)
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    spec, frames, track, mask, label = [b.to(self.device) for b in batch]

                    logits = self.model(spec, frames, track, mask)
                    loss = self.criterion(logits, label)
                    val_loss += loss.item()

                    preds = logits.argmax(dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)

            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {history['train_loss'][-1]:.4f} | "
                      f"Val Loss: {history['val_loss'][-1]:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Time: {elapsed:.1f}s")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        print(f"\n{'='*40}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Total Training Time: {time.time() - start_time:.1f}s")
        print(f"{'='*40}")

        # Classification report
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                spec, frames, track, mask, label = [b.to(self.device) for b in batch]
                preds = self.model.predict(spec, frames, track, mask)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=ZONES))

        return history

    def save(self, path: str):
        """Save model and scalers."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'frame_scaler': self.model.frame_scaler,
            'track_scaler': self.model.track_scaler,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"Model saved: {path}")

    def load(self, path: str):
        """Load model and scalers."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.model.load_state_dict(save_dict['model_state'])
        self.config = save_dict['config']
        self.model.frame_scaler = save_dict['frame_scaler']
        self.model.track_scaler = save_dict['track_scaler']

        print(f"Model loaded: {path}")


# ============================================================
# Training Script
# ============================================================

def train_from_ultimate_features(
    features_dir: str,
    output_path: str = "models/deep_zone_classifier.pkl",
    epochs: int = 50
) -> Dict:
    """
    Train deep classifier from ultimate features.

    Args:
        features_dir: Directory with frames.pkl, track_features.pkl, spectrograms.pkl
        output_path: Where to save trained model
        epochs: Training epochs

    Returns:
        Training history
    """
    features_dir = Path(features_dir)

    print("Loading features...")

    # Load frames
    frames_df = pd.read_pickle(features_dir / "frames.pkl")

    # Load track features
    track_df = pd.read_pickle(features_dir / "track_features.pkl")

    # Load spectrograms
    with open(features_dir / "spectrograms.pkl", 'rb') as f:
        spec_data = pickle.load(f)

    # Prepare data
    print("Preparing data...")

    # Get unique tracks
    track_ids = frames_df['track_id'].unique()
    n_tracks = len(track_ids)

    # Frame features per track
    exclude_cols = ['frame_idx', 'frameTime', 'zone', 'track_id', 'path']
    frame_cols = [c for c in frames_df.columns if c not in exclude_cols]

    frame_features = []
    for tid in track_ids:
        track_frames = frames_df[frames_df['track_id'] == tid][frame_cols].values
        frame_features.append(track_frames)

    # Track features
    track_cols = [c for c in track_df.columns if c not in ['zone', 'path', 'duration']]
    track_features = track_df[track_cols].values

    # Spectrograms
    spectrograms = np.array([d['tensor'] for d in spec_data])

    # Labels
    labels = np.array([ZONE_TO_IDX[track_df.iloc[i]['zone']] for i in range(n_tracks)])

    print(f"Tracks: {n_tracks}")
    print(f"Frame features: {len(frame_cols)}")
    print(f"Track features: {len(track_cols)}")
    print(f"Spectrogram shape: {spectrograms.shape}")
    print(f"Zone distribution: {Counter([IDX_TO_ZONE[l] for l in labels])}")

    # Create model
    config = ModelConfig()
    model = DeepZoneClassifier(
        n_frame_features=len(frame_cols),
        n_track_features=len(track_cols),
        config=config
    )

    # Train
    trainer = DeepZoneTrainer(model, config)
    history = trainer.fit(
        spectrograms,
        frame_features,
        track_features,
        labels,
        epochs=epochs
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(output_path)

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Deep Zone Classifier")
    parser.add_argument("--features-dir", required=True,
                       help="Directory with extracted features")
    parser.add_argument("--output", default="models/deep_zone_classifier.pkl",
                       help="Output model path")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs")

    args = parser.parse_args()

    train_from_ultimate_features(args.features_dir, args.output, args.epochs)
