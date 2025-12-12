#!/usr/bin/env python3
"""
CNN model for zone classification based on spectrograms.
Uses 3-channel input: mel spectrogram, energy curve, drop/buildup mask.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

logger = logging.getLogger(__name__)

# Zone label mapping
ZONE_TO_IDX = {'YELLOW': 0, 'GREEN': 1, 'PURPLE': 2}
IDX_TO_ZONE = {v: k for k, v in ZONE_TO_IDX.items()}


class SpectrogramCNN:
    """
    CNN classifier for audio spectrograms.
    Works with or without PyTorch/TensorFlow.
    Falls back to sklearn if deep learning not available.
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (3, 128, 512),
                 n_classes: int = 3,
                 model_path: Optional[str] = None):
        """
        Args:
            input_shape: (channels, height, width)
            n_classes: Number of output classes
            model_path: Path to load pre-trained model
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = None
        self.use_torch = False
        self.scaler = None

        # Try to import PyTorch
        try:
            import torch
            import torch.nn as nn
            self.use_torch = True
            self._build_torch_model()
            logger.info("Using PyTorch CNN model")
        except ImportError:
            logger.info("PyTorch not available, using sklearn fallback")
            self._build_sklearn_model()

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _build_torch_model(self):
        """Build PyTorch CNN model."""
        import torch
        import torch.nn as nn

        class ZoneCNN(nn.Module):
            def __init__(self, in_channels, n_classes):
                super().__init__()

                # Feature extraction
                self.features = nn.Sequential(
                    # Block 1: 3 -> 32
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.1),

                    # Block 2: 32 -> 64
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.2),

                    # Block 3: 64 -> 128
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.3),

                    # Block 4: 128 -> 256
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 8)),  # Fixed output size
                )

                # Classifier
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 4 * 8, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, n_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        self.model = ZoneCNN(self.input_shape[0], self.n_classes)
        # Device priority: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("🍎 Using Apple Metal GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("🎮 Using NVIDIA CUDA GPU")
        else:
            self.device = torch.device('cpu')
            logger.info("💻 Using CPU")
        self.model.to(self.device)

    def _build_sklearn_model(self):
        """Build sklearn fallback model (uses flattened features)."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()

    def _extract_sklearn_features(self, tensor: np.ndarray) -> np.ndarray:
        """Extract features from tensor for sklearn model."""
        # Compute statistics for each channel
        features = []

        for c in range(tensor.shape[0]):
            channel = tensor[c]

            # Basic stats
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel),
                np.median(channel),
            ])

            # Temporal stats (along width)
            temporal_mean = np.mean(channel, axis=0)
            features.extend([
                np.mean(temporal_mean),
                np.std(temporal_mean),
                np.max(temporal_mean) - np.min(temporal_mean),  # Range
            ])

            # Frequency stats (along height)
            freq_mean = np.mean(channel, axis=1)
            features.extend([
                np.mean(freq_mean),
                np.std(freq_mean),
            ])

            # Energy distribution
            total = np.sum(channel) + 1e-10
            features.append(np.sum(channel[:channel.shape[0]//2]) / total)  # Low freq ratio

        return np.array(features)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.2,
            class_weights: Optional[Dict[int, float]] = None) -> Dict:
        """
        Train the model.

        Args:
            X: Input tensors shape (N, C, H, W)
            y: Labels (zone indices)
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            class_weights: Optional class weight dict

        Returns:
            Training history dict
        """
        # Compute class weights if not provided
        if class_weights is None:
            counts = Counter(y)
            total = len(y)
            class_weights = {c: total / (len(counts) * count)
                           for c, count in counts.items()}

        if self.use_torch:
            return self._fit_torch(X, y, epochs, batch_size, validation_split, class_weights)
        else:
            return self._fit_sklearn(X, y)

    def _fit_torch(self, X, y, epochs, batch_size, val_split, class_weights):
        """Train PyTorch model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, stratify=y, random_state=42
        )

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        # Optimize DataLoader for Apple Silicon
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,  # MPS works best with num_workers=0
            'pin_memory': False if self.device.type == 'mps' else True,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        # Loss and optimizer
        weights = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(self.n_classes)])
        weights = weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._best_state = self.model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Val Loss: {history['val_loss'][-1]:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        # Restore best model
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state)

        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        return history

    def _fit_sklearn(self, X, y):
        """Train sklearn model."""
        # Extract features
        X_features = np.array([self._extract_sklearn_features(x) for x in X])

        # Scale
        X_scaled = self.scaler.fit_transform(X_features)

        # Train
        self.model.fit(X_scaled, y)

        # Evaluate
        preds = self.model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        print(f"Training accuracy: {acc:.4f}")

        return {'train_acc': [acc]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.use_torch:
            return self._predict_torch(X)
        else:
            return self._predict_sklearn(X)

    def _predict_torch(self, X, batch_size: int = 32):
        """Predict with PyTorch model (batched to avoid OOM)."""
        import torch

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                X_tensor = torch.FloatTensor(batch).to(self.device)
                outputs = self.model(X_tensor)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                # Clear GPU memory
                del X_tensor, outputs
                if self.device.type == 'mps':
                    torch.mps.empty_cache()

        return np.concatenate(all_preds)

    def _predict_sklearn(self, X):
        """Predict with sklearn model."""
        X_features = np.array([self._extract_sklearn_features(x) for x in X])
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.use_torch:
            return self._predict_proba_torch(X)
        else:
            return self._predict_proba_sklearn(X)

    def _predict_proba_torch(self, X, batch_size: int = 32):
        """Predict probabilities with PyTorch (batched to avoid OOM)."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                X_tensor = torch.FloatTensor(batch).to(self.device)
                outputs = self.model(X_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
                # Clear GPU memory
                del X_tensor, outputs
                if self.device.type == 'mps':
                    torch.mps.empty_cache()

        return np.concatenate(all_probs)

    def _predict_proba_sklearn(self, X):
        """Predict probabilities with sklearn."""
        X_features = np.array([self._extract_sklearn_features(x) for x in X])
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance."""
        preds = self.predict(X)

        results = {
            'accuracy': accuracy_score(y, preds),
            'confusion_matrix': confusion_matrix(y, preds),
            'classification_report': classification_report(
                y, preds,
                target_names=[IDX_TO_ZONE[i] for i in range(self.n_classes)],
                output_dict=True
            )
        }

        return results

    def save(self, path: str):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'use_torch': self.use_torch,
        }

        if self.use_torch:
            import torch
            save_dict['model_state'] = self.model.state_dict()
        else:
            save_dict['model'] = self.model
            save_dict['scaler'] = self.scaler

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.input_shape = save_dict['input_shape']
        self.n_classes = save_dict['n_classes']

        if save_dict['use_torch'] and self.use_torch:
            self.model.load_state_dict(save_dict['model_state'])
        elif not save_dict['use_torch']:
            self.model = save_dict['model']
            self.scaler = save_dict['scaler']
            self.use_torch = False

        logger.info(f"Model loaded from {path}")


def train_from_dataset(dataset_path: str,
                      output_model: str = "models/production/spectrogram_cnn.pkl",
                      epochs: int = 50,
                      batch_size: int = 32) -> Dict:
    """
    Train CNN model from spectrogram dataset.

    Args:
        dataset_path: Path to spectrogram_dataset.pkl
        output_model: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Training results dict
    """
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Prepare data
    X = np.array([d['tensor'] for d in dataset])
    y = np.array([ZONE_TO_IDX[d['zone']] for d in dataset])

    print(f"Dataset: {len(X)} samples")
    print(f"Input shape: {X.shape}")
    print(f"Zone distribution: {Counter([d['zone'] for d in dataset])}")

    # Create model
    model = SpectrogramCNN(input_shape=X.shape[1:])

    # Train
    print("\nTraining...")
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # Evaluate
    print("\nEvaluating...")
    results = model.evaluate(X, y)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    for zone, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"  {zone}: precision={metrics['precision']:.2f}, "
                  f"recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")

    # Save
    model.save(output_model)
    print(f"\nModel saved: {output_model}")

    return {
        'history': history,
        'results': results,
        'model_path': output_model
    }


def cross_validate(dataset_path: str, n_splits: int = 5) -> Dict:
    """Run cross-validation on spectrogram dataset."""
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    X = np.array([d['tensor'] for d in dataset])
    y = np.array([ZONE_TO_IDX[d['zone']] for d in dataset])

    print(f"Dataset: {len(X)} samples")
    print(f"Running {n_splits}-fold cross-validation...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = SpectrogramCNN(input_shape=X.shape[1:])
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

        results = model.evaluate(X_test, y_test)
        fold_results.append(results['accuracy'])
        print(f"Fold {fold+1} Accuracy: {results['accuracy']:.4f}")

    print(f"\n{'='*50}")
    print(f"CV Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")

    return {
        'fold_accuracies': fold_results,
        'mean_accuracy': np.mean(fold_results),
        'std_accuracy': np.std(fold_results)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train spectrogram CNN")
    parser.add_argument("--dataset", required=True, help="Path to spectrogram_dataset.pkl")
    parser.add_argument("--output", default="models/production/spectrogram_cnn.pkl",
                       help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation instead")

    args = parser.parse_args()

    if args.cv:
        cross_validate(args.dataset)
    else:
        train_from_dataset(args.dataset, args.output, args.epochs, args.batch_size)
