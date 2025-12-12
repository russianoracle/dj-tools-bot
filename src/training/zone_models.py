"""Machine learning models for energy zone classification."""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from ..utils import get_logger

logger = get_logger(__name__)

# Lazy import TensorFlow (only when NeuralZoneClassifier is used)
_tf_module = None
_keras_module = None

def _get_tensorflow():
    """Lazy import TensorFlow."""
    global _tf_module, _keras_module
    if _tf_module is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
            _tf_module = tf
            _keras_module = keras
        except Exception as e:
            logger.error(f"TensorFlow not available: {e}")
            logger.error("Neural network models will not work. Use XGBoost or Ensemble instead.")
            raise ImportError(f"TensorFlow not available: {e}")
    return _tf_module, _keras_module

# Zone label mapping
ZONE_LABELS = {
    'yellow': 0,
    'green': 1,
    'purple': 2
}

ZONE_NAMES = {0: 'yellow', 1: 'green', 2: 'purple'}


class ZoneClassifierBase(ABC):
    """Base class for zone classification models."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict zone labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict zone probabilities."""
        pass

    def save(self, path: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model loaded from {path}")


class XGBoostZoneClassifier(ZoneClassifierBase):
    """XGBoost-based zone classifier."""

    def __init__(self, **params):
        """
        Initialize XGBoost classifier.

        Args:
            **params: XGBoost parameters
        """
        super().__init__()
        self.params = params or {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.label_mapping = None
        self.inverse_label_mapping = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              grid_search: bool = False,
              early_stopping_rounds: int = 50) -> Dict:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels (0=yellow, 1=green, 2=purple)
            X_val: Validation features
            y_val: Validation labels
            grid_search: Perform hyperparameter search
            early_stopping_rounds: Early stopping patience

        Returns:
            Training metrics
        """
        # Check for class label issues and remap if needed
        unique_classes = np.unique(np.concatenate([y_train, y_val]))
        n_classes = len(unique_classes)

        # Store original label mapping for later use
        self.label_mapping = None
        self.inverse_label_mapping = None

        if n_classes < 3:
            # If we have only 2 classes, XGBoost expects them to be [0, 1]
            # But they might be [0, 2] or [1, 2], so we need to remap
            logger.info(f"⚙️  Detected {n_classes} classes: {unique_classes}. Remapping to [0, 1, ...{n_classes-1}]")

            # Create mapping: original_label -> new_label
            self.label_mapping = {old: new for new, old in enumerate(unique_classes)}
            self.inverse_label_mapping = {new: old for old, new in self.label_mapping.items()}

            # Remap labels
            y_train = np.array([self.label_mapping[label] for label in y_train])
            y_val = np.array([self.label_mapping[label] for label in y_val])

            logger.info(f"✓ Labels remapped: {self.label_mapping}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        if grid_search:
            logger.info("⚙️  Performing grid search for hyperparameters...")
            self.model = self._grid_search(X_train_scaled, y_train, X_val_scaled, y_val, n_classes)
        else:
            # Standard training
            logger.info(f"⚙️  Initializing XGBoost classifier with early_stopping_rounds={early_stopping_rounds}")

            # Add early_stopping_rounds to params (for XGBoost >= 2.0)
            train_params = self.params.copy()
            train_params['early_stopping_rounds'] = early_stopping_rounds

            # Update num_class if needed
            if n_classes < 3:
                train_params['num_class'] = n_classes
                logger.info(f"✓ Adjusted num_class to {n_classes}")

            self.model = xgb.XGBClassifier(**train_params)

            logger.info(f"⚙️  Training XGBoost on {len(y_train)} samples...")
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=False
            )
            logger.info(f"✓ XGBoost training complete")

        self.is_trained = True

        # Calculate metrics
        logger.info(f"⚙️  Evaluating model performance...")
        train_preds = self.model.predict(X_train_scaled)
        val_preds = self.model.predict(X_val_scaled)

        train_acc = np.mean(train_preds == y_train)
        val_acc = np.mean(val_preds == y_val)

        best_iter = getattr(self.model, 'best_iteration', self.params.get('n_estimators', 300))

        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'best_iteration': best_iter
        }

        logger.info(f"✓ XGBoost metrics: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, best_iter={best_iter}")
        return metrics

    def _grid_search(self, X_train, y_train, X_val, y_val, n_classes=3):
        """Perform hyperparameter grid search."""
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [200, 300, 400],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }

        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            random_state=42
        )

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        logger.info(f"Best params: {grid.best_params_}")

        return grid.best_estimator_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict zone labels."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Reverse label mapping if it was applied during training
        if self.inverse_label_mapping is not None:
            predictions = np.array([self.inverse_label_mapping[pred] for pred in predictions])

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict zone probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'label_mapping': self.label_mapping,
            'inverse_label_mapping': self.inverse_label_mapping
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        # Load label mappings (backward compatible - default to None if not present)
        self.label_mapping = model_data.get('label_mapping')
        self.inverse_label_mapping = model_data.get('inverse_label_mapping')
        logger.info(f"XGBoost model loaded from {path}")


class NeuralZoneClassifier(ZoneClassifierBase):
    """Neural network-based zone classifier."""

    def __init__(self, input_dim: int = 1000, hidden_dims: list = None):
        """
        Initialize neural network classifier.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.history = None

    def _build_model(self, input_dim: int):
        """Build neural network architecture."""
        tf, keras = _get_tensorflow()

        model = keras.Sequential()

        # Input layer
        model.add(keras.layers.Input(shape=(input_dim,)))
        model.add(keras.layers.BatchNormalization())

        # Hidden layers with dropout
        for hidden_dim in self.hidden_dims:
            model.add(keras.layers.Dense(hidden_dim, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.3))

        # Output layer (3 classes)
        model.add(keras.layers.Dense(3, activation='softmax'))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 200,
              batch_size: int = 32,
              patience: int = 20) -> Dict:
        """
        Train neural network.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience

        Returns:
            Training metrics
        """
        tf, keras = _get_tensorflow()

        # Scale features
        logger.info(f"⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        logger.info(f"✓ Features scaled")

        # Build model
        logger.info(f"⚙️  Building neural network architecture...")
        self.model = self._build_model(X_train_scaled.shape[1])
        logger.info(f"✓ Model built: input_dim={X_train_scaled.shape[1]}, layers={self.hidden_dims}")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]

        # Train
        logger.info(f"⚙️  Training neural network for up to {epochs} epochs (batch_size={batch_size})...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        logger.info(f"✓ Training complete")

        self.is_trained = True

        # Calculate final metrics
        logger.info(f"⚙️  Evaluating model performance...")
        train_loss, train_acc = self.model.evaluate(X_train_scaled, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val_scaled, y_val, verbose=0)

        epochs_trained = len(self.history.history['loss'])

        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs_trained': epochs_trained
        }

        logger.info(f"✓ Neural network metrics: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, epochs={epochs_trained}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict zone labels."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict zone probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)

    def save(self, path: str):
        """Save model to file."""
        # Save keras model separately
        model_path = Path(path)
        keras_path = model_path.parent / f"{model_path.stem}_keras.h5"

        self.model.save(keras_path)

        # Save rest with pickle
        model_data = {
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'keras_path': str(keras_path)
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Neural model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        tf, keras = _get_tensorflow()

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.input_dim = model_data['input_dim']
        self.hidden_dims = model_data['hidden_dims']

        # Load keras model
        keras_path = model_data['keras_path']
        self.model = keras.models.load_model(keras_path)

        logger.info(f"Neural model loaded from {path}")


class EnsembleZoneClassifier(ZoneClassifierBase):
    """Ensemble combining XGBoost and Neural Network."""

    def __init__(self, xgb_params: dict = None, nn_params: dict = None):
        """
        Initialize ensemble classifier.

        Args:
            xgb_params: XGBoost parameters
            nn_params: Neural network parameters
        """
        super().__init__()
        self.xgb_model = XGBoostZoneClassifier(**(xgb_params or {}))
        self.nn_model = None  # Created during training
        self.nn_params = nn_params or {}
        self.weights = [0.6, 0.4]  # XGBoost, NN weights

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """
        Train ensemble models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Training parameters

        Returns:
            Combined metrics
        """
        # Filter kwargs for each model type
        xgb_kwargs = {k: v for k, v in kwargs.items() if k in ['grid_search', 'early_stopping_rounds']}
        nn_kwargs = {k: v for k, v in kwargs.items() if k in ['epochs', 'batch_size', 'patience']}

        logger.info("Training XGBoost component...")
        xgb_metrics = self.xgb_model.train(X_train, y_train, X_val, y_val, **xgb_kwargs)

        logger.info("Training Neural Network component...")
        self.nn_model = NeuralZoneClassifier(input_dim=X_train.shape[1], **self.nn_params)
        nn_metrics = self.nn_model.train(X_train, y_train, X_val, y_val, **nn_kwargs)

        # Mark as trained so predict methods work
        self.is_trained = True

        # Optimize ensemble weights
        self._optimize_weights(X_val, y_val)

        # Ensemble validation accuracy
        val_preds = self.predict(X_val)
        ensemble_acc = np.mean(val_preds == y_val)

        metrics = {
            'xgb_val_accuracy': xgb_metrics['val_accuracy'],
            'nn_val_accuracy': nn_metrics['val_accuracy'],
            'ensemble_val_accuracy': ensemble_acc,
            'ensemble_weights': self.weights
        }

        logger.info(f"Ensemble training complete: val_acc={ensemble_acc:.3f}")
        return metrics

    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Optimize ensemble weights on validation set."""
        xgb_probs = self.xgb_model.predict_proba(X_val)
        nn_probs = self.nn_model.predict_proba(X_val)

        best_acc = 0
        best_weights = [0.5, 0.5]

        # Grid search over weights
        for w1 in np.arange(0.1, 1.0, 0.1):
            w2 = 1.0 - w1
            ensemble_probs = w1 * xgb_probs + w2 * nn_probs
            preds = np.argmax(ensemble_probs, axis=1)
            acc = np.mean(preds == y_val)

            if acc > best_acc:
                best_acc = acc
                best_weights = [w1, w2]

        self.weights = best_weights
        logger.info(f"Optimized weights: XGBoost={self.weights[0]:.2f}, NN={self.weights[1]:.2f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict zone labels using ensemble."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict zone probabilities using weighted average."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        xgb_probs = self.xgb_model.predict_proba(X)
        nn_probs = self.nn_model.predict_proba(X)

        ensemble_probs = self.weights[0] * xgb_probs + self.weights[1] * nn_probs
        return ensemble_probs

    def save(self, path: str):
        """Save ensemble to file."""
        model_path = Path(path)

        # Save individual models
        xgb_path = model_path.parent / f"{model_path.stem}_xgb.pkl"
        nn_path = model_path.parent / f"{model_path.stem}_nn.pkl"

        self.xgb_model.save(str(xgb_path))
        self.nn_model.save(str(nn_path))

        # Save ensemble metadata
        ensemble_data = {
            'weights': self.weights,
            'is_trained': self.is_trained,
            'xgb_path': str(xgb_path),
            'nn_path': str(nn_path)
        }

        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)

        logger.info(f"Ensemble saved to {path}")

    def load(self, path: str):
        """Load ensemble from file."""
        with open(path, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.weights = ensemble_data['weights']
        self.is_trained = ensemble_data['is_trained']

        # Load individual models
        self.xgb_model.load(ensemble_data['xgb_path'])
        self.nn_model = NeuralZoneClassifier()
        self.nn_model.load(ensemble_data['nn_path'])

        logger.info(f"Ensemble loaded from {path}")
