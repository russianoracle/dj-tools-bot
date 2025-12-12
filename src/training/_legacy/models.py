"""Machine learning models for BPM correction."""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from tensorflow import keras
    from tensorflow.keras import layers

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils import get_logger

logger = get_logger(__name__)

# Lazy TensorFlow import
TENSORFLOW_AVAILABLE = None
_tf_cache = None

def _get_tensorflow():
    """Lazy import TensorFlow for BPM models."""
    global TENSORFLOW_AVAILABLE, _tf_cache
    if _tf_cache is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            _tf_cache = (tf, keras, layers)
            TENSORFLOW_AVAILABLE = True
        except Exception as e:
            logger.error(f"TensorFlow not available: {e}")
            TENSORFLOW_AVAILABLE = False
            raise ImportError(f"TensorFlow not available: {e}")
    return _tf_cache


class BPMCorrectionModel(ABC):
    """Abstract base class for BPM correction models."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    def save(self, path: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
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
        self.feature_names = model_data['feature_names']
        logger.info(f"Model loaded from {path}")


class XGBoostBPMModel(BPMCorrectionModel):
    """XGBoost-based BPM correction model."""

    def __init__(self):
        super().__init__()
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              grid_search: bool = False,
              **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            X: Training features
            y: Training targets (BPM values)
            X_val: Validation features
            y_val: Validation targets
            grid_search: Whether to perform grid search for hyperparameters

        Returns:
            Training results with metrics
        """
        logger.info("Training XGBoost BPM correction model...")

        self.feature_names = X.columns.tolist()

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        if grid_search:
            logger.info("Performing grid search...")
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1)
            grid.fit(X_scaled_df, y)

            self.model = grid.best_estimator_
            best_params = grid.best_params_
            logger.info(f"Best parameters: {best_params}")
        else:
            # Default parameters
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )

            # Train with early stopping if validation set provided
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_scaled, y), (X_val_scaled, y_val)]
                self.model.fit(
                    X_scaled_df, y,
                    eval_set=eval_set,
                    eval_metric='mae',
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(X_scaled_df, y)

        # Calculate training metrics
        y_pred_train = self.model.predict(X_scaled_df)
        train_mae = np.mean(np.abs(y - y_pred_train))

        results = {
            'train_mae': train_mae,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }

        if X_val is not None and y_val is not None:
            y_pred_val = self.predict(X_val)
            val_mae = np.mean(np.abs(y_val - y_pred_val))
            results['val_mae'] = val_mae
            logger.info(f"Validation MAE: {val_mae:.2f} BPM")

        logger.info(f"Training MAE: {train_mae:.2f} BPM")
        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return dict(zip(self.feature_names, self.model.feature_importances_))


class NeuralBPMModel(BPMCorrectionModel):
    """Neural network-based BPM correction model."""

    def __init__(self):
        super().__init__()
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

    def _build_model(self, input_dim: int, hidden_layers: Tuple[int, ...] = (128, 64, 32)) -> keras.Model:
        """Build neural network architecture."""
        _, keras, layers = _get_tensorflow()

        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers with batch norm and dropout
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            if i < len(hidden_layers) - 1:
                dropout_rate = 0.3 - (i * 0.1)  # Decreasing dropout
                model.add(layers.Dropout(dropout_rate))

        # Output layer
        model.add(layers.Dense(1))

        return model

    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              epochs: int = 200,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              hidden_layers: Tuple[int, ...] = (128, 64, 32),
              **kwargs) -> Dict[str, Any]:
        """
        Train neural network model.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            hidden_layers: Hidden layer sizes

        Returns:
            Training results with history
        """
        logger.info("Training Neural Network BPM correction model...")

        _, keras, _ = _get_tensorflow()

        self.feature_names = X.columns.tolist()

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build model
        self.model = self._build_model(X.shape[1], hidden_layers)

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10
            )
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val.values)

        # Train
        history = self.model.fit(
            X_scaled, y.values,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Results
        results = {
            'history': history.history,
            'final_train_loss': history.history['loss'][-1],
        }

        if validation_data is not None:
            results['final_val_loss'] = history.history['val_loss'][-1]
            logger.info(f"Validation MAE: {results['final_val_loss']:.2f} BPM")

        logger.info(f"Training MAE: {results['final_train_loss']:.2f} BPM")
        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()


class EnsembleBPMModel(BPMCorrectionModel):
    """Ensemble model combining XGBoost and Neural Network."""

    def __init__(self, xgb_weight: float = 0.6, nn_weight: float = 0.4):
        """
        Initialize ensemble model.

        Args:
            xgb_weight: Weight for XGBoost predictions
            nn_weight: Weight for Neural Network predictions
        """
        super().__init__()
        self.xgb_model = XGBoostBPMModel()
        self.nn_model = NeuralBPMModel()
        self.xgb_weight = xgb_weight
        self.nn_weight = nn_weight

    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              **kwargs) -> Dict[str, Any]:
        """Train both models."""
        logger.info("Training Ensemble BPM correction model...")

        # Train XGBoost
        xgb_results = self.xgb_model.train(X, y, X_val, y_val, **kwargs)

        # Train Neural Network
        nn_results = self.nn_model.train(X, y, X_val, y_val, **kwargs)

        # Evaluate ensemble
        if X_val is not None and y_val is not None:
            y_pred_ensemble = self.predict(X_val)
            ensemble_mae = np.mean(np.abs(y_val - y_pred_ensemble))
        else:
            y_pred_ensemble = self.predict(X)
            ensemble_mae = np.mean(np.abs(y - y_pred_ensemble))

        results = {
            'xgb_results': xgb_results,
            'nn_results': nn_results,
            'ensemble_mae': ensemble_mae
        }

        logger.info(f"Ensemble MAE: {ensemble_mae:.2f} BPM")
        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        xgb_pred = self.xgb_model.predict(X)
        nn_pred = self.nn_model.predict(X)

        # Weighted average
        ensemble_pred = self.xgb_weight * xgb_pred + self.nn_weight * nn_pred
        return ensemble_pred

    def save(self, path: str):
        """Save ensemble model."""
        path = Path(path)
        base_path = path.parent / path.stem

        # Save individual models
        self.xgb_model.save(str(base_path) + '_xgb.pkl')
        self.nn_model.save(str(base_path) + '_nn.pkl')

        # Save weights
        weights = {
            'xgb_weight': self.xgb_weight,
            'nn_weight': self.nn_weight
        }
        with open(str(base_path) + '_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)

        logger.info(f"Ensemble model saved to {base_path}_*")

    def load(self, path: str):
        """Load ensemble model."""
        path = Path(path)
        base_path = path.parent / path.stem

        # Load individual models
        self.xgb_model.load(str(base_path) + '_xgb.pkl')
        self.nn_model.load(str(base_path) + '_nn.pkl')

        # Load weights
        with open(str(base_path) + '_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        self.xgb_weight = weights['xgb_weight']
        self.nn_weight = weights['nn_weight']

        logger.info(f"Ensemble model loaded from {base_path}_*")
