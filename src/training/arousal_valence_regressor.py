"""
Arousal-Valence Regressor

Регрессор для предсказания arousal и valence на основе аудио фичей.
Обучается на DEAM датасете и применяется к новым трекам.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from ..utils import get_logger

logger = get_logger(__name__)


class ArousalValenceRegressor:
    """
    Двухканальный регрессор для предсказания arousal и valence.

    Использует два независимых XGBoost регрессора:
    - arousal_model: предсказывает энергию [1.6, 8.1]
    - valence_model: предсказывает эмоциональную окраску [1.6, 8.4]
    """

    def __init__(self, model_dir: str = "models/arousal_valence"):
        """
        Initialize regressor.

        Args:
            model_dir: Директория для сохранения моделей
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Models (XGBoost regressors)
        self.arousal_model: Optional[xgb.XGBRegressor] = None
        self.valence_model: Optional[xgb.XGBRegressor] = None

        # Feature names (для валидации)
        self.feature_names: Optional[list] = None

        # Training history
        self.history: Dict = {
            'arousal': {},
            'valence': {}
        }

    def train(self,
             X_train: np.ndarray,
             y_arousal_train: np.ndarray,
             y_valence_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_arousal_val: Optional[np.ndarray] = None,
             y_valence_val: Optional[np.ndarray] = None,
             feature_names: Optional[list] = None,
             xgb_params: Optional[Dict] = None,
             progress_callback: Optional[Callable] = None,
             log_callback: Optional[Callable] = None) -> Dict:
        """
        Обучает оба регрессора (arousal и valence).

        Args:
            X_train: Training features (n_samples, n_features)
            y_arousal_train: Training arousal labels (n_samples,)
            y_valence_train: Training valence labels (n_samples,)
            X_val: Validation features
            y_arousal_val: Validation arousal labels
            y_valence_val: Validation valence labels
            feature_names: Названия фичей
            xgb_params: Параметры XGBoost (если None - используются default)
            progress_callback: Callback для прогресса (value, max_value, message)
            log_callback: Callback для логов (level, message)

        Returns:
            Dict с метриками обучения
        """
        self._log(log_callback, "INFO", "🎯 Starting Arousal-Valence Regressor Training")
        self._log(log_callback, "INFO", f"  Train samples: {len(X_train)}")
        self._log(log_callback, "INFO", f"  Features: {X_train.shape[1]}")

        if X_val is not None:
            self._log(log_callback, "INFO", f"  Val samples: {len(X_val)}")

        # Сохраняем feature names
        self.feature_names = feature_names

        # Default XGBoost parameters (оптимизированы для регрессии)
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }

        self._log(log_callback, "INFO", f"\n⚙️  XGBoost parameters:")
        for key, value in xgb_params.items():
            self._log(log_callback, "INFO", f"  {key}: {value}")

        # === TRAIN AROUSAL MODEL ===

        self._log(log_callback, "INFO", "\n📊 Training Arousal Regressor...")
        self._progress(progress_callback, 0, 2, "Training arousal model...")

        self.arousal_model = xgb.XGBRegressor(**xgb_params)

        # Prepare eval set
        eval_set_arousal = None
        if X_val is not None and y_arousal_val is not None:
            eval_set_arousal = [(X_train, y_arousal_train), (X_val, y_arousal_val)]

        # Train
        self.arousal_model.fit(
            X_train, y_arousal_train,
            eval_set=eval_set_arousal,
            verbose=False
        )

        # Evaluate arousal
        arousal_metrics = self._evaluate_model(
            self.arousal_model,
            X_train, y_arousal_train,
            X_val, y_arousal_val,
            "Arousal",
            log_callback
        )

        self.history['arousal'] = arousal_metrics

        # === TRAIN VALENCE MODEL ===

        self._log(log_callback, "INFO", "\n📊 Training Valence Regressor...")
        self._progress(progress_callback, 1, 2, "Training valence model...")

        self.valence_model = xgb.XGBRegressor(**xgb_params)

        # Prepare eval set
        eval_set_valence = None
        if X_val is not None and y_valence_val is not None:
            eval_set_valence = [(X_train, y_valence_train), (X_val, y_valence_val)]

        # Train
        self.valence_model.fit(
            X_train, y_valence_train,
            eval_set=eval_set_valence,
            verbose=False
        )

        # Evaluate valence
        valence_metrics = self._evaluate_model(
            self.valence_model,
            X_train, y_valence_train,
            X_val, y_valence_val,
            "Valence",
            log_callback
        )

        self.history['valence'] = valence_metrics

        self._progress(progress_callback, 2, 2, "Training completed!")

        # === SUMMARY ===

        self._log(log_callback, "INFO", "\n✅ Training Completed!")
        self._log(log_callback, "INFO", f"\n📈 Final Metrics:")
        self._log(log_callback, "INFO", f"  Arousal (val):  MAE={arousal_metrics['val_mae']:.3f}, RMSE={arousal_metrics['val_rmse']:.3f}, R²={arousal_metrics['val_r2']:.3f}")
        self._log(log_callback, "INFO", f"  Valence (val):  MAE={valence_metrics['val_mae']:.3f}, RMSE={valence_metrics['val_rmse']:.3f}, R²={valence_metrics['val_r2']:.3f}")

        return {
            'arousal': arousal_metrics,
            'valence': valence_metrics,
            'feature_names': self.feature_names
        }

    def _evaluate_model(self,
                       model: xgb.XGBRegressor,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: Optional[np.ndarray],
                       y_val: Optional[np.ndarray],
                       name: str,
                       log_callback: Optional[Callable]) -> Dict:
        """Evaluate single model (arousal or valence)."""

        # Train predictions
        y_train_pred = model.predict(X_train)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }

        self._log(log_callback, "INFO", f"  Train: MAE={train_mae:.3f}, RMSE={train_rmse:.3f}, R²={train_r2:.3f}")

        # Val predictions
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)

            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_r2 = r2_score(y_val, y_val_pred)

            metrics.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })

            self._log(log_callback, "INFO", f"  Val:   MAE={val_mae:.3f}, RMSE={val_rmse:.3f}, R²={val_r2:.3f}")

        return metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает arousal и valence для новых треков.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Tuple (arousal_predictions, valence_predictions)
            - arousal_predictions: array of shape (n_samples,)
            - valence_predictions: array of shape (n_samples,)
        """
        if self.arousal_model is None or self.valence_model is None:
            raise ValueError("Models not trained. Call train() or load() first.")

        arousal_pred = self.arousal_model.predict(X)
        valence_pred = self.valence_model.predict(X)

        # Clip to DEAM ranges
        arousal_pred = np.clip(arousal_pred, 1.6, 8.1)
        valence_pred = np.clip(valence_pred, 1.6, 8.4)

        return arousal_pred, valence_pred

    def predict_with_zones(self,
                          X: np.ndarray,
                          yellow_arousal: float = 4.0,
                          purple_arousal: float = 6.0,
                          valence_threshold: float = 4.5) -> pd.DataFrame:
        """
        Предсказывает arousal, valence И зоны.

        Args:
            X: Features (n_samples, n_features)
            yellow_arousal: Порог для Yellow зоны
            purple_arousal: Порог для Purple зоны
            valence_threshold: Порог negative/positive

        Returns:
            DataFrame with columns: arousal, valence, zone
        """
        arousal_pred, valence_pred = self.predict(X)

        # Map to zones
        zones = []
        for a, v in zip(arousal_pred, valence_pred):
            if a < yellow_arousal:
                zone = 'YELLOW_CHILL' if v >= valence_threshold else 'YELLOW_DARK'
            elif a > purple_arousal:
                zone = 'PURPLE_EUPHORIC' if v >= valence_threshold else 'PURPLE_AGGRESSIVE'
            else:
                zone = 'GREEN'
            zones.append(zone)

        return pd.DataFrame({
            'arousal': arousal_pred,
            'valence': valence_pred,
            'zone': zones
        })

    def save(self, suffix: str = ""):
        """
        Сохраняет обе модели и метаданные.

        Args:
            suffix: Суффикс для имени файла (например, "v1", "best")
        """
        if self.arousal_model is None or self.valence_model is None:
            raise ValueError("Models not trained. Nothing to save.")

        suffix_str = f"_{suffix}" if suffix else ""

        # Save arousal model
        arousal_path = self.model_dir / f"arousal_model{suffix_str}.pkl"
        with open(arousal_path, 'wb') as f:
            pickle.dump(self.arousal_model, f)
        logger.info(f"💾 Saved arousal model: {arousal_path}")

        # Save valence model
        valence_path = self.model_dir / f"valence_model{suffix_str}.pkl"
        with open(valence_path, 'wb') as f:
            pickle.dump(self.valence_model, f)
        logger.info(f"💾 Saved valence model: {valence_path}")

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'history': self.history,
            'arousal_range': [1.6, 8.1],
            'valence_range': [1.6, 8.4]
        }

        metadata_path = self.model_dir / f"metadata{suffix_str}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"💾 Saved metadata: {metadata_path}")

        logger.info(f"✅ All models saved to: {self.model_dir}")

    def load(self, suffix: str = ""):
        """
        Загружает обе модели и метаданные.

        Args:
            suffix: Суффикс имени файла
        """
        suffix_str = f"_{suffix}" if suffix else ""

        # Load arousal model
        arousal_path = self.model_dir / f"arousal_model{suffix_str}.pkl"
        if not arousal_path.exists():
            raise FileNotFoundError(f"Arousal model not found: {arousal_path}")

        with open(arousal_path, 'rb') as f:
            self.arousal_model = pickle.load(f)
        logger.info(f"✅ Loaded arousal model: {arousal_path}")

        # Load valence model
        valence_path = self.model_dir / f"valence_model{suffix_str}.pkl"
        if not valence_path.exists():
            raise FileNotFoundError(f"Valence model not found: {valence_path}")

        with open(valence_path, 'rb') as f:
            self.valence_model = pickle.load(f)
        logger.info(f"✅ Loaded valence model: {valence_path}")

        # Load metadata
        metadata_path = self.model_dir / f"metadata{suffix_str}.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.feature_names = metadata.get('feature_names')
            self.history = metadata.get('history', {'arousal': {}, 'valence': {}})
            logger.info(f"✅ Loaded metadata: {metadata_path}")

        logger.info(f"✅ All models loaded from: {self.model_dir}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Возвращает важность фичей для обеих моделей.

        Returns:
            DataFrame with columns: feature, arousal_importance, valence_importance
        """
        if self.arousal_model is None or self.valence_model is None:
            raise ValueError("Models not trained.")

        # Get importance
        arousal_imp = self.arousal_model.feature_importances_
        valence_imp = self.valence_model.feature_importances_

        # Create dataframe
        df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'f{i}' for i in range(len(arousal_imp))],
            'arousal_importance': arousal_imp,
            'valence_importance': valence_imp
        })

        # Sort by average importance
        df['avg_importance'] = (df['arousal_importance'] + df['valence_importance']) / 2
        df = df.sort_values('avg_importance', ascending=False)

        return df[['feature', 'arousal_importance', 'valence_importance', 'avg_importance']]

    def _log(self, callback: Optional[Callable], level: str, message: str):
        """Helper для логирования."""
        if callback:
            callback(level, message)
        else:
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(message)

    def _progress(self, callback: Optional[Callable], value: int, max_value: int, message: str):
        """Helper для прогресса."""
        if callback:
            callback(value, max_value, message)


def main():
    """Test function."""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 19)
    y_arousal_train = np.random.uniform(1.6, 8.1, 100)
    y_valence_train = np.random.uniform(1.6, 8.4, 100)

    X_val = np.random.randn(20, 19)
    y_arousal_val = np.random.uniform(1.6, 8.1, 20)
    y_valence_val = np.random.uniform(1.6, 8.4, 20)

    # Train
    regressor = ArousalValenceRegressor(model_dir="models/arousal_valence_test")

    metrics = regressor.train(
        X_train, y_arousal_train, y_valence_train,
        X_val, y_arousal_val, y_valence_val,
        feature_names=[f'feature_{i}' for i in range(19)]
    )

    print(f"\n✅ Training completed!")
    print(f"  Arousal R²: {metrics['arousal']['val_r2']:.3f}")
    print(f"  Valence R²: {metrics['valence']['val_r2']:.3f}")

    # Save
    regressor.save()

    # Load
    regressor2 = ArousalValenceRegressor(model_dir="models/arousal_valence_test")
    regressor2.load()

    # Predict
    X_test = np.random.randn(5, 19)
    arousal_pred, valence_pred = regressor2.predict(X_test)

    print(f"\n🎯 Predictions:")
    for i, (a, v) in enumerate(zip(arousal_pred, valence_pred)):
        print(f"  Track {i+1}: arousal={a:.2f}, valence={v:.2f}")

    # Predict with zones
    df = regressor2.predict_with_zones(X_test)
    print(f"\n🎨 Predictions with zones:")
    print(df)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
