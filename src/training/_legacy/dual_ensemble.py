"""
Dual-Source Ensemble Classifier

Комбинирует DEAM-trained и User-trained модели для классификации зон.

DEAM модель: хорошо различает YELLOW/GREEN/PURPLE (сбалансированный датасет)
User модель: специализирована на DJ-треках (преимущественно GREEN)
"""

import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger

logger = get_logger(__name__)


class DualSourceEnsembleClassifier:
    """
    Ensemble из двух моделей: DEAM и User.

    Комбинирует вероятности через взвешенное усреднение:
    P(zone) = w_deam * P_deam(zone) + w_user * P_user(zone)

    Attributes:
        deam_model: XGBoost модель, обученная на DEAM
        user_model: XGBoost модель, обученная на user tracks
        deam_scaler: StandardScaler для DEAM модели
        user_scaler: StandardScaler для User модели
        weights: [w_deam, w_user] - веса моделей (сумма = 1.0)
        feature_names: Список имён фичей
    """

    ZONE_NAMES = ['yellow', 'green', 'purple']

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize ensemble.

        Args:
            weights: [deam_weight, user_weight]. По умолчанию [0.7, 0.3]
                     DEAM доминирует, так как знает YELLOW/PURPLE
        """
        self.deam_model = None
        self.user_model = None
        self.deam_scaler = None
        self.user_scaler = None
        self.weights = weights or [0.7, 0.3]
        self.feature_names = None

    def load_models(self,
                   deam_dir: str = "models/deam_zone_classifier_19f",
                   user_dir: str = "models/user_zone_classifier"):
        """
        Загрузить обе модели.

        Args:
            deam_dir: Директория с DEAM моделью
            user_dir: Директория с User моделью
        """
        deam_path = Path(deam_dir)
        user_path = Path(user_dir)

        # Load DEAM model
        logger.info(f"Loading DEAM model from {deam_path}")

        deam_model_file = deam_path / "xgboost_model.pkl"
        if deam_model_file.exists():
            with open(deam_model_file, 'rb') as f:
                loaded = pickle.load(f)
                # Handle dict format from XGBoostZoneClassifier.save()
                if isinstance(loaded, dict) and 'model' in loaded:
                    self.deam_model = loaded['model']
                else:
                    self.deam_model = loaded
        else:
            # Try JSON format
            self.deam_model = xgb.Booster()
            self.deam_model.load_model(str(deam_path / "xgboost_model.json"))

        with open(deam_path / "scaler.pkl", 'rb') as f:
            self.deam_scaler = pickle.load(f)

        with open(deam_path / "feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)

        # Load User model
        logger.info(f"Loading User model from {user_path}")

        user_model_file = user_path / "xgboost_model.json"
        self.user_model = xgb.Booster()
        self.user_model.load_model(str(user_model_file))

        with open(user_path / "scaler.pkl", 'rb') as f:
            self.user_scaler = pickle.load(f)

        logger.info(f"Both models loaded. Features: {len(self.feature_names)}")
        logger.info(f"Weights: DEAM={self.weights[0]:.1%}, User={self.weights[1]:.1%}")

    def _get_model_proba(self, model, X_scaled: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from model, handling both Booster and XGBClassifier.

        Args:
            model: XGBoost model (Booster or XGBClassifier)
            X_scaled: Already normalized features

        Returns:
            Probability array (N, 3)
        """
        if isinstance(model, xgb.Booster):
            # Raw Booster - use DMatrix
            dmatrix = xgb.DMatrix(X_scaled)
            return model.predict(dmatrix)
        elif hasattr(model, 'predict_proba'):
            # sklearn-style (XGBClassifier)
            return model.predict_proba(X_scaled)
        else:
            # Fallback - try direct predict
            dmatrix = xgb.DMatrix(X_scaled)
            return model.predict(dmatrix)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказать вероятности зон.

        Args:
            X: Массив фичей (N, 19) - НЕ нормализованный

        Returns:
            Массив вероятностей (N, 3) для [yellow, green, purple]
        """
        # Normalize with each scaler
        X_deam = self.deam_scaler.transform(X)
        X_user = self.user_scaler.transform(X)

        # Get predictions from each model
        deam_proba = self._get_model_proba(self.deam_model, X_deam)
        user_proba = self._get_model_proba(self.user_model, X_user)

        # Weighted average
        ensemble_proba = (self.weights[0] * deam_proba +
                         self.weights[1] * user_proba)

        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказать зоны.

        Args:
            X: Массив фичей (N, 19) - НЕ нормализованный

        Returns:
            Массив меток зон (N,) - 0=yellow, 1=green, 2=purple
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Предсказать зоны с уверенностью.

        Returns:
            (predictions, confidences, all_probabilities)
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        return predictions, confidences, proba

    def optimize_weights(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        grid_step: float = 0.1) -> Tuple[List[float], float]:
        """
        Оптимизировать веса через grid search.

        Args:
            X: Validation features (НЕ нормализованные)
            y: True labels (0, 1, 2)
            grid_step: Шаг сетки поиска

        Returns:
            (best_weights, best_accuracy)
        """
        logger.info("Optimizing ensemble weights...")

        # Prepare normalized data
        X_deam = self.deam_scaler.transform(X)
        X_user = self.user_scaler.transform(X)

        # Get base predictions using helper
        deam_proba = self._get_model_proba(self.deam_model, X_deam)
        user_proba = self._get_model_proba(self.user_model, X_user)

        best_acc = 0
        best_weights = self.weights.copy()

        for w_deam in np.arange(0.0, 1.0 + grid_step/2, grid_step):
            w_user = 1.0 - w_deam

            proba = w_deam * deam_proba + w_user * user_proba
            preds = np.argmax(proba, axis=1)
            acc = np.mean(preds == y)

            logger.debug(f"  w_deam={w_deam:.1f}, w_user={w_user:.1f} -> acc={acc:.1%}")

            if acc > best_acc:
                best_acc = acc
                best_weights = [w_deam, w_user]

        self.weights = best_weights
        logger.info(f"Best weights: DEAM={best_weights[0]:.1%}, User={best_weights[1]:.1%}")
        logger.info(f"Best accuracy: {best_acc:.1%}")

        return best_weights, best_acc

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Оценить качество ensemble.

        Returns:
            Dict с метриками
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        predictions, confidences, proba = self.predict_with_confidence(X)

        acc = accuracy_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions, labels=[0, 1, 2])
        # Determine actual labels present
        actual_labels = sorted(set(y) | set(predictions))
        actual_names = [self.ZONE_NAMES[i] for i in actual_labels]

        report = classification_report(y, predictions,
                                       labels=actual_labels,
                                       target_names=actual_names,
                                       output_dict=True,
                                       zero_division=0)

        return {
            'accuracy': acc,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'mean_confidence': np.mean(confidences),
            'predictions': predictions,
            'probabilities': proba
        }

    def save(self, path: str):
        """Сохранить ensemble."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save weights and config
        config = {
            'weights': self.weights,
            'feature_names': self.feature_names
        }
        with open(save_path / "ensemble_config.pkl", 'wb') as f:
            pickle.dump(config, f)

        logger.info(f"Ensemble config saved to {save_path}")

    def load(self,
            config_path: str,
            deam_dir: str = "models/deam_zone_classifier_19f",
            user_dir: str = "models/user_zone_classifier"):
        """
        Загрузить ensemble.

        Args:
            config_path: Путь к ensemble_config.pkl
            deam_dir: Директория с DEAM моделью
            user_dir: Директория с User моделью
        """
        with open(Path(config_path) / "ensemble_config.pkl", 'rb') as f:
            config = pickle.load(f)

        self.weights = config['weights']
        self.feature_names = config['feature_names']

        self.load_models(deam_dir, user_dir)

    def get_model_contributions(self, X: np.ndarray) -> Dict:
        """
        Показать вклад каждой модели в предсказание.

        Returns:
            Dict с деталями вкладов
        """
        X_deam = self.deam_scaler.transform(X)
        X_user = self.user_scaler.transform(X)

        deam_proba = self._get_model_proba(self.deam_model, X_deam)
        user_proba = self._get_model_proba(self.user_model, X_user)

        ensemble_proba = self.weights[0] * deam_proba + self.weights[1] * user_proba

        return {
            'deam_proba': deam_proba,
            'user_proba': user_proba,
            'ensemble_proba': ensemble_proba,
            'deam_predictions': np.argmax(deam_proba, axis=1),
            'user_predictions': np.argmax(user_proba, axis=1),
            'ensemble_predictions': np.argmax(ensemble_proba, axis=1)
        }
