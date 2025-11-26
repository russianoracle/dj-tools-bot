#!/usr/bin/env python3
"""
Train Arousal-Valence Regressor on DEAM Dataset

Обучает двухканальный XGBoost регрессор для предсказания arousal и valence.
Использует готовые train/val/test splits из dataset/deam_processed/.
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Настройка кириллицы для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.arousal_valence_regressor import ArousalValenceRegressor


def load_deam_data(deam_dir: str = "dataset/deam_processed") -> Tuple[np.ndarray, ...]:
    """
    Загружает DEAM датасет из готовых splits.

    Returns:
        Tuple (X_train, y_arousal_train, y_valence_train,
               X_val, y_arousal_val, y_valence_val,
               X_test, y_arousal_test, y_valence_test,
               feature_names)
    """
    deam_dir = Path(deam_dir)

    print(f"📂 Loading DEAM dataset from: {deam_dir}")

    if not deam_dir.exists():
        raise FileNotFoundError(
            f"DEAM directory not found: {deam_dir}\n"
            f"Please run: python scripts/prepare_deam_dataset.py"
        )

    # Load splits
    train_df = pd.read_csv(deam_dir / "train.csv")
    val_df = pd.read_csv(deam_dir / "val.csv")
    test_df = pd.read_csv(deam_dir / "test.csv")

    print(f"✅ Loaded splits:")
    print(f"  Train: {len(train_df)} tracks")
    print(f"  Val:   {len(val_df)} tracks")
    print(f"  Test:  {len(test_df)} tracks")

    # Определяем колонки фичей - используем только 19 базовых аудио фичей
    # Исключаем 8 DEAM-специфичных фичей с data leakage (96.7% NaN, корреляция 0.951 с целевой)
    feature_columns = [
        'tempo',
        'zero_crossing_rate',
        'rms_energy',
        'spectral_centroid',
        'spectral_rolloff',
        'energy_variance',
        'mfcc_1_mean', 'mfcc_1_std',
        'mfcc_2_mean', 'mfcc_2_std',
        'mfcc_3_mean', 'mfcc_3_std',
        'mfcc_4_mean', 'mfcc_4_std',
        'mfcc_5_mean', 'mfcc_5_std',
        'low_energy',
        'brightness',
        'drop_intensity'
    ]

    print(f"\n📊 Features: {len(feature_columns)} (19 базовых аудио фичей)")
    print(f"  {', '.join(feature_columns[:5])}...")
    print(f"\n⚠️  Исключены 8 DEAM-фичей с data leakage:")
    print(f"     arousal_max_mean, arousal_max_std, arousal_min_mean, arousal_min_std")
    print(f"     valence_max_mean, valence_max_std, valence_min_mean, valence_min_std")

    # Extract X (features)
    X_train = train_df[feature_columns].values
    X_val = val_df[feature_columns].values
    X_test = test_df[feature_columns].values

    # Extract y (arousal + valence)
    y_arousal_train = train_df['arousal'].values
    y_valence_train = train_df['valence'].values

    y_arousal_val = val_df['arousal'].values
    y_valence_val = val_df['valence'].values

    y_arousal_test = test_df['arousal'].values
    y_valence_test = test_df['valence'].values

    # Статистика
    print(f"\n📈 Arousal range:")
    print(f"  Train: [{y_arousal_train.min():.2f}, {y_arousal_train.max():.2f}], mean={y_arousal_train.mean():.2f}")
    print(f"  Val:   [{y_arousal_val.min():.2f}, {y_arousal_val.max():.2f}], mean={y_arousal_val.mean():.2f}")
    print(f"  Test:  [{y_arousal_test.min():.2f}, {y_arousal_test.max():.2f}], mean={y_arousal_test.mean():.2f}")

    print(f"\n📈 Valence range:")
    print(f"  Train: [{y_valence_train.min():.2f}, {y_valence_train.max():.2f}], mean={y_valence_train.mean():.2f}")
    print(f"  Val:   [{y_valence_val.min():.2f}, {y_valence_val.max():.2f}], mean={y_valence_val.mean():.2f}")
    print(f"  Test:  [{y_valence_test.min():.2f}, {y_valence_test.max():.2f}], mean={y_valence_test.mean():.2f}")

    return (X_train, y_arousal_train, y_valence_train,
            X_val, y_arousal_val, y_valence_val,
            X_test, y_arousal_test, y_valence_test,
            feature_columns)


def visualize_predictions(y_true_arousal: np.ndarray,
                         y_pred_arousal: np.ndarray,
                         y_true_valence: np.ndarray,
                         y_pred_valence: np.ndarray,
                         output_dir: str = "results"):
    """
    Визуализирует предсказания vs ground truth.

    Args:
        y_true_arousal: Ground truth arousal
        y_pred_arousal: Predicted arousal
        y_true_valence: Ground truth valence
        y_pred_valence: Predicted valence
        output_dir: Директория для сохранения
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # === AROUSAL ===

    ax = axes[0]

    # Scatter plot
    ax.scatter(y_true_arousal, y_pred_arousal, alpha=0.5, s=20, c='blue', edgecolors='none')

    # Perfect prediction line
    min_val = min(y_true_arousal.min(), y_pred_arousal.min())
    max_val = max(y_true_arousal.max(), y_pred_arousal.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное предсказание')

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true_arousal, y_pred_arousal)
    rmse = np.sqrt(mean_squared_error(y_true_arousal, y_pred_arousal))
    r2 = r2_score(y_true_arousal, y_pred_arousal)

    ax.text(0.05, 0.95,
            f'MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR² = {r2:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12)

    ax.set_xlabel('Ground Truth Arousal', fontsize=13, weight='bold')
    ax.set_ylabel('Predicted Arousal', fontsize=13, weight='bold')
    ax.set_title('Arousal Prediction Quality', fontsize=15, weight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # === VALENCE ===

    ax = axes[1]

    # Scatter plot
    ax.scatter(y_true_valence, y_pred_valence, alpha=0.5, s=20, c='green', edgecolors='none')

    # Perfect prediction line
    min_val = min(y_true_valence.min(), y_pred_valence.min())
    max_val = max(y_true_valence.max(), y_pred_valence.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное предсказание')

    # Metrics
    mae = mean_absolute_error(y_true_valence, y_pred_valence)
    rmse = np.sqrt(mean_squared_error(y_true_valence, y_pred_valence))
    r2 = r2_score(y_true_valence, y_pred_valence)

    ax.text(0.05, 0.95,
            f'MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR² = {r2:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12)

    ax.set_xlabel('Ground Truth Valence', fontsize=13, weight='bold')
    ax.set_ylabel('Predicted Valence', fontsize=13, weight='bold')
    ax.set_title('Valence Prediction Quality', fontsize=15, weight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # === SAVE ===

    plt.tight_layout()
    output_path = output_dir / "arousal_valence_predictions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Prediction visualization saved: {output_path}")
    plt.close()


def visualize_feature_importance(regressor: ArousalValenceRegressor, output_dir: str = "results"):
    """
    Визуализирует важность фичей.

    Args:
        regressor: Trained regressor
        output_dir: Директория для сохранения
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get importance
    importance_df = regressor.get_feature_importance()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(importance_df))
    width = 0.35

    ax.barh(x - width/2, importance_df['arousal_importance'], width, label='Arousal', color='blue', alpha=0.7)
    ax.barh(x + width/2, importance_df['valence_importance'], width, label='Valence', color='green', alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=13, weight='bold')
    ax.set_title('Feature Importance for Arousal & Valence Prediction', fontsize=15, weight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature importance saved: {output_path}")
    plt.close()

    # Save CSV
    csv_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"✅ Feature importance CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Arousal-Valence Regressor on DEAM dataset'
    )
    parser.add_argument(
        '--deam-dir', type=str, default='dataset/deam_processed',
        help='DEAM processed directory (default: dataset/deam_processed)'
    )
    parser.add_argument(
        '--model-dir', type=str, default='models/arousal_valence',
        help='Model save directory (default: models/arousal_valence)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Output directory for visualizations (default: results)'
    )
    parser.add_argument(
        '--n-estimators', type=int, default=300,
        help='XGBoost n_estimators (default: 300)'
    )
    parser.add_argument(
        '--max-depth', type=int, default=6,
        help='XGBoost max_depth (default: 6)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.05,
        help='XGBoost learning rate (default: 0.05)'
    )

    args = parser.parse_args()

    try:
        print("=" * 80)
        print("🎯 AROUSAL-VALENCE REGRESSOR TRAINING")
        print("=" * 80)

        # 1. Load data
        (X_train, y_arousal_train, y_valence_train,
         X_val, y_arousal_val, y_valence_val,
         X_test, y_arousal_test, y_valence_test,
         feature_names) = load_deam_data(args.deam_dir)

        # 2. Create regressor
        print(f"\n🔧 Creating ArousalValenceRegressor...")
        regressor = ArousalValenceRegressor(model_dir=args.model_dir)

        # 3. Train
        print(f"\n🚀 Training regressor...")

        xgb_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
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

        metrics = regressor.train(
            X_train, y_arousal_train, y_valence_train,
            X_val, y_arousal_val, y_valence_val,
            feature_names=feature_names,
            xgb_params=xgb_params
        )

        # 4. Evaluate on test set
        print(f"\n📊 Evaluating on test set...")

        y_arousal_pred, y_valence_pred = regressor.predict(X_test)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Arousal metrics
        arousal_mae = mean_absolute_error(y_arousal_test, y_arousal_pred)
        arousal_rmse = np.sqrt(mean_squared_error(y_arousal_test, y_arousal_pred))
        arousal_r2 = r2_score(y_arousal_test, y_arousal_pred)

        # Valence metrics
        valence_mae = mean_absolute_error(y_valence_test, y_valence_pred)
        valence_rmse = np.sqrt(mean_squared_error(y_valence_test, y_valence_pred))
        valence_r2 = r2_score(y_valence_test, y_valence_pred)

        print(f"\n✅ Test Set Metrics:")
        print(f"  Arousal:  MAE={arousal_mae:.3f}, RMSE={arousal_rmse:.3f}, R²={arousal_r2:.3f}")
        print(f"  Valence:  MAE={valence_mae:.3f}, RMSE={valence_rmse:.3f}, R²={valence_r2:.3f}")

        # 5. Save model
        print(f"\n💾 Saving model...")
        regressor.save()

        # 6. Visualizations
        print(f"\n📊 Creating visualizations...")

        # Prediction quality
        visualize_predictions(
            y_arousal_test, y_arousal_pred,
            y_valence_test, y_valence_pred,
            output_dir=args.output_dir
        )

        # Feature importance
        visualize_feature_importance(regressor, output_dir=args.output_dir)

        # 7. Save summary report
        print(f"\n📄 Saving summary report...")

        report = {
            'dataset': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'n_features': len(feature_names)
            },
            'model_params': xgb_params,
            'metrics': {
                'arousal': {
                    'train_mae': metrics['arousal']['train_mae'],
                    'train_rmse': metrics['arousal']['train_rmse'],
                    'train_r2': metrics['arousal']['train_r2'],
                    'val_mae': metrics['arousal']['val_mae'],
                    'val_rmse': metrics['arousal']['val_rmse'],
                    'val_r2': metrics['arousal']['val_r2'],
                    'test_mae': arousal_mae,
                    'test_rmse': arousal_rmse,
                    'test_r2': arousal_r2
                },
                'valence': {
                    'train_mae': metrics['valence']['train_mae'],
                    'train_rmse': metrics['valence']['train_rmse'],
                    'train_r2': metrics['valence']['train_r2'],
                    'val_mae': metrics['valence']['val_mae'],
                    'val_rmse': metrics['valence']['val_rmse'],
                    'val_r2': metrics['valence']['val_r2'],
                    'test_mae': valence_mae,
                    'test_rmse': valence_rmse,
                    'test_r2': valence_r2
                }
            }
        }

        import json
        report_path = Path(args.output_dir) / "training_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved: {report_path}")

        print(f"\n" + "=" * 80)
        print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"=" * 80)
        print(f"\nModel saved to: {args.model_dir}")
        print(f"Visualizations saved to: {args.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Extract features from 415 user tracks")
        print(f"  2. Apply regressor to predict arousal/valence")
        print(f"  3. Visualize distribution compared to DEAM")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())