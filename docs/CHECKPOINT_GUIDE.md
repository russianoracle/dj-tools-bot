# Checkpoint & Resume Training Guide

## Схема сохранения промежуточных точек обучения

### 1. Автоматические Checkpoints

Система автоматически сохраняет:

#### **XGBoost:**
```
models/bpm_correctors/checkpoints/
├── xgboost_epoch_100_20250123_103045.pkl
├── xgboost_epoch_200_20250123_104012.pkl
├── xgboost_epoch_300_20250123_104938.pkl  ← финальная
├── checkpoint_metadata.json
├── training_history.json
└── features.pkl  ← кэш фичей (один раз)
```

**Частота:** каждые 100 деревьев (n_estimators)

#### **Neural Network:**
```
models/bpm_correctors/checkpoints/
├── neural_network_epoch_20_20250123_110521.pkl
├── neural_network_epoch_40_20250123_111205.pkl
├── neural_network_epoch_60_20250123_111849.pkl
├── neural_network_best.pkl  ← лучшая по val_loss
└── ...
```

**Частота:** каждые 20 эпох + лучшая модель (EarlyStopping)

---

### 2. Структура Checkpoint

#### checkpoint_metadata.json
```json
{
  "xgboost_epoch_300_20250123": {
    "epoch": 300,
    "algorithm": "xgboost",
    "metrics": {
      "train_mae": 1.85,
      "val_mae": 1.92
    },
    "timestamp": "20250123_104938",
    "model_path": "models/.../xgboost_epoch_300_20250123.pkl"
  },
  "neural_network_epoch_60_20250123": {
    "epoch": 60,
    "algorithm": "neural_network",
    "metrics": {
      "train_loss": 1.23,
      "val_loss": 1.31
    },
    "timestamp": "20250123_111849",
    "model_path": "models/.../neural_network_epoch_60_20250123.pkl"
  }
}
```

#### training_history.json
```json
{
  "xgboost": {
    "epochs": [100, 200, 300],
    "metrics": {
      "train_mae": [2.15, 1.98, 1.85],
      "val_mae": [2.21, 2.05, 1.92]
    }
  },
  "neural_network": {
    "epochs": [20, 40, 60, 80, 100],
    "metrics": {
      "loss": [2.45, 1.89, 1.52, 1.31, 1.25],
      "val_loss": [2.51, 1.95, 1.58, 1.35, 1.31]
    }
  }
}
```

---

### 3. Использование в коде

#### Базовое обучение с checkpoints:

```python
from src.training import BPMTrainer, CheckpointManager

# Создание менеджера checkpoints
checkpoint_manager = CheckpointManager("models/bpm_correctors/checkpoints")

# Обучение с автоматическими checkpoints
trainer = BPMTrainer("test_data_2000.txt")
trainer.run_full_training_pipeline(
    algorithms=['xgboost', 'neural_network'],
    save_dir='models/bpm_correctors',
    checkpoint_manager=checkpoint_manager,  # ← включить checkpoints
    checkpoint_frequency=20  # сохранять каждые 20 итераций/эпох
)
```

#### Возобновление обучения после прерывания:

```python
from src.training import BPMTrainer, CheckpointManager, TrainingResumer

# Менеджер и resumer
checkpoint_manager = CheckpointManager("models/bpm_correctors/checkpoints")
resumer = TrainingResumer(checkpoint_manager)

# Проверка возможности возобновления
if resumer.can_resume('xgboost'):
    print("Найден checkpoint, продолжаем обучение...")

    trainer = BPMTrainer("test_data_2000.txt")

    # Автоматическое возобновление
    trainer.run_full_training_pipeline(
        algorithms=['xgboost'],
        resume=True,  # ← возобновить с последнего checkpoint
        checkpoint_manager=checkpoint_manager
    )
else:
    print("Начинаем обучение с нуля...")
```

#### Ручное управление checkpoints:

```python
from src.training.models import XGBoostBPMModel

# Загрузка конкретного checkpoint
model = XGBoostBPMModel()
checkpoint_manager.load_checkpoint(
    "models/.../xgboost_epoch_200.pkl",
    model
)

# Продолжение обучения
model.train(X_train, y_train, X_val, y_val, n_estimators=100)  # еще 100 деревьев

# Сохранение нового checkpoint
checkpoint_manager.save_checkpoint(
    model, epoch=300,
    metrics={'val_mae': 1.8},
    algorithm='xgboost'
)
```

#### Получение лучшей модели:

```python
# Автоматический выбор checkpoint с лучшей метрикой
best_checkpoint = checkpoint_manager.get_best_checkpoint(
    algorithm='neural_network',
    metric='val_loss'  # минимизировать val_loss
)

if best_checkpoint:
    model = NeuralBPMModel()
    checkpoint_manager.load_checkpoint(best_checkpoint['model_path'], model)
    print(f"Загружена лучшая модель: val_loss={best_checkpoint['metrics']['val_loss']:.3f}")
```

---

### 4. CLI поддержка

#### Обучение с checkpoints:

```bash
# Базовое обучение (автоматически создает checkpoints)
python scripts/train_bpm_corrector.py data.txt \
    --algorithms xgboost neural_network \
    --checkpoint-dir models/checkpoints \
    --checkpoint-freq 20

# Возобновление прерванного обучения
python scripts/train_bpm_corrector.py data.txt \
    --resume \
    --checkpoint-dir models/checkpoints

# Продолжение с конкретного checkpoint
python scripts/train_bpm_corrector.py data.txt \
    --resume-from models/checkpoints/xgboost_epoch_200.pkl
```

#### Управление checkpoints:

```bash
# Очистка старых checkpoints (оставить последние 3)
python scripts/manage_checkpoints.py --cleanup \
    --checkpoint-dir models/checkpoints \
    --keep-last 3

# Показать историю обучения
python scripts/manage_checkpoints.py --show-history \
    --checkpoint-dir models/checkpoints \
    --algorithm xgboost

# Найти лучший checkpoint
python scripts/manage_checkpoints.py --best \
    --checkpoint-dir models/checkpoints \
    --algorithm neural_network \
    --metric val_loss
```

---

### 5. GUI поддержка

В окне обучения:

**Автоматические checkpoints:**
- ✅ Включены по умолчанию
- 📁 Сохраняются в `models/bpm_correctors/checkpoints/`
- 🔄 Частота: каждые 20 эпох/итераций

**Кнопки управления:**
- **Resume Training** - продолжить с последнего checkpoint
- **Clear Checkpoints** - удалить старые checkpoint'ы
- **View History** - график метрик по эпохам

**Индикаторы:**
```
Training Progress:
[████████████████░░░░] 60/100 epochs

Last Checkpoint: epoch 60 (val_loss: 1.31)
Best Checkpoint: epoch 55 (val_loss: 1.29)

Time elapsed: 45 min
Estimated remaining: 30 min
```

---

### 6. Кэширование Features

**Проблема:** Извлечение фичей из 2000 треков занимает ~1-2 часа.

**Решение:** Автоматическое кэширование:

```python
# Первый запуск - извлекает и сохраняет фичи
trainer.run_full_training_pipeline(...)
# Сохраняет: models/.../checkpoints/features.pkl

# Последующие запуски - загружает из кэша
trainer.run_full_training_pipeline(...)
# Загружает: models/.../checkpoints/features.pkl (мгновенно!)
```

**Очистка кэша:**
```bash
rm models/bpm_correctors/checkpoints/features.pkl
```

---

### 7. Схема работы при сбое

**Сценарий:** Обучение прервалось на 1500-м треке из 2000.

```
1. Извлечение фичей:
   [████████████████████] 2000/2000 треков (1.5 часа)
   ✅ Сохранено: features.pkl

2. Обучение XGBoost:
   [████████████████████] 300/300 деревьев (5 минут)
   ✅ Сохранено: xgboost_epoch_300.pkl

3. Обучение Neural Network:
   [████████████░░░░░░░░] 60/100 эпох
   ⚠️  ПРЕРЫВАНИЕ (crash, Ctrl+C, etc.)
   ✅ Последний checkpoint: neural_network_epoch_60.pkl

4. Возобновление:
   $ python scripts/train_bpm_corrector.py data.txt --resume

   ✅ Загружено: features.pkl (мгновенно, не пересчитывает!)
   ✅ XGBoost уже обучен, пропущен
   ✅ Neural Network: продолжение с эпохи 61
   [████████████████████] 61-100/100 эпох (осталось 40%)
```

**Экономия времени:** 1.5 часа (features) + 5 минут (XGBoost) + 36 минут (NN 0-60) = **1 час 41 минута**

---

### 8. Best Practices

#### ✅ Рекомендуемые настройки:

```python
# Для длительного обучения (2000+ треков)
checkpoint_frequency = 20       # каждые 20 эпох
keep_last_checkpoints = 5       # хранить последние 5
auto_cleanup = True             # автоочистка старых

# Для быстрых экспериментов (100-500 треков)
checkpoint_frequency = 50       # реже
keep_last_checkpoints = 2       # меньше места
```

#### ⚠️ Внимание:

- **Checkpoint'ы занимают место:** ~50-100 MB каждый
- **Features.pkl:** ~500 MB для 2000 треков
- **Рекомендуется:** периодическая очистка старых checkpoint'ов

#### 🔧 Оптимизация места:

```python
# Автоматическая очистка после успешного обучения
checkpoint_manager.cleanup_old_checkpoints(
    algorithm='xgboost',
    keep_last=1  # оставить только финальную модель
)
```

---

### 9. Мониторинг в real-time

GUI показывает:

```
═══════════════════════════════════════════════════════
Neural Network Training - Epoch 65/100

Current Metrics:
  Train Loss: 1.24
  Val Loss:   1.31  ← лучшая: 1.29 (epoch 55)

Checkpoints:
  ✅ Last saved: epoch 60 (5 epochs ago)
  ⏰ Next save:  epoch 80 (15 epochs)

Memory:
  Checkpoint size: 85.2 MB
  Total checkpoints: 187.5 MB (3 saved)
  Features cache: 523.1 MB
═══════════════════════════════════════════════════════
```

---

### 10. Troubleshooting

**Q: Checkpoint не создается?**
```bash
# Проверить права доступа
ls -la models/bpm_correctors/checkpoints/

# Проверить место на диске
df -h
```

**Q: Не находит checkpoint для resume?**
```python
# Проверить наличие
checkpoint_manager.get_latest_checkpoint('xgboost')
# Если None - checkpoint'ов нет
```

**Q: Как удалить все checkpoint'ы?**
```bash
rm -rf models/bpm_correctors/checkpoints/*
```

**Q: Features.pkl устарел (изменилась feature extraction)?**
```bash
# Удалить кэш
rm models/bpm_correctors/checkpoints/features.pkl
# При следующем запуске пересчитается
```

---

## Итого

**Схема checkpoint'ов обеспечивает:**

✅ **Защита от потери прогресса** - можно возобновить в любой момент
✅ **Кэширование фичей** - не пересчитывать 1-2 часа работы
✅ **Выбор лучшей модели** - автоматически по метрикам
✅ **История обучения** - графики и JSON
✅ **Управление памятью** - автоочистка старых checkpoint'ов
✅ **CLI & GUI** - полная поддержка в обоих интерфейсах

**Для 2000+ треков это критически важно!** 🚀
