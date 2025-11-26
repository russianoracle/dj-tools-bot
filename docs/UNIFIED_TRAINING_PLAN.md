# Unified Training Plan: Frame-Based Dataset Integration

## Концепция

Объединение двух датасетов на **общих фреймовых признаках** для обучения единой модели классификации зон.

### Датасеты

| Датасет | Треков | Источник меток | Формат фичей |
|---------|--------|----------------|--------------|
| **User** | 484 | Ручная разметка (YELLOW/GREEN/PURPLE) | Полное извлечение |
| **DEAM** | 1802 | Arousal/Valence → маппинг в зоны | openSMILE (0.5s фреймы) |
| **Объединённый** | 2286 | Комбинированный | Общие фреймовые фичи |

---

## Текущий статус

### Реализовано

1. **Фреймовое извлечение** (`src/training/zone_features.py`)
   - `extract_frames()` - DataFrame с фичами по фреймам 0.5s
   - `extract_frames_fast()` - агрегация фреймов до уровня трека
   - Drop detection из фреймов

2. **Общие фичи (21 признак)**
   ```
   tempo, zero_crossing_rate, low_energy, rms_energy,
   spectral_rolloff, brightness, spectral_centroid,
   mfcc_1_mean..mfcc_5_mean, mfcc_1_std..mfcc_5_std,
   energy_variance, drop_strength,
   drop_contrast_mean, drop_count, drop_intensity
   ```

3. **Тестирование на 100+100 треках**
   - Accuracy: **67.5%**
   - User треки: drop_count=42, rms_energy=0.245
   - DEAM треки: drop_count=6, rms_energy=0.114
   - **Вывод**: User треки значительно энергичнее DEAM

### Кэшированные данные

- `models/checkpoints/features.pkl` - 483 User трека
- `models/user_zone_v3/` - текущая модель

---

## План обучения

### Этап 1: Подготовка данных

#### 1.1 Извлечь фреймовые фичи для DEAM

```bash
python scripts/extract_features.py \
  --input dataset/MEMD_audio \
  --output dataset/deam_frames.pkl \
  --method frames \
  --frame-size 0.5
```

**Результат**: Фреймовые фичи для 1802 DEAM треков

#### 1.2 Пересчитать фичи для User датасета

```bash
python scripts/extract_features.py \
  --input tests/test_data.txt \
  --output dataset/user_frames.pkl \
  --method frames \
  --frame-size 0.5
```

**Результат**: Фреймовые фичи для 484 User треков (совместимые с DEAM)

#### 1.3 Маппинг DEAM arousal → зоны

```python
# Текущий маппинг (config/zone_mapping.yaml)
def arousal_to_zone(arousal, valence):
    if arousal < 4.0:
        return 'YELLOW'  # Низкая энергия
    elif arousal > 6.0 and valence > 4.5:
        return 'PURPLE'  # Высокая энергия, позитив
    else:
        return 'GREEN'   # Переходная зона
```

---

### Этап 2: Объединение датасетов

#### 2.1 Создать unified dataset

```python
import pandas as pd

# Загрузить оба датасета
user_df = pd.read_pickle('dataset/user_frames.pkl')
deam_df = pd.read_pickle('dataset/deam_frames.pkl')

# Выбрать общие колонки (21 фича)
common_features = [
    'tempo', 'zero_crossing_rate', 'low_energy', 'rms_energy',
    'spectral_rolloff', 'brightness', 'spectral_centroid',
    'mfcc_1_mean', 'mfcc_1_std', 'mfcc_2_mean', 'mfcc_2_std',
    'mfcc_3_mean', 'mfcc_3_std', 'mfcc_4_mean', 'mfcc_4_std',
    'mfcc_5_mean', 'mfcc_5_std', 'energy_variance', 'drop_strength',
    'drop_contrast_mean', 'drop_count', 'drop_intensity'
]

# Объединить
unified_df = pd.concat([
    user_df[common_features + ['zone', 'source']],
    deam_df[common_features + ['zone', 'source']]
])
```

#### 2.2 Балансировка классов

| Проблема | Решение |
|----------|---------|
| DEAM перевешивает User (1802 vs 484) | Взвешенное обучение или undersampling DEAM |
| Разное распределение энергии | Стратификация по source + zone |
| Domain shift | Domain adaptation или отдельные веса |

**Стратегия 1: Взвешенное обучение**
```python
# Больший вес для User треков (они точнее размечены)
sample_weights = df['source'].map({'user': 3.0, 'deam': 1.0})
```

**Стратегия 2: Balanced sampling**
```python
# По 400 треков из каждого источника
user_sample = user_df.sample(n=400)
deam_sample = deam_df.sample(n=400)
balanced_df = pd.concat([user_sample, deam_sample])
```

---

### Этап 3: Обучение модели

#### 3.1 Baseline: Только User датасет

```bash
python scripts/retrain_on_new_labels.py \
  --data tests/test_data.txt \
  --output models/user_only_v4
```

**Ожидаемая точность**: ~70-75%

#### 3.2 Experiment 1: User + DEAM (weighted)

```bash
python scripts/retrain_on_new_labels.py \
  --data dataset/unified_weighted.pkl \
  --output models/unified_weighted_v1 \
  --user-weight 3.0
```

**Ожидаемая точность**: ~75-80%

#### 3.3 Experiment 2: Transfer learning

```python
# 1. Предобучить на DEAM
model_deam = train_on_deam(deam_df)

# 2. Fine-tune на User
model_final = finetune(model_deam, user_df, lr=0.01)
```

**Ожидаемая точность**: ~78-82%

#### 3.4 Experiment 3: Ensemble

```python
# Две модели + голосование
model_user = train(user_df)
model_unified = train(unified_df)

def predict(track):
    p1 = model_user.predict_proba(track)
    p2 = model_unified.predict_proba(track)
    return weighted_average(p1, p2, weights=[0.6, 0.4])
```

**Ожидаемая точность**: ~80-85%

---

### Этап 4: Валидация

#### 4.1 Holdout set

- 20% User треков (97 штук) - **никогда не использовать для обучения**
- Стратифицировать по зонам

#### 4.2 Cross-validation

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

#### 4.3 Метрики успеха

| Метрика | Цель | Текущее |
|---------|------|---------|
| Accuracy | > 80% | 67.5% |
| F1 (weighted) | > 0.78 | - |
| PURPLE recall | > 85% | - |
| User holdout accuracy | > 80% | - |

---

### Этап 5: Анализ ошибок

#### 5.1 Confusion matrix

```bash
python scripts/analyze_predictions.py \
  --model models/unified_v1 \
  --data tests/test_data.txt \
  --output results/confusion_matrix.png
```

#### 5.2 Анализ domain shift

```python
# Сравнить accuracy по источникам
user_acc = accuracy_score(y_user, model.predict(X_user))
deam_acc = accuracy_score(y_deam, model.predict(X_deam))

print(f"User accuracy: {user_acc:.2%}")
print(f"DEAM accuracy: {deam_acc:.2%}")
```

#### 5.3 Feature importance

```python
# XGBoost feature importance
importances = model.feature_importances_
top_features = sorted(zip(feature_names, importances),
                     key=lambda x: x[1], reverse=True)[:10]
```

---

## Ключевые наблюдения из тестов

### Разница между датасетами (100+100)

| Признак | User | DEAM | Ratio |
|---------|------|------|-------|
| drop_count | 42 | 6 | 7x |
| rms_energy | 0.245 | 0.114 | 2.2x |
| energy_variance | выше | ниже | - |
| brightness | выше | ниже | - |

**Вывод**: User треки (EDM/DJ) намного энергичнее и имеют больше дропов чем DEAM (разные жанры).

### Рекомендации

1. **Не объединять напрямую** - domain shift слишком большой
2. **Использовать DEAM для предобучения** - базовое понимание энергии
3. **Fine-tune на User** - адаптация к DJ-контенту
4. **Взвешивать User выше** - точнее размечен

---

## Команды для запуска

```bash
# 1. Извлечь фреймовые фичи
python scripts/extract_features.py --method frames

# 2. Переобучить на новых данных
python scripts/retrain_on_new_labels.py --output models/v4

# 3. Анализ предсказаний
python scripts/analyze_predictions.py

# 4. Визуализация
python scripts/visualize_zone_mapping_2d.py
```

---

## Следующие шаги

- [ ] Извлечь фреймовые фичи для всего DEAM датасета
- [ ] Создать unified dataset с общими признаками
- [ ] Протестировать weighted training (User weight=3.0)
- [ ] Реализовать transfer learning pipeline
- [ ] Достичь 80% accuracy на User holdout
- [ ] Проанализировать feature importance
- [ ] Оптимизировать drop detection пороги

---

## Файловая структура

```
mood-classifier/
├── dataset/
│   ├── user_frames.pkl       # Фреймовые фичи User
│   ├── deam_frames.pkl       # Фреймовые фичи DEAM
│   └── unified.pkl           # Объединённый датасет
├── models/
│   ├── checkpoints/          # Промежуточные модели
│   ├── user_only_v4/         # Baseline модель
│   ├── unified_weighted_v1/  # Взвешенное обучение
│   └── transfer_v1/          # Transfer learning
├── results/
│   ├── confusion_matrix.png
│   └── feature_importance.csv
└── scripts/
    ├── retrain_on_new_labels.py
    └── analyze_predictions.py
```
