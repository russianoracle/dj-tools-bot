# Руководство по обучению ML модели для классификации энергетических зон

Это руководство объясняет, как обучить ML модель на вашей коллекции из 2000+ треков техно/хауса.

## Обзор процесса

```
Ваша коллекция треков с BPM
    ↓
1. Создание dataset.csv (извлечение BPM из метаданных)
    ↓
2. Извлечение audio features (16 характеристик из каждого трека)
    ↓
3. Ручная разметка 200-300 треков по энергетическим зонам
    ↓
4. Обучение ML модели
    ↓
5. Валидация и тестирование
    ↓
Обученная модель для автоматической классификации
```

---

## Шаг 1: Установка зависимостей

```bash
# Установить все необходимые пакеты
pip install -r requirements.txt

# Проверить установку
python -c "import librosa, sklearn, pandas, xgboost; print('Все установлено!')"
```

---

## Шаг 2: Создание датасета (извлечение BPM)

### Если BPM уже в метаданных файлов:

```bash
# Сканировать папку с музыкой и извлечь BPM из метаданных
python scripts/create_dataset.py /path/to/your/music/folder \
    --output dataset.csv \
    --recursive
```

**Что делает скрипт:**
- Сканирует все .mp3, .flac, .m4a, .wav файлы
- Читает BPM из ID3 tags, Vorbis comments, MP4 tags
- Создает CSV файл: `path,bpm,genre,zone`

**Ожидаемый результат:**
```
Found 2000 audio files
Files with BPM: 2000
Files without BPM: 0
Dataset saved to: dataset.csv
```

### Если BPM НЕ в метаданных:

Тогда приложение вычислит BPM автоматически на следующем шаге. Создайте CSV вручную:

```csv
path,bpm,genre,zone
/path/to/track1.mp3,,,
/path/to/track2.flac,,,
```

---

## Шаг 3: Извлечение audio features

Это самый долгий шаг - извлечение 16 audio features из каждого трека.

```bash
# Извлечь features с multiprocessing (8 ядер)
python scripts/extract_features.py dataset.csv \
    --output features.csv \
    --workers 8 \
    --verbose
```

**Параметры:**
- `--workers 8` - использовать 8 CPU ядер (укажите ваше число)
- `--format csv` - сохранить в CSV (или `pickle` для быстрой загрузки)
- `--verbose` - подробный вывод

**Время выполнения:**
- С оптимизированным STFT: ~8-10 сек на трек
- 2000 треков на 8 ядрах: **~4-5 часов**

**Что извлекается из каждого трека:**
1. **Temporal**: BPM, zero-crossing rate, low energy %, RMS energy
2. **Spectral**: rolloff, brightness, spectral centroid
3. **MFCC**: 5 коэффициентов (mean + std)
4. **Dynamic**: energy variance, drop intensity

**Ожидаемый результат:**
```
Extracting features: 100%|████████| 2000/2000 [4:23:15<00:00, 8.12s/it]
Successfully extracted features from 2000/2000 tracks

BPM Detection Accuracy:
  MAE (Mean Absolute Error): 1.85 BPM
  Within 2 BPM: 94.3%
  Within 5 BPM: 98.7%
```

---

## Шаг 4: Ручная разметка треков по энергетическим зонам

Откройте `features.csv` и разметьте **200-300 треков** в колонке `zone`:

### Критерии разметки:

#### 🟨 **YELLOW (rest zone)** - зона отдыха
- Низкая энергия, спокойные треки
- BPM обычно <110
- Примеры: ambient, deep house, downtempo
- Используется для warm-up, cool-down

#### 🟩 **GREEN (transitional)** - переходная зона
- Средняя энергия, постепенный build-up
- BPM обычно 110-128
- Примеры: progressive house, tech house
- Используется для transitions между зонами

#### 🟪 **PURPLE (energy/hits)** - энергетическая зона
- Высокая энергия, выраженные drops
- BPM обычно >128
- Примеры: energetic techno, electro house
- Используется для peak time, drops

### Как размечать:

```csv
path,bpm,genre,zone
/music/ambient/track1.mp3,95,Ambient,yellow
/music/techhouse/track2.mp3,125,Tech House,green
/music/techno/track3.flac,135,Techno,purple
...
```

### Рекомендации:
- **Минимум**: 50 треков каждой зоны (150 total)
- **Оптимум**: 70-100 треков каждой зоны (200-300 total)
- Размечайте **равномерно** - по ~33% каждой зоны
- Выбирайте **типичные** представители каждой зоны, не пограничные случаи

---

## Шаг 5: Обучение ML модели

```bash
# Базовое обучение с Random Forest
python scripts/train_zone_classifier.py features.csv \
    --model-output models/zone_classifier.pkl \
    --algorithm random_forest

# С grid search для оптимальных параметров (медленнее, но точнее)
python scripts/train_zone_classifier.py features.csv \
    --model-output models/zone_classifier.pkl \
    --algorithm xgboost \
    --grid-search

# Кастомный split (train/val/test)
python scripts/train_zone_classifier.py features.csv \
    --model-output models/zone_classifier.pkl \
    --train-size 0.7 \
    --val-size 0.15
```

**Доступные алгоритмы:**
- `random_forest` - быстрый, хорошая точность (рекомендуется)
- `gradient_boosting` - медленнее, немного точнее
- `xgboost` - best accuracy, требует больше данных

**Ожидаемый результат:**
```
Dataset split:
  Train: 140 samples
  Val:   30 samples
  Test:  30 samples

Training random_forest model...
Training accuracy: 0.957
Validation accuracy: 0.900

Top 10 most important features:
  tempo                    : 0.2543
  drop_intensity           : 0.1892
  energy_variance          : 0.1234
  brightness               : 0.0987
  ...

Evaluating on test set...
Test Set Performance:
  Accuracy:  0.900
  Precision: 0.895
  Recall:    0.900
  F1 Score:  0.897

Classification Report:
              precision    recall  f1-score   support
      yellow       0.92      0.92      0.92        12
       green       0.85      0.89      0.87         9
      purple       0.91      0.91      0.91         11

Confusion Matrix:
[[11  1  0]
 [ 1  8  0]
 [ 0  1 10]]

Model saved to: models/zone_classifier.pkl
```

---

## Шаг 6: Использование обученной модели

### Вариант A: CLI с ML моделью

```bash
# Проанализировать один трек
python main.py --file track.mp3 \
    --model models/zone_classifier.pkl

# Batch processing всей коллекции
python main.py --batch /path/to/music \
    --model models/zone_classifier.pkl \
    --write-metadata \
    --export-csv classified_tracks.csv
```

### Вариант B: GUI с ML моделью

Отредактируйте `config/default_config.yaml`:

```yaml
classification:
  model_path: models/zone_classifier.pkl  # Путь к обученной модели
  confidence_threshold: 0.7
```

Запустите GUI:
```bash
python main.py --gui
```

### Вариант C: Программный API

```python
from src.classification.classifier import EnergyZoneClassifier
from src.audio.loader import AudioLoader
from src.audio.extractors import FeatureExtractor

# Загрузить модель
classifier = EnergyZoneClassifier(model_path='models/zone_classifier.pkl')

# Анализировать трек
loader = AudioLoader()
extractor = FeatureExtractor()

audio_data = loader.load('track.mp3')
features = extractor.extract(audio_data.audio, audio_data.sample_rate)

result = classifier.classify(features)

print(f"Zone: {result.zone}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Method: {result.method}")  # 'ml' или 'rule-based'
```

---

## Шаг 7: Валидация на полной коллекции

После обучения протестируйте на всей коллекции:

```bash
# Классифицировать все 2000 треков
python main.py --batch /path/to/music \
    --model models/zone_classifier.pkl \
    --export-csv results.csv
```

Проверьте `results.csv`:
```csv
path,zone,confidence,bpm,method
/music/track1.mp3,yellow,0.95,98,ml
/music/track2.flac,purple,0.88,132,ml
...
```

### Анализ результатов:

```python
import pandas as pd

df = pd.read_csv('results.csv')

# Распределение по зонам
print(df['zone'].value_counts())

# Средняя уверенность
print(f"Avg confidence: {df['confidence'].mean():.2%}")

# Треки с низкой уверенностью (проверить вручную)
low_conf = df[df['confidence'] < 0.7]
print(f"Low confidence tracks: {len(low_conf)}")
```

---

## Переобучение модели

По мере использования вы можете собрать больше данных:

1. **Добавьте новые размеченные треки** в `features.csv`
2. **Переобучите модель:**
   ```bash
   python scripts/train_zone_classifier.py features.csv \
       --model-output models/zone_classifier_v2.pkl
   ```
3. **Сравните версии** на test set
4. **Используйте лучшую модель**

---

## Troubleshooting

### Проблема: "No labeled tracks found!"

**Решение**: Откройте `features.csv` и заполните колонку `zone` (yellow/green/purple) для 150+ треков.

### Проблема: Low accuracy (<80%)

**Причины:**
1. Мало размеченных данных - добавьте больше
2. Несбалансированные классы - разметьте равномерно
3. Неправильная разметка - перепроверьте критерии

**Решение:**
```bash
# Grid search для лучших параметров
python scripts/train_zone_classifier.py features.csv --grid-search
```

### Проблема: BPM detection error > 5 BPM

**Причина**: Сложные ритмы (breakbeat, polyrhythm)

**Решение**: Используйте BPM из вашего DJ софта (Rekordbox/Traktor):
1. Экспортируйте playlist в CSV с BPM
2. Обновите `dataset.csv` колонку `bpm`
3. Переизвлеките features

### Проблема: Feature extraction очень медленная

**Решение:**
```bash
# Используйте больше workers
python scripts/extract_features.py dataset.csv --workers 16

# Или сохраните в pickle (быстрее для повторной загрузки)
python scripts/extract_features.py dataset.csv \
    --output features.pkl \
    --format pickle
```

---

## Результаты

После завершения процесса вы получите:

✅ **Обученную ML модель** (`models/zone_classifier.pkl`)
✅ **Confusion matrix** показывающую точность по каждой зоне
✅ **Feature importance** - какие характеристики важнее всего
✅ **Автономное приложение** для классификации новых треков
✅ **Данные для всей коллекции** с зонами в метаданных

---

## Метрики качества

### Целевые показатели:

- **Accuracy**: >85% (доля правильных предсказаний)
- **F1 Score**: >0.85 (баланс precision/recall)
- **Per-class recall**: >80% для каждой зоны
- **BPM MAE**: <2 BPM (средняя абсолютная ошибка)

### Если метрики ниже:

1. Добавьте больше размеченных данных (300-500 треков)
2. Проверьте качество разметки (пограничные случаи)
3. Попробуйте XGBoost с grid search
4. Калибруйте пороговые значения в `config/default_config.yaml`

---

## Следующие шаги

1. **Active Learning**: Приложение показывает неуверенные предсказания для ручной проверки
2. **Genre-specific models**: Отдельные модели для techno, house, trance
3. **Online learning**: Модель учится на ваших правках
4. **Export to DJ software**: Импорт зон в Rekordbox/Serato/Traktor

---

## Поддержка

Если возникли проблемы, проверьте:
- Логи в консоли (с флагом `--verbose`)
- Файлы в папке `results/plots/` (confusion matrix, feature importance)
- Статистику в выводе `extract_features.py`

Удачи с обучением модели! 🎧🎵
