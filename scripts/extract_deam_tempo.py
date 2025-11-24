#!/usr/bin/env python3
"""
DEAM Tempo Extractor

Извлекает tempo (BPM) для всех треков из DEAM датасета.
Это единственная фича которая отсутствует в предрассчитанных DEAM фичах.

Время выполнения: ~5 секунд на трек × 1,802 треков = ~2.5 часа
"""

import sys
import argparse
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_tempo_single(audio_path: str, sample_rate: int = 22050) -> dict:
    """
    Извлекает tempo для одного трека

    Args:
        audio_path: Путь к audio файлу
        sample_rate: Sample rate для загрузки

    Returns:
        Dict с track_id, tempo, tempo_confidence
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=sample_rate, duration=45.0)  # DEAM треки 45 сек

        # Извлекаем tempo с использованием HPSS для лучшей точности
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # Проверяем октавные ошибки (из вашего fix_bpm_accuracy)
        # Если BPM слишком низкий или высокий, корректируем
        if tempo < 60:
            tempo *= 2
        elif tempo > 200:
            tempo /= 2

        track_id = int(Path(audio_path).stem)  # filename = track_id

        return {
            'track_id': track_id,
            'tempo': float(tempo.item() if hasattr(tempo, 'item') else tempo),
            'audio_path': audio_path,
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"Error extracting tempo from {audio_path}: {e}")
        return {
            'track_id': int(Path(audio_path).stem) if audio_path else -1,
            'tempo': None,
            'audio_path': audio_path,
            'success': False,
            'error': str(e)
        }


def extract_tempo_batch(audio_dir: str,
                       output_csv: str,
                       sample_rate: int = 22050,
                       limit: int = None,
                       checkpoint_every: int = 50):
    """
    Извлекает tempo для всех треков в директории

    Args:
        audio_dir: Директория с MP3 файлами DEAM
        output_csv: Путь для сохранения результатов
        sample_rate: Sample rate для извлечения
        limit: Максимальное количество треков (для тестирования)
        checkpoint_every: Сохранять промежуточные результаты каждые N треков
    """
    audio_dir = Path(audio_dir)
    output_csv = Path(output_csv)

    # Создаём директорию для output если не существует
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Получаем список всех MP3 файлов
    audio_files = sorted(audio_dir.glob("*.mp3"))

    if limit:
        audio_files = audio_files[:limit]

    logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")
    logger.info(f"Output will be saved to: {output_csv}")
    logger.info(f"Estimated time: {len(audio_files) * 5 / 60:.1f} minutes")

    # Загружаем существующие результаты если есть (для resume)
    if output_csv.exists():
        logger.info(f"Loading existing results from {output_csv}")
        existing_df = pd.read_csv(output_csv)
        existing_ids = set(existing_df['track_id'].tolist())
        logger.info(f"Found {len(existing_ids)} already processed tracks")
    else:
        existing_df = None
        existing_ids = set()

    # Фильтруем уже обработанные
    audio_files = [f for f in audio_files if int(f.stem) not in existing_ids]
    logger.info(f"Remaining tracks to process: {len(audio_files)}")

    if not audio_files:
        logger.info("All tracks already processed!")
        return

    # Обрабатываем треки
    results = []
    failed = 0

    for i, audio_file in enumerate(tqdm(audio_files, desc="Extracting tempo")):
        result = extract_tempo_single(str(audio_file), sample_rate=sample_rate)
        results.append(result)

        if not result['success']:
            failed += 1

        # Checkpoint: сохраняем промежуточные результаты
        if (i + 1) % checkpoint_every == 0:
            _save_results(results, existing_df, output_csv)
            logger.info(f"Checkpoint: saved {i + 1} results (failed: {failed})")

    # Финальное сохранение
    _save_results(results, existing_df, output_csv)

    logger.info(f"Extraction complete!")
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Successful: {len(results) - failed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results saved to: {output_csv}")

    # Статистика tempo
    successful_tempos = [r['tempo'] for r in results if r['success']]
    if successful_tempos:
        logger.info(f"Tempo statistics:")
        logger.info(f"  Mean: {np.mean(successful_tempos):.1f} BPM")
        logger.info(f"  Median: {np.median(successful_tempos):.1f} BPM")
        logger.info(f"  Min: {np.min(successful_tempos):.1f} BPM")
        logger.info(f"  Max: {np.max(successful_tempos):.1f} BPM")


def _save_results(new_results: list, existing_df: pd.DataFrame, output_path: Path):
    """Сохраняет результаты, объединяя с существующими если есть"""
    new_df = pd.DataFrame(new_results)

    if existing_df is not None:
        # Объединяем с существующими
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Сортируем по track_id
    combined_df = combined_df.sort_values('track_id')

    # Сохраняем
    combined_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Extract tempo from DEAM dataset')
    parser.add_argument('--audio-dir', type=str, default='dataset/MEMD_audio',
                       help='Directory with DEAM MP3 files')
    parser.add_argument('--output', type=str, default='dataset/deam_tempo.csv',
                       help='Output CSV file')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Sample rate for audio loading (default: 22050)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tracks (for testing)')
    parser.add_argument('--checkpoint-every', type=int, default=50,
                       help='Save checkpoint every N tracks (default: 50)')

    args = parser.parse_args()

    try:
        extract_tempo_batch(
            audio_dir=args.audio_dir,
            output_csv=args.output,
            sample_rate=args.sample_rate,
            limit=args.limit,
            checkpoint_every=args.checkpoint_every
        )
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
