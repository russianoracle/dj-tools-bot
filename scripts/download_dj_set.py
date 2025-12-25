#!/usr/bin/env python3
"""
Interactive DJ Set Downloader

Скачивает DJ сеты с SoundCloud/YouTube в каталог data/dj_sets/
и опционально запускает аудио-анализ.

Поддержка плейлистов SoundCloud с проверкой дубликатов.
Параллельная загрузка для плейлистов.
Организация по каталогам исполнителей.

Использование:
    python scripts/download_dj_set.py                    # интерактивный режим
    python scripts/download_dj_set.py <URL>              # скачать один сет
    python scripts/download_dj_set.py <PLAYLIST_URL>     # скачать плейлист
    python scripts/download_dj_set.py <PLAYLIST_URL> -d  # быстрая загрузка
    python scripts/download_dj_set.py --list             # показать загруженные
"""

import subprocess
import sys
import os
import json
from pathlib import Path
import re
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import shutil

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DJ_SETS_DIR = PROJECT_ROOT / "data" / "dj_sets"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "reference_sets"

# Ensure directories exist
DJ_SETS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Print lock for thread-safe output
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


def print_progress_bar(current: int, total: int, success: int, failed: int,
                       width: int = 30, prefix: str = ""):
    """Выводит прогресс-бар в одну строку."""
    if total == 0:
        return

    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)

    status = f"✅{success}"
    if failed > 0:
        status += f" ❌{failed}"

    line = f"\r{prefix}[{current}/{total}] {bar} {percent*100:.0f}% {status}"

    with print_lock:
        sys.stdout.write(line)
        sys.stdout.flush()
        if current >= total:
            print()  # New line at the end


def sanitize_filename(name: str) -> str:
    """Очищает имя файла от спецсимволов."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.lower().strip('-')
    return name


def normalize_for_comparison(name: str) -> str:
    """Нормализует имя для сравнения (убирает все спецсимволы)."""
    # Убираем всё кроме букв и цифр
    name = re.sub(r'[^a-zA-Z0-9а-яА-Я]', '', name.lower())
    return name


def extract_artist_from_url(url: str) -> str:
    """Извлекает имя исполнителя из URL."""
    # SoundCloud: https://soundcloud.com/artist-name/...
    if 'soundcloud.com' in url:
        match = re.search(r'soundcloud\.com/([^/]+)', url)
        if match:
            artist = match.group(1)
            # Clean up artist name
            artist = artist.replace('-', ' ').title()
            return artist
    return "Unknown"


def get_existing_sets() -> Dict[str, Path]:
    """Возвращает словарь существующих сетов {sanitized_name: path}."""
    sets = {}

    # Search in main directory and all subdirectories
    for ext in ['mp3', 'm4a', 'opus', 'wav']:
        # Main directory
        for f in DJ_SETS_DIR.glob(f"*.{ext}"):
            if '.part' not in f.name:
                sets[f.stem.lower()] = f

        # Artist subdirectories
        for f in DJ_SETS_DIR.glob(f"**/*.{ext}"):
            if '.part' not in f.name:
                sets[f.stem.lower()] = f

    return sets


def list_existing_sets():
    """Показывает существующие сеты."""
    sets = get_existing_sets()

    # Group by directory
    by_dir = {}
    for name, path in sorted(sets.items()):
        parent = path.parent.name if path.parent != DJ_SETS_DIR else "/"
        if parent not in by_dir:
            by_dir[parent] = []
        by_dir[parent].append(path)

    if sets:
        print(f"\n📁 Загруженные сеты ({len(sets)}):")
        for dir_name, files in sorted(by_dir.items()):
            if dir_name != "/":
                print(f"\n   📂 {dir_name}/")
            for path in sorted(files):
                size_mb = path.stat().st_size / (1024 * 1024)
                prefix = "      " if dir_name != "/" else "   "
                print(f"{prefix}• {path.name} ({size_mb:.1f} MB)")
    else:
        print("\n📁 Нет загруженных сетов")
    print()
    return sets


def get_playlist_info(url: str) -> Tuple[List[Dict], str]:
    """
    Получает информацию о треках в плейлисте.

    Returns:
        Tuple of (tracks list, artist name)
    """
    print(f"📋 Получаю информацию о плейлисте...")

    artist = extract_artist_from_url(url)

    # First try flat playlist (fast) to get count
    cmd_flat = ['yt-dlp', '--flat-playlist', '-J', url]
    try:
        result = subprocess.run(cmd_flat, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get('_type') == 'playlist':
                count = len(data.get('entries', []))
                title = data.get('title', 'Unknown')

                # Try to get artist from playlist owner
                uploader = data.get('uploader', '')
                if uploader:
                    artist = uploader

                print(f"   Плейлист: {title}")
                print(f"   Исполнитель: {artist}")
                print(f"   Треков: {count}")

                # For large playlists, suggest direct mode
                if count > 20:
                    print(f"   ⚠️  Большой плейлист! Рекомендуем флаг -d для быстрой загрузки")

                print(f"   Получаю детали...")
    except Exception as e:
        print(f"   ⚠️  Не удалось получить предварительную информацию: {e}")

    # Now get full info with progress
    cmd = [
        'yt-dlp',
        '-J',
        '--no-download',
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"❌ Ошибка: {result.stderr[:200] if result.stderr else 'unknown'}")
            return [], artist

        data = json.loads(result.stdout)

        # Get artist from response
        if data.get('uploader'):
            artist = data.get('uploader')

        if data.get('_type') == 'playlist':
            entries = data.get('entries', [])
            print(f"   ✅ Получено {len(entries)} треков")
            return entries, artist
        else:
            return [data], artist

    except subprocess.TimeoutExpired:
        print("❌ Таймаут (5 мин). Используйте флаг -d для прямой загрузки.")
        return [], artist
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга: {e}")
        return [], artist
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return [], artist


def check_duplicates(tracks: List[Dict], existing: Dict[str, Path], artist: str) -> Tuple[List[Dict], List[Dict]]:
    """Проверяет какие треки уже загружены."""
    new_tracks = []
    existing_tracks = []

    artist_prefix = sanitize_filename(artist)
    artist_normalized = normalize_for_comparison(artist)

    # Pre-normalize all existing names for faster comparison
    existing_normalized = {
        normalize_for_comparison(name): name
        for name in existing.keys()
    }

    for track in tracks:
        title = track.get('title', '')
        if not title:
            continue

        sanitized = sanitize_filename(title)
        full_name = f"{artist_prefix}---{sanitized}"

        # Normalized versions for fuzzy matching
        title_normalized = normalize_for_comparison(title)
        # Remove artist name from title for comparison (in case title includes artist)
        title_without_artist = title_normalized.replace(artist_normalized, '')

        # Check multiple patterns
        is_duplicate = False

        for existing_norm, existing_name in existing_normalized.items():
            # 1. Exact normalized match
            if title_normalized == existing_norm:
                is_duplicate = True
                break

            # 2. Title without artist matches
            if title_without_artist and len(title_without_artist) > 5:
                if title_without_artist in existing_norm or existing_norm in title_without_artist:
                    is_duplicate = True
                    break

            # 3. Normalized title is substring (for partial matches)
            if len(title_normalized) > 10:
                if title_normalized in existing_norm or existing_norm in title_normalized:
                    is_duplicate = True
                    break

            # 4. Original sanitized comparison (backward compat)
            if sanitized == existing_name or full_name == existing_name:
                is_duplicate = True
                break

        if is_duplicate:
            existing_tracks.append(track)
        else:
            new_tracks.append(track)

    return new_tracks, existing_tracks


def get_artist_dir(artist: str) -> Path:
    """Возвращает путь к каталогу исполнителя."""
    artist_dir = DJ_SETS_DIR / sanitize_filename(artist)
    artist_dir.mkdir(parents=True, exist_ok=True)
    return artist_dir


class EncodingPipeline:
    """
    Pipeline для параллельного скачивания и конвертации.

    Скачивание идёт в оригинальном формате (быстро),
    конвертация в mp3 — отдельным потоком через очередь.
    """

    def __init__(self, target_format: str = "mp3", quality: str = "192K",
                 download_workers: int = 3, encode_workers: int = 2):
        self.target_format = target_format
        self.quality = quality
        self.download_workers = download_workers
        self.encode_workers = encode_workers

        self.encode_queue: Queue = Queue()
        self.results: List[Path] = []
        self.failed: List[str] = []
        self.stats = {
            'downloaded': 0,
            'encoded': 0,
            'total': 0
        }
        self._stop_event = threading.Event()
        self._results_lock = threading.Lock()

    def _download_raw(self, url: str, output_dir: Path, filename: str) -> Optional[Path]:
        """Скачивает в оригинальном формате (без конвертации)."""
        output_template = str(output_dir / f"{filename}.%(ext)s")

        cmd = [
            "yt-dlp",
            "-x",  # Extract audio only
            # НЕ указываем --audio-format, чтобы сохранить оригинал
            "-o", output_template,
            "--quiet",
            "--no-warnings",
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                return None

            # Find downloaded file (could be opus, m4a, etc.)
            for ext in ['opus', 'm4a', 'webm', 'ogg', 'mp3']:
                path = output_dir / f"{filename}.{ext}"
                if path.exists():
                    return path

            # Fallback: find any file matching pattern
            files = list(output_dir.glob(f"{filename}.*"))
            if files:
                return files[0]

        except Exception:
            pass

        return None

    def _encode_worker(self):
        """Воркер конвертации — работает из очереди."""
        while not self._stop_event.is_set() or not self.encode_queue.empty():
            try:
                item = self.encode_queue.get(timeout=1)
            except:
                continue

            if item is None:  # Poison pill
                break

            raw_path, target_path, title = item

            try:
                # Convert using ffmpeg
                cmd = [
                    "ffmpeg",
                    "-i", str(raw_path),
                    "-codec:a", "libmp3lame",
                    "-b:a", self.quality,
                    "-y",  # Overwrite
                    "-loglevel", "error",
                    str(target_path)
                ]

                result = subprocess.run(cmd, capture_output=True, timeout=600)

                if result.returncode == 0 and target_path.exists():
                    # Remove raw file
                    raw_path.unlink()

                    with self._results_lock:
                        self.results.append(target_path)
                        self.stats['encoded'] += 1

                    safe_print(f"   🎵 Encoded: {title}")
                else:
                    with self._results_lock:
                        self.failed.append(f"{title} (encode)")

            except Exception as e:
                with self._results_lock:
                    self.failed.append(f"{title} (encode: {e})")

            finally:
                self.encode_queue.task_done()

    def _download_and_queue(self, track_info: Dict, artist: str,
                           artist_dir: Path, url_base: str) -> bool:
        """Скачивает трек и добавляет в очередь на конвертацию."""
        title = track_info.get('title', 'unknown')
        track_url = track_info.get('webpage_url') or track_info.get('url')

        if track_url and not track_url.startswith('http'):
            if 'soundcloud.com' in url_base:
                match = re.search(r'soundcloud\.com/([^/]+)', url_base)
                if match:
                    sc_artist = match.group(1)
                    track_url = f"https://soundcloud.com/{sc_artist}/{track_url}"
                else:
                    track_url = f"https://soundcloud.com{track_url}"

        if not track_url:
            return False

        artist_prefix = sanitize_filename(artist)
        filename = f"{artist_prefix}---{sanitize_filename(title)}"

        # Check if already exists
        target_path = artist_dir / f"{filename}.{self.target_format}"
        if target_path.exists():
            with self._results_lock:
                self.results.append(target_path)
                self.stats['downloaded'] += 1
                self.stats['encoded'] += 1
            return True

        # Download raw
        raw_path = self._download_raw(track_url, artist_dir, f"_raw_{filename}")

        if raw_path and raw_path.exists():
            with self._results_lock:
                self.stats['downloaded'] += 1

            safe_print(f"   ⬇️  Downloaded: {title}")

            # Queue for encoding
            self.encode_queue.put((raw_path, target_path, title))
            return True

        return False

    def download_playlist(self, tracks: List[Dict], artist: str,
                         url_base: str, artist_dir: Path) -> List[Path]:
        """
        Скачивает плейлист с pipeline: download || encode.
        """
        self.stats['total'] = len(tracks)

        print(f"\n🚀 Pipeline: {self.download_workers} downloaders + {self.encode_workers} encoders\n")

        # Start encoder threads
        encoder_threads = []
        for _ in range(self.encode_workers):
            t = threading.Thread(target=self._encode_worker, daemon=True)
            t.start()
            encoder_threads.append(t)

        # Download in parallel
        with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
            futures = {
                executor.submit(
                    self._download_and_queue,
                    track, artist, artist_dir, url_base
                ): track for track in tracks
            }

            for future in as_completed(futures):
                track = futures[future]
                try:
                    success = future.result()
                    if not success:
                        with self._results_lock:
                            self.failed.append(track.get('title', 'Unknown'))
                except Exception as e:
                    with self._results_lock:
                        self.failed.append(f"{track.get('title', 'Unknown')}: {e}")

        # Wait for encoding to finish
        print(f"\n⏳ Waiting for encoding to finish...")
        self.encode_queue.join()

        # Stop encoder threads
        self._stop_event.set()
        for _ in encoder_threads:
            self.encode_queue.put(None)  # Poison pills

        for t in encoder_threads:
            t.join(timeout=5)

        print(f"\n✅ Downloaded: {self.stats['downloaded']}/{self.stats['total']}")
        print(f"✅ Encoded: {self.stats['encoded']}/{self.stats['total']}")

        if self.failed:
            print(f"❌ Failed: {len(self.failed)}")
            for f in self.failed[:5]:
                print(f"   • {f}")

        return self.results


def download_playlist_pipeline(url: str, audio_format: str = "mp3", quality: str = "192K",
                              download_workers: int = 4, encode_workers: int = 2,
                              skip_confirm: bool = False) -> List[Path]:
    """
    Скачивает плейлист с pipeline (скачивание и конвертация параллельно).

    Быстрее чем обычный режим, т.к. скачивание не ждёт ffmpeg.
    """
    # Get playlist info
    tracks, artist = get_playlist_info(url)
    if not tracks:
        return []

    # Check for duplicates
    existing = get_existing_sets()
    new_tracks, duplicates = check_duplicates(tracks, existing, artist)

    print(f"\n📊 Результат проверки:")
    print(f"   👤 Исполнитель: {artist}")
    print(f"   ✅ Новых: {len(new_tracks)}")
    print(f"   ⏭️  Уже загружено: {len(duplicates)}")

    if not new_tracks:
        print("\n✨ Все треки уже загружены!")
        return []

    print(f"\n   Будут загружены:")
    for i, t in enumerate(new_tracks[:10], 1):
        print(f"      {i}. {t.get('title', 'Unknown')}")
    if len(new_tracks) > 10:
        print(f"      ... и ещё {len(new_tracks) - 10}")

    if not skip_confirm:
        confirm = input(f"\n▶️  Загрузить {len(new_tracks)} треков (pipeline mode)? [Y/n]: ").strip().lower()
        if confirm in ('n', 'no', 'н', 'нет'):
            print("❌ Отменено")
            return []

    artist_dir = get_artist_dir(artist)

    pipeline = EncodingPipeline(
        target_format=audio_format,
        quality=quality,
        download_workers=download_workers,
        encode_workers=encode_workers
    )

    return pipeline.download_playlist(new_tracks, artist, url, artist_dir)


def download_playlist_direct(url: str, artist: str = None, audio_format: str = "mp3",
                            quality: str = "192K", keep_original: bool = False) -> None:
    """
    Скачивает плейлист напрямую через yt-dlp.

    Args:
        keep_original: Если True, сохраняет в оригинальном формате (opus/m4a)
                      без конвертации — быстрее и librosa всё равно прочитает
    """
    if not artist:
        artist = extract_artist_from_url(url)

    artist_dir = get_artist_dir(artist)
    archive_file = DJ_SETS_DIR / "downloaded.txt"
    artist_prefix = sanitize_filename(artist)

    # Output template with artist prefix
    output_template = str(artist_dir / f"{artist_prefix}---%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "-o", output_template,
        "--download-archive", str(archive_file),
        "--progress",
        "--concurrent-fragments", "4",
        url
    ]

    # Add format conversion only if not keeping original
    if not keep_original:
        cmd.extend(["--audio-format", audio_format, "--audio-quality", quality])
        format_info = f"{audio_format} {quality}"
    else:
        format_info = "оригинальный формат (opus/m4a)"

    print(f"⬇️  Скачиваю плейлист в {artist_dir.name}/")
    print(f"   Формат: {format_info}")
    print(f"   Архив: {archive_file}")
    print(f"   (уже скачанные будут пропущены автоматически)\n")

    subprocess.run(cmd)
    print("\n✅ Готово!")


def download_single_track(track_info: Dict, artist: str, url_base: str,
                         keep_original: bool = True) -> Optional[Path]:
    """Скачивает один трек из плейлиста (для параллельной загрузки)."""
    title = track_info.get('title', 'unknown')

    # Try different URL fields
    track_url = track_info.get('webpage_url') or track_info.get('url')

    # For SoundCloud, construct full URL if needed
    if track_url and not track_url.startswith('http'):
        if 'soundcloud.com' in url_base:
            # Extract artist from base URL
            match = re.search(r'soundcloud\.com/([^/]+)', url_base)
            if match:
                sc_artist = match.group(1)
                track_url = f"https://soundcloud.com/{sc_artist}/{track_url}"
            else:
                track_url = f"https://soundcloud.com{track_url}"

    if not track_url:
        safe_print(f"   ⚠️  Нет URL для: {title}")
        return None

    return download_set(track_url, artist=artist, quiet=True, keep_original=keep_original)


def download_playlist(url: str, audio_format: str = "mp3", quality: str = "192K",
                     analyze: bool = False, skip_confirm: bool = False,
                     parallel: int = 3) -> List[Path]:
    """
    Скачивает плейлист с проверкой дубликатов и параллельной загрузкой.

    Args:
        parallel: Количество параллельных загрузок (default: 3)

    Returns:
        Список путей к скачанным файлам
    """
    # Get playlist info
    tracks, artist = get_playlist_info(url)
    if not tracks:
        return []

    # Check for duplicates
    existing = get_existing_sets()
    new_tracks, duplicates = check_duplicates(tracks, existing, artist)

    print(f"\n📊 Результат проверки:")
    print(f"   👤 Исполнитель: {artist}")
    print(f"   ✅ Новых: {len(new_tracks)}")
    print(f"   ⏭️  Уже загружено: {len(duplicates)}")

    if duplicates:
        print(f"\n   Пропускаем:")
        for t in duplicates[:5]:  # Show first 5
            print(f"      • {t.get('title', 'Unknown')}")
        if len(duplicates) > 5:
            print(f"      ... и ещё {len(duplicates) - 5}")

    if not new_tracks:
        print("\n✨ Все треки уже загружены!")
        return []

    print(f"\n   Будут загружены:")
    for i, t in enumerate(new_tracks, 1):
        print(f"      {i}. {t.get('title', 'Unknown')}")

    # Confirm
    if not skip_confirm:
        confirm = input(f"\n▶️  Загрузить {len(new_tracks)} треков (параллельно: {parallel})? [Y/n]: ").strip().lower()
        if confirm in ('n', 'no', 'н', 'нет'):
            print("❌ Отменено")
            return []

    # Download with parallel execution
    downloaded = []
    failed = []
    total = len(new_tracks)

    print(f"\n🚀 Параллельная загрузка ({parallel} потоков)...\n")

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(
                download_single_track,
                track, artist, url  # скачиваем в оригинале
            ): track for track in new_tracks
        }

        for i, future in enumerate(as_completed(futures), 1):
            track = futures[future]
            title = track.get('title', 'Unknown')
            try:
                result = future.result()
                if result:
                    downloaded.append(result)

                    if analyze:
                        analyze_set(result)
                else:
                    failed.append(title)
            except Exception as e:
                failed.append(title)

            # Update progress bar
            print_progress_bar(i, total, len(downloaded), len(failed), prefix="   ")

    print(f"\n✅ Загружено: {len(downloaded)}/{len(new_tracks)}")
    if failed:
        print(f"❌ Не удалось: {len(failed)}")
        for f in failed[:5]:
            print(f"   • {f}")

    return downloaded


def download_set(url: str, filename: str = None, artist: str = None,
                audio_format: str = None, quality: str = "192K",
                quiet: bool = False, keep_original: bool = True,
                output_dir: Path = None) -> Optional[Path]:
    """
    Скачивает один DJ сет.

    Args:
        keep_original: Если True (по умолчанию), сохраняет в оригинале без конвертации
        output_dir: Директория для сохранения (если None, используется dj_sets/artist/)
    """
    # Use custom output dir or default artist-based dir
    if output_dir:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        artist_prefix = ""  # No prefix for custom output dir
    else:
        # Extract artist from URL if not provided
        if not artist:
            artist = extract_artist_from_url(url)
        target_dir = get_artist_dir(artist)
        artist_prefix = sanitize_filename(artist)

    # Auto-generate filename if not provided
    if not filename:
        try:
            result = subprocess.run(
                ['yt-dlp', '--get-title', url],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                filename = sanitize_filename(result.stdout.strip())
        except:
            pass

        if not filename:
            filename = f"set_{len(list(target_dir.glob('*'))) + 1}"

    # Add artist prefix to filename (empty for custom output_dir)
    if artist_prefix:
        full_filename = f"{artist_prefix}---{filename}"
    else:
        full_filename = filename

    # Check if already exists (exact match)
    for ext in ['opus', 'm4a', 'mp3', 'ogg', 'webm']:
        existing = target_dir / f"{full_filename}.{ext}"
        if existing.exists():
            if not quiet:
                print(f"⏭️  Уже существует: {existing.name}")
            return existing
        # Also check old format without artist prefix (only if we have prefix)
        if artist_prefix:
            old_existing = target_dir / f"{filename}.{ext}"
            if old_existing.exists():
                if not quiet:
                    print(f"⏭️  Уже существует: {old_existing.name}")
                return old_existing

    # Check for similar files using normalized comparison (skip for custom output_dir)
    if not output_dir:
        filename_normalized = normalize_for_comparison(filename)
        for ext in ['opus', 'm4a', 'mp3', 'ogg', 'webm']:
            for existing_file in target_dir.glob(f"*.{ext}"):
                existing_normalized = normalize_for_comparison(existing_file.stem)
                # Check if normalized names match (fuzzy)
                if filename_normalized and len(filename_normalized) > 10:
                    if filename_normalized in existing_normalized or existing_normalized in filename_normalized:
                        if not quiet:
                            print(f"⏭️  Похожий файл уже существует: {existing_file.name}")
                        return existing_file

    output_template = str(target_dir / f"{full_filename}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "-o", output_template,
        "--progress" if not quiet else "--quiet",
        url
    ]

    # Add conversion only if not keeping original
    if not keep_original and audio_format:
        cmd.extend(["--audio-format", audio_format, "--audio-quality", quality])
        format_info = audio_format
    else:
        format_info = "original"

    if not quiet:
        print(f"⬇️  {full_filename} ({format_info})")
        print(f"   📂 {target_dir.name}/")

    try:
        result = subprocess.run(cmd, capture_output=quiet)

        if result.returncode != 0:
            if not quiet:
                print(f"❌ Ошибка загрузки")
            return None

        # Find the downloaded file
        for ext in [audio_format, 'opus', 'm4a', 'mp3']:
            output_path = target_dir / f"{full_filename}.{ext}"
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                if not quiet:
                    print(f"✅ {output_path.name} ({size_mb:.1f} MB)")
                return output_path

        # Fallback
        files = sorted(target_dir.glob(f"{full_filename}.*"),
                      key=lambda x: x.stat().st_mtime, reverse=True)
        if files:
            return files[0]

    except FileNotFoundError:
        if not quiet:
            print("❌ yt-dlp не установлен!")
        return None
    except Exception as e:
        if not quiet:
            print(f"❌ Ошибка: {e}")
        return None

    return None


def analyze_set(audio_path: Path) -> Optional[dict]:
    """Запускает аудио-анализ сета."""
    safe_print(f"🔍 Анализирую: {audio_path.name}")

    try:
        sys.path.insert(0, str(PROJECT_ROOT))

        import numpy as np
        from src.data.mix_audio_analyzer import MixAudioAnalyzer

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_, np.integer)):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        analyzer = MixAudioAnalyzer(sr=22050, segment_sec=30.0)
        analysis = analyzer.analyze(str(audio_path))
        style = analyzer.extract_style_vector(analysis)

        safe_print(f"   {analysis.duration_min:.0f} min, {style['avg_tempo']:.0f} BPM")

        # Save
        analysis_path = ANALYSIS_DIR / f"{audio_path.stem}_audio_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump({
                'source_file': str(audio_path),
                'analysis': analysis.to_dict(),
                'style_vector': style
            }, f, indent=2, cls=NumpyEncoder)

        safe_print(f"   ✅ Сохранено: {analysis_path.name}")
        return style

    except Exception as e:
        safe_print(f"   ❌ Ошибка: {e}")
        return None


def interactive_mode():
    """Интерактивный режим."""
    print("=" * 50)
    print("🎧 DJ Set Downloader")
    print("=" * 50)

    existing = list_existing_sets()

    url = input("🔗 URL (сет или плейлист): ").strip()
    if not url:
        print("❌ URL не указан")
        return

    # Check if playlist
    is_playlist = '/sets/' in url or '/playlist' in url or 'list=' in url

    print("\n📦 Формат:")
    print("   1. Оригинал (opus/m4a) — быстро, без конвертации [рекомендуется]")
    print("   2. MP3 192K")
    print("   3. MP3 320K")

    format_choice = input("   Выбор [1]: ").strip() or "1"
    formats = {
        "1": (None, None, True),    # (format, quality, keep_original)
        "2": ("mp3", "192K", False),
        "3": ("mp3", "320K", False)
    }
    audio_format, quality, keep_original = formats.get(format_choice, (None, None, True))

    analyze = input("\n🔍 Анализировать после загрузки? [y/N]: ").strip().lower()
    do_analyze = analyze in ('y', 'yes', 'д', 'да')

    if is_playlist:
        print("\n⚡ Режим загрузки плейлиста:")
        print("   1. Параллельная с проверкой дубликатов + прогресс-бар")
        print("   2. Быстрая прямая загрузка (yt-dlp archive)")
        mode = input("   Выбор [1]: ").strip() or "1"

        if mode == "2":
            download_playlist_direct(url, audio_format=audio_format or "mp3",
                                   quality=quality or "192K", keep_original=keep_original)
        else:
            parallel = input("\n🔄 Параллельных загрузок [3]: ").strip()
            parallel = int(parallel) if parallel.isdigit() else 3
            download_playlist(url, audio_format or "mp3", quality or "192K",
                            do_analyze, parallel=parallel)
    else:
        result = download_set(url, audio_format=audio_format, quality=quality or "192K",
                             keep_original=keep_original)
        if result and do_analyze:
            analyze_set(result)

    print("\n👋 Готово!")


def download_batch(urls: List[str], parallel: int = 3, analyze: bool = False,
                   keep_original: bool = True, output_dir: Path = None) -> List[Path]:
    """
    Скачать массив URL параллельно.

    Args:
        urls: Список URL для скачивания (формат: URL или "filename|URL")
        parallel: Количество параллельных загрузок
        analyze: Анализировать после загрузки
        keep_original: Сохранять оригинальный формат
        output_dir: Директория для сохранения (если None, используется dj_sets/artist/)

    Returns:
        Список путей к скачанным файлам
    """
    print(f"\n{'=' * 60}")
    print(f"BATCH DOWNLOAD ({len(urls)} URLs)")
    if output_dir:
        print(f"Output dir: {output_dir}")
    print(f"{'=' * 60}")

    # Parse URLs with optional filenames (format: "filename|URL" or just "URL")
    items = []
    for line in urls:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '|' in line:
            filename, url = line.split('|', 1)
            items.append((filename.strip(), url.strip()))
        else:
            items.append((None, line))

    downloaded = []
    failed = []

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(
                download_set,
                url,
                filename=filename,
                audio_format=None,
                keep_original=keep_original,
                quiet=True,
                output_dir=output_dir
            ): (filename, url) for filename, url in items
        }

        for i, future in enumerate(as_completed(futures), 1):
            filename, url = futures[future]
            display_name = filename or url[:50]
            try:
                result = future.result()
                if result:
                    downloaded.append(result)
                    safe_print(f"  [{i}/{len(futures)}] OK: {result.name}")
                    if analyze:
                        analyze_set(result)
                else:
                    failed.append(url)
                    safe_print(f"  [{i}/{len(futures)}] FAILED: {display_name}...")
            except Exception as e:
                failed.append(url)
                safe_print(f"  [{i}/{len(futures)}] ERROR: {display_name}... - {e}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Downloaded: {len(downloaded)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\n  Failed URLs:")
        for url in failed[:10]:
            print(f"    - {url[:60]}...")

    return downloaded


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DJ Set Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --list                              # показать загруженные
  %(prog)s <URL>                               # скачать один сет
  %(prog)s <PLAYLIST_URL>                      # скачать плейлист (параллельно)
  %(prog)s <PLAYLIST_URL> -d                   # быстрая прямая загрузка
  %(prog)s <URL> -a                            # скачать и анализировать
  %(prog)s <PLAYLIST_URL> -p 5                 # 5 параллельных загрузок
  %(prog)s --batch urls.txt                    # скачать из файла со списком URL
  %(prog)s --batch urls.txt -p 5               # batch с 5 параллельными загрузками
        """
    )
    parser.add_argument("url", nargs="?", help="URL сета или плейлиста")
    parser.add_argument("--name", "-n", help="Имя файла (для одного сета)")
    parser.add_argument("--format", "-f", default="mp3", choices=["mp3", "m4a", "opus"])
    parser.add_argument("--quality", "-q", default="192K")
    parser.add_argument("--analyze", "-a", action="store_true", help="Анализировать после загрузки")
    parser.add_argument("--list", "-l", action="store_true", help="Показать загруженные сеты")
    parser.add_argument("--yes", "-y", action="store_true", help="Пропустить подтверждение")
    parser.add_argument("--direct", "-d", action="store_true",
                       help="Прямая загрузка плейлиста (быстро, без предварительной проверки)")
    parser.add_argument("--parallel", "-p", type=int, default=3,
                       help="Количество параллельных загрузок (default: 3)")
    parser.add_argument("--convert", "-c", action="store_true",
                       help="Конвертировать в mp3 (по умолчанию сохраняется оригинал opus/m4a)")
    parser.add_argument("--batch", "-b", type=str,
                       help="Файл со списком URL (формат: URL или 'filename|URL', # комментарии)")
    parser.add_argument("--output-dir", "-o", type=str,
                       help="Директория для сохранения (для batch режима)")

    args = parser.parse_args()

    if args.list:
        list_existing_sets()
        return

    # Batch mode
    if args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"ERROR: File not found: {batch_file}")
            sys.exit(1)

        with open(batch_file) as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        if not urls:
            print("ERROR: No URLs found in file")
            sys.exit(1)

        print(f"Found {len(urls)} URLs in {batch_file}")
        output_dir = Path(args.output_dir) if args.output_dir else None
        download_batch(urls, parallel=args.parallel, analyze=args.analyze,
                      keep_original=not args.convert, output_dir=output_dir)
        return

    if args.url:
        # Check if playlist
        is_playlist = '/sets/' in args.url or '/playlist' in args.url or 'list=' in args.url

        if is_playlist:
            if args.direct:
                # Быстрая загрузка — по умолчанию в оригинальном формате
                # Конвертация только если указан -c
                download_playlist_direct(
                    args.url,
                    audio_format=args.format,
                    quality=args.quality,
                    keep_original=not args.convert  # по умолчанию оригинал, -c = конвертировать
                )
            else:
                download_playlist(args.url, args.format, args.quality, args.analyze,
                                args.yes, parallel=args.parallel)
        else:
            # По умолчанию оригинал, -c = конвертировать
            result = download_set(
                args.url,
                args.name,
                audio_format=args.format,
                quality=args.quality,
                keep_original=not args.convert
            )
            if result and args.analyze:
                analyze_set(result)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()