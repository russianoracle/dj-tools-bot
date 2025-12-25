#!/usr/bin/env python3
"""
DJ Tools - Interactive CLI Menu.

Usage:
    python main.py
    python scripts/generate_set_interactive.py
"""

import sys
import os
import time
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", message=".*aifc.*deprecated.*")
warnings.filterwarnings("ignore", message=".*audioop.*deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import io
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
try:
    from src.core.cache import CacheRepository
    from src.core.pipelines.set_generator import SetGeneratorPipeline, SetPlan, SetPhase
    from src.core.pipelines import BatchProgressDisplay, MixingStyle
    from src.services import AnalysisService, ProfilingService
finally:
    sys.stderr = _stderr_backup


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str, width: int = 60):
    print()
    print(f"{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * width}{Colors.RESET}")
    print()


def print_menu_item(key: str, text: str, indent: int = 0):
    spaces = "  " * indent
    print(f"{spaces}  {Colors.YELLOW}[{key}]{Colors.RESET} {text}")


def print_divider(width: int = 60):
    print(f"{Colors.DIM}{'─' * width}{Colors.RESET}")


def get_input(prompt: str, default: str = "") -> str:
    try:
        if default:
            result = input(f"{prompt} [{default}]: ").strip()
            return result if result else default
        return input(f"{prompt}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def get_int_input(prompt: str, default: int, min_val: int = 0, max_val: int = 999) -> int:
    while True:
        try:
            val = get_input(prompt, str(default))
            if not val:
                return default
            num = int(val)
            if min_val <= num <= max_val:
                return num
            print(f"{Colors.RED}Введите число от {min_val} до {max_val}{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Введите число{Colors.RESET}")


def get_file_path(prompt: str) -> str:
    path = get_input(prompt)
    if path:
        # Remove quotes and expand ~
        path = path.strip("'\"").strip()
        path = os.path.expanduser(path)
        if Path(path).exists():
            return path
        print(f"{Colors.RED}Файл не найден: {path}{Colors.RESET}")
    return ""


def get_folder_path(prompt: str) -> str:
    path = get_input(prompt)
    if path:
        # Remove quotes and expand ~
        path = path.strip("'\"").strip()
        path = os.path.expanduser(path)
        if Path(path).is_dir():
            return path
        print(f"{Colors.RED}Папка не найдена: {path}{Colors.RESET}")
    return ""


class InteractiveMenu:
    # Audio file extensions
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.aiff'}

    # Default directories
    DEFAULT_TRACKS_DIR = Path(__file__).parent.parent / "data" / "tracks"
    DEFAULT_SETS_DIR = Path(__file__).parent.parent / "data" / "dj_sets"
    SETTINGS_FILE = Path(__file__).parent.parent / "cache" / "menu_settings.json"

    def __init__(self):
        self.cache = CacheRepository()
        self.pipeline = SetGeneratorPipeline(cache_repo=self.cache)
        self.current_plan: SetPlan = None
        self.running = True

        # Load saved directories or use defaults
        self.tracks_dir: Path = self.DEFAULT_TRACKS_DIR
        self.sets_dir: Path = self.DEFAULT_SETS_DIR
        self._load_settings()

    def run(self):
        while self.running:
            self.show_main_menu()

    def show_main_menu(self):
        clear_screen()
        print_header("DJ TOOLS")

        print_menu_item("1", "Работа с треками")
        print_menu_item("2", "Работа с сетами")
        print_menu_item("3", "DJ профили")
        print_menu_item("4", "Треклисты")
        print_menu_item("5", "Скачать аудио")
        print_menu_item("6", "Обзор файлов")
        print_menu_item("7", "Кеш")
        print_divider()
        print_menu_item("H", "Помощь")
        print_menu_item("0", "Выход")
        print()

        choice = get_input("Выбор").lower()

        if choice == "1":
            self.tracks_menu()
        elif choice == "2":
            self.sets_menu()
        elif choice == "6":
            self.file_browser_menu()
        elif choice == "3":
            self.profiles_menu()
        elif choice == "4":
            self.tracklists_menu()
        elif choice == "5":
            self.download_menu()
        elif choice == "7":
            self.cache_menu()
        elif choice == "h":
            self.show_help()
        elif choice == "0":
            self.running = False

    # ==================== ТРЕКИ ====================
    def tracks_menu(self):
        clear_screen()
        print_header("РАБОТА С ТРЕКАМИ")

        print_menu_item("1", "Анализ одного трека")
        print_menu_item("2", "Пакетный анализ")
        print_divider()
        print_menu_item("0", "Назад")
        print()

        choice = get_input("Выбор")

        if choice == "1":
            self.analyze_single_track()
        elif choice == "2":
            self.analyze_batch_tracks()

    def analyze_single_track(self):
        """Анализ одного трека: BPM, beat grid, дропы, энергия."""
        clear_screen()
        print_header("АНАЛИЗ ТРЕКА")

        path = get_file_path("Путь к файлу")
        if not path:
            get_input("\nНажмите Enter")
            return

        # Спрашиваем про классификацию зон
        classify_zones = get_input("Классифицировать энергозону? [y/N]", "n").lower() in ("y", "да")

        print(f"\n{Colors.CYAN}Анализ {Path(path).name}...{Colors.RESET}")
        print(f"{Colors.DIM}Это может занять 10-30 секунд{Colors.RESET}\n")

        try:
            self._analyze_track(path, classify_zones)
        except Exception as e:
            print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()

        get_input("\nНажмите Enter")

    def _analyze_track(self, path: str, classify_zones: bool = False):
        """Анализ трека с использованием AnalysisService."""
        service = AnalysisService()

        # Проверяем кеш
        cached = self.cache.get_track(path)
        if cached:
            print(f"{Colors.GREEN}Загружено из кеша{Colors.RESET}")
            self._print_track_result(cached.to_dict(), classify_zones, path)
            return

        # Progress callback
        def progress_callback(progress: float, stage: str, message: str):
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r  [{bar}] {int(progress*100):3d}% {message[:40]:<40}", end="", flush=True)
            if progress >= 1.0:
                print()

        # Analyze using service
        result = service.analyze_track(
            path,
            use_cache=True,
            on_progress=progress_callback,
        )

        if result.success:
            print(f"\n{Colors.GREEN}Анализ завершён{Colors.RESET}")
            print(f"  Длительность: {result.duration_sec/60:.1f} мин")
            print(f"  Темп: {result.tempo:.1f} BPM")
            print(f"  Дропов: {result.n_drops}")
            print(f"  Зона: {result.zone.display_name if result.zone else 'N/A'}")
        else:
            print(f"\n{Colors.RED}Ошибка: {result.error}{Colors.RESET}")

        # Опциональная классификация зон
        if classify_zones:
            self._classify_track_zone(path)

    def _print_track_result(self, result: dict, classify_zones: bool, path: str):
        """Печать результатов из кеша."""
        duration = result.get('duration_sec', 0)
        print(f"  Длительность: {duration/60:.1f} мин ({duration:.0f} сек)")

        tempo = result.get('tempo', 0)
        n_bars = result.get('n_bars', 0)
        n_phrases = result.get('n_phrases', 0)
        if tempo > 0:
            print(f"\nBeat Grid:")
            print(f"  Темп: {tempo:.1f} BPM")
            print(f"  Тактов: {n_bars}")
            print(f"  Фраз: {n_phrases}")

        n_drops = result.get('n_drops', 0)
        drop_density = result.get('drop_density', 0)
        print(f"\nДропы:")
        print(f"  Дропов: {n_drops}")
        print(f"  Плотность: {drop_density:.2f}/мин")

        drop_times = result.get('drop_times', [])
        if drop_times:
            print("\n  Главные дропы:")
            for i, t in enumerate(drop_times[:5], 1):
                mins = int(t // 60)
                secs = int(t % 60)
                print(f"    {i}. {mins:02d}:{secs:02d}")

        print("\nЭнергетический профиль:")
        dist = result.get('drop_temporal_distribution', 0.5)
        if dist < 0.35:
            profile = "Раскачка в начале"
        elif dist > 0.65:
            profile = "Финальный взрыв"
        else:
            profile = "Равномерная энергия"
        print(f"  Тип: {profile}")
        print(f"  1-я половина: {result.get('drops_first_half', 0)} дропов")
        print(f"  2-я половина: {result.get('drops_second_half', 0)} дропов")

        if classify_zones:
            self._classify_track_zone(path)

    def _classify_track_zone(self, path: str):
        """Классификация трека по энергетическим зонам (опционально)."""
        try:
            from src.audio import AudioLoader, FeatureExtractor
            from src.classification import EnergyZoneClassifier
            from src.utils import get_config

            print(f"\n{Colors.CYAN}Классификация энергозоны...{Colors.RESET}")

            config = get_config()
            loader = AudioLoader(sample_rate=22050)
            extractor = FeatureExtractor(config)
            classifier = EnergyZoneClassifier(config)

            y, sr = loader.load(path)
            features = extractor.extract(y, sr)
            result = classifier.classify(features)

            print(f"  Зона: {result.zone.emoji} {result.zone.display_name}")
            print(f"  Уверенность: {result.confidence:.1%}")

        except Exception as e:
            print(f"{Colors.YELLOW}Не удалось классифицировать: {e}{Colors.RESET}")

    def analyze_batch_tracks(self):
        """Пакетный анализ треков - тот же анализ для каждого файла в папке."""
        clear_screen()
        print_header("ПАКЕТНЫЙ АНАЛИЗ ТРЕКОВ")

        folder = get_folder_path("Папка с треками")
        if not folder:
            get_input("\nНажмите Enter")
            return

        recursive = get_input("Включая подпапки? [y/N]", "n").lower() in ("y", "да")
        classify_zones = get_input("Классифицировать энергозоны? [y/N]", "n").lower() in ("y", "да")

        # Найти файлы
        files = []
        for ext in ['.mp3', '.wav', '.flac', '.m4a']:
            if recursive:
                files.extend(Path(folder).rglob(f'*{ext}'))
            else:
                files.extend(Path(folder).glob(f'*{ext}'))

        if not files:
            print(f"{Colors.YELLOW}Файлы не найдены{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        print(f"\n{Colors.CYAN}Найдено {len(files)} файлов{Colors.RESET}\n")

        # Анализ каждого файла
        for i, f in enumerate(files, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(files)}] {f.name}")
            print(f"{'='*60}\n")

            try:
                self._analyze_track(str(f), classify_zones)
            except Exception as e:
                print(f"{Colors.RED}Ошибка: {e}{Colors.RESET}")

        print(f"\n{Colors.GREEN}Обработано {len(files)} файлов{Colors.RESET}")
        get_input("\nНажмите Enter")

    # ==================== СЕТЫ ====================
    def sets_menu(self):
        clear_screen()
        print_header("РАБОТА С СЕТАМИ")

        print_menu_item("1", "Анализ одного сета")
        print_menu_item("2", "Пакетный анализ")
        print_divider()
        print_menu_item("0", "Назад")
        print()

        choice = get_input("Выбор")

        if choice == "1":
            self.analyze_single_set()
        elif choice == "2":
            self.analyze_batch_sets()

    def analyze_single_set(self):
        clear_screen()
        print_header("АНАЛИЗ DJ СЕТА")

        path = get_file_path("Путь к файлу")
        if not path:
            get_input("\nНажмите Enter")
            return

        print(f"\n{Colors.CYAN}Анализ {Path(path).name}...{Colors.RESET}")
        print(f"{Colors.DIM}Это может занять несколько минут{Colors.RESET}\n")

        try:
            from scripts.analyze_dj_set import analyze_dj_set
            analyze_dj_set(path, show_progress=True)
        except Exception as e:
            print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")

        get_input("\nНажмите Enter")

    def analyze_batch_sets(self):
        clear_screen()
        print_header("ПАКЕТНЫЙ АНАЛИЗ СЕТОВ")

        folder = get_folder_path("Папка с сетами")
        if not folder:
            get_input("\nНажмите Enter")
            return

        recursive = get_input("Включая подпапки? [y/N]", "n").lower() in ("y", "да")

        print(f"\n{Colors.CYAN}Запуск анализа...{Colors.RESET}\n")

        try:
            service = AnalysisService()

            # Create progress display
            folder_path = Path(folder)
            audio_files = [f for f in (folder_path.rglob('*') if recursive else folder_path.glob('*'))
                          if f.suffix.lower() in self.AUDIO_EXTENSIONS]

            if not audio_files:
                print(f"{Colors.YELLOW}Аудио файлы не найдены{Colors.RESET}")
                get_input("\nНажмите Enter")
                return

            display = BatchProgressDisplay(total_files=len(audio_files), max_log_lines=6)
            display.render()

            result = service.analyze_batch(
                folder,
                recursive=recursive,
                use_cache=True,
                display=display,
            )

            print(f"\n{Colors.GREEN}Анализ завершён!{Colors.RESET}")
            print(f"  Обработано: {result.processed}")
            print(f"  Из кеша: {result.cached}")
            print(f"  Ошибок: {result.failed}")

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Прервано{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")

        get_input("\nНажмите Enter")

    # ==================== ПРОФИЛИ ====================
    def profiles_menu(self):
        while True:
            clear_screen()
            print_header("DJ ПРОФИЛИ")

            profiles = self.cache.get_all_dj_profiles_info()

            if profiles:
                print("Существующие профили:")
                print_divider()
                for i, p in enumerate(profiles, 1):
                    updated = time.strftime("%d.%m.%Y", time.localtime(p['updated_at']))
                    print(f"  {Colors.YELLOW}[{i}]{Colors.RESET} {p['dj_name']:<20} "
                          f"({p['n_sets']} сетов, {p['total_hours']:.1f}ч, {updated})")
                print()

            print_divider()
            print_menu_item("C", "Создать профиль из кеша (по DJ)")
            print_menu_item("F", "Создать профиль из папки")
            print_menu_item("A", "Агрегировать все сеты по DJ")
            print_divider()
            print_menu_item("0", "Назад")
            print()

            choice = get_input("Выбор (номер для деталей)").lower()

            if choice == "0":
                break
            elif choice == "c":
                self._create_profile_from_cache()
            elif choice == "f":
                self._create_profile_from_folder()
            elif choice == "a":
                self._aggregate_all_profiles()
            elif choice.isdigit() and profiles and 1 <= int(choice) <= len(profiles):
                self._show_profile_details(profiles[int(choice) - 1])

    def _show_profile_details(self, profile_meta: dict):
        clear_screen()
        dj_name = profile_meta['dj_name']
        cached = self.cache.get_dj_profile(dj_name)

        if not cached:
            print(f"{Colors.RED}Профиль не найден{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        profile = cached.to_dict()
        print_header(f"ПРОФИЛЬ: {dj_name}")

        tempo = profile.get('tempo_distribution', {})
        print(f"  {Colors.CYAN}Темп:{Colors.RESET}")
        print(f"    Диапазон: {tempo.get('min_bpm', 0):.0f} - {tempo.get('max_bpm', 0):.0f} BPM")
        print(f"    Средний: {tempo.get('mean_bpm', 0):.1f} BPM")

        energy = profile.get('energy_arc', {})
        print(f"\n  {Colors.CYAN}Энергия:{Colors.RESET}")
        print(f"    Начало: {energy.get('opening_energy', 0):.0%}")
        print(f"    Пик: {energy.get('peak_energy', 0):.0%}")
        print(f"    Финал: {energy.get('closing_energy', 0):.0%}")

        get_input("\nНажмите Enter")

    def _create_profile_from_cache(self):
        """Create DJ profile from cached set analyses (grouped by DJ name from folder)."""
        clear_screen()
        print_header("СОЗДАТЬ ПРОФИЛЬ ИЗ КЕША")

        # Get all cached sets and group by DJ (folder name)
        cached_paths = self.cache.get_all_cached_sets()
        if not cached_paths:
            print(f"{Colors.YELLOW}Нет проанализированных сетов в кеше{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        # Group by DJ name (parent folder name)
        dj_sets = {}
        for path in cached_paths:
            p = Path(path)
            dj_name = p.parent.name  # Folder name as DJ name
            if dj_name not in dj_sets:
                dj_sets[dj_name] = []
            dj_sets[dj_name].append(path)

        if not dj_sets:
            print(f"{Colors.YELLOW}Не удалось определить DJ по папкам{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        # Show available DJs
        print("Найденные DJ (по папкам):")
        print_divider()
        dj_list = sorted(dj_sets.keys())
        for i, dj_name in enumerate(dj_list, 1):
            count = len(dj_sets[dj_name])
            # Check if profile exists
            existing = self.cache.get_dj_profile(dj_name)
            status = f"{Colors.GREEN}✓{Colors.RESET}" if existing else f"{Colors.DIM}○{Colors.RESET}"
            print(f"  {Colors.YELLOW}[{i}]{Colors.RESET} {status} {dj_name:<25} ({count} сетов)")
        print()

        choice = get_input("Номер DJ для создания профиля (0=отмена)")
        if not choice.isdigit() or int(choice) == 0:
            return

        idx = int(choice) - 1
        if idx < 0 or idx >= len(dj_list):
            return

        dj_name = dj_list[idx]
        sets = dj_sets[dj_name]

        print(f"\n{Colors.CYAN}Создание профиля для {dj_name} из {len(sets)} сетов...{Colors.RESET}\n")

        # Use ProfilingService
        service = ProfilingService()

        try:
            # Profile each set (using cached analysis)
            result = service.profile_dj(
                dj_name=dj_name,
                folder=str(Path(sets[0]).parent),  # Use folder from first set
                use_cache=True,
                verbose=True,
                on_progress=lambda d, t, p: print(f"  [{d}/{t}] {Path(p).name[:50]}")
            )

            if result.profile:
                print(f"\n{Colors.GREEN}Профиль создан!{Colors.RESET}")
                print(f"  Стиль: {result.profile.mixing_style}")
                print(f"  Энергия: {result.profile.energy_profile}")
                print(f"  Дропов/час: {result.profile.avg_drops_per_hour:.1f}")
            else:
                print(f"\n{Colors.YELLOW}Не удалось создать профиль{Colors.RESET}")

        except Exception as e:
            print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")

        get_input("\nНажмите Enter")

    def _create_profile_from_folder(self):
        """Create DJ profile by selecting a folder with sets."""
        clear_screen()
        print_header("СОЗДАТЬ ПРОФИЛЬ ИЗ ПАПКИ")

        folder = get_folder_path("Путь к папке с сетами DJ")
        if not folder:
            return

        # Default DJ name from folder
        default_name = Path(folder).name
        dj_name = get_input(f"Имя DJ [{default_name}]") or default_name

        # Count audio files
        folder_path = Path(folder)
        audio_files = [f for f in folder_path.rglob('*')
                       if f.suffix.lower() in self.AUDIO_EXTENSIONS]

        if not audio_files:
            print(f"{Colors.YELLOW}Аудио файлы не найдены{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        print(f"\nНайдено {len(audio_files)} файлов")

        # Check cache status
        cached_count = sum(1 for f in audio_files if self.cache.get_set(str(f.absolute())))
        uncached_count = len(audio_files) - cached_count

        print(f"  В кеше: {cached_count}")
        print(f"  Требует анализа: {uncached_count}")

        if uncached_count > 0:
            analyze = get_input(f"\nПроанализировать {uncached_count} файлов? [y/N]", "n")
            if analyze.lower() not in ("y", "да"):
                if cached_count == 0:
                    print(f"{Colors.YELLOW}Нет данных для профиля{Colors.RESET}")
                    get_input("\nНажмите Enter")
                    return

        print(f"\n{Colors.CYAN}Создание профиля...{Colors.RESET}\n")

        # Use ProfilingService
        service = ProfilingService()

        try:
            result = service.profile_dj(
                dj_name=dj_name,
                folder=folder,
                recursive=True,
                use_cache=True,
                verbose=True,
                on_progress=lambda d, t, p: print(f"  [{d}/{t}] {Path(p).name[:50]}")
            )

            if result.profile:
                print(f"\n{Colors.GREEN}Профиль '{dj_name}' создан!{Colors.RESET}")
                print(f"  Сетов: {result.profile.n_sets_analyzed}")
                print(f"  Стиль: {result.profile.mixing_style}")
                print(f"  Энергия: {result.profile.energy_profile}")
            else:
                print(f"\n{Colors.YELLOW}Не удалось создать профиль{Colors.RESET}")
                if result.errors:
                    print(f"  Ошибок: {len(result.errors)}")

        except Exception as e:
            print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")

        get_input("\nНажмите Enter")

    def _aggregate_all_profiles(self):
        """Aggregate profiles for all DJs found in sets directory."""
        clear_screen()
        print_header("АГРЕГАЦИЯ ПРОФИЛЕЙ")

        if not self.sets_dir.exists():
            print(f"{Colors.YELLOW}Каталог сетов не найден: {self.sets_dir}{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        # Find all DJ folders (immediate subdirectories)
        dj_folders = [d for d in self.sets_dir.iterdir() if d.is_dir()]

        if not dj_folders:
            print(f"{Colors.YELLOW}Не найдено папок DJ в {self.sets_dir}{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        print(f"Найдено {len(dj_folders)} папок DJ:")
        print_divider()

        # Show status for each
        dj_info = []
        for folder in sorted(dj_folders):
            dj_name = folder.name
            audio_files = [f for f in folder.rglob('*')
                          if f.suffix.lower() in self.AUDIO_EXTENSIONS]
            cached = sum(1 for f in audio_files if self.cache.get_set(str(f.absolute())))
            has_profile = self.cache.get_dj_profile(dj_name) is not None
            dj_info.append((dj_name, len(audio_files), cached, has_profile, folder))

            status = f"{Colors.GREEN}✓{Colors.RESET}" if has_profile else f"{Colors.DIM}○{Colors.RESET}"
            cache_status = f"{cached}/{len(audio_files)}" if audio_files else "0"
            print(f"  {status} {dj_name:<25} ({cache_status} в кеше)")

        print()
        confirm = get_input("Создать/обновить все профили? [y/N]", "n")
        if confirm.lower() not in ("y", "да"):
            return

        print(f"\n{Colors.CYAN}Агрегация профилей...{Colors.RESET}\n")

        service = ProfilingService()
        created = 0
        errors = 0

        for dj_name, total, cached, has_profile, folder in dj_info:
            if total == 0:
                continue

            print(f"\n{Colors.CYAN}{dj_name}{Colors.RESET} ({cached}/{total} в кеше)")

            try:
                result = service.profile_dj(
                    dj_name=dj_name,
                    folder=str(folder),
                    recursive=True,
                    use_cache=True,
                )

                if result.profile:
                    print(f"  {Colors.GREEN}✓{Colors.RESET} {result.profile.mixing_style}, "
                          f"{result.profile.avg_drops_per_hour:.1f} drops/h")
                    created += 1
                else:
                    print(f"  {Colors.YELLOW}○{Colors.RESET} Нет данных")

            except Exception as e:
                print(f"  {Colors.RED}✗{Colors.RESET} {str(e)[:50]}")
                errors += 1

        print(f"\n{Colors.GREEN}Готово!{Colors.RESET}")
        print(f"  Создано/обновлено: {created}")
        print(f"  Ошибок: {errors}")

        get_input("\nНажмите Enter")

    # ==================== ТРЕКЛИСТЫ ====================
    def tracklists_menu(self):
        clear_screen()
        print_header("ТРЕКЛИСТЫ")

        print_menu_item("1", "Создать новый сет")
        print_menu_item("2", "Показать текущий план")
        print_menu_item("3", "Экспорт в M3U")
        print_divider()
        print_menu_item("0", "Назад")
        print()

        if self.current_plan:
            print(f"{Colors.GREEN}●{Colors.RESET} Текущий план: {self.current_plan.dj_name}, "
                  f"{self.current_plan.n_tracks} треков\n")

        choice = get_input("Выбор")

        if choice == "1":
            self.create_tracklist()
        elif choice == "2" and self.current_plan:
            self.show_tracklist()
        elif choice == "3" and self.current_plan:
            self.export_tracklist()

    def create_tracklist(self):
        clear_screen()
        print_header("СОЗДАТЬ ТРЕКЛИСТ")

        profiles = self.cache.get_all_dj_profiles_info()
        if not profiles:
            print(f"{Colors.YELLOW}Нет DJ профилей для генерации{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        print("Выберите DJ профиль:")
        for i, p in enumerate(profiles, 1):
            print(f"  {Colors.YELLOW}[{i}]{Colors.RESET} {p['dj_name']}")

        idx = get_int_input("\nНомер", 1, 1, len(profiles))
        dj_name = profiles[idx - 1]['dj_name']

        duration = get_int_input("Длительность (мин)", 60, 15, 180)

        print(f"\n{Colors.CYAN}Генерация плана...{Colors.RESET}")
        self.current_plan = self.pipeline.generate_plan(dj_name, duration)

        if self.current_plan.n_tracks > 0:
            print(f"\n{Colors.GREEN}✓{Colors.RESET} Создан план: {self.current_plan.n_tracks} треков")
        else:
            print(f"\n{Colors.RED}Не удалось найти треки{Colors.RESET}")

        get_input("\nНажмите Enter")

    def show_tracklist(self):
        if not self.current_plan:
            return

        clear_screen()
        print_header(f"ТРЕКЛИСТ: {self.current_plan.dj_name}")

        print(f"  Треков: {self.current_plan.n_tracks}")
        print(f"  Длительность: {self.current_plan.actual_duration_min:.0f} мин")
        print()

        for track in self.current_plan.tracks[:20]:
            title = track.title[:30] + ".." if len(track.title) > 32 else track.title
            print(f"  {track.position:2}. {title:<32} {track.bpm:>5.0f} BPM  {track.camelot:>3}")

        if self.current_plan.n_tracks > 20:
            print(f"\n  ... и ещё {self.current_plan.n_tracks - 20} треков")

        get_input("\nНажмите Enter")

    def export_tracklist(self):
        if not self.current_plan:
            return

        name = get_input("Название плейлиста", f"{self.current_plan.dj_name} Set")
        if not name:
            return

        success = self.pipeline.export_to_rekordbox(self.current_plan, name)
        if success:
            print(f"\n{Colors.GREEN}✓ Плейлист '{name}' создан{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}Ошибка экспорта{Colors.RESET}")

        get_input("\nНажмите Enter")

    # ==================== СКАЧИВАНИЕ ====================
    def download_menu(self):
        clear_screen()
        print_header("СКАЧАТЬ АУДИО")

        print(f"{Colors.DIM}Поддержка: SoundCloud, YouTube, Mixcloud и др.{Colors.RESET}\n")

        print_menu_item("1", "Скачать один сет/трек")
        print_menu_item("2", "Скачать плейлист (SoundCloud /sets/ или YouTube)")
        print_menu_item("3", "Показать загруженные")
        print_divider()
        print_menu_item("0", "Назад")
        print()

        choice = get_input("Выбор")

        if choice == "1":
            self.download_single()
        elif choice == "2":
            self.download_playlist()
        elif choice == "3":
            self.show_downloaded()

    def download_single(self):
        """Скачать один сет/трек по URL."""
        clear_screen()
        print_header("СКАЧАТЬ СЕТ/ТРЕК")

        print(f"{Colors.DIM}Поддерживаются: SoundCloud, YouTube, Mixcloud, Bandcamp{Colors.RESET}\n")

        url = get_input("URL")
        if not url:
            return

        # Определяем платформу для информации
        platform = "Unknown"
        if "soundcloud.com" in url:
            platform = "SoundCloud"
        elif "youtube.com" in url or "youtu.be" in url:
            platform = "YouTube"
        elif "mixcloud.com" in url:
            platform = "Mixcloud"
        elif "bandcamp.com" in url:
            platform = "Bandcamp"

        print(f"\n{Colors.CYAN}Платформа: {platform}{Colors.RESET}")

        # Формат
        print(f"\n{Colors.CYAN}Формат:{Colors.RESET}")
        print("  1. Оригинал (opus/m4a) - быстро")
        print("  2. MP3 192K")
        print("  3. MP3 320K")
        fmt_choice = get_input("Выбор", "1")

        keep_original = fmt_choice == "1"
        audio_format = "mp3" if fmt_choice in ("2", "3") else None
        quality = "320K" if fmt_choice == "3" else "192K"

        # Анализировать после?
        analyze = get_input("Анализировать после загрузки? [y/N]", "n").lower() in ("y", "да")

        print(f"\n{Colors.CYAN}Загрузка...{Colors.RESET}\n")

        try:
            from scripts.download_dj_set import download_set, analyze_set

            result = download_set(
                url,
                audio_format=audio_format,
                quality=quality,
                keep_original=keep_original
            )

            if result:
                print(f"\n{Colors.GREEN}Загружено: {result.name}{Colors.RESET}")
                if analyze:
                    analyze_set(result)
            else:
                print(f"\n{Colors.RED}Ошибка загрузки{Colors.RESET}")

        except Exception as e:
            print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")

        get_input("\nНажмите Enter")

    def download_playlist(self):
        """Скачать плейлист с параллельной загрузкой."""
        clear_screen()
        print_header("СКАЧАТЬ ПЛЕЙЛИСТ")

        print(f"{Colors.DIM}SoundCloud: /sets/ URL")
        print(f"YouTube: playlist?list= или канал{Colors.RESET}\n")

        url = get_input("URL плейлиста")
        if not url:
            return

        # Определяем платформу
        is_soundcloud = "soundcloud.com" in url
        is_youtube = "youtube.com" in url or "youtu.be" in url

        # Проверка что это плейлист
        is_playlist = (
            '/sets/' in url or          # SoundCloud sets
            '/playlist' in url or        # YouTube playlist
            'list=' in url or           # YouTube playlist param
            '/@' in url or              # YouTube channel
            '/c/' in url                # YouTube channel
        )

        if not is_playlist:
            print(f"\n{Colors.YELLOW}Это не похоже на плейлист.{Colors.RESET}")
            print(f"{Colors.DIM}SoundCloud: URL должен содержать /sets/")
            print(f"YouTube: URL должен содержать list= или быть каналом{Colors.RESET}")
            confirm = get_input("\nПродолжить всё равно? [y/N]", "n")
            if confirm.lower() not in ("y", "да"):
                return

        platform = "YouTube" if is_youtube else "SoundCloud" if is_soundcloud else "Unknown"
        print(f"\n{Colors.CYAN}Платформа: {platform}{Colors.RESET}")

        # Формат
        print(f"\n{Colors.CYAN}Формат:{Colors.RESET}")
        print("  1. Оригинал (opus/m4a) - быстро, без конвертации")
        print("  2. MP3 192K")
        fmt_choice = get_input("Выбор", "1")
        keep_original = fmt_choice == "1"

        # Параллельность
        parallel = get_int_input("Параллельных загрузок", 4, 1, 10)

        # Режим загрузки
        print(f"\n{Colors.CYAN}Режим:{Colors.RESET}")
        print("  1. С проверкой дубликатов (рекомендуется)")
        print("  2. Быстрая прямая загрузка (yt-dlp archive)")
        mode = get_input("Выбор", "1")

        print(f"\n{Colors.CYAN}Загрузка плейлиста...{Colors.RESET}\n")

        try:
            from scripts.download_dj_set import (
                download_playlist,
                download_playlist_direct,
                extract_artist_from_url
            )

            if mode == "2":
                # Прямая загрузка
                download_playlist_direct(
                    url,
                    audio_format="mp3" if not keep_original else None,
                    quality="192K",
                    keep_original=keep_original
                )
            else:
                # С проверкой дубликатов
                results = download_playlist(
                    url,
                    audio_format="mp3",
                    quality="192K",
                    analyze=False,
                    skip_confirm=False,
                    parallel=parallel
                )
                if results:
                    print(f"\n{Colors.GREEN}Загружено {len(results)} файлов{Colors.RESET}")

        except Exception as e:
            print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()

        get_input("\nНажмите Enter")

    def show_downloaded(self):
        """Показать загруженные сеты."""
        clear_screen()
        print_header("ЗАГРУЖЕННЫЕ СЕТЫ")

        try:
            from scripts.download_dj_set import list_existing_sets
            list_existing_sets()
        except Exception as e:
            print(f"{Colors.RED}Ошибка: {e}{Colors.RESET}")

        get_input("\nНажмите Enter")

    # ==================== ФАЙЛОВЫЙ БРАУЗЕР ====================

    def _load_settings(self):
        """Load saved settings from JSON file."""
        import json
        try:
            if self.SETTINGS_FILE.exists():
                with open(self.SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    if settings.get('tracks_dir'):
                        path = Path(settings['tracks_dir']).expanduser()
                        if path.exists():
                            self.tracks_dir = path
                    if settings.get('sets_dir'):
                        path = Path(settings['sets_dir']).expanduser()
                        if path.exists():
                            self.sets_dir = path
        except Exception:
            pass

    def _save_settings(self):
        """Save settings to JSON file."""
        import json
        try:
            self.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump({
                    'tracks_dir': str(self.tracks_dir),
                    'sets_dir': str(self.sets_dir)
                }, f, indent=2)
        except Exception:
            pass

    def file_browser_menu(self):
        """File browser main menu."""
        while True:
            clear_screen()
            print_header("ОБЗОР ФАЙЛОВ")

            # Show current directories
            print(f"{Colors.DIM}Текущие каталоги:{Colors.RESET}")
            print(f"  Треки: {Colors.CYAN}{self.tracks_dir}{Colors.RESET}")
            print(f"  Сеты:  {Colors.CYAN}{self.sets_dir}{Colors.RESET}")
            print()

            print_menu_item("1", "Обзор треков")
            print_menu_item("2", "Обзор сетов")
            print_divider()
            print_menu_item("T", "Изменить каталог треков")
            print_menu_item("S", "Изменить каталог сетов")
            print_menu_item("C", "Статистика кеша")
            print_divider()
            print_menu_item("0", "Назад")
            print()

            choice = get_input("Выбор").lower()

            if choice == "1":
                self.browse_directory(self.tracks_dir, "ТРЕКИ", is_set=False)
            elif choice == "2":
                self.browse_directory(self.sets_dir, "СЕТЫ", is_set=True)
            elif choice == "t":
                self._change_directory("tracks")
            elif choice == "s":
                self._change_directory("sets")
            elif choice == "c":
                self._show_cache_stats()
            elif choice == "0":
                break

    def _change_directory(self, dir_type: str):
        """Change tracks or sets directory."""
        clear_screen()
        title = "КАТАЛОГ ТРЕКОВ" if dir_type == "tracks" else "КАТАЛОГ СЕТОВ"
        print_header(title)

        current = self.tracks_dir if dir_type == "tracks" else self.sets_dir
        print(f"Текущий: {Colors.CYAN}{current}{Colors.RESET}\n")

        path = get_folder_path("Новый путь (или Enter для отмены)")
        if path:
            new_path = Path(path).expanduser()
            if dir_type == "tracks":
                self.tracks_dir = new_path
            else:
                self.sets_dir = new_path
            self._save_settings()
            print(f"\n{Colors.GREEN}Каталог изменён{Colors.RESET}")
            get_input("\nНажмите Enter")

    def browse_directory(self, root_dir: Path, title: str, is_set: bool = False):
        """Browse a directory with cache status display."""
        current_dir = root_dir

        while True:
            clear_screen()
            print_header(f"{title}: {current_dir.name}")
            print(f"{Colors.DIM}{current_dir}{Colors.RESET}\n")

            # Gather items
            items = self._list_directory_with_cache(current_dir, is_set)

            if not items['dirs'] and not items['files']:
                print(f"{Colors.DIM}Каталог пуст{Colors.RESET}\n")
            else:
                # Print directories first
                if items['dirs']:
                    print(f"{Colors.BOLD}Папки:{Colors.RESET}")
                    for i, (name, count, cached_count) in enumerate(items['dirs'], 1):
                        cache_info = f" ({cached_count}/{count} в кеше)" if count > 0 else ""
                        print(f"  {Colors.YELLOW}[{i}]{Colors.RESET} 📁 {name}{Colors.DIM}{cache_info}{Colors.RESET}")
                    print()

                # Print files
                if items['files']:
                    print(f"{Colors.BOLD}Файлы:{Colors.RESET}")
                    start_idx = len(items['dirs']) + 1
                    for i, (name, is_cached, duration) in enumerate(items['files'], start_idx):
                        status = f"{Colors.GREEN}✓{Colors.RESET}" if is_cached else f"{Colors.DIM}○{Colors.RESET}"
                        dur_str = f" ({duration})" if duration else ""
                        print(f"  {Colors.YELLOW}[{i}]{Colors.RESET} {status} {name}{Colors.DIM}{dur_str}{Colors.RESET}")
                    print()

            print_divider()
            # Navigation options
            if current_dir != root_dir:
                print_menu_item("..", "Вверх")
            print_menu_item("A", f"Анализ всех некешированных ({items['uncached_count']})")
            print_menu_item("R", "Обновить")
            print_menu_item("0", "Назад")
            print()

            choice = get_input("Выбор (номер или команда)").lower()

            if choice == "0":
                break
            elif choice == ".." and current_dir != root_dir:
                current_dir = current_dir.parent
            elif choice == "r":
                continue  # Refresh
            elif choice == "a" and items['uncached_count'] > 0:
                self._analyze_uncached(current_dir, is_set, items)
            elif choice.isdigit():
                idx = int(choice)
                total_dirs = len(items['dirs'])
                if 1 <= idx <= total_dirs:
                    # Navigate to directory
                    dir_name = items['dirs'][idx - 1][0]
                    current_dir = current_dir / dir_name
                elif total_dirs < idx <= total_dirs + len(items['files']):
                    # Select file
                    file_idx = idx - total_dirs - 1
                    file_name = items['files'][file_idx][0]
                    file_path = current_dir / file_name
                    self._file_action_menu(file_path, is_set)

    def _list_directory_with_cache(self, directory: Path, is_set: bool) -> dict:
        """List directory contents with cache status."""
        result = {
            'dirs': [],
            'files': [],
            'uncached_count': 0
        }

        if not directory.exists():
            return result

        # Get all cached paths for quick lookup
        if is_set:
            cached_paths = set(self.cache.get_all_cached_sets())
        else:
            # For tracks, we need to check individually (no bulk method)
            cached_paths = set()

        entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        for entry in entries:
            if entry.name.startswith('.'):
                continue

            if entry.is_dir():
                # Count audio files in subdirectory
                audio_count = 0
                cached_count = 0
                for audio_file in entry.rglob('*'):
                    if audio_file.suffix.lower() in self.AUDIO_EXTENSIONS:
                        audio_count += 1
                        abs_path = str(audio_file.absolute())
                        if is_set:
                            if abs_path in cached_paths:
                                cached_count += 1
                        else:
                            if self.cache.get_track(abs_path):
                                cached_count += 1
                result['dirs'].append((entry.name, audio_count, cached_count))

            elif entry.suffix.lower() in self.AUDIO_EXTENSIONS:
                abs_path = str(entry.absolute())
                if is_set:
                    is_cached = abs_path in cached_paths
                    cached_data = self.cache.get_set(abs_path) if is_cached else None
                else:
                    cached_data = self.cache.get_track(abs_path)
                    is_cached = cached_data is not None

                # Get duration from cache if available
                duration = ""
                if cached_data:
                    dur_sec = getattr(cached_data, 'duration_sec', 0)
                    if dur_sec > 0:
                        mins = int(dur_sec // 60)
                        secs = int(dur_sec % 60)
                        duration = f"{mins}:{secs:02d}"

                result['files'].append((entry.name, is_cached, duration))
                if not is_cached:
                    result['uncached_count'] += 1

        return result

    def _file_action_menu(self, file_path: Path, is_set: bool):
        """Actions for a selected file."""
        while True:
            clear_screen()
            print_header(file_path.name)
            print(f"{Colors.DIM}{file_path.parent}{Colors.RESET}\n")

            # Check cache status
            abs_path = str(file_path.absolute())
            if is_set:
                cached = self.cache.get_set(abs_path)
            else:
                cached = self.cache.get_track(abs_path)

            if cached:
                print(f"{Colors.GREEN}✓ В кеше{Colors.RESET}\n")
                # Show basic info
                dur_sec = getattr(cached, 'duration_sec', 0)
                if dur_sec > 0:
                    mins = int(dur_sec // 60)
                    secs = int(dur_sec % 60)
                    print(f"  Длительность: {mins}:{secs:02d}")
                tempo = getattr(cached, 'tempo', 0)
                if tempo > 0:
                    print(f"  Темп: {tempo:.1f} BPM")
                n_drops = getattr(cached, 'n_drops', 0)
                if n_drops >= 0:
                    print(f"  Дропов: {n_drops}")
                print()
            else:
                print(f"{Colors.DIM}○ Не проанализирован{Colors.RESET}\n")

            print_menu_item("1", "Анализировать" if not cached else "Переанализировать")
            if cached:
                print_menu_item("2", "Показать детали")
                print_menu_item("3", "Удалить из кеша")
            print_divider()
            print_menu_item("0", "Назад")
            print()

            choice = get_input("Выбор")

            if choice == "0":
                break
            elif choice == "1":
                self._analyze_single_file(abs_path, is_set, force=cached is not None)
                get_input("\nНажмите Enter")
            elif choice == "2" and cached:
                self._show_cached_details(cached, is_set)
                get_input("\nНажмите Enter")
            elif choice == "3" and cached:
                if is_set:
                    self.cache.invalidate_set(abs_path)
                else:
                    self.cache.invalidate_track(abs_path)
                print(f"{Colors.GREEN}Удалено из кеша{Colors.RESET}")
                get_input("\nНажмите Enter")
                break

    def _show_cached_details(self, cached, is_set: bool):
        """Show detailed cached info."""
        clear_screen()
        print_header("ДЕТАЛИ АНАЛИЗА")

        data = cached.to_dict() if hasattr(cached, 'to_dict') else {}

        for key, value in data.items():
            if key.startswith('_') or key == 'file_path':
                continue
            # Format key
            display_key = key.replace('_', ' ').title()
            # Format value
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            elif isinstance(value, list):
                display_value = f"[{len(value)} элементов]"
            elif isinstance(value, dict):
                display_value = f"{{{len(value)} ключей}}"
            else:
                display_value = str(value)

            print(f"  {display_key}: {display_value}")

    def _analyze_single_file(self, path: str, is_set: bool, force: bool = False):
        """Unified single file analysis using services."""
        service = AnalysisService()

        print(f"\n{Colors.CYAN}Анализ: {Path(path).name}{Colors.RESET}\n")

        try:
            if is_set:
                # Progress callback for set analysis
                def progress_cb(progress: float, stage: str, message: str):
                    bar_width = 30
                    filled = int(bar_width * progress)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(f"\r  [{bar}] {int(progress*100):3d}% {message[:40]:<40}", end="", flush=True)
                    if progress >= 1.0:
                        print()

                result = service.analyze_set(
                    path,
                    use_cache=not force,
                    force=force,
                    on_progress=progress_cb,
                    verbose=True,
                )

                if result.success:
                    print(f"\n{Colors.GREEN}Готово!{Colors.RESET}")
                    print(f"  Длительность: {result.duration_sec/60:.1f} мин")
                    print(f"  Переходов: {result.n_transitions}")
                    print(f"  Дропов: {result.total_drops}")
                    print(f"  Время: {result.processing_time_sec:.1f}с")
                else:
                    print(f"\n{Colors.RED}Ошибка: {result.error}{Colors.RESET}")
            else:
                result = service.analyze_track(path, use_cache=not force)
                if result.success:
                    print(f"{Colors.GREEN}Готово!{Colors.RESET}")
                    print(f"  Темп: {result.tempo:.1f} BPM")
                    print(f"  Дропов: {result.n_drops}")
                else:
                    print(f"{Colors.RED}Ошибка: {result.error}{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}Ошибка: {e}{Colors.RESET}")

    def _analyze_uncached(self, directory: Path, is_set: bool, items: dict):
        """Analyze all uncached files in directory using AnalysisService."""
        clear_screen()
        print_header("ПАКЕТНЫЙ АНАЛИЗ")

        # Collect uncached files
        uncached_files = []
        for name, is_cached, _ in items['files']:
            if not is_cached:
                uncached_files.append(directory / name)

        # Check subdirectories
        for dir_name, total, cached in items['dirs']:
            if cached < total:
                subdir = directory / dir_name
                for f in subdir.rglob('*'):
                    if f.suffix.lower() in self.AUDIO_EXTENSIONS:
                        abs_path = str(f.absolute())
                        if is_set:
                            if not self.cache.get_set(abs_path):
                                uncached_files.append(f)
                        else:
                            if not self.cache.get_track(abs_path):
                                uncached_files.append(f)

        total = len(uncached_files)
        print(f"Файлов для анализа: {total}\n")

        if total == 0:
            print(f"{Colors.GREEN}Все файлы уже в кеше!{Colors.RESET}")
            get_input("\nНажмите Enter")
            return

        confirm = get_input(f"Начать анализ {total} файлов? [y/N]", "n")
        if confirm.lower() not in ("y", "да"):
            return

        print()

        # Use AnalysisService for batch analysis
        service = AnalysisService()

        if is_set:
            # Set analysis with BatchProgressDisplay
            display = BatchProgressDisplay(total_files=total, max_log_lines=6)
            display.render()

            for i, file_path in enumerate(uncached_files, 1):
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                display.start_file(file_path.name, size_mb=file_size_mb)

                try:
                    abs_path = str(file_path.absolute())
                    result = service.analyze_set(abs_path, use_cache=True)

                    if result.success:
                        display.update_batch(done=i, cached=0)
                        display.log_result(
                            file_path.name,
                            transitions=result.n_transitions,
                            drops=result.total_drops,
                            time_sec=result.processing_time_sec
                        )
                    else:
                        display.log_result(file_path.name, error=result.error[:30] if result.error else "Unknown")
                except Exception as e:
                    display.log_result(file_path.name, error=str(e)[:30])

            display.complete()
        else:
            # Track analysis - simple progress
            for i, file_path in enumerate(uncached_files, 1):
                print(f"\n{Colors.CYAN}[{i}/{total}] {file_path.name}{Colors.RESET}")
                try:
                    abs_path = str(file_path.absolute())
                    result = service.analyze_track(abs_path, use_cache=True)
                    if result.success:
                        print(f"  {Colors.GREEN}✓{Colors.RESET} {result.tempo:.0f} BPM, {result.n_drops} drops")
                    else:
                        print(f"  {Colors.RED}✗{Colors.RESET} {result.error}")
                except Exception as e:
                    print(f"  {Colors.RED}✗{Colors.RESET} {e}")

        print(f"\n{Colors.GREEN}Анализ завершён!{Colors.RESET}")
        get_input("\nНажмите Enter")

    def _show_cache_stats(self):
        """Show cache statistics."""
        clear_screen()
        print_header("СТАТИСТИКА КЕША")

        stats = self.cache.get_stats()

        print(f"  Сетов в кеше: {stats.get('set_count', 0)}")
        print(f"  Всего записей: {stats.get('total_entries', 0)}")

        # Count files in directories
        tracks_count = 0
        tracks_cached = 0
        for f in self.tracks_dir.rglob('*'):
            if f.suffix.lower() in self.AUDIO_EXTENSIONS:
                tracks_count += 1
                if self.cache.get_track(str(f.absolute())):
                    tracks_cached += 1

        sets_count = 0
        sets_cached = 0
        cached_set_paths = set(self.cache.get_all_cached_sets())
        for f in self.sets_dir.rglob('*'):
            if f.suffix.lower() in self.AUDIO_EXTENSIONS:
                sets_count += 1
                if str(f.absolute()) in cached_set_paths:
                    sets_cached += 1

        print()
        print(f"  Каталог треков: {tracks_cached}/{tracks_count} проанализировано")
        print(f"  Каталог сетов: {sets_cached}/{sets_count} проанализировано")

        get_input("\nНажмите Enter")

    # ==================== ПОМОЩЬ ====================
    # ==================== КЕШ ====================
    def cache_menu(self):
        while True:
            clear_screen()
            print_header("УПРАВЛЕНИЕ КЕШЕМ")

            # Показываем статистику
            stats = self.cache.get_stats()
            print(f"  {Colors.CYAN}Директория:{Colors.RESET} {self.cache.cache_dir}")
            print(f"  {Colors.CYAN}Сеты:{Colors.RESET} {stats.get('set_count', 0)}")
            print(f"  {Colors.CYAN}Предсказания:{Colors.RESET} {stats.get('prediction_count', 0)}")
            print(f"  {Colors.CYAN}Features:{Colors.RESET} {stats.get('feature_count', 0)}")
            print(f"  {Colors.CYAN}STFT файлы:{Colors.RESET} {stats.get('stft_count', 0)}")
            print(f"  {Colors.CYAN}Размер:{Colors.RESET} {stats.get('total_size_mb', 0):.1f} MB")
            print()

            print_divider()
            print_menu_item("1", "Очистить всё")
            print_menu_item("2", "Очистить анализы сетов")
            print_menu_item("3", "Очистить предсказания")
            print_menu_item("4", "Очистить features")
            print_menu_item("5", "Очистить STFT кеш")
            print_menu_item("6", "Очистить старые записи")
            print_menu_item("7", "Инвалидировать DJ")
            print_divider()
            print_menu_item("R", "Обновить статистику")
            print_menu_item("0", "Назад")
            print()

            choice = get_input("Выбор").lower()

            if choice == "1":
                confirm = get_input(f"{Colors.RED}Удалить ВСЕ данные кеша? [y/N]{Colors.RESET}", "n")
                if confirm.lower() == "y":
                    self.cache.clear_all()
                    print(f"{Colors.GREEN}Кеш очищен{Colors.RESET}")
                    get_input("\nНажмите Enter")
            elif choice == "2":
                self.cache.clear_sets()
                print(f"{Colors.GREEN}Анализы сетов очищены{Colors.RESET}")
                get_input("\nНажмите Enter")
            elif choice == "3":
                self.cache.clear_predictions()
                print(f"{Colors.GREEN}Предсказания очищены{Colors.RESET}")
                get_input("\nНажмите Enter")
            elif choice == "4":
                import shutil
                if self.cache.features_dir.exists():
                    shutil.rmtree(self.cache.features_dir)
                    self.cache.features_dir.mkdir(exist_ok=True)
                print(f"{Colors.GREEN}Features кеш очищен{Colors.RESET}")
                get_input("\nНажмите Enter")
            elif choice == "5":
                import shutil
                if self.cache.stft_dir.exists():
                    shutil.rmtree(self.cache.stft_dir)
                    self.cache.stft_dir.mkdir(exist_ok=True)
                print(f"{Colors.GREEN}STFT кеш очищен{Colors.RESET}")
                get_input("\nНажмите Enter")
            elif choice == "6":
                days = get_int_input("Удалить записи старше (дней)", 30, 1, 365)
                self.cache.cleanup(max_age_days=days)
                print(f"{Colors.GREEN}Записи старше {days} дней удалены{Colors.RESET}")
                get_input("\nНажмите Enter")
            elif choice == "7":
                profiles = self.cache.get_all_dj_profiles()
                if profiles:
                    print(f"\n{Colors.CYAN}DJ профили:{Colors.RESET}")
                    for i, name in enumerate(profiles, 1):
                        print(f"  {i}. {name}")
                    dj_name = get_input("\nИмя DJ для инвалидации")
                    if dj_name:
                        count = self.cache.invalidate_by_dj(dj_name)
                        print(f"{Colors.GREEN}Инвалидировано записей: {count}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}Нет DJ профилей{Colors.RESET}")
                get_input("\nНажмите Enter")
            elif choice == "r":
                continue  # Просто обновить экран
            elif choice == "0":
                break

    def show_help(self):
        clear_screen()
        print_header("ПОМОЩЬ")

        print(f"{Colors.CYAN}DJ Tools{Colors.RESET} - набор инструментов для анализа музыки\n")

        print(f"{Colors.BOLD}Работа с треками:{Colors.RESET}")
        print("  Анализ треков: BPM, дропы, энергетический профиль")
        print("  Обнаружение beat grid, классификация типов дропов")
        print("  Опционально: классификация по зонам (Yellow/Green/Purple)")
        print()

        print(f"{Colors.BOLD}Работа с сетами:{Colors.RESET}")
        print("  Анализ DJ миксов: обнаружение переходов и дропов")
        print("  Определение структуры сета и стиля сведения")
        print("  Тот же pipeline что и для треков")
        print()

        print(f"{Colors.BOLD}DJ профили:{Colors.RESET}")
        print("  Агрегированные характеристики стиля DJ")
        print("  Создаются из проанализированных сетов")
        print()

        print(f"{Colors.BOLD}Треклисты:{Colors.RESET}")
        print("  Генерация плана сета на основе DJ профиля")
        print("  Экспорт в M3U плейлист")
        print()

        print(f"{Colors.BOLD}Скачивание:{Colors.RESET}")
        print("  Загрузка треков/сетов с SoundCloud, YouTube, Mixcloud, Bandcamp")
        print("  Пакетная загрузка плейлистов с параллельной загрузкой (до 10 потоков)")
        print("  Автоматическое определение исполнителя из URL")
        print("  Проверка дубликатов перед загрузкой")
        print("  Сохранение в оригинальном формате (opus/m4a) или MP3")
        print()

        print(f"{Colors.BOLD}Обзор файлов:{Colors.RESET}")
        print("  Навигация по каталогам треков и сетов")
        print("  Просмотр статуса кеша (✓ проанализирован / ○ нет)")
        print("  Выбор файлов для анализа, пакетный анализ")
        print("  Настройка каталогов сохраняется между сессиями")
        print()

        print(f"{Colors.BOLD}Кеш:{Colors.RESET}")
        print("  Просмотр статистики кеша (размер, количество записей)")
        print("  Очистка всего кеша или его частей (сеты, features, STFT)")
        print("  Инвалидация записей по DJ или по возрасту")
        print("  CLI: python main.py cache --stats")
        print()

        print(f"{Colors.DIM}CLI команды:{Colors.RESET}")
        print("  python main.py analyze <file>     - анализ файла (трек/сет)")
        print("  python main.py classify -f <file> - классификация энергозоны")
        print("  python main.py generate --help    - генерация сета")
        print()
        print(f"{Colors.DIM}Скачивание (CLI):{Colors.RESET}")
        print("  python scripts/download_dj_set.py <URL>         - скачать трек/сет")
        print("  python scripts/download_dj_set.py <PLAYLIST>    - скачать плейлист")
        print("  python scripts/download_dj_set.py <URL> -p 5    - 5 параллельных загрузок")
        print("  python scripts/download_dj_set.py --list        - показать загруженные")

        get_input("\nНажмите Enter")


def main():
    try:
        menu = InteractiveMenu()
        menu.run()
        print(f"\n{Colors.CYAN}До встречи!{Colors.RESET}\n")
    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYAN}До встречи!{Colors.RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
