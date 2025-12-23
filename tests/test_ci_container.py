"""CI интеграционные тесты для Docker контейнера.

Тестирует РЕАЛЬНОЕ окружение внутри Docker:
    - Runtime зависимости (yt-dlp, ffmpeg)
    - Filesystem permissions (/data, /app/downloads)
    - ARQ worker connectivity

Эти тесты ДОЛЖНЫ запускаться ВНУТРИ Docker контейнера!
"""

import pytest
import subprocess
import os
import shutil


def is_inside_container():
    """Check if running inside a Docker container."""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == '1'


# Skip entire module if not in container
pytestmark = pytest.mark.skipif(
    not is_inside_container(),
    reason="CI container tests require Docker environment"
)


@pytest.mark.integration
@pytest.mark.container
class TestContainerDependencies:
    """Тесты runtime зависимостей в контейнере."""

    def test_ytdlp_installed(self):
        """Тест: yt-dlp установлен и работает.

        ЧТО ПРОВЕРЯЕМ:
            yt-dlp доступен в PATH и возвращает версию

        КАК ПРОВЕРЯЕМ:
            subprocess.run(['yt-dlp', '--version'])

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            Exit code 0, версия в stdout

        КРИТЕРИЙ УСПЕШНОСТИ:
            yt-dlp работает
        """
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"yt-dlp должен быть установлен. stderr: {result.stderr}"
        assert result.stdout.strip(), \
            "yt-dlp должен вернуть версию"

    def test_ffmpeg_installed(self):
        """Тест: ffmpeg установлен и работает.

        ЧТО ПРОВЕРЯЕМ:
            ffmpeg доступен в PATH

        КАК ПРОВЕРЯЕМ:
            subprocess.run(['ffmpeg', '-version'])

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            Exit code 0

        КРИТЕРИЙ УСПЕШНОСТИ:
            ffmpeg работает
        """
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"ffmpeg должен быть установлен. stderr: {result.stderr}"

    def test_ffprobe_installed(self):
        """Тест: ffprobe установлен (часть ffmpeg).

        ЧТО ПРОВЕРЯЕМ:
            ffprobe доступен для проверки аудио файлов

        КАК ПРОВЕРЯЕМ:
            shutil.which('ffprobe')

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            Путь к ffprobe найден

        КРИТЕРИЙ УСПЕШНОСТИ:
            ffprobe в PATH
        """
        ffprobe_path = shutil.which("ffprobe")
        assert ffprobe_path is not None, \
            "ffprobe должен быть установлен (часть ffmpeg)"


@pytest.mark.integration
@pytest.mark.container
class TestContainerFilesystem:
    """Тесты filesystem внутри контейнера."""

    def test_data_dir_exists(self):
        """Тест: /data директория существует.

        ЧТО ПРОВЕРЯЕМ:
            DATA_DIR директория доступна

        КАК ПРОВЕРЯЕМ:
            os.path.exists(DATA_DIR)

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            Директория существует

        КРИТЕРИЙ УСПЕШНОСТИ:
            Путь существует
        """
        data_dir = os.getenv("DATA_DIR", "/data")
        assert os.path.exists(data_dir), \
            f"DATA_DIR={data_dir} должен существовать"

    def test_data_dir_writable(self):
        """Тест: /data директория записываема.

        ЧТО ПРОВЕРЯЕМ:
            Можно создать файл в DATA_DIR

        КАК ПРОВЕРЯЕМ:
            Создаём и удаляем тестовый файл

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            Запись успешна

        КРИТЕРИЙ УСПЕШНОСТИ:
            Файл создан и удалён
        """
        data_dir = os.getenv("DATA_DIR", "/data")
        test_file = os.path.join(data_dir, ".write_test")

        try:
            with open(test_file, "w") as f:
                f.write("test")
            assert os.path.exists(test_file), \
                "Тестовый файл должен быть создан"
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_downloads_dir_exists(self):
        """Тест: downloads директория существует.

        ЧТО ПРОВЕРЯЕМ:
            DOWNLOADS_DIR доступен для скачивания

        КАК ПРОВЕРЯЕМ:
            os.makedirs + os.path.exists

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            Директория существует или создана

        КРИТЕРИЙ УСПЕШНОСТИ:
            Путь существует
        """
        downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        assert os.path.exists(downloads_dir), \
            f"DOWNLOADS_DIR={downloads_dir} должен существовать"


@pytest.mark.integration
@pytest.mark.container
class TestContainerPythonEnv:
    """Тесты Python окружения в контейнере."""

    def test_app_imports(self):
        """Тест: app модули импортируются.

        ЧТО ПРОВЕРЯЕМ:
            Все основные модули доступны

        КАК ПРОВЕРЯЕМ:
            import app.modules...

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            Нет ImportError

        КРИТЕРИЙ УСПЕШНОСТИ:
            Все импорты успешны
        """
        # Core
        from app.core.config import settings
        assert settings is not None

        # Bot
        from app.modules.bot.routers.main import create_bot, create_dispatcher
        assert create_bot is not None
        assert create_dispatcher is not None

        # Analysis
        from app.modules.analysis.pipelines.set_analysis import SetAnalysisPipeline
        assert SetAnalysisPipeline is not None

        # ARQ Worker
        from app.services.arq_worker import WorkerSettings
        assert WorkerSettings is not None

    def test_arq_worker_functions_registered(self):
        """Тест: ARQ worker функции зарегистрированы.

        ЧТО ПРОВЕРЯЕМ:
            WorkerSettings.functions содержит все tasks

        КАК ПРОВЕРЯЕМ:
            len(WorkerSettings.functions)

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            2 функции: analyze_set_task, download_and_analyze_task

        КРИТЕРИЙ УСПЕШНОСТИ:
            Обе функции зарегистрированы
        """
        from app.services.arq_worker import WorkerSettings

        assert len(WorkerSettings.functions) == 2, \
            "Должно быть 2 ARQ функции"

        func_names = [f.__name__ for f in WorkerSettings.functions]
        assert "analyze_set_task" in func_names, \
            "analyze_set_task должен быть зарегистрирован"
        assert "download_and_analyze_task" in func_names, \
            "download_and_analyze_task должен быть зарегистрирован"


@pytest.mark.integration
@pytest.mark.container
class TestYtdlpFunctionality:
    """Тесты yt-dlp функционала."""

    def test_ytdlp_extractors_available(self):
        """Тест: yt-dlp поддерживает нужные сайты.

        ЧТО ПРОВЕРЯЕМ:
            SoundCloud и YouTube extractors доступны

        КАК ПРОВЕРЯЕМ:
            yt-dlp --list-extractors | grep

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            YouTube и SoundCloud в списке

        КРИТЕРИЙ УСПЕШНОСТИ:
            Оба extractor найдены
        """
        result = subprocess.run(
            ["yt-dlp", "--list-extractors"],
            capture_output=True,
            text=True,
            timeout=30
        )

        extractors = result.stdout.lower()

        assert "youtube" in extractors, \
            "YouTube extractor должен быть доступен"
        assert "soundcloud" in extractors, \
            "SoundCloud extractor должен быть доступен"

    def test_ytdlp_can_extract_info(self):
        """Тест: yt-dlp может извлечь информацию (без скачивания).

        ЧТО ПРОВЕРЯЕМ:
            --dump-json работает для известного URL

        КАК ПРОВЕРЯЕМ:
            yt-dlp --dump-json --skip-download

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            JSON с информацией о видео

        КРИТЕРИЙ УСПЕШНОСТИ:
            JSON парсится, есть title

        NOTE: Этот тест требует сети!
        """
        # Используем короткое публичное видео
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo"

        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--skip-download", test_url],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            pytest.skip(f"Network test failed (expected in CI): {result.stderr[:200]}")

        import json
        info = json.loads(result.stdout)
        assert "title" in info, "Должен быть title в JSON"
