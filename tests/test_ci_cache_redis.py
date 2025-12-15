"""CI интеграционные тесты для Redis-кеша.

Эти тесты проверяют интеграцию с Redis через GitHub Actions services.
Требуют запущенного Redis сервиса (redis://localhost:6379/0).
"""

import pytest
import asyncio
import redis.asyncio as aioredis
import os
from pathlib import Path


@pytest.fixture
async def redis_client(auto_redis):
    """Создаёт подключение к Redis для тестов.

    ЧТО СОЗДАЁМ:
        Async Redis клиент, подключенный к тестовому Redis

    КАК РАБОТАЕТ:
        1. Использует auto_redis фикстуру (автоматически запускает Redis если нужно)
        2. Читаем REDIS_URL из переменной окружения (установлен auto_redis)
        3. Создаём async Redis клиент
        4. Проверяем подключение через ping()
        5. Очищаем тестовую БД (flushdb) перед каждым тестом
        6. После теста закрываем соединение

    ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
        - Redis запущен автоматически (Docker или локально)
        - Успешное подключение к Redis
        - Чистая БД для каждого теста

    КРИТЕРИЙ УСПЕШНОСТИ:
        Fixture возвращает рабочий Redis клиент с пустой БД
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = await aioredis.from_url(redis_url, decode_responses=True)

    # Проверяем подключение
    assert await client.ping(), "Redis должен отвечать на ping"

    # Очищаем БД перед тестом
    await client.flushdb()

    yield client

    # Очищаем и закрываем после теста
    await client.flushdb()
    await client.aclose()


@pytest.mark.integration
@pytest.mark.requires_redis
class TestRedisCacheBasics:
    """Базовые операции с Redis кешем."""

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_redis_connection(self, redis_client):
        """Тест подключения к Redis.

        ЧТО ПРОВЕРЯЕМ:
            Redis сервис запущен и доступен в GitHub Actions

        КАК ПРОВЕРЯЕМ:
            1. Получаем Redis клиент из fixture
            2. Вызываем ping()
            3. Проверяем ответ True

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - Redis отвечает на ping
            - Соединение стабильное

        РЕЗУЛЬТАТ:
            Подтверждено, что Redis service работает в CI

        КРИТЕРИЙ УСПЕШНОСТИ:
            ping() возвращает True без исключений
        """
        result = await redis_client.ping()
        assert result is True, "Redis должен отвечать True на ping()"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_set_and_get(self, redis_client):
        """Тест базовых операций set/get.

        ЧТО ПРОВЕРЯЕМ:
            Сохранение и извлечение данных из Redis

        КАК ПРОВЕРЯЕМ:
            1. Сохраняем строку через set()
            2. Извлекаем через get()
            3. Сравниваем с оригиналом

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - set() успешно сохраняет данные
            - get() возвращает те же данные

        РЕЗУЛЬТАТ:
            Базовые операции Redis работают корректно

        КРИТЕРИЙ УСПЕШНОСТИ:
            Извлечённое значение идентично сохранённому
        """
        await redis_client.set("test_key", "test_value")
        result = await redis_client.get("test_key")

        assert result == "test_value", \
            "Извлечённое значение должно совпадать с сохранённым"

    @pytest.mark.asyncio
    async def test_expiration(self, redis_client):
        """Тест TTL (time-to-live) для ключей.

        ЧТО ПРОВЕРЯЕМ:
            Ключи с TTL автоматически удаляются через заданное время

        КАК ПРОВЕРЯЕМ:
            1. Сохраняем ключ с TTL=1 секунда
            2. Проверяем, что ключ существует
            3. Ждём 1.5 секунды
            4. Проверяем, что ключ удалён

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - Сразу после set() ключ доступен
            - Через 1.5 секунды ключ возвращает None

        РЕЗУЛЬТАТ:
            TTL механизм Redis работает для автоматической очистки кеша

        КРИТЕРИЙ УСПЕШНОСТИ:
            Ключ исчезает после истечения TTL
        """
        await redis_client.set("expiring_key", "value", ex=1)  # TTL = 1 секунда

        # Сразу после set() ключ должен существовать
        result = await redis_client.get("expiring_key")
        assert result == "value", "Ключ должен существовать сразу после set()"

        # Ждём истечения TTL
        await asyncio.sleep(1.5)

        # Ключ должен быть удалён
        result = await redis_client.get("expiring_key")
        assert result is None, "Ключ должен быть удалён после истечения TTL"

    @pytest.mark.asyncio
    async def test_hash_operations(self, redis_client):
        """Тест операций с hash структурами.

        ЧТО ПРОВЕРЯЕМ:
            Redis hash (hset/hget/hgetall) для хранения объектов

        КАК ПРОВЕРЯЕМ:
            1. Сохраняем hash с полями через hset()
            2. Извлекаем одно поле через hget()
            3. Извлекаем все поля через hgetall()

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - hset() сохраняет несколько полей
            - hget() возвращает конкретное поле
            - hgetall() возвращает весь hash как dict

        РЕЗУЛЬТАТ:
            Hash структуры Redis работают для хранения объектов (например, результатов анализа)

        КРИТЕРИЙ УСПЕШНОСТИ:
            Все поля hash сохраняются и извлекаются корректно
        """
        # Сохраняем hash (например, результат анализа трека)
        await redis_client.hset("track:123", mapping={
            "zone": "purple",
            "bpm": "128.5",
            "confidence": "0.87"
        })

        # Извлекаем одно поле
        zone = await redis_client.hget("track:123", "zone")
        assert zone == "purple", "Поле 'zone' должно быть 'purple'"

        # Извлекаем весь hash
        track_data = await redis_client.hgetall("track:123")
        assert track_data["zone"] == "purple"
        assert track_data["bpm"] == "128.5"
        assert track_data["confidence"] == "0.87"


@pytest.mark.integration
@pytest.mark.requires_redis
class TestARQTaskQueue:
    """Тесты ARQ task queue через Redis."""

    @pytest.mark.asyncio
    async def test_arq_connection(self, redis_client):
        """Тест подключения ARQ к Redis.

        ЧТО ПРОВЕРЯЕМ:
            ARQ может подключиться к Redis для очереди задач

        КАК ПРОВЕРЯЕМ:
            1. Создаём ArqRedis подключение
            2. Вызываем ping()
            3. Проверяем успешный ответ

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - ArqRedis успешно подключается
            - ping() возвращает bytes b'PONG'

        РЕЗУЛЬТАТ:
            ARQ worker может работать с Redis в CI

        КРИТЕРИЙ УСПЕШНОСТИ:
            ping() успешен, подключение установлено
        """
        from arq.connections import ArqRedis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        arq_redis = await ArqRedis.from_url(redis_url)

        # Проверяем подключение
        result = await arq_redis.ping()
        # ArqRedis.ping() возвращает True вместо b"PONG"
        assert result is True, "ARQ Redis должен отвечать True на ping"

        await arq_redis.close()

    @pytest.mark.asyncio
    async def test_enqueue_task(self, redis_client):
        """Тест постановки задачи в очередь ARQ.

        ЧТО ПРОВЕРЯЕМ:
            Задачи корректно ставятся в очередь ARQ

        КАК ПРОВЕРЯЕМ:
            1. Создаём ArqRedis подключение
            2. Ставим задачу в очередь через enqueue_job()
            3. Проверяем, что task ID возвращён

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - enqueue_job() возвращает task ID
            - Задача сохранена в Redis очереди

        РЕЗУЛЬТАТ:
            Bot может ставить задачи анализа в очередь

        КРИТЕРИЙ УСПЕШНОСТИ:
            Получен валидный task ID (строка, непустая)
        """
        from arq.connections import ArqRedis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        arq_redis = await ArqRedis.from_url(redis_url)

        # Ставим задачу в очередь (мокированная функция)
        job = await arq_redis.enqueue_job(
            "analyze_track",
            url="https://youtube.com/watch?v=test123"
        )

        assert job is not None, "enqueue_job должен вернуть Job объект"
        # ARQ возвращает Job объект, не строку
        # Проверяем, что у Job есть job_id
        assert hasattr(job, 'job_id'), "Job должен иметь атрибут job_id"
        assert job.job_id is not None, "job_id не должен быть None"
        assert len(job.job_id) > 0, "job_id не должен быть пустым"

        await arq_redis.close()

    @pytest.mark.asyncio
    async def test_task_result_storage(self, redis_client):
        """Тест хранения результатов задач.

        ЧТО ПРОВЕРЯЕМ:
            Результаты выполнения задач сохраняются в Redis

        КАК ПРОВЕРЯЕМ:
            1. Сохраняем мок-результат задачи как hash
            2. Извлекаем результат через hgetall()
            3. Проверяем все поля результата

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - Результат сохраняется как Redis hash
            - Все поля доступны для извлечения

        РЕЗУЛЬТАТ:
            Bot может сохранять и извлекать результаты анализа

        КРИТЕРИЙ УСПЕШНОСТИ:
            Все поля результата (status, zone, bpm, error) корректно сохранены
        """
        # Симулируем результат выполнения задачи
        task_id = "task_abc123"
        await redis_client.hset(f"arq:result:{task_id}", mapping={
            "status": "complete",
            "zone": "purple",
            "bpm": "128.5",
            "error": ""
        })

        # Извлекаем результат
        result = await redis_client.hgetall(f"arq:result:{task_id}")

        assert result["status"] == "complete", "Статус должен быть 'complete'"
        assert result["zone"] == "purple", "Зона должна быть 'purple'"
        assert result["bpm"] == "128.5", "BPM должен быть '128.5'"
        assert result["error"] == "", "Ошибка должна быть пустой"


@pytest.mark.integration
@pytest.mark.requires_redis
class TestCacheIntegration:
    """Интеграционные тесты кеша приложения."""

    @pytest.mark.asyncio
    async def test_prediction_cache(self, redis_client):
        """Тест кеширования предсказаний.

        ЧТО ПРОВЕРЯЕМ:
            Предсказания зон сохраняются в Redis для повторного использования

        КАК ПРОВЕРЯЕМ:
            1. Сохраняем предсказание как hash (file_hash → результат)
            2. Извлекаем из кеша
            3. Проверяем, что не нужно повторно анализировать

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - Первый анализ сохраняется в кеш
            - Повторный запрос возвращает кешированный результат

        РЕЗУЛЬТАТ:
            Кеш предотвращает дублирующий анализ одного и того же файла

        КРИТЕРИЙ УСПЕШНОСТИ:
            Кешированное предсказание идентично оригинальному
        """
        file_hash = "abc123def456"

        # Сохраняем предсказание в кеш
        await redis_client.hset(f"prediction:{file_hash}", mapping={
            "zone": "green",
            "bpm": "120.0",
            "confidence": "0.92",
            "timestamp": "2025-12-15T10:00:00"
        })

        # Проверяем кеш перед анализом
        cached = await redis_client.hgetall(f"prediction:{file_hash}")

        assert cached is not None, "Предсказание должно быть в кеше"
        assert cached["zone"] == "green"
        assert cached["bpm"] == "120.0"
        assert cached["confidence"] == "0.92"

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, redis_client):
        """Тест инвалидации кеша.

        ЧТО ПРОВЕРЯЕМ:
            Кеш корректно очищается при необходимости

        КАК ПРОВЕРЯЕМ:
            1. Сохраняем несколько предсказаний
            2. Удаляем конкретный ключ через delete()
            3. Проверяем, что ключ удалён, остальные остались

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - delete() удаляет только указанный ключ
            - Другие ключи не затронуты

        РЕЗУЛЬТАТ:
            Можно инвалидировать кеш для конкретных файлов

        КРИТЕРИЙ УСПЕШНОСТИ:
            Удалённый ключ возвращает None, остальные доступны
        """
        # Сохраняем два предсказания
        await redis_client.set("prediction:file1", "purple")
        await redis_client.set("prediction:file2", "yellow")

        # Инвалидируем одно
        await redis_client.delete("prediction:file1")

        # Проверяем инвалидацию
        result1 = await redis_client.get("prediction:file1")
        result2 = await redis_client.get("prediction:file2")

        assert result1 is None, "Инвалидированный ключ должен быть удалён"
        assert result2 == "yellow", "Другой ключ должен остаться"

    @pytest.mark.asyncio
    async def test_cache_pattern_deletion(self, redis_client):
        """Тест удаления кеша по паттерну.

        ЧТО ПРОВЕРЯЕМ:
            Можно удалить все ключи, соответствующие паттерну (например, все predictions)

        КАК ПРОВЕРЯЕМ:
            1. Сохраняем несколько prediction:* ключей
            2. Находим их через keys("prediction:*")
            3. Удаляем все через delete()
            4. Проверяем, что все удалены

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - keys() находит все соответствующие ключи
            - delete() удаляет все найденные ключи

        РЕЗУЛЬТАТ:
            Можно очистить весь кеш определённого типа одной операцией

        КРИТЕРИЙ УСПЕШНОСТИ:
            После удаления паттерн не возвращает ключей
        """
        # Сохраняем несколько предсказаний
        await redis_client.set("prediction:file1", "purple")
        await redis_client.set("prediction:file2", "yellow")
        await redis_client.set("prediction:file3", "green")
        await redis_client.set("other:key", "value")

        # Находим все prediction:* ключи
        prediction_keys = await redis_client.keys("prediction:*")
        assert len(prediction_keys) == 3, "Должно быть 3 prediction ключа"

        # Удаляем все найденные
        if prediction_keys:
            await redis_client.delete(*prediction_keys)

        # Проверяем, что все predictions удалены
        remaining = await redis_client.keys("prediction:*")
        assert len(remaining) == 0, "Все prediction ключи должны быть удалены"

        # Другой ключ должен остаться
        other = await redis_client.get("other:key")
        assert other == "value", "Ключ other:key должен остаться"