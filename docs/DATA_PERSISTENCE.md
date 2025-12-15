# Data Persistence Guide

## Проблема

При обновлении приложения (deploy) данные НЕ должны перетираться:
- ❌ `cache/predictions.db` — база данных треков, DJ профилей, сет-планов
- ❌ `cache/stft/` — кешированные STFT матрицы
- ❌ `cache/features/` — ML features

**Если БД в папке с кодом → она перетрётся при деплое!**

---

## Решение: Разделение Code и Data

```
/app/               ← Код (обновляется при deploy)
/data/              ← Данные (НЕ трогаются при deploy)
    └── cache/
        ├── predictions.db
        ├── stft/
        └── features/
```

---

## Реализация

### Локальная разработка

По умолчанию данные в `./cache/`:

```bash
# Всё работает как раньше
python main.py
# БД: ./cache/predictions.db
```

### Production (Docker)

#### 1. Установить переменную окружения

```bash
export DATA_DIR=/data
```

Или в `.env`:
```
DATA_DIR=/data
```

#### 2. Docker Compose (рекомендуется)

```yaml
services:
  mood-classifier:
    environment:
      - DATA_DIR=/data
    volumes:
      - mood-data:/data  # ← Персистентный volume

volumes:
  mood-data:  # ← Создаётся автоматически
```

**Обновление приложения:**
```bash
docker-compose pull       # Новый образ
docker-compose up -d      # Рестарт
# ✅ Данные в mood-data сохранены!
```

#### 3. Docker (без compose)

```bash
# Создать volume
docker volume create mood-data

# Запуск
docker run -d \
  -e DATA_DIR=/data \
  -v mood-data:/data \
  mood-classifier

# Обновление
docker pull mood-classifier:latest
docker stop <container>
docker rm <container>
docker run -d -e DATA_DIR=/data -v mood-data:/data mood-classifier:latest
# ✅ БД сохранена в mood-data volume
```

---

## Проверка

### Где хранятся данные?

```python
from app.core.config import get_data_dir, get_db_path

print(get_data_dir())  # → /data (если DATA_DIR=/data)
print(get_db_path())   # → /data/predictions.db
```

### Инспекция Docker volume

```bash
# Где физически лежат файлы?
docker volume inspect mood-data

# Размер БД
docker run --rm -v mood-data:/data alpine du -sh /data
```

---

## Миграция существующих данных

Если БД уже есть в `./cache/`:

```bash
# Скопировать в volume
docker run --rm \
  -v $(pwd)/cache:/source \
  -v mood-data:/dest \
  alpine cp -a /source/. /dest/
```

---

## Бэкапы

### Бэкап БД

```bash
# Вариант 1: Копирование из volume
docker run --rm \
  -v mood-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/db-$(date +%Y%m%d).tar.gz -C /data .

# Вариант 2: SQLite dump
docker exec <container> sqlite3 /data/predictions.db .dump > backup.sql
```

### Восстановление

```bash
# Из tar.gz
docker run --rm \
  -v mood-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/db-20250101.tar.gz -C /data

# Из SQL dump
docker exec -i <container> sqlite3 /data/predictions.db < backup.sql
```

---

## Troubleshooting

### БД перетёрлась при деплое

**Причина:** DATA_DIR не установлен, БД в `./cache/`

**Решение:**
1. Установить `DATA_DIR=/data`
2. Примонтировать volume
3. Восстановить из бэкапа

### Permission denied

**Причина:** Docker контейнер не имеет прав на `/data`

**Решение:**
```bash
# В Dockerfile
RUN mkdir -p /data && chmod 777 /data
```

Или volume permissions:
```yaml
volumes:
  mood-data:
    driver: local
    driver_opts:
      type: none
      device: /path/on/host
      o: bind,uid=1000,gid=1000
```

---

## Best Practices

1. **Всегда используйте named volumes** для production
2. **Делайте бэкапы БД** перед обновлением
3. **Тестируйте восстановление** из бэкапов
4. **Мониторьте размер** `/data` volume
5. **Логируйте DATA_DIR** при старте приложения

---

## См. также

- [Dockerfile.data-persistence](../Dockerfile.data-persistence)
- [docker-compose.example.yml](../docker-compose.example.yml)
- [.env.example](../.env.example)