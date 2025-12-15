# Deploy Checklist

## ✅ Pre-Deploy

- [ ] Установить `DATA_DIR=/data` в production environment
- [ ] Создать Docker volume для данных
- [ ] Сделать бэкап текущей БД (если есть)
- [ ] Проверить requirements.txt (БЕЗ pyrekordbox!)

## ✅ Deploy

```bash
# 1. Build новый образ
docker-compose build

# 2. Остановить старый контейнер (данные сохранятся в volume)
docker-compose down

# 3. Запустить новый контейнер
docker-compose up -d

# 4. Проверить логи
docker-compose logs -f
```

## ✅ Post-Deploy

- [ ] Проверить что БД доступна: `docker exec <container> ls -lh /data/predictions.db`
- [ ] Проверить количество треков в кеше
- [ ] Протестировать генерацию сета
- [ ] Проверить Telegram bot (если используется)

## 🔥 Rollback

Если что-то сломалось:

```bash
# Вернуться к предыдущему образу
docker-compose down
docker tag mood-classifier:latest mood-classifier:broken
docker tag mood-classifier:previous mood-classifier:latest
docker-compose up -d
```

## 📊 Мониторинг

```bash
# Размер данных
docker exec <container> du -sh /data

# Статистика БД
docker exec <container> sqlite3 /data/predictions.db \
  "SELECT 'Tracks:', COUNT(*) FROM track_metadata
   UNION ALL
   SELECT 'Sets:', COUNT(*) FROM set_analysis_results
   UNION ALL
   SELECT 'Profiles:', COUNT(*) FROM dj_profiles"
```

## 🚨 Emergency: БД потеряна

```bash
# 1. Восстановить из бэкапа
docker run --rm -v mood-data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/db-latest.tar.gz -C /data

# 2. Если бэкапа нет — пересинхронизировать из Rekordbox
docker exec <container> python scripts/sync_rekordbox_metadata.py
```

---

## Environment Variables (Production)

Обязательные:
```bash
DATA_DIR=/data
```

Опциональные:
```bash
TELEGRAM_BOT_TOKEN=<token>
REDIS_HOST=redis
REDIS_PORT=6379
```