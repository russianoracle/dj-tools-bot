# DJ Tools Bot - Local Development Commands
# Updated with new deployment system (sync to dj-tools-bot)

.PHONY: install install-dev start-local restart-local stop-local status-local logs-worker logs-all start stop logs clean test deploy deploy-full

# ============================================================================
# HELP
# ============================================================================

.DEFAULT_GOAL := help

help:
	@echo "📚 DJ Tools Bot - Available Commands"
	@echo ""
	@echo "⚙️  Setup:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install dev dependencies"
	@echo ""
	@echo "🚀 Local Development:"
	@echo "  make start-local      - Start Redis + Bot + Worker locally"
	@echo "  make restart-local    - Restart all local services"
	@echo "  make stop-local       - Stop all local services + cleanup"
	@echo "  make status-local     - Check local services status"
	@echo ""
	@echo "🚀 Development (Legacy):"
	@echo "  make start        - Start full stack (Redis + Bot)"
	@echo "  make start-bot    - Start bot only"
	@echo "  make dev          - Development mode (interactive logs)"
	@echo "  make stop         - Stop all services"
	@echo "  make logs         - View logs"
	@echo "  make status       - Check service status"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test         - Run tests"
	@echo ""
	@echo "📦 Deployment:"
	@echo "  make deploy               - Quick deploy (app/ only) → GitHub Actions"
	@echo "  make deploy-full          - Full deploy (all files) → GitHub Actions"
	@echo "  make deploy-safe          - Deploy with DB backup"
	@echo "  make deploy-sync-only     - Sync files without git operations"
	@echo "  make pre-deploy           - Run pre-deploy checks"
	@echo ""
	@echo "💾 Data Management:"
	@echo "  make backup-db        - Backup production database"
	@echo "  make cache-stats      - Show cache statistics"
	@echo "  make cache-clean      - Clean old cache (30+ days)"
	@echo "  make clean-downloads  - Clean downloaded audio files"
	@echo "  make clean-all-cache  - Clean all cache (DB + downloads)"
	@echo "  make data-dir         - Show data directory location"
	@echo "  make sync-rekordbox   - Sync Rekordbox metadata (local only)"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean            - Clean cache and logs"
	@echo ""
	@echo "📖 Documentation:"
	@echo "  docs/DATA_PERSISTENCE.md   - Data persistence guide"
	@echo "  DEPLOY_CHECKLIST.md        - Deployment checklist"
	@echo "  .env.example               - Environment configuration"

# Python executable
PYTHON := /Applications/miniforge3/bin/python3

# GitHub CLI executable
GH_BIN := /opt/homebrew/bin/gh

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

# Install production dependencies
install:
	@echo "📦 Installing production dependencies..."
	@$(PYTHON) -m pip install -r requirements-prod-app.txt
	@echo "✅ Dependencies installed"

# Install dev dependencies
install-dev:
	@echo "📦 Installing dev dependencies..."
	@$(PYTHON) -m pip install -r requirements.txt
	@$(PYTHON) -m pip install -r requirements-dev.txt
	@echo "✅ Dev dependencies installed"

# ============================================================================
# LOCAL DEVELOPMENT (Bot + Worker)
# ============================================================================

# Start Redis + Bot + Worker locally
start-local:
	@echo "🚀 Starting local development environment..."
	@echo ""
	@echo "📦 Step 1/3: Starting Redis..."
	@redis-server --daemonize yes --port 6379 --dir ./cache --dbfilename dump.rdb
	@sleep 1
	@redis-cli ping > /dev/null && echo "  ✅ Redis started" || echo "  ❌ Redis failed to start"
	@echo ""
	@echo "🤖 Step 2/3: Starting Telegram Bot..."
	@REDIS_HOST=localhost REDIS_PORT=6379 DATA_DIR=./cache nohup $(PYTHON) -m app.main > bot.log 2>&1 &
	@echo "  ✅ Bot started (PID: $$!)"
	@sleep 2
	@echo ""
	@echo "⚙️  Step 3/3: Starting ARQ Worker..."
	@REDIS_HOST=localhost REDIS_PORT=6379 DATA_DIR=./cache nohup $(PYTHON) -m arq app.services.arq_worker.WorkerSettings > worker.log 2>&1 &
	@echo "  ✅ Worker started (PID: $$!)"
	@sleep 2
	@echo ""
	@echo "✅ All services started!"
	@echo ""
	@echo "📊 Status:"
	@$(MAKE) status-local
	@echo ""
	@echo "📄 Logs:"
	@echo "  Bot:    tail -f bot.log"
	@echo "  Worker: tail -f worker.log"

# Check local services status
status-local:
	@echo "📊 Local Services Status:"
	@echo ""
	@echo "Redis:"
	@redis-cli ping 2>/dev/null && echo "  ✅ Running" || echo "  ❌ Not running"
	@echo ""
	@echo "Bot:"
	@pgrep -f "python.*app.main" >/dev/null && echo "  ✅ Running (PID: $$(pgrep -f 'python.*app.main'))" || echo "  ❌ Not running"
	@echo ""
	@echo "Worker:"
	@pgrep -f "arq.*WorkerSettings" >/dev/null && echo "  ✅ Running (PID: $$(pgrep -f 'arq.*WorkerSettings'))" || echo "  ❌ Not running"

# Restart all local services
restart-local:
	@echo "🔄 Restarting local services..."
	@$(MAKE) stop-local
	@sleep 2
	@$(MAKE) start-local

# Stop all local services + cleanup
stop-local:
	@echo "🛑 Stopping local services..."
	@echo ""
	@echo "Stopping Bot..."
	@pkill -f "python.*app.main" || echo "  (not running)"
	@echo "Stopping Worker..."
	@pkill -f "arq.*WorkerSettings" || echo "  (not running)"
	@echo "Stopping Redis..."
	@redis-cli shutdown 2>/dev/null || echo "  (not running)"
	@echo ""
	@echo "🧹 Cleaning up..."
	@rm -f bot.log worker.log nohup.out
	@rm -f cache/dump.rdb
	@echo ""
	@echo "✅ All services stopped and cleaned"

# View worker logs
logs-worker:
	@tail -f worker.log

# View both logs
logs-all:
	@echo "📋 Bot logs (bot.log):"
	@tail -20 bot.log
	@echo ""
	@echo "📋 Worker logs (worker.log):"
	@tail -20 worker.log

# ============================================================================
# LEGACY COMMANDS (for backward compatibility)
# ============================================================================

# Start full stack (Redis + Bot)
start:
	@echo "🚀 Starting DJ Tools Bot..."
	@echo "📦 Starting Redis..."
	@redis-server --daemonize yes --port 6379 --dir ./cache --dbfilename dump.rdb
	@sleep 1
	@echo "🤖 Starting Telegram Bot..."
	@REDIS_URL=redis://localhost:6379/0 nohup $(PYTHON) -m app.main > bot.log 2>&1 &
	@echo "✅ Bot started! PID: $$!"
	@echo "📄 Logs: tail -f bot.log"
	@sleep 2
	@tail -20 bot.log

# Start bot only (assume Redis is running)
start-bot:
	@echo "🤖 Starting bot only..."
	@REDIS_URL=redis://localhost:6379/0 $(PYTHON) -m app.main 2>&1 | tee bot.log

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	@pkill -f "python3 -m app.main" || true
	@redis-cli shutdown || true
	@echo "✅ All stopped"

# View logs
logs:
	@tail -f bot.log

# Clean cache and logs
clean:
	@echo "🧹 Cleaning cache and logs..."
	@rm -f bot.log
	@rm -f cache/dump.rdb
	@echo "✅ Cleaned"

# Run tests
test:
	@$(PYTHON) -m pytest tests/ -v

# Health check
status:
	@echo "📊 Service Status:"
	@echo "Redis:"
	@redis-cli ping 2>/dev/null && echo "  ✅ Running" || echo "  ❌ Not running"
	@echo "Bot:"
	@pgrep -f "python3 -m app.main" >/dev/null && echo "  ✅ Running (PID: $$(pgrep -f 'python3 -m app.main'))" || echo "  ❌ Not running"

# Interactive mode (logs in terminal)
dev:
	@echo "🔧 Starting in development mode..."
	@REDIS_URL=redis://localhost:6379/0 LOG_LEVEL=DEBUG $(PYTHON) -m app.main

# ============================================================================
# DOCKER BASE IMAGE (rebuild only when dependencies change)
# ============================================================================

REGISTRY := cr.yandex
REGISTRY_ID := crp78hhr2t67jkeedad1

# Build and push app-base image (contains all dependencies, no app code)
# Run this only when requirements-prod-app.txt changes
build-base:
	@echo "🔧 Building app-base image..."
	docker buildx build \
		-f Dockerfile.base \
		--build-arg REGISTRY=$(REGISTRY) \
		--build-arg REGISTRY_ID=$(REGISTRY_ID) \
		--push \
		-t $(REGISTRY)/$(REGISTRY_ID)/app-base:3.12 \
		.
	@echo "✅ app-base:3.12 pushed to registry"

# Check if app-base exists in registry
check-base:
	@echo "🔍 Checking app-base image..."
	@docker pull $(REGISTRY)/$(REGISTRY_ID)/app-base:3.12 && echo "✅ app-base exists" || echo "❌ app-base not found, run: make build-base"

.PHONY: build-base check-base

# ============================================================================
# DEPLOYMENT COMMANDS (sync to dj-tools-bot)
# ============================================================================

DEPLOY_REPO := ../dj-tools-bot

# Pre-deploy checks
pre-deploy:
	@echo "🔍 Pre-deploy checks..."
	@echo "1. Checking for pyrekordbox in requirements.txt..."
	@grep -q "pyrekordbox" requirements.txt && echo "  ❌ ERROR: pyrekordbox found in requirements.txt! Use requirements-training.txt" && exit 1 || echo "  ✅ OK"
	@echo "2. Checking DATA_DIR setup..."
	@grep -q "DATA_DIR" .env.example && echo "  ✅ .env.example has DATA_DIR" || echo "  ⚠️  Warning: .env.example missing DATA_DIR"
	@echo "3. Checking docker-compose.yml for volumes..."
	@grep -q "volumes:" docker-compose.yml && echo "  ✅ docker-compose has volumes" || echo "  ⚠️  Warning: No volumes in docker-compose"
	@echo "4. Checking deploy repo exists..."
	@test -d $(DEPLOY_REPO) && echo "  ✅ $(DEPLOY_REPO) exists" || (echo "  ❌ ERROR: $(DEPLOY_REPO) not found!" && exit 1)
	@echo "✅ Pre-deploy checks passed!"

# Sync files from mood-classifier to dj-tools-bot
sync-to-deploy:
	@echo "📦 Syncing files to $(DEPLOY_REPO)..."
	@echo "  → Copying app/"
	@rsync -av --delete --exclude='__pycache__' --exclude='*.pyc' app/ $(DEPLOY_REPO)/app/
	@echo "  → Copying tests/"
	@rsync -av --delete --exclude='__pycache__' --exclude='*.pyc' tests/ $(DEPLOY_REPO)/tests/
	@echo "  → Copying .github/workflows/"
	@rsync -av --delete .github/workflows/ $(DEPLOY_REPO)/.github/workflows/
	@echo "  → Copying models/production/"
	@rsync -av --delete models/production/ $(DEPLOY_REPO)/models/production/
	@echo "  → Copying Docker files"
	@cp Dockerfile.unified Dockerfile.base docker-compose.yml $(DEPLOY_REPO)/ || true
	@echo "  → Copying config files"
	@cp fluent-bit.conf parsers.conf flatten-log.lua set-source.lua $(DEPLOY_REPO)/ || true
	@cp healthcheck_worker.py logging-config.yaml main.py pytest.ini $(DEPLOY_REPO)/ || true
	@cp requirements-prod.txt requirements-prod-app.txt $(DEPLOY_REPO)/ || true
	@cp .dockerignore .env.example .gitleaksignore README.md $(DEPLOY_REPO)/ || true
	@cp Makefile $(DEPLOY_REPO)/ || true
	@echo "  → Copying scripts/"
	@mkdir -p $(DEPLOY_REPO)/scripts
	@cp scripts/fetch-secrets.sh $(DEPLOY_REPO)/scripts/ || true
	@echo "✅ Files synced to $(DEPLOY_REPO)"

# Cancel running GitHub Actions workflows before deploy
cancel-running-jobs:
	@echo "🛑 Canceling running GitHub Actions workflows..."
	@cd $(DEPLOY_REPO) && $(GH_BIN) run list --status in_progress --json databaseId -q '.[].databaseId' | \
		xargs -I {} $(GH_BIN) api --method POST /repos/russianoracle/dj-tools-bot/actions/runs/{}/cancel 2>/dev/null || true
	@echo "✅ All running jobs canceled"

# Deploy only app/ folder to dj-tools-bot repo (production code only)
# Usage: make deploy MSG="your commit message"
deploy: pre-deploy cancel-running-jobs
	@echo "🚀 Quick deploy: app/ only"
	@echo ""
	@echo "📝 Step 1: Commit changes in mood-classifier (dev repo)"
	@git add app/
	@git commit -m "$(if $(MSG),$(MSG),deploy: update app/)" || echo "  ℹ️  No changes to commit in dev repo"
	@echo ""
	@echo "📦 Step 2: Sync app/ to dj-tools-bot"
	@rsync -av --delete --exclude='__pycache__' --exclude='*.pyc' app/ $(DEPLOY_REPO)/app/
	@echo ""
	@echo "📝 Step 3: Commit in dj-tools-bot (deploy repo)"
	@cd $(DEPLOY_REPO) && git add app/ && \
		git commit -m "$(if $(MSG),$(MSG),deploy: update app/ from mood-classifier)" || echo "  ℹ️  No changes to commit in deploy repo"
	@echo ""
	@echo "🚀 Step 4: Push to GitHub → triggers CI/CD"
	@cd $(DEPLOY_REPO) && git push origin main
	@echo ""
	@echo "✅ Deployed! Check GitHub Actions: https://github.com/russianoracle/dj-tools-bot/actions"

# Deploy full infrastructure (app/ + tests/ + docker + configs)
# Usage: make deploy-full MSG="your commit message"
deploy-full: pre-deploy cancel-running-jobs
	@echo "🚀 Full deploy: all production files"
	@echo ""
	@echo "📝 Step 1: Commit ALL changes in mood-classifier (dev repo)"
	@git add -A
	@git commit -m "$(if $(MSG),$(MSG),deploy: full deployment $(shell date +%Y-%m-%d))" || echo "  ℹ️  No changes to commit in dev repo"
	@echo ""
	@echo "📦 Step 2: Sync ALL files to dj-tools-bot"
	@$(MAKE) sync-to-deploy
	@echo ""
	@echo "📝 Step 3: Commit in dj-tools-bot (deploy repo)"
	@cd $(DEPLOY_REPO) && git add -A && \
		git commit -m "$(if $(MSG),$(MSG),deploy: full sync from mood-classifier $(shell date +%Y-%m-%d))" || echo "  ℹ️  No changes to commit in deploy repo"
	@echo ""
	@echo "🚀 Step 4: Push to GitHub → triggers CI/CD"
	@cd $(DEPLOY_REPO) && git push origin main
	@echo ""
	@echo "✅ Deployed! Check GitHub Actions: https://github.com/russianoracle/dj-tools-bot/actions"
	@echo ""
	@echo "🔔 IMPORTANT: Production setup checklist:"
	@echo "   1. GitHub secrets configured: YC_LOCKBOX_SECRET_ID, YC_FOLDER_ID"
	@echo "   2. Secrets loaded from Yandex Lockbox automatically"
	@echo "   3. See: docs/DATA_PERSISTENCE.md"

# Quick sync without git operations (for testing)
deploy-sync-only:
	@echo "📦 Syncing files only (no git operations)..."
	@$(MAKE) sync-to-deploy
	@echo "✅ Files synced. Run 'cd $(DEPLOY_REPO) && git status' to see changes"

# Backup production database before deploy
backup-db:
	@echo "💾 Backing up production database..."
	@mkdir -p backups
	@docker exec $$(docker ps -q -f name=mood-classifier) sqlite3 /data/predictions.db .dump > backups/db-backup-$(shell date +%Y%m%d-%H%M%S).sql
	@echo "✅ Backup saved to backups/"

# Deploy with database backup
deploy-safe: backup-db deploy-full
	@echo "✅ Safe deploy completed (with DB backup)"

# ============================================================================
# PRODUCTION MONITORING
# ============================================================================

# Show production container logs
logs-prod:
	@echo "📋 Fetching logs from production VM..."
	@ssh -i ~/.ssh/tender-bot-key ubuntu@158.160.122.216 "cd ~/app && docker-compose logs --tail=100 app"

# Follow production logs in real-time
logs-prod-follow:
	@echo "📋 Following production logs (Ctrl+C to stop)..."
	@ssh -i ~/.ssh/tender-bot-key ubuntu@158.160.122.216 "cd ~/app && docker-compose logs -f app"

# Show all services logs (app + fluent-bit)
logs-prod-all:
	@echo "📋 Fetching all service logs..."
	@ssh -i ~/.ssh/tender-bot-key ubuntu@158.160.122.216 "cd ~/app && docker-compose logs --tail=50"

# Show container status on production
status-prod:
	@echo "📊 Production container status:"
	@ssh -i ~/.ssh/tender-bot-key ubuntu@158.160.122.216 "cd ~/app && docker-compose ps"

# Restart production containers
restart-prod:
	@echo "🔄 Restarting production containers..."
	@ssh -i ~/.ssh/tender-bot-key ubuntu@158.160.122.216 "cd ~/app && docker-compose restart"
	@echo "✅ Restarted"

# ============================================================================
# DATA MANAGEMENT
# ============================================================================

# Show cache statistics
cache-stats:
	@echo "📊 Cache Statistics:"
	@$(PYTHON) -c "from app.core.connectors import CacheRepository; \
		repo = CacheRepository.get_instance(); \
		stats = repo._manager.get_track_metadata_stats(); \
		print(f'  Tracks: {stats[\"total_tracks\"]}'); \
		print(f'  Artists: {stats[\"unique_artists\"]}'); \
		print(f'  Genres: {stats[\"unique_genres\"]}'); \
		print(f'  Sources: {stats[\"by_source\"]}');"

# Clean old cache entries
cache-clean:
	@echo "🧹 Cleaning old cache entries (30+ days)..."
	@$(PYTHON) -c "from app.core.connectors import CacheRepository; \
		repo = CacheRepository.get_instance(); \
		repo._manager.cleanup_old_entries(max_age_days=30); \
		print('✅ Cleaned')"

# Clean downloaded audio files
clean-downloads:
	@echo "🧹 Cleaning downloaded audio files..."
	@rm -rf cache/downloads/* downloads/* 2>/dev/null || true
	@echo "✅ Downloads cleaned"

# Clean all cache (DB + downloads)
clean-all-cache:
	@echo "🧹 Cleaning all cache..."
	@$(MAKE) cache-clean
	@$(MAKE) clean-downloads
	@echo "✅ All cache cleaned"

# Show where data is stored
data-dir:
	@echo "📁 Data Directory:"
	@$(PYTHON) -c "from app.core.config import get_data_dir, get_db_path; \
		print(f'  DATA_DIR: {get_data_dir()}'); \
		print(f'  DB: {get_db_path()}');"

# Sync Rekordbox metadata to cache (local only)
sync-rekordbox:
	@echo "🔄 Syncing Rekordbox metadata to cache..."
	@$(PYTHON) scripts/sync_rekordbox_metadata.py
	@echo "✅ Sync complete!"

# Fetch secrets from Lockbox (run on VM)
fetch-secrets:
	@echo "🔐 Fetching secrets from Yandex Lockbox..."
	@bash scripts/fetch-secrets.sh --validate
	@echo "✅ Secrets loaded to .env"

# Validate secrets are present
validate-secrets:
	@echo "🔍 Validating secrets..."
	@$(PYTHON) -c "from app.core.secrets import validate_secrets; \
		ok, missing = validate_secrets(); \
		exit(0 if ok else 1)"

.PHONY: pre-deploy sync-to-deploy cancel-running-jobs deploy deploy-full deploy-sync-only backup-db deploy-safe cache-stats cache-clean data-dir sync-rekordbox fetch-secrets validate-secrets
