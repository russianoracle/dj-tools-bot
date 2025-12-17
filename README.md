# Mood Classifier - DJ Set Analysis Bot

AI-powered Telegram bot for analyzing DJ sets, detecting drops, transitions, and energy patterns using audio signal processing and machine learning.

## Recent Improvements (2025-12-17)

### Critical Fixes
- **Architecture**: Fixed KeyAnalysisTask to use STFTCache (no more runtime crashes)
- **Security**: Replaced unsafe `os.execv` with `sys.exit` for bot restart
- **Security**: Added domain allowlist for URL validation (SSRF protection)
- **Memory**: Added STFTCache cleanup to prevent memory leaks on long tracks

### Performance & Optimization
- **15-20% faster** analysis via vectorized mel filterbank computation
- **10x smaller** production Docker image (8GB → 800MB)
- **5x faster** CI/CD builds (25min → 5min)

### Dependencies
- Split requirements into production (`requirements.txt`) and development (`requirements-dev.txt`)
- Updated cryptography to 46.0.3 (security patches)
- Fixed yandex-cloud package name

### DevOps
- Made security scan non-blocking in CI/CD pipeline
- Added comprehensive audit reports (ARQ patterns, dependency analysis)
- Improved deployment workflow with proper error handling

---

## Features

- **Audio Analysis**: Spectral analysis using STFTCache with librosa
- **Drop Detection**: Automatic detection of energy drops with vectorized NMS
- **Transition Detection**: Identify smooth vs hard transitions between tracks
- **Energy Timeline**: Track energy levels throughout the set
- **Telegram Bot**: Async bot with ARQ task queue for background processing
- **Caching**: Redis-based caching for fast repeated analysis

---

## Architecture

```
app/
├── common/primitives/     # Pure math (numpy/scipy only)
│   ├── stft.py           # STFTCache - single librosa entry point
│   ├── dynamics.py       # Drop detection, buildup analysis
│   ├── rhythm.py         # Tempo, beat tracking
│   └── spectral.py       # Spectral features
├── modules/analysis/     # Analysis orchestration
│   ├── tasks/           # Feature extraction tasks
│   └── pipelines/       # Multi-stage analysis pipelines
├── modules/bot/         # Telegram bot handlers
│   └── handlers/        # Message and callback handlers
├── core/                # Core infrastructure
│   ├── cache/          # CacheRepository
│   ├── secrets/        # Yandex Lockbox integration
│   └── adapters/       # External integrations
└── services/           # Background services
    └── arq_worker.py   # Async task queue worker
```

### Key Architectural Principles
1. **STFTCache Centralization**: All librosa calls go through `stft.py`
2. **Layer Separation**: Primitives → Tasks → Pipelines
3. **Cache-First**: Use `CacheRepository`, never `CacheManager` directly
4. **M2 Optimization**: All arrays are `float32` + contiguous

---

## Development Setup

### Prerequisites
- Python 3.12+
- Redis 7+
- ffmpeg (for audio processing)

### Installation

```bash
# Install production dependencies
pip install -r requirements.txt

# For development (includes ML training, GUI, visualization)
pip install -r requirements-dev.txt
```

### Environment Variables

Create `.env` file:
```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_token_here
ADMIN_USER_ID=your_telegram_id

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Storage
DATA_DIR=/data
DOWNLOADS_DIR=/data/downloads

# Logging
LOG_LEVEL=INFO
LOG_JSON_FORMAT=true
```

---

## Running the Application

### Development Mode

```bash
# Start Redis
docker-compose up redis -d

# Start bot (with hot reload)
make dev

# Start ARQ worker
arq app.services.arq_worker.WorkerSettings
```

### Production Mode

```bash
# Full stack (Redis + Bot + Worker)
make start

# Check status
make status

# View logs
make logs

# Stop services
make stop
```

---

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_stft_cache.py -v

# Run CI tests only
pytest tests/test_ci_*.py -v
```

---

## Deployment

**IMPORTANT**: Always use `make deploy-full` for deployments. Never push directly to origin.

```bash
# Deploy full stack (recommended)
make deploy-full

# This will:
# 1. Run pre-deploy checks
# 2. Copy all files to dj-tools-bot repo
# 3. Commit and push changes
# 4. Trigger GitHub Actions CI/CD pipeline
```

### CI/CD Pipeline

GitHub Actions automatically:
1. Runs security scan (gitleaks)
2. Builds Docker image with test stage
3. Runs integration tests
4. Pushes production image to Yandex Container Registry
5. Deploys to production VM (if secrets configured)

**Pipeline Status**: [![CI/CD](https://github.com/russianoracle/dj-tools-bot/actions/workflows/deploy.yml/badge.svg)](https://github.com/russianoracle/dj-tools-bot/actions)

---

## Performance

### Build Performance
- **Docker build**: 1m23s (down from 25min)
- **Image size**: 800MB (down from 8GB)
- **Test execution**: ~30s for CI tests

### Analysis Performance
- **Short tracks** (3-5min): ~5-10s
- **DJ sets** (1-2hr): ~2-5min
- **Memory usage**: <6GB per worker (with cleanup)

### Optimization Techniques
- Vectorized numpy operations (no Python loops)
- STFTCache for feature reuse
- Redis caching for repeated analysis
- Apple Silicon (M2) optimizations

---

## Documentation

- [ARCHITECTURE_PLAN.md](docs/ARCHITECTURE_PLAN.md) - System architecture overview
- [REFACTORING_PLAN.md](docs/REFACTORING_PLAN.md) - Refactoring roadmap
- [ARQ_PATTERNS_REVIEW.md](ARQ_PATTERNS_REVIEW.md) - Workflow orchestration analysis
- [DEPENDENCY_AUDIT.md](DEPENDENCY_AUDIT.md) - Dependency optimization report
- [CLAUDE.md](CLAUDE.md) - AI assistant development guidelines

---

## Makefile Commands

### Core Commands
```bash
make start          # Start full stack (Redis + Bot + Worker)
make dev            # Development mode with logs
make stop           # Stop all services
make test           # Run test suite
make status         # Check service status
```

### Deployment Commands
```bash
make deploy-full    # Deploy full stack (USE THIS)
make deploy         # Quick deploy (app only)
make deploy-safe    # Deploy with database backup
make pre-deploy     # Run pre-deploy checks
```

### Utility Commands
```bash
make logs           # View logs
make clean          # Clean temp files
make backup-db      # Backup database
```

---

## Troubleshooting

### Memory Issues
If worker uses >6GB memory:
- Check STFTCache cleanup is enabled
- Monitor with `docker stats mood-arq-worker`
- Reduce `max_jobs` in arq_worker.py

### Bot Not Responding
1. Check bot token: `echo $TELEGRAM_BOT_TOKEN`
2. Verify bot is running: `docker ps | grep mood-classifier`
3. Check logs: `docker logs mood-classifier --tail 50`

### CI/CD Pipeline Failures
- **Security scan fails**: Non-blocking, pipeline continues
- **Build fails**: Check requirements.txt syntax
- **Deploy fails**: Verify Lockbox secrets configured

---

## Contributing

### Code Style
- Follow CLAUDE.md guidelines
- Use STFTCache for all librosa operations
- Write vectorized numpy code (no loops)
- Add tests for new features

### Commit Messages
```bash
# Format: <type>: <description>
fix: resolve KeyAnalysisTask crash
feat: add drop detection vectorization
perf: optimize mel filterbank computation
```

### Pre-commit Checks
```bash
# Lint code
ruff check app/ tests/

# Type check
mypy app/ --ignore-missing-imports

# Run tests
pytest tests/ -v
```

---

## License

Proprietary - All rights reserved

---

## Acknowledgments

- **librosa** - Audio analysis library
- **Telegram Bot API** - Bot framework
- **ARQ** - Async task queue
- **Redis** - Caching and task queue backend
- **Yandex Cloud** - Infrastructure and logging

---

## Support

For issues and questions:
- GitHub Issues: [dj-tools-bot/issues](https://github.com/russianoracle/dj-tools-bot/issues)
- Documentation: [docs/](docs/)

---

**Last Updated**: 2025-12-17
**Version**: 2.0.0 (Post-Audit Improvements)
