#!/bin/bash
# Deploy production files to dj-tools-bot repository
# Usage: ./scripts/deploy_to_bot.sh [commit message]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_BRANCH="main"
COMMIT_MSG="${1:-Auto-deploy from mood-classifier}"

cd "$PROJECT_ROOT"

echo "=== Deploying to dj-tools-bot ==="

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "Warning: You have uncommitted changes in mood-classifier"
    echo "Consider committing them first"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Files/folders to deploy (production only)
DEPLOY_FILES=(
    # Core application
    "src/services/__init__.py"
    "src/services/bot.py"
    "src/services/celery_app.py"
    "src/services/analysis.py"
    "src/services/downloader.py"
    "src/services/api.py"
    "src/services/profiling.py"

    # Core modules needed by services
    "src/__init__.py"
    "src/audio/"
    "src/classification/"
    "src/metadata/"
    "src/utils/"
    "src/core/"

    # Docker/deployment
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    ".env.example"

    # GitHub Actions
    ".github/workflows/deploy.yml"

    # ML models (production only)
    "models/production/"

    # Config
    "config/default_config.yaml"

    # Entry point
    "main.py"
)

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Using temp dir: $TEMP_DIR"

# Clone deploy repo
echo "Cloning dj-tools-bot..."
git clone --depth 1 git@github.com:russianoracle/dj-tools-bot.git "$TEMP_DIR/dj-tools-bot"

# Copy files
echo "Copying production files..."
for item in "${DEPLOY_FILES[@]}"; do
    src="$PROJECT_ROOT/$item"
    dst="$TEMP_DIR/dj-tools-bot/$item"

    if [[ -e "$src" ]]; then
        # Create parent directory
        mkdir -p "$(dirname "$dst")"

        if [[ -d "$src" ]]; then
            cp -r "$src" "$dst"
            echo "  [DIR]  $item"
        else
            cp "$src" "$dst"
            echo "  [FILE] $item"
        fi
    else
        echo "  [SKIP] $item (not found)"
    fi
done

# Create/update README for bot repo
cat > "$TEMP_DIR/dj-tools-bot/README.md" << 'EOF'
# DJ Tools Bot

Telegram bot for DJ set analysis - automatic energy zone classification.

## Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
TELEGRAM_BOT_TOKEN=your_bot_token
ADMIN_USER_ID=your_telegram_id
REDIS_URL=redis://redis:6379/0
```

## Architecture

- **Bot**: aiogram 3.x Telegram bot with inline keyboard UI
- **Celery**: Background task processing for audio analysis
- **Redis**: Message broker and result backend
- **Analysis**: ML-based energy zone classification (Yellow/Green/Purple)

---
Auto-deployed from [mood-classifier](https://github.com/russianoracle/mood-classifier)
EOF

# Create proper .gitignore for bot repo
cat > "$TEMP_DIR/dj-tools-bot/.gitignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/
.eggs/

# Environment
.env
.env.*
!.env.example
venv/
.venv/

# IDE
.idea/
.vscode/
*.swp

# Cache
cache/
*.log
.pytest_cache/
.coverage

# Data
data/
downloads/
*.backup

# macOS
.DS_Store
EOF

# Commit and push
cd "$TEMP_DIR/dj-tools-bot"

git add -A
if git diff --staged --quiet; then
    echo "No changes to deploy"
else
    git commit -m "$COMMIT_MSG"
    git push origin $DEPLOY_BRANCH
    echo ""
    echo "=== Deployed successfully! ==="
    echo "Commit: $(git rev-parse --short HEAD)"
fi

# Cleanup
rm -rf "$TEMP_DIR"
echo "Done!"