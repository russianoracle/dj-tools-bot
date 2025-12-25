#!/bin/bash
# Setup script for mood-classifier dual-repository structure
#
# Structure:
#   mood-classifier/          <- public (code, production models)
#   mood-classifier-private/  <- private (data, cache, secrets)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_REPO="$(dirname "$SCRIPT_DIR")"
PRIVATE_REPO="$(dirname "$PUBLIC_REPO")/mood-classifier-private"

echo "=== Mood Classifier Repository Setup ==="
echo "Public repo:  $PUBLIC_REPO"
echo "Private repo: $PRIVATE_REPO"
echo ""

# Check if private repo exists
if [ -d "$PRIVATE_REPO" ]; then
    echo "[OK] Private repo already exists"
else
    echo "[CREATE] Creating private repo structure..."
    mkdir -p "$PRIVATE_REPO"/{data,cache,secrets,models/experimental}

    # Initialize git
    cd "$PRIVATE_REPO"
    git init

    # Create .gitignore for private repo
    cat > .gitignore << 'EOF'
# Cache files (too large, regeneratable)
cache/stft/
cache/*.db-journal

# Temporary files
*.tmp
*.temp
temp/
tmp/

# macOS
.DS_Store

# Large audio files in data/ (keep JSON, exclude audio)
data/**/*.mp3
data/**/*.wav
data/**/*.flac
data/**/*.m4a
data/**/*.opus
EOF

    # Create README
    cat > README.md << 'EOF'
# mood-classifier-private

Private repository for mood-classifier project.

## Contents

- `data/` - Training data, DJ sets, ground truth labels
- `cache/` - SQLite database, feature cache
- `secrets/` - Environment variables, API keys, credentials
- `models/experimental/` - Models in development

## Setup

This repo is used alongside the public `mood-classifier` repo.

```bash
# From mood-classifier/
./scripts/setup-private-repo.sh
```

## Secrets

Store sensitive data in `secrets/`:
- `.env` - Environment variables
- `ARCHITECTURE_REPORT.md` - Docs with embedded secrets
- `*-iam-key.json` - Service account keys
EOF

    git add .
    git commit -m "Initial structure for private data"
    echo "[OK] Private repo initialized"
fi

# Move existing data to private repo
echo ""
echo "=== Moving data to private repo ==="

move_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$PUBLIC_REPO/$src" ] && [ ! -L "$PUBLIC_REPO/$src" ]; then
        echo "[MOVE] $src -> private repo"
        mv "$PUBLIC_REPO/$src" "$dst" 2>/dev/null || cp -r "$PUBLIC_REPO/$src" "$dst"
        rm -rf "$PUBLIC_REPO/$src" 2>/dev/null || true
    fi
}

# Move data directories
move_if_exists "data" "$PRIVATE_REPO/"
move_if_exists "cache" "$PRIVATE_REPO/"

# Move secrets
mkdir -p "$PRIVATE_REPO/secrets"
[ -f "$PUBLIC_REPO/.env" ] && mv "$PUBLIC_REPO/.env" "$PRIVATE_REPO/secrets/"
[ -f "$PUBLIC_REPO/docs/ARCHITECTURE_REPORT.md" ] && mv "$PUBLIC_REPO/docs/ARCHITECTURE_REPORT.md" "$PRIVATE_REPO/secrets/"

# Move experimental models
move_if_exists "models/ensemble" "$PRIVATE_REPO/models/experimental/"
move_if_exists "models/ensemble_current" "$PRIVATE_REPO/models/experimental/"
move_if_exists "models/genre" "$PRIVATE_REPO/models/experimental/"

# Create symlinks
echo ""
echo "=== Creating symlinks ==="

create_symlink() {
    local target="$1"
    local link="$2"
    if [ -L "$link" ]; then
        echo "[OK] $link (already linked)"
    elif [ -e "$link" ]; then
        echo "[SKIP] $link (exists, not a symlink)"
    else
        ln -s "$target" "$link"
        echo "[LINK] $link -> $target"
    fi
}

create_symlink "$PRIVATE_REPO/data" "$PUBLIC_REPO/data"
create_symlink "$PRIVATE_REPO/cache" "$PUBLIC_REPO/cache"
create_symlink "$PRIVATE_REPO/secrets/.env" "$PUBLIC_REPO/.env"

# Update public .gitignore to ignore symlinks
echo ""
echo "=== Verifying .gitignore ==="

if ! grep -q "# Symlinks to private repo" "$PUBLIC_REPO/.gitignore"; then
    cat >> "$PUBLIC_REPO/.gitignore" << 'EOF'

# Symlinks to private repo (actual data in mood-classifier-private)
data
cache
EOF
    echo "[UPDATE] Added symlink entries to .gitignore"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Structure:"
echo "  $PUBLIC_REPO/          <- git push origin (public)"
echo "  $PRIVATE_REPO/         <- git push private (data)"
echo ""
echo "Symlinks:"
echo "  data -> ../mood-classifier-private/data"
echo "  cache -> ../mood-classifier-private/cache"
echo "  .env -> ../mood-classifier-private/secrets/.env"
echo ""
echo "Next steps:"
echo "  1. cd $PRIVATE_REPO"
echo "  2. git remote add origin git@github.com:YOUR_USER/mood-classifier-private.git"
echo "  3. git push -u origin main"
