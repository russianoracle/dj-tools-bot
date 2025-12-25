#!/bin/bash
# Setup Yandex Unified Agent for monitoring
# Collects system metrics + application metrics → Yandex Cloud Monitoring

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# Configuration
# ============================================================================

FOLDER_ID="${FOLDER_ID:-}"
INSTALL_DIR="/opt/yandex-unified-agent"
CONFIG_FILE="$INSTALL_DIR/config.yml"
SERVICE_FILE="/etc/systemd/system/unified-agent.service"

# ============================================================================
# Pre-checks
# ============================================================================

log "Starting Unified Agent installation..."
echo ""

# Check if running on Yandex Cloud VM
if ! curl -s -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/instance/id >/dev/null 2>&1; then
    error "Not running on Yandex Cloud VM"
    error "This script must run on a Yandex Cloud Compute VM"
    exit 1
fi
success "Running on Yandex Cloud VM"

# Check if FOLDER_ID is set
if [ -z "$FOLDER_ID" ]; then
    error "FOLDER_ID environment variable not set"
    echo ""
    echo "Get your folder ID:"
    echo "  yc config get folder-id"
    echo ""
    echo "Then run:"
    echo "  FOLDER_ID=<your_folder_id> bash setup-monitoring.sh"
    exit 1
fi
success "Folder ID: $FOLDER_ID"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run as root (use sudo)"
    exit 1
fi
success "Running as root"

echo ""

# ============================================================================
# Install Unified Agent
# ============================================================================

log "Step 1/5: Installing Unified Agent..."

# Download latest version
log "Downloading latest Unified Agent..."
ua_version=$(curl --silent https://storage.yandexcloud.net/yc-unified-agent/latest-version)
log "Latest version: $ua_version"

# Create installation directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download binary
curl --silent --remote-name \
    "https://storage.yandexcloud.net/yc-unified-agent/releases/$ua_version/unified_agent"

# Make executable
chmod +x unified_agent

success "Unified Agent installed: $INSTALL_DIR/unified_agent"
echo ""

# ============================================================================
# Configure Unified Agent
# ============================================================================

log "Step 2/5: Configuring Unified Agent..."

# Copy configuration from project
if [ -f ~/app/monitoring/unified-agent-config.yml ]; then
    cp ~/app/monitoring/unified-agent-config.yml "$CONFIG_FILE"
    success "Configuration copied from project"
else
    error "Configuration file not found: ~/app/monitoring/unified-agent-config.yml"
    exit 1
fi

# Substitute FOLDER_ID in config
sed -i "s/\${FOLDER_ID}/$FOLDER_ID/g" "$CONFIG_FILE"
success "Configuration updated with FOLDER_ID"

# Validate configuration
if "$INSTALL_DIR/unified_agent" --config "$CONFIG_FILE" check-config; then
    success "Configuration is valid"
else
    error "Configuration validation failed"
    exit 1
fi

echo ""

# ============================================================================
# Create systemd service
# ============================================================================

log "Step 3/5: Creating systemd service..."

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Yandex Unified Agent
After=network.target

[Service]
Type=simple
User=root
ExecStart=$INSTALL_DIR/unified_agent --config $CONFIG_FILE
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

success "Systemd service created: $SERVICE_FILE"
echo ""

# ============================================================================
# Start service
# ============================================================================

log "Step 4/5: Starting Unified Agent service..."

# Reload systemd
systemctl daemon-reload

# Enable service
systemctl enable unified-agent

# Start service
systemctl start unified-agent

# Check status
sleep 3
if systemctl is-active --quiet unified-agent; then
    success "Unified Agent is running"
else
    error "Unified Agent failed to start"
    systemctl status unified-agent
    exit 1
fi

echo ""

# ============================================================================
# Verify metrics collection
# ============================================================================

log "Step 5/5: Verifying metrics collection..."

# Check health endpoint
if curl -s http://localhost:16241/status | grep -q "OK"; then
    success "Health check endpoint responding"
else
    warning "Health check endpoint not responding (this is normal during startup)"
fi

# Check if metrics are being collected
sleep 5
if systemctl status unified-agent | grep -q "started"; then
    success "Metrics collection started"
else
    warning "Check logs: journalctl -u unified-agent -f"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

log "=========================================="
success "Unified Agent installation complete!"
log "=========================================="
echo ""

log "Configuration:"
echo "  Config file:  $CONFIG_FILE"
echo "  Folder ID:    $FOLDER_ID"
echo "  Metrics port: 8000 (bot), 8001 (worker)"
echo ""

log "Service management:"
echo "  Status:   systemctl status unified-agent"
echo "  Logs:     journalctl -u unified-agent -f"
echo "  Restart:  systemctl restart unified-agent"
echo "  Stop:     systemctl stop unified-agent"
echo ""

log "Health check:"
echo "  curl http://localhost:16241/status"
echo ""

log "View metrics in Yandex Cloud:"
echo "  https://console.cloud.yandex.ru/folders/$FOLDER_ID/monitoring"
echo ""

success "Monitoring is now active!"
