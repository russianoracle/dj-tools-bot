#!/bin/bash
# ============================================================================
# Install GitHub Actions Runner on existing Bot VM
# ============================================================================
set -e

RUNNER_TOKEN="AIWS4CBCYMKSKKQJYZLHEGLJH5NKU"
GITHUB_REPO="russianoracle/dj-tools-bot"
RUNNER_NAME="yandex-bot-vm-runner"
BOT_VM_NAME="dj-tools-bot"

echo "=== Getting Bot VM IP ==="
BOT_IP=$(yc compute instance get ${BOT_VM_NAME} --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address')

if [ -z "$BOT_IP" ] || [ "$BOT_IP" == "null" ]; then
  echo "ERROR: Could not get bot VM IP"
  echo "Trying alternative VM names..."
  BOT_IP=$(yc compute instance list --format json | jq -r '.[0].network_interfaces[0].primary_v4_address.one_to_one_nat.address')
fi

echo "Bot VM IP: $BOT_IP"

echo "=== Installing GitHub Runner on Bot VM ==="

ssh -o StrictHostKeyChecking=no ubuntu@${BOT_IP} << 'ENDSSH'
  set -e

  echo "=== Checking if Docker is installed ==="
  if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker ubuntu
  fi

  echo "=== Installing GitHub Actions Runner ==="
  mkdir -p ~/actions-runner
  cd ~/actions-runner

  # Download if not exists
  if [ ! -f "config.sh" ]; then
    RUNNER_VERSION="2.311.0"
    curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L \
      https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
    tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
    rm actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
  fi

  # Remove old runner if exists
  if [ -f ".runner" ]; then
    echo "Removing old runner configuration..."
    sudo ./svc.sh stop || true
    sudo ./svc.sh uninstall || true
    ./config.sh remove --token RUNNER_TOKEN || true
  fi

  echo "=== Configuring Runner ==="
  ./config.sh \
    --url "https://github.com/GITHUB_REPO" \
    --token "RUNNER_TOKEN" \
    --name "RUNNER_NAME" \
    --labels "self-hosted,yandex-cloud,docker,bot-vm" \
    --work _work \
    --unattended \
    --replace

  echo "=== Installing as systemd service ==="
  sudo ./svc.sh install
  sudo ./svc.sh start

  echo "=== Checking status ==="
  sudo systemctl status actions.runner.*.service --no-pager || true

  echo "=== GitHub Runner installed successfully! ==="
ENDSSH

# Substitute variables in the SSH command
ssh -o StrictHostKeyChecking=no ubuntu@${BOT_IP} bash << ENDSSH2
  set -e
  cd ~/actions-runner

  # Remove old if exists
  if [ -f ".runner" ]; then
    echo "Removing old runner..."
    sudo ./svc.sh stop || true
    sudo ./svc.sh uninstall || true
    ./config.sh remove --token ${RUNNER_TOKEN} || true
  fi

  # Configure
  ./config.sh \
    --url "https://github.com/${GITHUB_REPO}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "self-hosted,yandex-cloud,docker,bot-vm" \
    --work _work \
    --unattended \
    --replace

  # Install service
  sudo ./svc.sh install
  sudo ./svc.sh start

  echo "Runner installed!"
  sudo systemctl status actions.runner.*.service --no-pager
ENDSSH2

echo ""
echo "=== Done! ==="
echo "Check runner status at:"
echo "  https://github.com/${GITHUB_REPO}/settings/actions/runners"
