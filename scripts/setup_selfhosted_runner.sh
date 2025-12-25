#!/bin/bash
# Setup self-hosted GitHub Actions runner в Yandex Cloud

set -e

echo "=== Setting up GitHub Actions self-hosted runner ==="

# 1. Создать VM в Yandex Cloud
echo "Creating VM in Yandex Cloud..."
yc compute instance create \
  --name github-runner \
  --zone ru-central1-a \
  --cores 4 \
  --memory 8 \
  --create-boot-disk image-folder-id=standard-images,image-family=ubuntu-2204-lts,size=50 \
  --network-interface subnet-name=default-ru-central1-a,nat-ip-version=ipv4 \
  --ssh-key ~/.ssh/id_rsa.pub

# 2. Получить IP адрес
VM_IP=$(yc compute instance get github-runner --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address')
echo "VM created with IP: $VM_IP"

# 3. Установить Docker и GitHub Actions runner
echo "Installing Docker and GitHub Actions runner..."
ssh -o StrictHostKeyChecking=no ubuntu@$VM_IP << 'ENDSSH'
  # Install Docker
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker ubuntu

  # Install GitHub Actions runner
  mkdir -p ~/actions-runner && cd ~/actions-runner
  RUNNER_VERSION="2.311.0"
  curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L \
    https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
  tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
  rm actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
ENDSSH

echo ""
echo "=== Next steps (MANUAL) ==="
echo "1. Go to: https://github.com/YOUR_REPO/settings/actions/runners/new"
echo "2. Copy the registration token"
echo "3. SSH to VM: ssh ubuntu@$VM_IP"
echo "4. Run: cd ~/actions-runner && ./config.sh --url https://github.com/YOUR_REPO --token YOUR_TOKEN"
echo "5. Install as service: sudo ./svc.sh install && sudo ./svc.sh start"
echo "6. Update workflow: runs-on: self-hosted"
echo ""
echo "VM IP: $VM_IP"
