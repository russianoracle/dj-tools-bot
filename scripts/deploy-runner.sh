#!/bin/bash
# ============================================================================
# Quick GitHub Actions Runner Deployment
# ============================================================================
set -e

RUNNER_TOKEN="AIWS4CBCYMKSKKQJYZLHEGLJH5NKU"
GITHUB_REPO="russianoracle/dj-tools-bot"
RUNNER_NAME="yandex-cloud-runner"
SSH_KEY="$HOME/.ssh/id_ed25519.pub"

echo "=== Creating GitHub Actions Runner VM ==="

# Read SSH key
SSH_PUB_KEY=$(cat ${SSH_KEY})

# Create cloud-init with substitutions
cat > /tmp/runner-cloud-init.yaml <<EOF
#cloud-config
users:
  - name: ubuntu
    ssh_authorized_keys:
      - ${SSH_PUB_KEY}
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: sudo, docker
    shell: /bin/bash

package_update: true
package_upgrade: true

packages:
  - curl
  - jq
  - git

runcmd:
  - |
    echo "=== Installing Docker ==="
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu
    systemctl enable docker
    systemctl start docker

  - |
    echo "=== Installing GitHub Runner ==="
    mkdir -p /home/ubuntu/actions-runner
    cd /home/ubuntu/actions-runner
    RUNNER_VERSION="2.311.0"
    curl -o actions-runner-linux-x64-\${RUNNER_VERSION}.tar.gz -L \\
      https://github.com/actions/runner/releases/download/v\${RUNNER_VERSION}/actions-runner-linux-x64-\${RUNNER_VERSION}.tar.gz
    tar xzf ./actions-runner-linux-x64-\${RUNNER_VERSION}.tar.gz
    rm actions-runner-linux-x64-\${RUNNER_VERSION}.tar.gz
    chown -R ubuntu:ubuntu /home/ubuntu/actions-runner

  - |
    echo "=== Configuring Runner ==="
    cd /home/ubuntu/actions-runner
    sudo -u ubuntu ./config.sh \\
      --url "https://github.com/${GITHUB_REPO}" \\
      --token "${RUNNER_TOKEN}" \\
      --name "${RUNNER_NAME}" \\
      --labels "self-hosted,yandex-cloud,docker" \\
      --work _work \\
      --unattended \\
      --replace

  - |
    echo "=== Installing as service ==="
    cd /home/ubuntu/actions-runner
    ./svc.sh install ubuntu
    ./svc.sh start
    systemctl status actions.runner.*.service

final_message: "GitHub Runner ready!"
EOF

# Create VM
yc compute instance create \
  --name github-runner \
  --zone ru-central1-a \
  --cores 4 \
  --memory 8 \
  --create-boot-disk image-folder-id=standard-images,image-family=ubuntu-2204-lts,size=50,type=network-ssd \
  --network-interface subnet-name=default-ru-central1-a,nat-ip-version=ipv4 \
  --metadata-from-file user-data=/tmp/runner-cloud-init.yaml \
  --async

echo "=== VM creation started! ==="
echo ""
echo "Wait ~3 minutes for setup, then check runner status:"
echo "  https://github.com/${GITHUB_REPO}/settings/actions/runners"
echo ""
echo "Get VM IP:"
echo "  yc compute instance get github-runner --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address'"
echo ""
echo "SSH to VM:"
echo "  ssh ubuntu@\$(yc compute instance get github-runner --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address')"
