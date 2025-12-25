#!/bin/bash
# Setup 2GB swap file on production VM
# Provides buffer for memory spikes without hard OOM kills
# Usage: ssh -i ~/.ssh/tender-bot-key ubuntu@158.160.122.216 'bash -s' < scripts/setup-swap.sh

set -e

echo "=========================================="
echo "SWAP CONFIGURATION SETUP"
echo "=========================================="
echo ""

# Check if swap already exists
if swapon --show | grep -q '/swapfile'; then
    echo "✅ Swap already configured:"
    swapon --show
    free -h
    exit 0
fi

echo "📦 Creating 2GB swap file..."

# Create 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

echo "✅ Swap file created and activated"
echo ""

# Make permanent (add to /etc/fstab)
if ! grep -q '/swapfile' /etc/fstab; then
    echo "📝 Adding to /etc/fstab for persistence..."
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Set swappiness to 10 (only swap when RAM critical)
echo "⚙️  Tuning swappiness to 10 (aggressive RAM usage)..."
sudo sysctl vm.swappiness=10

# Make swappiness permanent
if ! grep -q 'vm.swappiness' /etc/sysctl.conf; then
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
fi

echo ""
echo "=========================================="
echo "SWAP CONFIGURATION COMPLETE"
echo "=========================================="
echo ""
echo "Current status:"
swapon --show
echo ""
free -h
echo ""
echo "Configuration:"
echo "  Size: 2 GB"
echo "  Swappiness: 10 (only swap when RAM >90% full)"
echo "  Persistent: Yes (/etc/fstab)"
echo ""
echo "✅ System now has graceful degradation for memory spikes"
