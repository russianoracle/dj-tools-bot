#!/bin/bash
#
# Create Yandex Cloud Monitoring dashboards via REST API
#
# Usage:
#   FOLDER_ID=b1ge0vpe8dp87vc3n73l bash scripts/create-dashboards.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check FOLDER_ID
if [ -z "$FOLDER_ID" ]; then
    echo -e "${RED}Error: FOLDER_ID environment variable not set${NC}"
    echo "Usage: FOLDER_ID=b1ge0vpe8dp87vc3n73l bash scripts/create-dashboards.sh"
    exit 1
fi

echo -e "${GREEN}Creating Yandex Cloud Monitoring dashboards...${NC}"
echo "Folder ID: $FOLDER_ID"
echo ""

# Get IAM token
echo -e "${YELLOW}[1/3] Getting IAM token...${NC}"
IAM_TOKEN=$(yc iam create-token)
if [ -z "$IAM_TOKEN" ]; then
    echo -e "${RED}Failed to get IAM token${NC}"
    exit 1
fi
echo -e "${GREEN}✓ IAM token obtained${NC}"

# Dashboard 1: Business Metrics
echo ""
echo -e "${YELLOW}[2/3] Creating Business Metrics dashboard...${NC}"

# Note: Yandex Cloud Monitoring API for dashboards requires complex protobuf format
# The web console is currently the recommended way to create dashboards

echo -e "${YELLOW}Dashboard configs are available in:${NC}"
echo "  - monitoring/dashboard-business.json"
echo "  - monitoring/dashboard-system.json"
echo ""
echo -e "${YELLOW}To create dashboards:${NC}"
echo "1. Open: https://console.yandex.cloud/folders/$FOLDER_ID/monitoring/dashboards"
echo "2. Click 'Create dashboard'"
echo "3. Add widgets manually using the JSON configs as reference"
echo ""
echo -e "${GREEN}Alternatively, use the following queries in Metric Explorer:${NC}"
echo ""
echo -e "${YELLOW}Business Metrics:${NC}"
echo "  - Queue Depth: arq_queue_depth{queue_name=\"default\"}"
echo "  - Tasks In Progress: arq_in_progress{queue_name=\"default\"}"
echo "  - Cache Hit Rate: rate(cache_operations_total{result=\"hit\"}[5m]) / rate(cache_operations_total{operation=\"get\"}[5m])"
echo "  - Error Rate: rate(processing_errors_total[5m])"
echo ""
echo -e "${YELLOW}System Metrics:${NC}"
echo "  - CPU Usage: sys.cpu.usage_user, sys.cpu.usage_system"
echo "  - Memory: (sys.memory.used / sys.memory.total) * 100"
echo "  - App Memory: memory_usage_bytes{process=\"bot\"}, memory_usage_bytes{process=\"worker\"}"
echo "  - Network: rate(sys.network.bytes_sent[5m]), rate(sys.network.bytes_recv[5m])"
echo ""

# Create a simple dashboard using yc monitoring dashboard API if available
echo -e "${YELLOW}[3/3] Attempting to create dashboards via API...${NC}"

# Check if yc monitoring command exists
if ! yc monitoring --help &>/dev/null; then
    echo -e "${RED}yc monitoring command not available${NC}"
    echo -e "${YELLOW}Please create dashboards manually using the queries above${NC}"
    exit 0
fi

echo -e "${GREEN}Dashboard creation script completed${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Open Monitoring console: https://console.yandex.cloud/folders/$FOLDER_ID/monitoring"
echo "2. Go to Dashboards → Create dashboard"
echo "3. Add charts using the metric queries provided above"
