#!/bin/bash
# Fetch secrets from Yandex Lockbox and create .env file
# Supports multiple authentication methods:
#   1. VM metadata service (production)
#   2. Service account key file (local development)
#   3. yc CLI profile (manual testing)
#
# Usage:
#   ./fetch-secrets.sh                              # Use VM metadata service
#   ./fetch-secrets.sh --sa-key key.json            # Use service account key file
#   ./fetch-secrets.sh --secret-id XXX              # Use custom secret ID
#   ./fetch-secrets.sh --sa-id aje6e9iq034u4cvf3cpp # Use specific SA
#
# Required secrets in Lockbox:
#   - TELEGRAM_BOT_TOKEN (required)
#   - ADMIN_USER_ID (optional)
#   - YC_LOG_GROUP_ID (optional, for cloud logging)

set -e

# Configuration
SECRET_ID="${YC_LOCKBOX_SECRET_ID:-e6qrhl953e11s6flf61n}"
SERVICE_ACCOUNT_ID="${YC_SERVICE_ACCOUNT_ID:-aje6e9iq034u4cvf3cpp}"
ENV_FILE="${ENV_FILE:-/opt/mood-classifier/.env}"
YC_BIN="${YC_BIN:-yc}"
SA_KEY_FILE=""
AUTH_METHOD="metadata"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --secret-id)
            SECRET_ID="$2"
            shift 2
            ;;
        --sa-id)
            SERVICE_ACCOUNT_ID="$2"
            shift 2
            ;;
        --sa-key)
            SA_KEY_FILE="$2"
            AUTH_METHOD="key-file"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --yc-profile)
            AUTH_METHOD="profile"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage:"
            echo "  $0 [options]"
            echo ""
            echo "Options:"
            echo "  --secret-id ID       Lockbox secret ID (default: e6qrhl953e11s6flf61n)"
            echo "  --sa-id ID           Service account ID (default: aje6e9iq034u4cvf3cpp)"
            echo "  --sa-key FILE        Service account key JSON file"
            echo "  --env-file PATH      Output .env file path (default: /opt/mood-classifier/.env)"
            echo "  --yc-profile         Use yc CLI profile authentication"
            exit 1
            ;;
    esac
done

echo "=== Yandex Lockbox Secrets Loader ==="
echo "Secret ID: ${SECRET_ID}"
echo "Service Account: ${SERVICE_ACCOUNT_ID}"
echo "Auth method: $AUTH_METHOD"
echo "Output: $ENV_FILE"

# Get IAM token based on authentication method
echo ""
echo "[1/4] Authenticating with Yandex Cloud..."

case $AUTH_METHOD in
    metadata)
        # Production: Use VM metadata service
        IAM_TOKEN=$(curl -sf -H 'Metadata-Flavor: Google' \
            http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token \
            | jq -r '.access_token') || {
            echo "ERROR: Failed to get IAM token from metadata service"
            echo "  Make sure this script runs on a YC VM with service account attached"
            echo ""
            echo "For local development, use:"
            echo "  $0 --sa-key /path/to/key.json"
            exit 1
        }
        export IAM_TOKEN
        echo "  ✅ IAM token obtained from VM metadata service"
        ;;

    key-file)
        # Local development: Use service account key file
        if [ ! -f "$SA_KEY_FILE" ]; then
            echo "ERROR: Service account key file not found: $SA_KEY_FILE"
            exit 1
        fi

        IAM_TOKEN=$(curl -sf -X POST \
            -H "Content-Type: application/json" \
            -d @"$SA_KEY_FILE" \
            https://iam.api.cloud.yandex.net/iam/v1/tokens \
            | jq -r '.iamToken') || {
            echo "ERROR: Failed to get IAM token using service account key"
            echo "  Check key file format and service account permissions"
            exit 1
        }
        export IAM_TOKEN
        echo "  ✅ IAM token obtained using service account key file"
        ;;

    profile)
        # Manual testing: Use yc CLI profile
        if ! command -v "$YC_BIN" &> /dev/null; then
            echo "ERROR: yc CLI not found"
            echo "  Install: curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash"
            exit 1
        fi

        # yc CLI will use its configured profile automatically
        echo "  ✅ Using yc CLI profile authentication"
        ;;
esac

# Fetch secrets from Lockbox
echo ""
echo "[2/4] Fetching secrets from Lockbox..."

# Set up yc command with IAM token if available
if [ -n "$IAM_TOKEN" ]; then
    export YC_TOKEN="$IAM_TOKEN"
fi

SECRETS=$($YC_BIN lockbox payload get --id "$SECRET_ID" --format json 2>&1) || {
    echo "ERROR: Failed to fetch secrets from Lockbox"
    echo "  - Secret ID: $SECRET_ID"
    echo "  - Service Account: $SERVICE_ACCOUNT_ID"
    echo "  - Required permissions: lockbox.payloadViewer"
    echo "  - Error: $SECRETS"
    echo ""
    echo "To grant permissions:"
    echo "  yc lockbox secret add-access-binding \\"
    echo "    --id $SECRET_ID \\"
    echo "    --service-account-id $SERVICE_ACCOUNT_ID \\"
    echo "    --role lockbox.payloadViewer"
    exit 1
}

# Check if secrets response is valid
SECRET_COUNT=$(echo "$SECRETS" | jq -r '.entries | length')
if [ "$SECRET_COUNT" -eq 0 ]; then
    echo "ERROR: No secrets found in Lockbox secret $SECRET_ID"
    exit 1
fi
echo "  ✅ Found $SECRET_COUNT secret entries"

# Write secrets to .env file
echo ""
echo "[3/4] Writing secrets to $ENV_FILE..."
mkdir -p "$(dirname "$ENV_FILE")"

# Convert Lockbox entries to .env format
echo "# Auto-generated from Yandex Lockbox" > "$ENV_FILE"
echo "# Secret: ${SECRET_ID:0:8}..." >> "$ENV_FILE"
echo "# Generated: $(date -Iseconds)" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

echo "$SECRETS" | jq -r '.entries[] | "\(.key | ascii_upcase)=\(.text_value)"' >> "$ENV_FILE"

# Add static infrastructure env vars
cat >> "$ENV_FILE" << 'EOF'

# Infrastructure (static)
DATA_DIR=/data
DOWNLOADS_DIR=/data/downloads
REDIS_HOST=redis
REDIS_PORT=6379
LOG_LEVEL=INFO
LOG_JSON_FORMAT=true
REQUIRE_SECRETS=true

# Yandex Cloud
REGISTRY=cr.yandex
REGISTRY_ID=crp78hhr2t67jkeedad1
YC_FOLDER_ID=b1g9a77bmpk1llb76sab
EOF

# Add Lockbox secret ID for Python runtime loading
echo "YC_LOCKBOX_SECRET_ID=$SECRET_ID" >> "$ENV_FILE"

chmod 600 "$ENV_FILE"
echo "  ✅ Secrets written to $ENV_FILE"

# Validate required secrets
echo ""
echo "[4/4] Validating secrets..."

REQUIRED_SECRETS="TELEGRAM_BOT_TOKEN YC_LOG_GROUP_ID"
OPTIONAL_SECRETS="ADMIN_USER_ID YTDLP_PROXY"
# Note: Lockbox keys are lowercase, converted to UPPERCASE for env vars
MISSING_REQUIRED=""
MISSING_OPTIONAL=""

# Source the env file to check values
set -a
source "$ENV_FILE"
set +a

for secret in $REQUIRED_SECRETS; do
    value="${!secret}"
    if [ -z "$value" ]; then
        MISSING_REQUIRED="$MISSING_REQUIRED $secret"
    else
        echo "  ✅ $secret: set (${#value} chars)"
    fi
done

for secret in $OPTIONAL_SECRETS; do
    if [ -z "${!secret}" ]; then
        MISSING_OPTIONAL="$MISSING_OPTIONAL $secret"
        echo "  ⚪ $secret: not set (optional)"
    else
        echo "  ✅ $secret: set"
    fi
done

if [ -n "$MISSING_REQUIRED" ]; then
    echo ""
    echo "❌ ERROR: Missing required secrets:$MISSING_REQUIRED"
    echo "Add these secrets to Lockbox secret: $SECRET_ID"
    exit 1
fi

echo ""
echo "=== ✅ Secrets loaded successfully ==="
echo "  Required: $(echo $REQUIRED_SECRETS | wc -w | tr -d ' ') secrets"
echo "  Optional: $(echo $OPTIONAL_SECRETS | wc -w | tr -d ' ') secrets"
echo ""
echo "Next steps:"
echo "  1. docker-compose --env-file $ENV_FILE up -d"
echo "  2. docker logs mood-classifier"