#!/bin/bash
set -e

REPOSITORY_ID="crpcd8cjkmi6pvubugmn"
REPOSITORY_NAME="crp78hhr2t67jkeedad1/mood-classifier"
POLICY_NAME="cleanup-old-images"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RULES_FILE="$SCRIPT_DIR/registry-lifecycle-policy.json"

echo "Setting up lifecycle policy for $REPOSITORY_NAME..."

# Check if policy already exists
EXISTING_POLICY=$(yc container repository lifecycle-policy list \
  --repository-id "$REPOSITORY_ID" \
  --format json 2>/dev/null | jq -r '.[0].id // empty')

if [ -n "$EXISTING_POLICY" ]; then
  echo "Policy already exists (ID: $EXISTING_POLICY), updating..."
  yc container repository lifecycle-policy update "$EXISTING_POLICY" \
    --rules "$RULES_FILE" \
    --active
  echo "✅ Policy updated successfully"
else
  echo "Creating new lifecycle policy..."
  yc container repository lifecycle-policy create \
    --repository-id "$REPOSITORY_ID" \
    --name "$POLICY_NAME" \
    --description "Auto-cleanup: keep last 3 SHA images, delete untagged" \
    --rules "$RULES_FILE" \
    --active
  echo "✅ Policy created successfully"
fi

echo ""
echo "Current policy:"
yc container repository lifecycle-policy list \
  --repository-id "$REPOSITORY_ID"