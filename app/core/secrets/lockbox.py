"""Yandex Cloud Lockbox secrets loader.

Usage:
    # At application startup
    from app.core.secrets import init_secrets
    init_secrets()  # Uses YC_LOCKBOX_SECRET_ID env var

    # Or manually
    from app.core.secrets import load_secrets_to_env
    load_secrets_to_env("e6qrhl953e11s6flf61n")

Environment Variables:
    YC_LOCKBOX_SECRET_ID: Lockbox secret ID (default: e6qrhl953e11s6flf61n)
    YC_SERVICE_ACCOUNT_ID: Service account ID (default: aje6e9iq034u4cvf3cpp)
    YC_SA_KEY_FILE: Path to service account key JSON file (optional, for local dev)
"""

import os
from typing import Optional

from app.common.logging import get_logger

logger = get_logger(__name__)

# Default Yandex Cloud configuration
DEFAULT_SECRET_ID = "e6qrhl953e11s6flf61n"
DEFAULT_SERVICE_ACCOUNT_ID = "aje6e9iq034u4cvf3cpp"

# Lazy imports for yandexcloud SDK
_lockbox_client = None


def _get_lockbox_client():
    """Lazy init Lockbox client."""
    global _lockbox_client
    if _lockbox_client is not None:
        return _lockbox_client

    try:
        import yandexcloud
        from yandex.cloud.lockbox.v1.payload_service_pb2_grpc import PayloadServiceStub

        sa_key_file = os.getenv("YC_SA_KEY_FILE")

        if sa_key_file and os.path.exists(sa_key_file):
            import json
            with open(sa_key_file) as f:
                sa_key = json.load(f)
            sdk = yandexcloud.SDK(service_account_key=sa_key)
        else:
            # Use metadata service (for VMs/Cloud Functions/Containers)
            sdk = yandexcloud.SDK()

        _lockbox_client = sdk.client(PayloadServiceStub)
        return _lockbox_client

    except ImportError:
        logger.warning("yandexcloud SDK not installed, Lockbox disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to init Lockbox client: {e}")
        return None


def get_secret(secret_id: str, version_id: Optional[str] = None) -> dict[str, str]:
    """
    Get secret payload from Yandex Lockbox.

    Args:
        secret_id: Lockbox secret ID
        version_id: Optional specific version (default: latest)

    Returns:
        Dict mapping key names to values
    """
    client = _get_lockbox_client()
    if client is None:
        return {}

    try:
        from yandex.cloud.lockbox.v1.payload_service_pb2 import GetPayloadRequest

        request = GetPayloadRequest(secret_id=secret_id)
        if version_id:
            request.version_id = version_id

        response = client.Get(request)

        secrets = {}
        for entry in response.entries:
            if entry.text_value:
                secrets[entry.key] = entry.text_value
            elif entry.binary_value:
                secrets[entry.key] = entry.binary_value.decode("utf-8")

        return secrets

    except Exception as e:
        logger.error(f"Failed to get secret {secret_id}: {e}")
        return {}


def load_secrets_to_env(secret_id: str, prefix: str = "", override: bool = False) -> list[str]:
    """
    Load secrets from Lockbox and set as environment variables.

    Args:
        secret_id: Lockbox secret ID
        prefix: Optional prefix to add to env var names
        override: Whether to override existing env vars

    Returns:
        List of loaded variable names
    """
    secrets = get_secret(secret_id)
    loaded = []

    for key, value in secrets.items():
        env_key = f"{prefix}{key}" if prefix else key

        if override or env_key not in os.environ:
            os.environ[env_key] = value
            loaded.append(env_key)
            logger.debug(f"Loaded secret: {env_key}")

    logger.info(f"Loaded {len(loaded)} secrets from Lockbox")
    return loaded


# Required secrets for the application
REQUIRED_SECRETS = [
    "TELEGRAM_BOT_TOKEN",
]

# Optional secrets (won't fail if missing)
OPTIONAL_SECRETS = [
    "ADMIN_USER_ID",
    "YC_LOG_GROUP_ID",
    "YC_FOLDER_ID",
    "YTDLP_PROXY",
]


def validate_secrets() -> tuple[bool, list[str]]:
    """
    Validate that all required secrets are present in environment.

    Returns:
        Tuple of (success, list of missing required secrets)
    """
    missing = []

    for secret in REQUIRED_SECRETS:
        if not os.getenv(secret):
            missing.append(secret)

    if missing:
        logger.error(f"Missing required secrets: {missing}")
        return False, missing

    # Log optional secrets status
    for secret in OPTIONAL_SECRETS:
        if os.getenv(secret):
            logger.debug(f"Optional secret present: {secret}")
        else:
            logger.debug(f"Optional secret not set: {secret}")

    logger.info("All required secrets validated")
    return True, []


def init_secrets(
    secret_id: Optional[str] = None,
    validate: bool = True,
    fail_on_missing: bool = True,
) -> bool:
    """
    Initialize secrets from Lockbox and validate.

    Args:
        secret_id: Lockbox secret ID (uses env YC_LOCKBOX_SECRET_ID or default)
        validate: Whether to validate required secrets
        fail_on_missing: Raise exception if required secrets missing

    Returns:
        True if successful

    Raises:
        RuntimeError: If fail_on_missing=True and secrets are missing
    """
    sid = secret_id or os.getenv("YC_LOCKBOX_SECRET_ID") or DEFAULT_SECRET_ID

    if sid:
        loaded = load_secrets_to_env(sid)
        logger.info(f"Loaded {len(loaded)} secrets from Lockbox: {sid[:8]}...")
    else:
        logger.info("No Lockbox secret ID configured, using env vars only")

    if validate:
        success, missing = validate_secrets()
        if not success:
            msg = f"Missing required secrets: {missing}"
            if fail_on_missing:
                raise RuntimeError(msg)
            logger.warning(msg)
            return False

    return True