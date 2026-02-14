"""Implementation module for Label Studio integration.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

from dataset_tools.config import AppConfig, require_label_studio_api_key


def _import_label_studio_client() -> Any:
    """Internal helper for import label studio client.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    try:
        from label_studio_sdk import Client  # type: ignore

        return Client
    except ImportError:
        try:
            from label_studio_sdk._legacy import Client  # type: ignore

            return Client
        except ImportError as e:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "label_studio_sdk is not installed. Install it with `pip install label-studio-sdk`."
            ) from e


def _resolve_access_token(url: str, api_key: str) -> str:
    # If key looks like JWT refresh token, exchange it for access token.
    """Resolve access token from provided inputs.

Args:
    url: Value controlling url for this routine.
    api_key: Value controlling api key for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if not (api_key.startswith("eyJ") and len(api_key) > 50):
        return api_key

    if requests is None:
        return api_key

    refresh_url = f"{url.rstrip('/')}/api/token/refresh/"
    try:
        response = requests.post(refresh_url, json={"refresh": api_key}, timeout=10)
        if response.status_code == 200:
            access_token = response.json().get("access")
            if access_token:
                return access_token
    except requests.RequestException:
        # Fall through and return original token
        pass

    return api_key


def connect_to_label_studio(url: str, api_key: str):
    """Connect to to label studio and return a ready client.

Args:
    url: Value controlling url for this routine.
    api_key: Value controlling api key for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    Client = _import_label_studio_client()
    final_api_key = _resolve_access_token(url=url, api_key=api_key)

    ls = Client(url=url, api_key=final_api_key)
    if hasattr(ls, "check_connection"):
        if not ls.check_connection():
            raise RuntimeError(f"Failed to connect to Label Studio at {url}")

    return ls


def ensure_label_studio_client(config: AppConfig):
    """Ensure label studio client exists and return it.

Args:
    config: Configuration object controlling runtime behavior.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    api_key = require_label_studio_api_key(config)
    return connect_to_label_studio(
        url=config.label_studio.url,
        api_key=api_key,
    )
