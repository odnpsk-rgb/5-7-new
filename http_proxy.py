"""Настройка HTTP-прокси для VPN (httpx не поддерживает socks4 из системных переменных)."""

import os
from typing import Any, Optional

_PROXY_ENV_KEYS = ("TELEGRAM_PROXY", "OPENAI_PROXY", "ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY")


def normalize_proxy_url(url: str) -> str:
    """Большинство VPN принимают socks5 на том же порту, что и socks4."""
    url = url.strip()
    if url.startswith("socks4://"):
        return "socks5://" + url[len("socks4://") :]
    if url.startswith("socks4a://"):
        return "socks5h://" + url[len("socks4a://") :]
    return url


def get_proxy_url() -> Optional[str]:
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return normalize_proxy_url(value)
    return None


def get_httpx_client_kwargs() -> dict[str, Any]:
    """Параметры для httpx.Client / httpx.AsyncClient."""
    return {
        "proxy": get_proxy_url(),
        "trust_env": False,
    }
