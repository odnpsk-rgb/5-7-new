"""Создание OpenAI-клиента с поддержкой VPN/прокси."""

from typing import Optional

import httpx
from openai import OpenAI

from http_proxy import get_httpx_client_kwargs
import os


def create_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Создаёт OpenAI-клиент.

    Прокси берётся из OPENAI_PROXY, TELEGRAM_PROXY или стандартных переменных окружения.
    """
    http_client = httpx.Client(
        **get_httpx_client_kwargs(),
        timeout=httpx.Timeout(60.0, connect=30.0),
    )
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        http_client=http_client,
    )
