"""Reusable helpers for HSGuru scraping tasks."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import HEADERS, REQUEST_RETRIES, REQUEST_TIMEOUT_SECONDS, RETRY_BACKOFF_SECONDS
from utils.io import save_json


LOGGER = logging.getLogger(__name__)


def request_with_retry(url: str, retries: int = REQUEST_RETRIES, timeout: int = REQUEST_TIMEOUT_SECONDS) -> requests.Response:
    """Fetch a URL with a small retry policy."""

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            LOGGER.warning("Falha ao baixar %s (%s/%s): %s", url, attempt + 1, retries, error)
            time.sleep(RETRY_BACKOFF_SECONDS + attempt)

    assert last_error is not None
    raise last_error


def download_page(url: str) -> str:
    """Download a page and return its HTML content."""

    return request_with_retry(url).text


def parse_table(html: str) -> BeautifulSoup:
    """Parse HTML into a BeautifulSoup tree."""

    return BeautifulSoup(html, "html.parser")


def extract_deckcode(text: str) -> str | None:
    """Extract the Hearthstone deck code from arbitrary text."""

    match = re.search(r"AAECA[A-Za-z0-9+/=]+", text)
    return match.group(0) if match else None


def resolve_url(base_url: str, href: str) -> str:
    """Build an absolute URL from a base and a relative href."""

    return urljoin(base_url, href)


def save_json_file(data: Any, path: Path) -> None:
    """Persist JSON through the shared utility layer."""

    save_json(data, path)
