"""World Bank API client for macroeconomic indicators."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorldBankClient:
    """Small retrying client for the World Bank indicator API."""

    base_url: str = "https://api.worldbank.org/v2"
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_seconds: float = 1.5

    def fetch_indicator(
        self,
        country_code: str,
        indicator_code: str,
        *,
        per_page: int = 20000,
    ) -> list[dict[str, Any]]:
        """Fetch one indicator for one country from the World Bank API.

        Parameters
        ----------
        country_code:
            ISO-style country code used by the World Bank API.
        indicator_code:
            World Bank indicator code, for example ``NY.GDP.MKTP.KD.ZG``.
        per_page:
            Large page size used because annual country-indicator responses are
            small. Pagination metadata is still preserved in the raw response.

        Returns
        -------
        list[dict[str, Any]]
            The decoded World Bank JSON response. World Bank returns a
            two-element list: pagination metadata and observation records.
        """

        url = f"{self.base_url}/country/{country_code}/indicator/{indicator_code}"
        params: dict[str, Any] = {
            "format": "json",
            "per_page": per_page,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                LOGGER.info(
                    "Fetching World Bank indicator %s for %s (attempt %s/%s)",
                    indicator_code,
                    country_code,
                    attempt,
                    self.max_retries,
                )
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, list):
                    raise ValueError(f"Unexpected World Bank response shape: {payload!r}")

                if payload and isinstance(payload[0], dict) and "message" in payload[0]:
                    raise ValueError(f"World Bank API error: {payload[0]['message']!r}")

                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                LOGGER.warning(
                    "World Bank request failed for %s/%s on attempt %s: %s",
                    country_code,
                    indicator_code,
                    attempt,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)

        raise RuntimeError(
            f"Failed to fetch World Bank data for {country_code}/{indicator_code}"
        ) from last_error
