"""Small standard-library client for the monitoring integration API."""
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class MonitorAPIError(RuntimeError):
    """Raised when the monitor API rejects a request or cannot be reached."""


class MonitorClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def send_event(self, event: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/events", event)

    def poll_feedback(
        self,
        run_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = urlencode({"run_id": run_id, "limit": limit})
        response = self._request("GET", f"/v1/feedback?{query}")
        return response["feedback"]

    def get_audit(self, run_id: str) -> dict[str, Any]:
        query = urlencode({"run_id": run_id})
        return self._request("GET", f"/v1/audit?{query}")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = (
            json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if body is not None else None
        )
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                content = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MonitorAPIError(
                f"monitor API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise MonitorAPIError(f"monitor API request failed: {exc}") from exc
        decoded = json.loads(content.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise MonitorAPIError("monitor API response must be a JSON object")
        return decoded
