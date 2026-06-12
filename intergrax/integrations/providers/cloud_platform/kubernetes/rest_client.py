# © Artur Czarnecki. All rights reserved.

"""Kubernetes Deployment scale REST client (ECP-PROD.3)."""

from __future__ import annotations

from typing import Any

import httpx


class KubernetesDeploymentScaleClient:
    """Scale Deployments via the apps/v1 scale subresource."""

    def __init__(
        self,
        *,
        base_url: str,
        namespace: str,
        token: str = "",
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._namespace = namespace
        self._token = token
        self._timeout = timeout_seconds
        self._transport = transport

    def health(self) -> bool:
        return bool(self._base_url)

    def get_replicas(self, deployment: str, *, namespace: str | None = None) -> int:
        ns = namespace or self._namespace
        response = self._request(
            "GET",
            self._scale_path(deployment, ns),
        )
        response.raise_for_status()
        body = response.json()
        status = body.get("status") or {}
        return int(status.get("replicas", 0))

    def scale_workload(self, deployment: str, *, replicas: int, namespace: str | None = None) -> int:
        ns = namespace or self._namespace
        response = self._request(
            "PATCH",
            self._scale_path(deployment, ns),
            json={"spec": {"replicas": replicas}},
            headers={"Content-Type": "application/strategic-merge-patch+json"},
        )
        response.raise_for_status()
        return replicas

    def _scale_path(self, deployment: str, namespace: str) -> str:
        return f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale"

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        if self._token:
            headers.setdefault("Authorization", f"Bearer {self._token}")
        with httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            return client.request(method, path, headers=headers, **kwargs)
