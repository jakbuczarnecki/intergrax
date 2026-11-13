# © Artur Czarnecki. All rights reserved.

"""Shared helpers for the PDC-X B-D6 (runtime security) qualification tests."""

from __future__ import annotations

import secrets
from pathlib import Path

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.events.store import resolve_runtime_event_persistence

REPO_ROOT = Path(__file__).resolve().parents[4]


def generated_secret() -> str:
    """A fresh, generated test value (never a real credential)."""
    return "tv-" + secrets.token_urlsafe(24)


def product_env(**security_updates: object) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.product_defaults()
    if security_updates:
        env.security_profile = env.security_profile.model_copy(update=security_updates)
    return env


def lab_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults()


def open_evidence_plane(db_path: Path):
    """Canonical evidence persistence (SQLite runtime event store behind the validating port)."""
    port = resolve_runtime_event_persistence(db_path=db_path)
    assert port is not None
    return port


class CountingSecretsStore:
    """In-memory ``SecretsStore`` fake that counts reads (external provider stand-in)."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values: dict[str, str] = dict(values or {})
        self.reads: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.reads.append(path)
        return self.values[path]

    def put_secret(self, path: str, value: str) -> None:
        self.values[path] = value

    def delete_secret(self, path: str) -> None:
        self.values.pop(path, None)
