# © Artur Czarnecki. All rights reserved.

"""File-backed registry for harness online/shadow evaluation observations (W-OPS.11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationObservation,
    OnlineEvaluationRegistryStore,
)


class OnlineEvaluationRegistry(Protocol):
    """Append-only store for harness evaluation observations."""

    def append(self, observation: OnlineEvaluationObservation) -> None: ...

    def list_observations(self) -> list[OnlineEvaluationObservation]: ...

    def clear(self) -> None: ...


class InMemoryOnlineEvaluationRegistry:
    """In-process registry for unit tests."""

    def __init__(self) -> None:
        self._observations: list[OnlineEvaluationObservation] = []

    def append(self, observation: OnlineEvaluationObservation) -> None:
        self._observations.append(observation)

    def list_observations(self) -> list[OnlineEvaluationObservation]:
        return list(self._observations)

    def clear(self) -> None:
        self._observations.clear()


class FileOnlineEvaluationRegistry:
    """Persist observations under ``build/architecture_hardening/`` (gitignored)."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_registry_path()

    def append(self, observation: OnlineEvaluationObservation) -> None:
        store = self._load()
        store.observations.append(observation)
        self._save(store)

    def list_observations(self) -> list[OnlineEvaluationObservation]:
        return self._load().observations

    def clear(self) -> None:
        self._save(OnlineEvaluationRegistryStore())

    def _load(self) -> OnlineEvaluationRegistryStore:
        if not self._path.is_file():
            return OnlineEvaluationRegistryStore()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return OnlineEvaluationRegistryStore.model_validate(payload)

    def _save(self, store: OnlineEvaluationRegistryStore) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(store.model_dump_json(indent=2), encoding="utf-8")


def default_registry_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "architecture_hardening" / "online_evaluation_observations.json"


_default_file_registry: FileOnlineEvaluationRegistry | None = None


def default_online_evaluation_registry() -> FileOnlineEvaluationRegistry:
    global _default_file_registry
    if _default_file_registry is None:
        _default_file_registry = FileOnlineEvaluationRegistry()
    return _default_file_registry


def reset_default_online_evaluation_registry_for_tests() -> None:
    """Clear module singleton between tests."""
    global _default_file_registry
    _default_file_registry = None
