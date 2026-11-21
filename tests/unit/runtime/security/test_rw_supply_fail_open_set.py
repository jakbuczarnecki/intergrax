# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Part F: the fail-open set (NT-RW-SUP-10, NT-RW-SUP-12b, NT-RW-SUP-13a).

NT-RW-SUP-11, NT-RW-SUP-12a and NT-RW-SUP-13b are written in the ticket's draft folder and are held
out of the merge set (their fixer paths are outside the write set of this ticket).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.llm_adapters.governance import quota
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.architecture.online_evaluation_registry import FileOnlineEvaluationRegistry

pytestmark = [pytest.mark.unit]

_QUOTA_ENV = "INTERGRAX_LLM_TENANT_MAX_TOKENS"


# --------------------------------------------------------------------- NT-RW-SUP-10 (F-D3-12)


@pytest.mark.parametrize("raw", ["ten", "1e3", " ", "12abc"])
def test_sup_10a_malformed_quota_denies(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(_QUOTA_ENV, raw)
    with pytest.raises(quota.LLMQuotaExceeded) as excinfo:
        quota.check_llm_tenant_quota("tenant-a")
    assert type(excinfo.value).__name__ == "LLMQuotaConfigurationError"
    assert _QUOTA_ENV in str(excinfo.value)
    if raw.strip():
        assert raw.strip() not in str(excinfo.value)


def test_sup_10b_negative_quota_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_QUOTA_ENV, "-5")
    with pytest.raises(quota.LLMQuotaExceeded) as excinfo:
        quota.check_llm_tenant_quota("tenant-a")
    assert type(excinfo.value).__name__ == "LLMQuotaConfigurationError"


def test_sup_10c_unset_and_zero_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_QUOTA_ENV, raising=False)
    quota.check_llm_tenant_quota("tenant-a", additional_tokens=10**9)
    monkeypatch.setenv(_QUOTA_ENV, "0")
    quota.check_llm_tenant_quota("tenant-a", additional_tokens=10**9)


# --------------------------------------------------------------------- NT-RW-SUP-12b (F-D6-11)


def test_sup_12b_request_identity_tenant_normalised() -> None:
    identity = RequestIdentity(tenant_id="  t  ", user_id="u")
    assert identity.tenant_id == "t"
    with pytest.raises(ValueError):
        RequestIdentity(tenant_id="   ", user_id="u")


# --------------------------------------------------------------------- NT-RW-SUP-13a (F-D5-12)


def _observation(observation_id: str) -> OnlineEvaluationObservation:
    return OnlineEvaluationObservation(
        observation_id=observation_id,
        run_id="run-1",
        agent_id="agent-1",
        mode=OnlineEvaluationMode.SHADOW,
        scenario_id="scenario-1",
        passed=True,
        score=1.0,
    )


def test_sup_13a_foreign_release_keeps_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer whose lock was taken over by another writer must not delete the new owner's lock."""
    path = tmp_path / "registry.json"
    registry = FileOnlineEvaluationRegistry(path, lock_timeout_seconds=0.2, stale_lock_seconds=60.0)
    lock_path = path.with_name(path.name + ".lock")
    foreign_token = f"{os.getpid() + 1}:takeover"
    original_save = registry._save

    def save_then_lose_lock(store: Any) -> None:
        original_save(store)
        # Simulates a second writer that removed the lock as stale and took it over.
        lock_path.write_text(foreign_token, encoding="ascii")

    monkeypatch.setattr(registry, "_save", save_then_lose_lock)
    with pytest.warns(RuntimeWarning, match="taken over"):
        registry.append(_observation("obs-1"))

    assert lock_path.is_file()
    assert lock_path.read_text(encoding="ascii") == foreign_token
