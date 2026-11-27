# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B: explicit ``app_name`` wins over ``INTERGRAX_CELERY_APP_NAME`` (CeleryIntegrationConfig parity)."""

from __future__ import annotations

import pytest
from intergrax.applications._shared.production_queue_resolver import (
    UnsupportedQueueBackendError,
    require_celery_broker_url_for_strict,
    resolve_celery_queue_env_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_explicit_app_name_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_CELERY_APP_NAME", "from_env_app")
    cfg = resolve_celery_queue_env_config(app_name="explicit_wins")
    assert cfg.app_name == "explicit_wins"


def test_env_used_when_app_name_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_CELERY_APP_NAME", "env_only_app")
    cfg = resolve_celery_queue_env_config()
    assert cfg.app_name == "env_only_app"


def test_default_app_name_without_env_or_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_CELERY_APP_NAME", raising=False)
    cfg = resolve_celery_queue_env_config()
    assert cfg.app_name == "tier3_host_task_worker"


def test_strict_broker_fail_closed_unchanged() -> None:
    with pytest.raises(UnsupportedQueueBackendError):
        require_celery_broker_url_for_strict(broker_url="")
