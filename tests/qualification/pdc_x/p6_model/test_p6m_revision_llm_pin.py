# © Artur Czarnecki. All rights reserved.

"""B-P6-M finding 9: an activated agent revision pins the LLM profile it was built against."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from intergrax.agent_distribution import admin_service as admin_service_module
from intergrax.agent_distribution.errors import RuntimeActivationConflict
from intergrax.agent_distribution.runtime_revision import (
    RuntimeRevision,
    RuntimeRevisionState,
    build_llm_profile_digest,
)
from intergrax.llm_adapters.registry.profile import LLMProfile
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.xd._support import build_sqlite_agent_stack

PROFILE_A = LLMProfile(provider="p6m_a", model="model-a")
PROFILE_B = LLMProfile(provider="p6m_b", model="model-b")


class _Env:
    """Mutable stand-in for the environment configuration that owns the LLM profile."""

    def __init__(self, profile: LLMProfile | None) -> None:
        self.profile = profile

    def source(self, application_id: str, application_environment_id: str) -> LLMProfile | None:
        assert (application_id, application_environment_id) == (h.APP, h.ENV)
        return self.profile


@pytest.fixture
def with_llm_source(monkeypatch: pytest.MonkeyPatch) -> Callable[[_Env | None], None]:
    """Compose the real admin service with an ``llm_profile_source`` (public constructor argument only).

    ``install(None)`` composes services without a profile source again.
    """
    original = admin_service_module.AgentPlatformAdminService.__init__

    def install(env: _Env | None) -> None:
        if env is None:
            monkeypatch.setattr(admin_service_module.AgentPlatformAdminService, "__init__", original)
            return

        def init(self, **kwargs):  # type: ignore[no-untyped-def]
            original(self, llm_profile_source=env.source, **kwargs)

        monkeypatch.setattr(admin_service_module.AgentPlatformAdminService, "__init__", init)

    return install


def _ready(stack: h.D1Stack) -> None:
    h.install_bind_enable(stack, h.release("1.0.0"))


def _serving_revision(stack: h.D1Stack) -> str | None:
    return stack.service.inspect_serving(
        application_id=h.APP, application_environment_id=h.ENV
    ).traffic_serving_revision_id


def test_profile_digest_is_deterministic_and_profile_sensitive() -> None:
    digest = build_llm_profile_digest(PROFILE_A)
    assert digest == build_llm_profile_digest(LLMProfile(provider="p6m_a", model="model-a"))
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert digest != build_llm_profile_digest(PROFILE_B)
    assert digest != build_llm_profile_digest(LLMProfile(provider="p6m_a", model="model-a2"))
    assert digest != build_llm_profile_digest(LLMProfile(provider="p6m_a", model="model-a", options={"k": 1}))


def test_runtime_revision_pin_is_optional_and_legacy_payloads_validate() -> None:
    assert RuntimeRevision.model_fields["llm_profile_digest"].default is None
    candidate = RuntimeRevision(
        runtime_revision_id="r",
        application_id="a",
        application_environment_id="e",
        application_release_id="rel",
        platform_version="1",
        effective_roster_revision_id="ro",
        revision_state=RuntimeRevisionState.CANDIDATE,
    )
    assert candidate.llm_profile_digest is None
    legacy = candidate.model_dump()
    del legacy["llm_profile_digest"]
    assert RuntimeRevision.model_validate(legacy).llm_profile_digest is None
    pinned = candidate.model_copy(update={"llm_profile_digest": build_llm_profile_digest(PROFILE_A)})
    assert RuntimeRevision.model_validate_json(pinned.model_dump_json()).llm_profile_digest == pinned.llm_profile_digest


def test_revision_without_a_configured_llm_profile_builds_unpinned_and_activates() -> None:
    stack = h.build_stack()
    _ready(stack)
    h.build_revision(stack, "rev-1")
    stored = stack.revision_store.get_revision("rev-1")
    assert stored is not None and stored.llm_profile_digest is None
    h.activate_revision(stack, "rev-1")
    assert _serving_revision(stack) == "rev-1"


def test_build_pins_the_configured_profile_and_activation_accepts_a_matching_profile(with_llm_source) -> None:  # type: ignore[no-untyped-def]
    env = _Env(PROFILE_A)
    with_llm_source(env)
    stack = h.build_stack()
    _ready(stack)
    h.build_revision(stack, "rev-1")
    stored = stack.revision_store.get_revision("rev-1")
    assert stored is not None
    assert stored.llm_profile_digest == build_llm_profile_digest(PROFILE_A)
    assert stored.revision_state is RuntimeRevisionState.VALIDATED
    h.activate_revision(stack, "rev-1")
    assert _serving_revision(stack) == "rev-1"


def test_activation_fails_closed_when_the_profile_changed_after_the_build(with_llm_source) -> None:  # type: ignore[no-untyped-def]
    env = _Env(PROFILE_A)
    with_llm_source(env)
    stack = h.build_stack()
    _ready(stack)
    h.build_revision(stack, "rev-1")
    env.profile = PROFILE_B  # config edit that swaps the model after the revision was built
    with pytest.raises(RuntimeActivationConflict):
        h.activate_revision(stack, "rev-1")
    assert _serving_revision(stack) is None, "a blocked activation must not move traffic"
    env.profile = PROFILE_A  # the pin is restored: the same revision now activates
    h.activate_revision(stack, "rev-1")
    assert _serving_revision(stack) == "rev-1"


def test_activation_fails_closed_when_the_pinned_profile_no_longer_resolves(with_llm_source) -> None:  # type: ignore[no-untyped-def]
    env = _Env(PROFILE_A)
    with_llm_source(env)
    stack = h.build_stack()
    _ready(stack)
    h.build_revision(stack, "rev-1")
    env.profile = None
    with pytest.raises(RuntimeActivationConflict):
        h.activate_revision(stack, "rev-1")
    assert _serving_revision(stack) is None


def test_activation_fails_closed_when_no_profile_source_is_composed_for_a_pinned_revision(  # type: ignore[no-untyped-def]
    with_llm_source, tmp_path: Path
) -> None:
    db = tmp_path / "agents.sqlite3"
    with_llm_source(_Env(PROFILE_A))
    builder = build_sqlite_agent_stack(db)
    h.install_bind_enable(builder, h.release("1.0.0"))
    h.build_revision(builder, "rev-1")
    # The same durable state served by a service that has no profile source cannot verify the pin.
    with_llm_source(None)
    unverifiable = build_sqlite_agent_stack(db)
    with pytest.raises(RuntimeActivationConflict):
        h.activate_revision(unverifiable, "rev-1")
    assert _serving_revision(unverifiable) is None


def test_unpinned_revision_keeps_activating_after_a_profile_is_configured(with_llm_source) -> None:  # type: ignore[no-untyped-def]
    env = _Env(None)
    with_llm_source(env)
    stack = h.build_stack()
    _ready(stack)
    h.build_revision(stack, "rev-1")
    stored = stack.revision_store.get_revision("rev-1")
    assert stored is not None and stored.llm_profile_digest is None
    env.profile = PROFILE_B
    h.activate_revision(stack, "rev-1")
    assert _serving_revision(stack) == "rev-1"


def test_pin_survives_a_durable_store_reopen(with_llm_source, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    env = _Env(PROFILE_A)
    with_llm_source(env)
    db = tmp_path / "agents.sqlite3"
    stack = build_sqlite_agent_stack(db)
    h.install_bind_enable(stack, h.release("1.0.0"))
    h.build_revision(stack, "rev-1")
    reopened = build_sqlite_agent_stack(db)
    stored = reopened.revision_store.get_revision("rev-1")
    assert stored is not None
    assert stored.llm_profile_digest == build_llm_profile_digest(PROFILE_A)
