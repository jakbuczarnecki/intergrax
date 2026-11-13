# © Artur Czarnecki. All rights reserved.

"""PDC-D2-R008 / PDC-D6-R014: retrieval-poisoning defense is ON when the profile is absent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from intergrax.applications._shared.security_assembly_resolver import validate_security_wiring
from intergrax.applications._shared.security_wiring import wire_application_security
from intergrax.core.security.security_configuration import (
    retrieval_poisoning_defense_enabled_or_default,
)
from intergrax.tools.providers.rag import service as rag_service
from intergrax.tools.providers.rag.contracts import RagChunkResult, RagCitationResult
from intergrax.tools.registry.wiring import ToolWiringContext

from ._support import product_env

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_POISON_TEXT = "untrusted low-trust chunk"


def _chunks() -> tuple[list[RagChunkResult], list[RagCitationResult]]:
    chunks = [
        RagChunkResult(id="clean", text="quarterly revenue grew", score=0.9, rank=1),
        RagChunkResult(id="poison", text=_POISON_TEXT, score=0.05, rank=2),
    ]
    citations = [
        RagCitationResult(chunk_id="clean", source_id="a"),
        RagCitationResult(chunk_id="poison", source_id="b"),
    ]
    return chunks, citations


def _ctx(**kwargs: object) -> ToolWiringContext:
    return ToolWiringContext(**kwargs)  # type: ignore[arg-type]


def test_d6_r014_missing_profile_still_enforces_the_filter() -> None:
    chunks, citations = _chunks()
    ctx = _ctx()
    assert ctx.security_profile is None and "security_profile" not in ctx.extras
    kept, kept_citations, _reason, _warnings = rag_service._apply_retrieval_poisoning_filter(ctx, chunks, citations)
    assert [chunk.id for chunk in kept] == ["clean"]
    assert [citation.chunk_id for citation in kept_citations] == ["clean"]


def test_d6_r014_profile_without_the_toggle_still_enforces_the_filter() -> None:
    chunks, citations = _chunks()
    ctx = _ctx(security_profile=SimpleNamespace())
    kept, _citations, _reason, _warnings = rag_service._apply_retrieval_poisoning_filter(ctx, chunks, citations)
    assert [chunk.id for chunk in kept] == ["clean"]


def test_d6_r014_explicit_false_is_the_only_way_to_disable() -> None:
    chunks, citations = _chunks()
    ctx = _ctx(security_profile=SimpleNamespace(retrieval_poisoning_defense_enabled=False))
    kept, _citations, _reason, _warnings = rag_service._apply_retrieval_poisoning_filter(ctx, chunks, citations)
    assert [chunk.id for chunk in kept] == ["clean", "poison"]


@pytest.mark.parametrize("malformed", [None, "false", 0, "", object()])
def test_d6_r014_malformed_toggle_fails_closed(malformed: object) -> None:
    profile = SimpleNamespace(retrieval_poisoning_defense_enabled=malformed)
    assert retrieval_poisoning_defense_enabled_or_default(profile) is True


def test_d6_r014_default_resolver_contract() -> None:
    assert retrieval_poisoning_defense_enabled_or_default(None) is True
    enabled = SimpleNamespace(retrieval_poisoning_defense_enabled=True)
    disabled = SimpleNamespace(retrieval_poisoning_defense_enabled=False)
    assert retrieval_poisoning_defense_enabled_or_default(enabled) is True
    assert retrieval_poisoning_defense_enabled_or_default(disabled) is False


def test_d2_r008_security_assembly_uses_the_same_default() -> None:
    env = product_env()
    wiring = wire_application_security(env)
    assert wiring.options.retrieval_poisoning_defense_enabled is True
    assert validate_security_wiring(wiring, env).valid is True

    disabled = product_env(retrieval_poisoning_defense_enabled=False)
    disabled_wiring = wire_application_security(disabled)
    assert disabled_wiring.options.retrieval_poisoning_defense_enabled is False
    assert validate_security_wiring(disabled_wiring, disabled).valid is True
    # a wiring that silently turned the defense off against an enabled profile is rejected
    assert validate_security_wiring(disabled_wiring, env).valid is False
