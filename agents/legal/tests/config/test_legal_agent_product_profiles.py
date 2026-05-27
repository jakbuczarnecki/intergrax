# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from legal.config.legal_agent_config import LegalAgentConfig
from legal.config.legal_agent_product_profiles import LegalAgentProductProfile
from legal.memory.legal_memory_policy import LegalMemoryPolicy
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("profile", "attr", "expected"),
    [
        (LegalAgentProductProfile.FAST, "legal_loop_max_iterations", 2),
        (LegalAgentProductProfile.SAFE, "organization_allow_websearch", False),
        (LegalAgentProductProfile.RESEARCH, "legal_loop_max_iterations", 6),
        (LegalAgentProductProfile.STRICT_LEGAL, "legal_loop_early_exit_min_confidence", 0.95),
    ],
)
def test_apply_to_sets_expected_sku_fields(
    profile: LegalAgentProductProfile,
    attr: str,
    expected: object,
) -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter()
    base = LegalAgentConfig(session_manager=sm, llm_adapter=llm)
    updated = profile.apply_to(base)
    assert getattr(updated, attr) == expected
    assert updated is not base


def test_make_config_overrides_win_over_sku_defaults() -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter()
    cfg = LegalAgentProductProfile.FAST.make_config(
        session_manager=sm,
        llm_adapter=llm,
        legal_loop_max_iterations=9,
    )
    assert cfg.legal_loop_max_iterations == 9
    assert cfg.use_legal_run_evaluator is False


def test_apply_to_preserves_injected_handles() -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter()
    base = LegalAgentConfig(
        session_manager=sm,
        llm_adapter=llm,
        organization_allow_websearch=True,
        organization_allow_tools=True,
        legal_loop_max_iterations=3,
    )
    cfg = LegalAgentProductProfile.SAFE.apply_to(base)
    assert cfg.organization_allow_websearch is False
    assert cfg.organization_allow_tools is False
    assert cfg.session_manager is sm
    assert cfg.llm_adapter is llm


def test_parse_from_string_matches_enum_value() -> None:
    assert LegalAgentProductProfile("safe") is LegalAgentProductProfile.SAFE


def test_safe_sku_applies_minimal_exposure_memory_policy() -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter()
    cfg = LegalAgentProductProfile.SAFE.make_config(session_manager=sm, llm_adapter=llm)
    assert cfg.memory_policy.persist_workspace_snapshot_to_session is False
    assert cfg.memory_policy.hydrate_workspace_snapshot_from_session is False
    assert cfg.memory_policy.conversation_tail_message_limit == 6


def test_strict_legal_sku_applies_strict_workspace_memory_policy() -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter()
    cfg = LegalAgentProductProfile.STRICT_LEGAL.make_config(session_manager=sm, llm_adapter=llm)
    assert cfg.memory_policy.persist_workspace_snapshot_to_session is True
    assert cfg.memory_policy.conversation_tail_message_limit == 8


def test_make_config_memory_policy_overrides_sku() -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter()
    custom = LegalMemoryPolicy(
        persist_workspace_snapshot_to_session=True,
        hydrate_workspace_snapshot_from_session=True,
        conversation_tail_message_limit=99,
        conversation_snippet_max_chars_per_message=100,
    )
    cfg = LegalAgentProductProfile.SAFE.make_config(
        session_manager=sm,
        llm_adapter=llm,
        memory_policy=custom,
    )
    assert cfg.memory_policy is custom
    assert cfg.memory_policy.conversation_tail_message_limit == 99
