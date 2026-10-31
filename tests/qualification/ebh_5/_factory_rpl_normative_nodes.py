# © Artur Czarnecki. All rights reserved.

"""Normative test node bindings for factory one-pass RPL (no intergrax imports)."""

from __future__ import annotations

from typing import Final

FACTORY_RPL_ONE_PASS_CONCERNS: Final[frozenset[str]] = frozenset(
    {
        "llm.chat_completion",
        "llm.embedding",
        "llm.rerank",
        "llm.fallback_policy",
        "llm.model_routing_qualifier",
        "plugin.llm_adapter_registry",
        "plugin.integration_catalog",
        "env.integration_slug",
        "distributed.sqlite_kv",
        "execution.integration_bound_resolution",
        "tool.observability_backend",
        "speech.provider",
    }
)

_ONE_PASS_TEST_MODULE: Final[str] = "tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py"


def factory_rpl_positive_node(concern_id: str) -> str:
    return f"{_ONE_PASS_TEST_MODULE}::test_ebh5_factory_rpl_positive[{concern_id}]"


def factory_rpl_negative_node(concern_id: str) -> str:
    return f"{_ONE_PASS_TEST_MODULE}::test_ebh5_factory_rpl_negative[{concern_id}]"
