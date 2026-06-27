#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-16.2 — semantic compression in production profiles."""

from __future__ import annotations

import sys

from intergrax.applications._shared.context_runtime_bridge import apply_context_profiles_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.nexus.config import RuntimeConfig
from testing_support.builder import FakeLLMAdapter


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    ctx = env.context_profile
    if not ctx.semantic_compression_enabled:
        print("product_defaults must enable semantic_compression", file=sys.stderr)
        return 1
    if ctx.default_history_compression != "summarize_oldest":
        print("product_defaults must use summarize_oldest compression", file=sys.stderr)
        return 1

    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), enable_rag=False, enable_websearch=False)
    apply_context_profiles_from_environment(config, env)
    compression = config.metadata.get("semantic_compression.v1")
    if not isinstance(compression, dict) or not compression.get("enabled"):
        print("runtime config missing semantic_compression metadata", file=sys.stderr)
        return 1

    print("OK: semantic compression profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
