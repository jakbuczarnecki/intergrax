#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""M-LLM-X.7.3 — warn when adapter default models are missing from model catalog YAML."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
    from intergrax.llm_adapters.registry.model_catalog import get_model_catalog
    from intergrax.llm_adapters.registry.profile import LLMProfile

    catalog = get_model_catalog()
    warnings: list[str] = []
    for provider in LLMProvider:
        profile = LLMProfile(provider=provider)
        model = (profile.model or "").strip()
        if not model:
            continue
        if catalog.lookup_exact(model) is None and catalog.lookup_prefix(model) is None:
            warnings.append(f"{provider.value}/{model}")

    if warnings:
        print("model catalog coverage warnings (default adapter models missing from YAML):")
        for item in sorted(warnings):
            print(f"  - {item}")
        return 1
    print("model catalog coverage: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
