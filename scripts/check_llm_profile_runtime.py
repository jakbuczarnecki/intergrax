#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-6.7 — LLMProfile.validate_runtime() startup validation path."""

from __future__ import annotations

import sys


def main() -> int:
    from intergrax.llm_adapters.registry.profile import LLMProfile
    from intergrax.utils import attribute_access

    profile = LLMProfile.lab()
    if not attribute_access.is_callable_attr(profile, "validate_runtime"):
        print("LLMProfile.validate_runtime is missing or not callable", file=sys.stderr)
        return 1
    warnings = profile.validate_runtime()
    if not isinstance(warnings, list):
        print("validate_runtime must return list[str]", file=sys.stderr)
        return 1
    print(f"OK: LLMProfile.validate_runtime() ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
