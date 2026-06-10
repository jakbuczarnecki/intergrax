#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-17.2 — prompt diff / compare API for managed prompts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.prompt_diff_wiring import (
    compare_prompt_documents_for_host,
    prompt_compare_enabled,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.prompts.schema.prompt_governance import PromptRiskTier
from intergrax.prompts.schema.prompt_schema import LocalizedContent, LocalizedPromptDocument, PromptMeta


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    if not prompt_compare_enabled(env):
        print("product host must enable prompt compare", file=sys.stderr)
        return 1

    meta = PromptMeta(
        model_family="gpt",
        output_schema_id="text",
        tags=frozenset(),
        description="demo",
        owner_team="platform",
        owner_contact="platform@intergrax",
        risk_tier=PromptRiskTier.LOW,
    )
    left = LocalizedPromptDocument(
        id="demo.prompt",
        version=1,
        locales={"en": LocalizedContent(system="v1", developer=None, user_template=None)},
        meta=meta,
    )
    right = LocalizedPromptDocument(
        id="demo.prompt",
        version=2,
        locales={"en": LocalizedContent(system="v2", developer=None, user_template=None)},
        meta=meta,
    )
    result = compare_prompt_documents_for_host(left, right)
    if result.identical or not result.changed_fields:
        print("compare must detect prompt content changes", file=sys.stderr)
        return 1

    print(f"OK: prompt compare API ({len(result.changed_fields)} diffs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
