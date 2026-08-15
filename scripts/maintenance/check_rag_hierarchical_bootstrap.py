#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-14.4 — dual-index + hierarchical retriever bootstrap wiring."""

from __future__ import annotations

import sys


def main() -> int:
    from intergrax.rag.bootstrap.hierarchical_bootstrap import (
        profile_uses_hierarchical_index,
        resolve_toc_vectorstore_for_profile,
    )
    from intergrax.rag.profiles.rag_profile import RagProfile

    profile = RagProfile(hierarchical_index_enabled=True, retriever_id="hierarchical")
    if not profile_uses_hierarchical_index(profile):
        print("hierarchical profile flag not recognized", file=sys.stderr)
        return 1
    toc_store = resolve_toc_vectorstore_for_profile(
        profile,
        tenant_id="audit-tenant",
    )
    if toc_store is None:
        print("toc vectorstore bootstrap returned None", file=sys.stderr)
        return 1
    print("OK: hierarchical dual-index bootstrap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
