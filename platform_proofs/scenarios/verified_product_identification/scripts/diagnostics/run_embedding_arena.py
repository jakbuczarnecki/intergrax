"""Deprecated import path — use scripts.embedding.diagnostics.run_embedding_arena."""

from platform_proofs.scenarios.verified_product_identification.scripts.embedding.diagnostics.run_embedding_arena import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.embedding.diagnostics.run_embedding_arena")
    raise SystemExit(_m.main())
