"""Deprecated import path — use scripts.storage.storage_tooling.run_vector_db_roundtrip_qualification."""

from platform_proofs.scenarios.verified_product_identification.scripts.storage.storage_tooling.run_vector_db_roundtrip_qualification import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.storage.storage_tooling.run_vector_db_roundtrip_qualification")
    raise SystemExit(_m.main())
