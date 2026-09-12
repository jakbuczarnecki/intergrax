"""Deprecated import path — use scripts.storage.operator_lifecycle.bootstrap."""

from platform_proofs.scenarios.verified_product_identification.scripts.storage.operator_lifecycle.bootstrap import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.storage.operator_lifecycle.bootstrap")
    raise SystemExit(_m.main())
