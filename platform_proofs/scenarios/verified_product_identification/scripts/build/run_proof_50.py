"""Deprecated import path — use scripts.proof.proof_tooling.run_proof_50."""

from platform_proofs.scenarios.verified_product_identification.scripts.proof.proof_tooling.run_proof_50 import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.proof.proof_tooling.run_proof_50")
    raise SystemExit(_m.main())
