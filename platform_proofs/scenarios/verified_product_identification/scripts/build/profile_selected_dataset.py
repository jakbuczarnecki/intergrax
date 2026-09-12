"""Deprecated import path — use scripts.dataset.diagnostics.profile_selected_dataset."""

from platform_proofs.scenarios.verified_product_identification.scripts.dataset.diagnostics.profile_selected_dataset import *  # noqa: F403

if __name__ == "__main__":
    from importlib import import_module
    _m = import_module("platform_proofs.scenarios.verified_product_identification.scripts.dataset.diagnostics.profile_selected_dataset")
    raise SystemExit(_m.main())
