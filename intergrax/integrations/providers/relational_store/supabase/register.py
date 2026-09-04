# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register supabase in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.supabase.bundle import create_supabase_relational_store
from intergrax.integrations.providers.relational_store.supabase.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.relational_store.supabase.contract_spec import CONTRACT_SPECS


def register_supabase_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_supabase_relational_store, override=override, contract_specs=CONTRACT_SPECS)
