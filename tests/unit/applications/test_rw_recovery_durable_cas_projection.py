# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-E NT-RW-REC-05 (EX-D1-07, F-D1-11, G-08): control leg of the durable compare.

Only the control of the row is pinned here: a bundle identical to the durable authority registers and a
different integration slug is refused (both green before this ticket). The red legs of NT-RW-REC-05
(``revision_state`` / ``activated_at`` / a binding beyond its slug are refused) are held: comparing the
revision lifecycle turns the pinned flow
``tests/qualification/pdc_x/d1_repair/test_d1_repair_durable_projection_inputs.py::
test_second_composition_can_reregister_the_inputs_of_a_persisted_revision`` red (inputs built while
VALIDATED, durable authority ACTIVE), and the binding reduction to slug references happens in the
manifest JSON decode outside this ticket's write set. Plan section 6 STOP rule; question Q-RC-02.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from intergrax.agent_distribution.sqlite_stores import SqliteAgentDistributionStoreBundle
from intergrax.applications._shared.durable_agent_platform_runtime import DurableAgentPlatformRuntime
from intergrax.applications._shared.production_lifecycle import (
    build_descriptor_backed_projection_input_store,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    ReferenceProductionControlPlaneGovernance,
)
from intergrax.applications._shared.registry_projection import (
    RegistryProjectionError,
    RegistryProjectionInputBundle,
)
from tests.unit.applications.test_prod_q_r1_r7_d3_production_restart_reconciliation import (
    _durable_launcher,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _activate_bundle,
    _canonical_bundle,
)

pytestmark = [pytest.mark.unit]

_MISMATCH = "durable descriptor authority mismatch"


def _durable_authority(tmp_path: Path) -> tuple[SqliteAgentDistributionStoreBundle, RegistryProjectionInputBundle]:
    launcher, durable, governance = _durable_launcher(tmp_path, tmp_path / "rw-rec-05.db")
    assert isinstance(durable, DurableAgentPlatformRuntime)
    assert isinstance(governance, ReferenceProductionControlPlaneGovernance)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        revision_id="rev-rw-rec-05",
        stores=launcher.process_composition.agent_platform_runtime.stores,
    )
    revision_id = _activate_bundle(launcher, bundle, locator, principal=governance.principal)
    stores = durable.distribution_store_bundle
    reconstructed = build_descriptor_backed_projection_input_store(stores).get(revision_id)
    assert reconstructed is not None
    return stores, reconstructed


def _register(stores: SqliteAgentDistributionStoreBundle, bundle: RegistryProjectionInputBundle) -> None:
    # A fresh store per registration: the in-process fingerprint cache must not decide the outcome.
    build_descriptor_backed_projection_input_store(stores).register(bundle)


def test_rw_rec_05c_different_slug_is_still_refused(tmp_path: Path) -> None:
    stores, durable = _durable_authority(tmp_path)
    _register(stores, durable)  # identical to the durable authority: accepted

    profile = durable.manifest.integration_profile
    changed_profile = profile.model_copy(update={"async_task_index_slug": "rw-rec-05-different-slug"})
    assert changed_profile != profile
    changed = replace(durable, manifest=durable.manifest.model_copy(update={"integration_profile": changed_profile}))

    with pytest.raises(RegistryProjectionError, match=_MISMATCH):
        _register(stores, changed)
