# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Part D: a strict root refuses unattested security-defense plugins (NT-SEC-06-1, FRZ-SEC-06).

NT-SEC-06-1b (a strict root built through ``environment_wiring.py``) needs the caller edit G-05,
which is outside the write set of this ticket; it is held out of the merge set.
"""

from __future__ import annotations

import dataclasses

import pytest
from intergrax.core.plugins.discovery import EP_SECURITY_DEFENSES, EntryPointSpec
from intergrax.core.security.security_configuration import (
    PLUGIN_TRUST_POLICY_MISSING,
    SecurityConfigurationError,
)
from intergrax.runtime.security import defense_plugin_loader

pytestmark = [pytest.mark.unit]


def test_sec_06_1_strict_root_refuses_unattested_defense_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = EntryPointSpec(
        name="unattested-defense",
        group=EP_SECURITY_DEFENSES,
        value="unattested_defense:Plugin",
        distribution="unattested-defense",
    )
    loaded: list[str] = []
    monkeypatch.setattr(defense_plugin_loader, "iter_entry_point_specs", lambda _group: (spec,))

    def _load(group: str, **_kwargs: object) -> list[object]:
        loaded.append(group)
        return []

    monkeypatch.setattr(defense_plugin_loader, "load_entry_point_targets", _load)
    policy_type = defense_plugin_loader.SecurityDefenseAdmissionPolicy
    assert "require_provenance" in {f.name for f in dataclasses.fields(policy_type)}, (
        "a strict root cannot require provenance for security-defense plugins (FRZ-SEC-06)"
    )
    policy = dataclasses.replace(policy_type(), require_provenance=True)

    with pytest.raises(SecurityConfigurationError) as excinfo:
        defense_plugin_loader.load_security_defense_plugin_report(discover_entry_points=True, admission=policy)
    assert excinfo.value.reason_code == PLUGIN_TRUST_POLICY_MISSING
    assert loaded == []

    # Lab default and discovery off stay unchanged.
    defense_plugin_loader.load_security_defense_plugin_report(discover_entry_points=False, admission=policy)
    defense_plugin_loader.load_security_defense_plugin_report(discover_entry_points=True, admission=policy_type())
    assert loaded == [EP_SECURITY_DEFENSES]
