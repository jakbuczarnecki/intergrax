# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export — canonical: contracts.platform_plugin_selection."""

from __future__ import annotations

from intergrax.contracts.platform_plugin_selection import PlatformPluginSelectionRef
from intergrax.core.plugins.discovery import EntryPointSpec


def entry_point_spec_matches_selection_ref(
    spec: EntryPointSpec,
    ref: PlatformPluginSelectionRef,
) -> bool:
    if spec.group != ref.entry_point_group or spec.name != ref.entry_point_name:
        return False
    if spec.distribution is None:
        return False
    try:
        from intergrax.contracts.platform_plugin_selection import (
            normalize_distribution_package_name,
        )

        normalized = normalize_distribution_package_name(spec.distribution)
    except ValueError:
        return False
    return normalized == ref.distribution


def unresolved_entry_point_spec(ref: PlatformPluginSelectionRef) -> EntryPointSpec:
    return EntryPointSpec(
        name=ref.entry_point_name,
        group=ref.entry_point_group,
        value="<unresolved>",
        distribution=ref.distribution,
    )


setattr(
    PlatformPluginSelectionRef,
    "unresolved_entry_point_spec",
    lambda self: unresolved_entry_point_spec(self),
)

__all__ = [
    "PlatformPluginSelectionRef",
    "entry_point_spec_matches_selection_ref",
    "unresolved_entry_point_spec",
]
