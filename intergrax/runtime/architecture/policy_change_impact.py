# © Artur Czarnecki. All rights reserved.

"""Policy change impact visualization helpers (AUDIT-IDEAL-20.2)."""

from __future__ import annotations

from intergrax.runtime.architecture.capability_graph_lineage import CapabilityImpactReport


def render_policy_change_impact_visualization(
    report: CapabilityImpactReport,
    *,
    top_n: int = 12,
) -> str:
    """Render ASCII blast-radius summary for policy / capability change review."""
    ranked = sorted(
        report.impacts,
        key=lambda item: len(item.blast_radius_node_ids),
        reverse=True,
    )[:top_n]
    lines = ["Policy change impact (blast radius):", ""]
    for item in ranked:
        radius = len(item.blast_radius_node_ids)
        bar = "#" * min(radius, 40)
        lines.append(f"  {item.node_id:<40} {radius:>4}  {bar}")
        if item.blast_radius_node_ids:
            preview = ", ".join(item.blast_radius_node_ids[:5])
            suffix = " ..." if len(item.blast_radius_node_ids) > 5 else ""
            lines.append(f"    -> {preview}{suffix}")
    return "\n".join(lines)
