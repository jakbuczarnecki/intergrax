# Agent harness adapter packs (Delivery DevOS)

Generated fragments for **non-Cursor** harnesses. **Enforcement: ADVISORY** — operators merge manually; Cursor SSOT remains [`role_registry.yaml`](../role_registry.yaml) + [`generate_control_plane_matrix.py`](../../../scripts/governance/generate_control_plane_matrix.py).

| Harness | Output | Regenerate |
|---------|--------|------------|
| Cursor | [`../MATRIX.md`](../MATRIX.md) | `uv run python -m scripts.governance.igx_gov sync-adapters` |
| Claude | [`claude/DELIVERY_DEVOS.fragment.md`](claude/DELIVERY_DEVOS.fragment.md) | same |
| Codex | [`codex/AGENTS.fragment.md`](codex/AGENTS.fragment.md) | same |

Do not treat adapter stubs as certification evidence — see [`EVIDENCE-AND-CERTIFICATION.md`](../../../docs/project/technical/devos/EVIDENCE-AND-CERTIFICATION.md).
