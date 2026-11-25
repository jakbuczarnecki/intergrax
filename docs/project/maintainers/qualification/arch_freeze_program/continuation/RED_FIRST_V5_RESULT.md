# RED_FIRST V5 evidence index

- integration_head: `5935fa4f0cd3596852b3a6500f9d4e1d613a53df`
- Session logs: `.tmp/session/arch-freeze/red-first-v5/logs/`

## V5 remeasured (single-node)

| FRZ | prefix | exit | status | log |
|-----|--------|------|--------|-----|
| `FRZ-TEN-02` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | `.tmp/session/arch-freeze/red-first-v5/logs/FRZ-TEN-02.log` |

## V5 rerun still UNREPRODUCIBLE (corrected paths)

- `FRZ-CFG-02`, `FRZ-CFG-08`, `FRZ-OWN-02`: ENT-A node paths fixed (`test_ent_a_ent_a_*` not doubled); prefix exit 0 (semantic gate green at `754ee000`).
- `FRZ-PLG-01`: single node `test_validate_evaluation_wiring_requires_registry_when_online_enabled`; collection/plugin at prefix `22993254`.

Prior partial driver rows remain in `RED_FIRST_V5_RESULT.json` for audit; plan rows updated only where remeasured.