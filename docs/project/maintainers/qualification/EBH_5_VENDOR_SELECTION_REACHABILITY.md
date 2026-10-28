# EBH-5 — VENDOR_SELECTION_REACHABILITY_EVIDENCE

| Field | Value |
| --- | --- |
| **Evidence SHA** | Handoff @ [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) |
| **Tests** | [`test_ebh_5_reachability_gates.py`](../../../tests/qualification/ebh_5/test_ebh_5_reachability_gates.py) |

## Positive reachability

- Integration resolution: `resolve_from_profile` + `IntegrationProfile` ([`intergrax/integrations/registry/factory.py`](../../../intergrax/integrations/registry/factory.py)).
- ENV merge: `INTERGRAX_INTEGRATION_*` → `read_integration_slug_from_env` → profile merge (not parallel registry).
- Observability role backends: explicit `ToolWiringContext.observability_role_backends` / `observability_backend`.

## Negative reachability

- Unknown integration slug → resolution fails closed (`test_ebh5_reach_integration_unknown_slug_fail_closed`).
- Observability traces role without configured backend → `RuntimeError` (`observability_role_backend_not_configured`).
- RAG graph: no `graph_store_backend` shadow selector (when module present).

## CONFIG-X preservation

Historical **FRZ-CFG-01..08 PASS** @ `fac9a43f…` unchanged; `discover_active_blocker_path_keys()` empty (RPL-G gate).
