# T-IGX-036-Z — PDC-X P6 batch B-P6-I: omnibus repair of the cheap remaining debt

## Scope

One omnibus ticket for the cheap debt that P6 left (Operator decision 5 of `DELEGATION-PDC-WAVE2.md`): the `pdc_x` pytest process that does not exit after its summary line (finding R-05), tests that are red on the baseline for environment or stale-fixture reasons, and the unused `embedding_provider/ollama` manifest. The `compat_x` baseline counters were reproduced and are recorded as documented debt D1 of the plan (re-pinning them needs a gate-meaning change and the `roadmap_replay_x` lineage), not repaired here. Test-side fixes and one deletion of an unused product file. Everything else stays documented debt (see Out of scope). Product code: yes, exactly one path (`intergrax/integrations/providers/embedding_provider/ollama/manifest.py`, deletion). Plan: `PLAN_T-IGX-036-Z.md`.

## Plan digest

Plan digest: da824e288bcaf6872292a02ee1d12c4fa1334da947609f23e6a7344a894ec3ba

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-Z.md`](plans/PLAN_T-IGX-036-Z.md).

## Approved

Approved: operator-delegate @ 2026-11-14T09:39:38+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE2.md@ee26c8e20fcd606482971beb1418dc67e39915db

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_debt tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py tests/qualification/pdc_x/reg_p tests/qualification/pdc_x/p6_ecapfix tests/unit/integrations/providers/test_p5_m6_p4_providers.py tests/unit/rag/embedding/test_hf_embedding_provider_execution.py tests/integration/runtime/nexus/tracing/test_sqlite_run_trace_store.py applications/local_workspace_application/tests/test_run_artifact_metadata.py tests/unit/governance/test_iiac_g0_inventory.py -q
timeout 1500 uv run pytest tests/qualification/pdc_x -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Celery, Kafka and RabbitMQ identity and dead-letter mapping, strict-host `plugin_provenance`, the `ent_audit_x` `launcher_05` FREEZE timeout (documented debt, no ticket).
- `dist_scale_x` (Docker, C10/C15), `roadmap_replay_x` dispositions, F10 certification (T-IGX-036-Y), IIAC protocol header and activation (PDC-P7-00).
- Hotspot paths (`intergrax/contracts/`, `intergrax/applications/_shared/`), any new authority, any gate whose meaning would change, FRZ changes, OD signatures and certification.

## Lifecycle state

plan_approved
