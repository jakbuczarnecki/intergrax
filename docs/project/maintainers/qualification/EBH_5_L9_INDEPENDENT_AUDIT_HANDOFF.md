# EBH-5 — L9 Independent Audit Handoff (Protocol 03)

| Field | Value |
| --- | --- |
| **Parent** | EBH-5 — Replaceability & Vendor/Backend Certification |
| **Audit mode** | ECAP §2.B Parent Closure Audit |
| **CERTIFICATION_HANDOFF_SHA** | Publication commit documented in [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) |
| **Wave containment** | `git merge-base --is-ancestor 83a376b3e1b312e2e4f88da548c99c328f3589f7 HEAD` must succeed |

## Historical SHA (not audit entry)

| Label | SHA |
| --- | --- |
| Early dossier baseline | `5c9b66db632976ada039d406474530d7c6cb5f86` |
| Wave AUDIT_BASELINE | `83a376b3e1b312e2e4f88da548c99c328f3589f7` |

## Commands @ CERTIFICATION_HANDOFF_SHA

```powershell
git checkout development
git rev-parse HEAD
uv run pytest tests/qualification/ebh_5/ -q
uv run pytest tests/qualification/config_x/test_config_x_final_reconciliation_gates.py -q
```

## Mechanical exit §3.0.8 (verify)

| Criterion | Expected |
| --- | --- |
| Inventory complete, unclassified=0 | 69 concerns |
| Matrix extensibility promised | 56 rows |
| Reachability | gates green (1 skip: graph module optional) |
| Behavioral proof per QUALIFIED | 5 resolvable pytest nodes |
| Fixed substrates + SQLite annex | 38 touchpoints closed-world |
| RPL-G | green |
| Independent audit | **this review** |

## Five QUALIFIED — proof nodes

| concern_id | Behavioral proof (pytest node suffix) |
| --- | --- |
| `persistence.relational_sqlite_lab` | `test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile` |
| `int_config.configure_existing` | `test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service` |
| `observability.otlp_transport` | `test_ebh5_behavioral_otlp_transport_via_sanctioned_composition_wiring` |
| `tokenizer.selection` | `test_ebh5_behavioral_tokenizer_selection_via_bootstrap_composition` |
| `integration.key_value_cache` | `test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition` |

Registry SSOT: [`_behavioral_proof_registry.py`](../../../tests/qualification/ebh_5/_behavioral_proof_registry.py).

## Limitations (explicit)

- **51** matrix rows **NOT YET QUALIFIED** — deferred to EBH-7 scope; not false PASS.
- **RPL-F** **NA** on all five QUALIFIED rows (no mandatory failure/recovery L3 in EBH-5).
- **OTLP asymmetry:** empty endpoint + `ExporterKind.OTLP` disables transport without `ConfigurationError`; DISTRIBUTED_OTLP misconfig raises — see [`EBH_5_PROTOCOL03_FINAL_CORRECTION.md`](EBH_5_PROTOCOL03_FINAL_CORRECTION.md).
- **Reachability:** `integration.graph` module absent → 1 skipped test.
- **CONFIG-X** PASS @ `fac9a43f…` preserved; current-HEAD obligations additive only.
- **DIST-SCALE-X** remains operator scoped acceptance — do not rewrite as independent CLOSED.

## Findings disposition (auditor fills)

| Finding ID | Severity | Disposition | Follow-up |
| --- | --- | --- | --- |
| _empty_ | | ACCEPT / REJECT / DEFER | |

## Forbidden verdicts

- Global **FRZ PASS**
- EBH-5 **CLOSED — INDEPENDENTLY ACCEPTED** without operator OD-EBH5-08
- Implementation agent self-issuing **AUDIT PASS**

## Deliverable

Independent **AUDIT PASS** or **REJECT with findings** recorded by **operator** in roadmap §5 — not by implementation agent.
