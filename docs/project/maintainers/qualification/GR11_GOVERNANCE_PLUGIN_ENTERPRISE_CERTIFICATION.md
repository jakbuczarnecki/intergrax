# GR-11 — Governance Plugin Enterprise Certification

**Status:** `READY FOR AUDIT`  
**Parent:** GOV-X1 (CURRENT)  
**Mechanical SSOT:** `tests/qualification/governance/gr11/catalog.py`  
**Mechanical gate:** `tests/qualification/governance/gr11/test_gr11_plugin_enterprise_certification.py`

GR-11 certifies **Governance-relevant extensibility** only (not whole-platform plugin certification — EBH-5 / CONFIG-X / COMPAT-X / PROD-Q).

## Baseline

| Field | Value |
| ----- | ----- |
| Qualification bundle commit | set at wave commit on `development` |
| GR-12 closure evidence | `03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06` |
| GR-12 semantic baseline | `b706c2c72a900575ec360b7f217c97a3656c71b9` |

## Closed-world inventory

See `GR11_EXTENSION_SURFACES` in catalog SSOT (9 rows: root admission, inner guard, decision requirement, runtime policy evaluator, CLA-04 evaluator, continuation, provider invocation store, provider/reliability collaborators, reliability evidence observer).

## Historical reconciliation (GOV-FINAL-4 pluginability)

| Capability | Before | After (current HEAD) |
| ---------- | ------ | -------------------- |
| Continuation | PARTIAL | QUALIFIED |
| Provider integration | PARTIAL | QUALIFIED (typed contracts) |
| ProviderInvocationStore | PARTIAL | QUALIFIED |
| Reliability observer | PARTIAL | QUALIFIED |

## Authority matrix (summary)

| Extension class | May decide | Must not decide |
| --------------- | ---------- | --------------- |
| Governance permission plugins | ALLOW / DENY / REQUIRE_HUMAN / ESCALATE within contract | Execution lifecycle, provider mutation without governance, scope self-expansion |
| Decision requirement | Material requirement for MSE | Permission, execution |
| Execution continuation | Pause/resume lifecycle facts | Governance permission |
| ProviderInvocationStore | Durable invocation facts | Permission |
| Reliability observer | Emit reliability facts | Permission, execution truth |

## Discovery / selection / activation

Governance extension mechanisms are **composition-time** only (`COMPOSITION_TIME_ONLY`); sanctioned application/runtime/reliability composition owners activate implementations. No Governance self-activation or hot global registry.

## FRZ scoped evidence (global criteria remain OPEN)

Contributes scoped evidence toward FRZ-PLG-01..08, FRZ-RPL-01..04, FRZ-GOV-01..04/06/09, FRZ-CTR/TYP/REG where mechanically tested in GR-11 gate.

## Recommended status

`GR-11 = READY FOR AUDIT` · `GR-13 = NEXT` after independent GR-11 audit · `GOV-X1 = CURRENT`

**Not claimed:** GR-11 CLOSED · Governance Plane enterprise CLOSED · GOV-X1 CLOSED
