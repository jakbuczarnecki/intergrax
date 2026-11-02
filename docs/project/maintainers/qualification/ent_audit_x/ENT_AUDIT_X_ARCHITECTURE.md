# ENT-AUDIT-X — architecture (orchestration layer)

**Scope:** Tier-0 `intergrax/platform/enterprise_audit/` — closed-world manifest, read-only sandbox, report/drift, CLI. Does **not** duplicate governance, execution authority, or semantic registry.

**Entrypoints:** `scripts/enterprise_audit/run_enterprise_audit.py`; CI [`.github/workflows/enterprise-audit.yml`](../../../../.github/workflows/enterprise-audit.yml).

**Modes:** FAST (executable PG subset), FULL/FREEZE (includes pending/partial markers as findings), CHANGED-SINCE (EVOL-mapped check subset until EVOL B6 selector is wired).

**Evidence:** `.tmp/proof/ent-audit-x/<run-id>/`; baseline `tests/qualification/ent_audit_x/ENTERPRISE_AUDIT_BASELINE.json` (operator-promoted).

**Stage status @ implementation:** product runner EXISTS; FRZ-AUD-01..10 and Protocol 03 B6 remain **OPEN** until independent acceptance.
