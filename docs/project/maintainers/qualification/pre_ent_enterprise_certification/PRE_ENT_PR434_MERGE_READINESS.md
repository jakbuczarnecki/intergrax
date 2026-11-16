# PRE-ENT — PR #434 merge readiness (operator D-PREENT-01=b)

**PR:** [#434](https://github.com/jakbuczarnecki/intergrax/pull/434)  
**Head:** `cfd1d0fc62a5610aca501d640a4df0f2c4f0b048` (branch `pre-ent/formal-closure-checkpoint-034a`)  
**Base:** `development` @ `d799e49d5` (`0ad6511ac` ENT-AUDIT-X W4, per remote tip at poll)  
**AUDIT_HEAD (product pin):** `085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0`  
**Date:** 2026-11-10  
**Orchestrator:** T-IGX-034-A — **no merge**, **no Wave 3** until operator closes this gate.

**Status:** **MERGED** (operator 2026-11-10) — merge commit `03aa10959acf6ec3d81a7a0de8ac8224c8126973`. **Uwaga:** merge nastąpił przy **czerwonym** `governance-fast` / ACT-N11 (odchylenie od **D-PREENT-08=a**). Handoff IIAC nadal aktywny: [`PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md`](PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md).

---

## 1. CI (Regression + ENT audit)

**Evidence run (Regression gate):** `42793932297` @ head `cfd1d0fc6`.

| Check | Status @ poll | Notes |
| --- | --- | --- |
| `ci-smoke` | **PASS** | Fixed @ `cfd1d0fc6` (conftest `pytest_configure` hookspec) |
| `governance-policy-pin` | **PASS** | |
| `evol-x-b4-preflight` | **PASS** | |
| `governance-fast` | **FAIL** | 15 failed / 133 passed — DCP governance unit suite |
| `ent-audit-fast` | **PASS** | |
| `ent-audit-pr-gate` | **PASS** | |

**Regression gate:** **red** until `governance-fast` is green (required job on PR).

---

## 2. `governance-fast` root cause (job `129754569140`)

Failing step: **Governance unit suite (DCP)**.

Primary mechanism: **IIAC ACT-N11** — policy claims `IN_FORCE` but product paths differ from audited SHA (`verify_iiac_policy_activation --require-in-force`).

**Local reproduction @ `cfd1d0fc6`:**

```text
ACT-N11 … applications/shipped_product_manifest_catalog.py
ACT-N11 … intergrax/applications/_shared/product_manifest_registry.py
ACT-N11 … intergrax/cli/doctor_diff_app.py
ACT-N11 … intergrax/cli/doctor_health_app.py
ACT-N11 … intergrax/platform/enterprise_audit/baseline.py
ACT-N11 … intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json
ACT-N11 … intergrax/platform/enterprise_audit/data/schemas/enterprise_audit_report.v1.schema.json
```

**@ `origin/development` alone (pre-BND):** verify fails on **3 ENT paths** only (baseline + impact map + schema). Documented as known regression on `development` (`governance/tickets/T-IGX-024-B.md` W4-01 — separate IIAC re-activation, not ENT W4 merge blocker).

**PR increment:** **034-B BND-03B** adds **4** additional product paths — expected under DR-12=B / ACT-N11 sequencing (tickets 030-B, 032-A, 033-A, 034-B).

**Verdict:** failure is **not** a flaky test or doc typo; **not** fixable by weakening gates. A green `governance-fast` requires a **governed IIAC re-audit / re-activation batch** (ENT-AUDIT-X / MASTER stream), **outside** approved PRE-ENT scope given **D-PREENT-06=defer** and plan ban on editing material IIAC activation artifacts.

---

## 3. Sync and diff

| | |
| --- | --- |
| **Integracja** | Lokalna linia zawiera merge `origin/development` (`0ad6511ac`) @ `085be5ca9` |
| **PR delta** | ~44 paths — BND product/qual + PRE-ENT dossier + ticket checkpoints |
| **Fix commits on PR** | `cfd1d0fc6` — ci-smoke conftest hookspec only |
| **Push** | Branch on `origin`; ahead of `development` by full PR stack |

---

## 4. Merge readiness (honest)

| Gate | Ready? | Rationale |
| --- | --- | --- |
| **Operator diff review (D-PREENT-01=b)** | **Tak** | [`PRE_ENT_PR434_DIFF_REVIEW.md`](PRE_ENT_PR434_DIFF_REVIEW.md) |
| **ENT audit PR jobs** | **Tak** | `ent-audit-fast`, `ent-audit-pr-gate` green |
| **Smoke / policy pin / EVOL B4** | **Tak** | @ `cfd1d0fc6` |
| **Full Regression gate (GitHub)** | **Nie** | `governance-fast` ACT-N11 — blocker below |
| **Wave 3 audytorzy** | **Wstrzymane** | Per operator — po merge checkpoint |

---

## 5. Ryzyka po merge (jeśli operator merguje mimo czerwonego `governance-fast`)

1. **ACT-N11** na `development` pogłębia się o ścieżki 034-B (dopiero re-activation je legalizuje).
2. **AUDIT_HEAD** `085be5ca9` może wymagać aktualizacji w handoff po nowym merge SHA produktowym.
3. **F10 / OD-HIST04 / G-SKILL** — bez zmian (defer/block z §7).
4. **Nie** uzyskuje się zielonego CI na `development` do czasu IIAC batch (już znane na tipie zdalnym).

---

## 6. Authorization STOP — `governance-fast` green

**Orchestrator STOP** — brak minimalnej poprawki w zatwierdzonym planie PRE-ENT.

**Wymagana decyzja operatora (szkic — nowa linia, nie podpisana):**

| ID | Pytanie | Opcje |
| --- | --- | --- |
| **D-PREENT-08** | Jak traktować Regression gate przy ACT-N11? | **(a)** Wstrzymaj merge do zielonego `governance-fast` po **IIAC re-activation** (ENT/MASTER batch) · **(b)** Merge PR #434 z udokumentowanym ACT-N11 (analogia ENT W4-01 znany regression) · **(c)** Osobna linia na PRE-ENT-only IIAC re-activation (nowy batch, poza defer OD-HIST04-A/C) |

Bez **D-PREENT-08**: **nie** deklaruję „gotowe do merge pod pełny zielony CI”; możliwy jest wyłącznie merge świadomy ryzyka **(b)** po Twojej stronie.

---

## 7. Po Twoim merge (gdy autoryzujesz)

1. Zaktualizuj `PRE_ENT_CAMPAIGN_PROGRESS.md` / `AUDIT_HEAD` jeśli tip `development` się przesunie.
2. Dopiero wtedy Wave 3 — pięć briefów @ zaktualizowany SHA.
3. **Nie** startować certyfikacji ani macierzy końcowej.

---

## SSOT

- Diff review: [`PRE_ENT_PR434_DIFF_REVIEW.md`](PRE_ENT_PR434_DIFF_REVIEW.md)  
- Checkpoint §7: [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md)  
- ENT known regression: `governance/tickets/T-IGX-024-B.md` (W4-01)  
- IIAC batch: `docs/project/maintainers/qualification/historical_certification_campaign/HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`
