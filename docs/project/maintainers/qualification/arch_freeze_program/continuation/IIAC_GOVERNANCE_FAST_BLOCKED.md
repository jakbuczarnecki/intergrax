# ARCH-FREEZE — `governance-fast` BLOCKED (tor `development`)

**Recorded:** 2026-11-21  
**Worktree:** `af-dev-merge`  
**Pin:** `origin/development` @ `2100a85f93665f66afe518a50ad15c1549cd95ca` (post DF5 `ci-smoke` fix `Ref: T-IGX-043-J`)

## Status

| Check | On HEAD pin | Disposition |
|-------|-------------|-------------|
| `governance-fast` | **FAIL** (expected) | **BLOCKED** — poza autonomią ARCH-FREEZE |
| `ci-smoke` | see [latest run](#ci-smoke-head) | remediated w `2100a85f9` (osobny tor) |

## Root cause

| Element | Detail |
|---------|--------|
| Guard | **ACT-N11** — `verify_iiac_policy_activation` / `tests/unit/governance/test_iiac_negative_guards.py` |
| Audited SHA | `corrective_implementation_sha` = `969cbd964c679d563b02e55f065b452d46fdf90b` (`IIAC_POLICY_ACTIVATION_RECORD.json`; PDC-X P8-04 bind) |
| Mechanism | Dziesiątki ścieżek `intergrax/` zmienionych po `969cbd964`; pin aktywacji jest **stale** względem drzewa na `development`. |
| Remediation | **RB-2 re-pin** IIAC (sesja certifier + activation delegate; pasmo G w MEP-AF2) — **nie** mały hotfix testowy ani docs-only. |

Klasa znana: **KR-08** ([`KNOWN_RED_REGISTER.md`](../KNOWN_RED_REGISTER.md)); train/EQO kontekst: [`pr680_governance_fast.md`](../../enterprise_quality_oversight/eqo_closure/pr680_governance_fast.md), [`IIAC_RB2_BLOCK_READ.md`](../../enterprise_quality_oversight/eqo_closure/x/A0-007/IIAC_RB2_BLOCK_READ.md).

Operator boundary: **OD-AF-AUTO-01** — governance-fast na train wymaga zielonego checku lub wąskiego ACT-N11 na product paths; **docs nie bypassują** RB-2 na torze `development` przy czerwonym `governance-fast`.

## CI investigator findings

Pełna diagnoza runów `44880866642` / `44878706290` (ci-smoke vs governance-fast): [Diagnoza CI `development` (ci-investigator)](9dcc642c-21eb-48d1-a9d4-b7604bbf7fa1).

Skrót z sesji:

- **ci-smoke:** 7× `test_diag_foundation_5_destructive_proof.py` (G-09 `tenant_id`, FC-3 `vector_store` / embedding) — **FIX** @ `2100a85f9`.
- **governance-fast:** ACT-N11 @ `969cbd964` — **BLOCKED** do RB-2 re-pin; brak fixa w autonomii ARCH-FREEZE.

## ci-smoke @ HEAD

Po pushu tego pliku orchestrator sprawdza job `ci-smoke` w workflow **Regression gate** na `origin/development` HEAD (`gh run list --branch development --workflow unit-tests.yml`).

**Ref:** dokumentacja kontynuacji toru `development`; eskalacja RB-2 nie jest autoryzowana w tej sesji.