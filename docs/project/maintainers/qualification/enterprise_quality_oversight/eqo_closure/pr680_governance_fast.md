# PR #680 — governance-fast classification (W0-03)

**Run:** GitHub Actions workflow run `44690933116` (2026-11-20, head `af2/pt-fc1`).

**Check:** `governance-fast` — **FAILURE** (other PR checks mostly SUCCESS; enterprise gate jobs SKIPPED).

## Classification (plan §12, 11 reasons)

**Primary:** **2 — Regresja wprowadzona przez aktualne zmiany** (w kontekście oczekiwanego stanu IIAC przed RB-2).

Failures are dominated by IIAC activation and EVOL-X fresh-clone tests asserting **policy IN_FORCE** while **ACT-N11** reports stale audit after material governance / product-path changes on the PR branch, for example:

- `intergrax/llm_adapters/contracts/provider_error.py`
- `intergrax/runtime/human/persistence_errors.py`

Representative nodes:

- `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py` (multiple variants)
- `tests/unit/governance/test_iiac_activation_gate.py`
- `tests/unit/governance/test_iiac_negative_guards.py`

## Relation to KR-08

**KR-08** (known-red register on TRAIN_PIN): `ACT-N11: governance-fast reds after a product-path change (15 nodes)`.

This PR failure pattern is **consistent with KR-08** (governance-fast + ACT-N11 after product delta) but **not byte-identical** to the frozen KR register baseline (`7960ae7f`). Treat KR-08 as related cluster evidence, **not** an automatic waiver for #680.

## Secondary hypotheses (not primary)

- **8 — Stale gate expectation:** tests assume IIAC IN_FORCE on `development` while train remediation has not merged (FCS/RB-2 pending).
- **11 — Nieustalona:** full node list vs KR-08 inventory not reconciled in this pass.

## Evidence requirement

Re-run `governance-fast` at merge commit **after** RB-2 / IIAC activation (pasmo G) OR record exact blocker if red persists with IN_FORCE.

**W0-03 verdict:** regresja względem oczekiwanego stanu aktywacji IIAC na gałęzi PR; powiązanie z EQOA-A0-007 i X-03.
