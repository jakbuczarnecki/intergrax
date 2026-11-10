# PRE-ENT — PR 434 diff review (D-PREENT-01=b)

**PR:** [#434](https://github.com/jakbuczarnecki/intergrax/pull/434) — *PRE-ENT: formal closure checkpoint + origin sync (T-IGX-034-A)*  
**Base → head:** `development` ← `pre-ent/formal-closure-checkpoint-034a`  
**Reviewer:** T-IGX-034-A orchestrator (read-only; **no merge authority**)  
**Date:** 2026-11-10  
**AUDIT_HEAD (product):** `085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0`

**Operator:** D-PREENT-01=**b** — review diff first; merge when green after operator/CI, not before.

---

## Scope summary

| Area | Δ (approx.) | Risk | Notes |
| --- | ---: | --- | --- |
| **Product — BND-03B** | 1 catalog + registry refactor + gate test | **Medium** (boundary) | Removes Tier-0 `importlib` of Tier-3 manifests; adds `applications/shipped_product_manifest_catalog.py` + unit gate. Tier boundary change — appropriate for 034-B. |
| **Qual — replay L3** | `_pre_ent_l3_prod_delta.py`, disposition helpers | **Low** | SSOT for mechanical exit gates; no FRZ promotion in diff. |
| **Docs / governance** | PRE-ENT pack, tickets 034-A/B/E/F, predecessor checkpoints | **Low** | No PASS claims; formal checkpoint §7 unsigned until this batch. |
| **Merge commit** | `085be5ca9` integrates `origin/development` `0ad6511ac` (ENT W4 #433) | **Low** | `ort` merge; ENT adds `ent_audit_x` unit test only — orthogonal to PRE-ENT paths. |
| **Excluded from PR** | `CANDIDATE_ENTERPRISE_AUDIT_BASELINE.json` (untracked) | — | Per checkpoint §1 — stays off push. |

**Commits on PR (17):** product `01b3e1961` (034-B); qual replay `da8ab2061`; docs Wave 0–2–3; F10 pinned replay records; merge `085be5ca9`; checkpoint docs through progress HEAD.

**Files touched (41):** see `gh pr view 434 --json files`.

---

## Conflict / policy checks

- **Tier boundaries:** BND-03B moves manifest registration toward Tier-3 catalog — aligns with 033-A static CLI scope; doctor CLI still uses handler-scope bootstrap (documented residual in Wave 3 queue).
- **No F10 allowlist weakening** in diff — F10 remains doc/classification only; operator **D-PREENT-04=block** keeps TES ladder frozen.
- **No audit PASS / cert language** introduced in product code.
- **Ticket Ref trailers** present on product commits (034-B, replay qual).

---

## CI / merge recommendation

**At last orchestrator poll (2026-11-10):**

| Check | Status |
| --- | --- |
| `governance-policy-pin` | SUCCESS |
| `ent-audit-pr-gate` | SUCCESS |
| `ci-smoke` | **FAILURE** (Regression gate) |
| `governance-fast`, `ent-audit-fast`, `evol-x-b4-preflight` | IN_PROGRESS |

**Recommendation:** **Do not merge** until **Regression gate** is green on PR head (and any required fast tiers complete). After green:

1. Operator or delegate merges PR #434 to `development` (orchestrator **must not** merge per PRE-ENT constraints).
2. Re-pin Wave 3 `AUDIT_HEAD` if product SHA moves post-merge (docs-only follow-ups may keep `085be5ca9` as product replay pin per handoff queue policy).
3. Schedule five independent auditor chats per D-PREENT-05 (`PRE_ENT_WAVE3_AUDITOR_SESSION_BRIEFS.md`).

**Explicit:** This review is **not** G6 PASS, **not** certifier acceptance, **not** authorization to merge without green CI.

### CI follow-up — `ci-smoke` root cause (2026-11-10)

Run `42793468416` / job `129752532848`: `PluginValidationError` on `tests/unit/applications/conftest.py` — `pytest_configure(_config: …)` uses hook arg name `_config`, which does not match pytest hookspec (`config`). Introduced with **034-B** manifest registration in that conftest.

**Remedy on PR branch:** rename parameter to `config` (no behavior change). Re-run PR checks after push.

---

## Minimal re-verification (post-merge suggestion)

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py -q
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
```

---

## SSOT

- Checkpoint: [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md)  
- Ticket: `governance/tickets/T-IGX-034-A.md`  
- Operator line: D-PREENT-01=**b**
