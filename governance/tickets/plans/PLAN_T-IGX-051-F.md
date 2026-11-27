# PLAN T-IGX-051-F — T-AF2-RW-GATES

**Evidence pin:** `fc2ccb73dea161156d067019548f88e08f033fd9`. **Presence audit:** `.tmp/session/arch-freeze/execution/AF-20C/gate_presence.md`.

**MEP allocation (quote):**

| Field | Value |
|---|---|
| Node | T-AF2-RW-GATES |
| Wave | W3 |
| Predecessors | SCOPE, PT-WA |
| Ticket | T-IGX-051-F |
| Sensitivity | PROD + NONPRODUCT |
| Scope text | `scripts/gates`, scenario contract, event payloads, eval digests, NOT_COMPOSED gates; EX-D5-03, EX-L-01/11/13, EX-D3-04, D1-R015, D5-R002, D6-R015 |

**Acceptance (MEP § remediation matrix, row T-AF2-RW-GATES):** gate bound to stage plus ticket; one canonical signature; schema reconciled; digest enforced; non-composition gates — *an unrelated ticket gets DENIED; composing a NOT_COMPOSED feature raises*.

**Owner role (Delivery DevOS):** `implementer` (`governance/agent-control-plane/role_registry.yaml`).

## 1. Gap inventory @ pin (cited, no re-audit)

| Capability | State @ pin | Source |
|---|---|---|
| Stage + ticket promotion gating | PRESENT | `gate_presence.md` §1 |
| Canonical scenario `get_steps` | ABSENT | `gate_presence.md` §2 |
| `RuntimeEvent` schema | PRESENT | `gate_presence.md` §3 |
| Eval digest | PRESENT | `gate_presence.md` §4 |
| NOT_COMPOSED executable raise path | ABSENT (pass-through only) | `gate_presence.md` §5 |

## 2. Design rules

1. Gates fail closed; no silent pass-through for NOT_COMPOSED features covered by this node.
2. Promotion authorization remains bound to governance ticket `Approved:` line (existing MVP promotion gate pattern).
3. Any edit under `intergrax/contracts/**` is minimized; prefer gate scripts and runtime wiring. Contract edits trigger OD-AF-11 wave-packet check on merge.
4. No claim of FRZ closure, stage certification, or G6 PASS in this plan.

## 3. Write set (exact paths — firm unless marked conditional)

Paths align with MEP T-AF2-RW-GATES scope string and gap inventory. Conditional rows activate only if the implementer discovers an unavoidable contract edit (amendment, not silent scope creep).

### 3a. Gate scripts and maintenance (firm)

| Path | Intent |
|---|---|
| `scripts/gates/check_mvp_promotion_gates.py` | Harden stage-plus-ticket binding; unrelated ticket → DENIED |
| `scripts/gates/check_agent_promotion_eval_gate.py` | Keep promotion eval path; extend assertions per plan step |
| `scripts/gates/check_not_composed_reachability.py` | **New** — NOT_COMPOSED composition → raise / gate FAIL |
| `scripts/gates/check_eval_digest_enforcement.py` | **New or extend** — corrupted digest/signature → DENIED |

### 3b. Contracts and runtime surfaces (firm / conditional)

| Path | Intent | Firm? |
|---|---|---|
| `intergrax/contracts/evaluation_contracts.py` | Canonical eval scenario `get_steps(context)` signature (EX-D3-04) | firm |
| `intergrax/contracts/runtime_event.py` | Schema version / payload reconciliation with consumers | firm |
| `intergrax/runtime/events/payloads/canonical.py` | Payload canonicalization per EX-L-13 / event drift tests | firm |
| `intergrax/runtime/observability/qualification_runtime_trace.py` | Align trace emission with reconciled schema | firm |
| `intergrax/applications/_shared/deploy_activate_pipeline.py` | Replace NOT_COMPOSED pass-through with fail-closed path where gate requires | conditional — only if reachability test cannot be satisfied gate-only |

### 3c. Tests (firm, new unless noted)

| Path | Marker / name |
|---|---|
| `tests/unit/architecture/test_rw_gates_promotion_unrelated_ticket_denied.py` | `NT-RW-GATES-PROMO-DENY` |
| `tests/unit/architecture/test_rw_gates_eval_digest_denied.py` | `NT-RW-GATES-DIGEST-DENY` |
| `tests/unit/architecture/test_rw_gates_not_composed_raises.py` | `NT-RW-GATES-NC-RAISE` |
| `tests/unit/architecture/test_rw_gates_partial_inventory.py` | `NT-DEBT-04-1` (PARTIAL register inventory; not eval-digest guard) |
| Existing tests under `tests/unit/runtime/observability/test_qualification_runtime_observability_o*.py` | regression only (edited only if schema reconciliation requires) |

### 3d. Explicitly out of write set

`pyproject.toml`, `uv.lock`, bulk `intergrax/contracts/**` edits unrelated to the three gates capabilities, `.github/`, docs trees, agents/applications product refactors outside paths above.

## 4. Negative tests (required)

| Id | Behavior | Expected |
|---|---|---|
| `NT-RW-GATES-PROMO-DENY` | Promotion evidence cites ticket without valid `Approved:` or wrong stage pairing | Gate / evaluator returns **DENIED**; no promotion |
| `NT-RW-GATES-DIGEST-DENY` | Eval run with corrupted `inputs_digest`, `result_digest`, or comparison digest | `EvalDigestMismatchError` or gate **DENIED** |
| `NT-RW-GATES-NC-RAISE` | Exercise composition path for capability marked NOT_COMPOSED (e.g. trust admission / NC fixtures named in final plan) | **Raises** or gate failure; must not pass-through silently (`gate_presence.md` §5 baseline) |

## 5. Overlaps (record only)

- Semantic overlap with `T-IGX-051-N` (RES-AB-3 NOT_COMPOSED, `production_validation.py` glob) — merge order per RW-PLAN: either order, second rebases (`PLAN_T-IGX-051-N.md` §9).
- Lane 4 serialization: after PT-WA / PT-WB; parallel with PT-FC3, PT-WE per MEP W3 table.

## 6. Verification commands (plan contract; not run in prep pack)

Listed for the tracked ticket **Test contract** section — e.g. single `uv run --frozen pytest` bundling the three `test_rw_gates_*` files plus regression files in §3c; gate scripts invoked with exit code 0 in CI profile.

## Plan digest

PENDING_COMPUTE