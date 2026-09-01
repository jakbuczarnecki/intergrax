# POLICY_GOVERNANCE - Platform Audit

## Metadata

- **campaign_id:** `2026-08-18`
- **campaign_started_at:** `2026-08-18`
- **Layer code:** POLICY_GOVERNANCE
- **Tier(s):** cross-domain Tier-0 policy contracts · Tier-1 runtime policy / meaningful-side-effect evaluation · Tier-2/3 product adapters
- **layer_audited_at:** 2026-08-19
- **audited_sha:** `042cc9b50386cfcd4da30310c84d000dbf5d2718`
- **Status:** COMPLETE
- **Auditor:** independent ChatGPT platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 5 ACCEPTED 2026-08-19
- **Architecture doc(s):**
  - `docs/project/architecture/GOVERNED_EXECUTION.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/GOVERNED_EXECUTION.md`
- **Scope in:**
  - meaningful-side-effect authorization spine ownership
  - policy resolution precedence and matching semantics
  - scoped human-approval grant consumption
  - provider/backend abstraction posture on policy paths
- **Scope out:**
  - remediation implementation
  - re-interpretation of findings from later G5C commits on current HEAD
  - full security audit of all policy surfaces
- **Prior audit reference(s):** [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md) (governed execution boundary themes); [`IDENTITY_TRUST`](IDENTITY_TRUST.md) (authority/provenance spine)
- **architecture_sync:** COMPLETE after Commit A
- **plan_sync:** COMPLETE after Commit A
- **post_sync_sha:** `d7988045cfa550c4338eedc326b54933c4058541`

## Executive summary

**Verdict: FAIL.** Five accepted findings (4 HIGH, 1 MEDIUM) show duplicate meaningful-side-effect governance ownership, unsafe first-match policy resolution, action-only matching despite rich requests, non-consumption of canonical scoped approval grants by inspected External Work paths, and hidden `rule_id` suffix dispatch. Positive controls preserved: `MeaningfulSideEffectAuthorizationBoundary` as described canonical boundary, fail-closed External Work when policy is missing, ToolRuntime policy-before-handler posture, and provider-neutral policy contracts. No new independent vendor/backend leakage finding in this layer.

## Verdict

**FAIL** - 0 CRITICAL / 4 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-POLICY_GOVERNANCE-01

**Meaningful side effects have two parallel governance spines**

- **Severity:** HIGH
- **Category:** ARCHITECTURE DEFECT
- **Related classification:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Remediation block:** PG-FIX-A
- **Claim falsified:** Production meaningful-side-effect authorization has one canonical ownership path with identical authority semantics across all consumers.
- **Observation:** Platform has `MeaningfulSideEffectAuthorizationBoundary`, described as the canonical/shared production pre-side-effect authorization boundary; it composes `CollaborativeWorkEnforcementGate`. Reference `ExternalWorkAdapter` does not use that canonical boundary; it invokes its own injected `MeaningfulSideEffectEvaluator`. Governed-contractor wiring has a separate `meaningful_side_effect_policy` DI path rather than deriving from the normal canonical policy bundle. Local External Work path is fail-closed when policy is missing - defect is duplicate ownership and inconsistent authority semantics, not absence of all policy.
- **Location:**
  - `intergrax/runtime/policy/meaningful_side_effect_authorization.py:L44-L77` - `MeaningfulSideEffectAuthorizationBoundary` @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `agents/external_contractor_adapter/external_work_adapter.py:L128-L136` - separate `side_effect_policy` evaluator @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `agents/external_contractor_adapter/external_work_adapter.py:L735-L818` - direct `evaluate_meaningful_side_effect` @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `agents/external_contractor_adapter/external_contractor_adapter_agent.py:L67-L76` - separate DI wiring @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `agents/external_contractor_adapter/external_contractor_adapter_agent.py:L120-L124` - passes `side_effect_policy` to domain job @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
- **Reproduction:**
  1. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:intergrax/runtime/policy/meaningful_side_effect_authorization.py` - canonical boundary composes enforcement gate.
  2. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:agents/external_contractor_adapter/external_work_adapter.py` - adapter owns injected evaluator path.
  3. `git grep -n "MeaningfulSideEffectAuthorizationBoundary" 042cc9b50386cfcd4da30310c84d000dbf5d2718 -- agents/external_contractor_adapter/` - no consumer use on External Work path.
- **Impact:** Inconsistent authority semantics and duplicate policy ownership across meaningful-side-effect consumers undermine Governed Execution spine claims.
- **Confidence:** CONFIRMED

### AUDIT-20260818-POLICY_GOVERNANCE-02

**General ALLOW can shadow a later specific DENY**

- **Severity:** HIGH
- **Category:** SECURITY
- **Related classification:** ARCHITECTURE DEFECT · TEST GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** PG-FIX-B
- **Claim falsified:** Policy resolution cannot authorize an action when a more-specific applicable DENY exists merely because of rule list order.
- **Observation:** `MeaningfulSideEffectPolicyRule(action=None)` acts as wildcard. `RuntimePolicyEngine.evaluate_meaningful_side_effect()` uses first-match semantics and breaks after first matched rule. Therefore a broad ALLOW placed before a later action-specific DENY can authorize without evaluating the DENY. Tests encode first-match semantics and unrestricted-rule matching. No exploited production incident claimed.
- **Location:**
  - `intergrax/contracts/meaningful_side_effect_policy.py:L28-L35` - `action: str | None = None` wildcard @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `intergrax/runtime/policy/runtime_policy_engine.py:L66-L91` - first-match `break` @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `tests/unit/runtime/policy/test_meaningful_side_effect_policy.py:L221-L250` - first-match / unrestricted tests @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
- **Reproduction:**
  1. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:intergrax/runtime/policy/runtime_policy_engine.py` - loop breaks on first match.
  2. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:tests/unit/runtime/policy/test_meaningful_side_effect_policy.py` - `test_first_match_behavior`, `test_unrestricted_action_rule_matches_any_action`.
- **Impact:** Authorization can be weakened by rule ordering mistakes without explicit precedence semantics.
- **Confidence:** CONFIRMED

### AUDIT-20260818-POLICY_GOVERNANCE-03

**Meaningful-side-effect request is rich, but policy matching is effectively action-only**

- **Severity:** HIGH
- **Category:** ARCHITECTURE DEFECT
- **Related classification:** SECURITY
- **Status at publication:** ACCEPTED
- **Remediation block:** PG-FIX-A
- **Claim falsified:** Meaningful-side-effect authorization composes tenant/principal/resource/target/scope dimensions, not action-only matching.
- **Observation:** Request carries action, kinds, side_effect_scope_id/digest, task_id/run_id, principal_id, tenant_id, resource, external_target, correlation/context. Rule carries essentially rule_id, decision, action, reason. Matching selects by action rather than tenant/principal/resource/target/scope. `CollaborativeWorkAuthorityResolver` is a positive stronger pattern to reuse/compose - Intergrax does have authority resolver machinery.
- **Location:**
  - `intergrax/contracts/meaningful_side_effect.py:L35-L59` - rich request fields @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `intergrax/contracts/meaningful_side_effect_policy.py:L28-L35` - action-only rule shape @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `intergrax/runtime/policy/runtime_policy_engine.py:L66-L69` - action match only @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
- **Reproduction:**
  1. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:intergrax/contracts/meaningful_side_effect.py` - request field set.
  2. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:intergrax/runtime/policy/runtime_policy_engine.py` - matcher compares `rule.action` to `request.action` only.
- **Impact:** Rich request context does not constrain authorization; scope/tenant/target isolation claims are not enforced at this evaluator.
- **Confidence:** CONFIRMED

### AUDIT-20260818-POLICY_GOVERNANCE-04

**Canonical scoped approval grant exists but inspected External Work consumer does not consume it**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Related classification:** RELIABILITY · SECURITY
- **Status at publication:** ACCEPTED
- **Remediation block:** PG-FIX-C
- **Claim falsified:** After verified human approval, the exact scoped continuation grant authorizes only the matching side-effect continuation on all inspected production consumers.
- **Observation:** `GovernedContinuationApprovalGrant` binds exact continuation request, side-effect scope id/digest, task/run, operation/resource, policy rule, pause and human request. Coordinator mints it only after exact verified approval. External Work continuation accepts its own evidence shape and re-evaluates `MeaningfulSideEffectEvaluator`. `MeaningfulSideEffectRequest` has no canonical approval-grant carrier. Standard `RuntimePolicyEngine` does not interpret the scoped approval grant. A REQUIRE_HUMAN rule can therefore remain REQUIRE_HUMAN after approval unless host provides special semantics. Approval does not override DENY - it may authorize only the exact approved operation/scope. Later G5C commits on current development must not re-interpret this historical finding.
- **Location:**
  - `intergrax/contracts/governed_continuation_grant.py:L20-L43` - scoped grant fields @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `intergrax/runtime/human/governed_continuation_grant.py:L69-L102` - grant minting after approval @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `agents/external_contractor_adapter/external_work_adapter.py:L495-L529` - continuation evidence shape @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
  - `agents/external_contractor_adapter/external_work_adapter.py:L803-L818` - re-evaluates evaluator, no grant carrier @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
- **Reproduction:**
  1. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:intergrax/contracts/governed_continuation_grant.py` - grant binding contract.
  2. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:intergrax/runtime/human/governed_continuation_grant.py` - coordinator mint path.
  3. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:agents/external_contractor_adapter/external_work_adapter.py` - continuation forwards evidence / re-evaluates policy without grant consumption.
- **Impact:** Verified human approval may not close the exact continuation on External Work paths; REQUIRE_HUMAN can persist after approval.
- **Confidence:** CONFIRMED

### AUDIT-20260818-POLICY_GOVERNANCE-05

**Policy bundle contains hidden suffix matching via rule_id**

- **Severity:** MEDIUM
- **Category:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** PG-FIX-D
- **Claim falsified:** Critical policy matching uses explicit typed fields only; rule identifiers are not hidden dispatch instructions.
- **Observation:** Explicit typed `match_action` exists. Evaluator can still infer action by checking whether `rule_id` ends with `.<ACTION>`. This is backward-compatible magic-string dispatch in a critical policy contract. Current project clean-cut posture does not justify silent runtime semantics hidden inside identifier naming.
- **Location:**
  - `intergrax/runtime/policy/runtime_policy_bundle_evaluator.py:L133-L143` - `match_action` plus `rule_id.endswith` suffix @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`
- **Reproduction:**
  1. `git show 042cc9b50386cfcd4da30310c84d000dbf5d2718:intergrax/runtime/policy/runtime_policy_bundle_evaluator.py` - `_match_rule` suffix branch.
- **Impact:** Policy behavior depends on opaque identifier naming conventions rather than explicit contract fields.
- **Confidence:** CONFIRMED

## Provider / backend abstraction

| concern | canonical abstraction | provider boundary / selection | observed provider(s) | classification | evidence / finding |
|---------|-----------------------|-------------------------------|----------------------|----------------|--------------------|
| ExternalWorkIntegration | `ExternalWorkIntegration` | composition-owned adapter | product integration profiles | `ABSTRACTION_PRESERVED` | `agents/external_contractor_adapter/external_work_adapter.py` @ `042cc9b50386cfcd4da30310c84d000dbf5d2718`; no new vendor leak |
| Policy plugin registry / typed handler | declarative policy handler contracts | plugin registry selection | platform policy plugins | `ABSTRACTION_PRESERVED` | runtime policy bundle path @ audited SHA |
| Plugin selection | composition wiring | `COMPOSITION_ONLY` | host/profile wiring | `COMPOSITION_ONLY` | no generic runtime vendor coupling |
| Runtime policy contracts | `MeaningfulSideEffectRequest` / `PolicyDecision` | Tier-1 evaluator | platform runtime | `ABSTRACTION_PRESERVED` | findings are semantics/ownership, not vendor leakage |
| Authority repositories | `CollaborativeWorkAuthorityResolver` | Tier-1 authority | platform runtime | `ABSTRACTION_PRESERVED` | positive pattern for PG-FIX-A composition |

AUDIT-5 discovered no new independent provider/backend abstraction finding.

## Falsification log

Targets examined but **not** promoted to findings:

1. **ToolRuntime bypasses policy** - inspected tool execution enforces policy before handler (positive control).
2. **Declarative plugin failures open** - declarative plugin failures fail closed (positive control).
3. **REQUIRE_HUMAN never stops tool execution** - REQUIRE_HUMAN stops inspected tool execution path (positive control).
4. **External Work missing policy allows execution** - missing policy denies (positive control).
5. **Generic ToolRuntime bypass** - no generic ToolRuntime bypass finding in this layer.

## Prior-audit comparison

Prior campaign layers established governed execution boundary and identity/trust themes. This layer owns **policy/governance-specific** claims: single side-effect spine, resolution precedence, scoped approval consumption, and explicit matching. No prior canonical Protocol v2.2 `POLICY_GOVERNANCE` immutable snapshot existed before this layer.

## Open questions / blocked items

- Convergence of `CollaborativeWorkAuthorityResolver` with meaningful-side-effect matching - planning only (**PG-FIX-A**).
- Whether suffix `rule_id` migration requires a separately approved compatibility window - planning only (**PG-FIX-D**).
- No operator-disputed findings; no blocked evidence collection.

## Operator acceptance

- **Date:** 2026-08-19
- **Accepted findings:** all 5 (`AUDIT-20260818-POLICY_GOVERNANCE-01` … `AUDIT-20260818-POLICY_GOVERNANCE-05`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
- **Remediation blocks:** PG-FIX-A, PG-FIX-B, PG-FIX-C, PG-FIX-D - all **ACCEPTED / PLANNED** only; not implemented by this persistence task
