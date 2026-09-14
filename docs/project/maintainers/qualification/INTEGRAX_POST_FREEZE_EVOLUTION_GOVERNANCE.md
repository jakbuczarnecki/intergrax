# INTEGRAx-POST-FREEZE-EVOLUTION-GOVERNANCE

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-POST-FREEZE-EVOLUTION-GOVERNANCE` |
| **Date** | 2026-09-13 |
| **Branch policy** | `development` (unless operator directs otherwise) |
| **Frozen code baseline SHA** | `a185403d0c7524c29bea2fe09212f9508e6bccd8` |
| **Freeze record commit SHA** | `59fbf6f305b70d2b74adac7cd61dd21d352dba78` |
| **Freeze provenance correction SHA** | `e60fc0162e3302d5be0c320593755a9657de23ce` |
| **Parent freeze SSOT** | [`INTEGRAX_CORE_PLATFORM_FREEZE.md`](INTEGRAX_CORE_PLATFORM_FREEZE.md) |

**Status:** **Post-freeze evolution governance = ACTIVE**

**Semantic anchor (immutable unless formal re-freeze):**

```text
Certified Core Platform = FROZEN
Frozen code baseline SHA ≠ repository HEAD (automatic baseline advancement forbidden)
```

This document is the **single maintainer SSOT** for **how** changes after Core Platform Freeze are classified, gated, audited, and (when required) reopened. It does **not** reopen frozen core, modify production semantics, or replace the freeze record—it **governs evolution** around and through certified extension surfaces.

---

## Governance objective (four questions)

Every proposed change after freeze must answer:

| # | Question |
| - | -------- |
| 1 | Does the change touch **frozen core** (implementation or contract surface listed in Frozen Surface Registry)? |
| 2 | What **change class** applies (A, B, or C)? |
| 3 | Which **gates** must pass before merge? |
| 4 | Is a **scoped Architecture Reopen** required? |

---

## Fail-safe classification rule

```text
If a change cannot be classified unambiguously as Class A or Class B → DEFAULT = Class C
```

Do **not** choose the most liberal class when ambiguity exists.

---

## Change classification

### Class A — Safe extension

Change does **not** modify frozen **contract semantics** or **ownership**.

**Examples:**

- New plugin, provider, adapter, exporter, model provider, storage provider
- New strategy using existing contracts
- New domain module attached via contracts
- New composition-root binding of approved implementations

**Model:**

```text
existing contract → new implementation → composition registration
```

**Architecture reopen:** **Not required**

---

### Class B — Core-compatible change

Change touches **frozen core implementation** but does **not** change:

- Public contract semantics
- Lifecycle ownership
- Identity authority
- Canonical Decision → Execution path
- Persistence authority
- Governance authority

**Examples:** bugfix, performance optimization, internal refactor, hardening, deterministic validation improvement (same external behavior).

**Requires:**

```text
Class A minimum gates
+ frozen architecture gate families (impacted)
+ static quality (ruff, format, pyright in scope)
+ independent audit based on actual GitHub code
```

**Architecture reopen:** **Not required** (unless misclassified—when in doubt, Class C)

---

### Class C — Architecture / contract evolution

Change touches **any** of:

- Public contract (including breaking or semantic drift)
- Execution ownership or root execution path
- Lifecycle ownership
- Governance semantics (`DENY` / `REQUIRE_HUMAN` / `ALLOW`, authorization boundary)
- `DecisionExecutionAuthorization` semantics
- Persistence **contract** ownership or semantics
- Retry ownership or recovery ownership
- Identity authority (minting of run_id, execution_id, attempt_id, task_id)
- Canonical path: Decision → Governance → DecisionExecutionAuthorization → ExecutionRequest → ExecutionRuntime
- Any frozen invariant **INV-1 … INV-10** or **REL-1 … REL-8**

**Requires:**

```text
Architecture Reopen Record (scoped preferred)
→ migration analysis
→ impacted frozen gates
→ scoped or full recertification
→ independent audit based on actual GitHub code
→ updated freeze record (only after explicit re-freeze)
```

---

## Automatic Class C triggers (non-exhaustive)

| Trigger | Class |
| ------- | ----- |
| Root execution path or `ExecutionRuntime` ownership change | **C** |
| `DENY` / `REQUIRE_HUMAN` / `ALLOW` or `DecisionExecutionAuthorization` semantic change | **C** |
| Persistence **contract** semantic change | **C** |
| New retry owner or recovery executor (not policy impl on existing contract) | **C** |
| New identity minting outside approved owner | **C** (forbidden without reopen) |
| Evidence or observability path that **steers** execution control | **C** / forbidden without reopen |
| Change to public EE-B1.1 reliability contract semantics | **C** |
| Any change to an **existing** frozen public contract | **C** (even if “small”) |

---

## New contract rule

| Situation | Typical class |
| --------- | ------------- |
| Wholly new contract extending an extension surface **without** new ownership/authority | **A** or **B** (gate by touch surface) |
| New contract introducing new ownership, authority, or alternate execution/governance path | **C** |

---

## Frozen Surface Registry

Surfaces below are **frozen** at baseline `a185403d0…` unless a formal Class C reopen and re-freeze say otherwise.

| Surface | Frozen owner | Change class if modified (default) |
| ------- | ------------ | ---------------------------------- |
| Decision System core | Decision System | **C** (public contract / lifecycle); **B** (internal bugfix only) |
| Governance boundary (`DENY` / `REQUIRE_HUMAN` / `ALLOW`) | Governance | **C** |
| `DecisionExecutionAuthorization` | Decision System / Governance boundary | **C** |
| `ExecutionRequest` boundary | Execution Engine / contracts | **C** |
| `ExecutionRuntime` / root execution ownership | Execution Engine | **C** |
| Execution lifecycle ownership | Execution Engine | **C** |
| Retry ownership | Execution Engine (as qualified) | **C** |
| Recovery ownership | Recovery Plane / qualified boundary | **C** |
| Persistence **contract** semantics | Core contracts | **C** |
| Persistence **provider** implementation | Plugin / provider | **A**; **B** if core wiring bugfix |
| Identity authority (execution lineage IDs) | Approved owners only | **C** if new mint path |
| Tracing public contracts | Core contracts | **C** |
| Reliability contracts (EE-B1.1 @ baseline) | Core contracts + qualified semantics | **C** (public); **A** (replaceable classifier impl) |
| Nexus ownership / composition invariants | Composition owners (NPSC-4.2) | **C** |
| Plugin contract framework (DS-PLUGIN) | Platform plugin contract | **C** (framework); **A** (conforming plugin) |
| Production composition roots | Certified wiring | **C** (ownership change); **A** (new allowed binding) |
| Qualification harness (`testing_support/**`) | Qualification | **A/B** (not production baseline) |

**Touch rule:** If a diff modifies a row’s **frozen owner responsibility** or **public contract** → treat as **Class C** until proven otherwise.

---

## Extension Point Registry

Allowed evolution loci (must not rewrite frozen ownership in core).

| Extension point | Contract / port | Owner | Allowed class | Forbidden responsibility |
| --------------- | --------------- | ----- | ------------- | ------------------------ |
| **Plugin** | Existing plugin / platform contract | Plugin package | **A** (default) | Second decision authority; bypass governance; alternate lifecycle |
| **Provider** | Named port / protocol | Provider impl | **A** | Platform execution ownership; hidden global state; governance bypass |
| **Adapter** | Boundary adapter contract | Adapter module | **A** | Rewriting contract semantics in core |
| **Exporter** | Observability / evidence sink contract | Exporter impl | **A** | Evidence → execution control |
| **Strategy** | Strategy protocol on existing engine | Strategy impl | **A** | Owning root runtime or retry engine |
| **Domain module** | Domain contracts | Domain package | **A** | Direct frozen core mutation |
| **Composition root binding** | Wiring only | Composition owner | **A** | Parallel runtime; local governance; contract rewrite |
| **Optional capability** | Feature-flagged port | Capability module | **A** | Mandatory bypass of canonical path |

---

## Plugin extension rule

Every new plugin must:

- Implement an **existing** contract
- **Not** require frozen core changes for registration
- Be **replaceable** and **composable**
- **Not** bypass Governance → Authorization → ExecutionRequest → ExecutionRuntime
- **Not** create an alternate root lifecycle

---

## Provider extension rule

```text
contract → provider → vendor
```

Provider must **not**:

- Take platform execution ownership
- Drive execution outside its port
- Inject hidden global mutable state
- Bypass governance

**Provider swap:** **Class A**

---

## Composition rule

Composition roots **may:** select implementations, assemble dependencies, inject configuration.

Composition roots **must not:**

- Create a parallel runtime
- Rewrite contract semantics
- Instantiate local governance that overrides platform fail-closed rules

---

## Persistence governance

```text
Engine / Core → Persistence Port → Provider → Vendor
```

| Change | Class |
| ------ | ----- |
| New or swapped storage provider behind existing port | **A** |
| Persistence contract semantic or ownership change | **C** |

Post-freeze forbidden without reopen (see freeze SSOT): direct DB in engine core, concrete vendor in domain core, hidden lineage stores, alternate evidence persistence bypass.

---

## Execution governance

Any change to **root execution path** or **`ExecutionRuntime` ownership** → **Class C** automatically.

---

## Governance boundary

Any change to **`DENY` / `REQUIRE_HUMAN` / `ALLOW` semantics** or **`DecisionExecutionAuthorization`** → **Class C** automatically.

---

## Retry / recovery governance

| Change | Class |
| ------ | ----- |
| New policy **implementation** on existing retry/recovery contract | **A** or **B** |
| New retry **owner** or recovery **executor** | **C** |

---

## Identity governance

Minting `run_id`, `execution_id`, `attempt_id`, `task_id` outside the approved owner → **FORBIDDEN** → **Class C**.

---

## Evidence governance

| Change | Class |
| ------ | ----- |
| New exporter / sink (records truth only) | **A** |
| Evidence influences execution control | **C** / forbidden without reopen |

---

## Reliability governance

| Change | Class |
| ------ | ----- |
| New failure classifier (or similar) via existing Protocol | **A** |
| Public EE-B1.1 contract semantic change | **C** |

**EE-B1.2 (execution capacity / backpressure):** Not governed by this task’s implementation. **Before merge**, EE-B1.2 must be **classified** under this SSOT from the **actual diff**. If it touches frozen reliability semantics or ownership, expect **B** or **C**—do not assume Class A without analysis.

---

## Compatibility rules (core must stay vendor-neutral)

Future plugin/provider must **not** require in frozen core:

```text
if plugin_name == ...
if vendor == ...
```

If required → architecture smell → mandatory review (likely **C** or redesign to Class A).

---

## Forbidden hidden extension mechanisms

Without Architecture Reopen, **forbidden** in frozen core and production composition:

- Global registry with mutable ownership
- Service locator for architectural dispatch
- Monkey patching for behavior extension
- Runtime reflection for architectural dispatch
- `getattr` / string-based vendor logic as architecture dispatch

(Reflective dispatch policy aligned with [`INTEGRAX_CORE_PLATFORM_FREEZE.md`](INTEGRAX_CORE_PLATFORM_FREEZE.md).)

---

## Public contract policy (new or evolved)

New public contracts must be: explicit, typed, bounded, versionable, deterministic, fail-closed, vendor-neutral.

**Breaking** an existing frozen public contract → **always Class C**.

---

## Dependency direction

```text
core → contract

provider / plugin → contract

composition → provider / plugin
```

**Forbidden:**

```text
core → provider   (direct concrete vendor coupling for behavior)
```

---

## Frozen architecture gate matrix

Minimum gates **before merge** by class:

| Class | Minimum gates |
| ----- | ------------- |
| **A** | Contract tests; provider/plugin tests; targeted regression; static quality in change scope |
| **B** | All **A** gates + impacted **frozen architecture gate families** + static quality + **independent audit** |
| **C** | **Architecture Reopen Record** + impacted frozen gates + migration analysis + scoped/full recertification + independent audit + **freeze record update** only after explicit re-freeze |

---

## Frozen gate families (reference)

Point to **families**—do not treat “new tests pass” as sufficient alone.

| Family | Representative gate paths |
| ------ | --------------------------- |
| EE-A1 execution ownership | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` |
| NPSC-4.2 Nexus ownership | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py` |
| U5 zero-bypass | `tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py` |
| Decision / governance authorization | `tests/unit/contracts/test_decision_contract_architecture_gates.py`, decision qualification slices |
| Persistence abstraction | Architecture / contract gates per persistence SSOT |
| Tracing public contract | `tests/unit/contracts/test_tracing_public_contract.py` |
| EE-B1.1 reliability | `tests/unit/runtime/architecture/test_ee_b1_1_failure_semantics_certification.py` |
| Evidence must not control execution | `tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py` |

Full freeze-era index: [`INTEGRAX_CORE_PLATFORM_FREEZE.md`](INTEGRAX_CORE_PLATFORM_FREEZE.md) § Freeze enforcement gates.

**Architecture Guard Matrix SSOT (guard families, severity, archetypes, gate dependencies — does not alter classification semantics):** [`INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md`](INTEGRAX_POST_FREEZE_ARCHITECTURE_GUARD_MATRIX.md)

---

## Regression policy

A change is **not** acceptable because **only** new tests pass. It must also pass the **correct frozen regression gates** for its class and touched surfaces.

---

## Static quality policy

For **every** change class **A**, **B**, and **C** in production scope:

```text
ruff check
ruff format --check
pyright
```

(scope = touched packages/modules)

**No green static gate → NO MERGE** unless a formal accepted exception record exists.

---

## Independent audit policy

| Class | Audit |
| ----- | ----- |
| **B** | Required — based on **actual GitHub code**, not implementation report alone |
| **C** | Required — same |
| **A** | Optional; **recommended** for high-risk extension (security, persistence, auth) |

Audit evidence must include: commit SHA, diff scope, classification, touched frozen surfaces, tests run, static checks, baseline impact statement, architecture impact statement.

---

## Baseline and HEAD policy

```text
HEAD ≠ frozen baseline (except coincidental equality)
```

Future commits form a **post-freeze evolution chain**. They do **not** automatically advance:

```text
Frozen Core Baseline SHA: a185403d0c7524c29bea2fe09212f9508e6bccd8
```

Only **explicit re-freeze / recertification** may establish a **new** frozen baseline SHA.

---

## Post-freeze WIP rule

Untracked, local, or in-progress work (including feature branches not merged through governance):

```text
not certified · not frozen · not production baseline
```

until it completes classification, gates, audit, and merge policy.

---

## Architecture Reopen Record (Class C)

Required document (repo convention):

```text
docs/project/maintainers/qualification/ARCHITECTURE_REOPEN_<TASK>.md
```

(or equivalent maintainer qualification path)

**Must contain:**

- Trigger and **scoped** surface (avoid “whole platform reopened” when one contract suffices)
- Affected invariants (INV / REL / Nexus / path)
- Reason and alternatives considered
- Compatibility and migration plan
- Recertification scope
- Rollback plan

Freeze remains in force for **all unaffected** surfaces.

---

## Re-freeze policy

New frozen baseline only after:

```text
Class C change
→ recertification complete
→ final baseline selection
→ explicit re-freeze record commit
```

**Never** automatic on merge to `development`.

---

## Canonical path (unchanged)

```text
Decision
  → Governance
  → DecisionExecutionAuthorization
  → ExecutionRequest
  → ExecutionRuntime
```

No extension may omit this path without **Class C** reopen.

---

## Machine-readable policy

This repository does **not** add a new policy-engine framework in this task. **Document SSOT + existing architecture tests** are the enforcement surface. Optional manifests may be added only in a **separate** task if an established repo mechanism already exists.

---

## CI / bots

Do **not** invent new GitHub Actions, bots, or custom policy engines in this governance task.

---

## Production code changes under this governance SSOT task

When **establishing** this record only:

```text
Production code changes: NONE
```

Future tasks follow classification above.

---

## Final verdict (governance establishment)

```text
POST-FREEZE GOVERNANCE ESTABLISHED
```

---

## Findings template (for audits)

| Severity | Use |
| -------- | --- |
| **Critical** | Merge blocker; invariant breach or missing Class C reopen |
| **Major** | Wrong classification; missing frozen gates or audit |
| **Minor** | Documentation drift; non-blocking gate scope gap |
| **Observation** | Process improvement; WIP noted out of scope |
