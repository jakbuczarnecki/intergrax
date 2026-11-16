# IIAC ACT-N11 recovery batch — MCP-SERVER-X (P2–P4 product Δ)

**Campaign:** IIIC product re-bind after ENT Session B activation  
**Ticket anchors:** `governance/tickets/T-IGX-035-E.md`, `T-IGX-035-C.md`, `T-IGX-035-D.md`  
**Policy:** `IIAC-2026-11-05.1` — verifier **must not** be weakened.  
**Prior activation pin (`audit_execution_sha`):** `11ac8ad85f62a6925d25ea840c92d5c6a936cfe4` ([`IIAC_POLICY_ACTIVATION_RECORD.json`](../IIAC_POLICY_ACTIVATION_RECORD.json))  
**ENT Session B scope (closed):** seven PRE-ENT/ENT paths — [`audit_scope.json`](../iiac_evidence/ENT-AUDIT-X/act_n11_recovery/audit_scope.json)  
**Date:** 2026-11-11  
**Publisher:** ACT-N11 MCP corrective batch (governance evidence only — **no** activation record edit, **no** merge)

---

## 1. Verification snapshot

| Check | Result |
| --- | --- |
| `git fetch origin` | OK |
| `origin/development` @ batch | `7172f294fdd44fe35024ba60378fb9f4e0125963` |
| PR #454 head | `06be8bfd4ee56539efe2fe2b4e73eb563cef16f4` (`mcp-server-x-p4`) |
| `verify_iiac_policy_activation --require-in-force` @ `7172f294f` | **FAIL** — **7× ACT-N11** (MCP P2/P3 on `development`) |
| Same @ PR #454 head | **FAIL** — **16× ACT-N11** (P2–P4 cumulative) |
| CI `governance-fast` @ PR #454 | **FAIL** — run [`43010550205`](https://github.com/jakbuczarnecki/intergrax/actions/runs/43010550205) job `130488899947` (15 failed tests; root `verify_iiac` exit 1) |

**Mechanism (read-only):** `scripts/governance/iiac_activation_gate.py` — `product_paths_changed_between(audit_execution_sha, HEAD)`; scoped `audit_scope.json` does **not** narrow the gate.

---

## 2. Product path inventory vs `11ac8ad85f…`

### 2a. On `origin/development` (blocks `development` + any PR until re-bind)

| # | Path | Introducer (first change after pin) | Stream |
| --- | --- | --- | --- |
| 1 | `intergrax/contracts/mcp_server_definition.py` | `e21453caf` (P2 definition) · tip drift via P3/P4 | MCP **P2** |
| 2 | `intergrax/applications/_shared/mcp_server_engine.py` | `e21453caf` · tip `a21a6629a` (P4) | MCP **P2→P4** |
| 3 | `intergrax/applications/_shared/mcp_host_mount.py` | `bee38a76b` | MCP **P3** |
| 4 | `intergrax/applications/_shared/mcp_import_guard.py` | `bee38a76b` | MCP **P3** |
| 5 | `intergrax/applications/_shared/mcp_engine_concurrency.py` | `bee38a76b` | MCP **P3** |
| 6 | `intergrax/applications/_shared/settings_loader.py` | `bee38a76b` | MCP **P3** |
| 7 | `intergrax/applications/_shared/mcp_nexus_server.py` | — (unchanged on `development` vs pin) | _n/a on dev_ |

### 2b. Additional on PR #454 only (P4 facades)

| # | Path | Introducer | Stream |
| --- | --- | --- | --- |
| 8 | `intergrax/applications/_shared/mcp_application_server.py` | `a21a6629a` | MCP **P4** |
| 9 | `intergrax/applications/_shared/mcp_nexus_server.py` | `a21a6629a` | MCP **P4** |
| 10–17 | `applications/*/mcp/server.py` (8 apps) | `a21a6629a` | MCP **P4** |

Canonical list for auditor scope: [`iiac_evidence/MCP-SERVER-X/act_n11_mcp/audit_scope.json`](../iiac_evidence/MCP-SERVER-X/act_n11_mcp/audit_scope.json).

---

## 3. Minimal compliant re-bind (existing IIIC only)

| Step | Owner | Artifact |
| --- | --- | --- |
| 1 | **Independent auditor** (≠ P4 implementer) | `PASS_SCOPED` dossier + `iiac_audit_session_manifest.MCP-SERVER-X.act_n11_mcp.json` under `iiac_evidence/MCP-SERVER-X/act_n11_mcp/` |
| 2 | **Certifier** | Manifest validation + `verify_iiac_role_separation --strict-product-diff` logs |
| 3 | **Operator** | Personal line in [`IIAC_OD_ACTIVATION_MCP_ACT_N11_2026-11-11.md`](../IIAC_OD_ACTIVATION_MCP_ACT_N11_2026-11-11.md) — **not** delegate-only |
| 4 | **Operator / certifier** | Two-phase update of [`IIAC_POLICY_ACTIVATION_RECORD.json`](../IIAC_POLICY_ACTIVATION_RECORD.json) — new `audit_execution_sha` ≥ post-audit evidence SHA |
| 5 | **CI** | `verify_iiac_policy_activation --require-in-force` exit **0**; `governance-fast` **SUCCESS** on target merge head |

**Forbidden:** implementer edit of activation record, ACT-N11 allowlist/weakening, self-PASS, merge PR #454 before step 5.

**This batch delivered:** inventory, MCP `audit_scope.json`, activation dossier template, OD file `PREPARED` — **PENDING** steps 1–4.

---

## 4. Operator decision package **D-MCP-IIAC-01**

| Option | Action | Clears ACT-N11 for MCP? |
| --- | --- | --- |
| **(a) Shared MCP IIIC re-activation** | One independent session on §2 inventory @ merge target SHA; Operator signs; publish record | **Yes** |
| **(b) Merge PR #454 without IIIC** | Delegate merge only | **No** — stays FAIL on `development` and PR |
| **(c) Revert MCP P2–P4 product** | Reset paths to bytes @ `11ac8ad85f…` | Theoretically yes — **rejects** landed MCP work |

**Orchestrator recommendation:** **(a)** — aligns ENT Session B pattern and `MCP_SERVER_X_P4_CLOSURE_REPORT.md` blocker B-P4-GOV-01.

Record: `Approved: D-MCP-IIIC-01=a` (Operator chat) before auditor session starts.

---

## 5. Cross-links

- P4 closure: [`MCP_SERVER_X_P4_CLOSURE_REPORT.md`](./MCP_SERVER_X_P4_CLOSURE_REPORT.md)  
- ENT Session B SSOT: [`../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md`](../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md)  
- Activation dossier (empty `operator @`): [`../iiac_evidence/MCP-SERVER-X/act_n11_mcp/SESSION_MCP_ACTIVATION_DOSSIER.md`](../iiac_evidence/MCP-SERVER-X/act_n11_mcp/SESSION_MCP_ACTIVATION_DOSSIER.md)
