# HIST / ENT shared IIAC re-activation — coordination request

| Field | Value |
|---|---|
| Status | **PROPOSED_NO_CONSENT** |
| Policy version | `IIAC-2026-11-05.1` (`IN_FORCE`) |
| Activation / effective SHA (current) | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` |
| Audited SHA (current record) | `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0` |
| HIST campaign START_HEAD | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| Published | 2026-11-08 |

**This document is a factual coordination request, not consent.** No agent-to-agent channel exists; silence is not supersession.

## Parties

| Role | Identity |
|---|---|
| HIST campaign owner | Historical certification closure (HIST-01..04), this master plan execution |
| ENT-AUDIT-X execution owner | T-IGX-024-A stream (identity not assumed by HIST) |
| Operator | Required for IIAC `operator @` activation line; not delegable under current gate code |

## Why re-activation is needed

Between audited SHA `ae467c8c4` and verified `development` HEAD `fe0937517`, product-path and material-path diffs are **empty**, so `verify_iiac_policy_activation --require-in-force` can remain green today.

The planned HIST product batch (T-IGX-030-B, 030-C, 032-A, 033-A) changes paths under `intergrax/`, `agents/`, and `applications/`. After merge, ACT-N11 (`iiac_activation_gate.py` product loop) fails for **every consumer** of `development`, including ENT, until a new audited activation is published and the Operator signs the activation line.

Tests-only and docs-only commits on `development` do not trigger ACT-N11 product-path failures.

## Proposed product batch (HIST)

Isolated integration target branch: `hist-cert/product-batch` (lanes `hist-cert/c1-030b` … `c4-033a`). **Not merged to `development` until P2-WAIT clears.**

| Lane | Ticket | Product touch (summary) |
|---|---|---|
| C1 | T-IGX-030-B | Contract home for `RuntimeErrorCode` |
| C2 | T-IGX-030-C | Invoker no-boundary submit terminalization |
| C3 | T-IGX-032-A | Pin resolver KV precedence |
| C4 | T-IGX-033-A | Tier-0 CLI command-set seam |

Governance batch **excludes** T-IGX-021-B/C unless a later amendment says otherwise.

Target integration SHA: **TBD until P3-INT**.

## ENT stream (reference only)

ENT-AUDIT-X remains a separate stream. HIST does **not** edit ENT dossier, FRZ-AUD rows, or `IIAC_POLICY_ACTIVATION_RECORD.json`. ENT local worktrees (`ent-w1`, `ent-w2-fix`, `ent-w3-evid`, `ent-tes3`) are out of scope and not listed as agreed branches.

## Proposed sequencing (HIST does not choose alone)

Either:

1. **ENT-first:** ENT finishes and publishes its current non-product closure, then HIST merges one product batch and one shared re-activation; or
2. **Shared-final:** ENT explicitly accepts one shared final re-activation after both product sets land (requires ENT counter-signature on this file or a successor section).

## Acceptance gates (post-merge)

- Independent IIAC audit on the product-only delta (fresh session; audit owner TBD).
- `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` exit **0** on the new activated SHA (after Operator activation, not before).
- Prior activation record preserved in git history (no deletion).

## Rollback

Revert HIST product merge commits on `development`; do not delete the prior activation record.

## Operator decisions still required

- ENT consent or supersession record for sequencing (P2-WAIT).
- Personal IIAC activation line (`Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <time>`); delegate lines cannot activate.

## Consent record

| Party | Consent |
|---|---|
| HIST campaign | Proposes this request |
| ENT-AUDIT-X owner | **No in-repo consent recorded** |
| Operator | **No activation signature in this document** |

To accept: add a dated section with ENT session id, exact SHA accepted, and scope; or publish an Operator supersession record.
