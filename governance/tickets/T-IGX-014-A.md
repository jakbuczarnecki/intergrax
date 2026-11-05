# T-IGX-014-A — Harness ADR README index integrity (global hygiene)

## Scope

Reconcile **`docs/project/technical/adr/README.md`** index with on-disk **`entries/`** ADRs and align **`scripts/maintenance/check_harness_adr.py`** index parsing with documented ADR ID conventions (multi-segment IDs, slug filenames). **Documentation + validator contract only** — no product tiers, no ADR **`Accepted`** promotions, no FRZ/B6/ARCH-FREEZE claims.

**Trigger:** ASF-SEF-P1 closure prep @ `7de6b41f637df1d6949985d9dc567602a074e230` — `check_harness_adr.py` reports **24** pre-existing failures (see [`T-IGX-SEF-A`](T-IGX-SEF-A.md) closure dossier `.tmp/session/ASF-SEF-P1-CLOSURE/DOSSIER.md` § ADR_GLOBAL_BASELINE_RECONCILIATION).

## Plan digest

(none — hygiene batch; operator scope approval via ticket `Approved:` line)

## Approved

(pending — scaffold only)

## Test contract

**Positive (batch close requires exit 0):**

```powershell
uv run python scripts/maintenance/check_harness_adr.py
```

**Negative contract (must remain fail-closed until fixed):**

- Any ADR file under `entries/YYYY-MM-DD/` without a README index link that resolves to that file → validator exit **1**.
- Any README index link whose target file is missing → exit **1**.
- Stray `ADR-*.md` at `adr/` root → exit **1**.

**Regression guard (after parser fix):** unit or snapshot test that README rows for `ADR-GOVERNED-CONTINUATION-001`, `ADR-GR-10-003-*`, `ADR-UCA-6C-*` are counted as indexed (representative parser cases).

**Governance:**

```powershell
uv run python -m scripts.governance.igx_gov ticket
```

## Findings inventory (24 @ baselines `b342e5e7` and `7de6b41f`)

| Basename | Path under `docs/project/technical/adr/` | Cause |
|----------|--------------------------------------------|-------|
| ADR-AW-001.md | `entries/2026-09-02/ADR-AW-001.md` | missing README row |
| ADR-CTRL-X-001-MIDDLEWARE-HOOK-CONTRACT-OWNERSHIP.md | `entries/2026-10-03/…` | missing README row |
| ADR-EXECUTION-BOUNDARY-EVENT-001.md | `entries/2026-07-20/…` | parser `INDEX_LINK_RE` too narrow |
| ADR-GOVERNED-CONTINUATION-001.md | `entries/2026-07-20/…` | parser |
| ADR-GOVERNED-EXECUTION-001.md | `entries/2026-08-16/…` | parser |
| ADR-GOVERNED-EXECUTION-002.md | `entries/2026-08-17/…` | parser |
| ADR-GOVERNED-EXECUTION-003-ROOT-GOVERNANCE-IDENTITY-AUTHORITY.md | `entries/2026-09-18/…` | parser + slug path |
| ADR-GOVERNED-PROOF-001.md | `entries/2026-07-20/…` | parser |
| ADR-GR-10-001.md | `entries/2026-09-18/…` | parser |
| ADR-GR-10-002.md | `entries/2026-09-19/…` | parser |
| ADR-GR-10-003-gr10-gr13-governance-evidence-certification-scope.md | `entries/2026-09-21/…` | parser + slug path |
| ADR-GR-10-004-agentic-execution-model-uaep-canonical.md | `entries/2026-09-21/…` | parser + slug path |
| ADR-GR-5-001.md | `entries/2026-09-15/…` | parser |
| ADR-GR-8-001.md | `entries/2026-09-17/…` | parser |
| ADR-HOST-ATTESTATION-001.md | `entries/2026-07-20/…` | parser |
| ADR-MEM-XINT-002.md | `entries/2026-09-17/…` | parser |
| ADR-MP-006.md | `entries/2026-09-17/…` | missing README row |
| ADR-MP-007.md | `entries/2026-09-18/…` | missing README row |
| ADR-PLATFORM-PLUGIN-001.md | `entries/2026-08-14/…` | parser |
| ADR-POLICY-SIDE-EFFECT-001.md | `entries/2026-07-20/…` | parser |
| ADR-RUNTIME-POLICY-BUNDLE-001.md | `entries/2026-07-20/…` | parser |
| ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION.md | `entries/2026-09-23/…` | parser |
| ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY.md | `entries/2026-09-22/…` | parser |
| ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md | `entries/2026-09-22/…` | parser |

**Evidence logs:** `.tmp/session/ASF-SEF-P1-CLOSURE/check_harness_adr_audited.log`, `.tmp/session/ASF-SEF-P1-CORRECTIVE/check_harness_adr.log`; START_HEAD count **24** @ worktree `b342e5e7f`.

## Out of scope

ASF-SEF P2–P13 product work; ADR-SEF-001 **`Accepted`**; Option 4; FRZ row promotion; Protocol 03 B6; changing P1 publication semantics.

## Lifecycle state

draft

## Checkpoint

(scaffold — awaiting operator `Approved:` and implementation)

**Commit trailer:** `Ref: governance/tickets/T-IGX-014-A.md`
