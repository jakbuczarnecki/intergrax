# II-W0-A — Consolidated Closure Matrix (II-W0-01..04)

**Ticket:** T-IGX-079-D
**Batch:** II-W0-A-ARCHITECTURE-ACCEPTANCE-01
**Correction batch:** II-W0-A-CORRECTIVE-03 (mechanical final settlement); prior batches II-W0-A-CORRECTIVE-01, II-W0-A-CORRECTIVE-02 and original candidate preserved in Git ancestry
**Review organization:** OD-II-W0-A-AUDIT-01 authorizes pre-publication review in existing ChatGPT orchestration — **explicit review-organization exception**; not proof of separate mandatory G6 session separation.
**Latest scoped documentary review:** II-W0-A-SCOPED-FINAL-ACCEPTANCE-01 — **DOCUMENTARY_ACCEPTANCE_PASS_SCOPED** (not formal canonical G6)
**Independent G6:** PENDING — formal session separation not satisfied by scoped documentary acceptance alone
**Accepted audited candidate:** `00ae3f6b5a882021f500b6424f8213b26263111a`
**Evidence archive SHA-256:** `416c285e78e80510d6214726f792935fa5f82680d5aa9d3d95190aca470d12e2`
**Accepted documentary blobs:** II-W0-01 `a16bd1c7718ae2e9ddd2d5becb77dc8d62335dd6`; II-W0-02 `9a9d58e43d1d43b6d883ddda9f8452ee2154a371`; II-W0-03 `e1724a91fd61bf079cd890269acd6fb68bfc2692`; II-W0-04 `2aee4593fdba959e449ba5fa6c6bffe3b53071e2`
**F-W0A-43 (premature older-candidate publication):** Unaccepted deliverable chain already on `origin/development` before scoped acceptance — deliverable `3a3bfcb707e901798d66aab1a11eddb52998d47a`; qualification record `a3d9700c86a6717b2ca0c2609f8b491f2322bc40`; superseded blobs through preflight `8f03bae217386e26ce1d7804ca673f57f6bfccfc`. Corrective publication `a6e3a6e631d1e72aaeb779a189fa331fc5542938` verified on `origin/development`; history preserved.
**F-W0A-44 (evidence limitations — itemized):** (1) Historical diff-check EOF — resolved at integration range exit 0. (2) BOM/CRLF exports. (3) Workspace preservation not independently proven. (4) Q07 at `a6e3a6e` (1894/45/1w). Formal G6 pending.
**Formal package closure authority:** NOT AUTHORIZED
**Correction scope (this batch):** II-W0-01 technical normalization; II-W0-03 24-row FRZ/ECAP table and routing/capacity bindings; II-W0-04 canonical II-OPR-06 table; matrix/register consistency; II-W0-02 unchanged.

**Original candidate (preserved):** `1b81169a6dee214abf056cba0650d9892b199e26`
**Prior deliverable commit:** `3a3bfcb707e901798d66aab1a11eddb52998d47a`
**Prior corrected deliverable:** `7b5facb193249d0ec88e124b2c65f6d5cd36f361`
**Erroneous reported SHA (historical discrepancy):** `1b81169a86a6717b2ca0c2609f8b491f2322bc40`

**Historical review verdicts (preserved):** II-W0-A-PREPUBLICATION-AUDIT-01 `CHANGES_REQUIRED`; II-W0-A-PREPUBLICATION-REAUDIT-01 `TARGETED_CORRECTIONS_REQUIRED`; II-W0-A-PREPUBLICATION-FINAL-REVIEW-01 `LIMITED_CORRECTIONS_REQUIRED`.

**Program gates (unchanged):** `GATE-II-ENTRY`, `GATE-II-ARCH`, `GATE-II-CORE`, `GATE-II-ADAPTIVE`, `GATE-II-ENTERPRISE`, `GATE-INTELLIGENCE-CONSUME` — **NOT_REACHED**. W0-A formal package closures **0/4**; W0 formal package closures **0/6**; total roadmap formal closures **0/69**.

**Integration and publication (verified):** Original publication preflight baseline `8f03bae217386e26ce1d7804ca673f57f6bfccfc`; actual rebased integration parent `9efe643045c40255ce868e99400e9468d2fee4f4`; verified publication commit `a6e3a6e631d1e72aaeb779a189fa331fc5542938` (ancestor of remote `origin/development`); seven-path integration write-set only. Historical Cursor push rejection during II-W0-A-PUBLICATION-01 **superseded** — remote publication independently verified without attributing push actor. Settlement `74a5ff47956d48497368797dac406547924a2216`; II-W0-A-G6-HANDOFF-01 transitions ticket to `ready_for_audit` without formal G6 verdict.

| Package | Original acceptance criterion | Required output | Positive oracle (roadmap definition) | Negative oracle (roadmap definition) | Actual positive oracle result | Actual negative oracle result | Exact evidence locator | Candidate SHA | Publication SHA | Independent review | Residual findings | Actual lifecycle | Formal closure eligibility |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| II-W0-01 | Layer diagram and classification; Tier-0–Tier-3 unchanged | `II-W0-01_LAYER_TIER_BOUNDARIES.md` | Legal directions; Execution sole admission | Tier insert, reverse import, second root, private scheduler, alt admission | **PASS (documentary)** — legal directions and Execution admission (semantic unchanged after normalization) | **PASS (documentary)** — negative cases REJECT | `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-01_LAYER_TIER_BOUNDARIES.md` | `00ae3f6b5a882021f500b6424f8213b26263111a` | `a6e3a6e631d1e72aaeb779a189fa331fc5542938` | DOCUMENTARY_ACCEPTANCE_PASS_SCOPED | No runtime tier test | T-IGX-079-D `ready_for_audit` | **No** |
| II-W0-02 | Definition, scope, non-goals and adoption contract accepted | `II-W0-02_CAPABILITY_CONTRACT.md` | Deterministic and intelligent examples without authority transfer | Implicit permission, private harness, invented API, consumer lifecycle, duplicated criteria | **PASS (documentary)** — representative requests without authority transfer | **PASS (documentary)** — §10 negative cases REJECT | `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-02_CAPABILITY_CONTRACT.md` | `00ae3f6b5a882021f500b6424f8213b26263111a` | `a6e3a6e631d1e72aaeb779a189fa331fc5542938` | DOCUMENTARY_ACCEPTANCE_PASS_SCOPED | W1 port mapping not executed | `ready_for_audit` | **No** |
| II-W0-03 | Traceability to existing invariants; no conflicting authority | `II-W0-03_INVARIANT_TRACEABILITY.md` | 12/12 authority preserved + accountable evidence owners | Weakened controls; unbound claims; duplicate planes | **PASS (documentary)** — 12 rows; 24 explicit FRZ/ECAP bindings; domain sources incl. routing/capacity | **PASS (documentary)** — deferrals documented; no weakened controls | `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-03_INVARIANT_TRACEABILITY.md` | `00ae3f6b5a882021f500b6424f8213b26263111a` | `a6e3a6e631d1e72aaeb779a189fa331fc5542938` | DOCUMENTARY_ACCEPTANCE_PASS_SCOPED | Message-extension owner W1 | `ready_for_audit` | **No** |
| II-W0-04 | Single owner per concern; overlap and conflict matrix | `II-W0-04_SEMANTIC_OWNERSHIP.md` | One owner or explicit block per §7.1 concern | Duplicate identity/store/recovery/budget/permission planes | **PASS (documentary)** — 23/23 rows; 17-row canonical II-OPR-06 reproduction | **PASS (documentary)** — negative qualification REJECT | `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-04_SEMANTIC_OWNERSHIP.md` | `00ae3f6b5a882021f500b6424f8213b26263111a` | `a6e3a6e631d1e72aaeb779a189fa331fc5542938` | DOCUMENTARY_ACCEPTANCE_PASS_SCOPED | Cross-domain acceptance record-only | `ready_for_audit` | **No** |

**Oracle separation:** Roadmap definitions remain in the definition columns. **Actual** positive and negative oracle results are documentary qualification outcomes only; they do not imply runtime certification, formal package closure, or publication.

**Corrective publication (II-W0-A-PUBLICATION-01):** Preflight integration baseline `8f03bae217386e26ce1d7804ca673f57f6bfccfc`; rebased parent `9efe643045c40255ce868e99400e9468d2fee4f4`; verified publication `a6e3a6e631d1e72aaeb779a189fa331fc5542938`. **Integration Q07:** PASS at qualification SHA `a6e3a6e` — evidence `igx_gov_smoke_rebase.log` (**1894 passed, 45 deselected, 1 warning**).

**Formal G6:** PENDING — handoff at `.tmp/proof/II-W0-A-G6-HANDOFF-01/FORMAL_G6_HANDOFF.md` (OD-II-W0-A-G6-01; lifecycle `ready_for_audit`); not executed in this session. **Formal closure eligibility:** No for all packages.

No self-referential publication commit SHA embedded at authoring time. Implementer integration qualification does not close packages.
