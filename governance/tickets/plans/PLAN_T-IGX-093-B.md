# PLAN T-IGX-093-B — AHE W1-B01 cross-program integration

Batch: BATCH-ENT-AHE-W1-B01-CROSS-01.

Roadmap dependency: AHE-W1-01.

Purpose: integrate the independently qualified W1-B01-P candidate into development while resolving the SQLite accountability dependency and coordinating mandatory governance and enterprise qualification.

This plan does not create another AHE roadmap package.

## A. Qualified product identity

Preserve candidate:

`88b85392559783666c97878d6b620de7dc66517f`

Preserve exact contents of:

- `intergrax/runtime/adaptive/agent_design_search/__init__.py`
- `intergrax/runtime/adaptive/agent_design_search/contracts.py`
- `intergrax/runtime/adaptive/agent_design_search/archive.py`
- `intergrax/runtime/adaptive/agent_design_search/sqlite_archive.py`
- `tests/unit/runtime/adaptive/agent_design_search/test_contracts.py`
- `tests/unit/runtime/adaptive/agent_design_search/test_archive.py`
- `tests/unit/runtime/adaptive/agent_design_search/test_sqlite_archive.py`

No runtime redesign or test weakening.

Retain OD-AHE-03 and F-AHE-08 paired-publication boundaries.

Retain OD-AHE-04 as a limited retrospective exception for exactly the previously approved four commits.

No new historical exception.

## B. Integration topology

The selected architecture uses one explicitly authorized integration branch targeting development.

Proposed branch:

`integrate/ahe-w1-b01-cross-01`

The branch must be created from the exact externally authorized development base.

An authorized human integration action incorporates the pinned AHE candidate while preserving published history and all seven qualified file contents.

Do not merge PR #814 directly into development before the required accountability and governance reconciliation.

The existing Cursor integrator cannot integrate product candidates into development under its current target/kind allowlist.

Do not modify or bypass that allowlist.

Integration authorization, branch creation and human merge authority are separate prerequisites.

## C. Exact future EBH-5 correction

The only new implementation path proposed by T-IGX-093-B is:

`tests/qualification/ebh_5/_sqlite_touchpoint_registry.py`

Subject to explicit EBH-5 owner acceptance and Operator-approved plan digest.

Add exactly one new record to `_SQLITE_TOUCHPOINTS`:

```python
SqliteTouchpointRecord(
    "intergrax/runtime/adaptive/agent_design_search/sqlite_archive.py",
    SqliteTouchpointClassification.INTERNAL,
    None,
    "ADAS W1 LAB candidate archive — no production persistence authority",
),
```

Preserve the existing classification vocabulary.

Preserve all existing registry entries.

Preserve closed-world discovery and dynamic count calculation.

Do not change discovery rules, suppress the new path or hardcode the expected inventory count.

The registry correction and all seven qualified AHE files must coexist in the same integration candidate before qualification and publication to development.

An isolated registry addition without the corresponding ADAS path is prohibited.

Required acceptance:

- Exact new path discovered and registered once.
- Classification INTERNAL.
- No missing or stale registered paths.
- No production persistence or promotion authority granted.

Required test:

`tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py::test_ebh5_q09_sqlite_touchpoints_fully_accounted`

## D. Cross-program qualification sequence

Execute future authorized stages in this order.

### Stage 1 — Integration composition

- Pin exact development base.
- Incorporate the qualified AHE candidate.
- Apply the authorized EBH-5 registry correction.
- Verify all seven AHE file identities.
- Verify the integration delta contains no unrelated changes.

### Stage 2 — Scoped regression

Execute the focused ADAS and EBH-5 accountability tests.

Preserve the previous independent qualification as historical evidence.

Compare the two adjacent ExecutionId failures with their established baseline.

Do not waive or silently suppress failures.

### Stage 3 — Enterprise and EVOL-X reconciliation

Resolve the applicable external findings through their existing owner-controlled tickets and approvals.

Current finding inventory:

- ENT-A-PRD-01
- ENT-A-PRD-02
- ENT-A-REG-03
- ENT-A-TEN-01
- ENT-A-TEN-02
- TEN-IDENTITY
- VBN-SQLITE-EFFECTIVE
- EVOL-X A-18/A-19 findings
- Governance-fast residuals

T-IGX-093-B does not authorize implementation changes in these external domains.

Bind each disposition to actual evidence from the integration candidate.

Preserve previously measured baseline failures and platform limitations.

Do not assume unchanged failure counts establish absence of regression.

### Stage 4 — Audited IIAC rebind

After the final product tree is established, use the canonical independent IIAC procedure.

Include all four ADAS product paths in the full applicable product delta.

Bind actual product, audit evidence and certification identities according to the existing protocol.

Require independent audit and authorized Operator activation.

Do not mechanically repoint the activation record.

No T-IGX-093-B authority to modify IIAC certification records.

### Stage 5 — Final integration qualification

Require actual disposition of:

- EBH-5 SQLite accountability.
- ACT-N11.
- EVOL-X preflight.
- Governance-fast.
- Enterprise audit.
- Previously masked downstream CI.
- Mandatory local governance gates.
- Formal independent review and integration authorization.

Revalidate the target development identity before integration.

If development advances materially, reconcile the integration tree and rerun affected qualification.

### Stage 6 — Authorized integration

Merge only through an explicitly authorized human or repository-supported integration mechanism satisfying all mandatory branch protection and review requirements.

Verify the actual resulting development SHA and the seven AHE file identities.

Do not bypass protection or use unauthorized GitHub mutations.

Do not declare production certification.

## E. W1-B01-D dependency

After verified product integration, prepare the separate documentation authorization under OD-AHE-03 for exactly:

- `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`
- `docs/project/maintainers/plans/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`
- `docs/project/maintainers/plans/satellites/ADAPTIVE_HARNESS_EVOLUTION.md`

Bind the documentation to actual integration SHA, scoped independent verdict, all fourteen canonical ADAS obligations, actual evidence, OD-AHE-04 and residual limitations.

W1-B01 closes only after required product and documentation integration, independent acceptance and governance settlement.

## F. Next implementation batch

W1-B02 remains unauthorized until W1-B01 acceptance.

Its planned scope remains one consolidated implementation batch covering:

- AHE-W1-02.
- AHE-W1-03.
- AHE-W1-04.

Six proposed runtime modules and six focused test modules.

Do not split the batch into independent microtasks unless a mandatory dependency requires it.

## G. Future execution prerequisites

Before executing the integration plan require:

- Operator approval of the exact plan digest.
- EBH-5 owner acceptance.
- Explicit integration branch authorization.
- Authorized integration mechanism.
- External owner-controlled governance qualification scopes.

The preparation publication is not implementation authorization.

## H. Rollback and safety

Before development integration, preserve the original published AHE candidate and isolate all integration work.

Do not rewrite published history.

If integration fails, preserve exact candidate identities and findings.

Any later rollback must use the authorized repository procedure and reconcile affected certification evidence.

No force-push, hook bypass, test suppression or production activation.