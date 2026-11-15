# PLAN T-IGX-040-D — PDC-X finding F-35: classify the evidence-plane security audit trail as a COMPAT-X persistence site

Parent: PDC-X campaign, closing wave. Origin: finding F-35 (row F-35 of `PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_1.md`; `OD-MCP-RC-04` section 2 assigns the repair to the PDC-X owner on `development`). This plan is written by the implementer lane and is **unsigned** (`plan_pending`). It is not a certification verdict and certifies nothing.

## Problem

Ten COMPAT-X event-evolution gates are red on `development` (`61531828f`): four in `tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py` and six in `tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py`. They are green at the pin and red since PDC-X, because T-IGX-036-P (B-D6, PDC-D6-R020, commit `adaaa374e`) added `EvidencePlaneSecurityAuditTrail` to `intergrax/runtime/security/security_audit_trail.py`. The gates' AST discovery reports two sites in that class that the gate classifiers do not know:

- P0 persistence discovery: `security_audit_trail.py:213` `self._evidence.append(event, tenant_id=tenant)` → `unclassified persistence append …`, `persistence discovery includes UNCLASSIFIED append call sites`, `canonical production persistence bypass count 1 (expected 0)` (and the dependent probe K).
- R3-R1 constructor discovery: `security_audit_trail.py:198` `RuntimeEvent(...)` → `unclassified runtime event reader … append RuntimeEvent(...)`, `inventory persistence call sites include canonical production bypasses`.

(Seven further COMPAT-X gates that are red at the pin as well are not part of this finding and not touched.)

## Diagnosis: legitimate classified site, not a bypass

Read at `development` tip `61531828f` (no product file is changed by this plan):

1. **It is the canonical evidence plane, not a raw store.** The module imports only the contract `EvidencePersistencePort` (`intergrax/contracts/execution_evidence/persistence_port.py`); it imports no sqlite, memory, document or other backend store and calls no store `_inner`. The class docstring states "no security-owned store" and `test_d6_r020_entries_use_the_canonical_evidence_store_not_a_security_store` pins that the schema contains no security/audit table.
2. **The write passes through the platform validating boundary.** The port is injected. Canonical composition (`as_evidence_persistence_port` in `evidence_persistence_adapter.py`, `resolve_runtime_event_persistence`) wraps every port in `ValidatingEvidencePersistencePort` (prepare + strict schema validation) unless it already carries `CanonicalRuntimeEventWriteValidatedPort`. The D6 tests build the trail on exactly that composed port (`open_evidence_plane` → `resolve_runtime_event_persistence`), and the append succeeds under it.
3. **The payload is a registered typed schema.** `SecurityAuditPayloadV1` (`security.audit_entry.v1`) is registered with `register_payload_schema(..., extension=True)`; the event is `DOMAIN_SIGNAL` with `event_kind="security.audit_entry"` and `payload=payload.to_envelope()`, so the validating port accepts it under the extension contract (no ad-hoc payload shape).
4. **Same pattern as an already classified site.** `GovernanceEvidencePersistence` (`runtime/governance/governance_evidence_persistence.py:116`, `self.evidence_persistence.append(event, tenant_id=...)`) is classified `CANONICAL_VALIDATING_PATH` today with the same structure (injected port, event built from a typed fact).
5. **The constructor at :198 is a producer, not a persisted reconstruction.** `RuntimeEvent(...)` is built from live arguments (`validate_event_id(f"evt_{suffix}")`, the tenant stream, `datetime.now(UTC)`, `payload.to_envelope()`): no `**raw`/`**data` splat, no `json.loads`, no wire-derived `schema_version`, so `runtime_event_constructor_looks_like_persisted_reconstruction` is false.
6. **Reader side.** `list_entries` reads through the same port and decodes with `SecurityAuditPayloadV1.from_envelope`; the gates do not flag it (it constructs `SecurityAuditEntry`, not `RuntimeEvent`).

Conclusion: not a production persistence bypass and not a product defect; the gates' site lists simply predate the new class. **Residual observation (not a defect of this plan, no product change here):** the module itself accepts any `EvidencePersistencePort`, so the validating guarantee rests on the composition root (as for governance persistence). A runtime assertion of the `CanonicalRuntimeEventWriteValidatedPort` marker in the trail or its wiring would be a product hardening and would be its own finding; it is out of scope.

## Deliverable

Test-side classification only, additive and as narrow as the existing rules (no gate weakened or deleted):

- `tests/qualification/compat_x/_compat_x_event_evolution_inventory.py`: new evidence member `SECURITY_AUDIT_TRAIL_EVIDENCE_PLANE_PORT`, added to `_CANONICAL_VALIDATING_PATH_EVIDENCE`; one rule in `classify_runtime_event_append_call_site` that requires module suffix `runtime/security/security_audit_trail.py` **and** enclosing function `append` **and** callee `self._evidence.append`, giving `CANONICAL_VALIDATING_PATH` with that evidence.
- `tests/qualification/compat_x/_compat_x_event_evolution_r3_r1.py`: new evidence member `SECURITY_AUDIT_TRAIL_EVENT_RECORDER`, added to `_NON_PERSISTED_CONSTRUCTION_EVIDENCE`; one entry in `_CONSTRUCTOR_POSITIVE_EVIDENCE_RULES` binding (`runtime/security/security_audit_trail.py`, `append`) — module-wide trust stays 0, the rule matches exactly one site (no stale rule), and a persisted-reconstruction shape in that function is still `UNCLASSIFIED`.
- Hook-mandated normalization of `_compat_x_event_evolution_r3_r1.py` (the pre-commit hook lints every touched file and the file already carried debt): `ruff check --fix` import order (I001), `# noqa: UP042` on the two existing `(str, Enum)` classes (their base is kept, no `StrEnum` switch) and a `ruff format` pass. No semantic change: only line re-wrapping, import order and two `noqa` comments beyond the rule additions. Checked by comparing the parsed syntax trees of the file before and after: the import sets are equal and the only tree differences are the three additions (enum member, frozenset member, rule entry).
- New pinned tests, written before the code change (red run recorded under `.tmp/proof/pdc-x-f56/`): `tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py`. Positive: the real append and constructor sites are discovered and classified with the typed evidence; bypass count 0; rule integrity clean. Negative: other function / other callee (`self._store.append`, `self._inner.append`) / other module with the same names stay `UNCLASSIFIED`; a suspicious reconstruction in the classified function stays `UNCLASSIFIED`; an injected `UNCLASSIFIED` append site still fails the P0 gate and an injected unclassified constructor site still fails the R3-R1 gate.

## Write set

Exactly: the two helper files and the new test file above, plus the governance documents of this batch (`governance/tickets/T-IGX-040-D.md`, this plan). Not edited: any product path (`intergrax/`, `agents/`, `applications/`), the two gate test files `test_compat_x_event_evolution_r3_p0_gates.py` and `test_compat_x_event_evolution_r3_r1_gates.py` (byte-identical), `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the IIAC activation record, EBH-7 pins and allowlist.

## Gate-semantics note and signing

Adding a classification narrows what the gates report (one site class becomes accepted) and is therefore reviewed like a gate change. This plan is **not** covered by any `DELEGATION-PDC-*` record; signing returns to the Operator via the orchestrator. The implementer never writes an approval or delegation line and never merges.

**What the signer should check:** (a) the six diagnosis points above against the code (especially points 2 and 3: composition through the validating port, registered payload); (b) that each new rule is bound to exact module + function (+ callee for the append) and adds no module-wide trust; (c) that the two gate test files are unchanged and the negative tests exist; (d) the residual observation is accepted as out of scope; (e) the normalization of `_compat_x_event_evolution_r3_r1.py` is line re-wrapping, import order and `noqa` only (the syntax trees differ by the three additions only; the signer can repeat the comparison).

## Ordering and constraints

1. Local commit only until the Operator signs; then push and PR against `development` (separate step, not in this plan).
2. The change lies under `tests/`, which is not on the allowlist of the EBH-7 executable-identity gate; the gate stays red until the campaign's final EBH-7 re-pin (accepted by the orchestrator). No `MATERIAL_GOVERNANCE_PATHS` entry is touched.
3. No new registry, scheduler, engine, trust, policy or audit authority.

## Protocol

1. One `uv run pytest` per verification step, several paths in one command; log under `.tmp/proof/pdc-x-f56/`.
2. Verification: `uv run pytest tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py tests/qualification/pdc_x/d6/test_d6_audit_trail_durable.py -q`; ruff check and format check on the new test and the inventory helper; pyright on the touched files (pre-existing diagnostics recorded, not fixed).
3. Commit with exactly one `Ref: governance/tickets/T-IGX-040-D.md` trailer and no AI attribution.

## Acceptance

- The ten gates (four P0, six R3-R1) pass; the D6 audit trail tests stay green; the seven gates red at the pin remain as they were.
- The new tests pass, including all negatives; before the change the positive tests are red.
- `git diff --stat` shows only the write set; the two gate test files are byte-identical.

## Out of scope

- Product hardening of the audit trail (marker assertion), the seven gates red at the pin, any other finding, EBH-7 re-pin, IIAC audit, activation or certification.
