# T-IGX-018-A — Governance tests: retire stale pre-activation assertions, stop canonical-file pollution, CI history depth

## Scope

Separate, ticketed governance test hotfix (independent of the RA-B4 proof). Origin: B-4 re-audit findings
HF-01, HF-02, HF-03; specification `iiac_evidence/EVOL-X/b4_reaudit/GOVERNANCE_TEST_HOTFIX_TICKET_SPEC.md`.
None of these defects was introduced by `23feb2bbb`; all four HF-01 tests fail identically at `ca2b14782`
and `9a8fa3634`.

- **HF-01a/b/c** — `test_act_n01_*`, `test_positive_fixture_*`, `test_n12_*` asserted canonical `NOT_IN_FORCE`.
  The canonical policy is now `IN_FORCE`: the negative case runs on a `NOT_IN_FORCE` fixture/copy and the
  canonical record is asserted to verify under `--require-in-force`. Negative strength is preserved
  (inactive policy still fails default and `--require-in-force`; a forged `IN_FORCE` record still fails).
- **HF-01d** — `test_n09_*` globbed for pass-record filenames. Replaced by a content guard: each pass record must
  be bound by a manifest that passes the manifest and role-separation validators (or be its byte-identical
  auditor copy); unbound, invalid-manifest and tampered-copy cases are tested to fail.
- **HF-02** — `test_n05_*` / `test_n08_*` overwrote canonical files and restored them with `write_text`
  (LF→CRLF on Windows), leaving the `-text` activation record different from its blob. They now use temporary
  copies; a module-scoped guard asserts the canonical activation record is byte- and hash-identical afterwards.
- **HF-03** — `governance-fast` checks out with `fetch-depth: 0` (G0 inventory and activation checks need ancestry).

Not changed: IIAC activation gate, `iiac_common`, manifest / role validators, protocol, any evidence.

## Plan digest

Source: operator mission "INTEGRAX — EVOL-X RA-B4-01..04 Consolidated Final Corrective Implementation" section 9 @ `9a8fa3634713f69b56ce87ceaf9ed319fad27dae`

## Approved

Approved: operator @ 2026-11-06 (mission section 9: separate ticketed governance test hotfix prepared by the implementer; independent review required)

## Test contract

```powershell
uv run pytest tests/unit/governance -q
git status --short   # expected: clean apart from this change
git hash-object --no-filters docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json   # == HEAD blob
```

## Out of scope

- EVOL-X preflight (see T-IGX-017-B), Session C and P2 evidence, product paths
- Weakening or deleting any negative guard

## Lifecycle state

ready_for_audit

## Checkpoint

Prepared as a separate commit from the RA-B4 proof. Independent review by a session other than the implementer is required.
