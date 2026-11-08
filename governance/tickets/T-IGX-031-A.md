# T-IGX-031-A — TESTS-ONLY: restore `test_sec_ent.py` (HIST03-F-006) and isolate the integration catalog for it and for W5 b2/c/d (HIST03-F-016) (W1-PLAN step 1)

## Scope

TESTS-ONLY batch on top of `origin/development` (`6734971cf`) for two findings of the HIST-03 dossier that share one root cause, measured by the planner at that SHA:

- HIST03-F-006: `tests/unit/runtime/security/test_sec_ent.py` is red, 7 failed / 6 passed. Four independent test-side causes: bare `register_default_integrations()` fails on a clean catalog (slug `ollama` is registered by two providers); the patch target `intergrax.integrations.registry.factory.resolve_from_profile` does not bind where `security_runtime_bridge.py` looks the name up; a plain fake and `object()` are rejected by the pre-built secrets-store type guard; the spine test binds no execution identity.
- HIST03-F-016: the failed bootstrap in `test_sec_ent.py` leaves a half-filled global integration catalog, which makes the W5 b2/c/d tests fail with `Integration slug 'redis' is already registered` when they run in the same process (measured: `test_sec_ent.py` then b2/c/d gives the 8 W5 reds seen on the pin). The W5 files are victims; they pass alone.

Exact files (nothing under `intergrax/`, `agents/`, `applications/`, `scripts/`; ACT-N11 must stay green): `tests/unit/runtime/security/test_sec_ent.py`, `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py`, `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_c_event_export.py`, `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_d_exporter_composition.py`, and one new helper `tests/support/integration_catalog_isolation.py`. Root cause per file, per-assertion before/after ledger with the no-gate-weakening argument, test contract, escape clause, risks and residuals: [`plans/PLAN_T-IGX-031-A.md`](plans/PLAN_T-IGX-031-A.md).

Escape clause: any item whose only fix is a product-code edit leaves this batch and moves to the product batch (W1-PLAN step 3). The plan records one product-level residual (two providers share the slug `ollama`) and does not fix it.

No PASS, closure, FRZ promotion or stage-parent change is implied. The implementer is not the auditor; the independent audit is run by a fresh session, and the plan approver is a third session.

## Plan digest

Plan digest: b17c42288d4bf50dedae540a170904a253fd49b7695a7559b8b16a46b5ec7f5e

## Approved

(pending — no approval line is requested by this ticket's author; any approval line is written only by the Operator or by a commissioned `operator-delegate` session, never by the planner or the implementer.)

Authority cited for the scope (given in chat and in the accepted W1-PLAN sequence, not signature lines): W1-PLAN step 1 (a test-only batch for F-006 / F-016 with an audit proving no gate is weakened; the sequence approves no ticket plan), DR-12 = B (tests now, product edits batched with the next IIAC re-activation).

## Test contract

Phase 2 only, one pytest / ruff / uv process at a time, each under the heavy lock `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock`, pytest's own exit code captured, logs under `.tmp/session/t-igx-031-a/`:

```text
uv run --extra dev-ci pytest tests/unit/runtime/security/test_sec_ent.py -q                       # before (measured): 7 failed, 6 passed; after: 14 passed
uv run --extra dev-ci pytest <the three W5 b2/c/d files> -q                                       # after: 20 passed
uv run --extra dev-ci pytest <test_sec_ent.py + the three W5 files> -q                            # ONE process, isolation proof; before (measured): 15 failed, 18 passed; after: 34 passed
uv run --extra dev-ci pytest <the three W5 files reversed + test_sec_ent.py> -q                   # ONE process, order independence; after: 34 passed
negative probes (uncommitted, restored): restore each broken patch target / the object() input / drop the identity => exactly those tests red again; drop the fixtures => the HEAD red set again
ACT-N11 pre-check: verify_iiac_policy_activation --require-in-force; governance tests incl. test_iiac_g0_inventory; expected green
uv run --extra dev-ci ruff check <5 files> ; uv run --extra dev-ci ruff format --check <5 files>
git diff --name-only 6734971cf..HEAD    # only the five files + ticket/plan files
```

Exact commands, expected counts, the optional 27-path combined run and the stop rules are in section 7 of the plan.

## Out of scope

Everything under `intergrax/`, `agents/`, `applications/`, `scripts/`; the six IIAC material governance paths; `T-IGX-030-B`, `T-IGX-030-C`, the P2-C (F-015) corrective, BND-03, `T-IGX-021-B/C`; `conftest.py`, pytest markers (`no_ci` stays), `pyproject.toml`; selection manifests; ENT-AUDIT-X artefacts; IIAC records and activation; roadmap / FRZ edits and promotions; PASS / closure claims; Docker / vendor / production configuration (C10/C15); push, merge, rebase, PR.

## Lifecycle state

plan_pending
