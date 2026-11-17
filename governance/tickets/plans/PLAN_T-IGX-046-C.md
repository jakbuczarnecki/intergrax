# PLAN T-IGX-046-C — EQO-A remediation node PT-D12: seam canon documents

Status: DRAFT for Operator review. Nothing here is approved, applied or audited. Implemented as a draft pull request from branch `docs/eqo-a-pt-d12-seam-canon`.

## 1. Purpose

Record, in the canon and owner documents, what the Operator decided in the EQO-A remediation planning for the layer-boundary seams (decisions DP-13, DP-16, DP-18, DP-19) and relabel the scope of earlier CTRL-X, GOV-X2 and HARNESS-FINAL claims about the registry-bypass gate (finding EQOA-A3-016). The texts describe the target state decided by the Operator; the code and gate changes are not part of this ticket and are implemented by the code wave. This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no product path.

## 2. Scope (5 files, additive dated notes only)

| File | Decision | Change |
|---|---|---|
| `docs/project/architecture/PLATFORM_FOUNDATION.md` | DP-13 | One dated note after the CI-guard list of the dependency-direction enforcement paragraph; the canon text (the lower-layer rule and "no grandfather exceptions") is not edited |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | DP-13, DP-16, A3-016 | New subsection 5.7 with three dated notes; the invariant rows and the CI map are not edited |
| `docs/project/architecture/LLM_ADAPTERS.md` | DP-16 | One paragraph in Provider selection naming the Tier-1 `runtime/wiring` resolver as canonical host LLM resolver, Tier-3 delegating |
| `docs/project/architecture/INTEGRATIONS.md` | DP-18 | New subsection stating the closed-list rule; the list itself is not written (no Operator list exists); inventory counts at the head |
| `docs/project/architecture/APPLICATION_HOSTING.md` | DP-19 | New subsection declaring direct `uvicorn.run` entrypoints non-production, with the verified call-site and bind-default facts |

Every statement of present fact carries the head SHA it was read at and was re-checked against `origin/development` `2ab57e36d`; statements about the audit baseline are attributed to the finding.

## 3. Explicitly out of scope

- DP-14, DP-15, DP-17, DP-20, DP-21, DP-22 (other nodes of the plan).
- The named list of provider composition roots (DP-18): no source names it; left to the seam ledger gate batch and an Operator list.
- Roadmap, FRZ checklist, qualification records (including the CTRL-X record itself), `DOCUMENTATION_MAP.md`, any file under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`.

## 4. Write set

```text
docs/project/architecture/APPLICATION_HOSTING.md
docs/project/architecture/INTEGRATIONS.md
docs/project/architecture/LLM_ADAPTERS.md
docs/project/architecture/PLATFORM_FOUNDATION.md
docs/project/technical/guides/SYSTEM_INVARIANTS.md
```

Plus this plan and the ticket under `governance/tickets/`. None of the five is an input of the EBH-7 B6-F10 canonical inputs (`tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`): the inputs read the roadmap, the FRZ checklist, `AGENTS.md`, qualification matrices and the owner documents of the RF-* bindings, which are other files. Like other `docs/project/architecture/` files they are outside the identity allowlist of `verify_ebh7_p03_immutable_evidence.py`, so the PR has to merge before the final EBH-7 pin (DP-10, DP-61).

## 5. Method

Worktree outside the repository from `origin/development`; notes added by exact-anchor insertion, existing text untouched; relative links checked; commit with the `Ref:` trailer and no tool attribution.

## 6. Test contract

See the ticket. One process at a time under the shared lock.

## 7. Risks

- `PLATFORM_FOUNDATION.md` and `SYSTEM_INVARIANTS.md` are Operator-owned canon: the Operator must read the notes; they are additive and dated.
- The DP-19 note makes the production-serving status of two production-named commands explicit; wording was kept to what the code shows.
- Rollback: revert the PR; no code, data or configuration is affected.

## 8. Closure condition

PR merged after review. Finding EQOA-A3-016 is covered by the relabel note only; EQOA-A4-003, A4-016, A4-020 and A4-040 are not closed by this ticket (their code and gate work is separate).
