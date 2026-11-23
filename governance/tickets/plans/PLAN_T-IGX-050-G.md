# PLAN T-IGX-050-G — ARCH-FREEZE W2 node T-AF2-MANIFESTS (draft): contract, layer, ownership and composition manifests

Status: DRAFT for Operator review. Drafted by the implementing session. Nothing here is approved, applied or audited.

## 1. Purpose

The freeze needs four manifests (FRZ-FRZ-05 contract, FRZ-FRZ-06 layer, FRZ-FRZ-07 ownership, FRZ-FRZ-08 composition), and the plan records that no definition, generator or owner of them exists (Operator decision DP-40). This batch gives the RW product train and the W5 final generation a reviewed shape: it defines each artefact, generates a first draft of each from the tree at `origin/development` with a stdlib-only generator, and marks the entries that the plan section E dispositions and OD-AF-18 call `NOT_COMPOSED` or `NOT_CLAIMED`, and the neutral contracts of BND-CLOSURE as `PLANNED_MOVE (PT-WA)`. It freezes nothing, certifies nothing, promotes no FRZ criterion and edits no product path. The final manifests are generated from the G2 tree in W5 under T-IGX-052-B.

## 2. Scope

| Item | Path | Change |
|---|---|---|
| Manifests | `docs/project/maintainers/qualification/arch_freeze_program/manifests/{layer,contract,ownership,composition}_manifest.draft.json` | new; schema ids `af2.manifest.<kind>.v0-draft`, status `DRAFT` in every `meta` block |
| Generated summary | `docs/project/maintainers/qualification/arch_freeze_program/manifests/SUMMARY.md` | new; counts, generator SHA-256 |
| Folder README | `docs/project/maintainers/qualification/arch_freeze_program/manifests/README.md` | new; DRAFT label, mark vocabulary, what the draft surfaces, not-included list |
| Generator listing | `docs/project/maintainers/qualification/arch_freeze_program/manifests/GENERATOR_REFERENCE_LISTING.py.txt` | new; byte-identical copy of the generator kept under `.tmp/session/arch-freeze/manifests/`; a reference listing under `docs/` only, not under `scripts/` |
| Ticket and plan | `governance/tickets/T-IGX-050-G.md`, this file | new |

## 3. Method

1. The generator (`python -I`, stdlib only) lists tracked files with `git ls-files` and parses them with the AST. It imports and executes nothing from the tree. It writes no timestamp, so two runs give identical bytes.
2. Layer: tier by path rule from the AGENTS.md tier line; forbidden edges under R1 (intergrax must not import agents or applications), R2 (agents must not import applications) and R4 (the registry-bypass rule of `check_agent_registry_bypass.py`).
3. Contract: modules of `intergrax/contracts/` and `intergrax/applications/contracts/`; mechanical public/internal rule; `PLANNED_MOVE (PT-WA)` entries from closed-world `RESULT.md` section 1.4 B, each checked against the tree.
4. Ownership and composition: values read as AST literals from the CONFIG-X inventory, the EBH-5 platform concern register, the fixed-substrate decision register and the strict-production-roots table; every path they name is checked against the tree.
5. Marks come from the quoted `ARCH_FREEZE_IMPACT` column of `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md` Table A and from OD-AF-18. A row whose text maps to no mark is listed without one.
6. Every added factual sentence is checked against `origin/development` and recorded as `path@sha:line` in `FACT_CHECK.md` (kept under `.tmp/`, not committed).
7. The committed listing is run once more from the PR branch's copy against the baseline checkout; its output must equal the committed files byte for byte.

## 4. Out of scope

Any change under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`; `pyproject.toml`, `uv.lock`; the roadmap and the FRZ checklist; every `IIAC_*` file and script; the PDC-X record directories; a drift gate; non-composition gates; naming owners the plan does not name; a reviewed public/internal contract split; the final generation from the G2 tree; merging, approval and certification of any kind.

## 5. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/arch_freeze_program/manifests/GENERATOR_REFERENCE_LISTING.py.txt
docs/project/maintainers/qualification/arch_freeze_program/manifests/README.md
docs/project/maintainers/qualification/arch_freeze_program/manifests/SUMMARY.md
docs/project/maintainers/qualification/arch_freeze_program/manifests/composition_manifest.draft.json
docs/project/maintainers/qualification/arch_freeze_program/manifests/contract_manifest.draft.json
docs/project/maintainers/qualification/arch_freeze_program/manifests/layer_manifest.draft.json
docs/project/maintainers/qualification/arch_freeze_program/manifests/ownership_manifest.draft.json
governance/tickets/T-IGX-050-G.md
governance/tickets/plans/PLAN_T-IGX-050-G.md
```

## 6. Verification plan

One process at a time under the shared lock `D:/Projekty/_wt/_locks/pytest.lock`:

- `python -I scripts/governance/verify_arch_freeze_plan_traceability.py` (the plan validator)
- `python -I scripts/docs/check_docs_domain_pairs.py` (docs pairs)
- `python -I scripts/governance/verify_ticket_lifecycle.py governance/tickets/T-IGX-050-G.md`
- the generator rerun against the baseline checkout and a byte comparison with the committed files (a plain Python process, no pytest)

## 7. Risks

- The manifests describe the tree at the baseline commit; later commits on `development` change counts. The W5 generation from G2 replaces them.
- The contract public/internal split is a path rule, not a reviewed list, so `DRAFT_PUBLIC_CANDIDATE` over-approximates the public surface.
- The `PLANNED_MOVE` set is a closed-world proposal (the binding scope is T-AF2-RW-PLAN). One entry, `ExecutionMode`, is already defined at Tier-0 at the baseline commit; the draft says so.
- A `NOT_COMPOSED` mark without its gate would be an unsupported claim; every gate is `PENDING` and the README says so.
