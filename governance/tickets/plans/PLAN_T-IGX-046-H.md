# PLAN T-IGX-046-H — PT-G01 G2-only: registry bypass gate AST (DP-14)

EC-02/P1 slice. Nothing in this plan claims PASS, FRZ promotion, stage closure, independent verification, or EQO-X-01 closure.

**Operator decision:** OD-EC02-P1 assigns canonical ticket `T-IGX-046-H` for G2_ONLY within master-plan node PT-G01.

**Parent context:** `T-IGX-046-F` on `arch-freeze/rw-train` remains the broader PT-G01 W2 scaffold; `T-IGX-046-F` on `development` is unrelated (DB-C documentation) and is not authority for this work.

**Target branch:** `arch-freeze/rw-train` only for implementation commits.

**Baseline reference (EC-02/P1 authority read):** `origin/arch-freeze/rw-train` `68db9c0ef6207b023388eecc460644698a92b895`.

## 1. Purpose

Replace regex import detection in `scripts/maintenance/check_agent_registry_bypass.py` with AST classification that matches DP-14: Tier-2 `agents/` may import only `intergrax.integrations.contracts` and its submodules; deny other `intergrax.integrations.*` and all `intergrax.tools.*`. Remove false positives on six historical contract-leaf imports at the train baseline. No product-path edits.

## 2. Scope (G2 only)

| In scope | Out of scope (this ticket) |
|---|---|
| AST gate rewrite for `check_agent_registry_bypass.py` | `check_intergrax_no_applications_imports.py` (G1 unchanged) |
| New unit tests with isolated `tmp_path` fixtures | Q-WA-01 fifteen paths outside G1 roots (`OWNER_SCOPED_RESIDUAL`) |
| Preserve CLI and scan root `agents/` | Product paths, `.github/`, FRZ, roadmap, WB05, EQO closure matrix |
| | EBH-3, seam ledger, other PT-G01 slices, EC-02/P2, EC-03 |

## 3. Frozen implementation write set

```text
scripts/maintenance/check_agent_registry_bypass.py
tests/unit/maintenance/test_check_agent_registry_bypass_ast.py
```

No additional paths without a new approved plan.

## 4. Technical contract (gate)

### 4.1 Internal API

```text
check_imports_in_tree(root: Path) -> list[str]
```

- Scan `root/agents/**/*.py`.
- Exclusions: paths with `tests` in `path.parts`; path allowlist `agents/__init__.py` only.
- Return sorted deterministic findings as `relative_path:line: reason` (violations and parse/read failures).
- No file mutation; no subprocesses in the checker.
- CLI: resolve repo root, call `check_imports_in_tree`, print findings, exit `0` iff empty else nonzero (same invocation as today).

Use `ast` for import classification (not regex). Comments and string literals are not imports.

### 4.2 `ast.Import` — each `alias.name` independently

| Import | Result |
|---|---|
| `import intergrax.integrations.contracts` | ALLOW |
| `import intergrax.integrations.contracts.foo` | ALLOW |
| `import intergrax.integrations.providers` | DENY |
| `import intergrax.integrations._shared` | DENY |
| `import intergrax.integrations` | DENY |
| `import intergrax.tools` | DENY |
| `import intergrax.tools.providers` | DENY |
| `import intergrax.integrations.contracts.foo as alias` | ALLOW |

`as alias` does not change module-name policy.

### 4.3 `ast.ImportFrom` — `node.module`

| Import | Result |
|---|---|
| `from intergrax.integrations.contracts import Foo` | ALLOW |
| `from intergrax.integrations.contracts.foo import Bar` | ALLOW |
| `from intergrax.integrations.providers import Foo` | DENY |
| `from intergrax.integrations._shared import Foo` | DENY |
| `from intergrax.integrations import contracts` | DENY |
| `from intergrax import integrations` | DENY |
| `from intergrax.tools import Foo` | DENY |
| `from intergrax import tools` | DENY |

Relative imports: no new policy in this plan (no expanded relative-import interpretation).

### 4.4 Failure semantics

- `SyntaxError` → finding, exit nonzero.
- Other read/parse failures → finding, not PASS.
- Stable sorted `relative_path:line` in CLI output.
- No allowlist of six historical contract-leaf files.
- Zero findings → exit `0`.

### 4.5 Expected outcome

Six historical `intergrax.integrations.contracts.*` imports under `agents/` are no longer violations; true bypass imports remain blocked.

## 5. G1 residual

`scripts/maintenance/check_intergrax_no_applications_imports.py` is not edited. Fifteen product files outside current scan roots remain `OWNER_SCOPED_RESIDUAL` (Q-WA-01, `PLAN_T-IGX-043-G.md` gate notes). G1 maintenance is regression-only in the test contract below.

## 6. Test contract (after `plan_approved` only)

One process at a time; pytest under lock `D:/Projekty/_wt/_locks/pytest.lock`.

**Order:**

1. `uv run pytest tests/unit/maintenance/test_check_agent_registry_bypass_ast.py -p no:cacheprovider` — all §4.2–4.4 cases via `check_imports_in_tree` and `tmp_path` (no subprocess in fixture tests).
2. `uv run pytest tests/unit/agents/test_wa_agent_import_rule.py::test_wa_13_agents_tree_uses_only_integration_contract_leaves tests/unit/agents/test_wa_agent_import_rule.py::test_wa_17_old_gate_pattern_matches_only_contract_leaf_files -p no:cacheprovider`
3. `uv run python scripts/maintenance/check_agent_registry_bypass.py` (expect exit `0` post-change)
4. `uv run python scripts/maintenance/check_intergrax_no_applications_imports.py` (regression; script unchanged)

**Out of scope for new tests:** `intergrax.applications` import negatives (G1 obligation).

**Pre-push:** `uv run python -m scripts.governance.igx_gov ticket` on authorized checkout.

Implementer does not claim independent PASS; Session E / wave audit is separate.

## 7. Acceptance (implementer checkpoint, not G6 PASS)

- Exactly two implementation paths changed (plus governance ticket/plan).
- G2 maintenance exit `0` at implementation SHA.
- New AST tests and WA-13/17 green.
- G1 maintenance still exit `0`.
- No product diff.

## 8. Implementation notes

- Worktree on `arch-freeze/rw-train` per `/igx-worktree`.
- Commits touching `scripts/` include trailer `Ref: governance/tickets/T-IGX-046-H.md`.
- No AI-tool attribution in commits.