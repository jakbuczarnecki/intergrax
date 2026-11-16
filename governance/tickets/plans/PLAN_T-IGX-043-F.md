# PLAN T-IGX-043-F — EQO-A remediation: CONFIG-X semantic scan extension for vendor-slug defaults (gate-meaning change)

Status: DRAFT (revision 3: the Operator's ruling of 2026-11-16 is recorded in section 14 from `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a transcription by the orchestrating session of the Operator's chat answers of 2026-11-16, a session artefact, not yet a repository record); the approval line of this ticket is still the Operator's own and is not written; revision 2 after `FC/REVIEW/FC_DELTA_REVIEW_1.md`, section 5: the LAB exemption mechanism, the source of the vendor values, the false-positive list and an honest NT-FC3-CX7 expectation are defined in sections 3.1 to 3.4; the scratch measurement has its own authorization entry in `AUTHORIZATION_REQUESTS_T-IGX-043-F.md`). Drafted 2026-11-16 by the planning session of `T-IGX-043-C` after the independent delegate review (`FC/REVIEW/FC_DELEGATE_REVIEW.md`, section 4 check 3, finding X-5) found that Part G of FC-3 is a gate-meaning change. Read-only planning: no branch, no worktree, no commit, no test run. Nothing here is approved, applied or audited. This ticket is **not approvable by a delegate** and no implementer or delegate writes its approval line.

Notation. `B` = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, `H` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42`; `path@B:line` means the full SHA (FACT_CHECK.md spells them out). All files named below are `UNCHANGED` between B and H (`git diff --stat B H -- tests/qualification/config_x` is empty).

## 1. Purpose and classification

Purpose: make the CONFIG-X semantic production scan able to see silent vendor/backend slug defaults (finding EQOA-A4-019 and its blind-spot statement; FRZ-CFG-02, FRZ-CFG-06 and FRZ-CFG-08 are PASS under CONFIG-X at `fac9a43f` per `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md@H:211,215`; FRZ-CFG-08 claims that gates prevent reintroduction).

Classification: **gate-meaning change.** Today `_classify_resolved_semantic_string_default` returns `APPROVED_TYPED_CONFIGURATION` for every non-empty string except tenant `default` and localhost URLs (`tests/qualification/config_x/_config_x_semantic_production_scan.py@B:282-299`). After this ticket a string or enum-member default that names a registered Integrations catalog slug or a known vendor value on a semantic field becomes `HARD_CODED_SEMANTIC_PRODUCTION_SELECTION`, and the scan reaches code shapes it did not read before. That changes what the gate reports about FRZ-CFG-02/06/08, so under the standing rule gate-meaning changes are personal to the Operator (`WP-12/AUTHORIZATION_REQUESTS.md` line 7; `PT-G01`, `PT-PB`). A delegate or commissioned session may not approve this ticket; the approval line, if the Operator rules yes, is the Operator's own.

Decisions applied: OD-REM-03, C3 (negative tests, regression, independent audit at the exact SHA), C6; DP-10 (a gate that is red on day one lands only together with its fixes); DP-06 (a contradicted PASS row keeps PASS with an annotation and counts as BLOCKS in the entry check); DP-11 (one wave).

## 2. Dependency order and prerequisites

- Depends on `T-IGX-043-C` (the product fixes). The repo-level test of this ticket (NT-FC3-CX7) is green only after the selectors in C no longer default, so this ticket merges after C is merged, or in the same wave behind C with the Operator's ruling; never before C (DP-10). C does not depend on this ticket for its own correctness (plan C, section 3.7).
- Prerequisites: the Operator's ruling is recorded (section 14); the approval line is not written; milestone M1 and the wave scope as for the other wave tickets (whether PDC P9 is formally closed is not decided); the scratch measurement F-A2 is authorised and not yet run, and runs before the approval line so that the line is written on known consequences.

## 3. Design

Blind spots at B, all in `tests/qualification/config_x/_config_x_semantic_production_scan.py`:

1. `_classify_resolved_semantic_string_default@B:282-299` approves every non-empty string except tenant `default` and localhost URLs, so `"ollama"` is `APPROVED_TYPED_CONFIGURATION`.
2. `_is_environ_get_call@B:259-266` matches only `os.environ.get(...)`; `embedding_profile_from_env` uses `os.getenv(...)` (`intergrax/rag/embedding/registry/profile.py@B:54`).
3. `_from_env_semantic_constant_fallbacks@B:399-451` inspects only class methods literally named `from_env`; the FC-3 selectors are module-level functions (`resolve_notification_settings`, `resolve_inbound_verifier_settings`, `embedding_profile_from_env`).
4. `_resolve_string_constant@B:243-256` resolves only `Constant` and `Name`; `NotificationBackend.LOG.value` and enum-member defaults (`backend: NotificationBackend = NotificationBackend.LOG`, `mode: InboundVerifierMode = InboundVerifierMode.NONE`) return nothing.
5. `_SEMANTIC_PARAM_NAMES@B:44-59` lacks `mode`, `slug`, `store`, `channel`.

Change: resolve attribute access to enum members; handle `os.getenv` and `os.environ.get` (second argument and `or` fallback of the read) in any function; scan class and dataclass field defaults and parameter defaults on the extended name set; extend the name set by `mode`, `slug`, `store`, `channel` (not by model-name identifiers, section 3.4); classify a string or enum default equal to a registered catalog slug or a known vendor value (section 3.2) as `HARD_CODED_SEMANTIC_PRODUCTION_SELECTION` unless it is LAB-guarded in the syntactic sense of section 3.1. Plain `x or "literal"` fallbacks that do not read the environment are not scanned. The inventory (`_config_x_concern_inventory.py`) and `_config_x_current_classification.py` are not edited, so the closed-world counts of `test_cx_q01/q02/q16` stay unchanged; if the implementation proves otherwise, the ticket and this plan are amended and the digest recomputed before the edit. The `llm.embedding` class-A entry (`_config_x_concern_inventory.py@B:53-60`) stays; its contradiction is carried by the new gate and the DP-06 annotation.

### 3.1 LAB exemption mechanism (syntactic; no module or path exemption)

A default expression E (a string constant, an enum member, the `.value` of an enum member, or a name that resolves to a module string constant) is LAB-guarded if and only if both hold: (i) the enclosing function has a parameter named `lab_opt_in` annotated `LabSelectorOptIn | None` (or `Optional[LabSelectorOptIn]`); (ii) E sits inside the body branch of an `ast.If` or `ast.IfExp`, or the right operand of an `and`, whose test is `lab_opt_in`, `lab_opt_in is not None` or `not (lab_opt_in is None)`; the `else` branch is not exempt. A module-level string constant (the LAB value `ollama`) is exempt if and only if its name starts with `_LAB_`, it is not listed in `__all__`, and every load of it in the same module is LAB-guarded as above; a load outside a guard is flagged, and an import of the constant by another module is flagged because each module is scanned alone. Class and dataclass field defaults and parameter defaults are never LAB-guarded. No exemption exists by module path, module name, class name or decorator: a module named `lab_*` or placed under a `lab/` directory gets none, because that would be the provider allowlist that R3 forbids.

### 3.2 Sources of the vendor values

"Registered catalog slug" is `list_slugs()` of `intergrax.integrations.registry.catalog` (`catalog.py@B:143-151`, keyword-only `category`, no argument returns every registered slug) evaluated in the new test module after `register_default_integrations(preset="full")` (the pattern of `rag/embedding/bootstrap/default_embedding_engine.py@B:26-31`). "Known vendor value" is the set of `.value` of the selector enums imported by the test module: `NotificationBackend` (`runtime/notifications/backend_contract.py@B:18`), `InboundVerifierMode` (`runtime/interactions/verification/factory.py@B:29`) and `VisionProvider` (`intergrax/model_inference/registry/vision_provider.py@B:10`). Both sets are derived, not hand-written; they change when the catalog or the enums change.

### 3.3 False-positive risk (to be sized by the scratch measurement)

The values `log`, `none`, `stub`, `inmemory` and `auto` are catalog slugs or enum values and also ordinary words. Expected false positives: `mode: str = "none"` or `channel = "log"` parameters in unrelated modules (message labels, retry or compression modes); `backend: str = "inmemory"` in lab and `testing_support` modules, which `_EXCLUDE_PATH_PARTS` (`tests/qualification/config_x/_config_x_semantic_production_scan.py@B:24-34`) does not exclude (it excludes `/tests/`, `/qualification/`, `/examples/`, `/scaffold/` and `docker/runtime-context` only); `provider`/`backend` defaults in fixtures shipped inside `intergrax/`. These values are also exactly the silent downgrades that FC-3 removes (`log`, `none`, `inmemory`, `stub`), so they are targets, not noise, when they are the default of a selector. The measurement lists every hit per path; the Operator decides per hit (section 14, item 3). The coordinator label `notify_channel or "log"` is a plain `or` fallback and is not scanned.

### 3.4 What NT-FC3-CX7 can honestly assert after `T-IGX-043-C`

C keeps these vendor literals by decision: `triton_vision.py@B:46` (`... or "yolo"`, an attribute assignment, the model name), the Hugging Face `DEFAULT_MODEL = "facebook/detr-resnet-50"` (`huggingface_inference_vision.py@B:31`, FRZ-CFG-05, out of scope in C) and the LAB constant `ollama` (kept as `_LAB_*`, guarded). With the name set of section 3 (no `model_name`, `model_id`, `DEFAULT_MODEL`) the extended scan does not report the first two, so the gate does **not** cover vendor model-name literals; that is a documented gap tied to FRZ-CFG-05 and not a claim of the gate. NT-FC3-CX7 therefore asserts equality of the findings over the eight paths with an explicit expected list (`EXPECTED_RESIDUAL`), initially empty for the selector defaults of C; if the scratch measurement or a later widening of the name set shows residuals, the list is filled with each residual named and justified through a ticket amendment (digest recomputed), never a silent empty and never a loosened assertion. NT-FC3-CX7 can stay red after C if a residual exists that is neither fixed nor listed; that is the intended signal.

## 4. Write set

Tests only; none of these paths is in `PRODUCT_PATH_PREFIXES` (`scripts/governance/iiac_common.py@H:22`), so the change is **not ACT-N11 sensitive**. They are not on the EBH-7 executable-identity allowlist (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py@H:20-37`), so they merge before the final EBH-7 pin (DP-61).

```text
tests/qualification/config_x/_config_x_semantic_production_scan.py          (edit)
tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py      (new)
```

Forbidden: every product path (`intergrax/`, `agents/`, `applications/`), `scripts/`, `.github/`, `docs/`, `governance/` except this ticket and plan, the other files under `tests/qualification/config_x/` (inventory, classification, existing test files) unless an amendment is approved, `scripts/maintenance/check_*` and `tests/unit/architecture` (`PT-G01`), and every file of `T-IGX-043-A/B/C/D`.

## 5. Negative tests (new; red at `B`, green at the wave SHA that contains `T-IGX-043-C`)

| ID | Exercises | Expected result |
|---|---|---|
| NT-FC3-CX1 | synthetic module with `os.getenv("X_PROVIDER", "ollama")` | `HARD_CODED_SEMANTIC_PRODUCTION_SELECTION` |
| NT-FC3-CX2 | synthetic `backend: NotificationBackend = NotificationBackend.LOG` | flagged |
| NT-FC3-CX3 | synthetic `os.environ.get(ENV, Mode.NONE.value)` | flagged |
| NT-FC3-CX4 | synthetic module-level `resolve_*_settings` function | scanned (not only `from_env`) |
| NT-FC3-CX5 | synthetic parameter named `mode` with a vendor default | flagged |
| NT-FC3-CX6 | catalog-slug equality | a default equal to a registered slug is flagged, a non-slug string is not |
| NT-FC3-CX7 | repo-level scan over the eight FC-3 paths (`provider_authority.py`, `profile.py`, `resolver.py`, `integration_vectorstore.py`, `backend_contract.py`, `verification/factory.py`, `llm_guardrail/_factory.py`, `triton_vision.py`/`huggingface_inference_vision.py`) | findings at `B`; with `T-IGX-043-C` in the tree the findings equal the explicit list `EXPECTED_RESIDUAL` (initially empty; section 3.4) |
| NT-FC3-CX8 | mutation: re-introduce a default in a synthetic copy of `profile.py` | the gate fails |
| NT-FC3-CX9 | a `_LAB_*` constant loaded only inside `if lab_opt_in is not None:` of a function with the typed parameter | not flagged (positive control) |
| NT-FC3-CX10 | the same default outside the guard | flagged |
| NT-FC3-CX11 | the `_LAB_*` constant also loaded outside the guard, or imported by another module | flagged |
| NT-FC3-CX12 | an unguarded default in a module named `lab_*` or under a `lab/` path | flagged (no module or path exemption) |

Count: 12 negative tests (CX1..CX12). Synthetic snippets follow the style of `test_unresolved_named_constant_semantic_default_is_blind_spot` (`tests/qualification/config_x/test_config_x_r1_r1_r1_semantic_constant_resolution.py@B:52`).

## 6. Regression scope

The 11 existing CONFIG-X node IDs of `regression_nodes_F.txt` (verified to exist at B and at H): `test_frz_cfg_02`, `_05`, `_06`, `_08` (`test_config_x_final_reconciliation_gates.py`), `test_cx_q01`, `q16`, `q18` (`test_config_x_qualification_gates.py`), the three constant-resolution tests and `test_cx_a_missing_required_integration_config_fails_closed`. They must stay green with the extended scan; a red result outside the eight FC-3 paths means the extended scan found a real or false hit elsewhere (section 14, item 3).

## 7. Test commands (one process at a time; none run when drafted)

1. `uv run pytest tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py -p no:cacheprovider` (red at `B`, green with C in the tree).
2. `uv run pytest tests/qualification/config_x -p no:cacheprovider` (whole directory: seven files).
3. `uv run pytest <11 node IDs from regression_nodes_F.txt> -p no:cacheprovider`.
4. `uv run python -m scripts.governance.igx_gov ticket` before push.

## 8. Order of work

1. Scratch measurement F-A2 (authorised, not yet run), then the Operator's approval line (his own). 2. Branch from the wave integration branch that already contains `T-IGX-043-C`. 3. Write the new tests red-first. 4. Extend the scan. 5. Run commands 1 to 3. 6. Hand off to the independent audit.

## 9. ACT-N11 / IIAC / EBH-7

Not a product-path change; no IIAC rebind is caused by this ticket. It must merge before the final EBH-7 pin. No overlap with `PT-G01` or `PT-T01` (they own `scripts/maintenance/check_*`, `governance/architecture`, `tests/unit/architecture`); A4-025 (Campaign B-2 probe on `tests/qualification/config_x`) is not claimed.

## 10. ARCH-FREEZE impact

A4-019 (BLOCKS, effective): the closure condition "the CONFIG-X scan flags the vendor-slug defaults and is green after the fix" is met only by this ticket plus `T-IGX-043-C`; if the Operator rules this ticket out, the condition is not met by C alone and must be restated by the Operator. FRZ-CFG-02/06/08 stay PASS with an annotation (DP-06); nothing here changes a status, and no sentence claims closure.

## 11. Independent audit

At the exact wave SHA by a separate non-implementing session (the gate-meaning ruling itself is the Operator's, not the auditor's): replay NT-FC3-CX1..8 at `B` (red where stated) and at the wave SHA (green); run the mutation of NT-FC3-CX8; confirm the 11 regression nodes; compare the set of scan findings before and after the change over the whole inventory and list every new hit with its path (section 14, item 3). Label `PASS_SCOPED` only for the scope executed; the implementer records no PASS.

## 12. Risks

R1 The extended scan may report hits outside the eight FC-3 paths (count unknown, UNVERIFIED); each is either a real silent default (a new finding, not fixable inside this ticket) or a false positive (needs a classifier refinement). Resolved by the ruling (section 14, item 3): only hits on the eight paths fail the gate, each hit elsewhere is recorded as a finding and does not fail it; the verdict function for paths outside the eight is unchanged. R2 `test_frz_cfg_08_regression_protects_current_closed_world_classification` pins the current classification; the change must keep it valid or the ticket is amended. R3 A gate that can be satisfied by allowlisting providers would defeat its purpose; no allowlist of providers is part of this ticket. R4 Ordering: merging before C leaves the repository red.

## 13. Rollback

Revert the two-file commit: tests only, no product effect, no IIAC consequence. The old gate verdict function returns, together with the blind spot (A4-019 closure condition reopens).

## 14. The Operator's ruling (decided; transcription of 2026-11-16) and what stays open

Source: `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a transcription by the orchestrating session of the Operator's chat answers of 2026-11-16, a session artefact, not yet a repository record). The Operator's messages quoted there: "Akceptuję rekomendowane odpowiedzi na wszystkie pytania FC-1..4", "Akceptuję rekomendowane odpowiedzi na wszystkie pozycje". This session did not see an Operator-signed record.

1. Extension of the CONFIG-X scan as a gate-meaning change: **yes, scoped to the eight FC-3 paths** (not wider).
2. Scratch measurement F-A2: **authorised** (non-certifying, one process, read-only); **not yet run**; it runs before the approval line. It is not run by this draft.
3. Hits outside the eight FC-3 paths: **option (a)**: each is recorded as a finding and the gate stays scoped to the FC-3 paths; no allowlist of providers (R3). Design consequence: the extended rules are evaluated over the inventory, but only findings on the eight paths (and on the expected-residual list of section 3.4) fail the test; hits elsewhere are written to a record (a list in the test output and one finding per hit in the register), which does not fail the gate.
4. Timing: **same wave behind `T-IGX-043-C`** (merge order B, A, C, F).

Still open and not decided by that file: the approval line of this ticket (the Operator's own, not written), milestone M1 (PDC P9), the test authorisations other than F-A2, the publication of the transcription.

## 15. Effort

M (one existing 700-line scanner module plus one new test file); the measurement of R1 decides whether it grows.
