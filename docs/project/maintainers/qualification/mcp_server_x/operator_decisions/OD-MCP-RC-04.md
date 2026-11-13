# OD-MCP-RC-04 — PDC-P7-00 window decisions: baseline re-pin and freeze, red tests, bound verdict, delegation renewal

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-RC-04 |
| **Status** | **RECORDED (operator chat answers, 2026-11-13; not signed `Authorized:` lines)** |
| **Related** | `OD-MCP-RC-01` (D-1), `OD-MCP-RC-03`, `DELEGATION-MCP-RC-2`, `PDC_P7_00_CERTIFIER_PREFLIGHT.md` (sections 3, 5 and 7), `MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md` (branch `mcp-rc-activation`) |

English renderings below are by the orchestrator session and are not operator-written lines; the Polish option labels are quoted as selected.

## 1. Window and baseline

- The operator stated that the PDC-P7-00 window is open ("Okno PDC-P7-00 jest otwarte, wznów kroki"; "The PDC-P7-00 window is open, resume the steps").
- First baseline answer: "87f4d4f41 (development)", then, after PRs #523 and #524 landed: "Przepnij na head development (Recommended)" ("Re-pin to the development head").
- After PR #525 landed (`a156cbbd5`, two more product paths): "Przepnij i zamroź (Recommended)" ("Re-pin and freeze"). Effect: the integrated baseline is development `a156cbbd5`, merged into `mcp-rc` as `324d9dba3`; audit addendum 2 covers the two new paths. The operator declares, on the PDC-X side, that PDC-X changes no product path on `development` for the rest of the window; the orchestrator cannot enforce this. Any further product commit on `development` needs another addendum and preflight.

## 2. Red tests on the baseline

Question: who repairs the tests that are red on the baseline (F-07 PDC-X inventory, F-35 COMPAT-X persistence classification, F-09 pytest hang) so that `governance-fast` and `ent-audit-*` can be green on the PR head. Answer: "PDC-X na development (Recommended)" ("PDC-X on development"). Effect: the PDC-X owner repairs them on `development`; `mcp-rc` merges `development` again afterwards. No product path is touched by those repairs. MCP-RC does not repair PDC-X tests.

## 3. Bound verdict

Question: all four auditor manifests carry `CONDITIONAL` and the activation gate accepts only `PASS` in the binding manifest (ACT-N05); should the activation certifier author a new binding manifest with `verdict = PASS`, with the scoped limits in `limitations` and in the decision text. Answer: "Tak, wzór P6 (Recommended)" ("Yes, the P6 pattern"). Effect: the activation certifier writes that manifest and states the scope limits, the open findings, the red-test caveat and the accepted conditions explicitly. The auditors' scoped verdicts are not rewritten.

## 4. Delegation renewal

Question: `DELEGATION-MCP-RC-2` expires 24 hours after the commission; renew it for 24 hours from the start of the signing session with the same scope (activation line only, no rebind, no merge). Answer: "Tak, odnawiam (Recommended)" ("Yes, I renew"). Effect: the renewal applies from the start of the signing session and is transcribed into `DELEGATION-MCP-RC-2.md` by the signing session. Scope unchanged.

## Explicit non-actions

No signature, no activation line, no rebind, no merge, no product change, no FRZ change (counter stays 225).
