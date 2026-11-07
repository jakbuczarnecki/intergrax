# T-IGX-021-A — IIAC amendment: operator-designated independent signing/certification session (specification batch)

## Scope

Operator requirement (2026-11-07, chat): the operator may designate a **separate, independent agent session** to sign and certify **in the operator's name**. The only condition stated by the operator: the session is **new**, **dedicated solely to that certification task**, **not a continuation of any other session** and **not an implementation session**.

This batch is the **specification batch** and changes **no material governance path**:

1. New document `docs/project/maintainers/qualification/IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md`: normative text of the rule, state `PROPOSED - NOT IN FORCE`, with the operator's condition quoted verbatim in Polish and English.
2. New template `docs/project/maintainers/qualification/IIAC_OPERATOR_SIGNING_DELEGATION.template.md` (delegation record: stage, decision id, record path, SHA-256 of the text to be signed, designated session id, dedication statement, expiry).
3. Specification (text only, no code) of the validator/preflight/activation changes the rule requires, split into follow-up tickets (see Follow-ups).
4. Impact analysis: which pinned items the real change voids (see Constraints).

Agent-proposed safeguards, included in the draft **unless the operator strikes them here before audit** (they keep the rule auditable; the operator's own condition above is the normative core):

- S-1 the delegating record is signed by the operator once per delegation (decision id, record path, text digest, designated session id, expiry);
- S-2 a delegated signature line is distinguishable: `Approved <Decision ID>: operator-delegate @ <ISO-8601>` plus `Delegation: <path>@<sha>`;
- S-3 not delegable: the delegation record itself, IIAC policy activation, any amendment of this rule.

## Constraints discovered (must be honoured)

- `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_*` are `MATERIAL_GOVERNANCE_PATHS`. Editing any of them makes `verify_iiac_policy_activation --require-in-force` fail (stale audit) until a new audited activation. **This batch must not edit them.**
- `scripts/qualification/evol_x_closure_publication_preflight.py` and its tests are the pinned B-4 corrective (OD-3 condition 3). Any change voids the B-4 acceptance and needs its own ticket and audit. **This batch must not edit them.**
- The B6 record format requires `Approved <id>: operator @ <time>` "written by the operator, never by an agent" (`EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md`, `_RECORD_SIGNATURE`). The rule cannot take effect before the follow-up code and the new policy version are audited and activated.
- The rule applies only to certifications started after activation. It is not retroactive and does not unblock the running EVOL-X P3/B6 chain.

## Plan digest

(none — specification only; operator instruction in chat on 2026-11-07)

## Approved

(pending — operator adds the `Approved:` line. The implementer session was started on the operator's explicit chat instruction of 2026-11-07; its commits stay local on a worktree branch, unpushed, until the operator approves.)

## Test contract

```powershell
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
uv run python scripts/governance/sync_agent_env.py --check
git diff --name-only origin/development -- scripts/governance/iiac_activation_gate.py scripts/governance/iiac_common.py scripts/governance/verify_iiac_audit_manifest.py scripts/governance/verify_iiac_role_separation.py scripts/governance/verify_iiac_policy_activation.py docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md scripts/qualification/evol_x_closure_publication_preflight.py tests/unit/qualification
```

The last command must print nothing. One `uv run` per step (test-process budget).

## Follow-ups (not part of this batch)

- `T-IGX-021-B`: code — signature grammar, `operator-delegate`, delegation-record validation, role separation for the designated session (preflight, B6 formats, material verifiers). Voids OD-3 cond. 3 → new audit of B-4 delta.
- `T-IGX-021-C`: independent audit of 021-A/B (separate session, not the implementer).
- Operator: new policy version, `IIAC_OD_ACTIVATION_*.md`, two-phase SHA publication.

## Out of scope

FRZ promotion, OD-EVOL-12, OD-EVOLX-08, EVOL-X P3/B6 publication, any edit of the material paths above, the activation record, the preflight, Docker, vendor, production configuration, push to `development`.

## Lifecycle state

closed

Operator note: closed by the operator's direct instruction in chat on 2026-11-07 (rule adopted as policy text; independent audit T-IGX-021-C and operator activation record waived by that instruction, not performed; code ticket T-IGX-021-B not done). No `Approved:` line was written by an agent.

## Checkpoint (implementer session; not an audit, no PASS claimed)

Worktree `D:/Projekty/_wt/intergrax/t-igx-021-a`, branch `t-igx-021-a` (local, unpushed), base `origin/development` @ 2f6664d53.

Created: `docs/project/maintainers/qualification/IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` (status PROPOSED - NOT IN FORCE; scope items 1, 3, 4) and `docs/project/maintainers/qualification/IIAC_OPERATOR_SIGNING_DELEGATION.template.md` (scope item 2). S-1..S-3 are marked agent-proposed and strikeable.

```text
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
  -> exit 0 ; IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d
uv run python scripts/governance/sync_agent_env.py --check
  -> exit 0 ; [INFO] sync_agent_env --check: OK (39 adapters + CLAUDE.md block)
git diff --name-only origin/development -- <protected paths of the Test contract>
  -> (see below; must be empty)
```

Open for the operator: the `Approved:` line (not written here), strike or keep of S-1..S-3, open questions OQ-1..OQ-4 in the amendment. Independent audit (`T-IGX-021-C`) is a separate session.
