# RW train — standing signing delegation record 2 (packet 7 D1, wave 0 and wave 0b)

**Status:** record of standing Operator decisions, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chats; it is not a verbatim quote. It grants the signing delegate nothing beyond the rules below, it is not an audit verdict, and it claims no PASS, closure or certification. It supplements `RW_TRAIN_STANDING_DELEGATION_RECORD.md` and the per-wave records `RW_TRAIN_DELEGATION_RECORD*.md`, which stay unchanged.

## Decisions

On 2026-11-20 the Operator gave two answers that this record renders for the signing delegate:

- In the ARCH-FREEZE orchestrator chat, the Operator agreed to the standing mandate D1 and the decisions D2 as recommended, and asked for the work to resume (ARCH-FREEZE session record `OPERATOR_PACKET_7.md`, a session artefact under `.tmp/`, not tracked).
- In the EQO-A remediation orchestrator chat, the Operator accepted the recommended option of every group of decision wave 0 (groups A to D) and, in a second dialog, of every item of wave 0b, the ARCH-FREEZE additions (EQO-A session record `WAVE_0_DECISIONS.md`, a session artefact under `.tmp/`, not tracked).

The wave 0b mandate M-1 to M-3 and the packet 7 items D1-1, D1-2 and D1-4 state the same mandate; this record encodes it once. Of the other decisions, it restates only what concerns signing: the extra tickets of wave 0b, the stop conditions (packet 7 and A8) and the gates that stay with the Operator. Merge authority, reporting, session starts, account limits, residual dispositions, GitHub ruleset stages, pins and activations are not restated here.

## Series named by this record

This record names the ticket series of the RW product train: `T-IGX-043-*`, `T-IGX-044-*`, `T-IGX-046-*` and `T-IGX-051-*`. The wildcards exist so that the strict approval grammar accepts the record for any ticket of these series. They do **not** authorize signing by themselves.

This record applies only to approval commits on `arch-freeze/rw-train` made before the train is merged into `development` (FCS). It authorizes nothing on `development` or on any other branch, and it authorizes nothing new after FCS.

## Relation to the earlier records

- The per-wave records 1 to 4 (`RW_TRAIN_DELEGATION_RECORD.md` to `RW_TRAIN_DELEGATION_RECORD_4.md`) and the standing record `RW_TRAIN_STANDING_DELEGATION_RECORD.md` stay valid for the tickets and digests already signed under them. Nothing in this record voids or re-opens those signatures.
- For a signature made under this record, the conditions below replace the condition of the earlier records that a digest must be listed in an Operator-approved wave packet, the S2 amendment envelope of the standing record, the amendment rule A2 of wave 0, and the rule that a further amendment needs a new Operator approval and a new record file. The rule of the standing record that every new digest of T-IGX-044-F needs the Operator's own decision is replaced by the limits below, in particular the IIAC limit. Every exclusion of the earlier records that this record repeats under "Always the Operator" still applies.

## M-1 (D1-1). When the delegate may sign

The delegate writes the approval block of a ticket of the named series, at a given plan digest (a new ticket, an amendment or a fixer ticket), only when all of the following hold:

1. **The ticket belongs to the train.** It is a ticket of one of the named series whose work lands on `arch-freeze/rw-train`.
2. **The review is independent and bound to the digest.** An independent review verdict REVIEW_PASS, or DELTA_OK for an amendment, exists for the same plan digest, and for an amendment for the same amendment diff. A verdict for another digest does not count.
3. **No item of "Always the Operator" and no stop condition applies.**

A gate-meaning change is signable under this record only when it makes a check stricter: it adds a check or an assertion, narrows a guard baseline, allowlist, ratchet, threshold or exception register, or removes an entry from one of them. A gate-meaning change that does anything else, or whose direction is unclear to the delegate, is an Operator stop.

The delegate recomputes the plan digest from the plan file with `plan_digest.py`, refuses on any mismatch, and reports the signed ticket and digest to the orchestrator. The approval commit message names this record, the review verdict (its file and the reviewed head or diff) and the signed digest.

## M-2 (D1-2). When a plan may change

- A plan changes only to resolve a BLOCKER or MAJOR finding of an independent review. LOW and INFO findings are recorded as notes in the pull request body and do not change the plan.
- A ticket has at most 2 amendment rounds. An amendment round is one new plan digest made in response to review findings, counted per ticket from the first independent review verdict on its plan, including rounds made under the earlier records. If the plan still has a BLOCKER or MAJOR finding after the second round, the delegate does not sign, and the orchestrator stops and reports to the Operator.
- The delegate does not sign a digest that changes the plan for any other reason.

## M-3 (D1-4). Draft defaults

A plan may settle an Operator-level question by taking the draft default recorded for that question in a session record; the delegate may sign such a digest when the plan names the question and its default. A plan that settles an Operator-level question that has no recorded draft default, or departs from the recorded default, is an Operator stop.

## Extra tickets accepted in wave 0b and wave 0

The Operator accepted the following tickets. Each is signed under M-1 to M-3 like any other ticket of the named series, and only when its ticket id belongs to one of the named series and its approval commit is on `arch-freeze/rw-train`:

- **F-1:** the fixer for EX-D1-07 and G-08 (descriptor edit and PDC-D1 test treatment), which releases part D of the RW-RECOVERY ticket.
- **B3 and B4:** the two fixers for the A4-032 legs. Leg (a): in PRODUCT or STRICT an enabled OTLP exporter with an empty endpoint fails closed at wiring, with its own fixture and gate plan and a measured list first. Leg (b), with WP2-OBS-02: in PRODUCT or STRICT a composition with side-effect tool ids and no explicit sandbox root is refused and the bundle-only gap is closed; part 2 of T-IGX-043-E (the sandbox script) is released with it. Each is signed only while its plan stays inside that ruling.
- **F-3:** the test-hygiene ticket for `tests/unit/applications/test_ac3_phase5_production_lifecycle_e2e.py` (tests only). The Operator merges it on `development`; an approval commit that is not on `arch-freeze/rw-train` is outside this record and needs a record on its own branch.
- **F-4:** T-IGX-043-N, after its independent review.
- **F-5:** the residuals R-1 to R-8 of the RW-SUPPLY ticket and the two cursor-signing byte labels, with their recorded defaults.

Two accepted items are **not** signed under this record:

- **G-9:** ticket 052-E (standalone ECAP) lies outside the named series. It is signed only with its own one-time Operator authorisation, and the Operator merges it.
- **G-5:** A4-040 and Part G of the PT-WB ticket, accepted for wave 4. This record keeps them excluded (see below); the delegate signs that work only under a later record that names it.

## Always the Operator

The delegate never signs, and the orchestrator stops and asks the Operator, when a plan digest does any of the following or reaches any of these gates:

- It loosens or removes a check: it removes or weakens a check or an assertion, adds an entry to a guard baseline, allowlist, ratchet, threshold or exception register, relaxes a threshold, or adds a skip, an expected failure, a `conftest.py` change or a pytest configuration change that lets a failing check pass.
- It touches the IIAC protocol beyond the paths already approved: an IIAC material governance path that is not in the edit scope of a digest the Operator approved for the same ticket in a wave packet, or any change to the IIAC protocol itself beyond those paths.
- It touches governance under `scripts/` or `.github/` without a reason that the ticket names for that path in its own text.
- It touches any Operator-only file, including `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` and `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`.
- It falls under an exclusion of the standing record: Part G of the PT-WB ticket and A4-040 (the delegate does not sign a PT-WB digest whose plan has Part G as in-scope work), any PROD-Q run, and the RB-2 re-pin.
- The Operator's own hard gates: the merge of the train into `development` (FCS), the ENTRY check, and the admission and closure texts OD-ARCH-FREEZE-01, OD-AF-15, OD-AF-16b and DS-ADM EDER-D-0170, EDER-D-0172 and EDER-D-0173.

## Stop conditions

The delegate does not sign, and the orchestrator stops and reports to the Operator, when any of the following arises (packet 7 and rule A8 of wave 0):

- a red gate that is not registered and not explained by a base red;
- a BLOCKER in frozen scope that no ticket owns, or a plan defect after 2 amendment rounds;
- an authority conflict: documents disagree and the resolution changes a verdict;
- a decision outside these rules;
- a change of gate meaning or product semantics beyond the approved rulings, or a change that would loosen a gate or needs an Operator-only file;
- a tenant or security residual with no allowed disposition;
- secrets in evidence;
- a tool that writes to tracked paths unexpectedly;
- broken independence of a reviewer, auditor or certifier;
- a hard Operator gate reached;
- an API, account or machine limit that stops the work for more than 24 hours.

## Session independence

The delegate session is never the implementer session of the ticket it signs: the `Delegate session:` and `Implementer session:` values of the approval block differ, and the delegate session wrote no part of the ticket's implementation. The review that the signature relies on comes from a session that is neither the implementer nor the delegate. If independence is broken, the delegate does not sign and the orchestrator stops and reports to the Operator.

## Ancestry and merge rule

The approval commit of each ticket signed under this record must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `arch-freeze/rw-train`, or a later train commit that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request that adds this file merges with a merge commit, not a squash.
