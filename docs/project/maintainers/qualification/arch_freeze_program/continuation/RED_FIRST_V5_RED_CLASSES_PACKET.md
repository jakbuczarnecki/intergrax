# T-AF2-RED-CLASSES — recommendation packet (six residuals)

**Packet:** single recommendation under **T-AF2-RED-CLASSES** (not a signature).  
**Root:** [`RED_FIRST_V5_CLASSIFICATION_ROOT.md`](RED_FIRST_V5_CLASSIFICATION_ROOT.md)  
**Checkpoint cited in root:** `origin/development` `1bc28033cf0fdf95b169ec3b16b3ea5d930c0f09`  
**Inventory:** 39 `CODE_FIX` rows · **33** `RED_AT_PREFIX` · **6** `RED_FIRST_UNREPRODUCIBLE` (unchanged by this packet).

This packet does **not** mark any operator card **DECIDED**. It does **not** edit `rw_plan.v1.json`, `red_first_status`, or `scripts/governance/verify_rw_plan_phase_check3.py`. It does **not** remove inventory rows or authorize **SCOPE**, **G2**, **FCS**, **M3**, or **ARCH-FREEZE CLOSED**.

**Decision authority:** Operator under **T-AF2-RED-CLASSES**. No existing delegation in this packet signs that card.

---

## Packet scope

Mechanical prefix probes are exhausted (MEP-AF2 v2.1 §V5.2–V5.3). The six rows below are the remaining `RED_FIRST_UNREPRODUCIBLE` class. This packet recommends how the operator should disposition each row **out of the red-first obligation** (or re-home for FRZ-10). Application inside the machine plan or validator requires a separate **operator citation** naming the exact status change.

Until that citation:

- All six rows stay **`RED_FIRST_UNREPRODUCIBLE`**.
- The 39-row `CODE_FIX` inventory stays intact.
- **`RED_FIRST_EVIDENCE`** stays **`INCOMPLETE`**.

---

## FRZ-PLG-02

| Field | Content |
| --- | --- |
| **Recorded contract sentence** | External structural implementation works without a core patch. |
| **Probe that exists** | pytest exit 0 @ prefix `22993254b13a`: `register_plugins` accepts an explicit external class (see `tests/unit/applications/test_nt_plg_red_first_probes.py` and disposition in [`RED_FIRST_V5_DISPOSITION.md`](RED_FIRST_V5_DISPOSITION.md)). |
| **What the probe does not cover** | Admission, composition, or plugin runtime execution beyond registration acceptance at the prefix. |
| **Why no new replay is authorized** | The recorded red-first sentence is the **registration contract**; the prefix already satisfies it at exit 0. Further full-lifecycle replay has no additional evidence value on the mechanical path (V5.3 steps 1–5 exhausted). |
| **Recommended disposition** | **OUT** of the red-first obligation only. The wider plugin lifecycle remains a **platform requirement**. |
| **Canonical basis** | MEP-AF2 already records PLG-02 **N/A WITH EVIDENCE** under **SCOPE-03**; **DS-EXC** is still open, so N/A is **not yet authorized** as a closed qualification row—only OUT of red-first is recommended here. |
| **Effect on other requirements** | Full admission/composition obligations under PDC and PLG domain rows stay open; SCOPE-03 / DS-EXC sequencing unchanged until operator citation. |
| **Decision authority** | Operator under **T-AF2-RED-CLASSES** (this packet is recommendation only). |

---

## FRZ-PLG-07

| Field | Content |
| --- | --- |
| **Recorded contract sentence** | Runtime extensions cannot self-expand authority. |
| **Probe that exists** | pytest exit 0 @ prefix `22993254b13a`: `ToolAccessPolicy` refuses tool ids outside the declared contract (same probe family as PLG-02 remeasurement). |
| **What the probe does not cover** | Proof that every runtime authority bypass is impossible; only the declared contract boundary at the prefix. |
| **Why no new replay is authorized** | No historical bypass of that policy was located. The prefix enforcement already exit 0; another replay would not name a new defect on the mechanical path. |
| **Recommended disposition** | **OUT** of the red-first obligation. **Authority enforcement** remains a **platform requirement** (owner context: `T-IGX-043-J` in plan materials). |
| **Canonical basis** | [`RED_FIRST_V5_CLASSIFICATION_ROOT.md`](RED_FIRST_V5_CLASSIFICATION_ROOT.md); MEP-AF2 v2.1 §V5.3.6 (OUT after mechanical exhaustion). |
| **Effect on other requirements** | Broader PLG-07 platform evidence and any later-wave owner work stay in force; does not close SCOPE or G2. |
| **Decision authority** | Operator under **T-AF2-RED-CLASSES**. |

---

## FRZ-RPL-04

| Field | Content |
| --- | --- |
| **Recorded contract sentence** | Vendor/backend switch does not modify consumers. |
| **Probe that exists** | pytest exit 0 @ prefix `754ee000675d`: worker call sites bind `MessageConsumer` (ENT-iso / RPL red-first probe set). |
| **What the probe does not cover** | Execution of an actual vendor swap or end-to-end consumer graph under a live backend change. |
| **Why no new replay is authorized** | Constructor binding at the prefix already exit 0; no product remediation commit is named for this row on the red-first path. |
| **Recommended disposition** | **OUT** of red-first. **W3 product edits stay forbidden** while **SCOPE** is open. |
| **Canonical basis** | MEP **DP-36**; owner node **`T-AF2-RW-TYP` (W3)** remains on the plan. |
| **Effect on other requirements** | W3 typing/replaceability work stays gated; RPL domain requirements and DP-36 obligations unchanged until W3 is authorized after SCOPE. |
| **Decision authority** | Operator under **T-AF2-RED-CLASSES**. |

---

## FRZ-REL-07

| Field | Content |
| --- | --- |
| **Recorded contract sentence** | No hidden infinite retry/fallback. |
| **Probe that exists** | pytest exit 0 @ prefix `66664e2bffa1`: `RetryEngine.decide` stops at `max_retries`. |
| **What the probe does not cover** | Provider-level retry, fallback chains, or restart loops outside the `RetryEngine.decide` path exercised at the prefix. |
| **Why no new replay is authorized** | **REL-09** is **not** evidence for this row. No unbounded historical path was named for REL-07 at the prefix. Bounded decision behavior is already present at exit 0. |
| **Recommended disposition** | **OUT** of the red-first obligation. **Broader retry behavior** stays a **platform requirement** (REL family remains OPEN in MEP where applicable). |
| **Canonical basis** | [`RED_FIRST_V5_CLASSIFICATION_ROOT.md`](RED_FIRST_V5_CLASSIFICATION_ROOT.md); prefix-api / REL probe remeasurement records. |
| **Effect on other requirements** | REL-09 and other REL rows retain their own disposition; does not promote REL-07 platform closure. |
| **Decision authority** | Operator under **T-AF2-RED-CLASSES**. |

---

## FRZ-PRD-08

| Field | Content |
| --- | --- |
| **Recorded contract sentence** | Production-only bypass count = 0. |
| **Probe that exists** | Inventory gate already reports bypass count **0** at prefix `754ee000675d` (exit 0). |
| **What the probe does not cover** | Live production-graph bypass proof for unqualified operator profiles (**PROD-Q** / Path B live gate). |
| **Why no new replay is authorized** | The CODE_FIX red-first gate is already satisfied at the recorded prefix; repeating inventory replay adds no mechanical evidence. |
| **Recommended disposition** | **OUT** of the **CODE_FIX red-first obligation** only. The **zero-bypass architecture requirement** and later **PROD-Q** stay in force. |
| **Canonical basis** | MEP **split**: architecture gate plus live **PROD-Q** ([`RED_FIRST_V5_SPECIAL_DISPOSITION.md`](RED_FIRST_V5_SPECIAL_DISPOSITION.md) / Path B references). |
| **Effect on other requirements** | FRZ-PRD-08 live portion and related PRD/SEC Path B criteria remain OPEN until PROD-Q. |
| **Decision authority** | Operator under **T-AF2-RED-CLASSES**. |

---

## FRZ-FRZ-10

| Field | Content |
| --- | --- |
| **Recorded contract sentence** | Post-freeze change policy active. |
| **Probe that exists** | Historical qualification cited `_self_test_cases` **`AttributeError`** at the prefix (governance self-test surface), not a product policy implementation. |
| **What the probe does not cover** | Activation or verification of the real post-freeze policy product surface. |
| **Why no new replay is authorized** | The **`AttributeError` is not the policy**; replaying the broken self-test does not evidence FRZ-10 mechanism. Pre-SCOPE red-first demand on this row is circular if W5 activation is required only to clear the gate. |
| **Recommended disposition** | **Re-home** to **`T-B7-PFP-ACTIVATE` (W5)** and **`T-B7-PFP-VERIFY` (W8)**. **Do not implement W5** in this campaign to satisfy red-first. This removes the circular pre-SCOPE demand **only after** an operator citation. |
| **Canonical basis** | [`RED_FIRST_V5_CLASSIFICATION_ROOT.md`](RED_FIRST_V5_CLASSIFICATION_ROOT.md); B7 post-freeze program nodes on WP-12 DAG. |
| **Effect on other requirements** | W5/W8 stay out of current SCOPE-close path; FRZ-FRZ-10 should not block RED_FIRST mechanical completion once re-homed and cited. |
| **Decision authority** | Operator under **T-AF2-RED-CLASSES**. |

---

## Applying recommendations (operator citation required)

| Action | Requires operator citation? |
| --- | --- |
| Change `red_first_status` or row class inside `rw_plan.v1.json` | **Yes** |
| Alter `verify_rw_plan_phase_check3.py` expectations or phase-check contract | **Yes** (and explicitly out of scope for this ticket) |
| Mark **T-AF2-RED-CLASSES** operator card **DECIDED** | **Yes** (not done in this packet) |
| Remove or shrink the 39-row inventory | **Yes** — **not** recommended; keep full inventory until citation names exact deltas |

**Recommended operator actions (upon citation):**

1. Mark the red-first obligation of `FRZ-PLG-02`, `FRZ-PLG-07`, `FRZ-RPL-04`, `FRZ-REL-07`, and `FRZ-PRD-08` as **OUT**, with this packet and [`RED_FIRST_V5_CLASSIFICATION_ROOT.md`](RED_FIRST_V5_CLASSIFICATION_ROOT.md) as evidence references.
2. Re-home `FRZ-FRZ-10` to `T-B7-PFP-ACTIVATE` / `T-B7-PFP-VERIFY` so it is not a pre-SCOPE red-first row.
3. Keep the 39-row inventory until the citation names the exact status change.
4. Independent **T-AF2-RW-PLAN-AUDIT** full verdict stays **NOT_PERFORMED** until citation exists (structural pass does not replace it).

Until citation: **`RED_FIRST_EVIDENCE`** = **`INCOMPLETE`** · **`SCOPE`** = **`OPEN`** · W3/W5 product edits remain out of scope.

---

## Signatures and delegation

| Role | Status in this packet |
| --- | --- |
| **T-AF2-RED-CLASSES** operator card | **Not signed** — recommendation only |
| **Existing delegation chains** | **No** packet author signs on behalf of the operator |

**Recorded:** 2026-11-26. English recommendation packet for governance/tickets **T-IGX-073-E**.