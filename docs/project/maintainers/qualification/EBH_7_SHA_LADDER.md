# EBH-7 — SHA ladder

| Stage | SHA role | Value |
| --- | --- | --- |
| PLANNING_SHA | Annex + master plan baseline | `5df7018aa71f8107b07531c50507eada2cc76f86` |
| IMPLEMENTATION_SHA | Closure wave | `f913944ff608c97a4a4145aae6c2bbd610004f64` |
| SEMANTIC_CORRECTION_SHA | Fail-closed disposition SSOT | `edebda9d5b2fbd99564d8def50f60fd0d6964a9b` |
| AUTHORIZED_BASELINE_SHA | OD-EBH7-01 entry @ operator approval | `81679e67225f42dfbc1c508516e19488b6fbf453` |
| W1W2_RECON_SHA | Qualification-only correction | `9ecf8666a42320cd65ca9852b1fdbdd8bbfbf805` |
| DISPOSITION_PIN_SHA | SSOT test_execution_evidence pin | `9ecf8666a42320cd65ca9852b1fdbdd8bbfbf805` |
| EBH7_BATCH_AB_START_SHA | Full behavioral execution entry | `195046d46360cb720956cd306b72cda81d8cc3b4` |
| EBH7_BATCH_AB_IMPL_SHA | Batch A/B implementation + pytest evidence | `ab8f958ed8c1846e6a371a2bcba2884e18fe60bb` |
| EBH7_BATCH_C_START_SHA | RPL-H Batch C entry | `da1c29162a7a58a7e56674777bcf9259fd3ded28` |
| EBH7_BATCH_D_START_SHA | TEN/XDM Batch D entry | `4c884dbc509203ec301303388176275a6792bf45` |
| EBH7_BATCH_D_IMPL_SHA | Batch D D1 TEN-03/04/05 EXACT | `54da93b1b3aa1756d10a44834fabb83938092063` |
| EBH7_MASTER_WAVE_SHA | Master completion wave | `ad3e24a20d7d1b341f721ef2427b9c1f65506061` |
| EBH7_GAP_CLOSURE_START_SHA | Gap-closure program entry | `ad3e24a20d7d1b341f721ef2427b9c1f65506061` |
| EBH7_GAP_CLOSURE_IMPLEMENTATION_SHA | Gap-closure final implementation | `8bbda35dd1686b15bd6959b732ade3ba0f268632` |
| EBH7_REMEDIATION_BASELINE_SHA | Independent audit remediation entry @ pushed HEAD | `11b724dd18ba1a7ed5f53a3999dc18af1068391d` |
| EBH7_SEMANTIC_DISPOSITION_SHA | SSOT disposition pin | see `_semantic_disposition_registry.py` |
| EBH7_CLOSURE_IMPLEMENTATION_SHA | Terminal closure — 56/56 RPL @ operator dispositions | `d4937889fc0b07515649dbdddf602df2a4415489` |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA | Independent B6 F01/F02 + INTERNAL/FIXED sweep | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| EBH7_CLOSURE_G01_G09_IMPLEMENTATION_SHA | Closure evidence audit SSOT + gates | `72619e6c4a4b6cc8b4494f3c31b82dfedc1f0fe9` |
| EBH7_B6_PROOF_STRENGTH_CORRECTION_SHA | B6 F07/F08/F09 RPL-E/F + derived G07/G08 + G09 resolution | `83136ef7be70ff33d0a81b978c154ac8528930a2` |
| EBH7_F07_PROBE_FAILCLOSED_SHA | RPL-E operational probe fail-closed + adversarial gates | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| TEST_EXECUTION_SHA | Full ebh_7 L3 replay | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA | P03 dossier publication | `0c0075b3456bd6a8825c1231f172f33fae1ef24c` |
| INDEPENDENT_ACCEPTANCE_BASELINE_SHA | B6 audited tree | *pending B6 PASS* |
| FORMAL_CLOSURE_PUBLICATION_SHA | OD-EBH7-08 signed publication | *pending operator* |

**Deprecated alias:** `CLOSURE_SHA` must not denote implementation while B6/OD-08 are OPEN.

**Protocol 03 rule:** `TEST_EXECUTION_SHA` equals `EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA`; `EBH7_CLOSURE_IMPLEMENTATION_SHA` is a strict git ancestor (pre-B6 terminal closure @ `d4937889`), not the replayed tree after B6 corrective.

Integrity: `tests/qualification/ebh_7/_semantic_disposition_registry.py` pins `EBH7_SEMANTIC_DISPOSITION_SHA`; `tests/qualification/ebh_7/_ebh7_p03_evidence_integrity.py` pins SHA roles.
