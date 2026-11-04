# ENT-W8-B6-001 — R7 publication digest defect (preserved)

**Superseded handoff:** `qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d`  
**EVIDENCE_PUBLICATION_SHA:** `d18f01f2291328a073dcbdfbe05489faf98013b7`

Indexed `qual_wave_report.sha256` (`4a1107f0…`) ≠ SHA-256 of committed
`qual_wave_report.json` bytes (`b914bd8b…`) in the same commit. Root cause:
qualification report bytes and index digest were not published atomically
(post-index mutation or index built from a different qual_wave source).

Do **not** rewrite commit `d18f01f22`. Strict verification:
`verify_b5_publication_tree()` → fail-closed.

**W8 corrective B5:** `qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91`
(published with post-write `verify_b5_publication_tree()` gate).
