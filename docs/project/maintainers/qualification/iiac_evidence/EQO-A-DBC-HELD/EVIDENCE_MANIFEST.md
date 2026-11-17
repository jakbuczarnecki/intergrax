# EVIDENCE MANIFEST - EQO-A DB-C held hunks: publication of locally held test logs

Ticket: `governance/tickets/T-IGX-046-F.md`. Drafted 2026-11-17. Basis: EQO-A Campaign A dossier baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`; the earlier publication manifest `iiac_evidence/EQO-A-PT-D03/EVIDENCE_MANIFEST.md` is the layout model.

## What this manifest is and is not

- It lists three files copied byte-for-byte from local, untracked locations on the workstation where this change was prepared into this directory tree, with their sha256 values and sizes, and the result of a secret and personal-data scan. It also lists two files that were found but not published, and why.
- The tracked copy plus the sha256 in this manifest is a custody anchor for the file contents as they were found on 2026-11-17. It says nothing about whether the content is correct, about which commit the run was made at, or about which stage the run belongs to beyond what the file itself shows.
- This publication is not certification. It does not make any audit independent, does not verify any stage, does not close or promote any roadmap row, FRZ criterion or finding, and does not turn a scratch log into an evidence pin.
- The files were not reviewed for correctness by this publication. The scan below is a pattern scan plus a full read of the three small files; it is not a content review.

## Custody statement (statement of facts)

1. On 2026-11-17 each file below existed at the local path named in the column `Origin (local, untracked)`; the first two lie under `.tmp/` (excluded by `.gitignore` line 19) and the third lies in a scratch directory outside the repository. No tracked copy existed before this change. Whether another copy exists elsewhere is not known to this change.
2. The files are pytest console captures written by a Windows PowerShell session. They are UTF-16 little-endian text with a byte-order mark and carry no author, session or commit identifier.
3. Each file was copied with a byte copy and compared with its source and with the sha256 recorded here before the commit. `.gitattributes` line 1 sets `-text` for `iiac_evidence/**`, so git does not normalise line endings or encoding for these paths.
4. This publication did not re-run any command and did not verify who produced the files or at which commit.

## Published files (3)

| Path under `iiac_evidence/` | sha256 | Size (bytes) | Origin (local, untracked) | Last line of the file (decoded) | Cited by tracked record |
| --- | --- | ---: | --- | --- | --- |
| `EQO-A-DBC-HELD/logs/harness-w5-r1/v11-batch-a.log` | `464e96b23dc216efbeefd280e27dbe3f9dec7e3b23468097e26b92de7720ea1d` | 2642 | `.tmp/session/harness-w5-r1/v11-batch-a.log` | `106 passed, 3 warnings in 32.62s` | directory only: `ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_PARENT_RECERTIFICATION.md` L119 |
| `EQO-A-DBC-HELD/logs/harness-w5-r1/v12-batch-b.log` | `0cc6b83c3e1621837fa7c825f61df72f9943a68959222e40b3173c9fec58fec5` | 2642 | `.tmp/session/harness-w5-r1/v12-batch-b.log` | `106 passed, 3 warnings in 33.98s` | directory only: same record, L119 |
| `EQO-A-DBC-HELD/logs/pre-ent-w6-cross-audit/parent_regression_5gate.log` | `7c4ab15cff100f393a73dc5a123af89700eca8ed79fd41a033ff1c4454426f6d` | 1646 | `D:\Projekty\_wt\intergrax\_scratch\w6-cross-audit\parent_regression_5gate.log` | `41 passed in 107.13s (0:01:47)` | by path: `pre_ent_enterprise_certification/PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md` L79 and L160 |

The two W5 files differ only in the timing on their last line. None of the three files lists a run set of test files (the two W5 files name two test files only as warning sources), the commit that was tested or an exit code; the file names `v11-batch-a` and `v12-batch-b` are the only link to the labels "batch A" and "batch B" used by the tracked W5 record.

Scan of the three files, run on the decoded text: no private-key header, cloud-key or token pattern, bearer or JWT string, connection string with credentials, e-mail address, IP address, `Users` or `AppData` path, or personal-name pattern matched; the words `password`, `secret`, `token`, `api key`, `authorization` and `credential` did not occur. The two W5 files contain absolute workstation paths of the form `D:\Projekty\intergrax\intergrax\runtime\...` (a checkout location, no user name); the W6 file contains no absolute path. The decision to treat a checkout path as not personal data is the author's and is stated here so that a reviewer can overrule it. Result: CLEAN under that reading (pattern scan plus full read of three small files, not a content review).

## Held back (2 files, not published)

| Local file | Size (bytes) | sha256 of the local file as found | Reason |
| --- | ---: | --- | --- |
| `.tmp/session/HARNESS-W4-FINAL-CONSOLIDATION/t11-a.log` | 439232 | `1c4934fa5aca892547c2a95aab489b7973a7eb0ce54700920657cfd79fb892fe` | Scan hit: 200 lines contain a Windows user-profile path (`C:\Users\<profile>\AppData\...`). A log may not be redacted, because a modified log is no longer the artefact; so it is not published. |
| `.tmp/session/HARNESS-W4-FINAL-CONSOLIDATION/t11-b.log` | 439232 | `ad3ffb562bc7c7ea2901363cefb26158ccc74bac6cd941fd996f2e1db990adc8` | Same scan hit (200 lines). |

The two held files remain only on the local workstation. Their digests are an observation of local files and cannot be checked from the repository; they are recorded so that anyone holding the files can compare. The record that cites them is annotated in `ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W4_PARENT_RECERTIFICATION.md`.

## Verification performed by this publication

- Source sha256 and size of all five files were computed on 2026-11-17 with `sha256sum` and `wc -c`.
- After the commit, the blob stored in git for each published path is read back (`git cat-file blob`) and its sha256 compared with the value above; the result is recorded in the pull request body.
