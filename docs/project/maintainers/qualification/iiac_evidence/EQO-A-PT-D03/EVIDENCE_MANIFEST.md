# EVIDENCE MANIFEST - EQO-A PT-D03 publication of locally held audit artefacts

Ticket: `governance/tickets/T-IGX-046-A.md` (lifecycle `plan_pending`, approval line pending). Drafted 2026-11-16. Basis of the plan: EQO-A Campaign A dossier baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`; Operator decisions DP-07 and DP-44 and authorisation AR-20 (publication with sha256 and a secret/PII scan of every file).

## What this manifest is and is not

- It lists the files copied byte-for-byte from gitignored local session directories on the Operator machine into this directory tree, with their sha256 values and sizes, and the result of a secret/PII scan.
- The tracked copy plus the sha256 in this manifest is a custody anchor for the file contents as they were on the Operator machine on 2026-11-16. It says nothing about whether the content is correct.
- This publication is NOT certification. It does not make any audit independent, does not verify any stage, does not close or promote any roadmap row, FRZ criterion or finding, and does not accept any verdict a file carries (for example the CONDITIONAL / STOP verdicts of the TOOL-LIFE-X reports are published as found).
- The files were NOT independently reviewed by this publication. The scan below is a pattern scan plus a read of the lines that carry URLs, `@` characters, key or cookie words; it is not a content review.

## Custody statement (statement of facts)

1. Until 2026-11-16 every file below existed only on the Operator machine, in gitignored directories under `.tmp/session/` (`custody: Operator machine only`; no tracked copy, and no off-machine copy is known to the records).
2. Who produced each file, as far as the records show: the AUD-* reports and findings carry headers naming auditor sessions (`aud_qualx`, `aud_qualxp0`, `aud_skillx`, and the AUD-TOOLLIFE and AUD-TOOLLIFE2 sessions) and state that they are auditor records and not Operator closure signatures. The SIGN-* files name designated signing sessions, state `process_enforced` assurance and record transcriptions of decisions under a delegation record. These are statements made inside the files. This publication did not verify the identity, independence or authority of any producing session, and did not re-run any command or test recorded in them.
3. For 24 distinct sha256 values, the tracked record `docs/project/maintainers/qualification/CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md` cites a digest that equals the digest of a local file listed in the planning manifest. The column `Cited in tracked record` shows the line numbers at which each value occurs in that record at the publication head, computed by this publication. For files marked `no`, the digest is recorded here for the first time; no tracked record pinned them before.
4. Each file was copied with a byte copy and compared with its source and with the sha256 recorded here before the commit. `.gitattributes` line 1 sets `-text` for `iiac_evidence/**`, so git does not normalise line endings or encoding for these paths; files that contain CRLF line endings in the source keep them.
5. Files with a scan hit were not published (section Held back). A file that was not published is still held only on the Operator machine, with the custody described in item 1.

## Verification performed by this publication

- Source sha256 and size of all 33 files in the planning manifest were recomputed on 2026-11-16 and equal the values in `PUBLICATION_MANIFEST.md` of the DB-J plan.
- After the commit, the blob stored in git for each published path is read back (`git cat-file blob`) and its sha256 compared with the value below; the result is recorded in the pull request body.

## Published files (29)

Scan result for every file in this table: no private-key header, cloud or token pattern, bearer or JWT string, connection string with credentials, e-mail address, IP address or local profile path matched; the weak words `password`, `secret`, `api_key`, `authorization:` and `credential` did not occur. Lines that carry URLs, `@`, `key` or `cookie` words were read and contain only commit-reference `@` notation, `file:///` build lines, delegation-citation lines or prose. No personal name other than the cases under Held back was found by the name patterns. Result: CLEAN (pattern scan plus targeted read, not a full content review).

### QUAL-X

| Path under `iiac_evidence/` | sha256 | Size (bytes) | Origin (local, gitignored) | Cited in tracked record |
| --- | --- | ---: | --- | --- |
| `QUAL-X/audit/AUD-QUALX/AUDIT_REPORT.md` | `3661a2cd4281b6c89407afcf703e35df1b232ae804c6a49d29a0f994df4f3a4f` | 7184 | `.tmp/session/cert-campaign/AUD-QUALX/AUDIT_REPORT.md` | L54 |
| `QUAL-X/audit/AUD-QUALX/FINDINGS.json` | `89687115902c6d3cd2f10df49152f2286d7f711bde65849b67345163b2678dbb` | 2753 | `.tmp/session/cert-campaign/AUD-QUALX/FINDINGS.json` | L54 |
| `QUAL-X/audit/AUD-QUALX/logs/pytest-d9ea.log` | `b566d6b6ce8a8833ebbebe34ab8c414124efe9f46ade5e472405ffbec8227725` | 6521 | `.tmp/session/cert-campaign/AUD-QUALX/logs/pytest-d9ea.log` | L54 |

### QUAL-X-P0

| Path under `iiac_evidence/` | sha256 | Size (bytes) | Origin (local, gitignored) | Cited in tracked record |
| --- | --- | ---: | --- | --- |
| `QUAL-X-P0/audit/AUD-QUALXP0/AUDIT_REPORT.md` | `149647a7ea9b719ee17d17c2f9797d8aeffb25f1d2cb9df07c68b0cd4f40a574` | 6886 | `.tmp/session/cert-campaign/AUD-QUALXP0/AUDIT_REPORT.md` | L87 |
| `QUAL-X-P0/audit/AUD-QUALXP0/FINDINGS.json` | `4a2b46a87c1426271e463e3091fa4f39e361f9b09fa00e6321f2a6eaa126ab18` | 4927 | `.tmp/session/cert-campaign/AUD-QUALXP0/FINDINGS.json` | L87 |
| `QUAL-X-P0/audit/AUD-QUALXP0/AUDITOR_PASS_RECORD_DRAFT.md` | `cf213d0339a8a177896eccead3490823cf21e2405c0682344922948db20d807a` | 1012 | `.tmp/session/cert-campaign/AUD-QUALXP0/AUDITOR_PASS_RECORD_DRAFT.md` | L87 |

### SKILL-X

| Path under `iiac_evidence/` | sha256 | Size (bytes) | Origin (local, gitignored) | Cited in tracked record |
| --- | --- | ---: | --- | --- |
| `SKILL-X/audit/AUD-SKILLX/AUDIT_REPORT.md` | `04d41d5114c9ec391bd14f70991016ba49668cc44ae25f668479d0d27336fa82` | 7358 | `.tmp/session/cert-campaign/AUD-SKILLX/AUDIT_REPORT.md` | no |
| `SKILL-X/audit/AUD-SKILLX/FINDINGS.json` | `ba9bedc5b9cccde0ef5cd2c4b35b807c7cbb05706e818bbaba3babb0874553b6` | 2744 | `.tmp/session/cert-campaign/AUD-SKILLX/FINDINGS.json` | no |
| `SKILL-X/audit/AUD-SKILLX/logs/pytest-552e.log` | `b3841fa57fc3454ce547f19320966cd0596d0ba9cdd83bcf3877b16e57b88938` | 755 | `.tmp/session/cert-campaign/AUD-SKILLX/logs/pytest-552e.log` | no |

### TOOL-LIFE-X

| Path under `iiac_evidence/` | sha256 | Size (bytes) | Origin (local, gitignored) | Cited in tracked record |
| --- | --- | ---: | --- | --- |
| `TOOL-LIFE-X/audit/AUD-TOOLLIFE/AUDIT_REPORT.md` | `d4c26475e2594eb44a3f2740fe10c5d056b90367be6c1438b4163e63934c2fd6` | 10912 | `.tmp/session/cert-campaign/AUD-TOOLLIFE/AUDIT_REPORT.md` | no |
| `TOOL-LIFE-X/audit/AUD-TOOLLIFE/FINDINGS.json` | `dac3b07e13b9eaa1bd70f9e267ac92759f4fa1e71c009fff765f309424cd6f80` | 4873 | `.tmp/session/cert-campaign/AUD-TOOLLIFE/FINDINGS.json` | no |
| `TOOL-LIFE-X/audit/AUD-TOOLLIFE2/AUDIT_REPORT.md` | `f4ad68dc7e754824e6a43aee2946191c6b5dc308b14eac4ab3a61f795c366a2a` | 14873 | `.tmp/session/cert-campaign/AUD-TOOLLIFE2/AUDIT_REPORT.md` | no |
| `TOOL-LIFE-X/audit/AUD-TOOLLIFE2/FINDINGS.json` | `03d8abdc148cd3f1aa0e9589266f9be3395002a5a626b7ddb835619284edb904` | 4123 | `.tmp/session/cert-campaign/AUD-TOOLLIFE2/FINDINGS.json` | no |

### CERT-CAMPAIGN-2026-11-07

| Path under `iiac_evidence/` | sha256 | Size (bytes) | Origin (local, gitignored) | Cited in tracked record |
| --- | --- | ---: | --- | --- |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-2_SKILLX/SIGNED_DECISIONS.md` | `1b1f0d518e677385b23634e96c2741eee508d7760097fe58ffe2d309e51be204` | 4804 | `.tmp/session/cert-campaign/SIGNING/SIGN-2_SKILLX/SIGNED_DECISIONS.md` | L132 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-3_CTENT/SIGNED_DECISIONS.md` | `6ce86682eaad9dd1c1a304da4ec419d633a5f5696ce7c0f9e1712e1853b19650` | 6437 | `.tmp/session/cert-campaign/SIGNING/SIGN-3_CTENT/SIGNED_DECISIONS.md` | L31 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-4_TLX/SIGNED_DECISIONS.md` | `672518014018fb4ab88eaf22d4c87b47abbf7478423418ca26448bc4cd38c32c` | 13762 | `.tmp/session/cert-campaign/SIGNING/SIGN-4_TLX/SIGNED_DECISIONS.md` | L32 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/SIGNED_DECISIONS.md` | `3c8b94ccff3b89612d109a24425b2f109dabb0daf2b1306ba75671cfd65f796f` | 30334 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/SIGNED_DECISIONS.md` | L196 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-1_QUALX/texts/OD-QUALX-ACCEPT-2026-11-07.txt` | `16861352602eefd242b143b19a0bee15d083f8f9f27b7dae8a9c4299ea704f47` | 1362 | `.tmp/session/cert-campaign/SIGNING/SIGN-1_QUALX/texts/OD-QUALX-ACCEPT-2026-11-07.txt` | L49 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-1_QUALX/texts/OD-QUALXP0-VERIFIED-2026-11-07.txt` | `4ca3aa913899c63856425803ffdc48b199d57d1025d5eeb77ca0d49a12a38210` | 1724 | `.tmp/session/cert-campaign/SIGNING/SIGN-1_QUALX/texts/OD-QUALXP0-VERIFIED-2026-11-07.txt` | L82 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-2_SKILLX/SIGNED_TEXT_OD-SKILLX-ACCEPT-2026-11-07.txt` | `5b6ecb949a1d0dae54355986c4321ba6ddd6c44fdad71a792fb1fa297c694bf5` | 1528 | `.tmp/session/cert-campaign/SIGNING/SIGN-2_SKILLX/SIGNED_TEXT_OD-SKILLX-ACCEPT-2026-11-07.txt` | L140,180 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-ROUTE-DS0-2026-11-07.txt` | `fdda3f285518629de35d70d5cf1827f7e4efcdb04ad5d13b52a74a170778567b` | 1128 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-ROUTE-DS0-2026-11-07.txt` | L227 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-ROUTE-DS1-2026-11-07.txt` | `9a90f2a0162efe82b3e8d31fbc98ef48680e62ff395bbe5a328ef980b41780c4` | 1059 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-ROUTE-DS1-2026-11-07.txt` | L248 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-ROUTE-DS2-2026-11-07.txt` | `f50eb8443c6c145ffeda0e28398aa51df0a1337bb1789e6a629b667bfef6820b` | 1160 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-ROUTE-DS2-2026-11-07.txt` | L270 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-ROUTE-DS3-2026-11-07.txt` | `7b3546b01eee470424e9b2e573a7bdab220440df7db06a36721c6c83ccec02b9` | 1381 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-ROUTE-DS3-2026-11-07.txt` | L292 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-ROUTE-DS4-2026-11-07.txt` | `79b18a6fc405c21c2b645de87ee86359c2fdc4065a3a64fe4e95de5f130713f7` | 1458 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-ROUTE-DS4-2026-11-07.txt` | L314 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-ROUTE-DS5-2026-11-07.txt` | `01b3103634a0edb0697e91bd984a379245f74f0aa34ace00c9346a6b19128355` | 1140 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-ROUTE-DS5-2026-11-07.txt` | L336 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-SSOT-D1-2026-11-07.txt` | `dff6a49dc815c09201edaa54eced5977db8b066ff7e1acd53ff56bdb0faf1ed9` | 1468 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-SSOT-D1-2026-11-07.txt` | L358 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-SSOT-D2-2026-11-07.txt` | `ecb0dcbb4da4a689ebbd85f66f882d988d7bb7a867195219bf3f6d18b8268943` | 1342 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-SSOT-D2-2026-11-07.txt` | L380 |
| `CERT-CAMPAIGN-2026-11-07/signing/SIGN-5_ROUTE/texts/OD-SSOT-D3-2026-11-07.txt` | `1adc3b1f05ab79855186e4d8796ee913573fd8be02b3fe68efea4ee29defea0c` | 1444 | `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/texts/OD-SSOT-D3-2026-11-07.txt` | L402 |

## Held back (not published): scan hits

Recorded as path, line and kind only; the matched text is not reproduced. The source sha256 and size are listed so the file can be released later with the same digest.

| Local source (gitignored) | Line | Kind | sha256 of local file | Size (bytes) | Planned destination if released |
| --- | ---: | --- | --- | ---: | --- |
| `.tmp/session/cert-campaign/AUD-QUALXP0/logs/pytest-p0.log` | 89 | operating-system profile user name inside a local Windows path | `9e0a0a7a751f2451d869d8bd8e8c1716a19085c677e5fb77cfafa30be67299e9` | 8599 | `QUAL-X-P0/audit/AUD-QUALXP0/logs/pytest-p0.log` |
| `.tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md` | 5 | repository account handle (git/GitHub login that embeds a surname) | `5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab` | 2558 | `CERT-CAMPAIGN-2026-11-07/delegation/DELEGATION_2026-11-07_MASTER.md` |
| `.tmp/session/cert-campaign/SIGNING/SIGN-1_QUALX/SIGNED_DECISIONS.md` | 26 | repository account handle (git/GitHub login that embeds a surname) | `5acbb740f953ff34c6fa684db6cdff7ef9ce0edd7c2cc89cff0695e3fa399931` | 11943 | `CERT-CAMPAIGN-2026-11-07/signing/SIGN-1_QUALX/SIGNED_DECISIONS.md` |
| `.tmp/session/cert-campaign/SIGNING/SIGN-1_QUALX/SIGNED_DECISIONS.md` | 53 | repository account handle (git/GitHub login that embeds a surname) | `5acbb740f953ff34c6fa684db6cdff7ef9ce0edd7c2cc89cff0695e3fa399931` | 11943 | `CERT-CAMPAIGN-2026-11-07/signing/SIGN-1_QUALX/SIGNED_DECISIONS.md` |

Note: both kinds are identifiers that already appear in tracked repository files (the account handle in over a hundred tracked documents, including the record that cites these digests; the profile user name in several tracked qualification records). They were held back because the scoped rule is that any scan hit stops publication of that file. Releasing them needs an Operator decision that these identifiers may appear in the published copy.

## Not published (outside this batch)

| Local source | sha256 | Size (bytes) | Reason |
| --- | --- | ---: | --- |
| `.tmp/session/arch-freeze-g6-wave1-audit/G6_WAVE1_INDEPENDENT_AUDIT_REPORT.md` | `84121b78d13f5177bc2f86257445e5dd52bc211d2e2d991d55b851faee604b4a` | 13189 | Listed in the planning manifest as optional (finding A2-015, not one of the PT-D03 findings A2-026 and A7-026); belongs to the batch that owns the ARCH-FREEZE G6 record. Scan result: no hit. |

## Directory names

Stage directories `QUAL-X`, `QUAL-X-P0`, `SKILL-X` and `TOOL-LIFE-X` follow roadmap stage names. `CERT-CAMPAIGN-2026-11-07` is the name of a campaign, not a roadmap stage; it is used only because the planning manifest proposed it, and the Operator has not confirmed the name (open question in the pull request).
