# RW plan binding record

Status: historical binding authority, as cited; V6 supersedes train rules; live plan is RW_PLAN.md.

Tickets and approved plans bind scope to numbered versions of the RW plan that were never tracked. This folder publishes the cited versions so those references resolve (audit finding H3, ticket T-IGX-078-B). The text is unchanged; the only normalisation is CRLF to LF, so every line number stays valid. Both digests are recorded.

Limits: these files grant no write authority and are not maintained. Paths under `.tmp/` that they mention are untracked session artefacts and are not part of this record. The sub-revisions v18a and v18b are cited by no ticket or plan and are not published. The live plan is [`../RW_PLAN.md`](../RW_PLAN.md).

## Versions

| Version | Published file | Original untracked path | sha256 of the original (CRLF) | sha256 of the published copy (LF) | Lines |
|---|---|---|---|---|---|
| v6 | [`RW_PLAN_v6.md`](RW_PLAN_v6.md) | `.tmp/session/arch-freeze/rw-plan/work/v6/RW_PLAN.md` | `ddc8c482745305962f3f2b964c6a8970f4b395f578daedeb9773ea218a230d66` | `09511de15a74abec8c5b660292ea52e4250ceb10da7af6d85450e24610123446` | 1455 |
| v13 | [`RW_PLAN_v13.md`](RW_PLAN_v13.md) | `.tmp/session/arch-freeze/rw-plan/work/v13/RW_PLAN.md` | `ba9e4845fc23cb6f128ab580f0b78ee95efa32f89f55293fd738442ad049c61e` | `40f5e38ef6154bde62317f535e4d481d771cf79118b2ac0a5995e0ae9487fa81` | 1726 |
| v14 | [`RW_PLAN_v14.md`](RW_PLAN_v14.md) | `.tmp/session/arch-freeze/rw-plan/work/v14/RW_PLAN.md` | `3a211959f2d57a583e0242ac76477bd638d8020ad614d48c524f0fab1cda3462` | `6dc5f2973bae0d0ec4bd368a227c81be90fa0714796eb92cdab011b357c11a7f` | 1913 |
| v15 | [`RW_PLAN_v15.md`](RW_PLAN_v15.md) | `.tmp/session/arch-freeze/rw-plan/work/v15/RW_PLAN.md` | `392d1fa56b4374f08c368dae653c3833c9c42f274e2df9e04e38c79db5585eea` | `c3fdb5aa5c99504b34bd7956fe34bc4f8efc121c60fc958053a0b0c9978fc19b` | 1971 |
| v18 | [`RW_PLAN_v18.md`](RW_PLAN_v18.md) | `.tmp/session/arch-freeze/rw-plan/work/v18/RW_PLAN.md` | `3e0c59447cb5bca27e87cf75f1771c8ad6da1e4b354d584a6fb57945cd47a477` | `9b4e738d6e4105d6d028c2b0e21097aed457c62c6b452a3ca39ba5571a897aeb` | 2143 |
| v19 | [`RW_PLAN_v19.md`](RW_PLAN_v19.md) | `.tmp/session/arch-freeze/rw-plan/work/v19/RW_PLAN.md` | `0aeeb2944ffe749cb10864aeeefb07d59c4df46fe0b97e95ae6ffda7aee9b006` | `13628681d6d297ee4ee095d5eea8b405751dc96407008998f639c851dd416d7e` | 2184 |
| v21 | [`RW_PLAN_v21.md`](RW_PLAN_v21.md) | `.tmp/session/arch-freeze/rw-plan/RW_PLAN.md` | `a328394d80abac93facc409e8362c8f1911df92ecda240e06c6f3668b531d417` | `f47dd2e6d940902901030ff43cfa10e707217c5afcf9ebe5f8a301ef3aa20309` | 2212 |

## Section start lines

| Version | Sections |
|---|---|
| v6 | (a) L78, (b) L111, (c) L162, (d) L257, (e) L470, (f) L545, (g) L569, (h) L795, Appendix A1 L831 |
| v13 | (a) L168, (b) L201, (c) L254, (d) L349, (e) L575, (f) L661, (g) L685, (h) L911, Appendix A1 L952 |
| v14 | (a) L205, (b) L238, (c) L293, (d) L388, (e) L614, (f) L700, (g) L724, (h) L950, Appendix A1 L1071 |
| v15 | (a) L217, (b) L250, (c) L306, (d) L401, (e) L627, (f) L713, (g) L737, (h) L963, Appendix A1 L1095 |
| v18 | (a) L270, (b) L303, (c) L360, (d) L456, (e) L684, (f) L770, (g) L794, (h) L1021, Appendix A1 L1183 |
| v19 | (a) L298, (b) L331, (c) L388, (d) L485, (e) L713, (f) L799, (g) L823, (h) L1050, Appendix A1 L1222 |
| v21 | (a) L322, (b) L355, (c) L412, (d) L511, (e) L739, (f) L825, (g) L850, (h) L1077, Appendix A1 L1250 |

## Citing tickets and plans

`Plan lane` is the lane in the live `RW_PLAN.md`. `AF-CLOSE lane` is the WORKLIST row of the AF-CLOSE programme with its executing lane. The section column lists what the citing lines name next to a version.

| Citing file | Version | Sections, blocks and rows named | Ticket | Plan lane | AF-CLOSE lane |
|---|---|---|---|---|---|
| `governance/tickets/T-IGX-043-A.md` | [v13](RW_PLAN_v13.md) (first cite L5) | appendix A1 block `PT-FC1 (T-IGX-043-A)`; appendix A1 | T-IGX-043-A | lane_6 | WL-K1-E-01 (K1-A) |
| `governance/tickets/T-IGX-044-F.md` | [v13](RW_PLAN_v13.md) (first cite L7) | A1 block `PT-E05`; NT-ROW-010 | T-IGX-044-F | lane_4 | no WORKLIST row (landed or outside the pre-G2 lane work) |
| `governance/tickets/T-IGX-046-D.md` | [v13](RW_PLAN_v13.md) (first cite L5) | appendix A1 block `PT-D08 (T-IGX-046-D)` | T-IGX-046-D | lane_4 | no WORKLIST row (landed or outside the pre-G2 lane work) |
| `governance/tickets/T-IGX-051-A.md` | [v13](RW_PLAN_v13.md) (first cite L5), [v14](RW_PLAN_v14.md) (first cite L5), [v18](RW_PLAN_v18.md) (first cite L5) | A1 block `T-AF2-RW-TENANT (T-IGX-051-A)`; NT-ROW-065..074 | T-IGX-051-A | lane_11 | WL-K1-C-05 (K1-C) |
| `governance/tickets/T-IGX-051-B.md` | [v13](RW_PLAN_v13.md) (first cite L5) | A1 block `T-AF2-RW-SUPPLY (T-IGX-051-B)`; NT-ROW-075..096; NT-ROW-159 | T-IGX-051-B | lane_5 | WL-K1-C-03 (K1-C) |
| `governance/tickets/T-IGX-051-D.md` | [v13](RW_PLAN_v13.md) (first cite L5) | appendix A1 block `T-AF2-RW-RUNTIME (T-IGX-051-D)`; NT-ROW-005 | T-IGX-051-D | landed | WL-K1-A-08 (K1-A) |
| `governance/tickets/T-IGX-051-E.md` | [v13](RW_PLAN_v13.md) (first cite L5) | table T12; appendix A1 block `T-AF2-RW-RECOVERY (T-IGX-051-E)` | T-IGX-051-E | landed | WL-K1-A-08 (K1-A) |
| `governance/tickets/T-IGX-051-H.md` | [v13](RW_PLAN_v13.md) (first cite L5) | appendix A1 block `T-AF2-DEPS-BATCH (T-IGX-051-H)`; RWA-05 | T-IGX-051-H | lane_5 | WL-K1-C-02 (K1-C) |
| `governance/tickets/T-IGX-051-I.md` | [v15](RW_PLAN_v15.md) (first cite L5), [v18](RW_PLAN_v18.md) (first cite L11) | none named | T-IGX-051-I | lane_5 | WL-K1-C-09 (K1-C) |
| `governance/tickets/T-IGX-051-N.md` | [v21](RW_PLAN_v21.md) (first cite L47) | none named | T-IGX-051-N | lane_6 | WL-K1-C-03 (K1-C) |
| `governance/tickets/plans/PLAN_T-IGX-043-A.md` | [v13](RW_PLAN_v13.md) (first cite L3) | appendix A1 block `PT-FC1 (T-IGX-043-A)`; appendix A1; section c; section e | T-IGX-043-A | lane_6 | WL-K1-E-01 (K1-A) |
| `governance/tickets/plans/PLAN_T-IGX-043-B.md` | [v13](RW_PLAN_v13.md) (first cite L3) | A1 block `PT-FC2 (T-IGX-043-B)`; section (c) | T-IGX-043-B | lane_4 | WL-K1-E-02 (K1-A) |
| `governance/tickets/plans/PLAN_T-IGX-043-G.md` | [v6](RW_PLAN_v6.md) (first cite L21) | none named | T-IGX-043-G | lane_1 | WL-K1-E-07 (K1-B) |
| `governance/tickets/plans/PLAN_T-IGX-043-I.md` | [v13](RW_PLAN_v13.md) (first cite L35), [v21](RW_PLAN_v21.md) (first cite L35) | appendix A1 block `PT-WD (T-IGX-043-I)`; sections (b), (c), (d); section (d); NT-ROW-008 | T-IGX-043-I | lane_4 | WL-K1-E-09 (K3) |
| `governance/tickets/plans/PLAN_T-IGX-046-D.md` | [v13](RW_PLAN_v13.md) (first cite L12) | appendix A1 block `PT-D08 (T-IGX-046-D)` | T-IGX-046-D | lane_4 | no WORKLIST row (landed or outside the pre-G2 lane work) |
| `governance/tickets/plans/PLAN_T-IGX-051-A.md` | [v13](RW_PLAN_v13.md) (first cite L5), [v14](RW_PLAN_v14.md) (first cite L5), [v18](RW_PLAN_v18.md) (first cite L5) | A1 block `T-AF2-RW-TENANT (T-IGX-051-A)`; section a; NT-ROW-065..074; section d; section e; section c; :256-306; :163; A1 block `T-AF2-RW-TENANT` | T-IGX-051-A | lane_11 | WL-K1-C-05 (K1-C) |
| `governance/tickets/plans/PLAN_T-IGX-051-B.md` | [v13](RW_PLAN_v13.md) (first cite L5), [v18](RW_PLAN_v18.md) (first cite L3), [v19](RW_PLAN_v19.md) (first cite L283) | A1 block `T-AF2-RW-SUPPLY (T-IGX-051-B)`; section a; NT-ROW-075..096; NT-ROW-159; section d; section e | T-IGX-051-B | lane_5 | WL-K1-C-03 (K1-C) |
| `governance/tickets/plans/PLAN_T-IGX-051-D.md` | [v13](RW_PLAN_v13.md) (first cite L7) | appendix A1; NT-ROW-005; section g; appendix A1 block `T-AF2-RW-RUNTIME (T-IGX-051-D)` | T-IGX-051-D | landed | WL-K1-A-08 (K1-A) |
| `governance/tickets/plans/PLAN_T-IGX-051-E.md` | [v13](RW_PLAN_v13.md) (first cite L7) | table T12; appendix A1 block `T-AF2-RW-RECOVERY (T-IGX-051-E)`; NT-ROW-109..116; section (e); section (b); appendix A1 | T-IGX-051-E | landed | WL-K1-A-08 (K1-A) |
| `governance/tickets/plans/PLAN_T-IGX-051-H.md` | [v13](RW_PLAN_v13.md) (first cite L7) | appendix A1 block `T-AF2-DEPS-BATCH (T-IGX-051-H)`; RW_PLAN.md:1139-1146; appendix A1 | T-IGX-051-H | lane_5 | WL-K1-C-02 (K1-C) |
| `governance/tickets/plans/PLAN_T-IGX-051-I.md` | [v15](RW_PLAN_v15.md) (first cite L16), [v18](RW_PLAN_v18.md) (first cite L16) | none named | T-IGX-051-I | lane_5 | WL-K1-C-09 (K1-C) |
| `governance/tickets/plans/PLAN_T-IGX-051-N.md` | [v21](RW_PLAN_v21.md) (first cite L12) | A1 block `T-AF2-RW-SUPPLY`; :1929-1963; NT-ROW-075..078; :599-602; :1200; :1129; section c | T-IGX-051-N | lane_6 | WL-K1-C-03 (K1-C) |
