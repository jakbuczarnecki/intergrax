# EBH-5 — CROSS_LAYER_REPLACEABILITY_MATRIX (summary)

| Field | Value |
| --- | --- |
| **SSOT** | [`tests/qualification/ebh_5/_replaceability_matrix.py`](../../../tests/qualification/ebh_5/_replaceability_matrix.py) |
| **Matrix rows** | **56** (all `extensibility_promised ∈ {yes, partial}`) |
| **REPLACEABLE — QUALIFIED** | **5** — see registry SSOT |
| **QUALIFIED concern_ids** | `persistence.relational_sqlite_lab`, `int_config.configure_existing`, `observability.otlp_transport`, `tokenizer.selection`, `integration.key_value_cache` |
| **Gate** | `test_ebh5_q07_no_undocumented_qualified`, `test_ebh5_g06_03_matrix_md_lists_five_qualified` |

> **Historical summary note:** the five-row QUALIFIED figure above is the EBH-5 baseline (`33ecf19cc`) state. Later RPL-H dispositions (EBH-7 Package C and the fixed-substrate / internal ratifications; `GAP-RPL-51` **CLOSED** in [`EBH_7_GAP_REGISTER.md`](EBH_7_GAP_REGISTER.md)) are held in the SSOT module; do not read this summary as the current disposition. The historical content above is unchanged.

Full RPL-A..H columns per row in Python SSOT. Independent audit MUST verify each `REPLACEABLE — QUALIFIED` row has resolvable behavioral proof @ CERTIFICATION_HANDOFF_SHA.
