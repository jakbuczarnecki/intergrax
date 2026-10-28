# EBH-5 — CROSS_LAYER_REPLACEABILITY_MATRIX (summary)

| Field | Value |
| --- | --- |
| **SSOT** | [`tests/qualification/ebh_5/_replaceability_matrix.py`](../../../tests/qualification/ebh_5/_replaceability_matrix.py) |
| **Matrix rows** | **56** (all `extensibility_promised ∈ {yes, partial}`) |
| **QUALIFIED overrides** | `persistence.relational_sqlite_lab`, `int_config.configure_existing`, `observability.otlp_transport`, `tokenizer.selection` |
| **Gate** | `test_ebh5_q07_no_undocumented_qualified` |

Full RPL-A..H columns per row in Python SSOT. Independent audit MUST verify each `REPLACEABLE — QUALIFIED` row has non-monkeypatch behavioral proof @ closure SHA.
