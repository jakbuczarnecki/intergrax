# © Artur Czarnecki. All rights reserved.

"""Canonical behavioral proof bindings for EBH-5 REPLACEABLE — QUALIFIED rows."""

from __future__ import annotations

from typing import Final

_BEHAVIORAL_PROOFS_MODULE: Final[str] = "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py"

QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE: Final[dict[str, str]] = {
    "persistence.relational_sqlite_lab": (
        f"{_BEHAVIORAL_PROOFS_MODULE}::test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile"
    ),
    "int_config.configure_existing": (
        f"{_BEHAVIORAL_PROOFS_MODULE}::test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service"
    ),
    "observability.otlp_transport": (
        f"{_BEHAVIORAL_PROOFS_MODULE}::test_ebh5_behavioral_otlp_transport_port_injection"
    ),
    "tokenizer.selection": (
        f"{_BEHAVIORAL_PROOFS_MODULE}::test_ebh5_behavioral_tokenizer_alternate_without_consumer_change"
    ),
    "integration.key_value_cache": (
        f"{_BEHAVIORAL_PROOFS_MODULE}::test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition"
    ),
}

QUALIFIED_CONCERN_ALTERNATE_REF: Final[dict[str, str]] = {
    "persistence.relational_sqlite_lab": ("IntegrationProfile relational_store=SQLITE + register_sqlite_integration"),
    "int_config.configure_existing": ("ExistingCapabilityConfigurationRealizationStrategy SPI (sqlite reference)"),
    "observability.otlp_transport": "OtlpTransportPort injected RecordingTransport",
    "tokenizer.selection": "TokenizerRegistry default_tokenizer_id simple vs tiktoken",
    "integration.key_value_cache": "CustomMemoryKvPlugin via register_integration_plugin",
}

QUALIFIED_CONCERN_REGRESSION_GATES: Final[dict[str, tuple[str, ...]]] = {
    "persistence.relational_sqlite_lab": (
        "test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile",
        "test_ebh_5_l2_integrity_gates",
    ),
    "int_config.configure_existing": (
        "test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service",
        "test_ebh_5_l2_integrity_gates",
    ),
    "observability.otlp_transport": (
        "test_ebh5_behavioral_otlp_transport_port_injection",
        "test_ebh_5_l2_integrity_gates",
    ),
    "tokenizer.selection": (
        "test_ebh5_behavioral_tokenizer_alternate_without_consumer_change",
        "test_ebh_5_l2_integrity_gates",
    ),
    "integration.key_value_cache": (
        "test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition",
        "test_ebh_5_l2_integrity_gates",
    ),
}

HISTORICAL_SUPPLEMENTAL_PROOF_NOTES: Final[tuple[str, ...]] = (
    "INT-CONFIG-REAL-X qualification @ a59744517b92847f55def1db22826d17d89ee155 — supplemental; "
    "not bound to current HEAD without EBH-5 behavioral re-run.",
    "HARNESS-W5 RecordingTransport @ 5a87046 — supplemental OTLP harness evidence; "
    "EBH-5 uses test_ebh5_behavioral_otlp_transport_port_injection at execution SHA.",
)

QUALIFIED_CONCERN_IDS: Final[frozenset[str]] = frozenset(QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE)
