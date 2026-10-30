# © Artur Czarnecki. All rights reserved.

"""Canonical behavioral proof bindings for EBH-5 REPLACEABLE — QUALIFIED rows."""

from __future__ import annotations

from typing import Final

_L2: Final[tuple[str, ...]] = ("test_ebh_5_l2_integrity_gates",)

QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE: Final[dict[str, str]] = {
    "agent.effective_profile": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_agent_effective_profile"
    ),
    "application.composition_root": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_application_composition_root"
    ),
    "distributed.sqlite_kv": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_distributed_sqlite_kv"
    ),
    "env.integration_slug": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_env_integration_slug"
    ),
    "execution.integration_bound_resolution": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_execution_integration_bound_resolution"
    ),
    "int_config.configure_existing": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service"
    ),
    "integration.billing_meter": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_billing_meter"
    ),
    "integration.browser_automation": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_browser_automation"
    ),
    "integration.ci_cd": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_ci_cd"
    ),
    "integration.cloud_platform": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_cloud_platform"
    ),
    "integration.collaboration_suite": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_collaboration_suite"
    ),
    "integration.conversation_channel": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_conversation_channel"
    ),
    "integration.crm": ("tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_crm"),
    "integration.document_parser": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_document_parser"
    ),
    "integration.document_store": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_document_store"
    ),
    "integration.embedding_provider": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_embedding_provider"
    ),
    "integration.external_work": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_external_work"
    ),
    "integration.feature_flag": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_feature_flag"
    ),
    "integration.graph_store": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_graph_store"
    ),
    "integration.identity_provider": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_identity_provider"
    ),
    "integration.issue_tracker": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_issue_tracker"
    ),
    "integration.key_value_cache": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition"
    ),
    "integration.llm_guardrail": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_llm_guardrail"
    ),
    "integration.managed_retrieval": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_managed_retrieval"
    ),
    "integration.message_bus": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_message_bus"
    ),
    "integration.ml_inference_host": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_ml_inference_host"
    ),
    "integration.model_serving_runtime": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_model_serving_runtime"
    ),
    "integration.notification_channel": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_notification_channel"
    ),
    "integration.object_storage": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_object_storage"
    ),
    "integration.observability_backend": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_observability_backend"
    ),
    "integration.relational_store": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py"
        "::test_ebh5_pkgc_v2_relational_sqlite_duckdb_dual_slug; "
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py"
        "::test_ebh5_pkgc_integration_relational_store"
    ),
    "integration.rerank_provider": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_rerank_provider"
    ),
    "integration.sandbox_host": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_sandbox_host"
    ),
    "integration.search_provider": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_search_provider"
    ),
    "integration.secrets_store": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_secrets_store"
    ),
    "integration.security_scanner": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_security_scanner"
    ),
    "integration.speech_provider": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_speech_provider"
    ),
    "integration.vector_store": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_vector_store"
    ),
    "integration.vision_serving": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_vision_serving"
    ),
    "integration.wiki_knowledge": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_wiki_knowledge"
    ),
    "integration.workflow_orchestrator": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_integration_workflow_orchestrator"
    ),
    "llm.chat_completion": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_llm_chat_completion"
    ),
    "llm.embedding": ("tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_llm_embedding"),
    "llm.fallback_policy": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_llm_fallback_policy"
    ),
    "llm.model_routing_qualifier": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_llm_model_routing_qualifier"
    ),
    "llm.rerank": ("tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_llm_rerank"),
    "memory.sqlite.conversational": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_memory_sqlite_conversational"
    ),
    "observability.otlp_transport": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_otlp_transport_via_sanctioned_composition_wiring"
    ),
    "persistence.relational_sqlite_lab": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile"
    ),
    "plugin.integration_catalog": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_plugin_integration_catalog"
    ),
    "plugin.llm_adapter_registry": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_plugin_llm_adapter_registry"
    ),
    "prodq.colocated_runtime_adapter_isinstance": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_prodq_colocated_runtime_adapter_isinstance"
    ),
    "speech.provider": ("tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_speech_provider"),
    "tokenizer.selection": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_tokenizer_selection_via_bootstrap_composition"
    ),
    "tool.marketplace_activation": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_tool_marketplace_activation"
    ),
    "tool.observability_backend": (
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py::test_ebh5_pkgc_tool_observability_backend"
    ),
}

QUALIFIED_CONCERN_ALTERNATE_REF: Final[dict[str, str]] = {
    "agent.effective_profile": "Sanctioned factory @ Package C (EBH-7)",
    "application.composition_root": "Sanctioned factory @ Package C (EBH-7)",
    "distributed.sqlite_kv": "Sanctioned factory @ Package C (EBH-7)",
    "env.integration_slug": "Sanctioned factory @ Package C (EBH-7)",
    "execution.integration_bound_resolution": "Sanctioned factory @ Package C (EBH-7)",
    "int_config.configure_existing": "ExistingCapabilityConfigurationRealizationStrategy SPI",
    "integration.billing_meter": "Sanctioned factory @ Package C (EBH-7)",
    "integration.browser_automation": "Sanctioned factory @ Package C (EBH-7)",
    "integration.ci_cd": "Sanctioned factory @ Package C (EBH-7)",
    "integration.cloud_platform": "Sanctioned factory @ Package C (EBH-7)",
    "integration.collaboration_suite": "Sanctioned factory @ Package C (EBH-7)",
    "integration.conversation_channel": "Sanctioned factory @ Package C (EBH-7)",
    "integration.crm": "Sanctioned factory @ Package C (EBH-7)",
    "integration.document_parser": "Sanctioned factory @ Package C (EBH-7)",
    "integration.document_store": "Sanctioned factory @ Package C (EBH-7)",
    "integration.embedding_provider": "Sanctioned factory @ Package C (EBH-7)",
    "integration.external_work": "Sanctioned factory @ Package C (EBH-7)",
    "integration.feature_flag": "Sanctioned factory @ Package C (EBH-7)",
    "integration.graph_store": "Sanctioned factory @ Package C (EBH-7)",
    "integration.identity_provider": "Sanctioned factory @ Package C (EBH-7)",
    "integration.issue_tracker": "Sanctioned factory @ Package C (EBH-7)",
    "integration.key_value_cache": "CustomMemoryKvPlugin via register_integration_plugin",
    "integration.llm_guardrail": "Sanctioned factory @ Package C (EBH-7)",
    "integration.managed_retrieval": "Sanctioned factory @ Package C (EBH-7)",
    "integration.message_bus": "Sanctioned factory @ Package C (EBH-7)",
    "integration.ml_inference_host": "Sanctioned factory @ Package C (EBH-7)",
    "integration.model_serving_runtime": "Sanctioned factory @ Package C (EBH-7)",
    "integration.notification_channel": "Sanctioned factory @ Package C (EBH-7)",
    "integration.object_storage": "Sanctioned factory @ Package C (EBH-7)",
    "integration.observability_backend": "Sanctioned factory @ Package C (EBH-7)",
    "integration.relational_store": "Sanctioned factory @ Package C (EBH-7)",
    "integration.rerank_provider": "Sanctioned factory @ Package C (EBH-7)",
    "integration.sandbox_host": "Sanctioned factory @ Package C (EBH-7)",
    "integration.search_provider": "Sanctioned factory @ Package C (EBH-7)",
    "integration.secrets_store": "Sanctioned factory @ Package C (EBH-7)",
    "integration.security_scanner": "Sanctioned factory @ Package C (EBH-7)",
    "integration.speech_provider": "Sanctioned factory @ Package C (EBH-7)",
    "integration.vector_store": "Sanctioned factory @ Package C (EBH-7)",
    "integration.vision_serving": "Sanctioned factory @ Package C (EBH-7)",
    "integration.wiki_knowledge": "Sanctioned factory @ Package C (EBH-7)",
    "integration.workflow_orchestrator": "Sanctioned factory @ Package C (EBH-7)",
    "llm.chat_completion": "Sanctioned factory @ Package C (EBH-7)",
    "llm.embedding": "Sanctioned factory @ Package C (EBH-7)",
    "llm.fallback_policy": "Sanctioned factory @ Package C (EBH-7)",
    "llm.model_routing_qualifier": "Sanctioned factory @ Package C (EBH-7)",
    "llm.rerank": "Sanctioned factory @ Package C (EBH-7)",
    "memory.sqlite.conversational": "Sanctioned factory @ Package C (EBH-7)",
    "observability.otlp_transport": "ApplicationRuntimeEventDeliveryWiring + factory",
    "persistence.relational_sqlite_lab": "IntegrationProfile relational_store=SQLITE",
    "plugin.integration_catalog": "Sanctioned factory @ Package C (EBH-7)",
    "plugin.llm_adapter_registry": "Sanctioned factory @ Package C (EBH-7)",
    "prodq.colocated_runtime_adapter_isinstance": "Sanctioned factory @ Package C (EBH-7)",
    "speech.provider": "Sanctioned factory @ Package C (EBH-7)",
    "tokenizer.selection": "create_default_tokenizer_engine + TokenizerManager",
    "tool.marketplace_activation": "Sanctioned factory @ Package C (EBH-7)",
    "tool.observability_backend": "Sanctioned factory @ Package C (EBH-7)",
}

QUALIFIED_CONCERN_REGRESSION_GATES: Final[dict[str, tuple[str, ...]]] = {
    "agent.effective_profile": ("test_ebh5_pkgc_agent_effective_profile", *_L2),
    "application.composition_root": ("test_ebh5_pkgc_application_composition_root", *_L2),
    "distributed.sqlite_kv": ("test_ebh5_pkgc_distributed_sqlite_kv", *_L2),
    "env.integration_slug": ("test_ebh5_pkgc_env_integration_slug", *_L2),
    "execution.integration_bound_resolution": ("test_ebh5_pkgc_execution_integration_bound_resolution", *_L2),
    "int_config.configure_existing": (
        "test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service",
        "test_ebh_5_l2_integrity_gates",
    ),
    "integration.billing_meter": ("test_ebh5_pkgc_integration_billing_meter", *_L2),
    "integration.browser_automation": ("test_ebh5_pkgc_integration_browser_automation", *_L2),
    "integration.ci_cd": ("test_ebh5_pkgc_integration_ci_cd", *_L2),
    "integration.cloud_platform": ("test_ebh5_pkgc_integration_cloud_platform", *_L2),
    "integration.collaboration_suite": ("test_ebh5_pkgc_integration_collaboration_suite", *_L2),
    "integration.conversation_channel": ("test_ebh5_pkgc_integration_conversation_channel", *_L2),
    "integration.crm": ("test_ebh5_pkgc_integration_crm", *_L2),
    "integration.document_parser": ("test_ebh5_pkgc_integration_document_parser", *_L2),
    "integration.document_store": ("test_ebh5_pkgc_integration_document_store", *_L2),
    "integration.embedding_provider": ("test_ebh5_pkgc_integration_embedding_provider", *_L2),
    "integration.external_work": ("test_ebh5_pkgc_integration_external_work", *_L2),
    "integration.feature_flag": ("test_ebh5_pkgc_integration_feature_flag", *_L2),
    "integration.graph_store": ("test_ebh5_pkgc_integration_graph_store", *_L2),
    "integration.identity_provider": ("test_ebh5_pkgc_integration_identity_provider", *_L2),
    "integration.issue_tracker": ("test_ebh5_pkgc_integration_issue_tracker", *_L2),
    "integration.key_value_cache": (
        "test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition",
        "test_ebh_5_l2_integrity_gates",
    ),
    "integration.llm_guardrail": ("test_ebh5_pkgc_integration_llm_guardrail", *_L2),
    "integration.managed_retrieval": ("test_ebh5_pkgc_integration_managed_retrieval", *_L2),
    "integration.message_bus": ("test_ebh5_pkgc_integration_message_bus", *_L2),
    "integration.ml_inference_host": ("test_ebh5_pkgc_integration_ml_inference_host", *_L2),
    "integration.model_serving_runtime": ("test_ebh5_pkgc_integration_model_serving_runtime", *_L2),
    "integration.notification_channel": ("test_ebh5_pkgc_integration_notification_channel", *_L2),
    "integration.object_storage": ("test_ebh5_pkgc_integration_object_storage", *_L2),
    "integration.observability_backend": ("test_ebh5_pkgc_integration_observability_backend", *_L2),
    "integration.relational_store": ("test_ebh5_pkgc_integration_relational_store", *_L2),
    "integration.rerank_provider": ("test_ebh5_pkgc_integration_rerank_provider", *_L2),
    "integration.sandbox_host": ("test_ebh5_pkgc_integration_sandbox_host", *_L2),
    "integration.search_provider": ("test_ebh5_pkgc_integration_search_provider", *_L2),
    "integration.secrets_store": ("test_ebh5_pkgc_integration_secrets_store", *_L2),
    "integration.security_scanner": ("test_ebh5_pkgc_integration_security_scanner", *_L2),
    "integration.speech_provider": ("test_ebh5_pkgc_integration_speech_provider", *_L2),
    "integration.vector_store": ("test_ebh5_pkgc_integration_vector_store", *_L2),
    "integration.vision_serving": ("test_ebh5_pkgc_integration_vision_serving", *_L2),
    "integration.wiki_knowledge": ("test_ebh5_pkgc_integration_wiki_knowledge", *_L2),
    "integration.workflow_orchestrator": ("test_ebh5_pkgc_integration_workflow_orchestrator", *_L2),
    "llm.chat_completion": ("test_ebh5_pkgc_llm_chat_completion", *_L2),
    "llm.embedding": ("test_ebh5_pkgc_llm_embedding", *_L2),
    "llm.fallback_policy": ("test_ebh5_pkgc_llm_fallback_policy", *_L2),
    "llm.model_routing_qualifier": ("test_ebh5_pkgc_llm_model_routing_qualifier", *_L2),
    "llm.rerank": ("test_ebh5_pkgc_llm_rerank", *_L2),
    "memory.sqlite.conversational": ("test_ebh5_pkgc_memory_sqlite_conversational", *_L2),
    "observability.otlp_transport": (
        "test_ebh5_behavioral_otlp_transport_via_sanctioned_composition_wiring",
        "test_ebh_5_l2_integrity_gates",
    ),
    "persistence.relational_sqlite_lab": (
        "test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile",
        "test_ebh_5_l2_integrity_gates",
    ),
    "plugin.integration_catalog": ("test_ebh5_pkgc_plugin_integration_catalog", *_L2),
    "plugin.llm_adapter_registry": ("test_ebh5_pkgc_plugin_llm_adapter_registry", *_L2),
    "prodq.colocated_runtime_adapter_isinstance": ("test_ebh5_pkgc_prodq_colocated_runtime_adapter_isinstance", *_L2),
    "speech.provider": ("test_ebh5_pkgc_speech_provider", *_L2),
    "tokenizer.selection": (
        "test_ebh5_behavioral_tokenizer_selection_via_bootstrap_composition",
        "test_ebh_5_l2_integrity_gates",
    ),
    "tool.marketplace_activation": ("test_ebh5_pkgc_tool_marketplace_activation", *_L2),
    "tool.observability_backend": ("test_ebh5_pkgc_tool_observability_backend", *_L2),
}

HISTORICAL_SUPPLEMENTAL_PROOF_NOTES: Final[tuple[str, ...]] = (
    "EBH-7 Package C batch qualifies remaining REPLACEABLE NOT YET @ sanctioned composition paths.",
)

QUALIFIED_CONCERN_IDS: Final[frozenset[str]] = frozenset(QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE)
