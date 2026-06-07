# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Protocol conformance helpers for provider tests (Phase M.5)."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
from intergrax.integrations.contracts.issue_tracker import IssueCreator, IssueTracker
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.contracts.graph_store import GraphStore
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.integrations.contracts.vision_serving import VisionServingBackend
from intergrax.integrations.contracts.ml_inference_host import MlInferenceHostBackend
from intergrax.integrations.contracts.billing_meter import BillingMeterBackend
from intergrax.integrations.contracts.crm import CrmBackend


def assert_implements(instance: object, protocol: type[T]) -> T:
    if not isinstance(instance, protocol):
        raise AssertionError(
            f"Expected instance of {protocol.__name__}, got {type(instance)!r}"
        )
    return instance


def assert_relational_store(instance: object) -> RelationalStore:
    return assert_implements(instance, RelationalStore)


def assert_key_value_cache(instance: object) -> KeyValueCache:
    return assert_implements(instance, KeyValueCache)


def assert_message_bus(instance: object) -> MessageBus:
    return assert_implements(instance, MessageBus)


def assert_search_provider(instance: object) -> SearchProvider:
    return assert_implements(instance, SearchProvider)


def assert_notification_channel(instance: object) -> NotificationChannel:
    return assert_implements(instance, NotificationChannel)


def assert_interaction_surface(instance: object) -> InteractionSurface:
    return assert_implements(instance, InteractionSurface)


def assert_issue_tracker(instance: object) -> IssueTracker:
    return assert_implements(instance, IssueTracker)


def assert_issue_creator(instance: object) -> IssueCreator:
    return assert_implements(instance, IssueCreator)


def assert_wiki_knowledge(instance: object) -> WikiKnowledge:
    return assert_implements(instance, WikiKnowledge)


def assert_observability_backend(instance: object) -> ObservabilityBackend:
    return assert_implements(instance, ObservabilityBackend)


def assert_browser_automation(instance: object) -> BrowserAutomation:
    return assert_implements(instance, BrowserAutomation)


def assert_cloud_platform(instance: object) -> CloudPlatform:
    return assert_implements(instance, CloudPlatform)


def assert_collaboration_suite(instance: object) -> CollaborationSuite:
    return assert_implements(instance, CollaborationSuite)


def assert_document_store(instance: object) -> DocumentStore:
    return assert_implements(instance, DocumentStore)


def assert_object_storage(instance: object) -> ObjectStorage:
    return assert_implements(instance, ObjectStorage)


def assert_vector_store(instance: object) -> VectorStore:
    if not isinstance(instance, VectorStore):
        raise AssertionError(
            f"Expected instance of VectorStore, got {type(instance)!r}"
        )
    return instance


def assert_secrets_store(instance: object) -> SecretsStore:
    return assert_implements(instance, SecretsStore)


def assert_graph_store(instance: object) -> GraphStore:
    return assert_implements(instance, GraphStore)


def assert_feature_flag_backend(instance: object) -> FeatureFlagBackend:
    return assert_implements(instance, FeatureFlagBackend)


def assert_ci_cd_backend(instance: object) -> CiCdBackend:
    return assert_implements(instance, CiCdBackend)


def assert_security_scanner_backend(instance: object) -> SecurityScannerBackend:
    return assert_implements(instance, SecurityScannerBackend)


def assert_sandbox_host_backend(instance: object) -> SandboxHostBackend:
    return assert_implements(instance, SandboxHostBackend)


def assert_identity_provider_backend(instance: object) -> IdentityProviderBackend:
    return assert_implements(instance, IdentityProviderBackend)


def assert_speech_provider_backend(instance: object) -> SpeechProviderBackend:
    return assert_implements(instance, SpeechProviderBackend)


def assert_workflow_orchestrator_backend(instance: object) -> WorkflowOrchestratorBackend:
    return assert_implements(instance, WorkflowOrchestratorBackend)


def assert_vision_serving_backend(instance: object) -> VisionServingBackend:
    return assert_implements(instance, VisionServingBackend)


def assert_ml_inference_host_backend(instance: object) -> MlInferenceHostBackend:
    return assert_implements(instance, MlInferenceHostBackend)


def assert_billing_meter_backend(instance: object) -> BillingMeterBackend:
    return assert_implements(instance, BillingMeterBackend)


def assert_crm_backend(instance: object) -> CrmBackend:
    return assert_implements(instance, CrmBackend)
