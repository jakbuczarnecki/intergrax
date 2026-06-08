# © Artur Czarnecki. All rights reserved.

"""Enable Tier-1 catalog tools when IntegrationProfile P6 slots are configured."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.browser.service import BROWSER_FETCH_PAGE_TOOL_ID
from intergrax.tools.providers.cache.service import (
    CACHE_DELETE_TOOL_ID,
    CACHE_GET_TOOL_ID,
    CACHE_LIST_KEYS_TOOL_ID,
    CACHE_SET_TOOL_ID,
)
from intergrax.tools.providers.billing.service import BILLING_LIST_USAGE_TOOL_ID, BILLING_RECORD_USAGE_TOOL_ID
from intergrax.tools.providers.crm.service import CRM_GET_ACCOUNT_TOOL_ID, CRM_LIST_CONTACTS_TOOL_ID, CRM_LIST_TICKETS_TOOL_ID
from intergrax.tools.providers.collaboration.service import (
    COLLABORATION_CREATE_EVENT_TOOL_ID,
    COLLABORATION_GET_MESSAGE_TOOL_ID,
    COLLABORATION_GET_USER_TOOL_ID,
    COLLABORATION_LIST_CALENDAR_TOOL_ID,
    COLLABORATION_LIST_MESSAGES_TOOL_ID,
    COLLABORATION_REPLY_MESSAGE_TOOL_ID,
    COLLABORATION_SEND_MAIL_TOOL_ID,
)
from intergrax.tools.providers.database.service import DATABASE_DESCRIBE_SCHEMA_TOOL_ID, DATABASE_EXECUTE_TOOL_ID, DATABASE_QUERY_TOOL_ID
from intergrax.tools.providers.graph.service import GRAPH_GET_NODE_TOOL_ID, GRAPH_RUN_QUERY_TOOL_ID
from intergrax.tools.providers.issues.service import (
    ISSUES_ADD_COMMENT_TOOL_ID,
    ISSUES_CREATE_ISSUE_TOOL_ID,
    ISSUES_GET_ISSUE_TOOL_ID,
    ISSUES_SEARCH_TOOL_ID,
)
from intergrax.tools.providers.knowledge.service import KNOWLEDGE_GET_PAGE_TOOL_ID, KNOWLEDGE_SEARCH_TOOL_ID
from intergrax.tools.providers.message_bus.service import (
    MESSAGE_BUS_CANCEL_TOOL_ID,
    MESSAGE_BUS_ENQUEUE_TOOL_ID,
    MESSAGE_BUS_GET_RESULT_TOOL_ID,
    MESSAGE_BUS_GET_STATUS_TOOL_ID,
    MESSAGE_BUS_LIST_TASKS_TOOL_ID,
    MESSAGE_BUS_PURGE_COMPLETED_TOOL_ID,
)
from intergrax.tools.providers.notify.service import NOTIFY_SCHEDULE_TOOL_ID, NOTIFY_SEND_BATCH_TOOL_ID, NOTIFY_SEND_TOOL_ID
from intergrax.tools.providers.observability.service import (
    ERRORS_CAPTURE_TOOL_ID,
    LOGS_SEARCH_TOOL_ID,
    LOGS_TAIL_TOOL_ID,
    METRICS_QUERY_INSTANT_TOOL_ID,
    METRICS_QUERY_RANGE_TOOL_ID,
    TRACES_QUERY_TOOL_ID,
)
from intergrax.tools.providers.pagerduty.service import (
    PAGERDUTY_ACKNOWLEDGE_INCIDENT_TOOL_ID,
    PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID,
)
from intergrax.tools.providers.identity.service import (
    IDENTITY_GET_USER_TOOL_ID,
    IDENTITY_LIST_TENANTS_TOOL_ID,
    IDENTITY_VERIFY_TOKEN_TOOL_ID,
)
from intergrax.tools.providers.platform.service import (
    PLATFORM_CANCEL_WORKFLOW_RUN_TOOL_ID,
    PLATFORM_DELETE_SECRET_TOOL_ID,
    PLATFORM_EVALUATE_FEATURE_FLAG_TOOL_ID,
    PLATFORM_GET_SECRET_TOOL_ID,
    PLATFORM_GET_WORKFLOW_RUN_TOOL_ID,
    PLATFORM_LIST_CHECK_SUITES_TOOL_ID,
    PLATFORM_LIST_WORKFLOW_RUNS_TOOL_ID,
    PLATFORM_PUT_SECRET_TOOL_ID,
)
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from intergrax.tools.providers.rag.rerank_service import RAG_RERANK_TOOL_ID
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID
from intergrax.tools.providers.websearch.fetch_batch_service import WEBSEARCH_FETCH_BATCH_TOOL_ID
from intergrax.tools.providers.websearch.read_url_service import WEBSEARCH_READ_URL_TOOL_ID
from intergrax.tools.providers.websearch.service import WEBSEARCH_TOOL_ID
from intergrax.tools.providers.records.service import (
    RECORDS_DELETE_TOOL_ID,
    RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
    RECORDS_COUNT_TOOL_ID,
    RECORDS_GET_TOOL_ID,
    RECORDS_PUT_TOOL_ID,
    RECORDS_QUERY_TOOL_ID,
)
from intergrax.tools.providers.security.service import SECURITY_SCAN_TOOL_ID, SECURITY_SUMMARIZE_FINDINGS_TOOL_ID
from intergrax.tools.providers.storage.service import (
    STORAGE_DELETE_TOOL_ID,
    STORAGE_EXISTS_TOOL_ID,
    STORAGE_GET_TOOL_ID,
    STORAGE_PRESIGNED_URL_TOOL_ID,
    STORAGE_PUT_TOOL_ID,
)
from intergrax.tools.providers.workflow.service import (
    WORKFLOW_CANCEL_RUN_TOOL_ID,
    WORKFLOW_FETCH_LOGS_TOOL_ID,
    WORKFLOW_LIST_RUNS_TOOL_ID,
    WORKFLOW_POLL_TOOL_ID,
    WORKFLOW_TRIGGER_TOOL_ID,
)
from intergrax.tools.registry.profile import ToolProfile

_CATEGORY_TOOL_IDS: dict[IntegrationCategory, tuple[str, ...]] = {
    IntegrationCategory.SECURITY_SCANNER: (
        SECURITY_SCAN_TOOL_ID,
        SECURITY_SUMMARIZE_FINDINGS_TOOL_ID,
    ),
    IntegrationCategory.IDENTITY_PROVIDER: (
        IDENTITY_VERIFY_TOKEN_TOOL_ID,
        IDENTITY_GET_USER_TOOL_ID,
        IDENTITY_LIST_TENANTS_TOOL_ID,
    ),
    IntegrationCategory.SANDBOX_HOST: ("sandbox.exec",),
    IntegrationCategory.WORKFLOW_ORCHESTRATOR: (
        WORKFLOW_TRIGGER_TOOL_ID,
        WORKFLOW_POLL_TOOL_ID,
        WORKFLOW_FETCH_LOGS_TOOL_ID,
        WORKFLOW_LIST_RUNS_TOOL_ID,
        WORKFLOW_CANCEL_RUN_TOOL_ID,
    ),
    IntegrationCategory.WIKI_KNOWLEDGE: (
        KNOWLEDGE_GET_PAGE_TOOL_ID,
        KNOWLEDGE_SEARCH_TOOL_ID,
        "confluence.get_page",
        "confluence.search_pages",
        "confluence.search",
    ),
    IntegrationCategory.ISSUE_TRACKER: (
        ISSUES_GET_ISSUE_TOOL_ID,
        ISSUES_ADD_COMMENT_TOOL_ID,
        ISSUES_SEARCH_TOOL_ID,
        ISSUES_CREATE_ISSUE_TOOL_ID,
        "jira.get_issue",
        "jira.add_comment",
        "jira.search_tasks",
        "gitlab.create_issue",
    ),
    IntegrationCategory.OBJECT_STORAGE: (
        STORAGE_GET_TOOL_ID,
        STORAGE_PUT_TOOL_ID,
        STORAGE_PRESIGNED_URL_TOOL_ID,
        STORAGE_DELETE_TOOL_ID,
        STORAGE_EXISTS_TOOL_ID,
    ),
    IntegrationCategory.RELATIONAL_STORE: (
        DATABASE_QUERY_TOOL_ID,
        DATABASE_EXECUTE_TOOL_ID,
        DATABASE_DESCRIBE_SCHEMA_TOOL_ID,
    ),
    IntegrationCategory.DOCUMENT_STORE: (
        RECORDS_GET_TOOL_ID,
        RECORDS_PUT_TOOL_ID,
        RECORDS_DELETE_TOOL_ID,
        RECORDS_QUERY_TOOL_ID,
        RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
        RECORDS_COUNT_TOOL_ID,
    ),
    IntegrationCategory.BROWSER_AUTOMATION: (BROWSER_FETCH_PAGE_TOOL_ID,),
    IntegrationCategory.SECRETS_STORE: (
        PLATFORM_GET_SECRET_TOOL_ID,
        PLATFORM_PUT_SECRET_TOOL_ID,
        PLATFORM_DELETE_SECRET_TOOL_ID,
    ),
    IntegrationCategory.FEATURE_FLAG: (PLATFORM_EVALUATE_FEATURE_FLAG_TOOL_ID,),
    IntegrationCategory.CI_CD: (
        PLATFORM_GET_WORKFLOW_RUN_TOOL_ID,
        PLATFORM_LIST_CHECK_SUITES_TOOL_ID,
        PLATFORM_LIST_WORKFLOW_RUNS_TOOL_ID,
        PLATFORM_CANCEL_WORKFLOW_RUN_TOOL_ID,
    ),
    IntegrationCategory.MESSAGE_BUS: (
        MESSAGE_BUS_ENQUEUE_TOOL_ID,
        MESSAGE_BUS_GET_STATUS_TOOL_ID,
        MESSAGE_BUS_GET_RESULT_TOOL_ID,
        MESSAGE_BUS_LIST_TASKS_TOOL_ID,
        MESSAGE_BUS_CANCEL_TOOL_ID,
        MESSAGE_BUS_PURGE_COMPLETED_TOOL_ID,
    ),
    IntegrationCategory.GRAPH_STORE: (GRAPH_RUN_QUERY_TOOL_ID, GRAPH_GET_NODE_TOOL_ID),
    IntegrationCategory.COLLABORATION_SUITE: (
        COLLABORATION_SEND_MAIL_TOOL_ID,
        COLLABORATION_LIST_MESSAGES_TOOL_ID,
        COLLABORATION_GET_MESSAGE_TOOL_ID,
        COLLABORATION_LIST_CALENDAR_TOOL_ID,
        COLLABORATION_GET_USER_TOOL_ID,
        COLLABORATION_REPLY_MESSAGE_TOOL_ID,
        COLLABORATION_CREATE_EVENT_TOOL_ID,
    ),
    IntegrationCategory.KEY_VALUE_CACHE: (
        CACHE_GET_TOOL_ID,
        CACHE_SET_TOOL_ID,
        CACHE_DELETE_TOOL_ID,
        CACHE_LIST_KEYS_TOOL_ID,
    ),
    IntegrationCategory.NOTIFICATION_CHANNEL: (
        NOTIFY_SEND_TOOL_ID,
        NOTIFY_SEND_BATCH_TOOL_ID,
        NOTIFY_SCHEDULE_TOOL_ID,
        PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID,
        PAGERDUTY_ACKNOWLEDGE_INCIDENT_TOOL_ID,
    ),
    IntegrationCategory.OBSERVABILITY_BACKEND: (
        ERRORS_CAPTURE_TOOL_ID,
        LOGS_SEARCH_TOOL_ID,
        LOGS_TAIL_TOOL_ID,
        METRICS_QUERY_INSTANT_TOOL_ID,
        METRICS_QUERY_RANGE_TOOL_ID,
        TRACES_QUERY_TOOL_ID,
    ),
    IntegrationCategory.BILLING_METER: (
        BILLING_RECORD_USAGE_TOOL_ID,
        BILLING_LIST_USAGE_TOOL_ID,
    ),
    IntegrationCategory.CRM: (
        CRM_GET_ACCOUNT_TOOL_ID,
        CRM_LIST_CONTACTS_TOOL_ID,
        CRM_LIST_TICKETS_TOOL_ID,
    ),
    IntegrationCategory.RERANK_PROVIDER: (RAG_RERANK_TOOL_ID,),
    IntegrationCategory.SEARCH_PROVIDER: (
        WEBSEARCH_TOOL_ID,
        WEBSEARCH_READ_URL_TOOL_ID,
        WEBSEARCH_FETCH_BATCH_TOOL_ID,
    ),
    IntegrationCategory.DOCUMENT_PARSER: (RAG_INGEST_TOOL_ID,),
    IntegrationCategory.VECTOR_STORE: (RAG_RETRIEVE_TOOL_ID, RAG_INGEST_TOOL_ID),
}


def integration_category_configured(
    integration_profile: IntegrationProfile,
    category: IntegrationCategory,
) -> bool:
    """Return whether ``category`` has a slug binding or pre-built instance."""
    if integration_profile.instance_for_category(category) is not None:
        return True
    return integration_profile.slug_for_category(category) is not None


def extend_tool_profile_for_integration(
    tool_profile: ToolProfile,
    integration_profile: IntegrationProfile | None,
) -> ToolProfile:
    """Append P6 integration-backed tool_ids when matching categories are configured."""
    if integration_profile is None:
        return tool_profile

    additions: list[str] = []
    for category, tool_ids in _CATEGORY_TOOL_IDS.items():
        if integration_category_configured(integration_profile, category):
            additions.extend(tool_ids)

    if not additions:
        return tool_profile

    enabled = list(tool_profile.enabled)
    for tool_id in additions:
        if tool_id not in enabled:
            enabled.append(tool_id)
    return tool_profile.model_copy(update={"enabled": enabled})
