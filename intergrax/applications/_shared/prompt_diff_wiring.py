# © Artur Czarnecki. All rights reserved.

"""Prompt diff/compare wiring for managed catalogs (AUDIT-IDEAL-17.2)."""

from __future__ import annotations

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.prompts.registry.prompt_compare import PromptCompareResult, compare_prompt_documents
from intergrax.prompts.schema.prompt_schema import LocalizedPromptDocument


def prompt_compare_enabled(env: ApplicationEnvironmentProfile) -> bool:
    """Reference hosts expose prompt compare when catalog is loaded."""
    return env.application_profile in (ApplicationProfile.PRODUCT, ApplicationProfile.LAB)


def compare_prompt_documents_for_host(
    left: LocalizedPromptDocument,
    right: LocalizedPromptDocument,
) -> PromptCompareResult:
    """Compare two in-memory prompt documents (HTTP/API surface helper)."""
    return compare_prompt_documents(left, right)
