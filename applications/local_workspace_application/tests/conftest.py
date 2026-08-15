# © Artur Czarnecki. All rights reserved.

"""Pytest registration for local-workspace application test support."""

pytest_plugins = (
    "applications.local_workspace_application.tests._vendor_knowledge_e2e_plugin",
)
