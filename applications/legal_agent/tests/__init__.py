# © Artur Czarnecki. All rights reserved.
"""
Legal Agent test layout (mirrors production package domains):

- ``config/`` — :class:`LegalAgentConfig`, product profiles, budgets
- ``memory/`` — memory policy and workspace session snapshot metrics
- ``governance/`` — tool-plan / response governance and execution policy sources
- ``pipeline/`` — end-to-end runs, tool bridge, Nexus integration
- ``steps/`` — isolated legal pipeline steps (Ollama-backed where noted)
- ``support/`` — shared fixtures and runtime builders
"""
