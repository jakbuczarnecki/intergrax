# © Artur Czarnecki. All rights reserved.

"""Mechanical SQLite contract-authority durability gap evidence."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from intergrax.agent_distribution.in_memory_stores import InMemoryAgentArtifactMetadataStore
from intergrax.agent_distribution.sqlite_stores import SqliteAgentDistributionStoreBundle
from intergrax.agent_distribution.stores import AgentArtifactMetadataStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SQLITE_STORES = _REPO_ROOT / "intergrax/agent_distribution/sqlite_stores.py"


@dataclass(frozen=True, slots=True)
class ContractAuthorityDurabilityEvidence:
    protocol_exposes_get: bool
    protocol_exposes_persist: bool
    in_memory_implements_get: bool
    in_memory_implements_persist: bool
    sqlite_bundle_provides_metadata_store: bool
    sqlite_overrides_get: bool
    sqlite_overrides_persist: bool
    sqlite_table_for_contract_authority: bool
    in_scope_blocker: bool
    evidence_paths: tuple[str, ...]


def _protocol_has_method(protocol: type, name: str) -> bool:
    return hasattr(protocol, name) and name in getattr(protocol, "__protocol_attrs__", ()) or name in protocol.__dict__


def _class_defines_method(cls: type, name: str) -> bool:
    return name in cls.__dict__ and callable(cls.__dict__[name])


@lru_cache(maxsize=1)
def build_contract_authority_durability_evidence() -> ContractAuthorityDurabilityEvidence:
    proto_get = hasattr(AgentArtifactMetadataStore, "get_package_contract_authority")
    proto_persist = hasattr(AgentArtifactMetadataStore, "persist_package_contract_authority")

    in_mem_get = _class_defines_method(InMemoryAgentArtifactMetadataStore, "get_package_contract_authority")
    in_mem_persist = _class_defines_method(
        InMemoryAgentArtifactMetadataStore, "persist_package_contract_authority"
    )

    bundle = SqliteAgentDistributionStoreBundle
    sqlite_provides = hasattr(bundle, "artifact_metadata_store")
    # Concrete SQLite metadata store class inside bundle factory
    source = _SQLITE_STORES.read_text(encoding="utf-8")
    sqlite_overrides_get = "def get_package_contract_authority" in source
    sqlite_overrides_persist = "def persist_package_contract_authority" in source

    sqlite_table = (
        "package_contract_authority" in source.lower()
        or "contract_authority" in source
        and "CREATE TABLE" in source
        and "authority" in source.lower()
    )
    # Mechanical: no dedicated payload table for PackageAgentContractAuthorityRecord in SQLite DDL
    tree = ast.parse(source)
    table_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in ("CREATE TABLE IF NOT EXISTS ", "CREATE TABLE "):
                if match in node.value:
                    rest = node.value.split(match, 1)[-1].split("(", 1)[0].strip()
                    if rest:
                        table_names.add(rest.split()[0])
    sqlite_table = any("contract" in t and "authority" in t for t in table_names)

    blocker = (
        proto_get
        and proto_persist
        and in_mem_get
        and in_mem_persist
        and not sqlite_overrides_persist
        and not sqlite_table
    )

    return ContractAuthorityDurabilityEvidence(
        protocol_exposes_get=proto_get,
        protocol_exposes_persist=proto_persist,
        in_memory_implements_get=in_mem_get,
        in_memory_implements_persist=in_mem_persist,
        sqlite_bundle_provides_metadata_store=sqlite_provides or "AgentArtifactMetadataStore" in source,
        sqlite_overrides_get=sqlite_overrides_get,
        sqlite_overrides_persist=sqlite_overrides_persist,
        sqlite_table_for_contract_authority=sqlite_table,
        in_scope_blocker=blocker,
        evidence_paths=(
            "intergrax/agent_distribution/stores.py",
            "intergrax/agent_distribution/in_memory_stores.py",
            "intergrax/agent_distribution/sqlite_stores.py",
            "intergrax/agent_distribution/agent_contract_authority.py",
        ),
    )
