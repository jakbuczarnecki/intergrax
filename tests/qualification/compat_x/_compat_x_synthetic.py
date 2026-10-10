# © Artur Czarnecki. All rights reserved.

"""Synthetic AST fixtures for COMPAT-X adversarial gates (not production code)."""

from __future__ import annotations

from typing import Final

SYNTHETIC_PUBLIC_CONTRACT_SOURCE: Final[str] = '''
from typing import Literal
from pydantic import BaseModel

class SyntheticPublicContract(BaseModel):
    schema_version: Literal["synthetic.v1"] = "synthetic.v1"
'''

SYNTHETIC_PERSISTED_WITHOUT_VERSION_SOURCE: Final[str] = '''
from pydantic import BaseModel

class PersistedThing(BaseModel):
    value: str

def persist_thing(thing: PersistedThing) -> dict:
    return thing.model_dump()
'''

SYNTHETIC_UNSANCTIONED_MIGRATION_SOURCE: Final[str] = '''
def quantum_state_rewriter(payload: dict) -> dict:
    return payload

class QuantumMigrator:
    def migrate(self, data: dict) -> dict:
        return data
'''

SYNTHETIC_PARALLEL_AUTHORITY_SOURCE: Final[str] = '''
class LegacyResolver:
    def resolve_provider(self, name: str) -> str:
        return name
'''

SYNTHETIC_MODULE_PATH_PUBLIC: Final[str] = "synthetic/qualification/public_contract_probe.py"
SYNTHETIC_MODULE_PATH_PERSISTED: Final[str] = "synthetic/qualification/persisted_probe.py"
SYNTHETIC_MODULE_PATH_MIGRATION: Final[str] = "synthetic/qualification/migration_probe.py"
SYNTHETIC_MODULE_PATH_PARALLEL: Final[str] = "synthetic/qualification/parallel_authority_probe.py"
