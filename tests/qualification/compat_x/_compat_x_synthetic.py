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

SYNTHETIC_PERSISTED_CONTRACT_VERSIONED_SOURCE: Final[str] = '''
from typing import Literal
from pydantic import BaseModel

class PersistedContract(BaseModel):
    schema_version: Literal["persisted_contract.v1"] = "persisted_contract.v1"
    value: str

def persist_contract(value: PersistedContract):
    return value.model_dump()
'''

SYNTHETIC_PERSISTED_CONTRACT_VERSION_REMOVED_SOURCE: Final[str] = '''
from pydantic import BaseModel

class PersistedContract(BaseModel):
    value: str

def persist_contract(value: PersistedContract):
    return value.model_dump()
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

SYNTHETIC_PARALLEL_PROVIDER_SELECTOR_SOURCE: Final[str] = '''
class LegacyBackendSelector:
    def select_backend(self, tenant_id: str) -> str:
        return tenant_id
'''

SYNTHETIC_PARALLEL_REGISTRY_SOURCE: Final[str] = '''
class CompatibilityRegistry:
    def register(self, name: str, impl: object) -> None:
        pass

    def resolve(self, name: str) -> object:
        return object()
'''

SYNTHETIC_PARALLEL_EXECUTOR_SOURCE: Final[str] = '''
class LegacyExecutor:
    def execute(self, payload: dict) -> dict:
        return payload
'''

SYNTHETIC_PARALLEL_AUTHORIZER_SOURCE: Final[str] = '''
class CompatibilityAuthorizer:
    def authorize(self, principal: str, action: str) -> bool:
        return True
'''

SYNTHETIC_TRANSLATION_ONLY_SOURCE: Final[str] = '''
def from_legacy_document(raw: dict) -> dict:
    return raw

def to_current_document(doc: dict) -> dict:
    return doc
'''

SYNTHETIC_DECODE_NORMALIZATION_SOURCE: Final[str] = '''
def decode_legacy_payload(blob: bytes) -> dict:
    return {}
'''

SYNTHETIC_MODULE_PATH_PUBLIC: Final[str] = "synthetic/qualification/public_contract_probe.py"
SYNTHETIC_MODULE_PATH_PERSISTED: Final[str] = "synthetic/qualification/persisted_probe.py"
SYNTHETIC_MODULE_PATH_PERSISTED_CONTRACT_VERSIONED: Final[str] = (
    "synthetic/qualification/persisted_contract_versioned_probe.py"
)
SYNTHETIC_MODULE_PATH_PERSISTED_CONTRACT_UNVERSIONED: Final[str] = (
    "synthetic/qualification/persisted_contract_unversioned_probe.py"
)
SYNTHETIC_MODULE_PATH_MIGRATION: Final[str] = "synthetic/qualification/migration_probe.py"
SYNTHETIC_MODULE_PATH_PARALLEL: Final[str] = "synthetic/qualification/parallel_authority_probe.py"
SYNTHETIC_MODULE_PATH_PARALLEL_PROBES: Final[str] = "synthetic/qualification/compat_shim/authority_probes.py"
