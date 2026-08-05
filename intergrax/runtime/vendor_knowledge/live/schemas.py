from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.runtime.vendor_knowledge.live.identity import LIVE_CONTRACT_VERSION

_SCHEMA_REF_RE = re.compile(
    r"^schema://vendor-knowledge/live/"
    r"(?P<provider>[a-z][a-z0-9]*(?:_[a-z0-9]+)*)/"
    r"(?P<source>[a-z][a-z0-9]*(?:_[a-z0-9]+)*)/"
    r"(?P<operation>search|list|read|thread\.read|child\.read|content\.read)/"
    r"(?P<role>request|result)/v(?P<version>[1-9][0-9]*)$"
)


class SchemaRoleV1(StrEnum):
    REQUEST = "request"
    RESULT = "result"


class SchemaRegistrationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_ref: str = Field(..., min_length=1, max_length=256)
    role: SchemaRoleV1
    model: type[BaseModel]
    contract_version: str = Field(..., min_length=1, max_length=32)

    @field_validator("schema_ref")
    @classmethod
    def _valid_ref(cls, value: str) -> str:
        if _SCHEMA_REF_RE.fullmatch(value) is None:
            raise ValueError("live_schema_reference_invalid")
        return value

    @field_validator("contract_version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if value != LIVE_CONTRACT_VERSION:
            raise ValueError("live_schema_version_unsupported")
        return value

    @field_validator("model")
    @classmethod
    def _strict_model(cls, value: type[BaseModel]) -> type[BaseModel]:
        if not isinstance(value, type) or not issubclass(value, BaseModel):
            raise TypeError("live_schema_model_invalid")
        config = value.model_config
        if (
            config.get("extra") != "forbid"
            or config.get("frozen") is not True
            or config.get("strict") is not True
        ):
            raise ValueError("live_schema_model_not_strict_immutable")
        return value

    @property
    def parsed_role(self) -> SchemaRoleV1:
        match = _SCHEMA_REF_RE.fullmatch(self.schema_ref)
        assert match is not None
        return SchemaRoleV1(match.group("role"))

    @property
    def parsed_version(self) -> str:
        match = _SCHEMA_REF_RE.fullmatch(self.schema_ref)
        assert match is not None
        return match.group("version")

    def _validate_ref_contract(self) -> None:
        if self.parsed_role is not self.role:
            raise ValueError("live_schema_role_mismatch")
        if self.parsed_version != self.contract_version:
            raise ValueError("live_schema_version_mismatch")


class SchemaRegistryV1:
    """Immutable exact-reference registry published as one complete snapshot."""

    def __init__(self, registrations: tuple[SchemaRegistrationV1, ...] = ()) -> None:
        entries: dict[tuple[str, SchemaRoleV1], type[BaseModel]] = {}
        for registration in registrations:
            registration._validate_ref_contract()
            key = (registration.schema_ref, registration.role)
            if key in entries or any(
                existing_ref == registration.schema_ref
                for existing_ref, _existing_role in entries
            ):
                raise ValueError("duplicate_live_schema_reference")
            entries[key] = registration.model
        self._entries: Mapping[tuple[str, SchemaRoleV1], type[BaseModel]] = (
            MappingProxyType(entries)
        )

    def resolve(
        self,
        *,
        schema_ref: str,
        role: SchemaRoleV1,
        contract_version: str,
    ) -> type[BaseModel]:
        if not isinstance(schema_ref, str) or not isinstance(role, SchemaRoleV1):
            raise TypeError("live_schema_reference_or_role_invalid")
        if contract_version != LIVE_CONTRACT_VERSION:
            raise LookupError("live_schema_unavailable")
        match = _SCHEMA_REF_RE.fullmatch(schema_ref)
        if match is None or match.group("version") != contract_version:
            raise LookupError("live_schema_unavailable")
        if match.group("role") != role.value:
            raise LookupError("live_schema_role_mismatch")
        model = self._entries.get((schema_ref, role))
        if model is None:
            raise LookupError("live_schema_unavailable")
        return model

    def resolve_request(self, schema_ref: str, contract_version: str) -> type[BaseModel]:
        return self.resolve(
            schema_ref=schema_ref,
            role=SchemaRoleV1.REQUEST,
            contract_version=contract_version,
        )

    def resolve_result(self, schema_ref: str, contract_version: str) -> type[BaseModel]:
        return self.resolve(
            schema_ref=schema_ref,
            role=SchemaRoleV1.RESULT,
            contract_version=contract_version,
        )

    @property
    def registrations(self) -> Mapping[tuple[str, SchemaRoleV1], type[BaseModel]]:
        return self._entries
