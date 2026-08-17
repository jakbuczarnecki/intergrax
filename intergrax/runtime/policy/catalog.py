# © Artur Czarnecki. All rights reserved.

"""Runtime-neutral Policy Catalog resolution core (Governed Execution G2C-1)."""

from __future__ import annotations

from collections.abc import Iterable

from intergrax.contracts.policy_catalog import PolicyDefinition


class PolicyCatalogError(Exception):
    """Base error for Policy Catalog resolution failures."""


class UnknownPolicyDefinitionError(PolicyCatalogError):
    """No definition with this policy_id exists at any version."""

    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id
        super().__init__(f"unknown policy_id: {policy_id!r}")


class UnsupportedPolicyDefinitionVersionError(PolicyCatalogError):
    """policy_id exists, but the requested definition version does not."""

    def __init__(self, policy_id: str, version: str) -> None:
        self.policy_id = policy_id
        self.version = version
        super().__init__(
            f"unsupported policy definition version for policy_id {policy_id!r}: {version!r}"
        )


class PolicyDefinitionConflictError(PolicyCatalogError):
    """Two definitions attempted to own the same (policy_id, version) identity."""

    def __init__(self, policy_id: str, version: str) -> None:
        self.policy_id = policy_id
        self.version = version
        super().__init__(
            f"duplicate policy definition identity: {policy_id!r}@{version!r}"
        )


def _normalize_lookup_identity(policy_id: str, version: str) -> tuple[str, str]:
    return policy_id.strip(), version.strip()


class PolicyCatalog:
    """Immutable catalog of PolicyDefinition values with exact identity resolution."""

    def __init__(
        self,
        definitions: Iterable[PolicyDefinition] = (),
    ) -> None:
        lookup: dict[tuple[str, str], PolicyDefinition] = {}
        versions_by_policy_id: dict[str, frozenset[str]] = {}
        ordered: list[PolicyDefinition] = []

        for definition in definitions:
            key = (definition.policy_id, definition.version)
            if key in lookup:
                raise PolicyDefinitionConflictError(
                    definition.policy_id,
                    definition.version,
                )
            lookup[key] = definition
            ordered.append(definition)
            policy_versions = versions_by_policy_id.setdefault(definition.policy_id, frozenset())
            versions_by_policy_id[definition.policy_id] = policy_versions | frozenset(
                {definition.version}
            )

        self._lookup = lookup
        self._versions_by_policy_id = versions_by_policy_id
        self._definitions = tuple(
            sorted(ordered, key=lambda item: (item.policy_id, item.version))
        )

    def definitions(self) -> tuple[PolicyDefinition, ...]:
        """Return all catalog definitions in deterministic (policy_id, version) order."""
        return self._definitions

    def resolve(
        self,
        *,
        policy_id: str,
        version: str,
    ) -> PolicyDefinition:
        normalized_policy_id, normalized_version = _normalize_lookup_identity(
            policy_id,
            version,
        )
        if not normalized_policy_id:
            raise UnknownPolicyDefinitionError(normalized_policy_id)
        if not normalized_version:
            if normalized_policy_id in self._versions_by_policy_id:
                raise UnsupportedPolicyDefinitionVersionError(
                    normalized_policy_id,
                    normalized_version,
                )
            raise UnknownPolicyDefinitionError(normalized_policy_id)

        key = (normalized_policy_id, normalized_version)
        resolved = self._lookup.get(key)
        if resolved is not None:
            return resolved

        if normalized_policy_id in self._versions_by_policy_id:
            raise UnsupportedPolicyDefinitionVersionError(
                normalized_policy_id,
                normalized_version,
            )
        raise UnknownPolicyDefinitionError(normalized_policy_id)
