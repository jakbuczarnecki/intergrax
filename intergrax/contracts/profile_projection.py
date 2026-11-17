# © Artur Czarnecki. All rights reserved.

"""Declared projection of a wider profile model onto a narrower mirrored one.

A projection names every field of the source model: it is either carried to the
target (``projected``) or left out with a stated reason (``not_projected``).
``verify`` fails closed on any field that is in neither list, so a field added to
the source model can never be dropped silently, and it refuses a target that
ignores unknown keys. The module names no concrete profile class; callers pass the
classes in, so it stays a Tier-0 neutral contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel


class ProfileProjectionError(Exception):
    """Base class of every projection declaration error."""


class UnmappedProfileFieldError(ProfileProjectionError):
    """A source field is neither projected nor declared as not projected."""


class StaleProjectionEntryError(ProfileProjectionError):
    """A listed field does not exist on the source model."""


class ProjectionTargetFieldMissingError(ProfileProjectionError):
    """A projected field does not exist on the target model."""


class ProjectionTargetIgnoresExtrasError(ProfileProjectionError):
    """The target model does not forbid unknown keys."""


class DuplicateProjectionEntryError(ProfileProjectionError):
    """A field is listed both as projected and as not projected."""


class ProjectionReasonMissingError(ProfileProjectionError):
    """A not-projected field carries an empty reason."""


@dataclass(frozen=True)
class ProfileProjection[TargetT: BaseModel]:
    """Field map from a source profile model onto the target profile model."""

    name: str
    target: type[TargetT]
    projected: tuple[str, ...]
    not_projected: Mapping[str, str] = field(default_factory=dict)

    def verify(self, source_cls: type[BaseModel]) -> None:
        """Raise a typed error unless the map covers the source model exactly."""
        source_fields = set(source_cls.model_fields)
        projected = set(self.projected)
        left_out = set(self.not_projected)

        both = sorted(projected & left_out)
        if both:
            raise DuplicateProjectionEntryError(
                f"{self.name}: fields listed as projected and as not projected: {both}",
            )
        unmapped = sorted(source_fields - projected - left_out)
        if unmapped:
            raise UnmappedProfileFieldError(
                f"{self.name}: source fields neither projected nor declared: {unmapped}",
            )
        stale = sorted((projected | left_out) - source_fields)
        if stale:
            raise StaleProjectionEntryError(
                f"{self.name}: listed fields missing on {source_cls.__name__}: {stale}",
            )
        missing_on_target = sorted(projected - set(self.target.model_fields))
        if missing_on_target:
            raise ProjectionTargetFieldMissingError(
                f"{self.name}: projected fields missing on {self.target.__name__}: {missing_on_target}",
            )
        empty_reason = sorted(name for name, reason in self.not_projected.items() if not reason.strip())
        if empty_reason:
            raise ProjectionReasonMissingError(
                f"{self.name}: fields declared not projected without a reason: {empty_reason}",
            )
        if self.target.model_config.get("extra") != "forbid":
            raise ProjectionTargetIgnoresExtrasError(
                f"{self.name}: target {self.target.__name__} must set extra='forbid'",
            )

    def project(self, source_obj: BaseModel) -> TargetT:
        """Build the target from the declared fields of ``source_obj`` only."""
        payload = source_obj.model_dump(mode="json", include=set(self.projected))
        return self.target.model_validate(payload)
