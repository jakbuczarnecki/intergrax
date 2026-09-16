# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill domain errors for acquisition and composition lifecycle."""


class SkillDomainError(Exception):
    """Base error for Skill domain boundaries."""


class DynamicSkillAcquisitionResolutionError(SkillDomainError):
    """Exact release resolution failed — fail closed, no latest fallback."""


class DynamicSkillAcquisitionConflictError(SkillDomainError):
    """Operation replay conflicts with prior acquisition identity."""


class DynamicSkillAcquisitionBindingError(SkillDomainError):
    """Host profile skill binding rejected acquisition outcome."""


__all__ = [
    "DynamicSkillAcquisitionBindingError",
    "DynamicSkillAcquisitionConflictError",
    "DynamicSkillAcquisitionResolutionError",
    "SkillDomainError",
]
