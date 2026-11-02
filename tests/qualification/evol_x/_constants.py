# © Artur Czarnecki. All rights reserved.

"""Pinned SHAs and version constants for EVOL-X qualification."""

START_HEAD_SHA = "5a4e922ffacabef0ee24eb676e5c8705c20ef4a1"
W1_PUBLICATION_SHA = "d9474aad043e110eff467d27b34c04b9d00652fc"
EVOL_TOOL_VERSION = "evol-x-0.1.0"

FRZ_EVOL_IDS = tuple(f"FRZ-EVOL-{i:02d}" for i in range(1, 11))
# Part X (master plan v3): FRZ-EVOL-07 is a scoped qualification-layer claim, never a production rollback claim.
FRZ_EVOL_07_SCOPE_FOOTNOTE = (
    "FRZ-EVOL-07 scoped PASS: qualification-layer assessor (DM-01..07) only; "
    "production ActivationService.rollback durable-state enforcement is NOT claimed (OD-EVOL-05b Option 1)."
)
FRZ_EVOL_08_SCOPE_FOOTNOTE = "FRZ-EVOL-08: passive change dossier; continuous incident correlation remains ENT-AUDIT-X."
FRZ_EVOL_10_SCOPE_FOOTNOTE = (
    "FRZ-EVOL-10: executable triggers T1..T9 + freshness counter; scenario-wave audit remains ENT-AUDIT-X."
)
