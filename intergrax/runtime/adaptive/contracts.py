# © Artur Czarnecki. All rights reserved.

"""Adaptive Harness Intelligence runtime contracts (Phase W-ADAPT-0.3)."""

from __future__ import annotations

from enum import Enum

ADAPTIVE_PACKAGE_SCHEMA_VERSION = "1.0.0"


class AdaptiveLifecycleMode(str, Enum):
    """L4 runtime lifecycle modes (AHIA §12.1)."""

    OBSERVE = "l4_o"
    RECOMMEND = "l4_r"
    SHADOW = "l4_s"
    CANARY = "l4_c"
    APPLY = "l4_a"
    VERIFY = "l4_v"


class ProfileVersionStatus(str, Enum):
    """Profile version promotion states (AHIA §9.8, §12.2)."""

    DRAFT = "draft"
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"
    RETIRED = "retired"


class ProfileArtifactType(str, Enum):
    """Versioned harness profile artifact kinds (AHIA §9.8)."""

    ORCHESTRATION = "orchestration"
    RAG = "rag"
    LLM_ROUTING = "llm_routing"
    POLICY_FRAGMENT = "policy_fragment"
