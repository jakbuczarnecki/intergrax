# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraftProfile — Tier-3 host configuration (ECC-1 model, ECC-3 wiring)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CraftMode = Literal["disabled", "dry_run", "assist_only", "supervised", "autonomous"]
IsolationTier = Literal["local", "container", "cloud"]
NetworkEgress = Literal["deny", "allowlist"]

DEFAULT_FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "os",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "sys",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "importlib",
)


class CodeCraftProfile(BaseModel):
    """Host profile for ephemeral code craft (architecture CODE_CRAFT §6.2)."""

    model_config = ConfigDict(extra="forbid")

    mode: CraftMode = "disabled"
    isolation_tier: IsolationTier = "local"
    sandbox_host_slug: str | None = None
    allowed_languages: list[str] = Field(default_factory=lambda: ["python"])
    forbidden_imports: list[str] = Field(default_factory=lambda: list(DEFAULT_FORBIDDEN_IMPORTS))
    max_code_bytes: int = Field(default=32_768, ge=256, le=1_048_576)
    max_iterations: int = Field(default=8, ge=1, le=64)
    max_total_exec_time_s: float = Field(default=120.0, ge=1.0, le=3600.0)
    require_tests: bool = False
    test_command_template: str = "pytest {path}"
    network_egress: NetworkEgress = "deny"
    promotion_schema_ref: str | None = None
    codegen_llm_profile_ref: str | None = None
    require_hitl_before_exec: bool = False
    security_scan_before_exec: bool = False

    def exec_allowed(self) -> bool:
        return self.mode in ("supervised", "autonomous")

    def generation_allowed(self) -> bool:
        return self.mode not in ("disabled",)

    def exec_budget_exhausted(self, total_exec_time_s: float) -> bool:
        return total_exec_time_s >= self.max_total_exec_time_s

    def remaining_exec_time_s(self, total_exec_time_s: float) -> float:
        return max(0.0, self.max_total_exec_time_s - total_exec_time_s)
