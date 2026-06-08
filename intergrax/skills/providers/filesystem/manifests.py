# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

FILESYSTEM_LOCAL_IO = SkillManifest(
    skill_id="filesystem.local_io",
    version="1.0.0",
    description="Local filesystem IO: read/write text, glob paths, and list directories.",
    tool_ids=("filesystem.read_text", "filesystem.write_text", "filesystem.glob", "filesystem.list"),
    prompt_instruction_ids=("filesystem.local_io.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("filesystem", "local", "io"),
)

FILESYSTEM_STAT_AUDITOR = SkillManifest(
    skill_id="filesystem.stat_auditor",
    version="1.0.0",
    description="Filesystem audit: stat, list, and read for operator hosts.",
    tool_ids=("filesystem.stat", "filesystem.list", "filesystem.read_text"),
    prompt_instruction_ids=("filesystem.stat_auditor.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("filesystem", "stat", "auditor"),
)

