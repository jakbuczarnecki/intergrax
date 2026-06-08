# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

WORKSPACE_AUTHORING = SkillManifest(
    skill_id="workspace.authoring",
    version="1.0.0",
    description="Shadow workspace read/write/search with task memory persistence for drafts.",
    tool_ids=("workspace.read_file", "workspace.write_file", "workspace.search", "memory.write"),
    prompt_instruction_ids=("workspace.authoring.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("workspace", "authoring", "shadow"),
)

WORKSPACE_SNAPSHOT_MANAGER = SkillManifest(
    skill_id="workspace.snapshot_manager",
    version="1.0.0",
    description="Workspace lifecycle: snapshot state, list files, and delete stale artifacts.",
    tool_ids=("workspace.snapshot", "workspace.list_files", "workspace.delete_file"),
    prompt_instruction_ids=("workspace.snapshot_manager.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("workspace", "snapshot", "lifecycle"),
)

WORKSPACE_DRAFT_REVIEWER = SkillManifest(
    skill_id="workspace.draft_reviewer",
    version="1.0.0",
    description="Read-only draft review with workspace search and memory context.",
    tool_ids=("workspace.read_file", "workspace.search", "memory.read"),
    prompt_instruction_ids=("workspace.draft_reviewer.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("workspace", "draft", "review"),
)


WORKSPACE_ARTIFACT_EXPORTER = SkillManifest(
    skill_id="workspace.artifact_exporter",
    version="1.0.0",
    description="Export workspace artifacts to durable object storage.",
    tool_ids=("workspace.export_artifact", "storage.put", "workspace.list_files"),
    prompt_instruction_ids=("workspace.artifact_exporter.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("workspace", "export", "artifact"),
)

