# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

STORAGE_ARTIFACT_SYNC = SkillManifest(
    skill_id="storage.artifact_sync",
    version="1.0.0",
    description="Object storage sync with shadow workspace import/export for durable artifacts.",
    tool_ids=(
        "storage.get",
        "storage.put",
        "workspace.export_artifact",
        "workspace.import_artifact",
    ),
    prompt_instruction_ids=("storage.artifact_sync.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("storage", "artifact", "sync"),
)

STORAGE_OBJECT_LIFECYCLE = SkillManifest(
    skill_id="storage.object_lifecycle",
    version="1.0.0",
    description="Object storage lifecycle: exists check, presigned URLs, and delete.",
    tool_ids=("storage.exists", "storage.presigned_url", "storage.delete"),
    prompt_instruction_ids=("storage.object_lifecycle.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("storage", "lifecycle", "object"),
)

