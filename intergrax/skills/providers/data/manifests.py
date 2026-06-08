# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

DATA_SQL_ANALYST = SkillManifest(
    skill_id="data.sql_analyst",
    version="1.0.0",
    description="Structured data Q&A: schema discovery, SQL query, and workspace result export.",
    tool_ids=("database.query", "database.describe_schema", "workspace.write_file"),
    prompt_instruction_ids=("data.sql_analyst.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("data", "sql", "analyst"),
)
