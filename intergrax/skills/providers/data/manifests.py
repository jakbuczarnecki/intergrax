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

DATA_RECORDS_QUERY = SkillManifest(
    skill_id="data.records_query",
    version="1.0.0",
    description="Document store query: search records, fetch by id, and describe collections.",
    tool_ids=("records.query", "records.get", "records.describe_collection"),
    prompt_instruction_ids=("data.records_query.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("data", "records", "document_store"),
)
