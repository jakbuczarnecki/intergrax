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

DATA_SQL_MUTATOR = SkillManifest(
    skill_id="data.sql_mutator",
    version="1.0.0",
    description="SQL mutation runner: execute statements with schema guard and query fallback.",
    tool_ids=("database.execute", "database.describe_schema", "database.query"),
    prompt_instruction_ids=("data.sql_mutator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("data", "sql", "mutator"),
)


DATA_RECORDS_ADMIN = SkillManifest(
    skill_id="data.records_admin",
    version="1.0.0",
    description="Records store admin: put, delete, and count documents.",
    tool_ids=("records.put", "records.delete", "records.count"),
    prompt_instruction_ids=("data.records_admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("data", "records", "admin"),
)

DATA_PIPELINE_PROBE = SkillManifest(
    skill_id="data.pipeline_probe",
    version="1.0.0",
    description="Data pipeline health: SQL probe, records query, store check.",
    tool_ids=("database.query", "records.query", "health.check_relational_store"),
    prompt_instruction_ids=("data.pipeline_probe.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("data", "pipeline", "probe"),
)


DATA_SCHEMA_DOCUMENTER = SkillManifest(
    skill_id="data.schema_documenter",
    version="1.0.0",
    description="Schema documentation for SQL and records stores.",
    tool_ids=("database.describe_schema", "records.describe_collection", "workspace.write_file"),
    prompt_instruction_ids=("data.schema_documenter.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("data", "schema", "documenter"),
)

