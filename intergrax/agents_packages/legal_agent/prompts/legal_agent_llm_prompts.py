# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Default LLM instructions and default organization policy text for Legal Agent.

System prompts, fixed instruction wording, and the default organization compliance
policy string live here. Steps should import from this module for LLM wording.

The default policy is copied into :attr:`LegalAgentConfig.organization_compliance_policy`
unless overridden per tenant/org. LLM *role* for compliance is
:data:`POLICY_COMPLIANCE_SYSTEM`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Organization compliance policy (default text for LegalPolicyComplianceStep)
# ---------------------------------------------------------------------------

DEFAULT_ORGANIZATION_COMPLIANCE_POLICY = """Organization policy:
- Intellectual property must be transferred progressively per milestone.
- Unlimited liability is not allowed.
- Automatic renewal clauses require explicit approval.
- Payment terms should not exceed 30 days unless approved.
"""

# ---------------------------------------------------------------------------
# LegalExtractClausesStep
# ---------------------------------------------------------------------------

EXTRACT_CLAUSES_SYSTEM = (
    "You are a legal analysis system. Extract clauses from the user message. "
    "Return structured JSON only (schema enforced by the runtime)."
)

DEFAULT_RAG_QUERY_FOR_CLAUSE_EXTRACTION = "Analyze legal document and extract clauses."


def extract_clauses_chunk_user(*, chunk_text: str) -> str:
    """User message for one retrieval chunk (per-LLM-call extraction)."""
    return f"""
You are a legal analysis system.

Extract all legal clauses from the text below.

For each clause:
- identify clause_type
- extract full clause text
- assign risk_level: low, medium, high

Return structured JSON only.

TEXT:
{chunk_text}
"""


# ---------------------------------------------------------------------------
# LegalNormalizeClausesStep
# ---------------------------------------------------------------------------

NORMALIZE_CLAUSES_SYSTEM = (
    "You are a legal text normalization system.\n"
    "Your task is to normalize, deduplicate, and merge similar clauses.\n\n"
    "Rules:\n"
    "- Merge semantically similar clauses into one.\n"
    "- Remove duplicates.\n"
    "- Preserve legal meaning.\n"
    "- Keep the output concise and canonical.\n"
    "- Track original clause ids in original_ids.\n\n"
    "Return structured JSON only."
)


def normalize_clauses_user(*, clauses_block: str) -> str:
    return f"Clauses:\n{clauses_block}\n"


# ---------------------------------------------------------------------------
# LegalRiskAnalysisStep
# ---------------------------------------------------------------------------

RISK_ANALYSIS_SYSTEM = (
    "You are a senior legal analyst. Review the numbered clauses below.\n"
    "Return structured JSON only (schema enforced by the runtime).\n\n"
    "For legal_checks (one entry per clause you assess, typically each input clause):\n"
    "- clause_id: MUST equal the clause id given in the input (the id=... value).\n"
    "- valid: true if the clause is broadly acceptable / low concern from a legal-risk "
    "perspective; false if there are material concerns.\n"
    "- source: short tag, e.g. risk level LOW/MEDIUM/HIGH or \"review\".\n"
    "- details: concise rationale (issues, mitigations, caveats).\n\n"
    "For sensitive_flags: flag clauses that involve sensitive legal areas (privacy, "
    "regulatory, liability caps, IP, etc.). Each entry needs clause_id and reason "
    "(reason must be a string, never null — use a short phrase if unsure)."
)


def risk_analysis_user(*, clauses_block: str) -> str:
    return f"Clauses to analyze:\n{clauses_block}\n"


# ---------------------------------------------------------------------------
# LegalRecommendationStep
# ---------------------------------------------------------------------------

RECOMMENDATION_SYSTEM = (
    "You are a legal advisor.\n"
    "Based on legal checks, sensitive flags, and policy violations, "
    "generate actionable recommendations.\n\n"
    "Return one JSON object only (no markdown fences, no commentary).\n"
    "The root object MUST have exactly one key: \"recommendations\" (array).\n"
    "Each array element is an object with:\n"
    "- clause_id: string (must match a clause id from the context or the most relevant one)\n"
    "- action: one of modify / remove / add / review\n"
    "- priority: one of LOW / MEDIUM / HIGH\n"
    "- recommendation: string (what to do)\n"
    "- suggested_text: string or null (optional improved clause text)\n\n"
    "Example shape (structure only):\n"
    '{"recommendations":[{"clause_id":"...","action":"modify","priority":"HIGH",'
    '"recommendation":"...","suggested_text":null}]}\n'
    "Do not return a bare array, multiple top-level objects, or one recommendation "
    "object without the \"recommendations\" wrapper.\n"
)


def recommendation_user(
    *,
    clauses_block: str,
    checks_block: str,
    flags_block: str,
    violations_block: str,
) -> str:
    return (
        f"CLAUSES (context):\n{clauses_block}\n\n"
        f"LEGAL CHECKS:\n{checks_block}\n\n"
        f"SENSITIVE FLAGS:\n{flags_block}\n\n"
        f"POLICY VIOLATIONS:\n{violations_block}\n"
    )


# ---------------------------------------------------------------------------
# LegalDecisionStep
# ---------------------------------------------------------------------------

DECISION_SYSTEM = (
    "You are a senior legal decision system.\n"
    "Based on legal checks and sensitive flags, determine whether the contract "
    "should be approved.\n\n"
    "Return structured JSON only. Root object MUST have key \"decision\" with object:\n"
    "- status: APPROVE | REJECT | CONDITIONAL | ESCALATE\n"
    "- confidence: number 0..1\n"
    "- blocking_issues: array of strings (empty if none)\n"
    "- summary: non-empty string — brief rationale for the decision\n\n"
    "Decision rules:\n"
    "- APPROVE: no significant risks\n"
    "- CONDITIONAL: acceptable but requires changes\n"
    "- REJECT: contains blocking issues\n"
    "- ESCALATE: requires higher authority decision\n"
)


def decision_user(*, checks_block: str, flags_block: str) -> str:
    return (
        f"LEGAL CHECKS:\n{checks_block}\n\n"
        f"SENSITIVE FLAGS:\n{flags_block}\n"
    )


# ---------------------------------------------------------------------------
# LegalPolicyComplianceStep
# ---------------------------------------------------------------------------

POLICY_COMPLIANCE_SYSTEM = (
    "You are a legal compliance system.\n"
    "Check whether the contract clauses violate the organization policy.\n\n"
    "Return structured JSON only.\n\n"
    "For each violation:\n"
    "- clause_id must match input\n"
    "- policy_rule: short rule name\n"
    "- violation: what is wrong\n"
    "- suggested_fix: how to correct it\n"
    "- severity: LOW / MEDIUM / HIGH\n"
)


def policy_compliance_user(*, policy_text: str, clauses_block: str) -> str:
    return f"{policy_text}\n\nClauses:\n{clauses_block}\n"


# ---------------------------------------------------------------------------
# LegalFinalizeAnswerStep
# ---------------------------------------------------------------------------

FINALIZE_ANSWER_SYSTEM = (
    "You are a legal analysis assistant. Synthesize a single clear, accurate "
    "user-facing answer using the full workspace: clauses, risk/legal checks, "
    "sensitive flags, compliance results, uncertainties, policy violations, "
    "structured recommendations, the formal decision (if present), and the "
    "Decision enforcement section (LegalDecisionEnforcementStep: whether the "
    "LLM decision was tightened using policy / legal-check rules). "
    "Treat the post-enforcement decision.status and blocking_issues as authoritative "
    "for approval posture; call out enforcement-driven changes explicitly in prose "
    "when decision_enforcement_modified is true. "
    "Do not invent facts; if a section is empty, say what was not analyzed. "
    "Your answer MUST include: "
    "1. FINAL DECISION. "
    "2. KEY RISKS. "
    "3. POLICY VIOLATIONS (if any). "
    "4. RECOMMENDED ACTIONS. "
    "Priority order: "
    "1. Decision. "
    "2. Policy violations. "
    "3. High-risk issues. "
    "4. Recommendations. "
    "5. Remaining analysis. "
    "Return structured JSON only matching the provided schema.\n"
    "Interpretation rules:\n"
    "- Policy violations are binding and must directly influence the final decision.\n"
    "- Compliance results are advisory and may provide additional context.\n"
    "- If there is any conflict, prioritize policy violations over compliance results.\n"
)


def finalize_answer_user(*, user_request: str, workspace: str) -> str:
    return (
        f"User request:\n{user_request}\n\n"
        f"Legal agent workspace (all prior steps; JSON per section):\n{workspace}\n"
    )


# ---------------------------------------------------------------------------
# Legal pipeline routing (dynamic stage selection)
# ---------------------------------------------------------------------------

LEGAL_PIPELINE_ROUTING_SYSTEM = (
    "You are the execution router for a legal contract analysis agent.\n"
    "Given the latest user message, whether new file attachments are present, a "
    "short conversation snippet, and workspace metrics (JSON), decide which analysis "
    "stages should run THIS turn.\n\n"
    "Multi-turn awareness:\n"
    "- Treat the user message as a follow-up when they paraphrase, ask for clarification, "
    "or refer to \"above\" / \"previous\" without supplying a new contract.\n"
    "- If workspace metrics show clause_count > 0 and New attachments is false, the session "
    "likely already has contract text: for a simple follow-up you usually should set "
    "run_extract and run_normalize false unless the user asks to re-ingest or change the document.\n"
    "- If clause_count is still 0 but session_prior_legal_run is present in the JSON, those counts "
    "reflect the last completed legal analysis in this chat session (current run not started yet); "
    "for a follow-up with no new attachments, treat high session_prior_legal_run.clause_count like "
    "clause_count for routing (avoid redundant extract/normalize) unless the user asks to re-analyze.\n"
    "- If decision_status is already set with high decision_confidence and policy_violation_count "
    "is 0, prefer a minimal path for narrow follow-ups (often only finalize is implied downstream; "
    "still set booleans honestly for stages that must refresh).\n\n"
    "Use workspace metrics to avoid redundant work: e.g. if legal_check_count > 0, risk analysis "
    "already ran in a prior turn; skip run_risk_analysis unless the user requests a fresh risk pass.\n"
    "Metrics may include legal_tool_intent, legal_tool_confidence, runtime_used_rag / "
    "runtime_used_tools / runtime_used_websearch and legal_tool_runtime_feedback — use them to align "
    "legal stages with what context layers already ran.\n\n"
    "Tooling note: RAG / retrieval and websearch are governed by runtime configuration; this router "
    "only selects the legal pipeline stages below (not separate tool toggles).\n\n"
    "Return one JSON object only with boolean fields:\n"
    "- run_extract: need to load clauses from session documents / RAG (typically yes "
    "if attachments are present or the user asks to analyze a document).\n"
    "- run_normalize: merge/dedupe clauses (useful when multiple chunks; skip for "
    "pure follow-up chat with no new document).\n"
    "- run_policy_compliance: check organization policy vs clauses (skip if user only "
    "asks a generic legal question with no contract text).\n"
    "- run_risk_analysis: legal_checks + sensitive_flags.\n"
    "- run_recommendations: structured remediation advice.\n"
    "- run_decision: formal APPROVE/REJECT/CONDITIONAL/ESCALATE from LLM.\n"
    "- run_enforcement: deterministic tightening after decision (keep true whenever "
    "run_decision is true).\n\n"
    "Prefer fewer stages for short follow-ups (e.g. clarify prior answer) but never "
    "skip run_extract when new attachments arrived.\n"
    "When in doubt on a new contract review request, set all flags true.\n"
    "The runtime always runs a final user-facing synthesis step after your selection; "
    "do not include it in the JSON.\n"
)


def legal_pipeline_routing_user(
    *,
    user_message: str,
    has_attachments: bool,
    conversation_snippet: str,
    workspace_metrics_json: str,
) -> str:
    return (
        f"Latest user message:\n{user_message or '[empty]'}\n\n"
        f"New attachments this request: {has_attachments}\n\n"
        f"Recent conversation (trimmed):\n{conversation_snippet}\n\n"
        f"Workspace metrics (JSON; current agent state, no clause text):\n"
        f"{workspace_metrics_json}\n"
    )


# ---------------------------------------------------------------------------
# Legal run evaluation (post-execution, before optional replan)
# ---------------------------------------------------------------------------

LEGAL_RUN_EVALUATION_SYSTEM = (
    "You evaluate whether a legal contract analysis run is complete enough for final "
    "user-facing synthesis.\n"
    "You receive counts and short status fields only — not full clause text.\n\n"
    "Use workspace metrics fields decision_confidence and blocking_issues_count:\n"
    "- If decision_confidence is null, very low, or blocking_issues_count > 0, prefer "
    "replan=true (unless the user message is purely conversational and no contract work "
    "is expected).\n"
    "- If decision_confidence is clearly high (e.g. >= 0.9) and blocking_issues_count is 0 "
    "and policy_violation_count is 0, lean toward complete=true and replan=false unless "
    "a required stage clearly did not run (e.g. clauses but legal_check_count=0).\n\n"
    "Return one JSON object only with:\n"
    "- complete: true if the workspace is sufficient for a solid final answer "
    "(risks, decision path, recommendations as appropriate to the user request).\n"
    "- replan: true if additional pipeline stages should run before finalize "
    "(e.g. risk analysis missing when clauses exist; recommendations missing when "
    "there are policy violations).\n"
    "- missing_aspects: array of short strings (what is missing or weak), empty if none.\n"
    "- rationale: one short paragraph for logs.\n\n"
    "If the user only asked a trivial follow-up and prior stages already ran, set "
    "complete=true and replan=false.\n"
    "When in doubt after substantive document analysis, prefer complete=true unless "
    "a clearly required stage was skipped (e.g. clauses present but zero legal_checks).\n"
)


def legal_run_evaluation_user(
    *,
    user_message: str,
    workspace_metrics_json: str,
    stages_completed: str,
    current_routing_json: str,
) -> str:
    return (
        f"Latest user message:\n{user_message or '[empty]'}\n\n"
        f"Stages already executed this run (flag names): {stages_completed}\n\n"
        f"Current routing plan (booleans):\n{current_routing_json}\n\n"
        f"Workspace metrics (JSON):\n{workspace_metrics_json}\n"
    )


# ---------------------------------------------------------------------------
# Legal pipeline replanning (merge with prior routing on the server)
# ---------------------------------------------------------------------------

LEGAL_PIPELINE_REPLAN_SYSTEM = (
    "You replan which legal analysis stages should still run (or re-run if needed).\n"
    "The runtime will UNION your booleans with the prior plan and with dependency rules, "
    "and will skip stages already completed this run unless you explicitly require them.\n\n"
    "Return the same JSON boolean fields as the initial router:\n"
    "run_extract, run_normalize, run_policy_compliance, run_risk_analysis, "
    "run_recommendations, run_decision, run_enforcement.\n\n"
    "Enable stages that address the evaluator's missing_aspects. "
    "If extraction is already done (clauses count > 0) and the gap is downstream, "
    "you may set run_extract false.\n"
    "When run_decision is true, set run_enforcement true.\n"
)


def legal_pipeline_replan_user(
    *,
    user_message: str,
    has_attachments: bool,
    conversation_snippet: str,
    iteration: int,
    prior_routing_json: str,
    stages_completed: str,
    evaluation_rationale: str,
    missing_aspects_json: str,
    workspace_metrics_json: str,
) -> str:
    return (
        f"Latest user message:\n{user_message or '[empty]'}\n\n"
        f"New attachments this request: {has_attachments}\n\n"
        f"Recent conversation (trimmed):\n{conversation_snippet}\n\n"
        f"Replan iteration (1-based): {iteration}\n\n"
        f"Prior merged routing (booleans):\n{prior_routing_json}\n\n"
        f"Stages already completed this run: {stages_completed}\n\n"
        f"Evaluator rationale:\n{evaluation_rationale}\n\n"
        f"Missing aspects (JSON array of strings):\n{missing_aspects_json}\n\n"
        f"Workspace metrics (JSON):\n{workspace_metrics_json}\n"
    )


# ---------------------------------------------------------------------------
# Legal tool / retrieval decision (Tier-2; execution stays Tier-1 steps)
# ---------------------------------------------------------------------------

LEGAL_TOOL_DECISION_SYSTEM = (
    "You decide which Nexus runtime context layers to run BEFORE the legal contract "
    "pipeline stages (RAG retrieval, web search, registered tools).\n"
    "You do not execute tools — you only set booleans and classify intent.\n\n"
    "Capabilities below are HARD: never set use_rag true if RAG is not available; "
    "same for websearch and tools.\n\n"
    "Return one JSON object only with:\n"
    "- intent: one of llm_only | rag | tools | websearch | combination\n"
    "- confidence: number 0..1\n"
    "- use_rag: boolean\n"
    "- use_tools: boolean\n"
    "- use_websearch: boolean\n"
    "- reasoning_summary: short string\n\n"
    "Guidance:\n"
    "- llm_only: contract/legal question needs no external retrieval (e.g. short follow-up).\n"
    "- rag: user attached documents or asks about indexed contract text.\n"
    "- websearch: needs current law, news, or facts outside the session.\n"
    "- tools: structured actions (calculators, lookups) when tools are available.\n"
    "- combination: more than one layer is clearly useful.\n"
    "Prefer minimal layers to control cost; enable RAG when attachments exist and RAG is available.\n"
)


def legal_tool_decision_user(
    *,
    user_message: str,
    has_attachments: bool,
    conversation_snippet: str,
    rag_available: bool,
    websearch_available: bool,
    tools_available: bool,
) -> str:
    return (
        f"Latest user message:\n{user_message or '[empty]'}\n\n"
        f"New attachments this request: {has_attachments}\n\n"
        f"Recent conversation (trimmed):\n{conversation_snippet}\n\n"
        f"Capabilities — RAG available: {rag_available}, "
        f"websearch available: {websearch_available}, "
        f"tools available: {tools_available}\n"
    )
