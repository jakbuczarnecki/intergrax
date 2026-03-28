# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
**Product failure & degradation contract** for the Tier-2 Legal Agent.

This module is prescriptive: it states what **must** happen when data or subsystems are
missing, empty, clamped, or failing. Runtime code, traces, and tests should stay aligned
with :class:`LegalFailurePolicy` defaults. Hosts may subclass or replace the policy object
on :class:`~legal_agent.config.legal_agent_config.LegalAgentConfig`
only when they intentionally change SKU-level guarantees (and should re-validate tests).

Implementation anchors (for maintainers, not exhaustive):
:class:`~legal_agent.steps.legal_extract_clauses_step.LegalExtractClausesStep`,
:func:`~legal_agent.runtime.legal_tool_runtime_bridge.run_legal_tool_runtime_bridge`,
:class:`~intergrax.runtime.nexus.runtime_steps.tools_step.ToolsStep`,
:func:`~legal_agent.governance.legal_tool_plan_governance.enforce_legal_tool_plan_governance`,
:func:`~legal_agent.pipeline.legal_execution_loop._legal_post_wave_early_exit_ok`,
:func:`~legal_agent.pipeline.legal_execution_loop._evaluate_legal_run_llm`,
:class:`~legal_agent.steps.legal_finalize_answer_step.LegalFinalizeAnswerStep`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LegalFailureScenarioContract(BaseModel):
    """One scenario: what the user sees, what the pipeline does, what ops can rely on in traces."""

    model_config = ConfigDict(frozen=True)

    user_facing: str = Field(
        ...,
        description="End-user or API-visible outcome (tone, disclaimers, placeholders).",
    )
    pipeline: str = Field(
        ...,
        description="Guaranteed control flow: continue, skip, replace plan, cap loops, etc.",
    )
    telemetry: str = Field(
        ...,
        description="Trace/event expectations (components, levels, redaction rules).",
    )


class LegalFailurePolicy(BaseModel):
    """
    Named scenarios for degraded/failed inputs. Defaults are the **Intergrax Legal** product contract.

    Fields are stable API: changing default text is a product decision; changing behavior
    without updating this model breaks the contract.
    """

    model_config = ConfigDict(frozen=True)

    no_retrieval_hits: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "The agent continues analysis without retrieved attachment chunks. The answer "
                "must rely on general reasoning and any non-RAG context; it should not claim "
                "verbatim quotes from documents that were not retrieved. RouteInfo.used_rag may stay "
                "false even when a RAG step ran — that reflects honest telemetry, not a hard failure."
            ),
            pipeline=(
                "Clause extraction clears clauses, emits an INFO trace with outcome=no_hits, and "
                "returns early from the extract step; later stages run according to routing. "
                "The dynamic pipeline is not aborted solely for an empty index."
            ),
            telemetry=(
                "STEP LegalExtractClausesStep: INFO, payload records outcome=no_hits and zero chunks."
            ),
        ),
    )

    nexus_layer_disabled_vs_plan: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "If the runtime config disables a layer the plan requested (e.g. enable_rag=false "
                "while the plan asked for RAG), that layer is skipped. The user still receives a "
                "normal completion unless a later step fails hard."
            ),
            pipeline=(
                "The legal tool runtime bridge logs a PIPELINE WARNING and does not run the "
                "corresponding Nexus step (RagStep / WebsearchStep / ToolsStep)."
            ),
            telemetry=(
                "PIPELINE LegalToolBridge: WARNING with explicit 'skipping …' rationale per layer."
            ),
        ),
    )

    organization_tool_plan_clamp: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "Organization flags (organization_allow_rag / _websearch / _tools) may force layers "
                "off even when Tier-2 tool-decision LLM requested them. The user receives answers "
                "consistent with the **post**-governance plan (degraded tools/RAG/websearch as needed)."
            ),
            pipeline=(
                "After tool decision, enforce_legal_tool_plan_governance recomputes intent and "
                "may copy the plan with layers turned off; each clamp emits a dedicated trace."
            ),
            telemetry=(
                "PIPELINE LegalToolPlanGovernance: WARNING per clamped layer, payload "
                "LegalToolPlanGovernanceClampDiagV1 with reason_code organization_disallows_*."
            ),
        ),
    )

    optional_tool_plan_governance_hook: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "When legal_tool_plan_governance is set, the platform may further adjust the tool "
                "plan before Nexus. Behavior is implementation-defined but must remain deterministic "
                "per request inputs and must not crash the agent if adjustment is conservative."
            ),
            pipeline=(
                "Runs after static organization clamp and before run_legal_tool_runtime_bridge; "
                "last_legal_tool_plan stores the final plan."
            ),
            telemetry=(
                "Implementation-specific; host SHOULD emit trace or logs when adjusting plans."
            ),
        ),
    )

    nexus_tools_runtime_failure: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "Tools errors are non-fatal for the Legal Agent run: the user still gets a "
                "RuntimeAnswer with a substantive finalize answer when other stages succeed. "
                "Tool failure detail in traces is redacted; do not rely on raw exception text client-side."
            ),
            pipeline=(
                "Nexus ToolsStep catches failures, records diagnostics, clears tool summaries, and "
                "returns without aborting AgentEngine. Downstream legal stages proceed."
            ),
            telemetry=(
                "Step `tools`: WARNING or ERROR, ToolsSummaryDiagV1 with error_type preserved; "
                "error_message redacted per RuntimeState trace redaction policy."
            ),
        ),
    )

    noop_or_empty_tools: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "If tools run but no useful invocation occurs (noop registry, planner yields empty), "
                "the run still completes; answers must not pretend specific tool facts were retrieved."
            ),
            pipeline="ToolsStep runs at most per bridge contract; legal pipeline continues.",
            telemetry="Standard tools step traces; may be benign INFO/WARNING depending on Nexus.",
        ),
    )

    decision_low_confidence_no_early_exit: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "Low decision.confidence does not by itself stop the run. It **disables** "
                "legal_loop_early_exit: the evaluator/replan path may still run so the pipeline "
                "can self-correct within configured iteration caps."
            ),
            pipeline=(
                "_legal_post_wave_early_exit_ok requires decision.confidence >= "
                "legal_loop_early_exit_min_confidence (default 0.9) among other gates."
            ),
            telemetry=(
                "No mandatory user-visible signal; optional PIPELINE INFO when early exit is taken."
            ),
        ),
    )

    decision_escalate_or_blocking_issues_no_early_exit: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "status ESCALATE or non-empty decision.blocking_issues blocks early exit: the run "
                "continues through evaluator/replan within caps so human escalation context can be "
                "reflected in workspace metrics and finalize prompts."
            ),
            pipeline="_legal_post_wave_early_exit_ok returns false when ESCALATE or blocking_issues.",
            telemetry="Reflected in routing/decision diagnostics and workspace metrics JSON.",
        ),
    )

    policy_violations_no_early_exit: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "When policy_violations is non-empty, early exit is blocked so compliance-bearing "
                "stages are not short-circuited; finalize may still produce an answer that reflects violations."
            ),
            pipeline="_legal_post_wave_early_exit_ok returns false if len(policy_violations) > 0.",
            telemetry="Policy compliance and decision traces carry violation counts.",
        ),
    )

    legal_run_evaluator_llm_failure: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "If the structured evaluator LLM throws or returns an invalid payload, the product "
                "assumes the wave is complete_no_replan (conservative finish): the user still gets a "
                "finalize answer rather than a hard error from evaluator infrastructure."
            ),
            pipeline="_evaluate_legal_run_llm catches exceptions and returns LegalEvaluationResult("
            "complete=True, replan=False, rationale=...).",
            telemetry="Failure is swallowed silently at trace level; consider improving observability in-host.",
        ),
    )

    finalize_empty_llm_answer: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "An empty string from the finalize structured LLM is replaced with a visible placeholder "
                "prefix '[ERROR] Empty legal finalize answer.' before shaping/governance so clients "
                "detect the defect."
            ),
            pipeline=(
                "LegalFinalizeAnswerStep sets draft_answer to the placeholder when strip() is empty; "
                "then legal_response_governance (if any) shapes the client response."
            ),
            telemetry="LegalFinalizeAnswerStepDiagV1 outcome=empty_fallback when the draft was empty.",
        ),
    )

    response_governance_hook: LegalFailureScenarioContract = Field(
        default=LegalFailureScenarioContract(
            user_facing=(
                "When legal_response_governance is configured, draft finalize text is transformed into "
                "LegalShapedClientResponse (disclaimers, format version, optional redaction). Contract for "
                "exact wording is owned by the host implementation."
            ),
            pipeline=(
                "Runs after finalize LLM, before RouteInfo/RuntimeAnswer assembly; may compose "
                "multi-part client text via compose_legal_client_answer_text."
            ),
            telemetry="Diag includes legal_response_governance_applied and format_version fields.",
        ),
    )

    @classmethod
    def product_default(cls) -> LegalFailurePolicy:
        """Explicit factory for documentation and configs that want a named default."""
        return cls()
