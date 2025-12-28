# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------
# Enums
# -----------------------------

class StepAction(str, Enum):
    ASK_CLARIFYING_QUESTION = "ASK_CLARIFYING_QUESTION"
    USE_USER_LONGTERM_MEMORY_SEARCH = "USE_USER_LONGTERM_MEMORY_SEARCH"
    USE_ATTACHMENTS_RETRIEVAL = "USE_ATTACHMENTS_RETRIEVAL"
    USE_RAG_RETRIEVAL = "USE_RAG_RETRIEVAL"
    USE_WEBSEARCH = "USE_WEBSEARCH"
    USE_TOOLS = "USE_TOOLS"
    SYNTHESIZE_DRAFT = "SYNTHESIZE_DRAFT"
    VERIFY_ANSWER = "VERIFY_ANSWER"
    FINALIZE_ANSWER = "FINALIZE_ANSWER"


class FailurePolicyKind(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    RETRY = "retry"
    REPLAN = "replan"


class WebSearchStrategy(str, Enum):
    SNIPPETS = "snippets"
    OPEN_PAGES = "open_pages"
    HYBRID = "hybrid"


class OutputFormat(str, Enum):
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"


class PlanMode(str, Enum):
    CLARIFY = "clarify"
    EXECUTE = "execute"


# -----------------------------
# Shared small models
# -----------------------------

class FailurePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: FailurePolicyKind
    max_retries: int = Field(ge=0)
    retry_backoff_ms: int = Field(ge=0)
    replan_reason: Optional[str] = None


class StepBudgets(BaseModel):
    """
    Step-level budgets.
    HARD: Only these keys are allowed.
    """
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=0, ge=0)
    max_chars: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(default=0, ge=0)
    max_web_queries: int = Field(default=0, ge=0)


class PlanBudgets(BaseModel):
    """
    Plan-level budgets.
    """
    model_config = ConfigDict(extra="forbid")

    max_total_steps: int = Field(default=6, ge=1)
    max_total_tool_calls: int = Field(default=0, ge=0)
    max_total_web_queries: int = Field(default=0, ge=0)
    max_total_chars_context: int = Field(default=12000, ge=0)
    max_total_tokens_output: Optional[int] = Field(default=None, ge=0)


class StopConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop_on_clarifying_question_answered: bool = True
    stop_on_verifier_pass: bool = True
    stop_on_budget_exhausted: bool = True


class VerifyCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: VerifySeverity


# -----------------------------
# Params per action
# -----------------------------

class AskClarifyingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    # Optional UX helpers (allowed but not required)
    choices: Optional[List[str]] = None
    must_answer_to_continue: bool = True


class LtmSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1)
    score_threshold: Optional[float] = None
    include_debug: bool = False


class AttachmentsRetrievalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)


class RagRetrievalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)


class WebSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    recency_days: Optional[int] = Field(default=None, ge=0)
    max_results: int = Field(default=5, ge=1)
    strategy: WebSearchStrategy = WebSearchStrategy.HYBRID
    domains_allowlist: Optional[List[str]] = None


class ToolsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Dict[str, Any] = Field(default_factory=dict)


class SynthesizeDraftParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(min_length=1)
    must_include: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)


class VerifyAnswerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: List[VerifyCriterion] = Field(min_length=1)
    strict: bool = True


class FinalizeAnswerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(min_length=1)
    format: OutputFormat = OutputFormat.MARKDOWN


# Mapping for deterministic validation
_ACTION_TO_PARAMS_MODEL = {
    StepAction.ASK_CLARIFYING_QUESTION: AskClarifyingParams,
    StepAction.USE_USER_LONGTERM_MEMORY_SEARCH: LtmSearchParams,
    StepAction.USE_ATTACHMENTS_RETRIEVAL: AttachmentsRetrievalParams,
    StepAction.USE_RAG_RETRIEVAL: RagRetrievalParams,
    StepAction.USE_WEBSEARCH: WebSearchParams,
    StepAction.USE_TOOLS: ToolsParams,
    StepAction.SYNTHESIZE_DRAFT: SynthesizeDraftParams,
    StepAction.VERIFY_ANSWER: VerifyAnswerParams,
    StepAction.FINALIZE_ANSWER: FinalizeAnswerParams,
}


_RETRIEVAL_OR_TOOLS_ACTIONS = {
    StepAction.USE_USER_LONGTERM_MEMORY_SEARCH,
    StepAction.USE_ATTACHMENTS_RETRIEVAL,
    StepAction.USE_RAG_RETRIEVAL,
    StepAction.USE_WEBSEARCH,
    StepAction.USE_TOOLS,
}


# -----------------------------
# Execution Step
# -----------------------------

class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: StepId
    action: StepAction
    enabled: bool = True
    depends_on: List[StepId]

    budgets: StepBudgets = Field(default_factory=StepBudgets)
    inputs: Dict[str, Any] = Field(default_factory=dict)

    # NOTE: LLM returns dict; we validate and coerce into a typed params model in validator.
    params: Dict[str, Any] = Field(default_factory=dict)

    expected_output_type: ExpectedOutputType
    rationale_type: RationaleType

    on_failure: FailurePolicy = Field(default_factory=lambda: FailurePolicy(
        policy=FailurePolicyKind.RETRY, max_retries=1, retry_backoff_ms=0, replan_reason=None
    ))

    @field_validator("depends_on", mode="before")
    @classmethod
    def _depends_on_to_list(cls, v: Any) -> List[str]:
        if v is None:
            return []

        def to_str(x: Any) -> str:
            # prefer Enum.value over str(EnumMember)
            if isinstance(x, Enum):
                return str(x.value)
            return str(x)

        if isinstance(v, list):
            return [to_str(x) for x in v]

        # common mistake: single value instead of list
        return [to_str(v)]

    @model_validator(mode="after")
    def _validate_params_match_action(self) -> "ExecutionStep":
        """
        Validate 'params' dict shape by action and reject missing/extra keys.
        """
        model = _ACTION_TO_PARAMS_MODEL.get(self.action)
        if model is None:
            raise ValueError(f"Unsupported action: {self.action}")

        # Validate and normalize params via corresponding params model
        try:
            parsed = model.model_validate(self.params)
        except Exception as e:
            raise ValueError(f"Invalid params for action={self.action}: {e}") from e

        # Replace dict with normalized dict (still JSON-serializable)
        self.params = parsed.model_dump()

        return self


# -----------------------------
# Execution Plan
# -----------------------------

class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    intent: PlanIntent
    mode: PlanMode
    steps: List[ExecutionStep] = Field(min_length=1)

    budgets: PlanBudgets
    stop_conditions: StopConditions
    final_answer_style: str = Field(default="concise_technical", min_length=1)

    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_plan(self) -> "ExecutionPlan":
        # 1) Max steps budget
        if self.budgets and self.budgets.max_total_steps:
            if len(self.steps) > int(self.budgets.max_total_steps):
                raise ValueError("Number of steps exceeds max_total_steps budget.")

        # 2) Step ids must be unique
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate step_id in steps.")

        # 3) depends_on must reference existing step_ids and not create forward references
        seen = set()
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in ids:
                    raise ValueError(f"Step '{s.step_id}' depends_on unknown step_id '{dep}'.")
                # forward dep check: dep must already be seen in traversal order
                if dep not in seen:
                    raise ValueError(f"Step '{s.step_id}' has forward/invalid dependency '{dep}'.")
            seen.add(s.step_id)

        # 4) Mode constraints
        if self.mode == PlanMode.EXECUTE:
            if self.steps[-1].action != StepAction.FINALIZE_ANSWER:
                raise ValueError("Execute mode must end with FINALIZE_ANSWER.")
        elif self.mode == PlanMode.CLARIFY:
            if self.steps[0].action != StepAction.ASK_CLARIFYING_QUESTION:
                raise ValueError("Clarify mode must start with ASK_CLARIFYING_QUESTION.")
            # Clarify mode cannot include retrieval/tools steps
            for s in self.steps:
                if s.action in _RETRIEVAL_OR_TOOLS_ACTIONS:
                    raise ValueError("Clarify mode cannot include retrieval/tools steps.")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # 5) Websearch ordering: if present, it must occur before first SYNTHESIZE_DRAFT
        web_idx = None
        first_draft_idx = None
        for i, s in enumerate(self.steps):
            if web_idx is None and s.action == StepAction.USE_WEBSEARCH:
                web_idx = i
            if first_draft_idx is None and s.action == StepAction.SYNTHESIZE_DRAFT:
                first_draft_idx = i
        if web_idx is not None and first_draft_idx is not None:
            if web_idx > first_draft_idx:
                raise ValueError("USE_WEBSEARCH must appear before the first SYNTHESIZE_DRAFT.")

        return self


@dataclass(frozen=True)
class EngineHints:
    """
    Engine-level decision output: what capabilities are available/required for this run.

    Notes:
    - This is NOT a plan. It's only gating + intent hints.
    - StepPlanner must be deterministic given (user_message, hints).
    """
    enable_websearch: bool = False
    enable_ltm: bool = False
    enable_rag: bool = False
    enable_tools: bool = False

    # Routing decision from EnginePlanner
    intent: Optional[PlanIntent] = None

    # Optional debug
    intent_reason: Optional[str] = None


class StepId(str, Enum):
    WEBSEARCH = "websearch"
    LTM_SEARCH = "ltm_search"
    DRAFT = "draft"
    VERIFY = "verify"
    FINAL = "final"
    CLARIFY = "clarify"


class PlanIntent(str, Enum):
    GENERIC = "generic"
    FRESHNESS = "freshness"
    PROJECT_ARCHITECTURE = "project_architecture"
    CLARIFY = "clarify"


class ExpectedOutputType(str, Enum):
    DRAFT = "draft"
    VERIFIED = "verified"
    FINAL = "final"
    SEARCH_RESULTS = "search_results"
    LTM_RESULTS = "ltm_results"
    CLARIFYING_QUESTION = "clarifying_question"

class RationaleType(str, Enum):
    PRODUCE_DRAFT = "produce_draft"
    VERIFY_QUALITY = "verify_quality"
    FINALIZE = "finalize"
    RETRIEVE_WEB = "retrieve_web"
    RETRIEVE_LTM = "retrieve_ltm"
    ASK_CLARIFICATION = "ask_clarification"

class VerifySeverity(str, Enum):
    ERROR = "error"
    WARN = "warn"