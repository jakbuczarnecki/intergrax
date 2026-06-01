# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.braintrust.contracts import BraintrustLogEvalInput, BraintrustLogEvalOutput
from intergrax.tools.registry.wiring import ToolWiringContext

BRAINTRUST_LOG_EVAL_TOOL_ID = "braintrust.log_eval"


def braintrust_log_eval(ctx: ToolWiringContext, params: BraintrustLogEvalInput) -> BraintrustLogEvalOutput:
    from intergrax.tools.providers.observability.resolve import resolve_observability_backend

    backend = resolve_observability_backend(ctx, role="eval")
    log_eval = getattr(backend, "log_eval", None)
    if log_eval is None:
        raise RuntimeError("observability_backend_does_not_support_eval_logging")
    log_id = str(
        log_eval(
            name=params.name.strip(),
            score=params.score,
            metadata=params.metadata,
            project=params.project or None,
        )
    )
    return BraintrustLogEvalOutput(log_id=log_id)
