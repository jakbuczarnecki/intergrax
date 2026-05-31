# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.braintrust.contracts import BraintrustLogEvalInput, BraintrustLogEvalOutput
from intergrax.tools.providers.braintrust.service import braintrust_log_eval


class BraintrustLogEvalHandler(ServiceToolHandler[BraintrustLogEvalInput, BraintrustLogEvalOutput]):
    _service = braintrust_log_eval
