# © Artur Czarnecki. All rights reserved.

from testing_support.change_safety.contracts.enums import WorkflowEvolutionDecision


def classify_inflight_definition_change(*, has_inflight: bool, migration_declared: bool) -> WorkflowEvolutionDecision:
    if has_inflight and not migration_declared:
        return WorkflowEvolutionDecision.EXPLICITLY_UNSUPPORTED
    if has_inflight:
        return WorkflowEvolutionDecision.VERSIONED_DEPLOY
    return WorkflowEvolutionDecision.MIGRATE
