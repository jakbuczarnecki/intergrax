# © Artur Czarnecki. All rights reserved.

"""Hard assembly scope isolation (tenant / user / execution)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentScopeRef,
    ContextPolicyReasonCode,
    replace_context_fragment,
)


def canonical_assembly_scope_ref(request: ContextAssemblyRequest) -> ContextFragmentScopeRef:
    user_id = request.user_id.strip()
    return ContextFragmentScopeRef(
        tenant_id=request.tenant_id,
        user_id=user_id,
        execution_scope_key=f"{request.run_id}:{request.task_id}",
    )


def _scope_compatible(
    fragment_ref: ContextFragmentScopeRef,
    canonical: ContextFragmentScopeRef,
) -> bool:
    if fragment_ref.tenant_id != canonical.tenant_id:
        return False
    if canonical.user_id and fragment_ref.user_id and fragment_ref.user_id != canonical.user_id:
        return False
    if fragment_ref.user_id and canonical.user_id and fragment_ref.user_id != canonical.user_id:
        return False
    if (
        fragment_ref.execution_scope_key
        and canonical.execution_scope_key
        and fragment_ref.execution_scope_key != canonical.execution_scope_key
    ):
        return False
    return True


def isolate_assembly_scope(
    fragments: list[ContextFragment],
    request: ContextAssemblyRequest,
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
    canonical = canonical_assembly_scope_ref(request)
    kept: list[ContextFragment] = []
    excluded: list[tuple[ContextFragment, str]] = []
    for fragment in fragments:
        ref = fragment.scope_ref
        if ref is None:
            kept.append(
                replace_context_fragment(fragment, scope_ref=canonical),
            )
            continue
        if not _scope_compatible(ref, canonical):
            excluded.append((fragment, ContextPolicyReasonCode.SCOPE_INCOMPATIBLE.value))
            continue
        kept.append(fragment)
    return kept, excluded
