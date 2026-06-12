# © Artur Czarnecki. All rights reserved.

"""Workspace context provider (CE-7.2)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    content_hash_for_text,
)
from intergrax.context.providers.workspace_index import WorkspaceIndexResult, build_workspace_index


class WorkspaceContextProvider:
    """Collects workspace file chunks under budget."""

    provider_id = "builtin.workspace"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset({ContextFragmentSource.WORKSPACE})

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        files = ctx.handles.get("workspace_files")
        if not isinstance(files, dict) or not files:
            return []
        index: WorkspaceIndexResult = build_workspace_index(
            {str(k): str(v) for k, v in files.items()}
        )
        max_chunks = int(ctx.handles.get("workspace_max_chunks") or 32)
        fragments: list[ContextFragment] = []
        for chunk in index.chunks[:max_chunks]:
            header = f"{chunk.path}:{chunk.start_line}-{chunk.end_line}\n"
            body = chunk.content or ""
            text = header + body
            fragments.append(
                ContextFragment(
                    fragment_id=f"ws-{chunk.path}-{chunk.start_line}",
                    source=ContextFragmentSource.WORKSPACE,
                    source_id=chunk.path,
                    content=text,
                    token_estimate=max(1, len(text) // 4),
                    relevance_score=0.8,
                    freshness_score=0.9,
                    confidence_score=0.85,
                    mandatory=False,
                    metadata={
                        "merkle_root": index.root_merkle,
                        "content_hash": chunk.content_hash,
                    },
                    content_hash=content_hash_for_text(text),
                )
            )
        return fragments
