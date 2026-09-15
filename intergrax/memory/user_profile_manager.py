# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Dict, Any, List, Union

from intergrax.memory.user_profile_memory import (
    UserProfile,
    UserProfileMemoryEntry,
    UserProfileMemoryEntryNotFoundError,
)
from intergrax.memory.memory_temporal import filter_active_memory_entries, is_memory_entry_active
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryLifecycleOperation,
    MemoryLifecycleOutcome,
    MemoryReconciliationOutcome,
    UserProfileMemoryMutationResult,
    UserProfileMemoryProjection,
)
from intergrax.memory.user_profile_memory_lifecycle import UserProfileMemoryLifecycleCoordinator
from intergrax.memory.memory_vector_namespace import LTM_INDEX_DOMAIN, resolve_memory_index_collection
from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreScope,
)


class UserProfileManager:
    """
    High-level facade for working with user profiles.

    Responsibilities:
      - provide convenient methods to:
          * load or create a UserProfile for a given user_id,
          * persist profile changes,
          * manage long-term user memory entries,
          * manage system-level instructions derived from the profile;
      - hide direct interaction with the underlying UserProfileStore.

    It intentionally does NOT:
      - call LLMs directly,
      - perform RAG over long-term user memory,
      - decide *when* the profile should be updated (this is a policy concern
        for higher-level components such as the runtime or application logic).
    """

    def __init__(
            self, 
            store: UserProfileStore,
            *,
            embedding_manager: Optional[EmbeddingManager] = None,
            vectorstore_manager: Optional[VectorstoreManager] = None,
            retrieval_service: Optional[RetrievalService] = None,
            rag_profile: Optional[RagProfile] = None,
            longterm_top_k: int = 6,
            longterm_score_threshold: float = 0.25,
            tenant_id: str = "default",
            vector_index_namespace: str | None = None,
            workspace_id: str | None = None,
            memory_projections: Sequence[UserProfileMemoryProjection] | None = None,
    ) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._vector_index_namespace = vector_index_namespace
        self._workspace_id = workspace_id
        self._ltm_collection_name = resolve_memory_index_collection(
            vector_index_namespace=vector_index_namespace,
            tenant_id=tenant_id,
            domain=LTM_INDEX_DOMAIN,
        )

        # Optional Long-Term Memory RAG dependencies
        self._embedding_manager = embedding_manager
        self._vectorstore_manager = vectorstore_manager
        self._retrieval_service = retrieval_service
        self._rag_profile = rag_profile or (retrieval_service.profile if retrieval_service else None)

        # Retrieval defaults (can be overridden per call)
        self._longterm_top_k = int(longterm_top_k)
        self._longterm_score_threshold = float(longterm_score_threshold)
        self._memory_lifecycle = UserProfileMemoryLifecycleCoordinator(
            projections=self._resolve_memory_projections(memory_projections),
        )

    def _resolve_memory_projections(
        self,
        configured: Sequence[UserProfileMemoryProjection] | None,
    ) -> tuple[UserProfileMemoryProjection, ...]:
        if configured is not None:
            return tuple(configured)
        if self._embedding_manager is not None and self._vectorstore_manager is not None:
            return (
                UserProfileLtmVectorProjection(
                    embedding_manager=self._embedding_manager,
                    vectorstore_manager=self._vectorstore_manager,
                    tenant_id=self._tenant_id,
                    vector_index_namespace=self._vector_index_namespace,
                    workspace_id=self._workspace_id,
                ),
            )
        return ()

    async def _get_store_profile(self, user_id: str) -> UserProfile:
        return await self._store.get_profile(tenant_id=self._tenant_id, user_id=user_id)

    async def _save_store_profile(self, profile: UserProfile) -> None:
        await self._store.save_profile(tenant_id=self._tenant_id, profile=profile)

    async def _delete_store_profile(self, user_id: str) -> None:
        await self._store.delete_profile(tenant_id=self._tenant_id, user_id=user_id)

    def _vector_scope(self) -> VectorStoreScope:
        return VectorStoreScope(
            tenant_id=self._tenant_id,
            namespace=self._vector_index_namespace,
            workspace_id=self._workspace_id,
        )


    def is_longterm_rag_enabled(self) -> bool:
        if self._retrieval_service is not None:
            return True
        return self._embedding_manager is not None and self._vectorstore_manager is not None

    async def _search_longterm_via_retrieval_service(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
    ) -> Dict[str, Any]:
        from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter

        service = self._retrieval_service
        assert service is not None
        request = RetrievalRequest(
            query=query,
            final_top_k=top_k,
            score_threshold=score_threshold,
            scope=self._vector_scope(),
            metadata_filter=MetadataFilter(
                conditions={
                    "user_id": user_id,
                    "deleted": 0,
                    "index_domain": LTM_INDEX_DOMAIN,
                    "collection_name": self._ltm_collection_name,
                }
            ),
        )
        result = service.retrieve(request)
        profile = await self._get_store_profile(user_id)
        by_id = {
            e.entry_id: e
            for e in profile.memory_entries
            if is_memory_entry_active(e)
        }
        hits: List[UserProfileMemoryEntry] = []
        scores: List[float] = []
        for chunk in result.chunks:
            entry_id = str((chunk.metadata or {}).get("entry_id") or chunk.id or "")
            entry = by_id.get(entry_id)
            if entry is None:
                continue
            hits.append(entry)
            scores.append(float(chunk.score or 0.0))
        used = bool(hits)
        debug = {
            "enabled": True,
            "used": used,
            "reason": result.reason or ("hits" if used else "no_hits"),
            "retrieval_service": True,
            "route_tier": result.trace.route_tier if result.trace else None,
            "hits_count": len(hits),
        }
        return {
            "used_longterm": used,
            "hits": hits,
            "scores": scores,
            "debug": debug,
        }

    
    async def search_longterm_memory(
        self,
        user_id: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Vector-based retrieval over user's long-term memory entries.

        Contract (engine-friendly):
        - debug.used is the canonical flag (like rag_debug_info["used"])
        - hits contains canonical UserProfileMemoryEntry objects from the profile store

        Returns:
        {
            "used_longterm": bool,   # kept for backward compatibility
            "hits": List[UserProfileMemoryEntry],
            "scores": List[float],
            "debug": {
                "enabled": bool,
                "used": bool,
                "reason": str,
                ...
            }
        }
        """
        q = (query or "").strip()
        enabled = self.is_longterm_rag_enabled()

        if not q or not enabled:
            reason = "empty_query" if not q else "disabled"
            debug = {
                "enabled": bool(enabled),
                "used": False,
                "reason": reason,
                "hits_count": 0,
            }
            return {
                "used_longterm": False,
                "hits": [],
                "scores": [],
                "debug": debug,
            }

        k = int(top_k if top_k is not None else self._longterm_top_k)

        # IMPORTANT: score_threshold may be None (as in RuntimeConfig.longterm_score_threshold).
        # Treat None as "no threshold" (keep all).
        thr: Optional[float]
        if score_threshold is None:
            thr = None
        else:
            thr = float(score_threshold)

        if self._retrieval_service is not None:
            return await self._search_longterm_via_retrieval_service(
                user_id=user_id,
                query=q,
                top_k=k,
                score_threshold=thr,
            )

        # Embed query
        q_emb = self._embedding_manager.embed_texts([q])
        embedding = q_emb[0].tolist() if hasattr(q_emb[0], "tolist") else list(q_emb[0])

        # Filter strictly to this user, and exclude deleted entries.
        metadata_filter = MetadataFilter(
            conditions={
                "user_id": user_id,
                "deleted": 0,
                "index_domain": LTM_INDEX_DOMAIN,
                "collection_name": self._ltm_collection_name,
            },
        )
        raw_hits = self._vectorstore_manager.query(
            embedding,
            scope=self._vector_scope(),
            top_k=k,
            metadata_filter=metadata_filter,
        )

        filtered: List[tuple[str, float]] = []
        for hit in raw_hits:
            entry_id = hit.document.identity.document_id
            if not entry_id:
                continue
            score = float(hit.similarity_score)
            if thr is None or score >= thr:
                filtered.append((entry_id, score))

        if not filtered:
            debug = {
                "enabled": True,
                "used": False,
                "reason": "no_hits",
                "metadata_filter": metadata_filter.conditions,
                "top_k": k,
                "filtered_count": 0,
            }
            return {
                "used_longterm": False,
                "hits": [],
                "scores": [],
                "debug": debug,
            }

        # Map ids -> canonical entries from the stored profile (source of truth)
        profile = await self._get_store_profile(user_id)
        by_id = {
            e.entry_id: e
            for e in profile.memory_entries
            if is_memory_entry_active(e)
        }

        hits: List[UserProfileMemoryEntry] = []
        hit_scores: List[float] = []
        for entry_id, sc in filtered:
            e = by_id.get(entry_id)
            if e is not None:
                hits.append(e)
                hit_scores.append(sc)

        used = bool(hits)

        debug = {
            "enabled": True,
            "used": used,
            "reason": "hits" if used else "all_filtered_or_missing_in_profile",
            "metadata_filter": metadata_filter.conditions,
            "top_k": k,
            "threshold": thr,
            "raw_count": len(raw_hits),
            "hits_count": len(hits),
        }

        return {
            "used_longterm": used,
            "hits": hits,
            "scores": hit_scores,
            "debug": debug,
        }




    # ---------------------------------------------------------------------
    # Core profile APIs
    # ---------------------------------------------------------------------

    async def get_profile(self, user_id: str) -> UserProfile:
        """
        Load the user profile for the given user_id.

        Implementations of UserProfileStore are expected to return an
        initialized profile even if no data exists yet for that user.
        """
        return await self._get_store_profile(user_id)

    async def save_profile(self, profile: UserProfile) -> None:
        """
        Persist the given UserProfile aggregate.

        This MUST overwrite any previously stored profile for the same user.
        """
        await self._save_store_profile(profile)

    async def delete_profile(self, user_id: str) -> None:
        """
        Remove any stored profile data for the given user_id.

        This operation is typically used for cleanup or account deletion flows.
        """
        profile = await self._get_store_profile(user_id)
        entry_ids = [entry.entry_id for entry in profile.memory_entries]
        await self._delete_store_profile(user_id)
        outcome = await self._memory_lifecycle.apply_after_primary_deletes(
            operation=MemoryLifecycleOperation.DELETE_PROFILE,
            user_id=user_id,
            entry_ids=entry_ids,
        )
        self._memory_lifecycle.raise_if_partial(outcome)

    async def reconcile_memory_projections(self, user_id: str) -> MemoryReconciliationOutcome:
        """Rebuild derived projections from authoritative profile state."""
        profile = await self._get_store_profile(user_id)
        return await self._memory_lifecycle.reconcile_user(user_id=user_id, profile=profile)

    # ---------------------------------------------------------------------
    # System instructions management
    # ---------------------------------------------------------------------

    async def get_system_instructions_for_user(self, user_id: str) -> str:
        """
        Return a compact system-level instruction string for the given user.

        Behavior:
          - loads the user's profile from the store,
          - uses the profile's `system_instructions` if set,
          - otherwise builds a deterministic fallback based on identity
            and preferences via `UserProfile.build_default_system_instructions()`.

        This method does NOT call any LLM and does NOT use long-term memory.
        Higher-level components may choose to update `system_instructions`
        using LLMs and then persist the result via `update_system_instructions()`.
        """
        profile = await self._get_store_profile(user_id)
        return self._build_default_system_instructions(profile)

    async def update_system_instructions(
        self,
        user_id: str,
        instructions: str,
    ) -> UserProfile:
        """
        Update the `system_instructions` field of the user's profile.

        This method assumes that some higher-level component (e.g. the runtime
        or a batch job) has already decided *what* the new instructions should be,
        possibly by calling an LLM over `memory_entries` and other data.

        The manager is responsible only for:
          - loading the profile,
          - updating the field,
          - persisting the aggregate.

        Returns the updated UserProfile for convenience.
        """
        profile = await self._get_store_profile(user_id)
        normalized = instructions.strip()
        profile.system_instructions = normalized or None
        profile.modified=True
        await self._save_store_profile(profile)
        profile.modified=False
        return profile

    # ---------------------------------------------------------------------
    # Long-term memory management
    # ---------------------------------------------------------------------

    async def add_memory_entry_with_lifecycle(
        self,
        user_id: str,
        entry_or_content: Union[UserProfileMemoryEntry, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserProfileMemoryMutationResult:
        """Append memory entry and return lifecycle outcome (no raise on partial projection)."""
        profile = await self._get_store_profile(user_id)

        if isinstance(entry_or_content, UserProfileMemoryEntry):
            entry = entry_or_content
            if entry.metadata is None:
                entry.metadata = {}
        else:
            from intergrax.memory.contracts.enterprise_memory_record import MemoryProvenance

            entry = UserProfileMemoryEntry(
                content=str(entry_or_content),
                metadata=metadata or {},
                provenance=MemoryProvenance(),
            )

        profile.memory_entries.append(entry)

        await self._save_store_profile(profile)

        outcome = await self._memory_lifecycle.apply_after_primary_upsert(
            operation=MemoryLifecycleOperation.WRITE,
            user_id=user_id,
            entry=entry,
        )
        return UserProfileMemoryMutationResult(entry=entry, lifecycle=outcome)

    async def add_memory_entry(
        self,
        user_id: str,
        entry_or_content: Union[UserProfileMemoryEntry, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserProfileMemoryEntry:
        """
        Append a new long-term memory entry to the user's profile.

        This method only updates the profile aggregate and persists it via
        the store. It does NOT call any LLM and does NOT update
        `system_instructions` automatically.

        Returns the updated UserProfile for convenience.
        """
        mutation = await self.add_memory_entry_with_lifecycle(
            user_id,
            entry_or_content,
            metadata=metadata,
        )
        self._memory_lifecycle.raise_if_partial(mutation.lifecycle)
        assert mutation.entry is not None
        return mutation.entry

    async def update_memory_entry(
        self,
        user_id: str,
        entry_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        Update a single long-term memory entry identified by `entry_id`.
        """
        profile = await self._get_store_profile(user_id)

        matched: UserProfileMemoryEntry | None = None
        semantic_change = False
        for entry in profile.memory_entries:
            if entry.entry_id == entry_id:
                if content is not None and content != entry.content:
                    entry.content = content
                    semantic_change = True
                if metadata is not None and metadata != entry.metadata:
                    entry.metadata = metadata
                    semantic_change = True
                if semantic_change:
                    entry.bump_revision_for_semantic_change()
                entry.modified = True
                matched = entry
                break

        if matched is None:
            raise UserProfileMemoryEntryNotFoundError(entry_id)

        await self._save_store_profile(profile)

        if semantic_change:
            outcome = await self._memory_lifecycle.apply_after_primary_upsert(
                operation=MemoryLifecycleOperation.UPDATE,
                user_id=user_id,
                entry=matched,
            )
            self._memory_lifecycle.raise_if_partial(outcome)

        matched.modified = False

        return profile

    async def remove_memory_entry_with_lifecycle(
        self,
        user_id: str,
        entry_id: str,
    ) -> UserProfileMemoryMutationResult:
        """Soft-delete entry and return lifecycle outcome (no raise on partial projection)."""
        profile = await self._get_store_profile(user_id)

        found = False
        for entry in profile.memory_entries:
            if entry.entry_id == entry_id:
                entry.deleted = True
                found = True
                break

        if not found:
            return UserProfileMemoryMutationResult(
                entry=None,
                lifecycle=MemoryLifecycleOutcome(
                    operation=MemoryLifecycleOperation.DELETE_ENTRY,
                    disposition=MemoryLifecycleDisposition.UNCHANGED,
                    user_id=user_id,
                    memory_entity_ids=(),
                    primary_applied=False,
                    projection_evidence=(),
                ),
            )

        await self._save_store_profile(profile)

        outcome = await self._memory_lifecycle.apply_after_primary_deletes(
            operation=MemoryLifecycleOperation.DELETE_ENTRY,
            user_id=user_id,
            entry_ids=(entry_id,),
        )
        return UserProfileMemoryMutationResult(entry=None, lifecycle=outcome)

    async def remove_memory_entry(
        self,
        user_id: str,
        entry_id: str,
    ) -> UserProfile:
        """
        Remove a single long-term memory entry identified by `entry_id`.
        """
        mutation = await self.remove_memory_entry_with_lifecycle(user_id, entry_id)
        self._memory_lifecycle.raise_if_partial(mutation.lifecycle)
        return await self._get_store_profile(user_id)


    async def clear_memory(self, user_id: str) -> UserProfile:
        """
        Remove all long-term memory entries for the given user.

        This is usually used for privacy/cleanup flows or when the application
        decides to reset user-level memory.
        """
        profile = await self._get_store_profile(user_id)

        entry_ids = [entry.entry_id for entry in profile.memory_entries if not entry.deleted]
        changed = False
        for entry in profile.memory_entries:
            if not entry.deleted:
                entry.deleted = True
                changed = True

        if changed:
            await self._save_store_profile(profile)
            outcome = await self._memory_lifecycle.apply_after_primary_deletes(
                operation=MemoryLifecycleOperation.CLEAR,
                user_id=user_id,
                entry_ids=entry_ids,
            )
            self._memory_lifecycle.raise_if_partial(outcome)

        return profile
    

    def _build_default_system_instructions(self, profile: UserProfile) -> str:
        """
        Deterministic, non-LLM helper that builds system instructions
        from the given profile (identity + preferences) when the profile
        does not yet have explicit system_instructions.
        """
        if profile.system_instructions:
            return profile.system_instructions.strip()

        identity = profile.identity
        prefs = profile.preferences

        lines: list[str] = []

        # Identity
        if identity.display_name:
            lines.append(f"You are talking to {identity.display_name}.")
        else:
            lines.append(f"You are talking to a user with id '{identity.user_id}'.")

        if identity.role:
            lines.append(f"The user is: {identity.role}.")
        if identity.domain_expertise:
            lines.append(f"Domain expertise: {identity.domain_expertise}.")

        # Language / style
        if prefs.preferred_language:
            lines.append(
                f"Always answer in {prefs.preferred_language} unless explicitly asked otherwise."
            )
        if prefs.tone:
            lines.append(f"Default tone: {prefs.tone}.")
        if prefs.answer_length:
            lines.append(f"Default answer length: {prefs.answer_length}.")

        # Formatting rules
        if prefs.no_emojis_in_code:
            lines.append("Never use emojis in code blocks.")
        if prefs.no_emojis_in_docs:
            lines.append("Avoid emojis in technical documentation.")
        if prefs.default_project_context:
            lines.append(
                f"Assume the default project context is: {prefs.default_project_context}."
            )

        if not lines:
            lines.append(
                "You are talking to a user. Use a helpful, concise, and technical style by default."
            )

        return " ".join(lines)


    async def purge_deleted_memory_entries(self, user_id: str) -> UserProfile:
        """
        Permanently remove entries marked as deleted=True from the profile aggregate.

        This is a maintenance operation. Normal read flows should still ignore
        deleted entries even if purge is not called.
        """
        profile = await self._get_store_profile(user_id)

        before = len(profile.memory_entries)
        profile.memory_entries = [e for e in profile.memory_entries if not e.deleted]
        after = len(profile.memory_entries)

        if after != before:
            await self._save_store_profile(profile)

        return profile