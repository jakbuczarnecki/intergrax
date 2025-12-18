# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List, Union
from langchain.schema import Document

from intergrax.memory.user_profile_memory import (
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.rag.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore_manager import VectorstoreManager


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
            longterm_top_k: int = 6,
            longterm_score_threshold: float = 0.25,
    ) -> None:
        self._store = store

        # Optional Long-Term Memory RAG dependencies
        self._embedding_manager = embedding_manager
        self._vectorstore_manager = vectorstore_manager

        # Retrieval defaults (can be overridden per call)
        self._longterm_top_k = int(longterm_top_k)
        self._longterm_score_threshold = float(longterm_score_threshold)


    def is_longterm_rag_enabled(self) -> bool:
        return self._embedding_manager is not None and self._vectorstore_manager is not None
    

    async def _index_upsert_entry(self, user_id: str, entry: UserProfileMemoryEntry) -> None:
        """
        Upsert a single memory entry into the vector store (if enabled).
        Engine does not know about this.
        """
        if not self.is_longterm_rag_enabled():
            return
        if entry.deleted:
            return

        text = (entry.content or "").strip()
        if not text:
            return

        meta = dict(entry.metadata or {})
        meta.update(
            {
                "user_id": user_id,
                "entry_id": entry.entry_id,
                "kind": getattr(entry.kind, "value", str(entry.kind)),
                "deleted": bool(entry.deleted),
            }
        )
        
        # Create a Document so VectorstoreManager handles provider specifics consistently.
        meta = self._sanitize_vectorstore_metadata(meta)
        doc = Document(page_content=text, metadata=meta)    

        emb = self._embedding_manager.embed_texts([text])  # np.ndarray [1, D] or list[list[float]]
        self._vectorstore_manager.add_documents(
            documents=[doc],
            embeddings=emb,
            ids=[entry.entry_id],
            base_metadata=None,
        )

    def _sanitize_vectorstore_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chroma (and some other vectorstores) accept only scalar metadata values:
        str | int | float | bool | None.
        We normalize lists/dicts to stable strings.
        """
        out: Dict[str, Any] = {}
        for k, v in (meta or {}).items():
            if v is None or isinstance(v, (str, int, float, bool)):
                out[k] = v
                continue

            if isinstance(v, (list, tuple)):
                # Preserve information but keep scalar type
                out[k] = ",".join(str(x) for x in v)
                continue

            if isinstance(v, dict):
                out[k] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                continue

            out[k] = str(v)

        return out

    async def _index_delete_entry(self, entry_id: str) -> None:
        """
        Delete a memory entry vector by id (if enabled).
        """
        if not self.is_longterm_rag_enabled():
            return
        if not entry_id:
            return
        self._vectorstore_manager.delete([entry_id])

    
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

        # Embed query
        q_emb = self._embedding_manager.embed_texts([q])

        # Filter strictly to this user, and exclude deleted entries.
        where = {"user_id": user_id, "deleted": False}

        res = self._vectorstore_manager.query(q_emb, top_k=k, where=where)

        ids = (res.get("ids") or [[]])[0] or []
        scores = (res.get("scores") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        docs = (res.get("documents") or [[]])[0] or []

        # Apply threshold (if any)
        filtered: List[tuple[str, float]] = []
        for i, entry_id in enumerate(ids):
            try:
                sc = float(scores[i])
            except Exception:
                continue

            if thr is None or sc >= thr:
                filtered.append((str(entry_id), sc))

        if not filtered:
            debug = {
                "enabled": True,
                "used": False,
                "reason": "no_hits",
                "where": where,
                "top_k": k,
                "threshold": thr,
                "raw_count": len(ids),
                "filtered_count": 0,
            }
            return {
                "used_longterm": False,
                "hits": [],
                "scores": [],
                "debug": debug,
            }

        # Map ids -> canonical entries from the stored profile (source of truth)
        profile = await self._store.get_profile(user_id)
        by_id = {e.entry_id: e for e in profile.memory_entries if not e.deleted}

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
            "where": where,
            "top_k": k,
            "threshold": thr,
            "raw_ids": ids,
            "raw_scores": scores,
            "raw_metadatas": metas,
            "raw_documents_preview": [str(d)[:200] for d in docs],
            "returned_count": len(hits),
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
        return await self._store.get_profile(user_id)

    async def save_profile(self, profile: UserProfile) -> None:
        """
        Persist the given UserProfile aggregate.

        This MUST overwrite any previously stored profile for the same user.
        """
        await self._store.save_profile(profile)

    async def delete_profile(self, user_id: str) -> None:
        """
        Remove any stored profile data for the given user_id.

        This operation is typically used for cleanup or account deletion flows.
        """
        await self._store.delete_profile(user_id)

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
        profile = await self._store.get_profile(user_id)
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
        profile = await self._store.get_profile(user_id)
        normalized = instructions.strip()
        profile.system_instructions = normalized or None
        profile.modified=True
        await self._store.save_profile(profile)
        profile.modified=False
        return profile

    # ---------------------------------------------------------------------
    # Long-term memory management
    # ---------------------------------------------------------------------

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
        profile = await self._store.get_profile(user_id)

        if isinstance(entry_or_content, UserProfileMemoryEntry):
            entry = entry_or_content
            # Ensure metadata dict exists (avoid None)
            if entry.metadata is None:
                entry.metadata = {}
        else:
            entry = UserProfileMemoryEntry(
                content=str(entry_or_content),
                metadata=metadata or {},
            )

        profile.memory_entries.append(entry)

        await self._store.save_profile(profile)

        # Long-term memory vector index (optional)
        await self._index_upsert_entry(user_id=user_id, entry=entry)
        
        return entry

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
        profile = await self._store.get_profile(user_id)

        for entry in profile.memory_entries:
            if entry.entry_id == entry_id:
                if content is not None:
                    entry.content = content
                if metadata is not None:
                    entry.metadata = metadata
                entry.modified=True                
                break

        await self._store.save_profile(profile)

        # If content changed, refresh vector index
        if content is not None:
            await self._index_upsert_entry(user_id=user_id, entry=entry)

        if entry:
            entry.modified=False

        return profile

    async def remove_memory_entry(
        self,
        user_id: str,
        entry_id: str,
    ) -> UserProfile:
        """
        Remove a single long-term memory entry identified by `entry_id`.
        """
        profile = await self._store.get_profile(user_id)

        found = False
        for entry in profile.memory_entries:
            if entry.entry_id == entry_id:
                entry.deleted = True
                found = True
                break

        if not found:
            return profile
       
        await self._store.save_profile(profile)

        # Keep rerieval deterministic: remove from vector index on soft delete
        await self._index_delete_entry(entry_id=entry_id)

        return profile


    async def clear_memory(self, user_id: str) -> UserProfile:
        """
        Remove all long-term memory entries for the given user.

        This is usually used for privacy/cleanup flows or when the application
        decides to reset user-level memory.
        """
        profile = await self._store.get_profile(user_id)     
        
        changed = False
        for entry in profile.memory_entries:
            if not entry.deleted:
                entry.deleted=True  
                changed = True
        
        if changed:
            await self._store.save_profile(profile)
            # profile.memory_entries.clear()

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
        profile = await self._store.get_profile(user_id)

        before = len(profile.memory_entries)
        profile.memory_entries = [e for e in profile.memory_entries if not e.deleted]
        after = len(profile.memory_entries)

        if after != before:
            await self._store.save_profile(profile)

        return profile