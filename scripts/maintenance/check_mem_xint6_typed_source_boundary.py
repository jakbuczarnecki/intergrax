# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-6-R: canonical builtin providers must not read semantic handles."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_PATH = REPO_ROOT / "intergrax" / "context" / "providers" / "builtin.py"

SEMANTIC_HANDLE_CONSTANTS = (
    "LTM_ENTRIES_HANDLE",
    "RAG_CHUNKS_HANDLE",
    "TOOL_OUTPUT_BLOCKS_HANDLE",
    "WEBSEARCH_BLOCKS_HANDLE",
    "SESSION_HISTORY_MESSAGES_HANDLE",
    "SYSTEM_INSTRUCTIONS_HANDLE",
    "POLICY_OVERLAY_FRAGMENTS_HANDLE",
    "ATTACHMENT_SUMMARIES_HANDLE",
    "SHARED_CONTEXT_READS_HANDLE",
    "PRIOR_OUTPUT_RECORDS_HANDLE",
)


def main() -> int:
    text = BUILTIN_PATH.read_text(encoding="utf-8")
    violations: list[str] = []
    semantic_handle_literals = (
        "ltm_entries",
        "rag_chunks",
        "tool_output_blocks",
        "websearch_blocks",
        "session_history_messages",
        "system_instructions",
        "policy_overlay_fragments",
        "attachment_summaries",
        "shared_context_reads",
        "prior_output_records",
        "session_history_snapshot",
    )
    for literal in semantic_handle_literals:
        if f'handles.get("{literal}"' in text or f"handles.get('{literal}'" in text:
            violations.append(f"builtin.py: semantic ctx.handles.get({literal!r})")
    if "legacy_bridge" in text:
        violations.append("builtin.py: legacy_bridge import")
    for constant in SEMANTIC_HANDLE_CONSTANTS:
        if constant in text:
            violations.append(f"builtin.py: imports or references {constant}")
    if "ContextProviderSourceInputs" not in text and "ctx.sources" not in text:
        violations.append("builtin.py: missing typed ctx.sources reads")
    if violations:
        sys.stderr.write("MEM-XINT-6-R typed source boundary guard failed:\n")
        sys.stderr.write("\n".join(violations) + "\n")
        return 1
    if not re.search(r"\bctx\.sources\.", text):
        sys.stderr.write("MEM-XINT-6-R typed source boundary guard failed: no ctx.sources access\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
