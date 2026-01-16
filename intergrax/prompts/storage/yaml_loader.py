# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import yaml

from intergrax.prompts.schema.prompt_schema import (
    PromptContent,
    PromptDocument,
    PromptMeta,
)
from intergrax.prompts.storage.models import (
    LoadedPrompt,
    PromptParseError,
    PromptValidationError,
)


class YamlPromptLoader:
    """
    Production-grade YAML loader for prompt documents.

    No dynamic typing, no getattr, explicit contract only.
    """

    def load(self, path: Path) -> LoadedPrompt:
        try:
            raw = self._read_yaml(path)
        except Exception as e:
            raise PromptParseError(str(e)) from e

        # Validation errors must NOT be wrapped
        doc = self._parse_document(raw)

        try:
            h = self._compute_hash(doc)
        except Exception as e:
            raise PromptParseError(str(e)) from e

        return LoadedPrompt(document=doc, content_hash=h)


    # ---------------------------------------------------------------------

    def _read_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise PromptValidationError("YAML root must be a mapping")

        return data

    # ---------------------------------------------------------------------

    def _parse_document(self, data: Dict[str, Any]) -> PromptDocument:
        try:
            pid = str(data["id"])
            version = int(data["version"])

            content = data["content"]
            meta = data["meta"]

            return PromptDocument(
                id=pid,
                version=version,
                content=PromptContent(
                    system=str(content["system"]),
                    developer=self._opt_str(content.get("developer")),
                    user_template=self._opt_str(
                        content.get("user_template")
                    ),
                ),
                meta=PromptMeta(
                    model_family=str(meta["model_family"]),
                    output_schema_id=str(meta["output_schema_id"]),
                    tags=frozenset(str(t) for t in meta.get("tags", [])),
                    description=self._opt_str(meta.get("description")),
                ),
            )

        except KeyError as e:
            raise PromptValidationError(
                f"Missing required field: {e}"
            ) from e

    # ---------------------------------------------------------------------

    def _compute_hash(self, doc: PromptDocument) -> str:
        h = hashlib.sha256()

        h.update(doc.id.encode("utf-8"))
        h.update(str(doc.version).encode("utf-8"))

        h.update(doc.content.system.encode("utf-8"))

        if doc.content.developer:
            h.update(doc.content.developer.encode("utf-8"))

        if doc.content.user_template:
            h.update(doc.content.user_template.encode("utf-8"))

        h.update(doc.meta.model_family.encode("utf-8"))
        h.update(doc.meta.output_schema_id.encode("utf-8"))

        for tag in sorted(doc.meta.tags):
            h.update(tag.encode("utf-8"))

        return h.hexdigest()

    # ---------------------------------------------------------------------

    def _opt_str(self, v: Any) -> str | None:
        if v is None:
            return None
        return str(v)
