# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Local JSONL/file observability export sink (OBS-EXPORT-3)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.observability.export_boundary import ObservabilityExportEnvelope


class JsonlObservabilityExporter:
    """Append-only JSONL exporter for normalized observability export envelopes."""

    def __init__(
        self,
        output_path: Path | str,
        *,
        append: bool = True,
        create_parent_dirs: bool = False,
    ) -> None:
        self._output_path = Path(output_path)
        self._append = append
        self._create_parent_dirs = create_parent_dirs

    @property
    def output_path(self) -> Path:
        return self._output_path

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        if self._create_parent_dirs:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self._append else "w"
        line = envelope.model_dump_json()
        with self._output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
