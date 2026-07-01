# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-owned durable sinks for Elasticsearch failed observability delivery (OBS-VENDOR-6C-B1)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchFailedDeliveryRecord,
)


class FileElasticsearchFailedDeliverySink:
    """Append-only JSONL sink for safe Elasticsearch failed-delivery diagnostics."""

    def __init__(
        self,
        output_path: Path | str,
        *,
        create_parent_dirs: bool = True,
    ) -> None:
        self._output_path = Path(output_path)
        self._create_parent_dirs = create_parent_dirs

    @property
    def output_path(self) -> Path:
        return self._output_path

    def record_failed_delivery(self, record: ElasticsearchFailedDeliveryRecord) -> None:
        if self._create_parent_dirs:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._output_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
