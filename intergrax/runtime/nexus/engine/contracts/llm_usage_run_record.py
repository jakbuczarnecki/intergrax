from dataclasses import dataclass
from typing import List
from datetime import datetime

from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageReport


@dataclass(frozen=True)
class LLMUsageRunRecord:
    seq: int
    ts_utc: datetime
    run_id: str
    session_id: str
    user_id: str
    report: LLMUsageReport

    def pretty(self) -> str:
        lines: List[str] = []
        lines.append(f"Run #{self.seq}")
        lines.append(f"  ts_utc     : {self.ts_utc.isoformat()}")
        lines.append(f"  run_id     : {self.run_id}")
        lines.append(f"  session_id : {self.session_id}")
        lines.append(f"  user_id    : {self.user_id}")
        lines.append(self.report.pretty())        

        return "\n".join(lines)