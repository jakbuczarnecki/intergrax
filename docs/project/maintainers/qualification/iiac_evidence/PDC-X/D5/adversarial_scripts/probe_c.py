"""IA-D5 probe C: multi-process append to one file registry (R018 across processes)."""
import multiprocessing as mp, sys, tempfile
from pathlib import Path
sys.path.insert(0, "D:/Projekty/intergrax/.claude/worktrees/pdc-x-ia-d5-fb28f3")
from intergrax.runtime.architecture.online_evaluation_registry import FileOnlineEvaluationRegistry, OnlineEvaluationRegistryBusyError
from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationObservation, OnlineEvaluationMode

def work(args):
    path, pid_n = args
    reg = FileOnlineEvaluationRegistry(Path(path), lock_timeout_seconds=30)
    busy = 0
    for i in range(25):
        try:
            reg.append_for_tenant(OnlineEvaluationObservation(observation_id=f"p{pid_n}-{i}", run_id="r", agent_id="a", mode=OnlineEvaluationMode.SHADOW, scenario_id="s", passed=True, score=1.0), f"t{pid_n%2}")
        except OnlineEvaluationRegistryBusyError:
            busy += 1
    return busy

if __name__ == "__main__":
    d = Path(tempfile.mkdtemp(prefix="iad5c_")); path = d / "obs.json"
    with mp.Pool(4) as pool:
        busy = pool.map(work, [(str(path), n) for n in range(4)])
    reg = FileOnlineEvaluationRegistry(path)
    n = len(reg.list_observations())
    print("rejected(busy):", sum(busy), "stored:", n, "expected:", 100 - sum(busy), "lock leftover:", (d / "obs.json.lock").exists())
