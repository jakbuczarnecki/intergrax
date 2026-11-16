"""IA-D4 probe D: hard-kill the lease holder process, then acquire (crash recovery of the instance guard)."""
import asyncio, json, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from intergrax.hosting import HostedApplicationEventType
from tests.qualification.pdc_x.d4._support import APP_ID, JournalRuntime, build_engine, definition_of, make_profile
out = {}
async def main():
    tmp = Path(tempfile.mkdtemp(prefix="iad4h_"))
    (tmp / "run").mkdir()
    holder = subprocess.Popen([sys.executable, "-u", str(ROOT / "tests/qualification/pdc_x/d4/_lock_holder_process.py"),
                               str(tmp / "run"), APP_ID, "30"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        banner = holder.stdout.readline().strip()
        out["holder_banner"] = banner
        holder.kill(); holder.wait(timeout=5)
        out["holder_returncode"] = holder.returncode
        eng, pub, _ = build_engine(definition_of(make_profile(runtime=JournalRuntime())), tmp, instance_id="after-crash")
        try:
            await asyncio.wait_for(eng.start(), 10)
            out["start_after_holder_crash"] = "READY" if eng.lifecycle_snapshot().state.value == "ready" else eng.lifecycle_snapshot().state.value
            out["events"] = [e.event_type.value for e in pub.events if e.event_type.value.startswith("hosting.instance")]
            await eng.stop()
        except Exception as e:
            out["start_after_holder_crash"] = "REJECTED:" + type(e).__name__ + ":" + str(getattr(e.__cause__, "snapshot", e.__cause__))
    finally:
        if holder.poll() is None: holder.kill()
        holder.stdout.close()
asyncio.run(main())
print(json.dumps(out, indent=1))
