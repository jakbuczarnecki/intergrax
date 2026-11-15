from types import SimpleNamespace
from intergrax.runtime.execution.agent_runtime_context import ArtifactStore
import intergrax.runtime.execution.agent_runtime_context as m
cls = [v for k, v in vars(m).items() if isinstance(v, type) and "create_artifact" in vars(v)]
print("classes with create_artifact:", [c.__name__ for c in cls])
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
c = cls[0]
fake_self = SimpleNamespace(artifact_store=InMemoryArtifactStore())
state = SimpleNamespace(run_id="run-1")
try:
    c.create_artifact(fake_self, state=state, kind="k", mime_type="text/plain", data=b"x")
except Exception as e:
    print("create_artifact ->", type(e).__name__, e)
