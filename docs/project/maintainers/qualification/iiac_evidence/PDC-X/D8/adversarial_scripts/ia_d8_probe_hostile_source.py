from tests.qualification.pdc_x.d8._d8_support import *
from intergrax.contracts.artifact_ref import ArtifactClass
from intergrax.runtime.nexus.artifacts.governed_access import *
from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
store = InMemoryArtifactStore(); a = make_artifact(tenant_id=TENANT_B, artifact_id="b-art"); store.put(a)
class Hostile:
    def read(self, request):  # ignores tenant partition, returns tenant-B content for a tenant-A request
        art = store.get(TENANT_B, "b-art")
        from intergrax.runtime.nexus.artifacts.governed_access import GovernedArtifactContent
        return GovernedArtifactContent(ref=art.to_durable_ref(), content=art.data)
h = make_governance(granted=reader_grants())
port = GovernedArtifactReadPort(governance=h.boundary, audit=h.recorder, sources={ArtifactClass.EXECUTION_EVIDENCE: Hostile()})
try:
    port.read(make_context(), ArtifactReadRequest(artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="b-art"))
    print("HOSTILE_READ_ALLOWED")
except ArtifactAccessDenied as e:
    print("denied:", e.reason_code)
