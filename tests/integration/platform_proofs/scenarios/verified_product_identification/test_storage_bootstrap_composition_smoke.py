"""Tiny real composition smoke for PostgreSQL + Qdrant storage bootstrap."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from intergrax.integrations.providers.vector_store.qdrant.opens import (
    open_qdrant_control_plane_client,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    storage_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.adapter import (
    QdrantVectorStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    QdrantBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.target_mapping import (
    physical_collection_name,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchSize,
    BootstrapFinalStatus,
    BootstrapRequest,
    RelationalTargetId,
    ResumeMode,
    VectorTargetId,
    VerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.filesystem_reader import (
    FilesystemDataPackBootstrapReader,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.service import (
    StorageBootstrapDependencies,
    StorageBootstrapService,
)
from tests.integration.platform_proofs.scenarios.verified_product_identification.conftest import (
    drop_postgresql_schema,
    runtime_qualification_target_name,
    write_runtime_qualification_pack,
)

pytestmark = [pytest.mark.integration]

_RECORD_COUNT = 5


def _composition_available() -> bool:
    return storage_environment_available()


@pytest.mark.skipif(not _composition_available(), reason="PostgreSQL + Qdrant not configured")
def test_storage_bootstrap_composition_fresh_and_resume(tmp_path: Path) -> None:
    pack_root = write_runtime_qualification_pack(tmp_path / "pack", record_count=_RECORD_COUNT)
    checkpoint_root = tmp_path / "checkpoint"
    schema_name = runtime_qualification_target_name()
    tenant_id = runtime_qualification_target_name()
    previous_tenant = os.environ.get("INTERGRAX_QDRANT_TENANT_ID")
    os.environ["INTERGRAX_QDRANT_TENANT_ID"] = tenant_id
    qdrant_configuration = QdrantBootstrapConfiguration.from_env(
        logical_collection_name="vpi-product-embeddings",
    )
    relational = PostgreSqlRelationalStorageAdapter.from_env(schema_name=schema_name)
    vector = QdrantVectorStorageAdapter.from_env(
        logical_collection_name="vpi-product-embeddings",
    )
    reader = FilesystemDataPackBootstrapReader(pack_root)
    checkpoint_store = FilesystemBootstrapCheckpointStore(checkpoint_root)
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=relational,
            vector=vector,
            checkpoint_store=checkpoint_store,
        )
    )
    request = BootstrapRequest(
        artifact_root=pack_root,
        relational_target=RelationalTargetId("vpi-products"),
        vector_target=VectorTargetId("vpi-product-embeddings"),
        batch_size=BootstrapBatchSize(3),
        resume_mode=ResumeMode.FRESH,
        verification_mode=VerificationMode.STRICT,
        plan_only=False,
    )
    try:
        fresh = service.run(request)
        assert fresh.status is BootstrapFinalStatus.SUCCESS
        assert fresh.total_relational_written == _RECORD_COUNT
        assert fresh.total_vectors_written == _RECORD_COUNT
        assert fresh.committed_batches > 0

        resume = service.run(
            BootstrapRequest(
                artifact_root=request.artifact_root,
                relational_target=request.relational_target,
                vector_target=request.vector_target,
                batch_size=request.batch_size,
                resume_mode=ResumeMode.RESUME,
                verification_mode=request.verification_mode,
                plan_only=False,
            )
        )
        assert resume.status is BootstrapFinalStatus.SUCCESS
        assert resume.total_relational_written == 0
        assert resume.total_vectors_written == 0
    finally:
        reader.close()
        vector.close()
        drop_postgresql_schema(relational)
        client = open_qdrant_control_plane_client(qdrant_configuration.integration)
        try:
            physical_name = physical_collection_name(
                qdrant_configuration.logical_collection_name,
                qdrant_configuration.integration.tenant_id,
            )
            collections = {item.name for item in client.get_collections().collections}
            if physical_name in collections:
                client.delete_collection(physical_name)
        finally:
            client.close()
        if previous_tenant is None:
            os.environ.pop("INTERGRAX_QDRANT_TENANT_ID", None)
        else:
            os.environ["INTERGRAX_QDRANT_TENANT_ID"] = previous_tenant
