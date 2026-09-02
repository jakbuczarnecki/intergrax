"""PostgreSQL reference catalog bootstrap adapter — provider imports isolated here."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLIsolationLevel,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    CatalogIngestBatch,
    CatalogIngestBatchResult,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    structured_attribute_identity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
    VpiBootstrapManifest,
)
from platform_proofs.scenarios.verified_product_identification.integrations.catalog_store.postgresql.schema import (
    SCHEMA_DDL_STATEMENTS,
)


@dataclass(slots=True)
class PostgreSQLCatalogBootstrapAdapter:
  """Reference ``CatalogBootstrapPort`` over platform PostgreSQL session boundary."""

  _provider: PostgreSQLConnectionProvider
  _ingestion_batch_label: str

  @classmethod
  def from_env(
      cls,
      *,
      schema_name: str,
      ingestion_batch_label: str,
  ) -> PostgreSQLCatalogBootstrapAdapter:
      config = PostgreSQLIntegrationConfig.from_env(tenant_schema=schema_name)
      provider = PostgreSQLConnectionProvider(config, tenant_schema=schema_name)
      return cls(_provider=provider, _ingestion_batch_label=ingestion_batch_label)

  def probe_readiness(self) -> ValidationReport:
      try:
          with self._provider.connection() as session:
              session.execute("SELECT 1")
      except Exception as exc:
          raise VpiBootstrapProviderError("PostgreSQL readiness probe failed") from exc
      return ValidationReport.from_checks(
          (
              ValidationCheck(
                  name="postgresql_reachable",
                  status=ValidationStatus.PASS,
                  detail="SELECT 1 succeeded",
              ),
          )
      )

  def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
      _ = manifest
      with self._provider.transaction(isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED) as session:
          self._provider.ensure_schema_exists(session, self._provider.tenant_schema)
          for statement in SCHEMA_DDL_STATEMENTS:
              session.execute(statement)
      return ValidationReport.from_checks(
          (
              ValidationCheck(
                  name="postgresql_schema_ready",
                  status=ValidationStatus.PASS,
                  detail="schema DDL applied idempotently",
              ),
          )
      )

  def ingest_batch(self, batch: CatalogIngestBatch) -> CatalogIngestBatchResult:
      source_count = 0
      identifier_count = 0
      structured_count = 0
      batch_label = f"{self._ingestion_batch_label}:{batch.batch_ordinal}"

      with self._provider.transaction(isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED) as session:
          for record in batch.records:
              source_ref = record.representation.source_ref
              session.execute(
                  """
                  INSERT INTO vpi_source_offer (
                      catalog_id, offer_id, source_revision, record_json,
                      derivation_version, ingestion_batch, global_row_index
                  ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                  ON CONFLICT (catalog_id, offer_id) DO UPDATE SET
                      source_revision = EXCLUDED.source_revision,
                      record_json = EXCLUDED.record_json,
                      derivation_version = EXCLUDED.derivation_version,
                      ingestion_batch = EXCLUDED.ingestion_batch,
                      global_row_index = EXCLUDED.global_row_index
                  """,
                  (
                      source_ref.catalog_id,
                      source_ref.offer_id.value,
                      source_ref.source_revision,
                      record.record_json,
                      record.representation.derivation_version,
                      batch_label,
                      record.global_row_index,
                  ),
              )
              source_count += 1

              for term in record.representation.exact.terms:
                  session.execute(
                      """
                      INSERT INTO vpi_identifier (
                          catalog_id, offer_id, identifier_type, source_value,
                          lookup_value, source_field, source_revision
                      ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                      ON CONFLICT (
                          catalog_id, offer_id, identifier_type, lookup_value, source_field
                      ) DO NOTHING
                      """,
                      (
                          source_ref.catalog_id,
                          source_ref.offer_id.value,
                          term.identifier_type.value,
                          term.source_value,
                          term.lookup_value,
                          term.source_field,
                          source_ref.source_revision,
                      ),
                  )
                  identifier_count += 1

              for attribute in record.representation.structured.attributes:
                  typed_text = (
                      str(attribute.typed_value)
                      if attribute.typed_value is not None
                      else None
                  )
                  session.execute(
                      """
                      INSERT INTO vpi_structured_attribute (
                          catalog_id, offer_id, attr_identity, canonical_key,
                          source_key, source_value, normalized_text_value,
                          typed_value_text, source_field, source_revision
                      ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                      ON CONFLICT (catalog_id, offer_id, attr_identity) DO UPDATE SET
                          canonical_key = EXCLUDED.canonical_key,
                          source_key = EXCLUDED.source_key,
                          source_value = EXCLUDED.source_value,
                          normalized_text_value = EXCLUDED.normalized_text_value,
                          typed_value_text = EXCLUDED.typed_value_text,
                          source_field = EXCLUDED.source_field,
                          source_revision = EXCLUDED.source_revision
                      """,
                      (
                          source_ref.catalog_id,
                          source_ref.offer_id.value,
                          structured_attribute_identity(attribute),
                          attribute.canonical_key,
                          attribute.source_key,
                          attribute.source_value,
                          attribute.normalized_text_value,
                          typed_text,
                          attribute.source_field,
                          source_ref.source_revision,
                      ),
                  )
                  structured_count += 1

      return CatalogIngestBatchResult(
          source_offers_ingested=source_count,
          identifiers_ingested=identifier_count,
          structured_attributes_ingested=structured_count,
      )

  def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport:
      with self._provider.connection() as session:
          source_row = session.execute(
              "SELECT COUNT(*) AS count FROM vpi_source_offer WHERE catalog_id = %s",
              (manifest.catalog_id,),
          ).fetchone()
          identifier_row = session.execute(
              "SELECT COUNT(*) AS count FROM vpi_identifier WHERE catalog_id = %s",
              (manifest.catalog_id,),
          ).fetchone()
          structured_row = session.execute(
              "SELECT COUNT(*) AS count FROM vpi_structured_attribute WHERE catalog_id = %s",
              (manifest.catalog_id,),
          ).fetchone()

      source_count = int(source_row["count"]) if source_row else 0
      identifier_count = int(identifier_row["count"]) if identifier_row else 0
      structured_count = int(structured_row["count"]) if structured_row else 0
      expected = manifest.target_max_records or manifest.checkpoint_rows_processed

      checks = (
          ValidationCheck(
              name="source_offer_count",
              status=ValidationStatus.PASS if source_count >= expected else ValidationStatus.FAIL,
              detail=f"source_count={source_count} expected>={expected}",
          ),
          ValidationCheck(
              name="identifier_rows_present",
              status=ValidationStatus.PASS if identifier_count > 0 else ValidationStatus.FAIL,
              detail=f"identifier_count={identifier_count}",
          ),
          ValidationCheck(
              name="structured_rows_present",
              status=ValidationStatus.PASS if structured_count > 0 else ValidationStatus.FAIL,
              detail=f"structured_count={structured_count}",
          ),
      )
      return ValidationReport.from_checks(checks)

  def read_manifest(self) -> VpiBootstrapManifest | None:
      with self._provider.connection() as session:
          row = session.execute(
              "SELECT * FROM vpi_catalog_manifest WHERE manifest_id = 1"
          ).fetchone()
      if row is None:
          return None
      return _manifest_from_row(row)

  def write_manifest(self, manifest: VpiBootstrapManifest) -> None:
      with self._provider.transaction(isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED) as session:
          session.execute(
              """
              INSERT INTO vpi_catalog_manifest (
                  manifest_id, state, dataset_path, dataset_checksum, dataset_record_count,
                  search_representation_derivation_version, embedding_configuration_version,
                  embedding_provider, embedding_model, embedding_dimension,
                  catalog_schema_version, search_index_schema_version,
                  bootstrap_implementation_version, catalog_id, source_revision,
                  checkpoint_batch_ordinal, checkpoint_rows_processed, target_max_records,
                  catalog_source_offer_count, catalog_identifier_count,
                  catalog_structured_attribute_count, search_point_count,
                  failure_stage, failure_detail, updated_at
              ) VALUES (
                  1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
              )
              ON CONFLICT (manifest_id) DO UPDATE SET
                  state = EXCLUDED.state,
                  dataset_path = EXCLUDED.dataset_path,
                  dataset_checksum = EXCLUDED.dataset_checksum,
                  dataset_record_count = EXCLUDED.dataset_record_count,
                  search_representation_derivation_version = EXCLUDED.search_representation_derivation_version,
                  embedding_configuration_version = EXCLUDED.embedding_configuration_version,
                  embedding_provider = EXCLUDED.embedding_provider,
                  embedding_model = EXCLUDED.embedding_model,
                  embedding_dimension = EXCLUDED.embedding_dimension,
                  catalog_schema_version = EXCLUDED.catalog_schema_version,
                  search_index_schema_version = EXCLUDED.search_index_schema_version,
                  bootstrap_implementation_version = EXCLUDED.bootstrap_implementation_version,
                  catalog_id = EXCLUDED.catalog_id,
                  source_revision = EXCLUDED.source_revision,
                  checkpoint_batch_ordinal = EXCLUDED.checkpoint_batch_ordinal,
                  checkpoint_rows_processed = EXCLUDED.checkpoint_rows_processed,
                  target_max_records = EXCLUDED.target_max_records,
                  catalog_source_offer_count = EXCLUDED.catalog_source_offer_count,
                  catalog_identifier_count = EXCLUDED.catalog_identifier_count,
                  catalog_structured_attribute_count = EXCLUDED.catalog_structured_attribute_count,
                  search_point_count = EXCLUDED.search_point_count,
                  failure_stage = EXCLUDED.failure_stage,
                  failure_detail = EXCLUDED.failure_detail,
                  updated_at = NOW()
              """,
              _manifest_to_params(manifest),
          )

  def close(self) -> None:
      return None


def _manifest_from_row(row: dict[str, object]) -> VpiBootstrapManifest:
    return VpiBootstrapManifest(
        state=BootstrapState(str(row["state"])),
        dataset_path=str(row["dataset_path"]),
        dataset_checksum=str(row["dataset_checksum"]),
        dataset_record_count=int(row["dataset_record_count"]),
        search_representation_derivation_version=str(
            row["search_representation_derivation_version"]
        ),
        embedding_configuration_version=str(row["embedding_configuration_version"]),
        embedding_provider=str(row["embedding_provider"]),
        embedding_model=str(row["embedding_model"]),
        embedding_dimension=int(row["embedding_dimension"]),
        catalog_schema_version=str(row["catalog_schema_version"]),
        search_index_schema_version=str(row["search_index_schema_version"]),
        bootstrap_implementation_version=str(row["bootstrap_implementation_version"]),
        catalog_id=str(row["catalog_id"]),
        source_revision=str(row["source_revision"]) if row.get("source_revision") else None,
        checkpoint_batch_ordinal=(
            int(row["checkpoint_batch_ordinal"])
            if row.get("checkpoint_batch_ordinal") is not None
            else None
        ),
        checkpoint_rows_processed=int(row["checkpoint_rows_processed"]),
        target_max_records=(
            int(row["target_max_records"]) if row.get("target_max_records") is not None else None
        ),
        catalog_source_offer_count=int(row["catalog_source_offer_count"]),
        catalog_identifier_count=int(row["catalog_identifier_count"]),
        catalog_structured_attribute_count=int(row["catalog_structured_attribute_count"]),
        search_point_count=int(row["search_point_count"]),
        failure_stage=str(row["failure_stage"]) if row.get("failure_stage") else None,
        failure_detail=str(row["failure_detail"]) if row.get("failure_detail") else None,
    )


def _manifest_to_params(manifest: VpiBootstrapManifest) -> tuple[object, ...]:
    return (
        manifest.state.value,
        manifest.dataset_path,
        manifest.dataset_checksum,
        manifest.dataset_record_count,
        manifest.search_representation_derivation_version,
        manifest.embedding_configuration_version,
        manifest.embedding_provider,
        manifest.embedding_model,
        manifest.embedding_dimension,
        manifest.catalog_schema_version,
        manifest.search_index_schema_version,
        manifest.bootstrap_implementation_version,
        manifest.catalog_id,
        manifest.source_revision,
        manifest.checkpoint_batch_ordinal,
        manifest.checkpoint_rows_processed,
        manifest.target_max_records,
        manifest.catalog_source_offer_count,
        manifest.catalog_identifier_count,
        manifest.catalog_structured_attribute_count,
        manifest.search_point_count,
        manifest.failure_stage,
        manifest.failure_detail,
    )
