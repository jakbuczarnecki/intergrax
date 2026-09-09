"""Bounded PostgreSQL runtime qualification for structured CONTAINS capability."""

from __future__ import annotations

import json

import pytest

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    import_psycopg,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
    StructuredSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    postgres_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    StructuredAttributeTableSpec,
    pg_trgm_extension_available,
    structured_constraint_search_dml,
    structured_contains_capability_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.structured_search_adapter import (
    PostgreSqlStructuredCandidateSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalBatch,
    RelationalLoadRecord,
    RelationalTargetId,
)
from tests.integration.platform_proofs.scenarios.verified_product_identification.conftest import (
    drop_postgresql_schema,
    runtime_qualification_target_name,
)

pytestmark = [pytest.mark.integration]

_CATALOG_ID = "vpi-contains-runtime-catalog"


def _record_json(*, offer_id: str, brand: str, model: str) -> str:
    return json.dumps(
        {
            "id": offer_id,
            "title": f"{brand} {model}",
            "brand": brand,
            "keyValuePairs": {
                "Brand": brand,
                "Model": model,
            },
        },
        ensure_ascii=False,
    )


def _relational_record(*, index: int, offer_id: str, brand: str, model: str) -> RelationalLoadRecord:
    return RelationalLoadRecord(
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(offer_id),
            catalog_id=_CATALOG_ID,
            source_revision="rev-contains-runtime",
        ),
        global_row_index=index,
        record_json=_record_json(offer_id=offer_id, brand=brand, model=model),
        semantic_text=f"semantic-{offer_id}",
        semantic_text_hash=f"hash-{offer_id}",
        derivation_version="v2",
    )


@pytest.mark.skipif(not postgres_environment_available(), reason="PostgreSQL not configured")
def test_postgresql_structured_contains_runtime_qualification() -> None:
    schema_name = runtime_qualification_target_name()
    configuration = PostgreSqlBootstrapConfiguration.from_env(schema_name=schema_name)
    storage = PostgreSqlRelationalStorageAdapter.from_configuration(configuration)
    search = PostgreSqlStructuredCandidateSearchAdapter.from_configuration(configuration)
    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
    )
    table_spec = StructuredAttributeTableSpec(
        schema_name=configuration.schema_name,
        table_name=configuration.structured_attribute_table_name,
    )
    try:
        storage.prepare_target(RelationalTargetId("vpi-products"))
        batch = RelationalBatch(
            batch_number=0,
            target=RelationalTargetId("vpi-products"),
            records=tuple(
                _relational_record(
                    index=index,
                    offer_id=f"offer-contains-{index}",
                    brand="Samsung",
                    model="990 PRO",
                )
                for index in range(3)
            ),
        )
        write_result = storage.write_batch(batch)
        assert write_result.failed_count == 0

        with provider.connection() as session:
            pg_trgm_present = pg_trgm_extension_available(session)
            contains_capable = structured_contains_capability_available(session, table_spec)

        if not pg_trgm_present:
            pytest.skip("pg_trgm extension not installed in runtime PostgreSQL")

        assert contains_capable is True

        equals_result = search.search(
            StructuredSearchQuery(
                constraints=(
                    StructuredAttributeConstraint(
                        attribute_name="Brand",
                        operator=StructuredConstraintOperator.EQUALS,
                        value="Samsung",
                    ),
                ),
                limit=5,
            )
        )
        assert equals_result.failure is None

        contains_result = search.search(
            StructuredSearchQuery(
                constraints=(
                    StructuredAttributeConstraint(
                        attribute_name="Model",
                        operator=StructuredConstraintOperator.CONTAINS,
                        value="pro",
                    ),
                ),
                limit=5,
            )
        )
        assert contains_result.failure is None
        assert len(contains_result.candidates) >= 1

        with provider.connection() as session:
            _, _, _, sql_module = import_psycopg()
            explain_statement = sql_module.SQL("EXPLAIN {}").format(
                structured_constraint_search_dml(
                    table_spec,
                    include_contains_branch=True,
                )
            )
            session.execute("SET LOCAL enable_seqscan = off")
            rows = session.execute(
                explain_statement,
                ([1], ["Model"], ["contains"], ["pro"], 5),
            ).fetchall()
        plan_text = " ".join(str(row["QUERY PLAN"]) for row in rows).lower()
        assert "vpi_structured_value_trgm_idx" in plan_text
    finally:
        drop_postgresql_schema(schema_name)
