"""Strict Parquet schema and row-type hardening tests for Data Pack bootstrap reader."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    write_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    write_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.schema import (
    embedding_parquet_field_types,
    relational_parquet_field_types,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderSchemaError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.parquet_row_decode import (
    ParquetIntegerScalar,
    ParquetStringScalar,
    ParquetVectorScalar,
    extract_parquet_scalar,
    require_bool,
    require_float_vector,
    require_int,
    require_optional_string,
    require_string,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.parquet_streaming import (
    iter_embedding_records,
    iter_relational_records,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_load import (
    _build_embedding,
    _build_relational,
)

pytestmark = pytest.mark.unit

_EMBEDDING_DIMENSION = 8
_SHARD_PATH = Path("/tmp/test-shard.parquet")


def _write_relational(path: Path, *, count: int = 1) -> None:
    records = tuple(_build_relational(index, str(index)) for index in range(count))
    write_relational_parquet(path, records)


def _write_embedding(path: Path, *, count: int = 1) -> None:
    records = tuple(
        _build_embedding(_build_relational(index, str(index))) for index in range(count)
    )
    write_embedding_parquet(path, records, embedding_dimension=_EMBEDDING_DIMENSION)


def _replace_relational_column(
    path: Path,
    column_name: str,
    values: list[object],
    column_type: pa.DataType,
) -> None:
    table = pq.read_table(path)
    column_index = table.schema.get_field_index(column_name)
    table = table.set_column(
        column_index,
        column_name,
        pa.array(values, type=column_type),
    )
    pq.write_table(table, path)


def _replace_embedding_column(
    path: Path,
    column_name: str,
    values: list[object],
    column_type: pa.DataType,
) -> None:
    table = pq.read_table(path)
    column_index = table.schema.get_field_index(column_name)
    table = table.set_column(
        column_index,
        column_name,
        pa.array(values, type=column_type),
    )
    pq.write_table(table, path)


def _drop_relational_column(path: Path, column_name: str) -> None:
    table = pq.read_table(path)
    remaining = [name for name in table.column_names if name != column_name]
    pq.write_table(table.select(remaining), path)


# --- RELATIONAL failures ---


def test_catalog_id_int_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "catalog_id", [123], pa.int64())
    with pytest.raises(DataPackReaderSchemaError, match="catalog_id"):
        list(iter_relational_records(path))


def test_catalog_id_null_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "catalog_id", [None], pa.string())
    with pytest.raises(DataPackReaderSchemaError, match="catalog_id"):
        list(iter_relational_records(path))


def test_offer_id_wrong_type_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "offer_id", [1.5], pa.float64())
    with pytest.raises(DataPackReaderSchemaError, match="offer_id"):
        list(iter_relational_records(path))


def test_global_row_index_string_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "global_row_index", ["17"], pa.string())
    with pytest.raises(DataPackReaderSchemaError, match="global_row_index"):
        list(iter_relational_records(path))


def test_global_row_index_bool_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataPackReaderSchemaError, match="global_row_index"):
        require_int(True, shard_path=_SHARD_PATH, row_index=0, column="global_row_index")


def test_record_json_null_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "record_json", [None], pa.string())
    with pytest.raises(DataPackReaderSchemaError, match="record_json"):
        list(iter_relational_records(path))


def test_semantic_text_hash_numeric_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "semantic_text_hash", [123], pa.int64())
    with pytest.raises(DataPackReaderSchemaError, match="semantic_text_hash"):
        list(iter_relational_records(path))


def test_has_identifiers_false_string_rejected() -> None:
    with pytest.raises(DataPackReaderSchemaError, match="has_identifiers"):
        require_bool("false", shard_path=_SHARD_PATH, row_index=0, column="has_identifiers")


def test_has_spec_table_int_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "has_spec_table", [1], pa.int64())
    with pytest.raises(DataPackReaderSchemaError, match="has_spec_table"):
        list(iter_relational_records(path))


def test_has_structured_attributes_null_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "has_structured_attributes", [None], pa.bool_())
    with pytest.raises(DataPackReaderSchemaError, match="has_structured_attributes"):
        list(iter_relational_records(path))


# --- NULLABILITY passes ---


def test_nullable_source_revision_accepted(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "source_revision", [None], pa.string())
    records = list(iter_relational_records(path))
    assert records[0].source_ref.source_revision is None


def test_nullable_brand_category_description_accepted(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "brand", [None], pa.string())
    _replace_relational_column(path, "category", [None], pa.string())
    _replace_relational_column(path, "description", [None], pa.string())
    records = list(iter_relational_records(path))
    assert records[0].brand is None
    assert records[0].category is None
    assert records[0].description is None


# --- EMBEDDING failures ---


def test_logical_point_id_numeric_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path)
    _replace_embedding_column(path, "logical_point_id", [42], pa.int64())
    with pytest.raises(DataPackReaderSchemaError, match="logical_point_id"):
        list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))


def test_embedding_provider_null_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path)
    _replace_embedding_column(path, "embedding_provider", [None], pa.string())
    with pytest.raises(DataPackReaderSchemaError, match="embedding_provider"):
        list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))


def test_embedding_dimension_string_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path)
    _replace_embedding_column(path, "embedding_dimension", ["1024"], pa.string())
    with pytest.raises(DataPackReaderSchemaError, match="embedding_dimension"):
        list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))


def test_embedding_dimension_float_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path)
    _replace_embedding_column(path, "embedding_dimension", [1024.0], pa.float64())
    with pytest.raises(DataPackReaderSchemaError, match="embedding_dimension"):
        list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))


def test_vector_wrong_element_count_rejected() -> None:
    with pytest.raises(DataPackReaderSchemaError, match="dense_embedding"):
        require_float_vector(
            [0.1] * (_EMBEDDING_DIMENSION - 1),
            shard_path=_SHARD_PATH,
            row_index=0,
            column="dense_embedding",
            expected_length=_EMBEDDING_DIMENSION,
        )


def test_vector_string_element_rejected() -> None:
    vector = ["0.1"] + [0.0] * (_EMBEDDING_DIMENSION - 1)
    with pytest.raises(DataPackReaderSchemaError, match="dense_embedding\\[0\\]"):
        require_float_vector(
            vector,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="dense_embedding",
            expected_length=_EMBEDDING_DIMENSION,
        )


def test_vector_bool_element_rejected() -> None:
    vector = [True] + [0.0] * (_EMBEDDING_DIMENSION - 1)
    with pytest.raises(DataPackReaderSchemaError, match="dense_embedding\\[0\\]"):
        require_float_vector(
            vector,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="dense_embedding",
            expected_length=_EMBEDDING_DIMENSION,
        )


def test_vector_null_element_rejected() -> None:
    vector = [None] + [0.0] * (_EMBEDDING_DIMENSION - 1)
    with pytest.raises(DataPackReaderSchemaError, match="dense_embedding\\[0\\]"):
        require_float_vector(
            vector,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="dense_embedding",
            expected_length=_EMBEDDING_DIMENSION,
        )


def test_vector_nan_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path)
    vector = [float("nan")] + [0.0] * (_EMBEDDING_DIMENSION - 1)
    vector_type = embedding_parquet_field_types(_EMBEDDING_DIMENSION)["dense_embedding"]
    _replace_embedding_column(path, "dense_embedding", [vector], vector_type)
    with pytest.raises(DataPackReaderSchemaError, match="finite"):
        list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))


def test_vector_positive_inf_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path)
    vector = [math.inf] + [0.0] * (_EMBEDDING_DIMENSION - 1)
    vector_type = embedding_parquet_field_types(_EMBEDDING_DIMENSION)["dense_embedding"]
    _replace_embedding_column(path, "dense_embedding", [vector], vector_type)
    with pytest.raises(DataPackReaderSchemaError, match="finite"):
        list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))


def test_vector_negative_inf_rejected(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path)
    vector = [-math.inf] + [0.0] * (_EMBEDDING_DIMENSION - 1)
    vector_type = embedding_parquet_field_types(_EMBEDDING_DIMENSION)["dense_embedding"]
    _replace_embedding_column(path, "dense_embedding", [vector], vector_type)
    with pytest.raises(DataPackReaderSchemaError, match="finite"):
        list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))


# --- SCHEMA ---


def test_wrong_arrow_field_type_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _replace_relational_column(path, "derivation_version", [1], pa.int64())
    with pytest.raises(DataPackReaderSchemaError, match="derivation_version"):
        list(iter_relational_records(path))


def test_missing_column_rejected(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path)
    _drop_relational_column(path, "semantic_text")
    with pytest.raises(DataPackReaderSchemaError, match="semantic_text"):
        list(iter_relational_records(path))


# --- REGRESSION passes ---


def test_valid_canonical_relational_shard(tmp_path: Path) -> None:
    path = tmp_path / "relational.parquet"
    _write_relational(path, count=3)
    records = list(iter_relational_records(path))
    assert len(records) == 3
    assert records[0].global_row_index == 0


def test_valid_canonical_embedding_shard(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_embedding(path, count=2)
    records = list(iter_embedding_records(path, expected_dimension=_EMBEDDING_DIMENSION))
    assert len(records) == 2
    assert len(records[0].dense_embedding) == _EMBEDDING_DIMENSION


def test_optional_string_helper_accepts_null() -> None:
    assert (
        require_optional_string(
            None,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="source_revision",
        )
        is None
    )


def test_required_string_helper_rejects_int() -> None:
    invalid_value: ParquetStringScalar = 123  # type: ignore[assignment]
    with pytest.raises(DataPackReaderSchemaError, match="catalog_id"):
        require_string(invalid_value, shard_path=_SHARD_PATH, row_index=0, column="catalog_id")


def test_global_row_index_string_helper_rejected() -> None:
    invalid_value: ParquetIntegerScalar = "17"  # type: ignore[assignment]
    with pytest.raises(DataPackReaderSchemaError, match="global_row_index"):
        require_int(invalid_value, shard_path=_SHARD_PATH, row_index=0, column="global_row_index")


def test_extract_parquet_scalar_string() -> None:
    scalar = pa.scalar("catalog-a")
    assert (
        extract_parquet_scalar(
            scalar,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="catalog_id",
        )
        == "catalog-a"
    )


def test_extract_parquet_scalar_nullable_string() -> None:
    scalar = pa.scalar(None, type=pa.string())
    assert (
        extract_parquet_scalar(
            scalar,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="source_revision",
        )
        is None
    )


def test_extract_parquet_scalar_int_and_bool() -> None:
    assert (
        extract_parquet_scalar(
            pa.scalar(17),
            shard_path=_SHARD_PATH,
            row_index=0,
            column="global_row_index",
        )
        == 17
    )
    assert (
        extract_parquet_scalar(
            pa.scalar(True),
            shard_path=_SHARD_PATH,
            row_index=0,
            column="has_identifiers",
        )
        is True
    )


def test_extract_parquet_scalar_vector_numeric() -> None:
    vector = [1.0, 2.0, 3.0]
    scalar = pa.scalar(vector, type=pa.list_(pa.float32(), len(vector)))
    extracted = extract_parquet_scalar(
        scalar,
        shard_path=_SHARD_PATH,
        row_index=0,
        column="dense_embedding",
    )
    assert isinstance(extracted, list)
    assert len(extracted) == len(vector)
    assert all(isinstance(element, float) for element in extracted)


def test_extract_parquet_scalar_unsupported_shape_rejected() -> None:
    unsupported_cell = Mock(spec=pa.Scalar)
    unsupported_cell.as_py.return_value = {"unexpected": "dict"}
    with pytest.raises(DataPackReaderSchemaError, match="canonical Parquet scalar"):
        extract_parquet_scalar(
            unsupported_cell,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="catalog_id",
        )


def test_extract_parquet_scalar_vector_string_element_rejected() -> None:
    invalid_vector_cell = Mock(spec=pa.Scalar)
    invalid_vector_cell.as_py.return_value = ["0.1", 0.0, 0.0]
    with pytest.raises(DataPackReaderSchemaError, match="dense_embedding\\[0\\]"):
        extract_parquet_scalar(
            invalid_vector_cell,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="dense_embedding",
        )


def test_vector_helper_rejects_non_list() -> None:
    invalid_value: ParquetVectorScalar = 42  # type: ignore[assignment]
    with pytest.raises(DataPackReaderSchemaError, match="dense_embedding"):
        require_float_vector(
            invalid_value,
            shard_path=_SHARD_PATH,
            row_index=0,
            column="dense_embedding",
            expected_length=_EMBEDDING_DIMENSION,
        )


def test_forbidden_type_coercion_patterns_absent() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    reader_root = (
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/data_pack_load/reader"
    )
    forbidden_fragments = (
        "str(batch.column",
        "str(source_revision_raw)",
        "str(model_revision_raw)",
        "bool(batch.column",
        "int(batch.column",
        "tuple(float(value)",
        "float(value)",
        ": object",
        "-> object",
        "dict[str, object]",
        "Mapping[str, object]",
        "Sequence[object]",
        "tuple[object, ...]",
        "getattr(",
        "setattr(",
        "hasattr(",
        "import inspect",
        "import importlib",
        "except Exception",
        "from typing import Any",
        ": Any",
    )
    reader_modules = tuple(
        path.name
        for path in reader_root.iterdir()
        if path.suffix == ".py" and path.name != "__init__.py"
    )
    for module_name in reader_modules:
        source = (reader_root / module_name).read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{module_name} contains forbidden pattern {fragment}"


def test_schema_helpers_match_codec_types() -> None:
    relational = relational_parquet_field_types()
    assert relational["global_row_index"] == pa.int64()
    assert relational["catalog_id"] == pa.string()
    assert relational["has_identifiers"] == pa.bool_()
    embedding = embedding_parquet_field_types(_EMBEDDING_DIMENSION)
    assert embedding["embedding_dimension"] == pa.int32()
    assert embedding["dense_embedding"] == pa.list_(pa.float32(), _EMBEDDING_DIMENSION)
