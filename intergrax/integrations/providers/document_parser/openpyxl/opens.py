# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Only this module may import ``pandas`` / ``openpyxl`` for spreadsheet parsing."""

from __future__ import annotations

import json

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.openpyxl.config import OpenpyxlIntegrationConfig

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def parse_openpyxl_file(config: OpenpyxlIntegrationConfig, source: str) -> list[ParsedDocumentFragment]:
    if pd is None:
        raise ImportError("pandas is required for openpyxl integration (pip install pandas openpyxl)")

    loader = _ExcelLoader(
        source,
        mode=config.mode,
        header=config.header,
        sheet=config.sheet,
        na_filter=config.na_filter,
        max_rows_per_sheet=config.max_rows_per_sheet,
        encoding=config.encoding,
        delimiter=config.delimiter,
    )
    return [
        ParsedDocumentFragment(
            text=doc.page_content,
            metadata={"parser_backend": "openpyxl", **(doc.metadata or {})},
        )
        for doc in loader.load()
    ]


class _ExcelLoader:
    def __init__(
        self,
        path: str,
        *,
        mode: str,
        header: int,
        sheet: str | int | None,
        na_filter: bool,
        max_rows_per_sheet: int | None,
        encoding: str | None,
        delimiter: str | None,
    ) -> None:
        self.path = path
        self.mode = mode
        self.header = header
        self.sheet = sheet
        self.na_filter = na_filter
        self.max_rows_per_sheet = max_rows_per_sheet
        self.encoding = encoding
        self.delimiter = delimiter

    def _is_excel(self) -> bool:
        low = self.path.lower()
        return low.endswith(".xlsx") or low.endswith(".xls")

    def _is_tsv(self) -> bool:
        return self.path.lower().endswith(".tsv")

    def _read_excel(self) -> dict:
        kwargs = dict(sheet_name=self.sheet if self.sheet is not None else None, header=self.header, engine=None)
        try:
            dfs = pd.read_excel(self.path, **kwargs)
        except ImportError as exc:
            message = str(exc).lower()
            if "openpyxl" in message or "xlrd" in message:
                raise IntegrationDependencyError(
                    "Provider 'openpyxl' requires optional office parser dependencies. "
                    "Install Intergrax-ai[parsing-office].",
                    integration_name="openpyxl",
                ) from exc
            raise
        if isinstance(dfs, pd.DataFrame):
            return {"Sheet1": dfs}
        return dfs

    def _read_csv_like(self) -> dict:
        sep = self.delimiter
        if sep is None:
            sep = "\t" if self._is_tsv() else ","
        df = pd.read_csv(self.path, sep=sep, header=self.header, encoding=self.encoding)
        return {"csv": df}

    @staticmethod
    def _dtype_map(df) -> dict:
        return {str(c): str(t) for c, t in df.dtypes.items()}

    @staticmethod
    def _row_to_json(df, idx: int) -> str:
        def _safe(v):
            if pd.isna(v):
                return None
            if hasattr(v, "isoformat"):
                try:
                    return v.isoformat()
                except Exception:
                    return str(v)
            return v if isinstance(v, (int, float, bool, str)) else str(v)

        rec = {str(k): _safe(v) for k, v in df.iloc[idx].to_dict().items()}
        return json.dumps(rec, ensure_ascii=False)

    @staticmethod
    def _row_to_markdown(df, idx: int) -> str:
        rec = df.iloc[idx].to_dict()
        parts = []
        for k, v in rec.items():
            if pd.isna(v):
                continue
            parts.append(f"- **{k}**: {v}")
        return "\n".join(parts) if parts else "- (empty row)"

    @staticmethod
    def _df_to_markdown(df, max_rows: int | None = None) -> str:
        _df = df if max_rows is None else df.head(max_rows)
        try:
            return _df.to_markdown(index=False)
        except Exception:
            header = " | ".join(map(str, _df.columns))
            sep = " | ".join(["---"] * len(_df.columns))
            rows = [" | ".join(map(lambda x: "" if pd.isna(x) else str(x), r)) for _, r in _df.iterrows()]
            return "\n".join([header, sep] + rows)

    def load(self) -> list:
        try:
            from langchain_core.documents import Document
        except ModuleNotFoundError as exc:
            if exc.name == "langchain_core":
                raise RuntimeError(
                    "Provider 'openpyxl' requires optional dependency group "
                    "'rag-langchain-loaders'. Install Intergrax with "
                    "'rag-langchain-loaders'."
                ) from exc
            raise

        docs: list[Document] = []
        is_excel = self._is_excel()
        sheets = self._read_excel() if is_excel else self._read_csv_like()

        for sname, df in sheets.items():
            if self.na_filter:
                df = df.dropna(how="all")
            n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
            headers = [str(c) for c in df.columns]
            dtype_map = self._dtype_map(df)
            row_cap = self.max_rows_per_sheet if self.max_rows_per_sheet is not None else n_rows

            if self.mode == "sheets":
                content = self._df_to_markdown(df, max_rows=row_cap)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_path": self.path,
                            "sheet_name": sname,
                            "n_rows": n_rows,
                            "n_cols": n_cols,
                            "headers_json": json.dumps(headers, ensure_ascii=False),
                            "dtype_map_json": json.dumps(dtype_map, ensure_ascii=False),
                            "excel_mode": "sheets",
                        },
                    )
                )
                continue

            max_i = min(row_cap, n_rows)
            for i in range(max_i):
                content = self._row_to_json(df, i) if self.mode == "rows" else self._row_to_markdown(df, i)
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_path": self.path,
                            "sheet_name": sname,
                            "row_ix": i,
                            "n_rows": n_rows,
                            "n_cols": n_cols,
                            "headers_json": json.dumps(headers, ensure_ascii=False),
                            "dtype_map_json": json.dumps(dtype_map, ensure_ascii=False),
                            "excel_mode": self.mode,
                        },
                    )
                )
        return docs
