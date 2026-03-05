# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import Literal, Sequence

from langchain_core.documents import Document
from pandas import DataFrame

from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata
try:
    import pandas as pd
except Exception:
    pd = None

from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser


EXTRACTION_STRATEGY = Literal["rows", "sheets", "markdown"]

class ExcelSmartParser(BaseDocumentParser):

    def __init__(
        self,
        *,
        mode: EXTRACTION_STRATEGY,
        header: int,
        sheet: str | int | None,
        na_filter: bool,
        max_rows_per_sheet: int | None,
        encoding: str | None,
        delimiter: str | None,
    ):

        self._mode = mode
        self._header = header
        self._sheet = sheet
        self._na_filter = na_filter
        self._max_rows_per_sheet = max_rows_per_sheet
        self._encoding = encoding
        self._delimiter = delimiter

    @classmethod
    def parser_id(cls) -> str:
        return "excel_smart"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        loader = ExcelSmartLoader(
            source,
            mode=self._mode,
            header=self._header,
            sheet=self._sheet,
            na_filter=self._na_filter,
            max_rows_per_sheet=self._max_rows_per_sheet,
            encoding=self._encoding,
            delimiter=self._delimiter,
        )

        docs = loader.load()

        result: list[Document] = []

        for i, d in enumerate(docs):

            metadata = build_loader_metadata(
                source=source,
                parser=self.parser_id(),
                position=i,
            )

            metadata.update(d.metadata or {})

            result.append(
                Document(
                    page_content=d.page_content,
                    metadata=metadata,
                )
            )

        return result


class ExcelSmartLoader:
    """
    Excel/CSV → list of Documents.
    Modes:
      - rows:    1 doc = 1 row (page_content = JSON; best for RAG)
      - sheets:  1 doc = 1 sheet as a Markdown table (smaller files / overview)
      - markdown:1 doc = 1 row as readable text "col: value"
    Supports: .xlsx, .xls, .csv, .tsv
    """

    def __init__(
        self,
        path: str,
        *,
        mode: EXTRACTION_STRATEGY = "rows",            # "rows" | "sheets" | "markdown"
        header: int = 0,               # header row (as in pandas), None = no headers
        sheet: str | int | None = None,# sheet name/index; None => all
        na_filter: bool = True,        # drop empty rows (all NaN/empty)
        max_rows_per_sheet: int | None = None,  # limit for huge files
        encoding: str | None = None,   # for CSV/TSV
        delimiter: str | None = None,  # for CSV/TSV; None => auto (, / \t)
    ):
        if pd is None:
            raise ImportError("pandas is required for ExcelSmartLoader (pip install pandas openpyxl)")

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

    def _read_excel(self) -> dict[str, DataFrame]:
        # sheet_name=None => dict of DataFrames
        kwargs = dict(sheet_name=self.sheet if self.sheet is not None else None, header=self.header, engine=None)
        # pandas will automatically choose openpyxl/xlrd if available
        dfs = pd.read_excel(self.path, **kwargs)
        if isinstance(dfs, DataFrame):
            # single sheet → wrap in dict
            return {"Sheet1": dfs}
        return dfs  # {sheet_name: df}

    def _read_csv_like(self) -> dict[str, DataFrame]:
        # Treat CSV/TSV as "one sheet"
        sep = self.delimiter
        if sep is None:
            sep = "\t" if self._is_tsv() else ","  # simple heuristic choice
        df = pd.read_csv(self.path, sep=sep, header=self.header, encoding=self.encoding)
        return {"csv": df}

    @staticmethod
    def _dtype_map(df: DataFrame) -> dict:
        return {str(c): str(t) for c, t in df.dtypes.items()}

    @staticmethod
    def _row_to_json(df: DataFrame, idx: int) -> str:
        rec = df.iloc[idx].to_dict()
        # safe JSON (strings, numbers, dates → str)
        def _safe(v):
            if pd.isna(v):
                return None
            if hasattr(v, "isoformat"):
                try:
                    return v.isoformat()
                except Exception:
                    return str(v)
            return v if isinstance(v, (int, float, bool, str)) else str(v)
        rec = {str(k): _safe(v) for k, v in rec.items()}
        return json.dumps(rec, ensure_ascii=False)

    @staticmethod
    def _row_to_markdown(df: DataFrame, idx: int) -> str:
        rec = df.iloc[idx].to_dict()
        parts = []
        for k, v in rec.items():
            if pd.isna(v):
                continue
            parts.append(f"- **{k}**: {v}")
        return "\n".join(parts) if parts else "- (empty row)"

    @staticmethod
    def _df_to_markdown(df: DataFrame, max_rows: int | None = None) -> str:
        _df = df if max_rows is None else df.head(max_rows)
        try:
            return _df.to_markdown(index=False)
        except Exception:
            # fallback (without tabulate dependency)
            header = " | ".join(map(str, _df.columns))
            sep = " | ".join(["---"] * len(_df.columns))
            rows = [" | ".join(map(lambda x: "" if pd.isna(x) else str(x), r)) for _, r in _df.iterrows()]
            return "\n".join([header, sep] + rows)

    def load(self) -> list[Document]:
        docs: list[Document] = []
        is_excel = self._is_excel()

        sheets = self._read_excel() if is_excel else self._read_csv_like()

        for sname, df in sheets.items():
            if self.na_filter:
                df = df.dropna(how="all")
            n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
            headers = [str(c) for c in df.columns]
            dtype_map = self._dtype_map(df)

            # hard cap (huge files)
            row_cap = self.max_rows_per_sheet if self.max_rows_per_sheet is not None else n_rows

            if self.mode == "sheets":
                content = self._df_to_markdown(df, max_rows=row_cap)
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source_name": self.path.split("/")[-1],
                        "source_path": self.path,
                        "ext": ".xlsx" if is_excel else ".csv",
                        "sheet_name": sname,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        # list/dict → JSON string:
                        "headers_json": json.dumps(headers, ensure_ascii=False),
                        "dtype_map_json": json.dumps(dtype_map, ensure_ascii=False),
                        "excel_mode": "sheets",
                    }
                ))
                continue

            # rows / markdown
            max_i = min(row_cap, n_rows)
            for i in range(max_i):
                content = (
                    self._row_to_json(df, i) if self.mode == "rows"
                    else self._row_to_markdown(df, i)
                )
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source_name": self.path.split("/")[-1],
                        "source_path": self.path,
                        "ext": ".xlsx" if is_excel else ".csv",
                        "sheet_name": sname,
                        "row_ix": i,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        # list/dict → JSON string:
                        "headers_json": json.dumps(headers, ensure_ascii=False),
                        "dtype_map_json": json.dumps(dtype_map, ensure_ascii=False),
                        "excel_mode": self.mode,
                    }
                ))

        return docs