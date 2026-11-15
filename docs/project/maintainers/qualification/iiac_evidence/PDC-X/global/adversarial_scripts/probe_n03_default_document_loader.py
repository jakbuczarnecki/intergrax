"""IA-GLOBAL probe N-03: can the default document loader be built at the audited SHA?

Run with the repository venv python from the checkout root (no network, no secrets):
    python -I probe_n03_default_document_loader.py
Expected at cba32d247: ImportError for PipelineDocumentHandler (finding N-03).
"""

import sys

sys.path.insert(0, ".")
try:
    from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_documents_loader

    create_default_documents_loader()
    print("loader built")
except Exception as exc:  # noqa: BLE001 - probe prints the failure class
    print("LOADER FAIL", type(exc).__name__, str(exc)[:200])
