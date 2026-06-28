# yt-dlp (yt_dlp)

Category: `document_parser`

## Legacy facade

- `create_yt_dlp_document_parser()` remains backward-compatible.

## Contract-based integration

- `YtDlpDocumentParserIntegration` derives from the category-specific contract.
- Factory: `create_yt_dlp_document_parser_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
