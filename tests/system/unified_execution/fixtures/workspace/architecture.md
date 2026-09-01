# Local Workspace Architecture

The Local Workspace application indexes tenant documents, retrieves evidence, and answers
operational questions through the `local.workspace.search` capability.

## Components

- `local_indexer` - ingestion and chunking
- `local_search` - retrieval and evidence ranking
- `local_synthesizer` - optional synthesis from evidence
