# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Automatic Project Structure Documentation Generator for the Intergrax framework.

Purpose:
- Recursively scan the project directory.
- Collect all relevant source files (Python, Jupyter Notebooks, configurable).
- For each file:
    - Read the source code.
    - Send content + metadata to an LLM adapter.
    - Generate a structured summary: purpose, domain, responsibilities.

Output:
- Creates `PROJECT_STRUCTURE.md` containing a clear, navigable, human-readable
  and LLM-friendly description of the entire project layout.
- The result can be used:
    - for onboarding new developers,
    - as a blueprint for architectural decision-making,
    - as a context file when chatting with a model (ChatGPT, Intergrax agent),
      so the model immediately understands the architecture and components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm.auto import tqdm

from intergrax.globals.settings import GlobalSettings
from intergrax.llm_adapters import LangChainOllamaAdapter


@dataclass
class ProjectOverviewConfig:
    root_dir: Path = field(default_factory=lambda: Path(".").resolve())
    output_md: Path = field(default_factory=lambda: Path("PROJECT_STRUCTURE.md"))
    include_exts: Sequence[str] = field(default_factory=lambda: [".py", ".ipynb"])
    exclude_dirs: Sequence[str] = field(
        default_factory=lambda: [
            ".git",
            ".idea",
            ".vscode",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "venv",
            ".venv",
            "env",
            ".env",
            ".tmp",
        ]
    )
    exclude_files_substring: Sequence[str] = field(
        default_factory=lambda: ["PROJECT_STRUCTURE.md", "README.md"]
    )
    max_chars_per_file: int = 8000   # max content sent to LLM per file
    model_max_tokens: int = 1024     # expected max summary response tokens


@dataclass
class SimpleChatMessage:
    role: str
    content: str


class ProjectStructureDocumenter:
    def __init__(
        self,
        adapter: LangChainOllamaAdapter,
        config: Optional[ProjectOverviewConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or ProjectOverviewConfig()

    # -----------------------------
    # 1) Collect project files
    # -----------------------------
    def iter_project_files(self) -> List[Path]:
        root = self.config.root_dir
        files: List[Path] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Filter excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.config.exclude_dirs]

            for filename in filenames:
                path = Path(dirpath) / filename

                if self.config.include_exts and path.suffix not in self.config.include_exts:
                    continue

                if any(substr in str(path) for substr in self.config.exclude_files_substring):
                    continue

                files.append(path)

        files.sort()
        return files

    # -----------------------------
    # 2) Generate summary for a file via LLM
    # -----------------------------
    def summarize_file(self, file_path: Path) -> str:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Unable to read file (error: {exc})"

        if len(text) > self.config.max_chars_per_file:
            text = text[: self.config.max_chars_per_file]

        rel_path = file_path.relative_to(self.config.root_dir)

        prompt = f"""
You are a senior AI systems architect documenting the Intergrax framework.

You will receive:
- File path within the project
- Content of the file (may be truncated if large)

Your output must provide a **concise but meaningful** technical explanation.

Task:
1. Write a 1-2 sentence high-level description of the file's purpose.
2. Identify the domain (e.g., "LLM adapters", "RAG logic", "data ingestion", "agents", "configuration", "utility modules").
3. List the key responsibilities / main functionality in bullet points.
4. If the file appears to be experimental, auxiliary, legacy, or incomplete - clearly state that.

Formatting rules (strict):

- "Description:" - one short paragraph.
- "Domain:" - one short phrase.
- "Key Responsibilities:" - bullet list.

Example format (do not invent new structure):

Description: This module provides utilities ...
Domain: xxx
Key Responsibilities:
- ...
- ...

FILE PATH:
{rel_path}

CONTENT:
\"\"\"
{text}
\"\"\"
""".strip()

        messages = [SimpleChatMessage(role="user", content=prompt)]

        response = self.adapter.generate_messages(
            messages,
            temperature=None,
            max_tokens=self.config.model_max_tokens,
        )

        return str(response).strip()

    # -----------------------------
    # 3) Build the Markdown file
    # -----------------------------
    def generate_markdown(self) -> str:
        files = self.iter_project_files()

        lines: List[str] = []
        lines.append(f"# Intergrax — Project Structure Overview\n")
        lines.append(
            "This document was generated automatically by the Intergrax "
            "Project Structure Document Generator."
        )
        lines.append("")
        lines.append("## Purpose")
        lines.append(
            "- Provide a clean overview of the current codebase structure.\n"
            "- Enable new developers to understand architectural roles quickly.\n"
            "- Serve as context for LLM agents (e.g., ChatGPT, Intergrax agents) "
            "to reason about and improve the project."
        )
        lines.append("")
        lines.append("## File Index")
        lines.append("")

        for f in files:
            rel = f.relative_to(self.config.root_dir)
            lines.append(f"- `{rel}`")

        lines.append("")
        lines.append("## Detailed File Documentation\n")

        for file_path in tqdm(files, desc="Generating file summaries", unit="file"):
            rel_path = file_path.relative_to(self.config.root_dir)
            lines.append(f"### `{rel_path}`\n")

            summary = self.summarize_file(file_path)
            lines.append(summary)
            lines.append("")

        return "\n".join(lines)

    # -----------------------------
    # 4) Write to disk
    # -----------------------------
    def run(self) -> Path:
        content = self.generate_markdown()
        self.config.output_md.write_text(content, encoding="utf-8")
        return self.config.output_md


# ================================
# Standalone CLI usage
# ================================
if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    adapter = LangChainOllamaAdapter(
        chat=ChatOllama(
            model=GlobalSettings.default_ollama_model
        )
    )

    config = ProjectOverviewConfig(
        root_dir=Path(".").resolve(),
        output_md=Path("PROJECT_STRUCTURE.md"),
        max_chars_per_file=8000,
        model_max_tokens=1024,
    )

    documenter = ProjectStructureDocumenter(adapter=adapter, config=config)
    output_file = documenter.run()
    print(f"Generated: {output_file}")
