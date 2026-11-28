# © Artur Czarnecki. All rights reserved.
"""Generate adapters and install this repository's local plugin, preserving user config."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = "intergrax-codex@intergrax-codex-local"


def without_plugin(data: dict) -> dict:
    # Compare structure, never print config values or write a secret-bearing backup.
    result = dict(data)
    plugins = dict(result.get("plugins", {}))
    plugins.pop(PLUGIN, None)
    if plugins:
        result["plugins"] = plugins
    else:
        result.pop("plugins", None)
    return result


def main() -> int:
    codex = shutil.which("codex")
    if not codex:
        print("Codex CLI is required on PATH.", file=sys.stderr)
        return 1
    generated = subprocess.run(
        [sys.executable, str(ROOT / "scripts/governance/sync_agent_env.py")],
        cwd=ROOT,
        check=False,
    )
    if generated.returncode:
        return generated.returncode
    config = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"
    original = config.read_bytes() if config.is_file() else None
    original_data = tomllib.loads(original.decode("utf-8")) if original is not None else {}
    installed = subprocess.run(
        [
            codex,
            "-c",
            'marketplaces.intergrax-codex-local.source_type="local"',
            "-c",
            f"marketplaces.intergrax-codex-local.source={json.dumps(str(ROOT))}",
            "plugin",
            "add",
            PLUGIN,
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    # `plugin add` writes enablement to user config. The repository owns enablement;
    # restore user config only if all semantic changes are confined to our plugin.
    if config.is_file():
        after = config.read_bytes()
        if after != original:
            after_data = tomllib.loads(after.decode("utf-8"))
            if without_plugin(after_data) != without_plugin(original_data):
                print(
                    "User config changed outside this plugin during setup; left it intact for review.", file=sys.stderr
                )
                return 1
            if config.read_bytes() != after:
                print("User config changed concurrently; left it intact for review.", file=sys.stderr)
                return 1
            if original is None:
                config.unlink()
            else:
                config.write_bytes(original)
    if installed.returncode:
        print("Codex could not install the local plugin. Inspect it with codex plugin list.", file=sys.stderr)
        return installed.returncode
    print("Installed Intergrax Codex plugin in the local cache; user config preserved.")
    print("Reopen the project in Codex and review/trust .codex/hooks.json in Settings > Hooks or /hooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
