# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from intergrax.prompts.storage.yaml_loader import YamlPromptLoader
from intergrax.prompts.storage.models import LoadedPrompt, PromptNotFound
from intergrax.prompts.registry.pin_config import PromptPinConfig


class YamlPromptRegistry:
    """
    Production registry backed by YAML catalog.
    """

    def __init__(
        self,
        catalog_dir: Path,
        loader: Optional[YamlPromptLoader] = None,
    ) -> None:
        self._catalog_dir = catalog_dir
        self._loader = loader or YamlPromptLoader()
        self._cache: Dict[str, Dict[int, LoadedPrompt]] = {}

    # ---------------------------------------------------------------------

    def load_all(self) -> None:
        """
        Scan catalog directory and load all prompt versions.
        Structure expected:

        catalog/
          prompt_id/
            1.yaml
            2.yaml
            stable.yaml  -> contains {"stable": 2}
        """
        for prompt_dir in self._catalog_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            pid = prompt_dir.name
            self._cache[pid] = {}

            for f in prompt_dir.glob("*.yaml"):
                if f.name == "stable.yaml":
                    continue

                lp = self._loader.load(f)
                self._cache[pid][lp.document.version] = lp

    # ---------------------------------------------------------------------

    def resolve(
        self,
        prompt_id: str,
        pin: Optional[PromptPinConfig] = None,
    ) -> LoadedPrompt:

        if prompt_id not in self._cache:
            raise PromptNotFound(prompt_id)

        versions = self._cache[prompt_id]

        # 1. Pinned version
        if pin:
            v = pin.get(prompt_id)
            if v is not None and v in versions:
                return versions[v]

        # 2. Stable version
        stable = self._read_stable(prompt_id)
        if stable in versions:
            return versions[stable]

        # 3. Safety: no implicit latest
        raise PromptNotFound(
            f"No resolvable version for '{prompt_id}'"
        )

    # ---------------------------------------------------------------------

    def _read_stable(self, prompt_id: str) -> int:
        path = self._catalog_dir / prompt_id / "stable.yaml"

        if not path.exists():
            raise PromptNotFound(
                f"Missing stable.yaml for '{prompt_id}'"
            )

        data = self._loader._read_yaml(path)

        v = data.get("stable")
        if not isinstance(v, int):
            raise PromptNotFound(
                f"Invalid stable.yaml for '{prompt_id}'"
            )

        return v
