# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
import threading
from typing import Dict, Optional

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.prompts.schema.prompt_schema import (
    LocalizedContent,
)
from intergrax.prompts.storage.yaml_loader import YamlPromptLoader
from intergrax.prompts.storage.models import LoadedPrompt, PromptNotFound
from intergrax.prompts.registry.pin_config import PromptPinConfig


class YamlPromptRegistry:
    """
    Production registry backed by YAML catalog.
    """

    _default_instance : Dict[Path, "YamlPromptRegistry"] = {}
    _default_lock = threading.Lock()

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

        Expected structure:

        catalog/
          prompt_id/
            1.yaml
            2.yaml
            stable.yaml  -> { stable: 2 }
        """
        for prompt_dir in self._catalog_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            prompt_id = prompt_dir.name
            self._cache[prompt_id] = {}

            for f in prompt_dir.glob("*.yaml"):
                if f.name == "stable.yaml":
                    continue

                loaded = self._loader.load(f)
                self._cache[prompt_id][loaded.document.version] = loaded

    # ---------------------------------------------------------------------

    def resolve(
        self,
        prompt_id: str,
        pin: Optional[PromptPinConfig] = None,
    ) -> LoadedPrompt:
        """
        Resolve prompt version (pin -> stable).
        Does NOT handle localization.
        """

        if prompt_id not in self._cache:
            raise PromptNotFound(prompt_id)

        versions = self._cache[prompt_id]

        # 1. Pinned version
        if pin is not None:
            pinned_version = pin.get(prompt_id)
            if pinned_version is not None and pinned_version in versions:
                return versions[pinned_version]

        # 2. Stable version
        stable_version = self._read_stable(prompt_id)
        if stable_version in versions:
            return versions[stable_version]

        # 3. Safety: no implicit latest
        raise PromptNotFound(f"No resolvable version for '{prompt_id}'")

    # ---------------------------------------------------------------------

    def resolve_localized(
        self,
        prompt_id: str,
        language: Optional[str] = None,
        pin: Optional[PromptPinConfig] = None,
    ) -> LocalizedContent:
        """
        Resolve prompt and return localized content.

        Language resolution order:
        1) explicit `language`
        2) GLOBAL_SETTINGS.default_language
        3) fallback to 'en'

        Contract:
        - Registry operates on LocalizedPromptDocument artifacts.
        """

        loaded = self.resolve(prompt_id, pin=pin)
        doc = loaded.document

        lang = language or GLOBAL_SETTINGS.default_language
        if lang:
            lang = lang.lower()
        else:
            lang = "en"

        locales = doc.locales

        if lang in locales:
            return locales[lang]

        if "en" in locales:
            return locales["en"]

        raise PromptNotFound(
            f"No locale '{lang}' and no 'en' fallback for prompt '{prompt_id}'"
        )


    # ---------------------------------------------------------------------

    def _read_stable(self, prompt_id: str) -> int:
        path = self._catalog_dir / prompt_id / "stable.yaml"

        if not path.exists():
            raise PromptNotFound(f"Missing stable.yaml for '{prompt_id}'")

        # NOTE: _read_yaml is an internal loader method, but we keep usage local.
        data = self._loader._read_yaml(path)

        version = data.get("stable")
        if not isinstance(version, int):
            raise PromptNotFound(f"Invalid stable.yaml for '{prompt_id}'")

        return version
    

    @classmethod
    def create_default(
        cls, 
        path: Optional[str] = None, 
        load:bool = True
    )-> YamlPromptRegistry:

        target_path = Path(path or "intergrax/prompts/catalog")

        if target_path in cls._default_instance:
            return cls._default_instance[target_path]

        with cls._default_lock:
            instance = cls(catalog_dir=target_path)

            if load:
                instance.load_all()

            cls._default_instance[target_path] = instance

            return instance
        

    @classmethod
    def reset_defaults(cls, path: Optional[str] = None) -> None:
        if path is None:
            cls._default_instance.clear()
        else:
            cls._default_instance.pop(Path(path), None)
