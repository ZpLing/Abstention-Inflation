"""Thin compatibility wrapper around :mod:`loader.dataset_loader`.

Historical runners (``paired_pass``, the model sweep, …) ask a ``DataHandler`` for samples via ``load_dataset(name)``. In
this paper-aligned package the entire dataset story is one unified loader, so
the handler is just a small adaptor that:

* delegates to :py:func:`loader.dataset_loader.load_dataset`, and
* implements legacy ``DataHandler._get_path`` lookups by routing
  ``raw_dataset_template`` through the bundled ``dataset/`` directory.

A YAML config can still declare ``paths.raw_dataset_template`` (e.g.
``"dataset/{dataset_name}.json"``); leaving it unset is fine — the default
resolves to the same place.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from loader.dataset_loader import (
    DEFAULT_DATASET_ROOT,
    PAPER_DATASETS,
    Sample,
    load_dataset as _load_dataset,
)


class DataHandler:
    """Lightweight stand-in for the legacy ``src.data_handler.DataHandler``."""

    JUDGE_NAMES = {"FLD", "FLD_unknown", "FOLIO", "FOLIO_unknown"}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get("dataset_name", "")
        self.model_name = config.get("model_name", "")
        paths = config.get("paths") or {}
        template = paths.get("raw_dataset_template")
        self._raw_template = template

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _get_path(self, key: str) -> Path:
        paths = self.config.get("paths") or {}
        tpl = paths.get(key, "")
        if not tpl:
            raise KeyError(f"paths.{key} not configured in YAML.")
        return Path(
            tpl.format(dataset_name=self.dataset_name,
                       model_name=self.model_name)
        )

    # ------------------------------------------------------------------
    # Unified dataset access — the only entry point the new runners need
    # ------------------------------------------------------------------

    def load_dataset(self, name: str) -> List[Sample]:
        """Return all items of ``dataset/<name>.json`` as ``Sample`` instances."""
        if self._raw_template:
            # Honour a YAML-supplied template so non-default datasets still work.
            path = Path(self._raw_template.format(
                dataset_name=name, model_name=self.model_name))
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    items = json.load(f)
                return [Sample.from_dict(d) for d in items]
        return _load_dataset(name)

    # ------------------------------------------------------------------
    # Result writers (kept minimal — runners that need richer file I/O bring
    # their own helpers; these two are used by the AB / supplementary paths).
    # ------------------------------------------------------------------

    @staticmethod
    def save_json(obj: Any, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    @staticmethod
    def load_json(path: Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))
