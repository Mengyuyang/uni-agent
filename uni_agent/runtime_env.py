"""Helpers for constructing Ray worker runtime environments."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterable


def pythonpath_with_resolved_package_roots(
    package_names: Iterable[str],
    *existing_pythonpaths: str | None,
) -> str:
    """Prepend physical package roots to an existing ``PYTHONPATH``.

    Editable installs can be represented by synthetic import hooks. Those hooks may
    resolve differently inside Ray workers (for example, ``vllm`` can become the
    namespace package ``/vllm``). Resolving the package in the driver and forwarding
    its physical source root makes nested workers import the same implementation.
    """
    entries: list[str] = []

    for package_name in package_names:
        module = importlib.import_module(package_name)
        module_file = getattr(module, "__file__", None)
        if module_file:
            entries.append(str(Path(module_file).resolve().parent.parent))

    for pythonpath in existing_pythonpaths:
        if pythonpath:
            entries.extend(entry for entry in pythonpath.split(os.pathsep) if entry)

    return os.pathsep.join(dict.fromkeys(entries))
