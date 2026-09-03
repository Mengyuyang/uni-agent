import os
import sys
from types import ModuleType

from uni_agent.runtime_env import pythonpath_with_resolved_package_roots


def test_pythonpath_prepends_physical_editable_package_root(tmp_path, monkeypatch):
    package_root = tmp_path / "vllm-source"
    package_dir = package_root / "vllm"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("", encoding="utf-8")

    module = ModuleType("fake_vllm")
    module.__file__ = str(init_file)
    monkeypatch.setitem(sys.modules, "fake_vllm", module)

    existing = os.pathsep.join(("/vllm", "/existing"))
    result = pythonpath_with_resolved_package_roots(("fake_vllm",), existing, "/existing")

    assert result.split(os.pathsep) == [str(package_root.resolve()), "/vllm", "/existing"]


def test_pythonpath_ignores_namespace_package_without_file(monkeypatch):
    module = ModuleType("namespace_only")
    module.__file__ = None
    monkeypatch.setitem(sys.modules, "namespace_only", module)

    assert pythonpath_with_resolved_package_roots(("namespace_only",), "/existing") == "/existing"
