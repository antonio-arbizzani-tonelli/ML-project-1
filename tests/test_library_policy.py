"""Keep every project implementation within the challenge import policy."""

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {
    "__future__",
    "argparse",
    "collections",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "gc",
    "hashlib",
    "implementations",
    "json",
    "math",
    "numpy",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "re",
    "src",
    "time",
    "typing",
}


class TestLibraryPolicy(unittest.TestCase):
    def test_project_logic_imports_only_permitted_libraries(self):
        paths = [
            PROJECT_ROOT / "implementations.py",
            PROJECT_ROOT / "run.py",
            *sorted((PROJECT_ROOT / "src").glob("*.py")),
        ]
        for path in paths:
            with self.subTest(path=path.name):
                syntax = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(syntax):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        modules = [node.module]
                    else:
                        continue
                    for module in modules:
                        root = module.split(".", 1)[0]
                        if root == "matplotlib":
                            self.assertEqual(path.name, "analyze_dataset.py")
                        else:
                            self.assertIn(root, ALLOWED_ROOTS, f"{path.name}: {module}")


if __name__ == "__main__":
    unittest.main()
