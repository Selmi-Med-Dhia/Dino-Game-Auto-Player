import ast
from pathlib import Path


def test_app_parses():
    src = Path(__file__).parents[1] / "app.py"
    ast.parse(src.read_text(encoding="utf-8"))
