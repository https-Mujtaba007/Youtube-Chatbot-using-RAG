import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "youtube_chat.py"


def _streamlit_calls(tree):
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
        ):
            names.add(node.func.attr)
    return names


def test_no_misspelled_streamlit_tile_call():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assert "tile" not in _streamlit_calls(tree)


def test_title_is_rendered_with_st_title():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assert "title" in _streamlit_calls(tree)
