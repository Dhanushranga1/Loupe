"""Tests for parsing/ast_utils.py's class_field_annotations (a shared helper
for E7/E8/E9's zero-cost static analysis pack, docs/PhaseX/zero-cost-static-analysis-pack.md)."""

from loupe_core.graph.builder import parse_file
from loupe_core.parsing.ast_utils import class_field_annotations, symbol_nodes


def _fields_for(tmp_path, source: str, class_name: str = "Settings"):
    f = tmp_path / "a.py"
    f.write_text(source)
    pf = parse_file(str(f))
    class_node = next(node for node, symbol in symbol_nodes(pf) if symbol.qualified_name == class_name)
    return class_field_annotations(class_node, pf.source_bytes)


def test_annotated_field_without_default(tmp_path):
    fields = _fields_for(tmp_path, "class Settings:\n    api_key: str\n")
    assert len(fields) == 1
    assert fields[0].name == "api_key"
    assert fields[0].type_text == "str"
    assert fields[0].has_default is False


def test_annotated_field_with_default(tmp_path):
    fields = _fields_for(tmp_path, "class Settings:\n    debug: bool = False\n")
    assert len(fields) == 1
    assert fields[0].name == "debug"
    assert fields[0].has_default is True


def test_multiple_fields_in_order(tmp_path):
    fields = _fields_for(
        tmp_path, "class Settings:\n    api_key: str\n    debug: bool = False\n    timeout: int = 30\n"
    )
    assert [f.name for f in fields] == ["api_key", "debug", "timeout"]


def test_unannotated_assignment_is_not_a_field(tmp_path):
    fields = _fields_for(tmp_path, "class Settings:\n    CONSTANT = 5\n    api_key: str\n")
    assert [f.name for f in fields] == ["api_key"]


def test_method_inside_class_is_not_a_field(tmp_path):
    fields = _fields_for(tmp_path, "class Settings:\n    api_key: str\n\n    def validate(self):\n        return True\n")
    assert [f.name for f in fields] == ["api_key"]


def test_class_with_no_fields_returns_empty_list(tmp_path):
    fields = _fields_for(tmp_path, "class Settings:\n    def validate(self):\n        return True\n")
    assert fields == []
