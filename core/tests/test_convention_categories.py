"""Tests for the shared convention-category taxonomy (docs/loupe-scaffold.md §1)."""

from loupe_core.adapters.fastapi.convention_categories import ConventionCategory


def test_taxonomy_names_the_five_documented_categories():
    assert {c.value for c in ConventionCategory} == {
        "error_handling",
        "docstring_style",
        "import_style",
        "config_management",
        "dependency_injection",
    }


def test_categories_are_plain_string_enums_usable_as_dict_keys_or_tags():
    assert ConventionCategory.ERROR_HANDLING == "error_handling"
    assert isinstance(ConventionCategory.ERROR_HANDLING, str)
