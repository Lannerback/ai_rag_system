from types import SimpleNamespace

import pytest

from src.ai.document_loaders.element_cleaner import ElementCleaner


def _el(text, category="NarrativeText"):
    return SimpleNamespace(text=text, category=category)


@pytest.fixture
def cleaner():
    return ElementCleaner()


@pytest.mark.parametrize("category", ["Header", "Footer", "PageNumber", "PageBreak", "Image"])
def test_drops_layout_chrome(cleaner, category):
    assert cleaner.clean([_el("some text", category)]) == []


@pytest.mark.parametrize("category", ["Title", "NarrativeText", "ListItem", "Table"])
def test_keeps_real_content(cleaner, category):
    element = _el("A substantive clause of the regulation.", category)
    assert cleaner.clean([element]) == [element]


@pytest.mark.parametrize("marker", ["(9)", "(7) (8) (9)", "(34)\n\n(35)\n\n(18)", "12."])
def test_drops_pure_marker_fragments(cleaner, marker):
    assert cleaner.clean([_el(marker)]) == []


def test_drops_running_header_leak(cleaner):
    assert cleaner.clean([_el("(Artificial\n\nIntelligence Act)")]) == []
    assert cleaner.clean([_el("Artificial Intelligence Act")]) == []


def test_drops_too_short_fragments(cleaner):
    assert cleaner.clean([_el("short")]) == []


def test_drops_eli_footer(cleaner):
    assert cleaner.clean([_el("ELI: http://data.europa.eu/eli/reg/2024/1689/oj")]) == []


def test_keeps_marker_prefixed_real_text(cleaner):
    element = _el("(a) the placing on the market of an AI system that deploys subliminal techniques")
    assert cleaner.clean([element]) == [element]
