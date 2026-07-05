"""Noise filtering for partitioned document elements.

Legal PDFs (e.g. the EU AI Act) yield layout chrome and margin artifacts that
pollute retrieval: running headers, page footers, and standalone recital/footnote
number markers such as ``(9)`` or ``(7) (8) (9)``. Embedding these produces
near-duplicate, meaningless vectors that dilute precision. This cleaner drops
them before chunking, where element categories are still available.
"""
import re
from typing import Iterable, List

# Layout chrome with no semantic value.
_DROP_CATEGORIES = frozenset({"Header", "Footer", "PageNumber", "PageBreak", "Image"})

# Standalone marker fragments: only digits, parentheses, dots, whitespace.
_PURE_MARKER = re.compile(r"^[\(\)\d\s.]+$")

# Running-header leak ("(Artificial Intelligence Act)") miscategorized as body text.
_HEADER_NOISE = re.compile(r"^\(?\s*artificial\s+intelligence\s+act\s*\)?$", re.IGNORECASE)

# ELI permalink footer ("ELI: http://data.europa.eu/eli/...") that hi_res emits as body text.
_ELI_FOOTER = re.compile(r"^ELI:\s*https?://", re.IGNORECASE)

# Elements shorter than this after stripping are dropped as fragments.
_MIN_ELEMENT_CHARS = 15


class ElementCleaner:
    """Filters out layout chrome and margin-number noise from partitioned elements."""

    def clean(self, elements: Iterable) -> List:
        return [element for element in elements if self._keep(element)]

    def _keep(self, element) -> bool:
        if getattr(element, "category", None) in _DROP_CATEGORIES:
            return False

        text = (getattr(element, "text", None) or "").strip()
        if len(text) < _MIN_ELEMENT_CHARS:
            return False
        if _PURE_MARKER.fullmatch(text):
            return False

        normalized = text.replace("\n", " ").strip()
        if _HEADER_NOISE.fullmatch(normalized):
            return False
        if _ELI_FOOTER.match(normalized):
            return False
        return True
