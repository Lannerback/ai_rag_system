"""Prepends the governing Article heading to each chunk of a legal document.

`chunk_by_title` starts a new section at every Article title, but long articles
split across multiple chunks, and only the first chunk keeps the heading. The
continuation chunks then read as orphaned clauses (e.g. "1. interface tools..."),
which embed poorly and lose their legal anchor. This tagger tracks the current
Article across chunks and prefixes it where missing, so every chunk is a
self-contained, retrievable legal unit.
"""
import re
from typing import List

# Matches "Article 14" / "Article 5" as a heading token, optionally followed by a
# short title on the same or next line (e.g. "Article 14\n\nHuman oversight").
_ARTICLE_HEADING = re.compile(r"\bArticle\s+\d+\b", re.IGNORECASE)


class LegalSectionTagger:
    """Ensures each chunk is prefixed with the Article heading that governs it."""

    def tag(self, texts: List[str]) -> List[str]:
        tagged: List[str] = []
        current_heading: str | None = None

        for text in texts:
            heading = self._extract_heading(text)
            if heading is not None:
                current_heading = heading
                tagged.append(text)
            elif current_heading is not None and not _ARTICLE_HEADING.search(text):
                tagged.append(f"{current_heading}\n\n{text}")
            else:
                tagged.append(text)
        return tagged

    @staticmethod
    def _extract_heading(text: str) -> str | None:
        """Return the leading Article heading (with its inline title) if the chunk starts one."""
        stripped = text.lstrip()
        match = _ARTICLE_HEADING.match(stripped)
        if match is None:
            return None
        # Capture "Article N" plus an immediately following short title line.
        remainder = stripped[match.end():]
        title_line = remainder.split("\n", 2)[0].strip() if remainder else ""
        heading = match.group(0)
        if title_line and len(title_line) <= 80 and not title_line[0].isdigit():
            heading = f"{heading} {title_line}"
        return heading
