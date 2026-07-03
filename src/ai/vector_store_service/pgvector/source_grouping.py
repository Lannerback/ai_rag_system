from collections import OrderedDict
from typing import Dict, List, Tuple

SourceGroups = OrderedDict[str, List[Tuple[str, Dict]]]


def group_by_source(texts: List[str], metadatas: List[Dict]) -> SourceGroups:
    """Group (text, metadata) pairs by their canonical `source`, preserving order."""
    grouped: SourceGroups = OrderedDict()
    for text, metadata in zip(texts, metadatas):
        source = (metadata or {}).get("source", "unknown")
        grouped.setdefault(source, []).append((text, metadata or {}))
    return grouped


def document_metadata(metadatas: List[Dict]) -> Dict:
    """Derive document-level metadata: shared keys minus chunk-specific ones."""
    first = dict(metadatas[0]) if metadatas else {}
    first.pop("page", None)
    first.pop("source", None)
    return first
