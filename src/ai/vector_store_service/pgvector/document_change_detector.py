import hashlib
from typing import List


class DocumentChangeDetector:
    """Computes content checksums to skip re-embedding unchanged documents."""

    def checksum(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def checksum_many(self, contents: List[str]) -> str:
        """Stable checksum over an ordered list of chunk contents."""
        digest = hashlib.sha256()
        for content in contents:
            digest.update(content.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def has_changed(self, stored_checksum: str | None, current_checksum: str) -> bool:
        return stored_checksum != current_checksum
