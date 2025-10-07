from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseDocumentLoader(ABC):

    @property
    @abstractmethod
    def load_documents(self) -> Tuple[List[str], List[dict]]:
        """Load documents from the directory, split into chunks, and return texts + metadata."""
        pass