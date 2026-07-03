from src.ai.embedders.base_embedder import BaseEmbedder
from src.ai.vector_store_service.pgvector.models import EmbeddingCollection


class CollectionMismatchError(ValueError):
    """Raised when an embedder does not match a collection's bound identity."""


class CollectionValidator:
    """Ensures the active embedder matches the collection's provider/model/dimension.

    A collection is permanently bound to the embedding configuration that produced
    it; querying or ingesting with a different embedder would corrupt results.
    """

    def validate(self, collection: EmbeddingCollection, embedder: BaseEmbedder) -> None:
        mismatches = []
        if collection.provider != embedder.provider:
            mismatches.append(f"provider {collection.provider!r} != {embedder.provider!r}")
        if collection.model != embedder.model:
            mismatches.append(f"model {collection.model!r} != {embedder.model!r}")
        if collection.dimensions != embedder.dimension:
            mismatches.append(f"dimensions {collection.dimensions} != {embedder.dimension}")

        if mismatches:
            raise CollectionMismatchError(
                f"Embedder does not match collection '{collection.name}': "
                + "; ".join(mismatches)
                + ". Use a new collection and re-ingest instead."
            )
