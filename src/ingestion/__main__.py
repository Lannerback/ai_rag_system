import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from src.ai.service_factory import ServiceFactory  # noqa: E402  (after load_dotenv)
from src.common.config import CONFIG  # noqa: E402
from src.ingestion.ingestion_service import IngestionService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingestion")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into a pgvector collection.")
    parser.add_argument(
        "--collection",
        default=CONFIG["vector_store"]["collection"],
        help="Target collection name (defaults to vector_store.collection in config.yaml).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-embed and replace every document, even if unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    embedder = ServiceFactory.get_embedder()
    service = IngestionService(embedder, args.collection)

    logger.info("Starting ingestion into collection '%s' (rebuild=%s)", args.collection, args.rebuild)
    report = service.ingest(rebuild=args.rebuild)
    logger.info(
        "Ingestion complete: created=%d updated=%d skipped=%d removed=%d failed=%d",
        report.created,
        report.updated,
        report.skipped,
        report.removed,
        len(report.failed),
    )
    if report.failed:
        logger.warning("Failed sources: %s", ", ".join(report.failed))


if __name__ == "__main__":
    main()
