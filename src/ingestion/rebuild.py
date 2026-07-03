"""Wipe and reinitialize the pgvector schema from scratch.

Run with `python -m src.ingestion.rebuild --yes`.

Destructive: drops every pgvector table and recreates the schema (including the
`vector` extension) from the Alembic migration. It does NOT re-ingest documents —
run `python -m src.ingestion --collection <name>` afterwards to repopulate.
"""
import argparse
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()

from alembic import command  # noqa: E402  (after load_dotenv)
from alembic.config import Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"


def _safe_target(database_url: str) -> str:
    """Return host:port/db without credentials, safe for logging."""
    parts = urlsplit(database_url)
    return f"{parts.hostname}:{parts.port}{parts.path}"


def rebuild() -> None:
    """Drop all tables and recreate the schema via Alembic (downgrade base, upgrade head)."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set; it is required to rebuild the database.")

    config = Config(str(_ALEMBIC_INI))

    logger.warning("Wiping pgvector schema at %s", _safe_target(database_url))
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    logger.info(
        "Database wiped and schema reinitialized. "
        "Run `python -m src.ingestion --collection <name>` to repopulate."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wipe and reinitialize the pgvector schema from scratch (destructive)."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the destructive wipe. Required; without it the command aborts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.yes:
        logger.error("Refusing to wipe the database without --yes. Re-run with --yes to confirm.")
        raise SystemExit(1)
    rebuild()


if __name__ == "__main__":
    main()
