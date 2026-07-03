"""Dev server entrypoint with hot reload. Run with `uv run dev`."""
from dotenv import load_dotenv

load_dotenv()

import uvicorn


def main() -> None:
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["src"])


if __name__ == "__main__":
    main()
