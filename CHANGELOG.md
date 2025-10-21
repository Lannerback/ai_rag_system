# Changelog

All notable changes to this project are documented here. Dates use YYYY-MM-DD. This file summarizes the most important items from git history and recent refactors performed in this session.

## [2025-10-21] Increased recall and tuning
- Increased recall by adjusting chunk size and `k` in retrieval.
- Added `top_p` to Azure configuration.
- Merged PR: `patch/increase_recall`.

## [2025-10-12] PDF/OCR support
- Added OCR support for scanned PDFs.
- Added multi-document extension support.
- Validated folder existence before processing to prevent crashes.
- Merged PR: `feat/pdf-support`.

## [2025-10-10]
- Refactored code changing sructeadn paadigms, removed FASTAPI context
- Centralized startup wiring moved to `src/startup.py` with `initialize_vector_store()` and `initialize_rag_facade()`.
- `RagFacade` simplified to a pure facade; receives `RagService` via constructor.
- `RagService` now takes `llm` and `vector_store` explicitly via constructor.
- `VectorStoreFacade` now wraps a provided `BaseVectorStore` (no internal context access).
- `VectorStoreInitializer` accepts `FaissVectorStore` and `DocumentLoaderFacade` via constructor.
- `api.py` lifespan composes app-scoped singletons once and exposes only the top-level `RagFacade`.



## [2025-10-08] LLM extractor and architecture cleanup
- Introduced FastAPI context loader to reduce tight coupling (later improved to explicit DI).
- Added LLM extractor (vision-based extraction for scanned PDFs).
- Finalized refactoring round focused on document processing and extraction paths.
- Merged PR: `feat/llm_extractor`.

## [2025-10-02] Refactors and fixes
- Heavy refactor of codebase and fixing LLM extraction edge cases.
- Removed dotenv loading from builder and cleaned environment handling.
- Removed limit check from Gemini model.
- Refactored `config.yaml` and several naming inconsistencies.

## [2025-10-01] Mid-refactor checkpoints
- Large-scale refactor in progress; stabilized abstraction boundaries (first draft).

## [2025-08-05] Documentation improvements
- Improved and fixed documentation (README and usage notes).
- Refactored Azure configuration handling.

## [2025-07-31] Dependencies and libraries refactor
- Installed correct dependencies and adjusted `requirements.txt`.
- Refactoring of libraries and code; applied proper API key naming.
- Merged PR: `refactoring-libraries`.

## [2025-07-30] Configuration driven setup
- Added configuration-driven providers and updated README.
- Fixed Azure config reading.
- Merged PR: `feat/add-config`.

## [2025-07-27] Gemini support & initial refactors
- Added Gemini provider support.
- Externalized service selection by configuration.
- Refactored facades and services; first working draft committed.
- Merged PR: `refactoring`.

## Notes
- This file highlights impactful changes (features, refactors, integrations). For the full history, see `git log`.
