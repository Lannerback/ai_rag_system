# AGENT.md

This file defines how agents and contributors should work in this repository.

## Product Context

This project is a reusable Retrieval-Augmented Generation product. It is intended to be used by other developers and teams as a configurable foundation for building documentation-backed AI assistants.

The system must stay extensible so users can plug in different LLM providers, embedding models, vector stores, document loaders, and retrieval strategies without rewriting the application core. Provider-specific behavior belongs behind clear abstractions, not scattered through conditional logic.

## Tech Stack

- Python
- FastAPI
- LangChain
- FAISS
- Gemini
- Azure OpenAI
- YAML/env-driven configuration

## Non-Negotiable Rules

1. Keep the project config-driven.
2. Do not hardcode provider/model choices in business logic.
3. Any new dependency must be added to both:
   - `requirements.txt`
   - `environment.yml`
4. Keep API compatibility for `/ask` unless explicitly changing the contract.
5. Avoid destructive git operations (`reset --hard`, force cleanups) unless explicitly requested.
6. Never use `any` as a type. Use explicit, precise types.
7. Never hardcode secrets, API keys, credentials, or environment-specific values.

## Code Design Rules

1. Write reusable, composable code. Avoid one-off implementations when the behavior is likely to be reused.
2. Prefer OOP interfaces and concrete implementations when functionality can be extended by multiple implementers.
3. Keep business logic behind abstractions. Do not leak provider-specific details into unrelated layers.
4. Use design patterns when they reduce complexity, remove branching, or isolate behavior.
5. Prefer Chain of Responsibility for ordered fallback/handler pipelines.
6. Prefer Adapter when integrating external SDKs, APIs, or incompatible interfaces.
7. Prefer Facade when exposing a simple API over a complex subsystem.
8. Prefer Factory or Strategy when selecting providers, models, loaders, retrievers, or algorithms.
9. Avoid long `if`/`elif` chains for extensible behavior. Replace them with polymorphism, registries, factories, or handlers.
10. Keep modules focused on one responsibility. Do not create god files.

## Logging Guidelines

- Keep startup logs informative but not noisy.
- Avoid turning on verbose parser debug logs in normal runs.
- Use `INFO` for lifecycle and `DEBUG` only for targeted diagnostics.

## Development Workflow

1. Read config and affected modules before edits.
2. Make minimal, targeted changes.
3. Run quick syntax checks:
   - `python -m py_compile <changed_python_files>`
4. After Python changes, run the Python linting skill before finalizing.
5. Run SonarLint after every change.
6. If behavior changes, run focused tests first, then broader tests when practical.
7. Summarize behavior changes, verification performed, and any migration/rebuild requirements.
