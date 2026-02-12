# LinJ Project Knowledge Base

**Generated:** 2026-02-12 21:20
**Type:** Python Monorepo (LangGraph + AutoGen)

## Overview

Agent workflow framework with shared core, LangGraph adapters, and AutoGen integration. Supports both LangGraph and AutoGen executors with unified state management.

## Structure

```
./
├── shared/           # Shared core (state, edges, conditions, executor)
├── autogen/          # AutoGen adapter layer
├── langgraph/        # LangGraph adapters
├── examples/         # Usage examples
├── config/           # Config storage
└── theory/           # Design docs
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| State management | `shared/core/state.py`, `langgraph/state.py` | Unified state interface |
| Executor patterns | `shared/executor/` | Scheduler, evaluator, adapter |
| AutoGen integration | `autogen/src/linj_autogen/` | Bridge layer |
| Workflow definition | `examples/**/*.py`, `examples/**/workflow.yaml` | YAML + Python |
| Contitext | `shared/contitext/`, `autogen/src/linj_autogen/contitext/` | Context management |

## Conventions

- **Python**: 3.11+, strict typing (mypy strict mode)
- **Formatting**: ruff (line-length 88), double quotes
- **Testing**: pytest-asyncio, `tests/` and `autogen/tests/`
- **Async**: `asyncio_mode = auto` in pytest
- **Pydantic**: v2 (`pydantic>=2.12.5`)

## Anti-Patterns (THIS PROJECT)

- DO NOT mutate state directly; always use copy/update patterns
- DO NOT bypass contract validators in core modules
- DO NOT use implicit Optional types
- DO NOT commit without type checking

## Commands

```bash
pytest -v --tb=short          # Run tests
ruff check .                   # Lint
ruff format .                 # Format
mypy .                        # Type check
```

## Dependencies

- langgraph>=0.2.0, langchain>=0.3.0 (core)
- pydantic>=2.12.5, structlog>=25.5.0
- httpx>=0.27.0 (async HTTP)
