# Shared Core Module

**Role:** Shared abstractions for state, edges, conditions, and execution

## Overview

Shared utilities consumed by both `autogen/` and `langgraph/` packages. Contains core domain models.

## Structure

```
shared/
├── core/          # Domain models (state, edges, nodes, conditions)
├── executor/       # Execution layer (scheduler, evaluator, adapters)
├── contitext/     # Context management
├── exceptions/    # Error types
└── types/         # Common types
```

## Where to Look

| Task | Location |
|------|----------|
| State schema | `shared/core/state.py` |
| Edge definitions | `shared/core/edges.py` |
| Condition logic | `shared/core/condition.py` |
| Scheduler | `shared/executor/scheduler.py` |
| Adapter patterns | `shared/executor/adapter.py` |

## Conventions

- Follows root project conventions (ruff, mypy strict)
- State classes use Pydantic v2 models
- All async functions use explicit type annotations

## Key Files

- `shared/core/condition.py` (755 lines) - Complex condition evaluation
- `shared/core/tracing.py` (613 lines) - OpenTelemetry integration
- `shared/executor/scheduler.py` (468 lines) - Core scheduling logic
