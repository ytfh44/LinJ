# Shared Executor Layer

**Role:** Unified execution patterns (scheduler, evaluator, adapters)

## Overview

Execution engine abstractions supporting both LangGraph and AutoGen backends. Contains adapter patterns and scheduling logic.

## Where to Look

| Task | Location |
|------|----------|
| Main scheduler | `shared/executor/scheduler.py` |
| Evaluator | `shared/executor/evaluator.py` |
| Adapter base | `shared/executor/adapter.py` |
| LangGraph adapter | `shared/executor/langgraph_adapter.py` |
| AutoGen adapter | `shared/executor/autogen_adapter.py` |
| Context | `shared/executor/context.py` |

## Key Files

- `shared/executor/autogen_scheduler.py` (560 lines) - AutoGen-specific scheduling
- `shared/executor/evaluator.py` (539 lines) - Evaluation logic
- `shared/executor/context.py` (452 lines) - Execution context

## Conventions

- Async-first design
- Adapters implement common interface
- Context passed explicitly to all operations
