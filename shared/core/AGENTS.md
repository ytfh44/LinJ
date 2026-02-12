# Shared Core Domain Models

**Role:** Base domain abstractions (state, edges, nodes, changeset, conditions)

## Overview

Core domain types used across the monorepo. Contains the foundational data models.

## Where to Look

| Task | Location |
|------|----------|
| State definition | `shared/core/state.py` |
| Edge types | `shared/core/edges.py` |
| Condition evaluation | `shared/core/condition.py` |
| Document model | `shared/core/document.py` |
| Changeset | `shared/core/changeset.py` |
| Path utilities | `shared/core/path.py` |

## Key Files

- `shared/core/condition.py` (755 lines) - Complex conditional logic
- `shared/core/tracing.py` (613 lines) - Tracing integration

## Conventions

- All classes are Pydantic v2 models
- Strict typing enforced (mypy)
- No mutable state; use `.model_copy(update=...)`
