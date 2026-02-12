# LangGraph Integration

**Role:** LangGraph backend adapters and workflow execution

## Overview

LangGraph-specific adapters and workflows. Consumes `shared/` core abstractions.

## Structure

```
langgraph/
├── adapters/       # Contitext + executor adapters
├── nodes/          # Node implementations
├── utils/          # LangGraph utilities
└── workflows/      # Workflow definitions
```

## Where to Look

| Task | Location |
|------|----------|
| Executor adapter | `langgraph/adapters/executor_adapter.py` |
| Contitext adapter | `langgraph/adapters/contitext_adapter.py` |
| Base adapter | `langgraph/adapters/base.py` |
| State schema | `langgraph/state.py` |
| Node base | `langgraph/nodes/base.py` |

## Key Files

- `examples/langgraph_backend.py` (1242 lines) - Full example
- `langgraph/adapters/executor_adapter.py` - Executor → LangGraph bridge

## Conventions

- Uses langgraph>=0.2.0 API
- State extends `shared/core/state.py` patterns
- Adapters implement `shared/executor/adapter.py` interface
