# AutoGen Integration Layer

**Role:** ContiText/AutoGen adapter implementation

## Overview

AutoGen-specific implementation of ContiText patterns. Consumes `shared/` core abstractions and provides AutoGen backend integration.

## Structure

```
autogen/
├── src/linj_autogen/
│   ├── core/          # Core domain (changeset, edges, nodes, state)
│   ├── contitext/     # Context management (engine, signal, continuation)
│   ├── executor/      # Execution (scheduler, evaluator, runner)
│   └── autogen/       # AutoGen bridge (agent, backend, bridge)
├── tests/
│   ├── unit/
│   └── integration/
└── main.py            # CLI entry point
```

## Where to Look

| Task | Location |
|------|----------|
| CLI entry point | `autogen/main.py`, `autogen/src/linj_autogen/cli.py` |
| Contract validation | `autogen/src/linj_autogen/core/contract_validator.py` |
| Contitext engine | `autogen/src/linj_autogen/contitext/engine.py` |
| AutoGen bridge | `autogen/src/linj_autogen/autogen/bridge.py` |
| Scheduler | `autogen/src/linj_autogen/executor/scheduler.py` (504 lines) |
| Document model | `autogen/src/linj_autogen/core/document.py` (508 lines) |

## Key Files

- `autogen/src/linj_autogen/executor/scheduler.py` (504 lines)
- `autogen/src/linj_autogen/core/document.py` (508 lines)
- `autogen/src/linj_autogen/contitext/commit_manager.py` (561 lines)

## Conventions

- Inherits root project conventions (ruff, mypy strict)
- CLI via `click` (`linj-autogen` command)
- Tests: `pytest-asyncio` with `asyncio_mode = auto`

## Dependencies

- Core: click, pydantic, pyyaml, structlog
- Test: pytest, pytest-asyncio

## Entry Point

```bash
linj-autogen  # From autogen/main.py
```
