"""
Migration Guide for AutoGen Backend Refactoring

This document describes how to update imports when migrating from autogen-specific 
components to shared components.

## Import Changes

### Before (autogen-specific imports)
```python
from linj_autogen.core.document import LinJDocument
from linj_autogen.core.nodes import Node, ToolNode
from linj_autogen.core.state import StateManager
from linj_autogen.contitext.engine import ContiTextEngine
```

### After (shared component imports)
```python
from shared.core.document import LinJDocument
from shared.core.nodes import Node, ToolNode
from shared.core.state import StateManager
from shared.contitext.engine import ContiTextEngine
```

### Compatibility Layer (recommended transition path)
```python
# Use compatibility layer for smooth transition
from linj_autogen.core.compat import LinJDocument, Node, ToolNode, StateManager
from linj_autogen.contitext.compat import ContiTextEngine
```

## Key Differences

1. **Error Classes**: Moved to `shared.exceptions.errors` module
2. **Path Resolution**: Core path utilities remain in `shared.core.path`
3. **Executor Backend**: New shared backend interface in `shared.executor.backend`
4. **Type Definitions**: Common types moved to `shared.types.common`

## Migration Steps

1. **Phase 1**: Use compatibility layers (compat.py modules)
2. **Phase 2**: Gradually update imports to shared components
3. **Phase 3**: Remove compatibility layers once migration is complete

## Backward Compatibility

The refactoring maintains full backward compatibility:
- All existing APIs continue to work
- No breaking changes to public interfaces
- Existing examples and tests should continue to run

## New Features

The refactoring enables:
- Multi-backend support (AutoGen, LangGraph, custom backends)
- Shared execution engine abstractions
- Better code reuse between different AI frameworks
- Unified type definitions and error handling

## Example Usage

```python
# New recommended way using AutoGenExecutionBackend
from autogen.autogen.backend import AutoGenExecutionBackend
from shared.core.document import LinJDocument

# Create backend
backend = AutoGenExecutionBackend(enable_contitext=True, enable_parallel=True)
backend.set_document(document)

# Execute
result = await backend.execute_document(initial_state)
```
"""