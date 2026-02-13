# Shared Execution Engine

Provides a shared execution engine abstraction layer for multi-backend architectures, supporting a unified execution interface for different AI frameworks.

## Overview

This module defines the core abstract interfaces of the execution engine, achieving the following goals:

1. **Multi-Backend Support**: Unified abstract interfaces support implementation of different execution backends.
2. **Framework Compatibility**: Adapts to mainstream AI frameworks like AutoGen and LangChain.
3. **Flexible Scheduling**: Supports multiple scheduling strategies and concurrent execution modes.
4. **State Management**: Unified state operations and context management.
5. **Type Safety**: Complete type annotations and runtime validation.

## Core Components

### 1. Execution Backend (Backend)

- **ExecutionBackend**: Execution engine abstract interface.
- **BaseExecutionBackend**: Base implementation providing common functionality.
- **DummyExecutionBackend**: Dummy implementation for testing and development.

```python
# Usage Example
backend = DummyExecutionBackend()
result = await backend.execute_node(node, context)
```

### 2. Tool Adapter (Adapter)

- **ToolAdapter**: Interface for tool function adaptation.
- **BaseToolAdapter**: Base implementation supporting tool registration and management.
- **LangChainToolAdapter**: Adapter for LangChain framework.
- **AutoGenToolAdapter**: Adapter for AutoGen framework.

```python
# Usage Example
adapter = AutoGenToolAdapter()
adapter.register_tool("echo", echo_function)
result = await adapter.execute_tool("echo", {"message": "hello"})
```

### 3. Scheduler (Scheduler)

- **Scheduler**: Scheduler abstract base class.
- **BaseScheduler**: Base scheduling logic.
- **DeterministicScheduler**: Deterministic scheduler.
- **ParallelScheduler**: Parallel scheduler.
- **PriorityScheduler**: Priority scheduler.
- **AutoGenDeterministicScheduler**: AutoGen-compatible scheduler.

```python
# Usage Example
scheduler = DeterministicScheduler(nodes)
decision = scheduler.select_nodes(ready_nodes, context)
```

### 4. Evaluator (Evaluator)

- **Evaluator**: Abstract interface for condition expression evaluation.
- **BaseEvaluator**: Base evaluation implementation.
- **SimpleEvaluator**: Simple expression evaluator.
- **RegexEvaluator**: Advanced evaluator based on regex.
- **AutoGenConditionEvaluator**: AutoGen-compatible evaluator.

```python
# Usage Example
evaluator = AutoGenConditionEvaluator(state)
result = evaluator.evaluate("$.value > 10 AND $.status == 'active'")
```

### 5. Context Management (Context)

- **ExecutionContext**: Execution context data structure.
- **StateManager**: Abstract interface for state management.
- **BaseStateManager**: In-memory state management implementation.
- **ContextManager**: Context lifecycle management.
- **PathResolver**: Path resolution and logical operations.

```python
# Usage Example
state_manager = BaseStateManager()
context_manager = ContextManager(state_manager)
context = context_manager.create_context("session1")
```

## Architecture Design

### Layered Architecture

```
┌─────────────────────────────────────┐
│           Application Layer         │
├─────────────────────────────────────┤
│            Abstract Layer           │
│  ExecutionBackend │ ToolAdapter     │
│  Scheduler        │ Evaluator       │
│  StateManager     │ ContextManager  │
├─────────────────────────────────────┤
│       Implementation Layer          │
│  AutoGenBackend   │ LangChainBackend│
│  AutoGenScheduler │ ParallelScheduler│
│  BaseStateManager │ DummyBackend    │
├─────────────────────────────────────┤
│              Core Layer             │
│     Types  │ Errors  │ Utils        │
└─────────────────────────────────────┘
```

### Key Design Principles

1. **Interface Separation**: Clear responsibility boundaries for each component.
2. **Dependency Injection**: Supports flexible combination of different implementations.
3. **Backward Compatibility**: Maintains compatibility with existing AutoGen code.
4. **Extensibility**: Easy to add new backend implementations.
5. **Type Safety**: Compile-time and runtime type checking.

## Usage Guidelines

### Basic Usage

```python
from shared.executor import (
    AutoGenDeterministicScheduler,
    AutoGenConditionEvaluator,
    BaseStateManager,
    ContextManager
)

# Create components
state_manager = BaseStateManager()
context_manager = ContextManager(state_manager)
scheduler = AutoGenDeterministicScheduler(nodes)
evaluator = AutoGenConditionEvaluator(state)

# Create execution context
context = context_manager.create_context("main", initial_state)

# Scheduling and Execution
decision = scheduler.select_nodes(ready_nodes, context)
for node in decision.selected_nodes:
    # Execute node logic
    pass
```

### Custom Backend

```python
from shared.executor import BaseExecutionBackend, ExecutionResult

class CustomBackend(BaseExecutionBackend):
    async def execute_node(self, node, context, tools=None):
        # Custom execution logic
        result = await self._custom_execute(node, context)
        return ExecutionResult(success=True, data=result)
    
    def validate_node(self, node, context):
        # Custom validation logic
        return True
    
    # Implement other required methods...
```

### Tool Adaptation

```python
from shared.executor import AutoGenToolAdapter

adapter = AutoGenToolAdapter()

# Register custom tool
def my_tool(param1: str, param2: int) -> dict:
    return {"result": f"{param1}_{param2}"}

adapter.register_tool("my_tool", my_tool)

# Execute tool
result = await adapter.execute_tool("my_tool", {
    "param1": "hello",
    "param2": 42
})
```

## Compatibility

### AutoGen Compatibility

- Maintains original scheduling logic and state management methods.
- Supports existing node types and dependency relationships.
- Compatible with existing condition expression syntax.

### Framework Adaptation

- **AutoGen**: Fully compatible with existing API.
- **LangChain**: Provides specialized adapter.
- **Generic**: Supports custom framework integration.

## Performance Features

### Caching Mechanism

- Expression evaluation result caching.
- Tool execution result caching.
- Path resolution result caching.

### Concurrency Support

- Multi-node parallel execution.
- Safety checks and conflict avoidance.
- Configurable concurrency levels.

### State Management

- Incremental state updates.
- Change logs and rollback.
- Observer pattern support.

## Extension Guide

### Adding a New Scheduler

```python
from shared.executor import BaseScheduler, SchedulingDecision

class CustomScheduler(BaseScheduler):
    def select_nodes(self, ready_nodes, context, max_concurrency=None):
        # Custom scheduling logic
        return SchedulingDecision(
            selected_nodes=selected,
            execution_order=[n.id for n in selected],
            concurrency_level=len(selected),
            strategy=SchedulingStrategy.CUSTOM,
            metadata={}
        )
```

### Adding a New Evaluator

```python
from shared.executor import BaseEvaluator, EvaluationResult

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, expression, context, strategy=None):
        # Custom evaluation logic
        return EvaluationResult(success=True, value=custom_eval_result)
    
    def tokenize(self, expression):
        # Custom lexical analysis
        pass
    
    def parse(self, tokens):
        # Custom syntax analysis
        pass
```

## Best Practices

1. **Type Annotations**: Always use complete type annotations.
2. **Error Handling**: Provide detailed error messages and recovery mechanisms.
3. **Docstrings**: Provide detailed documentation for all public APIs.
4. **Unit Tests**: Write comprehensive tests for each component.
5. **Performance Monitoring**: Utilize built-in statistics and monitoring functions.

## Troubleshooting

### Common Issues

1. **Import Errors**: Check if dependent modules are correctly installed.
2. **Type Mismatch**: Ensure correct data types are used.
3. **Dependency Cycles**: Use the scheduler's dependency check feature.
4. **State Conflicts**: Utilize conflict detection in the state manager.

### Debugging Tools

- Built-in statistics collection.
- Detailed logging.
- Execution history tracking.
- Performance metrics monitoring.
