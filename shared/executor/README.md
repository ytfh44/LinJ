# Shared Execution Engine

提供多后端架构的共享执行引擎抽象层，支持不同AI框架的统一执行接口。

## 概述

本模块定义了执行引擎的核心抽象接口，实现以下目标：

1. **多后端支持**：统一的抽象接口支持不同执行后端实现
2. **框架兼容性**：适配AutoGen、LangChain等主流AI框架
3. **灵活调度**：支持多种调度策略和并发执行模式
4. **状态管理**：统一的状态操作和上下文管理
5. **类型安全**：完整的类型注解和运行时验证

## 核心组件

### 1. 执行后端 (Backend)

- **ExecutionBackend**: 执行引擎抽象接口
- **BaseExecutionBackend**: 基础实现，提供通用功能
- **DummyExecutionBackend**: 虚拟实现，用于测试和开发

```python
# 使用示例
backend = DummyExecutionBackend()
result = await backend.execute_node(node, context)
```

### 2. 工具适配器 (Adapter)

- **ToolAdapter**: 工具函数适配接口
- **BaseToolAdapter**: 基础实现，支持工具注册和管理
- **LangChainToolAdapter**: LangChain框架适配器
- **AutoGenToolAdapter**: AutoGen框架适配器

```python
# 使用示例
adapter = AutoGenToolAdapter()
adapter.register_tool("echo", echo_function)
result = await adapter.execute_tool("echo", {"message": "hello"})
```

### 3. 调度器 (Scheduler)

- **Scheduler**: 调度器抽象基类
- **BaseScheduler**: 基础调度逻辑
- **DeterministicScheduler**: 决定性调度器
- **ParallelScheduler**: 并行调度器
- **PriorityScheduler**: 优先级调度器
- **AutoGenDeterministicScheduler**: AutoGen兼容调度器

```python
# 使用示例
scheduler = DeterministicScheduler(nodes)
decision = scheduler.select_nodes(ready_nodes, context)
```

### 4. 求值器 (Evaluator)

- **Evaluator**: 条件表达式求值抽象接口
- **BaseEvaluator**: 基础求值实现
- **SimpleEvaluator**: 简单表达式求值器
- **RegexEvaluator**: 基于正则的高级求值器
- **AutoGenConditionEvaluator**: AutoGen兼容求值器

```python
# 使用示例
evaluator = AutoGenConditionEvaluator(state)
result = evaluator.evaluate("$.value > 10 AND $.status == 'active'")
```

### 5. 上下文管理 (Context)

- **ExecutionContext**: 执行上下文数据结构
- **StateManager**: 状态管理抽象接口
- **BaseStateManager**: 内存状态管理实现
- **ContextManager**: 上下文生命周期管理
- **PathResolver**: 路径解析和操作

```python
# 使用示例
state_manager = BaseStateManager()
context_manager = ContextManager(state_manager)
context = context_manager.create_context("session1")
```

## 架构设计

### 分层架构

```
┌─────────────────────────────────────┐
│           应用层 (Application)        │
├─────────────────────────────────────┤
│          抽象层 (Abstract)          │
│  ExecutionBackend │ ToolAdapter      │
│  Scheduler       │ Evaluator        │
│  StateManager    │ ContextManager   │
├─────────────────────────────────────┤
│         实现层 (Implementation)     │
│  AutoGenBackend │ LangChainBackend  │
│  AutoGenScheduler│ ParallelScheduler │
│  BaseStateManager│ DummyBackend     │
├─────────────────────────────────────┤
│           核心层 (Core)            │
│     Types │ Errors │ Utils          │
└─────────────────────────────────────┘
```

### 关键设计原则

1. **接口分离**: 每个组件都有清晰的职责边界
2. **依赖注入**: 支持不同实现的灵活组合
3. **向后兼容**: 保持与现有AutoGen代码的兼容性
4. **扩展性**: 易于添加新的后端实现
5. **类型安全**: 编译时和运行时的类型检查

## 使用指南

### 基本用法

```python
from shared.executor import (
    AutoGenDeterministicScheduler,
    AutoGenConditionEvaluator,
    BaseStateManager,
    ContextManager
)

# 创建组件
state_manager = BaseStateManager()
context_manager = ContextManager(state_manager)
scheduler = AutoGenDeterministicScheduler(nodes)
evaluator = AutoGenConditionEvaluator(state)

# 创建执行上下文
context = context_manager.create_context("main", initial_state)

# 调度和执行
decision = scheduler.select_nodes(ready_nodes, context)
for node in decision.selected_nodes:
    # 执行节点逻辑
    pass
```

### 自定义后端

```python
from shared.executor import BaseExecutionBackend, ExecutionResult

class CustomBackend(BaseExecutionBackend):
    async def execute_node(self, node, context, tools=None):
        # 自定义执行逻辑
        result = await self._custom_execute(node, context)
        return ExecutionResult(success=True, data=result)
    
    def validate_node(self, node, context):
        # 自定义验证逻辑
        return True
    
    # 实现其他必需方法...
```

### 工具适配

```python
from shared.executor import AutoGenToolAdapter

adapter = AutoGenToolAdapter()

# 注册自定义工具
def my_tool(param1: str, param2: int) -> dict:
    return {"result": f"{param1}_{param2}"}

adapter.register_tool("my_tool", my_tool)

# 执行工具
result = await adapter.execute_tool("my_tool", {
    "param1": "hello",
    "param2": 42
})
```

## 兼容性

### AutoGen兼容性

- 保持原有调度逻辑和状态管理方式
- 支持现有的节点类型和依赖关系
- 兼容现有的条件表达式语法

### 框架适配

- **AutoGen**: 完全兼容现有API
- **LangChain**: 提供专用适配器
- **通用**: 支持自定义框架集成

## 性能特性

### 缓存机制

- 表达式求值结果缓存
- 工具执行结果缓存
- 路径解析结果缓存

### 并发支持

- 多节点并行执行
- 安全性检查和冲突避免
- 可配置的并发级别

### 状态管理

- 增量状态更新
- 变更日志和回滚
- 观察者模式支持

## 扩展指南

### 添加新调度器

```python
from shared.executor import BaseScheduler, SchedulingDecision

class CustomScheduler(BaseScheduler):
    def select_nodes(self, ready_nodes, context, max_concurrency=None):
        # 自定义调度逻辑
        return SchedulingDecision(
            selected_nodes=selected,
            execution_order=[n.id for n in selected],
            concurrency_level=len(selected),
            strategy=SchedulingStrategy.CUSTOM,
            metadata={}
        )
```

### 添加新求值器

```python
from shared.executor import BaseEvaluator, EvaluationResult

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, expression, context, strategy=None):
        # 自定义求值逻辑
        return EvaluationResult(success=True, value=custom_eval_result)
    
    def tokenize(self, expression):
        # 自定义词法分析
        pass
    
    def parse(self, tokens):
        # 自定义语法分析
        pass
```

## 最佳实践

1. **类型注解**: 始终使用完整的类型注解
2. **错误处理**: 提供详细的错误信息和恢复机制
3. **文档字符串**: 为所有公共API提供详细文档
4. **单元测试**: 为每个组件编写全面的测试
5. **性能监控**: 利用内置的统计和监控功能

## 故障排除

### 常见问题

1. **导入错误**: 检查依赖模块是否正确安装
2. **类型不匹配**: 确保使用正确的数据类型
3. **依赖循环**: 使用调度器的依赖检查功能
4. **状态冲突**: 利用状态管理器的冲突检测

### 调试工具

- 内置统计信息收集
- 详细的日志记录
- 执行历史追踪
- 性能指标监控