# ContiText Framework

## 概述

ContiText 是 LinJ 规范中定义的续体执行框架，实现第 17-26 节的并行执行语义。本框架提供框架无关的实现，可跨不同的 AI 框架（如 AutoGen、LangGraph 等）使用。

## 核心特性

### 续体管理 (第 17-18 节)
- **续体 (Continuation)**: 表示执行的状态和上下文
- **续体视图 (ContinuationView)**: 为续体提供对主状态对象的受控视图
- **状态管理**: 支持续体的生命周期管理

### 基本操作 (第 19 节)
- **派生 (derive)**: 创建子续体
- **挂起 (suspend)**: 暂停续体执行
- **恢复 (resume)**: 继续被挂起的续体
- **合流 (join)**: 等待多个续体完成
- **取消 (cancel)**: 取消续体执行

### 信号与等待 (第 21 节)
- **信号 (Signal)**: 用于续体间通信的异步消息
- **等待条件 (WaitCondition)**: 定义信号匹配规则
- **信号队列 (SignalQueue)**: 管理信号的发送和等待

### 变更提交 (第 20、24.3 节)
- **变更集提交管理**: 实现决定性提交规则
- **基准规则**: 按 step_id 升序串行接受
- **只读优化**: 空变更集可立即接受
- **非相交优化**: 不相交变更集可提前接受
- **冲突检测**: 自动检测和处理变更冲突

### LinJ 映射 (第 23-26 节)
- **LinJ 到 ContiText 映射**: 将 LinJ 文档映射到续体执行
- **决定性调度**: 确保执行的一致性和可重现性
- **并行执行**: 支持真正的并行执行，同时保证一致性
- **资源域约束**: 支持资源分配和约束验证

## 架构设计

### 框架无关设计
- 使用 Protocol 定义抽象接口，支持不同的实现
- 不依赖特定框架的数据结构
- 提供默认实现，支持快速集成

### 核心组件

```
ContiText Framework
├── Continuation          # 续体核心
├── ContinuationView      # 续体视图
├── ContiTextEngine      # 执行引擎
├── Signal & WaitCondition # 信号机制
├── CommitManager        # 变更提交管理
└── LinJToContiTextMapper # LinJ 映射器
```

## 使用示例

### 基本续体操作

```python
from contitext import ContiTextEngine, Continuation

# 创建引擎
engine = ContiTextEngine()

# 派生根续体
root = engine.derive()

# 派生子续体
child = engine.derive(root)

# 挂起续体
engine.suspend(child, wait_condition=some_condition)

# 恢复续体
engine.resume(child.handle, input_data={"key": "value"})

# 等待完成
results = await engine.join([child.handle])
```

### LinJ 文档执行

```python
from contitext import LinJToContiTextMapper

# 创建映射器
mapper = LinJToContiTextMapper(document)

# 执行文档
final_state = await mapper.execute(initial_state={"counter": 0})

# 并行执行
final_state = await mapper.execute_parallel(initial_state={"data": []})
```

### 变更集提交

```python
# 创建变更集
changeset = MyChangeSet(...)

# 提交变更集
result = engine.submit_changeset(
    step_id=1,
    changeset=changeset,
    handle=continuation.handle
)

if result.success:
    print(f"Changeset committed at revision {result.new_revision}")
else:
    print(f"Commit failed: {result.error}")
```

## 协议接口

### StateManager 协议
```python
class StateManager(Protocol):
    def get_full_state(self) -> Dict[str, Any]: ...
    def get_revision(self) -> int: ...
    def apply(self, changeset: Any, step_id: Optional[int] = None) -> None: ...
```

### ChangeSet 协议
```python
class ChangeSet(Protocol):
    def is_empty(self) -> bool: ...
    def intersects_with(self, other: Any) -> bool: ...
    def apply_to_state(self, state: Dict[str, Any]) -> Dict[str, Any]: ...
```

### LinJDocument 协议
```python
class LinJDocument(Protocol):
    @property
    def linj_version(self) -> str: ...
    
    @property
    def nodes(self) -> List[Any]: ...
    
    @property
    def policies(self) -> Optional[Any]: ...
```

## 集成指南

### 与 AutoGen 集成

```python
# 实现 AutoGen 特定的状态管理器
class AutoGenStateManager:
    def __init__(self, agent_group):
        self.agent_group = agent_group
    
    def get_full_state(self):
        return self.agent_group.get_shared_state()
    
    def get_revision(self):
        return self.agent_group.get_revision()
    
    def apply(self, changeset, step_id=None):
        self.agent_group.apply_changeset(changeset, step_id)

# 使用 AutoGen 状态管理器
state_manager = AutoGenStateManager(agent_group)
engine = ContiTextEngine(state_manager)
```

### 与 LangGraph 集成

```python
# 实现 LangGraph 特定的变更集
class LangGraphChangeSet:
    def __init__(self, node_updates):
        self.node_updates = node_updates
    
    def is_empty(self):
        return not self.node_updates
    
    def intersects_with(self, other):
        return bool(set(self.node_updates.keys()) & set(other.node_updates.keys()))
    
    def apply_to_state(self, state):
        new_state = state.copy()
        new_state.update(self.node_updates)
        return new_state
```

## 执行保证

### 决定性执行
- 同一文档 + 同一初始状态 = 一致的最终主状态
- step_id 决定性分配确保执行顺序的可重现性
- 变更集提交的串行化保证状态一致性

### 并行安全
- 非相交优化允许安全的并行执行
- 冲突检测防止并发修改导致的数据竞争
- 续体隔离确保执行的原子性

### 容错机制
- 续体取消的幂等性
- 变更集冲突的自动检测和处理
- 执行状态的完整跟踪

## 性能优化

### 只读优化
- 空变更集立即接受，无需串行化
- 读取操作不产生状态冲突

### 非相交优化
- 不相交的写入操作可以并行执行
- 提前接受非相交变更集，减少等待时间

### 批量处理
- 支持批量处理待提交的变更集
- 减少状态管理的开销

## 错误处理

### 冲突错误
```python
from contitext import ConflictError

try:
    result = engine.submit_changeset(step_id, changeset, handle)
except ConflictError as e:
    print(f"Conflict: {e.message}")
    print(f"Details: {e.details}")
```

### 续体错误
```python
try:
    engine.resume(handle)
except ValueError as e:
    print(f"Cannot resume continuation: {e}")
```

## 扩展性

### 自定义信号评估器
```python
def custom_evaluator(predicate: str, state: Dict[str, Any]) -> bool:
    # 实现自定义的条件评估逻辑
    return evaluate_with_custom_engine(predicate, state)

wait_condition = WaitCondition(
    predicate="$.signal.payload.value > 10",
    evaluator=custom_evaluator
)
```

### 自定义调度器
```python
class CustomScheduler:
    def select_from_ready(self, ready_nodes):
        # 实现自定义的节点选择逻辑
        return self.select_by_priority(ready_nodes)
    
    def allocate_step_id(self):
        # 实现自定义的 step_id 分配逻辑
        return self.next_step_id()
```

## 版本兼容性

- 支持 LinJ 规范版本 1.0+
- 向后兼容的 API 设计
- 框架无关的抽象接口

## 许可证

本框架遵循项目整体许可证。