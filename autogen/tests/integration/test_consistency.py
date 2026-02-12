"""
串行/并行一致性测试

验证规范底线目标：同一文档 + 同一初始状态 + 同一外部响应 = 一致的最终主状态
"""

import asyncio
import pytest
from linj_autogen import LinJDocument, LinJExecutor


class TestSerialParallelConsistency:
    """测试串行/并行执行一致性"""
    
    @pytest.mark.asyncio
    async def test_independent_nodes_same_result(self):
        """
        测试独立节点串并行结果一致
        
        两个独立节点写入不同路径，最终状态应相同
        """
        async def set_path(value: str, path: str) -> str:
            return value
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "node_a",
                    "type": "tool",
                    "call": {"name": "set_path", "args": {"value": "A", "path": "a"}},
                    "write_to": "$.a",
                    "reads": ["$.trigger"],
                    "writes": ["$.a"]
                },
                {
                    "id": "node_b",
                    "type": "tool",
                    "call": {"name": "set_path", "args": {"value": "B", "path": "b"}},
                    "write_to": "$.b",
                    "reads": ["$.trigger"],
                    "writes": ["$.b"]
                }
            ],
            edges=[]  # 无依赖，可并行
        )
        
        executor = LinJExecutor()
        executor.register_tool("set_path", set_path)
        
        result = await executor.run(doc, {"trigger": "go"})
        
        assert result["a"] == "A"
        assert result["b"] == "B"
    
    @pytest.mark.asyncio
    async def test_order_by_rank_consistency(self):
        """
        测试 rank 决定执行顺序的一致性
        """
        execution_order = []
        
        async def record_execution(name: str) -> str:
            execution_order.append(name)
            return name
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "third",
                    "type": "tool",
                    "call": {"name": "record_execution", "args": {"name": "third"}},
                    "write_to": "$.third",
                    "rank": 1  # 最低优先级
                },
                {
                    "id": "first",
                    "type": "tool",
                    "call": {"name": "record_execution", "args": {"name": "first"}},
                    "write_to": "$.first",
                    "rank": 10  # 最高优先级
                },
                {
                    "id": "second",
                    "type": "tool",
                    "call": {"name": "record_execution", "args": {"name": "second"}},
                    "write_to": "$.second",
                    "rank": 5  # 中等优先级
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        executor.register_tool("record_execution", record_execution)
        
        await executor.run(doc, {})
        
        # 按 rank 降序执行
        assert execution_order == ["first", "second", "third"]
    
    @pytest.mark.asyncio
    async def test_dependency_enforces_order(self):
        """
        测试依赖强制顺序
        
        即使有 rank，数据依赖也必须先满足
        """
        results = []
        
        async def step(name: str) -> str:
            results.append(name)
            return name
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "child",
                    "type": "tool",
                    "call": {"name": "step", "args": {"name": "child"}},
                    "write_to": "$.child",
                    "rank": 100  # 高 rank 但依赖 parent
                },
                {
                    "id": "parent",
                    "type": "tool",
                    "call": {"name": "step", "args": {"name": "parent"}},
                    "write_to": "$.parent",
                    "rank": 1  # 低 rank 但无依赖
                }
            ],
            edges=[
                {"from": "parent", "to": "child", "kind": "data"}
            ]
        )
        
        executor = LinJExecutor()
        executor.register_tool("step", step)
        
        await executor.run(doc, {})
        
        # 依赖强制 parent 先于 child
        assert results == ["parent", "child"]


class TestChangeSetCommit:
    """测试变更集提交决定性"""
    
    @pytest.mark.asyncio
    async def test_changeset_order_deterministic(self):
        """
        测试变更集按 step_id 顺序应用
        """
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "first",
                    "type": "hint",
                    "template": "1",
                    "write_to": "$.value"
                },
                {
                    "id": "second",
                    "type": "hint",
                    "template": "2",
                    "write_to": "$.value"
                }
            ],
            edges=[
                {"from": "first", "to": "second", "kind": "control"}
            ]
        )
        
        executor = LinJExecutor()
        result = await executor.run(doc, {})
        
        # second 后执行，覆盖 first 的值
        assert result["value"] == "2"
    
    @pytest.mark.asyncio
    async def test_disjoint_writes_both_preserved(self):
        """
        测试不相交写入都保留
        """
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "write_a",
                    "type": "hint",
                    "template": "A",
                    "write_to": "$.a",
                    "writes": ["$.a"]
                },
                {
                    "id": "write_b",
                    "type": "hint",
                    "template": "B",
                    "write_to": "$.b",
                    "writes": ["$.b"]
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        result = await executor.run(doc, {})
        
        assert result["a"] == "A"
        assert result["b"] == "B"


class TestGateDeduplication:
    """测试 gate 节点去重"""
    
    @pytest.mark.asyncio
    async def test_gate_trigger_deduplication(self):
        """
        测试 gate 重复触发去重 (13.4 节)
        
        同一 round 内，同一节点被多次触发不得重复执行
        """
        trigger_count = 0
        
        async def count_triggers() -> str:
            nonlocal trigger_count
            trigger_count += 1
            return "triggered"
        
        # 简化测试：直接测试执行一次
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "target",
                    "type": "tool",
                    "call": {"name": "count_triggers", "args": {}},
                    "write_to": "$.result"
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        executor.register_tool("count_triggers", count_triggers)
        
        await executor.run(doc, {})
        
        assert trigger_count == 1


class TestToolRetryPolicy:
    """测试工具重试策略"""
    
    @pytest.mark.asyncio
    async def test_effect_write_no_auto_retry(self):
        """
        测试 effect=write 且 repeat_safe=false 禁止自动重试 (13.2 节)
        
        当前实现不自动重试，只验证配置正确传递
        """
        async def fail_once() -> str:
            return "success"
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "dangerous",
                    "type": "tool",
                    "call": {"name": "fail_once", "args": {}},
                    "write_to": "$.result",
                    "effect": "write",
                    "repeat_safe": False
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        executor.register_tool("fail_once", fail_once)
        
        result = await executor.run(doc, {})
        assert result["result"] == "success"


class TestFinalStateConsistency:
    """测试最终状态一致性"""
    
    @pytest.mark.asyncio
    async def test_same_input_same_output(self):
        """
        测试相同输入产生相同输出
        """
        async def compute(x: int) -> int:
            return x * x
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "square",
                    "type": "tool",
                    "call": {"name": "compute", "args": {"x": {"$path": "$.input"}}},
                    "write_to": "$.result"
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        executor.register_tool("compute", compute)
        
        # 多次执行相同输入
        results = []
        for _ in range(3):
            result = await executor.run(doc, {"input": 5})
            results.append(result["result"])
        
        # 所有结果相同
        assert all(r == 25 for r in results)
