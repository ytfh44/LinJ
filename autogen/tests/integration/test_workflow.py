"""
工作流集成测试

测试完整工作流执行
"""

import pytest
from linj_autogen import LinJExecutor, LinJDocument


class TestSimpleWorkflow:
    """测试简单工作流"""
    
    @pytest.mark.asyncio
    async def test_hint_node_workflow(self):
        """测试 hint 节点工作流"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "greeting",
                    "type": "hint",
                    "template": "Hello, {{name}}!",
                    "vars": {"name": {"$path": "$.user_name"}},
                    "write_to": "$.greeting"
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        result = await executor.run(doc, {"user_name": "World"})
        
        assert result["greeting"] == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_tool_node_workflow(self):
        """测试 tool 节点工作流"""
        async def double(x: int) -> int:
            return x * 2
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "calc",
                    "type": "tool",
                    "call": {"name": "double", "args": {"x": {"$path": "$.input"}}},
                    "write_to": "$.result"
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        executor.register_tool("double", double)
        result = await executor.run(doc, {"input": 5})
        
        assert result["result"] == 10
    
    @pytest.mark.asyncio
    async def test_gate_node_workflow(self):
        """测试 gate 节点条件分支"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "check",
                    "type": "gate",
                    "condition": "$.value > 10",
                    "then": ["high"],
                    "else": ["low"]
                },
                {
                    "id": "high",
                    "type": "hint",
                    "template": "High",
                    "write_to": "$.result"
                },
                {
                    "id": "low",
                    "type": "hint",
                    "template": "Low",
                    "write_to": "$.result"
                }
            ],
            edges=[
                {"from": "check", "to": "high", "kind": "control"},
                {"from": "check", "to": "low", "kind": "control"}
            ]
        )
        
        executor = LinJExecutor()
        
        # 测试 then 分支
        result = await executor.run(doc, {"value": 15})
        assert result["result"] == "High"
        
        # 测试 else 分支
        result = await executor.run(doc, {"value": 5})
        assert result["result"] == "Low"


class TestDependencyWorkflow:
    """测试依赖工作流"""
    
    @pytest.mark.asyncio
    async def test_linear_dependency(self):
        """测试线性依赖链"""
        async def add_one(x: int) -> int:
            return x + 1
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "step1",
                    "type": "tool",
                    "call": {"name": "add_one", "args": {"x": {"$path": "$.input"}}},
                    "write_to": "$.step1_result"
                },
                {
                    "id": "step2",
                    "type": "tool",
                    "call": {"name": "add_one", "args": {"x": {"$path": "$.step1_result"}}},
                    "write_to": "$.step2_result"
                },
                {
                    "id": "step3",
                    "type": "tool",
                    "call": {"name": "add_one", "args": {"x": {"$path": "$.step2_result"}}},
                    "write_to": "$.final"
                }
            ],
            edges=[
                {"from": "step1", "to": "step2", "kind": "data"},
                {"from": "step2", "to": "step3", "kind": "data"}
            ]
        )
        
        executor = LinJExecutor()
        executor.register_tool("add_one", add_one)
        result = await executor.run(doc, {"input": 0})
        
        assert result["step1_result"] == 1
        assert result["step2_result"] == 2
        assert result["final"] == 3
    
    @pytest.mark.asyncio
    async def test_join_node_workflow(self):
        """测试 join 节点"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "source",
                    "type": "hint",
                    "template": "test content",
                    "write_to": "$.content"
                },
                {
                    "id": "joiner",
                    "type": "join",
                    "input_from": "$.content",
                    "output_to": "$.joined"
                }
            ],
            edges=[
                {"from": "source", "to": "joiner", "kind": "data"}
            ]
        )
        
        executor = LinJExecutor()
        result = await executor.run(doc, {})
        
        assert result["joined"] == "test content"


class TestComplexWorkflow:
    """测试复杂工作流"""
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """测试并行执行"""
        async def slow_task(name: str, delay: float = 0.01) -> str:
            import asyncio
            await asyncio.sleep(delay)
            return f"{name}_done"
        
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "task_a",
                    "type": "tool",
                    "call": {"name": "slow_task", "args": {"name": "A"}},
                    "write_to": "$.result_a",
                    "reads": ["$.input"],
                    "writes": ["$.result_a"]
                },
                {
                    "id": "task_b",
                    "type": "tool",
                    "call": {"name": "slow_task", "args": {"name": "B"}},
                    "write_to": "$.result_b",
                    "reads": ["$.input"],
                    "writes": ["$.result_b"]
                },
                {
                    "id": "merge",
                    "type": "hint",
                    "template": "{{a}}-{{b}}",
                    "vars": {
                        "a": {"$path": "$.result_a"},
                        "b": {"$path": "$.result_b"}
                    },
                    "write_to": "$.final"
                }
            ],
            edges=[
                {"from": "task_a", "to": "merge", "kind": "data"},
                {"from": "task_b", "to": "merge", "kind": "data"}
            ]
        )
        
        executor = LinJExecutor()
        executor.register_tool("slow_task", slow_task)
        result = await executor.run(doc, {"input": "start"})
        
        assert result["result_a"] == "A_done"
        assert result["result_b"] == "B_done"
        assert result["final"] == "A_done-B_done"
    
    @pytest.mark.asyncio
    async def test_nested_hint(self):
        """测试嵌套 hint"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "outer",
                    "type": "hint",
                    "template": "Outer: {{inner}}",
                    "vars": {"inner": {"$path": "$.inner_value"}},
                    "write_to": "$.outer_result"
                },
                {
                    "id": "inner",
                    "type": "hint",
                    "template": "Inner: {{base}}",
                    "vars": {"base": {"$path": "$.base_value"}},
                    "write_to": "$.inner_value"
                }
            ],
            edges=[
                {"from": "inner", "to": "outer", "kind": "data"}
            ]
        )
        
        executor = LinJExecutor()
        result = await executor.run(doc, {"base_value": "Base"})
        
        assert result["inner_value"] == "Inner: Base"
        assert result["outer_result"] == "Outer: Inner: Base"


class TestErrorHandling:
    """测试错误处理"""
    
    @pytest.mark.asyncio
    async def test_missing_tool(self):
        """测试缺失工具"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "bad",
                    "type": "tool",
                    "call": {"name": "nonexistent", "args": {}},
                    "write_to": "$.result"
                }
            ],
            edges=[]
        )
        
        executor = LinJExecutor()
        
        with pytest.raises(Exception) as exc_info:
            await executor.run(doc, {})
        
        assert "not found" in str(exc_info.value).lower()
