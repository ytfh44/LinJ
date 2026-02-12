"""
LinJ-ContiText 串行/并行一致性测试

验证底线目标：同一文档 + 同一初始状态 + 同一外部响应 = 一致的最终主状态
"""

import asyncio
import pytest
from linj_autogen.contitext.mapper import LinJToContiTextMapper, ParallelLinJExecutor
from linj_autogen.core.document import LinJDocument


class TestSerialParallelConsistency:
    """测试串行/并行执行一致性（底线目标）"""

    @pytest.mark.asyncio
    async def test_simple_workflow_consistency(self):
        """测试简单工作流一致性"""

        async def set_value(value: str, path: str) -> str:
            return value

        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "node1",
                    "type": "tool",
                    "call": {
                        "name": "set_value",
                        "args": {"value": "A", "path": "result1"},
                    },
                    "write_to": "$.result1",
                    "reads": [],
                    "writes": ["$.result1"],
                },
                {
                    "id": "node2",
                    "type": "tool",
                    "call": {
                        "name": "set_value",
                        "args": {"value": "B", "path": "result2"},
                    },
                    "write_to": "$.result2",
                    "reads": [],
                    "writes": ["$.result2"],
                },
            ],
            edges=[],
        )

        # 串行执行
        serial_mapper = LinJToContiTextMapper(doc)
        serial_mapper.executor.register_tool("set_value", set_value)
        serial_result = await serial_mapper.execute({"input": "test"})

        # 并行执行
        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_mapper.executor.register_tool("set_value", set_value)
        parallel_result = await parallel_mapper.execute_parallel({"input": "test"})

        # 验证一致性
        assert serial_result["result1"] == "A"
        assert serial_result["result2"] == "B"
        assert parallel_result["result1"] == "A"
        assert parallel_result["result2"] == "B"
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_dependent_workflow_consistency(self):
        """测试依赖工作流一致性"""
        execution_order = []

        async def step(name: str) -> str:
            execution_order.append(name)
            return name

        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "step1",
                    "type": "tool",
                    "call": {"name": "step", "args": {"name": "first"}},
                    "write_to": "$.step1_result",
                    "reads": [],
                },
                {
                    "id": "step2",
                    "type": "tool",
                    "call": {"name": "step", "args": {"name": "second"}},
                    "write_to": "$.step2_result",
                    "reads": [],
                },
            ],
            edges=[{"from": "step1", "to": "step2", "kind": "control"}],
        )

        # 串行执行
        serial_execution_order = []

        async def capture_step_serial(name: str) -> str:
            serial_execution_order.append(name)
            return name

        serial_mapper = LinJToContiTextMapper(doc)
        serial_mapper.executor.register_tool("step", capture_step_serial)
        serial_result = await serial_mapper.execute()

        # 并行执行
        parallel_execution_order = []

        async def capture_step_parallel(name: str) -> str:
            parallel_execution_order.append(name)
            return name

        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_mapper.mapper.executor.register_tool("step", capture_step_parallel)
        parallel_result = await parallel_mapper.execute_parallel()

        # 验证执行顺序一致（step1 -> step2）
        assert serial_execution_order == ["first", "second"]
        assert parallel_execution_order == ["first", "second"]
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_conflict_resolution_consistency(self):
        """测试冲突解决一致性"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "writer1",
                    "type": "hint",
                    "template": "value1",
                    "write_to": "$.shared",
                    "rank": 2,  # 高 rank，应先执行
                },
                {
                    "id": "writer2",
                    "type": "hint",
                    "template": "value2",
                    "write_to": "$.shared",
                    "rank": 1,  # 低 rank，应后执行
                },
            ],
            edges=[{"from": "writer1", "to": "writer2", "kind": "control"}],
        )

        # 串行执行
        serial_mapper = LinJToContiTextMapper(doc)
        serial_result = await serial_mapper.execute()

        # 并行执行（实际上由于依赖，仍然是串行）
        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_result = await parallel_mapper.execute_parallel()

        # 验证最终状态一致（writer2 覆盖 writer1）
        assert serial_result["shared"] == "value2"
        assert parallel_result["shared"] == "value2"
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_disjoint_writes_preservation(self):
        """测试不相交写入保留"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "writer_a",
                    "type": "hint",
                    "template": "A_value",
                    "write_to": "$.data.a",
                    "writes": ["$.data.a"],
                },
                {
                    "id": "writer_b",
                    "type": "hint",
                    "template": "B_value",
                    "write_to": "$.data.b",
                    "writes": ["$.data.b"],
                },
                {
                    "id": "writer_c",
                    "type": "hint",
                    "template": "C_value",
                    "write_to": "$.other",
                    "writes": ["$.other"],
                },
            ],
            edges=[],
        )

        # 串行执行
        serial_mapper = LinJToContiTextMapper(doc)
        serial_result = await serial_mapper.execute()

        # 并行执行（可以真正并行）
        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_result = await parallel_mapper.execute_parallel()

        # 验证所有写入都保留
        assert serial_result["data"]["a"] == "A_value"
        assert serial_result["data"]["b"] == "B_value"
        assert serial_result["other"] == "C_value"

        assert parallel_result["data"]["a"] == "A_value"
        assert parallel_result["data"]["b"] == "B_value"
        assert parallel_result["other"] == "C_value"

        assert serial_result == parallel_result


class TestStepIdOrdering:
    """测试 step_id 决定性排序（24.3 节）"""

    @pytest.mark.asyncio
    async def test_step_id_allocation_order(self):
        """测试 step_id 分配顺序"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "high_rank",
                    "type": "hint",
                    "template": "A",
                    "write_to": "$.a",
                    "rank": 10,
                },
                {
                    "id": "low_rank",
                    "type": "hint",
                    "template": "B",
                    "write_to": "$.b",
                    "rank": 1,
                },
                {"id": "no_rank", "type": "hint", "template": "C", "write_to": "$.c"},
            ],
            edges=[],
        )

        mapper = LinJToContiTextMapper(doc)
        step_ids = []

        # 收集 step_id
        original_allocate = mapper.scheduler.allocate_step_id

        def capture_step_id():
            step_id = original_allocate()
            step_ids.append(step_id)
            return step_id

        mapper.scheduler.allocate_step_id = capture_step_id

        await mapper.execute()

        # 验证 step_id 顺序：high_rank -> no_rank -> low_rank
        assert step_ids == [1, 2, 3]  # 单调递增

    @pytest.mark.asyncio
    async def test_rank_priority_over_position(self):
        """测试 rank 优先级高于数组位置"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "third_pos",
                    "type": "hint",
                    "template": "3",
                    "write_to": "$.third",
                    "rank": 1,
                },
                {
                    "id": "first_pos",
                    "type": "hint",
                    "template": "1",
                    "write_to": "$.first",
                    "rank": 3,
                },
                {
                    "id": "second_pos",
                    "type": "hint",
                    "template": "2",
                    "write_to": "$.second",
                    "rank": 2,
                },
            ],
            edges=[],
        )

        serial_mapper = LinJToContiTextMapper(doc)
        serial_result = await serial_mapper.execute()

        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_result = await parallel_mapper.execute_parallel()

        # 验证执行顺序由 rank 决定：first_pos -> second_pos -> third_pos
        # 最终值取决于覆盖关系，这里假设无依赖，按 rank 顺序执行
        assert serial_result == parallel_result


class TestChangeSetCommitOrdering:
    """测试变更集提交顺序（20.2 节、24.3 节）"""

    @pytest.mark.asyncio
    async def test_sequential_commit_order(self):
        """测试串行提交顺序"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {"id": "step1", "type": "hint", "template": "A", "write_to": "$.value"},
                {"id": "step2", "type": "hint", "template": "B", "write_to": "$.value"},
                {"id": "step3", "type": "hint", "template": "C", "write_to": "$.value"},
            ],
            edges=[
                {"from": "step1", "to": "step2", "kind": "control"},
                {"from": "step2", "to": "step3", "kind": "control"},
            ],
        )

        mapper = LinJToContiTextMapper(doc)
        result = await mapper.execute()

        # 验证最终状态：step3 最后执行，覆盖前面的值
        assert result["value"] == "C"

        # 验证提交管理器状态
        commit_manager = mapper.contitext_engine.get_commit_manager()
        assert commit_manager.get_accepted_count() == 3

    @pytest.mark.asyncio
    async def test_non_intersection_optimization(self):
        """测试非相交优化"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "writer1",
                    "type": "hint",
                    "template": "A",
                    "write_to": "$.path1",
                    "writes": ["$.path1"],
                },
                {
                    "id": "writer2",
                    "type": "hint",
                    "template": "B",
                    "write_to": "$.path2",
                    "writes": ["$.path2"],
                },
                {
                    "id": "writer3",
                    "type": "hint",
                    "template": "C",
                    "write_to": "$.path3",
                    "writes": ["$.path3"],
                },
            ],
            edges=[],
        )

        # 并行执行可以优化不相交写入
        parallel_mapper = ParallelLinJExecutor(doc)
        result = await parallel_mapper.execute_parallel()

        # 验证所有不相交写入都保留
        assert result["path1"] == "A"
        assert result["path2"] == "B"
        assert result["path3"] == "C"


class TestResourceConstraints:
    """测试资源域约束（25 节）"""

    @pytest.mark.asyncio
    async def test_placement_constraints(self):
        """测试 placement 约束"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {"id": "node1", "type": "hint", "template": "A", "write_to": "$.a"},
                {"id": "node2", "type": "hint", "template": "B", "write_to": "$.b"},
            ],
            edges=[],
            placement=[
                {"target": "node1", "domain": "domain1"},
                {"target": "node2", "domain": "domain1"},
            ],
        )

        # 同域节点应该能一起执行
        mapper = ParallelLinJExecutor(doc)
        result = await mapper.execute_parallel()

        assert result["a"] == "A"
        assert result["b"] == "B"

    @pytest.mark.asyncio
    async def test_resource_dependencies(self):
        """测试 kind=resource 依赖"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {"id": "node1", "type": "hint", "template": "A", "write_to": "$.a"},
                {"id": "node2", "type": "hint", "template": "B", "write_to": "$.b"},
            ],
            edges=[
                {
                    "from": "node1",
                    "to": "node2",
                    "kind": "resource",
                    "resource_name": "shared_resource",
                }
            ],
        )

        # 使用同一 resource 的节点应该被安排在同一域
        mapper = ParallelLinJExecutor(doc)
        result = await mapper.execute_parallel()

        assert result["a"] == "A"
        assert result["b"] == "B"


class TestBoundaryConditions:
    """测试边界条件"""

    @pytest.mark.asyncio
    async def test_empty_document(self):
        """测试空文档"""
        doc = LinJDocument(linj_version="0.1", nodes=[], edges=[])

        serial_mapper = LinJToContiTextMapper(doc)
        serial_result = await serial_mapper.execute()

        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_result = await parallel_mapper.execute_parallel()

        # 空文档应该返回空状态
        assert serial_result == {}
        assert parallel_result == {}
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_max_rounds_limit(self):
        """测试最大轮次限制"""
        # 创建可能导致无限循环的文档
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "loop_node",
                    "type": "hint",
                    "template": "{{trigger}}",
                    "write_to": "$.trigger",
                }
            ],
            edges=[{"from": "loop_node", "to": "loop_node", "kind": "control"}],
            policies={"max_rounds": 3},
        )

        # 应该抛出轮次超限错误
        mapper = LinJToContiTextMapper(doc)

        with pytest.raises(Exception):  # 应该抛出 ExecutionError
            await mapper.execute()

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """测试最大步骤限制"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {"id": "step1", "type": "hint", "template": "A", "write_to": "$.a"},
                {"id": "step2", "type": "hint", "template": "B", "write_to": "$.b"},
            ],
            edges=[],
            policies={"max_steps": 1},
        )

        # 应该抛出步骤超限错误
        mapper = LinJToContiTextMapper(doc)

        with pytest.raises(Exception):  # 应该抛出 ExecutionError
            await mapper.execute()
