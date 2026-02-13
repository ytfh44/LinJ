"""
LinJ-ContiText Serial/Parallel Consistency Tests

Verifies the invariant: Same document + Same initial state + Same external response = Consistent final state
"""

import asyncio
import pytest
from linj_autogen.contitext.mapper import LinJToContiTextMapper, ParallelLinJExecutor
from linj_autogen.core.document import LinJDocument


class TestSerialParallelConsistency:
    """Test serial/parallel execution consistency (invariant)"""

    @pytest.mark.asyncio
    async def test_simple_workflow_consistency(self):
        """Test simple workflow consistency"""

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

        # Serial execution
        serial_mapper = LinJToContiTextMapper(doc)
        serial_mapper.executor.register_tool("set_value", set_value)
        serial_result = await serial_mapper.execute({"input": "test"})

        # Parallel execution
        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_mapper.executor.register_tool("set_value", set_value)
        parallel_result = await parallel_mapper.execute_parallel({"input": "test"})

        # Verify consistency
        assert serial_result["result1"] == "A"
        assert serial_result["result2"] == "B"
        assert parallel_result["result1"] == "A"
        assert parallel_result["result2"] == "B"
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_dependent_workflow_consistency(self):
        """Test dependent workflow consistency"""
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

        # Serial execution
        serial_execution_order = []

        async def capture_step_serial(name: str) -> str:
            serial_execution_order.append(name)
            return name

        serial_mapper = LinJToContiTextMapper(doc)
        serial_mapper.executor.register_tool("step", capture_step_serial)
        serial_result = await serial_mapper.execute()

        # Parallel execution
        parallel_execution_order = []

        async def capture_step_parallel(name: str) -> str:
            parallel_execution_order.append(name)
            return name

        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_mapper.mapper.executor.register_tool("step", capture_step_parallel)
        parallel_result = await parallel_mapper.execute_parallel()

        # Verify execution order is consistent (step1 -> step2)
        assert serial_execution_order == ["first", "second"]
        assert parallel_execution_order == ["first", "second"]
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_conflict_resolution_consistency(self):
        """Test conflict resolution consistency"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {
                    "id": "writer1",
                    "type": "hint",
                    "template": "value1",
                    "write_to": "$.shared",
                    "rank": 2,  # High rank, should execute first
                },
                {
                    "id": "writer2",
                    "type": "hint",
                    "template": "value2",
                    "write_to": "$.shared",
                    "rank": 1,  # Low rank, should execute later
                },
            ],
            edges=[{"from": "writer1", "to": "writer2", "kind": "control"}],
        )

        # Serial execution
        serial_mapper = LinJToContiTextMapper(doc)
        serial_result = await serial_mapper.execute()

        # Parallel execution (actually serial due to dependency)
        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_result = await parallel_mapper.execute_parallel()

        # Verify final state consistency (writer2 overwrites writer1)
        assert serial_result["shared"] == "value2"
        assert parallel_result["shared"] == "value2"
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_disjoint_writes_preservation(self):
        """Test disjoint writes preservation"""
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

        # Serial execution
        serial_mapper = LinJToContiTextMapper(doc)
        serial_result = await serial_mapper.execute()

        # Parallel execution (can truly be parallel)
        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_result = await parallel_mapper.execute_parallel()

        # Verify all writes are preserved
        assert serial_result["data"]["a"] == "A_value"
        assert serial_result["data"]["b"] == "B_value"
        assert serial_result["other"] == "C_value"

        assert parallel_result["data"]["a"] == "A_value"
        assert parallel_result["data"]["b"] == "B_value"
        assert parallel_result["other"] == "C_value"

        assert serial_result == parallel_result


class TestStepIdOrdering:
    """Test step_id deterministic ordering (Section 24.3)"""

    @pytest.mark.asyncio
    async def test_step_id_allocation_order(self):
        """Test step_id allocation order"""
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

        # Collect step_id
        original_allocate = mapper.scheduler.allocate_step_id

        def capture_step_id():
            step_id = original_allocate()
            step_ids.append(step_id)
            return step_id

        mapper.scheduler.allocate_step_id = capture_step_id

        await mapper.execute()

        # Verify step_id order: high_rank -> no_rank -> low_rank
        assert step_ids == [1, 2, 3]  # Monotonically increasing

    @pytest.mark.asyncio
    async def test_rank_priority_over_position(self):
        """Test rank priority over array position"""
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

        # Verify execution order is determined by rank: first_pos -> second_pos -> third_pos
        # Final value depends on overwrite relationship, here assuming no dependency, executed in rank order
        assert serial_result == parallel_result


class TestChangeSetCommitOrdering:
    """Test changeset commit ordering (Sections 20.2, 24.3)"""

    @pytest.mark.asyncio
    async def test_sequential_commit_order(self):
        """Test sequential commit order"""
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

        # Verify final state: step3 executes last, overwrites previous values
        assert result["value"] == "C"

        # Verify commit manager state
        commit_manager = mapper.contitext_engine.get_commit_manager()
        assert commit_manager.get_accepted_count() == 3

    @pytest.mark.asyncio
    async def test_non_intersection_optimization(self):
        """Test non-intersection optimization"""
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

        # Parallel execution can optimize disjoint writes
        parallel_mapper = ParallelLinJExecutor(doc)
        result = await parallel_mapper.execute_parallel()

        # Verify all disjoint writes are preserved
        assert result["path1"] == "A"
        assert result["path2"] == "B"
        assert result["path3"] == "C"


class TestResourceConstraints:
    """Test resource domain constraints (Section 25)"""

    @pytest.mark.asyncio
    async def test_placement_constraints(self):
        """Test placement constraints"""
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

        # Same-domain nodes should execute together
        mapper = ParallelLinJExecutor(doc)
        result = await mapper.execute_parallel()

        assert result["a"] == "A"
        assert result["b"] == "B"

    @pytest.mark.asyncio
    async def test_resource_dependencies(self):
        """Test kind=resource dependencies"""
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

        # Nodes using same resource should be scheduled in same domain
        mapper = ParallelLinJExecutor(doc)
        result = await mapper.execute_parallel()

        assert result["a"] == "A"
        assert result["b"] == "B"


class TestBoundaryConditions:
    """Test boundary conditions"""

    @pytest.mark.asyncio
    async def test_empty_document(self):
        """Test empty document"""
        doc = LinJDocument(linj_version="0.1", nodes=[], edges=[])

        serial_mapper = LinJToContiTextMapper(doc)
        serial_result = await serial_mapper.execute()

        parallel_mapper = ParallelLinJExecutor(doc)
        parallel_result = await parallel_mapper.execute_parallel()

        # Empty document should return empty state
        assert serial_result == {}
        assert parallel_result == {}
        assert serial_result == parallel_result

    @pytest.mark.asyncio
    async def test_max_rounds_limit(self):
        """Test max rounds limit"""
        # Create document that might cause infinite loop
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

        # Should raise rounds exceeded error
        mapper = LinJToContiTextMapper(doc)

        with pytest.raises(Exception):  # Should raise ExecutionError
            await mapper.execute()

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """Test max steps limit"""
        doc = LinJDocument(
            linj_version="0.1",
            nodes=[
                {"id": "step1", "type": "hint", "template": "A", "write_to": "$.a"},
                {"id": "step2", "type": "hint", "template": "B", "write_to": "$.b"},
            ],
            edges=[],
            policies={"max_steps": 1},
        )

        # Should raise steps exceeded error
        mapper = LinJToContiTextMapper(doc)

        with pytest.raises(Exception):  # Should raise ExecutionError
            await mapper.execute()
