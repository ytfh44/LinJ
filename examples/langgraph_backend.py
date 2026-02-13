"""
LangGraph Backend Adapter for Examples

Complete LangGraph backend implementation aligned with LinJ.md specification:
- Deterministic scheduling (11.3)
- State paths (5)
- ChangeSet atomicity (9.2)
- Node types (13): hint/tool/join/gate
- Condition expressions (14): exists/len/value
- Dependency graph (8)
- Loops (12)
- Thread safety (11.5)

Provides fully compatible execution interface with AutoGen version.
"""

import asyncio
import re
import sys
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple, Union
from dataclasses import dataclass, field

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure autogen/src is in path
autogen_src = project_root / "autogen" / "src"
if str(autogen_src) not in sys.path:
    sys.path.insert(0, str(autogen_src))


# ============================================================================
# Path Resolver (aligned with LinJ.md 5.1-5.4)
# ============================================================================


class PathResolver:
    """State path resolver - implements Sections 5.1-5.4 specification"""

    @staticmethod
    def _parse_path_parts(path: str) -> List[str]:
        """Parse path into list of parts, supporting array indices"""
        if not path.startswith("$."):
            path = f"$.{path}"

        # Remove $.
        path = path[2:]

        parts = []
        current = ""
        in_bracket = False

        for char in path:
            if char == "[":
                in_bracket = True
                if current:
                    parts.append(current)
                    current = ""
            elif char == "]":
                in_bracket = False
                if current:
                    parts.append(current)
                    current = ""
            elif char == ".":
                if not in_bracket:
                    if current:
                        parts.append(current)
                        current = ""
            else:
                current += char

        if current:
            parts.append(current)

        return parts

    @staticmethod
    def get(state: Dict[str, Any], path: str) -> Any:
        """Read path value (5.2) - returns null if not exists"""
        parts = PathResolver._parse_path_parts(path)
        current: Any = state

        # Handle root path $
        if len(parts) == 0:
            return state

        for part in parts:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return None
                except ValueError:
                    return None
            else:
                return None

        return current

    @staticmethod
    def set(state: Dict[str, Any], path: str, value: Any) -> None:
        """Write path value (5.3)"""
        parts = PathResolver._parse_path_parts(path)
        current: Any = state

        # Traverse to second-to-last part
        for i, part in enumerate(parts[:-1]):
            if part not in current or current[part] is None:
                # Determine whether to create object or array based on next part
                next_part = parts[i + 1]
                try:
                    idx = int(next_part)
                    current[part] = []
                    while len(current[part]) <= idx:
                        current[part].append(None)
                except ValueError:
                    current[part] = {}

            current = current[part]

        # Set the last part
        last_part = parts[-1]
        if isinstance(current, list):
            idx = int(last_part)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[last_part] = value

    @staticmethod
    def delete(state: Dict[str, Any], path: str) -> None:
        """Delete path value (5.4) - set to null"""
        parts = PathResolver._parse_path_parts(path)

        # Find parent of target path
        current: Any = state
        target_path = parts[-1]

        for i, part in enumerate(parts[:-1]):
            if current is None:
                return
            if isinstance(current, dict):
                if part not in current:
                    return
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return
                except ValueError:
                    return
            else:
                return

        # Set to null
        if isinstance(current, dict):
            if target_path in current:
                current[target_path] = None
        elif isinstance(current, list):
            try:
                idx = int(target_path)
                if 0 <= idx < len(current):
                    current[idx] = None
            except ValueError:
                pass

    def delete(state: Dict[str, Any], path: str) -> None:
        """Delete path value (5.4) - set to null"""
        if not path.startswith("$."):
            path = f"$.{path}"

        parts = path[2:].split(".")
        current = state
        target_path = parts[-1]

        for i, part in enumerate(parts[:-1]):
            if current is None:
                return
            if isinstance(current, dict):
                if part not in current:
                    return
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return
                except ValueError:
                    return
            else:
                return

        # Set to null (not delete key, because arrays cannot be shortened)
        if isinstance(current, dict):
            if target_path in current:
                current[target_path] = None
        elif isinstance(current, list):
            try:
                idx = int(target_path)
                if 0 <= idx < len(current):
                    current[idx] = None
            except ValueError:
                pass

    @staticmethod
    def intersect(path_a: str, path_b: str) -> bool:
        """Check if two paths intersect (11.4)"""
        # Prefix check
        if path_a.startswith(path_b) or path_b.startswith(path_a):
            return True
        if path_a == path_b:
            return True

        # Different array indices don't intersect
        try:
            # Extract path parts and array indices
            parts_a = path_a.replace("$.", "").split(".")
            parts_b = path_b.replace("$.", "").split(".")

            # Check if array indices are different
            min_len = min(len(parts_a), len(parts_b))
            for i in range(min_len):
                try:
                    idx_a = int(parts_a[i])
                    idx_b = int(parts_b[i])
                    if idx_a != idx_b:
                        # Different indices, if previous parts match, don't intersect
                        prefix_match = True
                        for j in range(i):
                            if parts_a[j] != parts_b[j]:
                                prefix_match = False
                                break
                        if prefix_match:
                            return False
                except ValueError:
                    pass

        except Exception:
            pass

        return True


# ============================================================================
# ChangeSet (aligned with LinJ.md 9)
# ============================================================================


@dataclass
class WriteOp:
    """Write operation"""

    path: str
    value: Any


@dataclass
class DeleteOp:
    """Delete operation"""

    path: str


@dataclass
class ChangeSet:
    """
    ChangeSet (9.1)

    - writes: [{path, value}]
    - deletes: [{path}]
    """

    writes: List[WriteOp] = field(default_factory=list)
    deletes: List[DeleteOp] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.writes) == 0 and len(self.deletes) == 0

    def get_write_paths(self) -> Set[str]:
        return {op.path for op in self.writes}

    def get_delete_paths(self) -> Set[str]:
        return {op.path for op in self.deletes}

    def get_all_modified_paths(self) -> Set[str]:
        return self.get_write_paths() | self.get_delete_paths()

    def intersects_with(self, other: "ChangeSet") -> bool:
        """Check path intersection (11.4)"""
        for path_a in self.get_all_modified_paths():
            for path_b in other.get_all_modified_paths():
                if PathResolver.intersect(path_a, path_b):
                    return True
        return False

    def apply_to(self, state: Dict[str, Any]) -> None:
        """Apply ChangeSet (9.2 atomicity guaranteed by caller)"""
        # Delete first
        for op in self.deletes:
            PathResolver.delete(state, op.path)
        # Write last
        for op in self.writes:
            PathResolver.set(state, op.path, op.value)


# ============================================================================
# Condition Expression Evaluator (aligned with LinJ.md 14)
# ============================================================================


class ConditionEvaluator:
    """Condition expression evaluator (14)"""

    # Comparison operators
    COMP_OPS = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b if a is not None and b is not None else False,
        ">=": lambda a, b: a >= b if a is not None and b is not None else False,
        "<": lambda a, b: a < b if a is not None and b is not None else False,
        "<=": lambda a, b: a <= b if a is not None and b is not None else False,
    }

    # Logical operators
    LOGIC_OPS = {
        "AND": lambda a, b: a and b,
        "OR": lambda a, b: a or b,
    }

    # Path functions
    PATH_FUNCS = {
        "exists": lambda path: path is not None,
        "len": lambda path: len(path) if isinstance(path, list) else 0,
        "value": lambda path: path,
    }

    @classmethod
    def evaluate(cls, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate condition expression"""
        if not condition or not condition.strip():
            return True

        # Preprocess: expand function calls
        expr = condition

        # Expand exists(path)
        def replace_exists(match):
            path = match.group(1).strip()
            value = PathResolver.get(state, path)
            return "True" if value is not None else "False"

        expr = re.sub(r"exists\(\s*([^)]+)\s*\)", replace_exists, expr)

        # Expand len(path)
        def replace_len(match):
            path = match.group(1).strip()
            value = PathResolver.get(state, path)
            length = len(value) if isinstance(value, list) else 0
            return str(length)

        expr = re.sub(r"len\(\s*([^)]+)\s*\)", replace_len, expr)

        # Expand value(path)
        def replace_value(match):
            path = match.group(1).strip()
            value = PathResolver.get(state, path)
            if value is None:
                return "None"
            if isinstance(value, str):
                return f'"{value}"'
            return str(value)

        expr = re.sub(r"value\(\s*([^)]+)\s*\)", replace_value, expr)

        # Safely evaluate expression
        try:
            # Replace operators with Python syntax
            expr = expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not")

            # Add safe environment
            safe_env = {
                "true": True,
                "false": False,
                "null": None,
                "True": True,
                "False": False,
                "None": None,
            }

            result = eval(expr, {"__builtins__": {}}, safe_env)
            return bool(result)

        except Exception as e:
            logging.warning(f"Condition evaluation error: {e}, condition: {condition}")
            return False


# ============================================================================
# Node Executor (aligned with LinJ.md 13)
# ============================================================================


class NodeExecutor:
    """Node executor - implements Section 13 node type semantics"""

    def __init__(self, tools: Dict[str, Callable], state: Dict[str, Any]):
        self.tools = tools
        self.state = state

    def can_retry(self, node: Dict[str, Any], attempt: int) -> bool:
        """
        Check if tool call can be retried at current attempt count (Section 13.2)

        Rules:
        - If effect is write and repeat_safe is false: even if globally allowed, executor must not auto-retry
        - If effect is none or read: can retry according to policy

        Args:
            node: Tool node configuration
            attempt: Current attempt count (0-based)

        Returns:
            Whether retry is allowed
        """
        effect = node.get("effect", "read")
        repeat_safe = node.get("repeat_safe", False)

        # If effect is write and repeat_safe is false: forbid retry
        if effect == "write" and not repeat_safe:
            return False

        # First attempt doesn't need retry
        if attempt == 0:
            return True

        # effect is none or read: allow retry (but max retry count needs to be controlled at call site)
        return effect in ("none", "read")

    def execute_tool(self, node: Dict[str, Any], retry_count: int = 0) -> ChangeSet:
        """
        Execute tool node (13.2)

        Args:
            node: Tool node configuration
            retry_count: Current retry count (used to determine if retry is allowed)
        """
        call = node.get("call", {})
        name = call.get("name")
        args_dict = call.get("args", {})
        write_to = node.get("write_to")
        effect = node.get("effect", "read")

        # Parse arguments
        args = {}
        for key, value in args_dict.items():
            if isinstance(value, dict) and "$path" in value:
                value = PathResolver.get(self.state, value["$path"])
            elif isinstance(value, str) and value.startswith("$."):
                value = PathResolver.get(self.state, value)
            args[key] = value

        # Call tool
        result = None
        error = None
        if name in self.tools:
            try:
                tool_func = self.tools[name]
                if asyncio.iscoroutinefunction(tool_func):
                    result = asyncio.run(tool_func(**args))
                else:
                    result = tool_func(**args)
            except Exception as e:
                error = str(e)
                logging.error(f"Tool {name} execution failed: {e}")
        else:
            error = f"Tool {name} not found"
            logging.warning(f"Tool {name} not registered")

        # Handle errors and retries
        if error:
            # Check if retry is allowed (Section 13.2 rules)
            if self.can_retry(node, retry_count):
                # Retry count limit (configurable in policy)
                max_retries = 3
                if retry_count < max_retries:
                    # Simple exponential backoff
                    import time

                    time.sleep(0.1 * (2**retry_count))
                    return self.execute_tool(node, retry_count + 1)

            # Not retryable or max retries reached, return error result
            result = f"Error: {error}"

        # Apply changes
        if write_to:
            return ChangeSet(writes=[WriteOp(path=write_to, value=result)])
        return ChangeSet()

    def execute_join(self, node: Dict[str, Any]) -> Tuple[ChangeSet, Optional[str]]:
        """Execute join node (13.3)"""
        input_from = node.get("input_from")
        output_to = node.get("output_to")
        glossary = node.get("glossary")

        # Read input
        text = PathResolver.get(self.state, input_from) or ""
        text_str = str(text)

        # Validate forbidden items
        forbidden = None
        if glossary:
            for item in glossary:
                if item.get("forbid"):
                    for word in item["forbid"]:
                        if word in text_str:
                            forbidden = word
                            break

        # Output
        if output_to:
            cs = ChangeSet(writes=[WriteOp(path=output_to, value=text)])
        else:
            cs = ChangeSet()

        return cs, forbidden

    def execute_gate(self, node: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Execute gate node (13.4)"""
        condition = node.get("condition", "")
        then = node.get("then", [])
        else_ = node.get("else", [])

        # Evaluate condition
        result = ConditionEvaluator.evaluate(condition, self.state)

        return result, then, else_

    def execute(self, node: Dict[str, Any]) -> Tuple[ChangeSet, Optional[str], Any]:
        """
        Execute single node

        Returns:
            - changeset: ChangeSet
            - forbidden: Forbidden item (join only)
            - gate_result: Gate result (gate only)
        """
        node_type = node.get("type")

        if node_type == "hint":
            return self.execute_hint(node), None, None
        elif node_type == "tool":
            return self.execute_tool(node), None, None
        elif node_type == "join":
            cs, forbidden = self.execute_join(node)
            return cs, forbidden, None
        elif node_type == "gate":
            gate_result, then, else_ = self.execute_gate(node)
            return ChangeSet(), None, (gate_result, then, else_)
        else:
            logging.warning(f"Unknown node type: {node_type}")
            return ChangeSet(), None, None


# ============================================================================
# Dependency Graph (aligned with LinJ.md 8)
# ============================================================================


class DependencyGraph:
    """Dependency graph management"""

    def __init__(self, edges: List[Dict[str, Any]]):
        self.edges = edges
        self.adjacency: Dict[str, List[str]] = {}  # node -> [dependents]
        self.reverse_adj: Dict[str, List[str]] = {}  # node -> [dependencies]
        self.edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self._build_graph()

    def _build_graph(self) -> None:
        """Build dependency graph"""
        for edge in self.edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            kind = edge.get("kind", "data")

            # Add edge
            self.edge_map[(from_node, to_node)] = edge

            # Update adjacency list
            if from_node not in self.adjacency:
                self.adjacency[from_node] = []
            if to_node not in self.reverse_adj:
                self.reverse_adj[to_node] = []

            self.adjacency[from_node].append(to_node)
            self.reverse_adj[to_node].append(from_node)

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get prerequisites for node"""
        return self.reverse_adj.get(node_id, [])

    def get_dependents(self, node_id: str) -> List[str]:
        """Get nodes that depend on this node"""
        return self.adjacency.get(node_id, [])

    def get_incoming_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all edges entering this node"""
        result = []
        for edge in self.edges:
            if edge.get("to") == node_id:
                result.append(edge)
        return result


# ============================================================================
# Deterministic Scheduler (aligned with LinJ.md 11.3)
# ============================================================================


class DeterministicScheduler:
    """Deterministic scheduler - 11.3"""

    def __init__(self, nodes: List[Dict[str, Any]], dependency_graph: DependencyGraph):
        self.nodes = nodes
        self.dependency_graph = dependency_graph
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.executed_steps: List[Dict[str, Any]] = []

    def are_dependencies_satisfied(self, node_id: str, completed: Set[str]) -> bool:
        """Check if dependencies are satisfied"""
        deps = self.dependency_graph.get_dependencies(node_id)
        return all(dep in completed for dep in deps)

    def get_ready_nodes(self, completed: Set[str]) -> List[Dict[str, Any]]:
        """Get all ready nodes"""
        ready = []
        for node in self.nodes:
            node_id = node.get("id")
            if node_id in completed or node_id in self.failed:
                continue
            if self.are_dependencies_satisfied(node_id, completed):
                ready.append(node)
        return ready

    def select_node(
        self, ready_nodes: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select next node to execute

        Deterministic order (11.3):
        rank first ( 1. Highertreat missing as 0)
        2. Earlier in nodes array first
        3. If still same, sort by node_id lexicographically
        """
        if not ready_nodes:
            return None

        # Sort
        ready_nodes.sort(
            key=lambda n: (
                n.get("rank", 0) or 0,  # rank
                self.nodes.index(n),  # nodes array order
                n.get("id", ""),  # node_id lexicographic order
            ),
            reverse=True,
        )

        return ready_nodes[0]

    def can_execute_concurrently(
        self, node_a: Dict[str, Any], node_b: Dict[str, Any]
    ) -> bool:
        """
        Check if two nodes can execute concurrently (11.5)

        Requirements:
        - Write sets must be disjoint
        - Read sets must not intersect with other's write sets
        """
        reads_a = set(node_a.get("reads", []) or [])
        writes_a = set(node_a.get("writes", []) or [])
        reads_b = set(node_b.get("reads", []) or [])
        writes_b = set(node_b.get("writes", []) or [])

        # Write sets must be disjoint
        for path_a in writes_a:
            for path_b in writes_b:
                if PathResolver.intersect(path_a, path_b):
                    return False

        # Reads must not intersect with other's writes
        for path_r in reads_a:
            for path_w in writes_b:
                if PathResolver.intersect(path_r, path_w):
                    return False

        for path_r in reads_b:
            for path_w in writes_a:
                if PathResolver.intersect(path_r, path_w):
                    return False

        return True


# ============================================================================
# Signal System (aligned with LinJ.md 21)
# ============================================================================


@dataclass
class Signal:
    """Signal structure (21.1)"""

    name: str
    payload: Any = None
    correlation: Optional[str] = None


@dataclass
class WaitCondition:
    """Wait condition (21.2)"""

    name: Optional[str] = None
    correlation: Optional[str] = None
    predicate: Optional[str] = None

    def matches(self, signal: Signal, state: Dict[str, Any]) -> bool:
        """Check if signal matches wait condition"""
        if self.name and signal.name != self.name:
            return False
        if self.correlation and signal.correlation != self.correlation:
            return False
        if self.predicate:
            test_state = {**state, "$.signal.payload": signal.payload}
            if not ConditionEvaluator.evaluate(self.predicate, test_state):
                return False
        return True


class SignalQueue:
    """Signal queue"""

    def __init__(self):
        self._signals: List[Signal] = []
        self._waiters: Dict[str, WaitCondition] = {}

    def send(self, signal: Signal) -> None:
        self._signals.append(signal)

    def register_waiter(self, handle: str, condition: WaitCondition) -> None:
        self._waiters[handle] = condition

    def unregister_waiter(self, handle: str) -> None:
        self._waiters.pop(handle, None)

    def check_waiter(self, handle: str, state: Dict[str, Any]) -> Optional[Signal]:
        if handle not in self._waiters:
            return None
        condition = self._waiters[handle]
        for signal in self._signals:
            if condition.matches(signal, state):
                return signal
        return None


# ============================================================================
# LangGraph Backend Implementation (complete alignment)
# ============================================================================


class BaseBackend:
    """Backend base class"""

    def __init__(
        self, enable_tracing: bool = True, config: Optional[Dict[str, Any]] = None
    ):
        self._enable_tracing = enable_tracing
        self._config = config or {}
        self._tools: Dict[str, Callable] = {}
        self._tracer: Optional[logging.Logger] = None

        if enable_tracing:
            self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging"""
        self._tracer = logging.getLogger("linj.langgraph")
        self._tracer.setLevel(logging.INFO)
        if not self._tracer.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self._tracer.addHandler(handler)

    def register_tool(self, name: str, func: Callable) -> None:
        """Register tool function"""
        self._tools[name] = func

    def register_tools(self, tools: Dict[str, Callable]) -> None:
        """Batch register tools"""
        self._tools.update(tools)

    async def run(
        self,
        document: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute workflow (subclass must implement)"""
        raise NotImplementedError

    def run_sync(
        self,
        document: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronous execution"""
        return asyncio.run(self.run(document, initial_state, **kwargs))


class AutoGenBackend(BaseBackend):
    """
    AutoGen backend - uses linj_autogen.executor.runner.LinJExecutor
    """

    async def run(
        self,
        document: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from linj_autogen.executor.runner import LinJExecutor

        executor = LinJExecutor(enable_tracing=self._enable_tracing)

        for name, func in self._tools.items():
            executor.register_tool(name, func)

        return await executor.run(document, initial_state or {}, **kwargs)


class LangGraphBackend(BaseBackend):
    """
    LangGraph backend - complete alignment with LinJ.md specification

    Key implementation points:
    - Deterministic scheduling (11.3)
    - State paths (5)
    - ChangeSet atomicity (9.2)
    - Node types (13)
    - Condition expressions (14)
    - Loop handling (12)
    - Thread safety (11.5)
    """

    def __init__(
        self, enable_tracing: bool = True, config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(enable_tracing, config)
        self._tracer = logging.getLogger("linj.langgraph.backend")

    async def run(
        self,
        document: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute workflow

        Args:
            document: LinJ document
            initial_state: Initial state

        Returns:
            Final state
        """
        return await self._execute(document, initial_state or {})

    def _is_loop_entry_node(
        self,
        node_id: str,
        loop_states: Dict[str, Any],
        dependency_graph: DependencyGraph,
    ) -> bool:
        """
        Check if node is loop entry node

        Loop entry node: has edges pointing to loop members, but node itself is not in loop
        For entry nodes, use OR semantics to check dependencies (only one dependency needs to be satisfied)
        """
        for loop_id, loop_state in loop_states.items():
            # Check if there are edges from outside the loop to loop members
            for member_id in loop_state["members"]:
                incoming = dependency_graph.get_incoming_edges(member_id)
                for edge in incoming:
                    from_node = (
                        edge.get("from") if isinstance(edge, dict) else edge.from_
                    )
                    if from_node == node_id:
                        # Check if this node is in the loop
                        if node_id not in loop_state["members"]:
                            return True
        return False

    def _check_dependencies_satisfied(
        self,
        node_id: str,
        completed: Set[str],
        activated: Set[str],
        loop_states: Dict[str, Any],
        dependency_graph: DependencyGraph,
    ) -> bool:
        """
        Check if node dependencies are satisfied

        For loop entry nodes: use OR semantics (only one dependency needs to be satisfied)
        For normal nodes: use AND semantics (all dependencies must be satisfied)
        """
        deps = dependency_graph.get_dependencies(node_id)
        if not deps:
            return True

        # Check if it's a loop entry node
        is_loop_entry = self._is_loop_entry_node(node_id, loop_states, dependency_graph)

        if is_loop_entry:
            # Loop entry node: OR semantics
            return any(dep in completed or dep in activated for dep in deps)
        else:
            # Normal node: AND semantics
            return all(dep in completed or dep in activated for dep in deps)

        max_steps = policies.get("max_steps", 1000)
        max_rounds = policies.get("max_rounds", 100)

        # 3. Build dependency graph
        dependency_graph = DependencyGraph(edges)

        # 4. Create scheduler
        scheduler = DeterministicScheduler(nodes, dependency_graph)

        # 5. Initialize tracing and conflict detection
        trace = []
        step_id = 0
        current_round = 0
        activated = set()  # Activated nodes
        completed = set()  # Completed nodes
        failed = set()  # Failed nodes
        applied_changesets: List[ChangeSet] = []  # Applied ChangeSets

        # Initialize loop states
        loop_states = {}
        for loop in loops:
            loop_states[loop["id"]] = {
                "entry": loop["entry"],
                "members": set(loop["members"]),
                "max_rounds": loop.get("max_rounds", max_rounds),
                "current_round": 0,
                "executed": set(),
            }

        self._tracer.info(
            f"Starting workflow execution: {len(nodes)} nodes, {len(edges)} edges"
        )

        # 6. Main execution loop
        while current_round < max_rounds and step_id < max_steps:
            round_ready = set()

            # 6.1 Collect ready nodes
            for node in nodes:
                node_id = node.get("id")
                if node_id in completed or node_id in failed or node_id in activated:
                    continue

                # Check dependencies (support OR semantics for loop entry nodes)
                if not self._check_dependencies_satisfied(
                    node_id, completed, activated, loop_states, dependency_graph
                ):
                    continue

                # Check loop member limits
                in_loop = False
                for loop_id, loop_state in loop_states.items():
                    if node_id in loop_state["members"]:
                        in_loop = True
                        if node_id in loop_state["executed"]:
                            # Don't execute again in same round
                            continue
                        # Inside loop, check round limit
                        if loop_state["current_round"] >= loop_state["max_rounds"]:
                            continue

                round_ready.add(node)

            if not round_ready:
                # No ready nodes, check if we can advance
                can_advance = False
                for loop_id, loop_state in loop_states.items():
                    if loop_state["executed"]:
                        # Loop advances
                        loop_state["current_round"] += 1
                        loop_state["executed"].clear()
                        can_advance = True

                if not can_advance:
                    # Workflow complete
                    break
                else:
                    current_round += 1
                    continue

            # 6.2 Deterministic node selection (11.3)
            ready_list = list(round_ready)
            ready_list.sort(
                key=lambda n: (n.get("rank", 0) or 0, nodes.index(n), n.get("id", "")),
                reverse=True,
            )

            # 6.3 Sequential execution (guarantees determinism)
            for node in ready_list:
                if step_id >= max_steps:
                    break

                node_id = node.get("id")
                node_type = node.get("type")

                self._tracer.info(
                    f"Executing node: {node_id} ({node_type}), step_id={step_id}"
                )

                try:
                    # Execute node
                    executor = NodeExecutor(self._tools, state)
                    changeset, forbidden, gate_result = executor.execute(node)

                    # Check join forbidden items
                    if forbidden:
                        raise ValueError(f"Forbidden term found: {forbidden}")

                    # Apply ChangeSet (9.2 atomicity)
                    if not changeset.is_empty():
                        # Conflict detection (22.1: rejection policy)
                        conflict = False
                        for prev_cs in applied_changesets:
                            if changeset.intersects_with(prev_cs):
                                conflict = True
                                break

                        if conflict:
                            # Conflict produces error
                            raise ValueError(
                                f"ConflictError: changeset for node {node_id} at step {step_id} "
                                f"intersects with previously applied changesets"
                            )

                        # No conflict, apply ChangeSet
                        changeset.apply_to(state)
                        applied_changesets.append(changeset)

                    # Record trace (27)
                    trace.append(
                        {
                            "step_id": step_id,
                            "round": current_round,
                            "node_id": node_id,
                            "status": "completed",
                            "writes": list(changeset.get_write_paths()),
                        }
                    )

                    # Update loop states
                    for loop_id, loop_state in loop_states.items():
                        if node_id in loop_state["members"]:
                            loop_state["executed"].add(node_id)

                    activated.add(node_id)
                    step_id += 1

                    # Handle gate activation
                    if node_type == "gate" and gate_result:
                        condition_met, then_nodes, else_nodes = gate_result
                        target_nodes = then_nodes if condition_met else else_nodes
                        for target_id in target_nodes:
                            if target_id not in completed:
                                activated.add(target_id)

                except Exception as e:
                    self._tracer.error(f"Node {node_id} failed: {e}")
                    trace.append(
                        {
                            "step_id": step_id,
                            "round": current_round,
                            "node_id": node_id,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    failed.add(node_id)
                    step_id += 1

            # 6.4 Complete this round's activated nodes
            completed.update(activated)
            activated.clear()

            current_round += 1

        # 7. Add trace information to state
        state["$.trace"] = trace
        state["$.execution_stats"] = {
            "total_steps": step_id,
            "total_rounds": current_round,
            "completed_nodes": len(completed),
            "failed_nodes": len(failed),
        }

        self._tracer.info(
            f"Workflow completed: {step_id} steps, {current_round} rounds, "
            f"{len(completed)} completed, {len(failed)} failed"
        )

        return state


def create_backend(
    backend_type: str = "autogen",
    enable_tracing: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> BaseBackend:
    """
    Create backend executor of specified type

    Args:
        backend_type: "autogen" or "langgraph"
        enable_tracing: Whether to enable tracing
        config: Optional configuration

    Returns:
        Backend executor instance
    """
    backend_type = backend_type.lower()

    if backend_type == "langgraph":
        return LangGraphBackend(enable_tracing=enable_tracing, config=config)
    else:
        return AutoGenBackend(enable_tracing=enable_tracing, config=config)


def load_document(path: str) -> Dict[str, Any]:
    """
    Load YAML format LinJ document

    Args:
        path: YAML file path

    Returns:
        Parsed document object
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def run_workflow(
    yaml_path: str,
    tools: Dict[str, Callable],
    initial_state: Dict[str, Any],
    backend_type: str = "autogen",
    enable_tracing: bool = True,
) -> Dict[str, Any]:
    """Convenient workflow execution function"""
    doc = load_document(yaml_path)

    backend = create_backend(backend_type=backend_type, enable_tracing=enable_tracing)
    backend.register_tools(tools)

    return await backend.run(doc, initial_state)


def run_workflow_sync(
    yaml_path: str,
    tools: Dict[str, Callable],
    initial_state: Dict[str, Any],
    backend_type: str = "autogen",
    enable_tracing: bool = True,
) -> Dict[str, Any]:
    """Synchronous version of workflow execution function"""
    return asyncio.run(
        run_workflow(
            yaml_path,
            tools,
            initial_state,
            backend_type=backend_type,
            enable_tracing=enable_tracing,
        )
    )


# Export public interfaces
__all__ = [
    "PathResolver",
    "ChangeSet",
    "WriteOp",
    "DeleteOp",
    "ConditionEvaluator",
    "NodeExecutor",
    "DependencyGraph",
    "DeterministicScheduler",
    "load_document",
    "BaseBackend",
    "AutoGenBackend",
    "LangGraphBackend",
    "create_backend",
    "run_workflow",
    "run_workflow_sync",
]
