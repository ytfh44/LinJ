"""
LangGraph Backend Adapter for Examples

完整对齐 LinJ.md 规范的 LangGraph 后端实现：
- 决定性调度 (11.3)
- 状态路径 (5)
- 变更集原子性 (9.2)
- 节点类型 (13): hint/tool/join/gate
- 条件表达式 (14): exists/len/value
- 依赖图 (8)
- 循环 (12)
- 并发安全 (11.5)

提供与 AutoGen 版本完全兼容的执行接口。
"""

import asyncio
import re
import sys
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple, Union
from dataclasses import dataclass, field

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 确保 autogen/src 在路径中
autogen_src = project_root / "autogen" / "src"
if str(autogen_src) not in sys.path:
    sys.path.insert(0, str(autogen_src))


# ============================================================================
# 路径解析器 (对齐 LinJ.md 5.1-5.4)
# ============================================================================


class PathResolver:
    """状态路径解析器 - 实现 5.1-5.4 节规范"""

    @staticmethod
    def _parse_path_parts(path: str) -> List[str]:
        """解析路径为部分列表，支持数组索引"""
        if not path.startswith("$."):
            path = f"$.{path}"

        # 移除 $.
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
        """读取路径值 (5.2) - 不存在返回 null"""
        parts = PathResolver._parse_path_parts(path)
        current: Any = state

        # 处理根路径 $
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
        """写入路径值 (5.3)"""
        parts = PathResolver._parse_path_parts(path)
        current: Any = state

        # 遍历到倒数第二个部分
        for i, part in enumerate(parts[:-1]):
            if part not in current or current[part] is None:
                # 根据下一部分判断创建对象还是数组
                next_part = parts[i + 1]
                try:
                    idx = int(next_part)
                    current[part] = []
                    while len(current[part]) <= idx:
                        current[part].append(None)
                except ValueError:
                    current[part] = {}

            current = current[part]

        # 设置最后一部分
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
        """删除路径值 (5.4) - 设为 null"""
        parts = PathResolver._parse_path_parts(path)

        # 找到目标路径的父级
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

        # 设为 null
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
        """删除路径值 (5.4) - 设为 null"""
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

        # 设为 null (不是删除键，因为数组不能缩短)
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
        """判断两路径是否相交 (11.4)"""
        # 前缀判定
        if path_a.startswith(path_b) or path_b.startswith(path_a):
            return True
        if path_a == path_b:
            return True

        # 数组下标不同不相交
        try:
            # 提取路径部分和数组下标
            parts_a = path_a.replace("$.", "").split(".")
            parts_b = path_b.replace("$.", "").split(".")

            # 检查数组下标是否不同
            min_len = min(len(parts_a), len(parts_b))
            for i in range(min_len):
                try:
                    idx_a = int(parts_a[i])
                    idx_b = int(parts_b[i])
                    if idx_a != idx_b:
                        # 下标不同，如果前面部分相同则不相交
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
# 变更集 (对齐 LinJ.md 9)
# ============================================================================


@dataclass
class WriteOp:
    """写操作"""

    path: str
    value: Any


@dataclass
class DeleteOp:
    """删除操作"""

    path: str


@dataclass
class ChangeSet:
    """
    变更集 (9.1)

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
        """检查路径相交 (11.4)"""
        for path_a in self.get_all_modified_paths():
            for path_b in other.get_all_modified_paths():
                if PathResolver.intersect(path_a, path_b):
                    return True
        return False

    def apply_to(self, state: Dict[str, Any]) -> None:
        """应用变更集 (9.2 原子性由调用方保证)"""
        # 先删除
        for op in self.deletes:
            PathResolver.delete(state, op.path)
        # 后写入
        for op in self.writes:
            PathResolver.set(state, op.path, op.value)


# ============================================================================
# 条件表达式评估器 (对齐 LinJ.md 14)
# ============================================================================


class ConditionEvaluator:
    """条件表达式评估器 (14)"""

    # 比较操作符
    COMP_OPS = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b if a is not None and b is not None else False,
        ">=": lambda a, b: a >= b if a is not None and b is not None else False,
        "<": lambda a, b: a < b if a is not None and b is not None else False,
        "<=": lambda a, b: a <= b if a is not None and b is not None else False,
    }

    # 逻辑操作符
    LOGIC_OPS = {
        "AND": lambda a, b: a and b,
        "OR": lambda a, b: a or b,
    }

    # 路径函数
    PATH_FUNCS = {
        "exists": lambda path: path is not None,
        "len": lambda path: len(path) if isinstance(path, list) else 0,
        "value": lambda path: path,
    }

    @classmethod
    def evaluate(cls, condition: str, state: Dict[str, Any]) -> bool:
        """求值条件表达式"""
        if not condition or not condition.strip():
            return True

        # 预处理：展开函数调用
        expr = condition

        # 展开 exists(path)
        def replace_exists(match):
            path = match.group(1).strip()
            value = PathResolver.get(state, path)
            return "True" if value is not None else "False"

        expr = re.sub(r"exists\(\s*([^)]+)\s*\)", replace_exists, expr)

        # 展开 len(path)
        def replace_len(match):
            path = match.group(1).strip()
            value = PathResolver.get(state, path)
            length = len(value) if isinstance(value, list) else 0
            return str(length)

        expr = re.sub(r"len\(\s*([^)]+)\s*\)", replace_len, expr)

        # 展开 value(path)
        def replace_value(match):
            path = match.group(1).strip()
            value = PathResolver.get(state, path)
            if value is None:
                return "None"
            if isinstance(value, str):
                return f'"{value}"'
            return str(value)

        expr = re.sub(r"value\(\s*([^)]+)\s*\)", replace_value, expr)

        # 安全地求值表达式
        try:
            # 替换操作符为 Python 语法
            expr = expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not")

            # 添加安全的环境
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
# 节点执行器 (对齐 LinJ.md 13)
# ============================================================================


class NodeExecutor:
    """节点执行器 - 实现 13 节节点类型语义"""

    def __init__(self, tools: Dict[str, Callable], state: Dict[str, Any]):
        self.tools = tools
        self.state = state

    def can_retry(self, node: Dict[str, Any], attempt: int) -> bool:
        """
        检查工具调用是否可以在当前尝试次数时重试 (13.2 节)

        规则：
        - 若 effect 为 write 且 repeat_safe 为 false：即使全局允许重试，执行器必须不自动重试
        - 若 effect 为 none 或 read：可按策略重试

        Args:
            node: 工具节点配置
            attempt: 当前尝试次数（0-based）

        Returns:
            是否可以重试
        """
        effect = node.get("effect", "read")
        repeat_safe = node.get("repeat_safe", False)

        # 若 effect 为 write 且 repeat_safe 为 false：禁止重试
        if effect == "write" and not repeat_safe:
            return False

        # 首次尝试不需要重试
        if attempt == 0:
            return True

        # effect 为 none 或 read：允许重试（但需要在调用处控制最大重试次数）
        return effect in ("none", "read")

    def execute_tool(self, node: Dict[str, Any], retry_count: int = 0) -> ChangeSet:
        """
        执行 tool 节点 (13.2)

        Args:
            node: 工具节点配置
            retry_count: 当前重试次数（用于判断是否可重试）
        """
        call = node.get("call", {})
        name = call.get("name")
        args_dict = call.get("args", {})
        write_to = node.get("write_to")
        effect = node.get("effect", "read")

        # 解析参数
        args = {}
        for key, value in args_dict.items():
            if isinstance(value, dict) and "$path" in value:
                value = PathResolver.get(self.state, value["$path"])
            elif isinstance(value, str) and value.startswith("$."):
                value = PathResolver.get(self.state, value)
            args[key] = value

        # 调用工具
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

        # 处理错误和重试
        if error:
            # 检查是否可以重试（13.2 节规则）
            if self.can_retry(node, retry_count):
                # 重试次数限制（可在策略中配置）
                max_retries = 3
                if retry_count < max_retries:
                    # 简单指数退避
                    import time

                    time.sleep(0.1 * (2**retry_count))
                    return self.execute_tool(node, retry_count + 1)

            # 不可重试或已达到最大重试次数，返回错误结果
            result = f"Error: {error}"

        # 应用变更
        if write_to:
            return ChangeSet(writes=[WriteOp(path=write_to, value=result)])
        return ChangeSet()

    def execute_join(self, node: Dict[str, Any]) -> Tuple[ChangeSet, Optional[str]]:
        """执行 join 节点 (13.3)"""
        input_from = node.get("input_from")
        output_to = node.get("output_to")
        glossary = node.get("glossary")

        # 读取输入
        text = PathResolver.get(self.state, input_from) or ""
        text_str = str(text)

        # 验证禁止项
        forbidden = None
        if glossary:
            for item in glossary:
                if item.get("forbid"):
                    for word in item["forbid"]:
                        if word in text_str:
                            forbidden = word
                            break

        # 输出
        if output_to:
            cs = ChangeSet(writes=[WriteOp(path=output_to, value=text)])
        else:
            cs = ChangeSet()

        return cs, forbidden

    def execute_gate(self, node: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """执行 gate 节点 (13.4)"""
        condition = node.get("condition", "")
        then = node.get("then", [])
        else_ = node.get("else", [])

        # 求值条件
        result = ConditionEvaluator.evaluate(condition, self.state)

        return result, then, else_

    def execute(self, node: Dict[str, Any]) -> Tuple[ChangeSet, Optional[str], Any]:
        """
        执行单个节点

        Returns:
            - changeset: 变更集
            - forbidden: 禁止项（仅 join）
            - gate_result: 门控结果（仅 gate）
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
# 依赖图 (对齐 LinJ.md 8)
# ============================================================================


class DependencyGraph:
    """依赖图管理"""

    def __init__(self, edges: List[Dict[str, Any]]):
        self.edges = edges
        self.adjacency: Dict[str, List[str]] = {}  # node -> [dependents]
        self.reverse_adj: Dict[str, List[str]] = {}  # node -> [dependencies]
        self.edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self._build_graph()

    def _build_graph(self) -> None:
        """构建依赖图"""
        for edge in self.edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            kind = edge.get("kind", "data")

            # 添加边
            self.edge_map[(from_node, to_node)] = edge

            # 更新邻接表
            if from_node not in self.adjacency:
                self.adjacency[from_node] = []
            if to_node not in self.reverse_adj:
                self.reverse_adj[to_node] = []

            self.adjacency[from_node].append(to_node)
            self.reverse_adj[to_node].append(from_node)

    def get_dependencies(self, node_id: str) -> List[str]:
        """获取节点的前置依赖"""
        return self.reverse_adj.get(node_id, [])

    def get_dependents(self, node_id: str) -> List[str]:
        """获取依赖该节点的节点"""
        return self.adjacency.get(node_id, [])

    def get_incoming_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """获取进入该节点的所有边"""
        result = []
        for edge in self.edges:
            if edge.get("to") == node_id:
                result.append(edge)
        return result


# ============================================================================
# 决定性调度器 (对齐 LinJ.md 11.3)
# ============================================================================


class DeterministicScheduler:
    """决定性调度器 - 11.3 节"""

    def __init__(self, nodes: List[Dict[str, Any]], dependency_graph: DependencyGraph):
        self.nodes = nodes
        self.dependency_graph = dependency_graph
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.executed_steps: List[Dict[str, Any]] = []

    def are_dependencies_satisfied(self, node_id: str, completed: Set[str]) -> bool:
        """检查依赖是否满足"""
        deps = self.dependency_graph.get_dependencies(node_id)
        return all(dep in completed for dep in deps)

    def get_ready_nodes(self, completed: Set[str]) -> List[Dict[str, Any]]:
        """获取所有就绪的节点"""
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
        选择下一个执行的节点

        决定性顺序 (11.3):
        1. rank 较大者优先（未提供视为 0）
        2. nodes 数组中靠前者优先
        3. 若仍相同，按 node_id 字典序
        """
        if not ready_nodes:
            return None

        # 排序
        ready_nodes.sort(
            key=lambda n: (
                n.get("rank", 0) or 0,  # rank
                self.nodes.index(n),  # nodes 数组顺序
                n.get("id", ""),  # node_id 字典序
            ),
            reverse=True,
        )

        return ready_nodes[0]

    def can_execute_concurrently(
        self, node_a: Dict[str, Any], node_b: Dict[str, Any]
    ) -> bool:
        """
        检查两节点是否可以并发执行 (11.5)

        要求:
        - 写入集合互不相交
        - 读取集合不得与对方写入集合相交
        """
        reads_a = set(node_a.get("reads", []) or [])
        writes_a = set(node_a.get("writes", []) or [])
        reads_b = set(node_b.get("reads", []) or [])
        writes_b = set(node_b.get("writes", []) or [])

        # 写入集合必须不相交
        for path_a in writes_a:
            for path_b in writes_b:
                if PathResolver.intersect(path_a, path_b):
                    return False

        # 读取不得与对方写入相交
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
# 信号系统 (对齐 LinJ.md 21)
# ============================================================================


@dataclass
class Signal:
    """信号结构 (21.1 节)"""

    name: str
    payload: Any = None
    correlation: Optional[str] = None


@dataclass
class WaitCondition:
    """等待条件 (21.2 节)"""

    name: Optional[str] = None
    correlation: Optional[str] = None
    predicate: Optional[str] = None

    def matches(self, signal: Signal, state: Dict[str, Any]) -> bool:
        """检查信号是否匹配等待条件"""
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
    """信号队列"""

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
# LangGraph 后端实现 (完整对齐)
# ============================================================================


class BaseBackend:
    """后端基类"""

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
        """配置日志"""
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
        """注册工具函数"""
        self._tools[name] = func

    def register_tools(self, tools: Dict[str, Callable]) -> None:
        """批量注册工具"""
        self._tools.update(tools)

    async def run(
        self,
        document: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """执行工作流（子类必须实现）"""
        raise NotImplementedError

    def run_sync(
        self,
        document: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """同步执行"""
        return asyncio.run(self.run(document, initial_state, **kwargs))


class AutoGenBackend(BaseBackend):
    """
    AutoGen 后端 - 使用 linj_autogen.executor.runner.LinJExecutor
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
    LangGraph 后端 - 完整对齐 LinJ.md 规范

    关键实现点:
    - 决定性调度 (11.3)
    - 状态路径 (5)
    - 变更集原子性 (9.2)
    - 节点类型 (13)
    - 条件表达式 (14)
    - 循环处理 (12)
    - 并发安全 (11.5)
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
        执行工作流

        Args:
            document: LinJ 文档
            initial_state: 初始状态

        Returns:
            最终状态
        """
        return await self._execute(document, initial_state or {})

    def _is_loop_entry_node(
        self,
        node_id: str,
        loop_states: Dict[str, Any],
        dependency_graph: DependencyGraph,
    ) -> bool:
        """
        检查节点是否是循环入口节点

        循环入口节点：有边指向循环members，但自身不在循环内的节点
        对于入口节点，使用OR语义检查依赖（只要有一个依赖满足即可）
        """
        for loop_id, loop_state in loop_states.items():
            # 检查是否有从循环外的节点指向该循环成员的边
            for member_id in loop_state["members"]:
                incoming = dependency_graph.get_incoming_edges(member_id)
                for edge in incoming:
                    from_node = (
                        edge.get("from") if isinstance(edge, dict) else edge.from_
                    )
                    if from_node == node_id:
                        # 检查该节点是否在循环内
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
        检查节点的依赖是否已满足

        对于循环入口节点：使用OR语义（只要有一个依赖满足即可）
        对于普通节点：使用AND语义（所有依赖都必须满足）
        """
        deps = dependency_graph.get_dependencies(node_id)
        if not deps:
            return True

        # 判断是否是循环入口节点
        is_loop_entry = self._is_loop_entry_node(node_id, loop_states, dependency_graph)

        if is_loop_entry:
            # 循环入口节点：OR语义
            return any(dep in completed or dep in activated for dep in deps)
        else:
            # 普通节点：AND语义
            return all(dep in completed or dep in activated for dep in deps)

        max_steps = policies.get("max_steps", 1000)
        max_rounds = policies.get("max_rounds", 100)

        # 3. 构建依赖图
        dependency_graph = DependencyGraph(edges)

        # 4. 创建调度器
        scheduler = DeterministicScheduler(nodes, dependency_graph)

        # 5. 初始化追踪和冲突检测
        trace = []
        step_id = 0
        current_round = 0
        activated = set()  # 已激活的节点
        completed = set()  # 已完成的节点
        failed = set()  # 失败的节点
        applied_changesets: List[ChangeSet] = []  # 已应用的变更集

        # 初始化循环状态
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

        # 6. 主执行循环
        while current_round < max_rounds and step_id < max_steps:
            round_ready = set()

            # 6.1 收集就绪节点
            for node in nodes:
                node_id = node.get("id")
                if node_id in completed or node_id in failed or node_id in activated:
                    continue

                # 检查依赖（支持循环入口节点的OR语义）
                if not self._check_dependencies_satisfied(
                    node_id, completed, activated, loop_states, dependency_graph
                ):
                    continue

                # 检查循环成员限制
                in_loop = False
                for loop_id, loop_state in loop_states.items():
                    if node_id in loop_state["members"]:
                        in_loop = True
                        if node_id in loop_state["executed"]:
                            # 同一轮内不重复执行
                            continue
                        # 在循环内，检查 round 限制
                        if loop_state["current_round"] >= loop_state["max_rounds"]:
                            continue

                round_ready.add(node)

            if not round_ready:
                # 没有就绪节点，检查是否可以推进
                can_advance = False
                for loop_id, loop_state in loop_states.items():
                    if loop_state["executed"]:
                        # 循环前进
                        loop_state["current_round"] += 1
                        loop_state["executed"].clear()
                        can_advance = True

                if not can_advance:
                    # 工作流完成
                    break
                else:
                    current_round += 1
                    continue

            # 6.2 决定性选择节点 (11.3)
            ready_list = list(round_ready)
            ready_list.sort(
                key=lambda n: (n.get("rank", 0) or 0, nodes.index(n), n.get("id", "")),
                reverse=True,
            )

            # 6.3 顺序执行（保证决定性）
            for node in ready_list:
                if step_id >= max_steps:
                    break

                node_id = node.get("id")
                node_type = node.get("type")

                self._tracer.info(
                    f"Executing node: {node_id} ({node_type}), step_id={step_id}"
                )

                try:
                    # 执行节点
                    executor = NodeExecutor(self._tools, state)
                    changeset, forbidden, gate_result = executor.execute(node)

                    # 检查 join 的禁止项
                    if forbidden:
                        raise ValueError(f"Forbidden term found: {forbidden}")

                    # 应用变更集 (9.2 原子性)
                    if not changeset.is_empty():
                        # 冲突检测 (22.1 节：拒绝策略)
                        conflict = False
                        for prev_cs in applied_changesets:
                            if changeset.intersects_with(prev_cs):
                                conflict = True
                                break

                        if conflict:
                            # 冲突产生错误
                            raise ValueError(
                                f"ConflictError: changeset for node {node_id} at step {step_id} "
                                f"intersects with previously applied changesets"
                            )

                        # 无冲突，应用变更集
                        changeset.apply_to(state)
                        applied_changesets.append(changeset)

                    # 记录追踪 (27)
                    trace.append(
                        {
                            "step_id": step_id,
                            "round": current_round,
                            "node_id": node_id,
                            "status": "completed",
                            "writes": list(changeset.get_write_paths()),
                        }
                    )

                    # 更新循环状态
                    for loop_id, loop_state in loop_states.items():
                        if node_id in loop_state["members"]:
                            loop_state["executed"].add(node_id)

                    activated.add(node_id)
                    step_id += 1

                    # 处理 gate 激活
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

            # 6.4 完成本轮激活的节点
            completed.update(activated)
            activated.clear()

            current_round += 1

        # 7. 添加追踪信息到状态
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
    创建指定类型的后端执行器

    Args:
        backend_type: "autogen" 或 "langgraph"
        enable_tracing: 是否启用追踪
        config: 可选配置

    Returns:
        后端执行器实例
    """
    backend_type = backend_type.lower()

    if backend_type == "langgraph":
        return LangGraphBackend(enable_tracing=enable_tracing, config=config)
    else:
        return AutoGenBackend(enable_tracing=enable_tracing, config=config)


def load_document(path: str) -> Dict[str, Any]:
    """
    加载 YAML 格式的 LinJ 文档

    Args:
        path: YAML 文件路径

    Returns:
        解析后的文档对象
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
    """便捷的工作流执行函数"""
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
    """同步版本的工作流执行函数"""
    return asyncio.run(
        run_workflow(
            yaml_path,
            tools,
            initial_state,
            backend_type=backend_type,
            enable_tracing=enable_tracing,
        )
    )


# 导出公共接口
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
