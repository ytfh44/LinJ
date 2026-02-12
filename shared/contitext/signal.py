"""
信号与等待机制

实现 LinJ 规范第 21 节定义的信号结构和等待条件
框架无关的信号处理实现
"""

from typing import Any, Dict, Optional, Callable, Protocol, runtime_checkable
from pydantic import BaseModel


class Signal(BaseModel):
    """
    信号 (21.1 节)

    - name: 信号名称
    - payload: 载荷数据
    - correlation: 可选关联标识
    框架无关的信号实现
    """

    name: str
    payload: Optional[Any] = None
    correlation: Optional[str] = None


class WaitCondition(BaseModel):
    """
    等待条件 (21.2 节)

    支持：
    - 按 name 精确匹配
    - 按 correlation 精确匹配
    - 谓词表达式
    框架无关的等待条件实现
    """

    name: Optional[str] = None
    correlation: Optional[str] = None
    predicate: Optional[str] = None  # 条件表达式
    evaluator: Optional[Callable[[str, Dict[str, Any]], bool]] = None

    def matches(self, signal: Signal, state: Optional[Dict] = None) -> bool:
        """
        检查信号是否匹配等待条件 (21.2 节)

        支持：
        - 按 name 精确匹配
        - 按 correlation 精确匹配
        - 谓词表达式：使用条件表达式语义

        Args:
            signal: 待检查的信号
            state: 当前状态（用于谓词求值）

        Returns:
            是否匹配
        """
        # 检查 name（精确匹配）
        if self.name and signal.name != self.name:
            return False

        # 检查 correlation（精确匹配）
        if self.correlation and signal.correlation != self.correlation:
            return False

        # 检查谓词表达式
        if self.predicate and state is not None:
            # 将信号载荷注入状态，按照规范使用 $.signal.payload 访问
            eval_state = dict(state)
            eval_state["signal"] = {
                "name": signal.name,
                "payload": signal.payload,
                "correlation": signal.correlation,
            }

            try:
                if self.evaluator:
                    # 使用自定义评估器
                    return self.evaluator(self.predicate, eval_state)
                else:
                    # 默认简单评估器（仅支持基本表达式）
                    return self._evaluate_condition_simple(self.predicate, eval_state)
            except Exception:
                # 谓词求值失败视为不匹配
                return False

        return True

    def _evaluate_condition_simple(self, predicate: str, state: Dict[str, Any]) -> bool:
        """
        简单的条件评估器

        支持基本的表达式求值，不支持复杂的 LINJ 语法
        实际使用时应该根据具体框架替换为合适的评估器
        """
        # 这里实现一个简单的表达式评估器
        # 仅支持基本的比较和逻辑操作

        try:
            # 替换常见的路径表达式
            expr = predicate.replace("$.signal.payload", "signal['payload']")
            expr = expr.replace("$.signal.name", "signal['name']")
            expr = expr.replace("$.signal.correlation", "signal['correlation']")

            # 限制可用的内置函数和变量
            safe_dict = {
                "signal": state.get("signal", {}),
                "state": state,
                "len": len,
                "str": str,
                "int": int,
                "bool": bool,
            }

            # 使用 eval 进行简单求值（注意：在生产环境中需要更安全的实现）
            return bool(eval(expr, {"__builtins__": {}}, safe_dict))
        except Exception:
            return False


class SignalQueue:
    """
    信号队列

    管理信号的发送和等待
    框架无关的信号队列实现
    """

    def __init__(self):
        self._signals: list[Signal] = []
        self._waiters: dict[str, WaitCondition] = {}  # handle -> condition

    def send(self, signal: Signal) -> None:
        """发送信号"""
        self._signals.append(signal)

    def check_waiter(self, handle: str, signal: Signal, state: Dict) -> bool:
        """检查等待者是否等到了信号"""
        if handle not in self._waiters:
            return False

        condition = self._waiters[handle]
        return condition.matches(signal, state)

    def register_waiter(self, handle: str, condition: WaitCondition) -> None:
        """注册等待者"""
        self._waiters[handle] = condition

    def unregister_waiter(self, handle: str) -> None:
        """取消等待者注册"""
        self._waiters.pop(handle, None)

    def find_matching_signal(
        self, condition: WaitCondition, state: Dict
    ) -> Optional[Signal]:
        """查找匹配条件的信号"""
        for signal in self._signals:
            if condition.matches(signal, state):
                return signal
        return None

    def get_signals_for_waiter(self, handle: str, state: Dict) -> list[Signal]:
        """获取等待者匹配的所有信号"""
        if handle not in self._waiters:
            return []

        condition = self._waiters[handle]
        matching_signals = []

        for signal in self._signals:
            if condition.matches(signal, state):
                matching_signals.append(signal)

        return matching_signals

    def clear_signals(self) -> None:
        """清除所有信号"""
        self._signals.clear()

    def get_pending_waiters(self) -> list[str]:
        """获取等待中的句柄列表"""
        return list(self._waiters.keys())

    def has_waiter(self, handle: str) -> bool:
        """检查是否有指定句柄的等待者"""
        return handle in self._waiters
