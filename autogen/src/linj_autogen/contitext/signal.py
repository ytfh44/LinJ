"""
信号与等待机制

实现 21 节定义的信号结构和等待条件
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class Signal(BaseModel):
    """
    信号 (21.1 节)

    - name: 信号名称
    - payload: 载荷数据
    - correlation: 可选关联标识
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
    """

    name: Optional[str] = None
    correlation: Optional[str] = None
    predicate: Optional[str] = None  # 条件表达式

    def matches(self, signal: Signal, state: Optional[Dict] = None) -> bool:
        """
        检查信号是否匹配等待条件 (21.2 节)

        支持：
        - 按 name 精确匹配
        - 按 correlation 精确匹配
        - 谓词表达式：使用 LinJ 条件表达式语义

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

            from ..executor.evaluator import evaluate_condition

            try:
                return evaluate_condition(self.predicate, eval_state)
            except Exception:
                # 谓词求值失败视为不匹配
                return False

        return True


class SignalQueue:
    """
    信号队列

    管理信号的发送和等待
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
