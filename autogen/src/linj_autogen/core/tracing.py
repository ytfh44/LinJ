"""
诊断追踪系统

实现 27 节建议的追踪记录：
- 执行日志
- 性能监控
- 冲突记录
"""

import time
import logging
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from ..core.errors import LinJError


class LogLevel(str, Enum):
    """日志级别"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceEntry:
    """追踪条目 (27 节)"""

    def __init__(
        self,
        step_id: int,
        round: int,
        node_id: str,
        attempt: int = 1,
        status: str = "started",
        reads_actual: Optional[List[str]] = None,
        writes_actual: Optional[List[str]] = None,
        ts_start_ms: Optional[int] = None,
        ts_end_ms: Optional[int] = None,
        error: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.step_id = step_id
        self.round = round
        self.node_id = node_id
        self.attempt = attempt
        self.status = status
        self.reads_actual = reads_actual or []
        self.writes_actual = writes_actual or []
        self.ts_start_ms = ts_start_ms or int(time.time() * 1000)
        self.ts_end_ms = ts_end_ms
        self.error = error
        self.metadata = metadata or {}

    def complete(
        self,
        status: str,
        writes: Optional[List[str]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        """完成追踪条目"""
        self.status = status
        self.ts_end_ms = int(time.time() * 1000)
        if writes:
            self.writes_actual = writes
        if error:
            self.error = error

    def duration_ms(self) -> Optional[int]:
        """获取执行时长（毫秒）"""
        if self.ts_end_ms is not None:
            return self.ts_end_ms - self.ts_start_ms
        return None


class PerformanceMetrics:
    """性能指标"""

    def __init__(self):
        self.total_steps = 0
        self.total_duration_ms = 0
        self.average_step_duration_ms = 0.0
        self.concurrent_groups = 0
        self.conflict_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def add_step(self, duration_ms: int) -> None:
        """添加步骤指标"""
        self.total_steps += 1
        self.total_duration_ms += duration_ms
        self.average_step_duration_ms = self.total_duration_ms / self.total_steps

    def add_conflict(self) -> None:
        """增加冲突计数"""
        self.conflict_count += 1

    def add_cache_hit(self) -> None:
        """增加缓存命中计数"""
        self.cache_hits += 1

    def add_cache_miss(self) -> None:
        """增加缓存未命中计数"""
        self.cache_misses += 1

    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


class ConflictRecord:
    """冲突记录 (22.2 节)"""

    def __init__(
        self,
        step_id_a: int,
        step_id_b: int,
        paths: List[str],
        resolution: str,
        selected_step_id: int,
        timestamp: Optional[int] = None,
    ):
        self.step_id_a = step_id_a
        self.step_id_b = step_id_b
        self.paths = paths
        self.resolution = resolution  # "accepted_a", "accepted_b", "merged"
        self.selected_step_id = selected_step_id
        self.timestamp = timestamp or int(time.time() * 1000)


class DiagnosticTracer:
    """
    诊断追踪器

    实现 27 节建议的追踪记录功能
    """

    def __init__(self, enable_detailed_logging: bool = True):
        """
        初始化诊断追踪器

        Args:
            enable_detailed_logging: 是否启用详细日志
        """
        self.enable_detailed_logging = enable_detailed_logging
        self._traces: List[TraceEntry] = []
        self._performance_metrics = PerformanceMetrics()
        self._conflicts: List[ConflictRecord] = []
        self._current_step_id = 0
        self._current_round = 0

        # 设置日志处理器
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if enable_detailed_logging:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def start_step(
        self,
        node_id: str,
        reads: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceEntry:
        """
        开始追踪步骤

        Args:
            node_id: 节点 ID
            reads: 实际读取的路径
            metadata: 额外元数据

        Returns:
            追踪条目
        """
        self._current_step_id += 1

        trace = TraceEntry(
            step_id=self._current_step_id,
            round=self._current_round,
            node_id=node_id,
            reads_actual=reads,
            metadata=metadata,
        )

        self._traces.append(trace)

        if self.enable_detailed_logging:
            self.logger.debug(f"Started step {trace.step_id}: node {node_id}")

        return trace

    def complete_step(
        self,
        trace: TraceEntry,
        status: str,
        writes: Optional[List[str]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        完成步骤追踪

        Args:
            trace: 追踪条目
            status: 完成状态
            writes: 实际写入的路径
            error: 错误信息
        """
        error_dict = None
        if error:
            error_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "details": getattr(error, "details", None),
            }

        trace.complete(status, writes, error_dict)

        # 更新性能指标
        if trace.duration_ms():
            self._performance_metrics.add_step(trace.duration_ms())

        if self.enable_detailed_logging:
            self.logger.debug(
                f"Completed step {trace.step_id}: {status} ({trace.duration_ms()}ms)"
            )

            if error:
                self.logger.error(f"Step {trace.step_id} failed: {error}")

    def start_round(self) -> None:
        """开始新轮次"""
        self._current_round += 1

        if self.enable_detailed_logging:
            self.logger.info(f"Starting round {self._current_round}")

    def record_conflict(
        self,
        step_id_a: int,
        step_id_b: int,
        paths: List[str],
        resolution: str,
        selected_step_id: int,
    ) -> None:
        """
        记录冲突 (22.2 节)

        Args:
            step_id_a: 第一个步骤 ID
            step_id_b: 第二个步骤 ID
            paths: 相交路径
            resolution: 解决方案
            selected_step_id: 选中的步骤 ID
        """
        conflict = ConflictRecord(
            step_id_a=step_id_a,
            step_id_b=step_id_b,
            paths=paths,
            resolution=resolution,
            selected_step_id=selected_step_id,
        )

        self._conflicts.append(conflict)
        self._performance_metrics.add_conflict()

        if self.enable_detailed_logging:
            self.logger.warning(
                f"Conflict between step {step_id_a} and {step_id_b} "
                f"on paths {paths}: {resolution}"
            )

    def record_cache_operation(self, hit: bool) -> None:
        """记录缓存操作"""
        if hit:
            self._performance_metrics.add_cache_hit()
        else:
            self._performance_metrics.add_cache_miss()

    def record_concurrent_group(self, node_count: int) -> None:
        """记录并发组"""
        self._performance_metrics.concurrent_groups += 1

        if self.enable_detailed_logging:
            self.logger.info(f"Concurrent group with {node_count} nodes")

    def get_trace_summary(self) -> Dict[str, Any]:
        """获取追踪摘要"""
        if not self._traces:
            return {}

        # 统计各状态的数量
        status_counts = {}
        for trace in self._traces:
            status_counts[trace.status] = status_counts.get(trace.status, 0) + 1

        # 计算统计信息
        successful_steps = sum(
            1 for trace in self._traces if trace.status in ("completed", "success")
        )
        failed_steps = sum(
            1 for trace in self._traces if trace.status in ("failed", "error")
        )

        durations = [
            trace.duration_ms()
            for trace in self._traces
            if trace.duration_ms() is not None
        ]

        return {
            "total_steps": len(self._traces),
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "status_distribution": status_counts,
            "average_duration_ms": sum(durations) / len(durations) if durations else 0,
            "total_rounds": self._current_round,
            "performance_metrics": {
                "total_steps": self._performance_metrics.total_steps,
                "total_duration_ms": self._performance_metrics.total_duration_ms,
                "average_step_duration_ms": self._performance_metrics.average_step_duration_ms,
                "concurrent_groups": self._performance_metrics.concurrent_groups,
                "conflict_count": self._performance_metrics.conflict_count,
                "cache_hit_rate": self._performance_metrics.get_cache_hit_rate(),
            },
        }

    def get_recent_traces(self, limit: int = 50) -> List[TraceEntry]:
        """获取最近的追踪记录"""
        return self._traces[-limit:]

    def get_conflicts(self) -> List[ConflictRecord]:
        """获取所有冲突记录"""
        return self._conflicts.copy()

    def export_to_state(self, state: Dict[str, Any]) -> None:
        """
        导出追踪信息到状态 (27 节)

        Args:
            state: 目标状态对象
        """
        # 按照 27 节建议，在主状态中维护 $.trace[]
        trace_data = {
            "steps": [
                {
                    "step_id": trace.step_id,
                    "round": trace.round,
                    "node_id": trace.node_id,
                    "attempt": trace.attempt,
                    "status": trace.status,
                    "reads_actual": trace.reads_actual,
                    "writes_actual": trace.writes_actual,
                    "ts_start_ms": trace.ts_start_ms,
                    "ts_end_ms": trace.ts_end_ms,
                    "duration_ms": trace.duration_ms(),
                    "error": trace.error,
                    "metadata": trace.metadata,
                }
                for trace in self._traces
            ],
            "conflicts": [
                {
                    "step_id_a": conflict.step_id_a,
                    "step_id_b": conflict.step_id_b,
                    "paths": conflict.paths,
                    "resolution": conflict.resolution,
                    "selected_step_id": conflict.selected_step_id,
                    "timestamp": conflict.timestamp,
                }
                for conflict in self._conflicts
            ],
            "summary": self.get_trace_summary(),
            "export_timestamp": int(time.time() * 1000),
        }

        state["$.trace"] = trace_data

        if self.enable_detailed_logging:
            self.logger.info(f"Exported {len(self._traces)} traces to state")

    def clear_traces(self) -> None:
        """清除所有追踪记录"""
        self._traces.clear()
        self._conflicts.clear()
        self._performance_metrics = PerformanceMetrics()
        self._current_step_id = 0
        self._current_round = 0

        if self.enable_detailed_logging:
            self.logger.info("Cleared all traces")


class TracingMixin:
    """
    追踪混入类

    为其他组件提供追踪能力的混入
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer: Optional[DiagnosticTracer] = None

    def set_tracer(self, tracer: DiagnosticTracer) -> None:
        """设置追踪器"""
        self.tracer = tracer

    def trace_step_start(
        self, node_id: str, reads: Optional[List[str]] = None
    ) -> Optional[TraceEntry]:
        """开始步骤追踪（如果设置了追踪器）"""
        if self.tracer:
            return self.tracer.start_step(node_id, reads)
        return None

    def trace_step_complete(
        self,
        trace: Optional[TraceEntry],
        status: str,
        writes: Optional[List[str]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """完成步骤追踪（如果设置了追踪器）"""
        if self.tracer and trace:
            self.tracer.complete_step(trace, status, writes, error)

    def trace_conflict(
        self,
        step_id_a: int,
        step_id_b: int,
        paths: List[str],
        resolution: str,
        selected_step_id: int,
    ) -> None:
        """记录冲突（如果设置了追踪器）"""
        if self.tracer:
            self.tracer.record_conflict(
                step_id_a, step_id_b, paths, resolution, selected_step_id
            )
