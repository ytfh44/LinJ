"""
Diagnostic Tracing System

Implements Section 27's tracing recommendations:
- Execution logs
- Performance monitoring
- Conflict recording

And Section 26's non-replayable diagnostic recording:
- $.diagnostics.non_replayable
"""

import time
import logging
import uuid
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime

from ..exceptions.errors import LinJError
from .path import PathResolver


class LogLevel(str, Enum):
    """Log level"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceEntry:
    """Trace entry (Section 27)"""

    def __init__(
        self,
        step_id: int,
        round: int,
        node_id: str,
        attempt: int = 1,
        status: str = "pending",
        reads_actual: Optional[List[str]] = None,
        writes_actual: Optional[List[str]] = None,
        reads_declared: Optional[List[str]] = None,
        writes_declared: Optional[List[str]] = None,
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
        self.reads_declared = reads_declared or []
        self.writes_declared = writes_declared or []
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
        """Complete trace entry"""
        self.status = status
        self.ts_end_ms = int(time.time() * 1000)
        if writes:
            self.writes_actual = writes
        if error:
            self.error = error

    def duration_ms(self) -> Optional[int]:
        """Get execution duration (milliseconds)"""
        if self.ts_end_ms is not None:
            return self.ts_end_ms - self.ts_start_ms
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace entry to dictionary"""
        return {
            "step_id": self.step_id,
            "round": self.round,
            "node_id": self.node_id,
            "attempt": self.attempt,
            "status": self.status,
            "reads_actual": self.reads_actual,
            "writes_actual": self.writes_actual,
            "reads_declared": self.reads_declared,
            "writes_declared": self.writes_declared,
            "ts_start_ms": self.ts_start_ms,
            "ts_end_ms": self.ts_end_ms,
            "duration_ms": self.duration_ms(),
            "error": self.error,
            "metadata": self.metadata,
        }


class PerformanceMetrics:
    """Performance metrics"""

    def __init__(self):
        self.total_steps = 0
        self.total_duration_ms = 0
        self.average_step_duration_ms = 0.0
        self.concurrent_groups = 0
        self.conflict_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def add_step(self, duration_ms: int) -> None:
        """Add step metric"""
        self.total_steps += 1
        self.total_duration_ms += duration_ms
        self.average_step_duration_ms = self.total_duration_ms / self.total_steps

    def add_conflict(self) -> None:
        """Increase conflict count"""
        self.conflict_count += 1

    def add_cache_hit(self) -> None:
        """Increase cache hit count"""
        self.cache_hits += 1

    def add_cache_miss(self) -> None:
        """Increase cache miss count"""
        self.cache_misses += 1

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


class ConflictRecord:
    """Conflict record (Section 22.2)"""

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
    Diagnostic Tracer

    Implements Section 27's tracing functionality
    """

    def __init__(self, enable_detailed_logging: bool = True):
        """
        Initialize diagnostic tracer

        Args:
            enable_detailed_logging: Whether to enable detailed logging
        """
        self.enable_detailed_logging = enable_detailed_logging
        self._traces: List[TraceEntry] = []
        self._performance_metrics = PerformanceMetrics()
        self._conflicts: List[ConflictRecord] = []
        self._current_step_id = 0
        self._current_round = 0

        # Set up logging handler
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
        Start tracing step

        Args:
            node_id: Node ID
            reads: Actual paths read
            metadata: Additional metadata

        Returns:
            Trace entry
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
        Complete step tracing

        Args:
            trace: Trace entry
            status: Completion status
            writes: Actual paths written
            error: Error information
        """
        error_dict = None
        if error:
            error_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "details": getattr(error, "details", None),
            }

        trace.complete(status, writes, error_dict)

        # Update performance metrics
        duration = trace.duration_ms()
        if duration is not None:
            self._performance_metrics.add_step(duration)

        if self.enable_detailed_logging:
            self.logger.debug(
                f"Completed step {trace.step_id}: {status} ({trace.duration_ms()}ms)"
            )

            if error:
                self.logger.error(f"Step {trace.step_id} failed: {error}")

    def start_round(self) -> None:
        """Start new round"""
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
        Record conflict (Section 22.2)

        Args:
            step_id_a: First step ID
            step_id_b: Second step ID
            paths: Intersecting paths
            resolution: Resolution strategy
            selected_step_id: Selected step ID
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
        """Record cache operation"""
        if hit:
            self._performance_metrics.add_cache_hit()
        else:
            self._performance_metrics.add_cache_miss()

    def record_concurrent_group(self, node_count: int) -> None:
        """Record concurrent group"""
        self._performance_metrics.concurrent_groups += 1

        if self.enable_detailed_logging:
            self.logger.info(f"Concurrent group with {node_count} nodes")

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get trace summary"""
        if not self._traces:
            return {}

        # Count status distribution
        status_counts = {}
        for trace in self._traces:
            status_counts[trace.status] = status_counts.get(trace.status, 0) + 1

        # Calculate statistics
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
            "average_duration_ms": sum(d for d in durations if d is not None)
            / len(durations)
            if durations
            else 0,
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
        """Get recent trace records"""
        return self._traces[-limit:]

    def get_conflicts(self) -> List[ConflictRecord]:
        """Get all conflict records"""
        return self._conflicts.copy()

    def export_to_state(self, state: Dict[str, Any]) -> None:
        """
        Export trace information to state (Section 27)

        Args:
            state: Target state object
        """
        # Per Section 27, maintain $.trace[] in the main state
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
        """Clear all trace records"""
        self._traces.clear()
        self._conflicts.clear()
        self._performance_metrics = PerformanceMetrics()
        self._current_step_id = 0
        self._current_round = 0

        if self.enable_detailed_logging:
            self.logger.info("Cleared all traces")


class TracingMixin:
    """
    Tracing Mixin

    Provides tracing capability to other components as a mixin
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer: Optional[DiagnosticTracer] = None

    def set_tracer(self, tracer: DiagnosticTracer) -> None:
        """Set tracer"""
        self.tracer = tracer

    def trace_step_start(
        self, node_id: str, reads: Optional[List[str]] = None
    ) -> Optional[TraceEntry]:
        """Start step tracing (if tracer is set)"""
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
        """Complete step tracing (if tracer is set)"""
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
        """Record conflict (if tracer is set)"""
        if self.tracer:
            self.tracer.record_conflict(
                step_id_a, step_id_b, paths, resolution, selected_step_id
            )


class NonReplayableDiagnostic:
    """
    Non-Replayable Diagnostic Record (Section 26)

    When a tool call encounters external non-replayable conditions during recovery or retry
    (e.g., credential expiration, external state changed) that prevent consistency requirements
    from being met, diagnostic information should be written to the main state:

    `$.diagnostics.non_replayable`: Object, must contain node_id, tool_name, reason, at_step_id
    """

    def __init__(
        self,
        node_id: str,
        tool_name: str,
        reason: str,
        at_step_id: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = node_id
        self.tool_name = tool_name
        self.reason = reason
        self.at_step_id = at_step_id
        self.details = details or {}
        self.timestamp_ms = int(time.time() * 1000)
        self.timestamp = datetime.now().isoformat()
        self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "tool_name": self.tool_name,
            "reason": self.reason,
            "at_step_id": self.at_step_id,
            "timestamp_ms": self.timestamp_ms,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class DiagnosticsRecorder:
    """
    Diagnostics Recorder

    Implements Section 26's non-replayable diagnostic recording functionality
    """

    def __init__(self, state_manager: Optional[Any] = None):
        self._state_manager = state_manager
        self._diagnostics: List[NonReplayableDiagnostic] = []

    def record_non_replayable(
        self,
        node_id: str,
        tool_name: str,
        reason: str,
        at_step_id: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> NonReplayableDiagnostic:
        """
        Record non-replayable diagnostic

        Args:
            node_id: Node ID
            tool_name: Tool name
            reason: Failure reason
            at_step_id: Step ID
            details: Additional details

        Returns:
            Diagnostic record
        """
        diagnostic = NonReplayableDiagnostic(
            node_id=node_id,
            tool_name=tool_name,
            reason=reason,
            at_step_id=at_step_id,
            details=details,
        )

        self._diagnostics.append(diagnostic)

        # Write to state (if state manager provided)
        if self._state_manager:
            self.write_to_state(self._state_manager.get_full_state())

        return diagnostic

    def get_diagnostics(self) -> List[NonReplayableDiagnostic]:
        """Get all diagnostic records"""
        return self._diagnostics.copy()

    def get_diagnostics_for_node(self, node_id: str) -> List[NonReplayableDiagnostic]:
        """Get diagnostic records for specified node"""
        return [d for d in self._diagnostics if d.node_id == node_id]

    def clear(self) -> None:
        """Clear diagnostic records"""
        self._diagnostics.clear()

    def write_to_state(
        self, state: Dict[str, Any], path: str = "$.diagnostics.non_replayable"
    ) -> None:
        """
        Write diagnostics to state

        Args:
            state: Target state dictionary
            path: Diagnostic path
        """
        diagnostic_list = [d.to_dict() for d in self._diagnostics]
        PathResolver.set(state, path, diagnostic_list)

    def get_summary(self) -> Dict[str, Any]:
        """Get diagnostic summary"""
        return {
            "total_diagnostics": len(self._diagnostics),
            "by_node": self._group_by_node(),
            "by_tool": self._group_by_tool(),
        }

    def _group_by_node(self) -> Dict[str, int]:
        return {d.node_id: self._count_node(d.node_id) for d in self._diagnostics}

    def _group_by_tool(self) -> Dict[str, int]:
        return {d.tool_name: self._count_tool(d.tool_name) for d in self._diagnostics}

    def _count_node(self, node_id: str) -> int:
        return sum(1 for d in self._diagnostics if d.node_id == node_id)

    def _count_tool(self, tool_name: str) -> int:
        return sum(1 for d in self._diagnostics if d.tool_name == tool_name)
