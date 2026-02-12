"""
LangGraph State Management

This module provides state management specifically for LangGraph workflows,
integrating with the shared core state management while providing LangGraph-specific
extensions and optimizations.
"""

from typing import Any, Dict, List, Optional, Union, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

from shared.core.state import StateManager, StateView
from shared.core.changeset import ChangeSet


class WorkflowState(TypedDict, total=False):
    """TypedDict for LangGraph workflow state structure"""

    # Core workflow information
    workflow_id: str
    step_id: int
    node_id: str

    # Execution context
    messages: List[Dict[str, Any]]
    intermediate_steps: List[Dict[str, Any]]

    # Configuration
    config: Dict[str, Any]

    # Results and outputs
    output: Optional[Dict[str, Any]]
    error: Optional[str]

    # Metadata
    metadata: Dict[str, Any]
    timestamp: str


class NodeStatus(Enum):
    """Status of node execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeExecutionState:
    """State for individual node execution"""

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    input_state: Optional[Dict[str, Any]] = None
    output_state: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "input_state": self.input_state,
            "output_state": self.output_state,
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_count": self.execution_count,
        }


class LangGraphStateManager(StateManager):
    """
    Extended state manager for LangGraph workflows

    Provides LangGraph-specific state management while maintaining
    compatibility with the shared core state management.
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(initial_state)

        self.workflow_id = workflow_id or str(uuid.uuid4())
        self._node_states: Dict[str, NodeExecutionState] = {}
        self._current_step = 0
        self._current_node: Optional[str] = None

        # Initialize workflow-specific state
        self._ensure_workflow_state()

    def _ensure_workflow_state(self) -> None:
        """Ensure basic workflow state structure exists"""
        workflow_path = "$.workflow"
        if self.get(workflow_path) is None:
            workflow_data = {
                "id": self.workflow_id,
                "step_id": 0,
                "current_node": None,
                "messages": [],
                "intermediate_steps": [],
                "config": {},
                "output": None,
                "error": None,
                "metadata": {},
                "timestamp": datetime.now().isoformat(),
            }
            for key, value in workflow_data.items():
                # Use PathResolver directly since StateManager doesn't have set method
                PathResolver.set(self._state, f"{workflow_path}.{key}", value)

    def get_workflow_id(self) -> str:
        """Get the workflow ID"""
        return self.workflow_id

    def get_current_step(self) -> int:
        """Get current step ID"""
        return self._current_step

    def increment_step(self) -> int:
        """Increment step counter and return new step ID"""
        self._current_step += 1
        self.set("$.workflow.step_id", self._current_step)
        return self._current_step

    def set_current_node(self, node_id: Optional[str]) -> None:
        """Set current executing node"""
        self._current_node = node_id
        self.set("$.workflow.current_node", node_id)

    def get_current_node(self) -> Optional[str]:
        """Get current executing node"""
        return self._current_node

    def get_node_state(self, node_id: str) -> Optional[NodeExecutionState]:
        """Get execution state for a specific node"""
        return self._node_states.get(node_id)

    def set_node_state(self, node_state: NodeExecutionState) -> None:
        """Set execution state for a specific node"""
        self._node_states[node_state.node_id] = node_state

    def update_node_status(
        self, node_id: str, status: NodeStatus, error_message: Optional[str] = None
    ) -> None:
        """Update node status"""
        if node_id not in self._node_states:
            self._node_states[node_id] = NodeExecutionState(node_id=node_id)

        node_state = self._node_states[node_id]
        node_state.status = status
        node_state.error_message = error_message

        if status == NodeStatus.RUNNING:
            node_state.start_time = datetime.now()
            node_state.execution_count += 1
        elif status in [NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED]:
            node_state.end_time = datetime.now()

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the workflow state"""
        messages = self.get("$.workflow.messages") or []
        messages.append(
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                **message,
            }
        )
        self.set("$.workflow.messages", messages)

    def add_intermediate_step(self, step: Dict[str, Any]) -> None:
        """Add an intermediate step to the workflow state"""
        steps = self.get("$.workflow.intermediate_steps") or []
        steps.append(
            {
                "id": str(uuid.uuid4()),
                "step_id": self._current_step,
                "timestamp": datetime.now().isoformat(),
                **step,
            }
        )
        self.set("$.workflow.intermediate_steps", steps)

    def get_workflow_state(self) -> WorkflowState:
        """Get the current workflow state as a TypedDict"""
        return WorkflowState(
            workflow_id=self.workflow_id,
            step_id=self._current_step,
            node_id=self._current_node or "",
            messages=self.get("$.workflow.messages") or [],
            intermediate_steps=self.get("$.workflow.intermediate_steps") or [],
            config=self.get("$.workflow.config") or {},
            output=self.get("$.workflow.output"),
            error=self.get("$.workflow.error"),
            metadata=self.get("$.workflow.metadata") or {},
            timestamp=self.get("$.workflow.timestamp") or datetime.now().isoformat(),
        )

    def create_state_view(
        self,
        step_id: Optional[int] = None,
        pending_changes: Optional[List[ChangeSet]] = None,
    ) -> "LangGraphStateView":
        """Create a LangGraph-specific state view"""
        if step_id is None:
            step_id = self._current_step

        return LangGraphStateView(
            state_manager=self, step_id=step_id, pending_changes=pending_changes
        )


class LangGraphStateView(StateView):
    """
    Extended state view for LangGraph workflows

    Provides LangGraph-specific state access methods while maintaining
    compatibility with the shared core state view.
    """

    def __init__(
        self,
        state_manager: LangGraphStateManager,
        step_id: int,
        pending_changes: Optional[List[ChangeSet]] = None,
    ):
        super().__init__(state_manager, step_id, pending_changes)
        self._state_manager = state_manager

    def get_workflow_state(self) -> WorkflowState:
        """Get workflow state from this view"""
        return WorkflowState(
            workflow_id=self.read("$.workflow.id"),
            step_id=self.read("$.workflow.step_id"),
            node_id=self.read("$.workflow.current_node") or "",
            messages=self.read("$.workflow.messages") or [],
            intermediate_steps=self.read("$.workflow.intermediate_steps") or [],
            config=self.read("$.workflow.config") or {},
            output=self.read("$.workflow.output"),
            error=self.read("$.workflow.error"),
            metadata=self.read("$.workflow.metadata") or {},
            timestamp=self.read("$.workflow.timestamp") or "",
        )

    def get_messages_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Get messages relevant to a specific node"""
        messages = self.read("$.workflow.messages") or []
        # Filter messages that are relevant to this node
        return [
            msg
            for msg in messages
            if msg.get("node_id") == node_id or msg.get("target_node") == node_id
        ]

    def get_node_execution_context(self, node_id: str) -> Dict[str, Any]:
        """Get execution context for a specific node"""
        node_state = self._state_manager.get_node_state(node_id)
        return {
            "node_id": node_id,
            "step_id": self._step_id,
            "workflow_id": self.read("$.workflow.id"),
            "messages": self.get_messages_for_node(node_id),
            "config": self.read("$.workflow.config") or {},
            "previous_attempts": node_state.execution_count if node_state else 0,
            "previous_error": node_state.error_message if node_state else None,
        }
