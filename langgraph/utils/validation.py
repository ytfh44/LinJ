"""
Validation Utilities

This module provides validation functions for LangGraph workflows.
"""

from typing import Any, Dict, List, Optional, Union

from ..types import ExecutionContext, NodeConfig, NodeType
from ..state import NodeStatus


def validate_state_transition(
    from_state: Dict[str, Any],
    to_state: Dict[str, Any],
    allowed_changes: Optional[List[str]] = None,
) -> bool:
    """
    Validate that a state transition is allowed

    Args:
        from_state: Current state
        to_state: Target state
        allowed_changes: List of allowed field paths (optional)

    Returns:
        True if transition is valid, False otherwise
    """
    if allowed_changes is None:
        return True  # Allow all changes if no restrictions

    # Create a simple diff to check what changed
    changed_fields = []

    for key in to_state:
        if key not in from_state or from_state[key] != to_state[key]:
            changed_fields.append(key)

    # Check if all changed fields are allowed
    for field in changed_fields:
        if field not in allowed_changes:
            return False

    return True


def validate_node_input(
    input_data: Dict[str, Any], node_config: NodeConfig, strict: bool = False
) -> List[str]:
    """
    Validate input data for a node

    Args:
        input_data: Input data to validate
        node_config: Node configuration
        strict: If True, reject unknown fields

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required inputs
    for input_name, state_path in node_config.inputs.items():
        if input_name not in input_data:
            errors.append(f"Missing required input: {input_name}")
        elif input_data[input_name] is None:
            errors.append(f"Required input cannot be None: {input_name}")

    # Validate input types based on node type
    type_errors = _validate_input_types(input_data, node_config)
    errors.extend(type_errors)

    # Check for unknown inputs if strict mode
    if strict:
        known_inputs = set(node_config.inputs.keys())
        provided_inputs = set(input_data.keys())
        unknown_inputs = provided_inputs - known_inputs

        for unknown in unknown_inputs:
            errors.append(f"Unknown input field: {unknown}")

    return errors


def _validate_input_types(
    input_data: Dict[str, Any], node_config: NodeConfig
) -> List[str]:
    """
    Validate input data types based on node type

    Args:
        input_data: Input data to validate
        node_config: Node configuration

    Returns:
        List of type validation errors
    """
    errors = []

    # Different node types have different input requirements
    if node_config.node_type == NodeType.DECISION:
        # Decision nodes should have a condition
        if "condition" not in input_data:
            errors.append("Decision node requires 'condition' input")

    elif node_config.node_type == NodeType.ACTION:
        # Action nodes should have an action to perform
        if "action" not in input_data:
            errors.append("Action node requires 'action' input")

    elif node_config.node_type == NodeType.CONDITION:
        # Condition nodes should have a condition to evaluate
        if "condition" not in input_data:
            errors.append("Condition node requires 'condition' input")
        elif not isinstance(input_data.get("condition"), (str, dict)):
            errors.append("Condition must be a string expression or dict")

    elif node_config.node_type == NodeType.TRANSFORM:
        # Transform nodes should have data and transformation
        if "data" not in input_data:
            errors.append("Transform node requires 'data' input")
        if "transformation" not in input_data:
            errors.append("Transform node requires 'transformation' input")

    return errors


def validate_node_config(node_config: NodeConfig) -> List[str]:
    """
    Validate node configuration

    Args:
        node_config: Node configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if not node_config.node_id:
        errors.append("Node ID is required")

    if not node_config.node_type:
        errors.append("Node type is required")

    # Validate node_id format
    if node_config.node_id and not isinstance(node_config.node_id, str):
        errors.append("Node ID must be a string")

    # Validate timeout
    if node_config.timeout is not None and node_config.timeout <= 0:
        errors.append("Timeout must be positive")

    # Validate retry settings
    if node_config.max_retries < 0:
        errors.append("Max retries cannot be negative")

    if node_config.retry_delay <= 0:
        errors.append("Retry delay must be positive")

    if node_config.retry_backoff_factor <= 0:
        errors.append("Retry backoff factor must be positive")

    # Validate node type specific requirements
    type_errors = _validate_node_type_requirements(node_config)
    errors.extend(type_errors)

    return errors


def _validate_node_type_requirements(node_config: NodeConfig) -> List[str]:
    """
    Validate node type specific requirements

    Args:
        node_config: Node configuration

    Returns:
        List of type-specific validation errors
    """
    errors = []

    if node_config.node_type == NodeType.DECISION:
        # Decision nodes should have outputs for different paths
        if not node_config.outputs:
            errors.append("Decision node requires outputs for decision paths")

    elif node_config.node_type == NodeType.SPLITTER:
        # Splitter nodes should have multiple outputs
        if len(node_config.outputs) < 2:
            errors.append("Splitter node requires at least 2 outputs")

    elif node_config.node_type == NodeType.JOINER:
        # Joiner nodes should have multiple dependencies
        if len(node_config.dependencies) < 2:
            errors.append("Joiner node requires at least 2 dependencies")

    elif node_config.node_type == NodeType.AGGREGATOR:
        # Aggregator nodes should have dependencies and aggregation logic
        if not node_config.dependencies:
            errors.append("Aggregator node requires dependencies to aggregate")
        if "aggregation_method" not in node_config.metadata:
            errors.append("Aggregator node requires aggregation_method in metadata")

    return errors


def validate_execution_context(context: ExecutionContext) -> List[str]:
    """
    Validate execution context

    Args:
        context: Execution context to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if not context.workflow_id:
        errors.append("Workflow ID is required")

    if not context.node_id:
        errors.append("Node ID is required")

    if context.step_id is None or context.step_id < 0:
        errors.append("Valid step ID is required")

    if not context.node_config:
        errors.append("Node configuration is required")

    if not context.workflow_config:
        errors.append("Workflow configuration is required")

    if not context.state_view:
        errors.append("State view is required")

    # Validate consistency
    if context.node_config and context.node_id != context.node_config.node_id:
        errors.append("Context node ID doesn't match node config node ID")

    return errors


def validate_workflow_config(workflow_config) -> List[str]:
    """
    Validate workflow configuration

    Args:
        workflow_config: Workflow configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if not workflow_config.workflow_id:
        errors.append("Workflow ID is required")

    if not workflow_config.name:
        errors.append("Workflow name is required")

    # Validate nodes
    if workflow_config.nodes:
        for node_id, node_config in workflow_config.nodes.items():
            node_errors = validate_node_config(node_config)
            for error in node_errors:
                errors.append(f"Node {node_id}: {error}")

    # Validate edges reference valid nodes
    if workflow_config.edges and workflow_config.nodes:
        node_ids = set(workflow_config.nodes.keys())
        for edge in workflow_config.edges:
            if isinstance(edge, dict):
                source = edge.get("source")
                target = edge.get("target")

                if source and source not in node_ids:
                    errors.append(f"Edge references unknown source node: {source}")

                if target and target not in node_ids:
                    errors.append(f"Edge references unknown target node: {target}")

    return errors
