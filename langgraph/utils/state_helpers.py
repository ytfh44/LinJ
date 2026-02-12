"""
State Helper Utilities

This module provides helper functions for state management in LangGraph workflows.
"""

from typing import Any, Dict, List, Optional, Union
import copy

from ..state import LangGraphStateView, NodeStatus


class StateHelper:
    """
    Helper class for state operations in LangGraph workflows

    Provides utility methods for common state manipulation and validation tasks.
    """

    @staticmethod
    def merge_states(
        base_state: Dict[str, Any], update_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two state dictionaries

        Args:
            base_state: Base state dictionary
            update_state: Updates to apply

        Returns:
            Merged state dictionary
        """
        result = copy.deepcopy(base_state)
        StateHelper._deep_merge(result, update_state)
        return result

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively deep merge update into base"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                StateHelper._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    @staticmethod
    def get_nested_value(state: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Get value from nested dictionary using dot notation

        Args:
            state: State dictionary
            path: Dot-separated path (e.g., "user.profile.name")
            default: Default value if path not found

        Returns:
            Value at path or default
        """
        keys = path.split(".")
        current = state

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    @staticmethod
    def set_nested_value(state: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set value in nested dictionary using dot notation

        Args:
            state: State dictionary to modify
            path: Dot-separated path (e.g., "user.profile.name")
            value: Value to set
        """
        keys = path.split(".")
        current = state

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    @staticmethod
    def validate_state_structure(
        state: Dict[str, Any], required_paths: List[str]
    ) -> List[str]:
        """
        Validate that required paths exist in state

        Args:
            state: State dictionary to validate
            required_paths: List of required dot-separated paths

        Returns:
            List of missing paths (empty if all valid)
        """
        missing_paths = []

        for path in required_paths:
            if StateHelper.get_nested_value(state, path) is None:
                missing_paths.append(path)

        return missing_paths

    @staticmethod
    def create_state_diff(
        old_state: Dict[str, Any], new_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create diff between two state dictionaries

        Args:
            old_state: Original state
            new_state: Updated state

        Returns:
            Dictionary representing the changes
        """
        diff = {}

        # Find changed and added keys
        all_keys = set(old_state.keys()) | set(new_state.keys())

        for key in all_keys:
            old_value = old_state.get(key)
            new_value = new_state.get(key)

            if old_value != new_value:
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    nested_diff = StateHelper.create_state_diff(old_value, new_value)
                    if nested_diff:
                        diff[key] = nested_diff
                else:
                    diff[key] = {
                        "old": old_value,
                        "new": new_value,
                        "action": "changed" if key in old_state else "added",
                    }

        # Find removed keys
        for key in set(old_state.keys()) - set(new_state.keys()):
            diff[key] = {"old": old_state[key], "new": None, "action": "removed"}

        return diff

    @staticmethod
    def apply_state_diff(state: Dict[str, Any], diff: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a diff to a state dictionary

        Args:
            state: Original state
            diff: Diff to apply

        Returns:
            Updated state dictionary
        """
        result = copy.deepcopy(state)

        for key, change in diff.items():
            if isinstance(change, dict) and "action" in change:
                # Simple change
                if change["action"] == "removed":
                    result.pop(key, None)
                else:
                    result[key] = change["new"]
            else:
                # Nested diff
                if key in result and isinstance(result[key], dict):
                    result[key] = StateHelper.apply_state_diff(result[key], change)
                else:
                    result[key] = change

        return result

    @staticmethod
    def get_node_state_summary(state_view: LangGraphStateView) -> Dict[str, Any]:
        """
        Get a summary of node states from the state view

        Args:
            state_view: State view to query

        Returns:
            Summary of node states
        """
        try:
            workflow_state = state_view.get_workflow_state()

            # Count nodes by status
            status_counts = {}
            total_nodes = 0

            # Check if we have node status information in state
            node_statuses = state_view.read("$.node_statuses") or {}

            for node_id, status in node_statuses.items():
                status_str = status.value if hasattr(status, "value") else str(status)
                status_counts[status_str] = status_counts.get(status_str, 0) + 1
                total_nodes += 1

            return {
                "total_nodes": total_nodes,
                "status_counts": status_counts,
                "current_step": workflow_state.get("step_id", 0),
                "current_node": workflow_state.get("node_id"),
                "workflow_id": workflow_state.get("workflow_id"),
            }

        except Exception as e:
            return {
                "error": f"Failed to get node state summary: {e}",
                "total_nodes": 0,
                "status_counts": {},
            }

    @staticmethod
    def is_workflow_complete(state_view: LangGraphStateView, total_nodes: int) -> bool:
        """
        Check if workflow is complete

        Args:
            state_view: State view to check
            total_nodes: Total number of nodes in workflow

        Returns:
            True if workflow is complete, False otherwise
        """
        try:
            node_statuses = state_view.read("$.node_statuses") or {}

            completed_count = 0
            failed_count = 0

            for status in node_statuses.values():
                status_str = status.value if hasattr(status, "value") else str(status)
                if status_str == NodeStatus.COMPLETED.value:
                    completed_count += 1
                elif status_str == NodeStatus.FAILED.value:
                    failed_count += 1

            # Workflow is complete if all nodes are either completed or failed
            total_processed = completed_count + failed_count
            return total_processed >= total_nodes

        except Exception:
            return False

    @staticmethod
    def get_failed_nodes(state_view: LangGraphStateView) -> List[str]:
        """
        Get list of failed node IDs

        Args:
            state_view: State view to check

        Returns:
            List of failed node IDs
        """
        try:
            node_statuses = state_view.read("$.node_statuses") or {}

            failed_nodes = []
            for node_id, status in node_statuses.items():
                status_str = status.value if hasattr(status, "value") else str(status)
                if status_str == NodeStatus.FAILED.value:
                    failed_nodes.append(node_id)

            return failed_nodes

        except Exception:
            return []
