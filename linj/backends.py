"""Backend selection with lazy loading.

This module provides a unified interface for loading different backends
(AutoGen, LangGraph) with lazy import to avoid hard dependencies.
"""

from typing import Any, Dict, Optional


def get_backend(name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Load backend by name with lazy import.

    Args:
        name: Backend name, either "autogen" or "langgraph"
        config: Optional backend configuration

    Returns:
        Backend instance

    Raises:
        ImportError: If the backend dependencies are not installed
        ValueError: If the backend name is unknown

    Example:
        >>> backend = get_backend("autogen")
        >>> result = await backend.run(doc, initial_state)
    """
    if name == "autogen":
        try:
            from linj_autogen.executor.runner import LinJExecutor

            return LinJExecutor(config)
        except ImportError as e:
            raise ImportError(
                "AutoGen backend not installed. Install with: pip install linj[autogen]"
            ) from e

    elif name == "langgraph":
        try:
            from shared.executor.langgraph_adapter import LangGraphExecutorAdapter

            return LangGraphExecutorAdapter(config)
        except ImportError as e:
            raise ImportError(
                "LangGraph backend not installed. "
                "Install with: pip install linj[langgraph]"
            ) from e

    else:
        raise ValueError(f"Unknown backend: {name}. Choose from: autogen, langgraph")


def create_backend(
    backend_type: str = "autogen",
    enable_tracing: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create backend executor of specified type.

    This is a convenience function that wraps get_backend with additional
    configuration options.

    Args:
        backend_type: Backend type, either "autogen" or "langgraph"
        enable_tracing: Whether to enable tracing (not used in all backends)
        config: Optional configuration dictionary

    Returns:
        Backend executor instance

    Raises:
        ImportError: If the backend dependencies are not installed
        ValueError: If the backend type is unknown

    Example:
        >>> backend = create_backend("langgraph", enable_tracing=True)
        >>> result = await backend.run(doc, initial_state)
    """
    backend_type = backend_type.lower()

    if backend_type == "langgraph":
        try:
            from examples.langgraph_backend import LangGraphBackend

            return LangGraphBackend(enable_tracing=enable_tracing, config=config)
        except ImportError as e:
            raise ImportError(
                "LangGraph backend not installed. "
                "Install with: pip install linj[langgraph]"
            ) from e
    elif backend_type == "autogen":
        try:
            from examples.langgraph_backend import AutoGenBackend

            return AutoGenBackend(enable_tracing=enable_tracing, config=config)
        except ImportError as e:
            raise ImportError(
                "AutoGen backend not installed. Install with: pip install linj[autogen]"
            ) from e
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. Choose from: autogen, langgraph"
        )
