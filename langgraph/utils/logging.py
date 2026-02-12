"""
Logging Utilities

This module provides logging utilities for LangGraph workflows.
"""

import sys
from typing import Any, Optional, Dict
import logging.config


def get_logger(name: str) -> Any:
    """
    Get a logger for the specified name

    Args:
        name: Logger name (usually module or component name)

    Returns:
        Logger instance
    """
    try:
        # Try to use structlog if available (preferred)
        import structlog

        # Configure structlog if not already configured
        if not structlog.is_configured():
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

        return structlog.get_logger(name)

    except ImportError:
        # Fallback to standard library logging
        return _get_standard_logger(name)


def _get_standard_logger(name: str) -> logging.Logger:
    """
    Get a standard library logger

    Args:
        name: Logger name

    Returns:
        Standard library logger
    """
    logger = logging.getLogger(name)

    # Configure basic logging if not already configured
    if not logging.getLogger().handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    return logger


def configure_logging(
    level: str = "INFO", format_string: Optional[str] = None, use_structlog: bool = True
) -> None:
    """
    Configure logging for LangGraph components

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (for standard logging)
        use_structlog: Whether to use structlog if available
    """
    if use_structlog:
        try:
            import structlog

            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
            ]

            # Add console renderer
            processors.append(structlog.dev.ConsoleRenderer())

            structlog.configure(
                processors=processors,
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

            # Set logging level for standard library logging
            logging.basicConfig(
                level=getattr(logging, level.upper()),
                format=format_string
                or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            return

        except ImportError:
            pass  # Fall back to standard logging

    # Standard logging configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def log_execution_start(logger: Any, node_id: str, context: Dict[str, Any]) -> None:
    """
    Log the start of node execution

    Args:
        logger: Logger instance
        node_id: ID of the node being executed
        context: Execution context information
    """
    if hasattr(logger, "info"):
        # Structlog or similar logger
        logger.info(
            "Starting node execution",
            node_id=node_id,
            workflow_id=context.get("workflow_id"),
            step_id=context.get("step_id"),
            node_type=context.get("node_type"),
        )
    else:
        # Standard logging logger
        logger.info(
            f"Starting node execution: {node_id} "
            f"(workflow: {context.get('workflow_id')}, "
            f"step: {context.get('step_id')}, "
            f"type: {context.get('node_type')})"
        )


def log_execution_end(
    logger: Any,
    node_id: str,
    success: bool,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log the end of node execution

    Args:
        logger: Logger instance
        node_id: ID of the node that was executed
        success: Whether execution was successful
        duration_ms: Execution duration in milliseconds
        error: Error message if execution failed
    """
    if hasattr(logger, "info"):
        # Structlog or similar logger
        if success:
            logger.info(
                "Node execution completed successfully",
                node_id=node_id,
                duration_ms=duration_ms,
            )
        else:
            logger.error(
                "Node execution failed",
                node_id=node_id,
                duration_ms=duration_ms,
                error=error,
            )
    else:
        # Standard logging logger
        status = "completed successfully" if success else "failed"
        duration_str = f" in {duration_ms:.2f}ms" if duration_ms else ""
        error_str = f" - Error: {error}" if error else ""

        level = logger.info if success else logger.error
        level(f"Node {node_id} execution {status}{duration_str}{error_str}")


def log_workflow_event(
    logger: Any,
    workflow_id: str,
    event_type: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a workflow-level event

    Args:
        logger: Logger instance
        workflow_id: ID of the workflow
        event_type: Type of event (e.g., "started", "completed", "failed")
        details: Additional event details
    """
    if hasattr(logger, "info"):
        # Structlog or similar logger
        logger.info(
            f"Workflow {event_type}",
            workflow_id=workflow_id,
            event_type=event_type,
            **(details or {}),
        )
    else:
        # Standard logging logger
        details_str = f" - {details}" if details else ""
        logger.info(f"Workflow {workflow_id} {event_type}{details_str}")


def create_workflow_logger(workflow_id: str) -> Any:
    """
    Create a logger specific to a workflow

    Args:
        workflow_id: ID of the workflow

    Returns:
        Workflow-specific logger
    """
    logger_name = f"workflow.{workflow_id}"
    logger = get_logger(logger_name)

    # Bind workflow context if using structlog
    if hasattr(logger, "bind"):
        logger = logger.bind(workflow_id=workflow_id)

    return logger


def create_node_logger(workflow_id: str, node_id: str) -> Any:
    """
    Create a logger specific to a node within a workflow

    Args:
        workflow_id: ID of the workflow
        node_id: ID of the node

    Returns:
        Node-specific logger
    """
    logger_name = f"workflow.{workflow_id}.node.{node_id}"
    logger = get_logger(logger_name)

    # Bind context if using structlog
    if hasattr(logger, "bind"):
        logger = logger.bind(
            workflow_id=workflow_id,
            node_id=node_id,
        )

    return logger
