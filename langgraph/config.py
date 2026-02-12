"""
LangGraph Configuration

This module provides configuration management for LangGraph workflows.
Includes default configurations, environment-based settings, and validation.
"""

import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

from .types import WorkflowConfig, NodeConfig, RetryPolicy, NodeType


@dataclass
class LangGraphConfig:
    """Main configuration class for LangGraph components"""

    # Core settings
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # State management
    state_backend: str = "memory"  # "memory", "file", "redis", "database"
    state_persistence_path: Optional[str] = None
    state_ttl_seconds: Optional[int] = None

    # Execution settings
    max_concurrent_workflows: int = 10
    default_timeout_seconds: float = 300.0  # 5 minutes
    checkpoint_interval: int = 10

    # Retry settings
    default_max_retries: int = 3
    default_retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    default_retry_delay: float = 1.0
    default_retry_backoff_factor: float = 2.0
    max_retry_delay: float = 60.0

    # Memory and performance
    max_state_size_mb: float = 100.0
    enable_state_compression: bool = False
    garbage_collection_interval: int = 100

    # Monitoring and tracing
    enable_tracing: bool = True
    tracing_backend: str = "console"  # "console", "file", "jaeger", "opentelemetry"
    tracing_endpoint: Optional[str] = None
    metrics_collection: bool = True

    # Security
    enable_authentication: bool = False
    api_key_required: bool = False
    allowed_node_types: List[NodeType] = field(default_factory=lambda: list(NodeType))

    # External services
    external_services: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)


def get_default_config() -> LangGraphConfig:
    """Get default LangGraph configuration"""
    return LangGraphConfig()


def load_config_from_file(config_path: Union[str, Path]) -> LangGraphConfig:
    """
    Load configuration from a JSON or YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        LangGraphConfig instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    return _config_from_dict(data)


def load_config_from_env() -> LangGraphConfig:
    """
    Load configuration from environment variables

    Environment variables should be prefixed with LANGGRAPH_
    For example: LANGGRAPH_DEBUG=true, LANGGRAPH_LOG_LEVEL=DEBUG

    Returns:
        LangGraphConfig instance
    """
    config = LangGraphConfig()

    # Environment variable mappings
    env_mappings = {
        "LANGGRAPH_DEBUG": ("debug", lambda x: x.lower() in ["true", "1", "yes"]),
        "LANGGRAPH_LOG_LEVEL": ("log_level", str),
        "LANGGRAPH_LOG_FORMAT": ("log_format", str),
        "LANGGRAPH_STATE_BACKEND": ("state_backend", str),
        "LANGGRAPH_STATE_PERSISTENCE_PATH": ("state_persistence_path", str),
        "LANGGRAPH_STATE_TTL_SECONDS": ("state_ttl_seconds", int),
        "LANGGRAPH_MAX_CONCURRENT_WORKFLOWS": ("max_concurrent_workflows", int),
        "LANGGRAPH_DEFAULT_TIMEOUT_SECONDS": ("default_timeout_seconds", float),
        "LANGGRAPH_CHECKPOINT_INTERVAL": ("checkpoint_interval", int),
        "LANGGRAPH_DEFAULT_MAX_RETRIES": ("default_max_retries", int),
        "LANGGRAPH_DEFAULT_RETRY_POLICY": ("default_retry_policy", RetryPolicy),
        "LANGGRAPH_DEFAULT_RETRY_DELAY": ("default_retry_delay", float),
        "LANGGRAPH_DEFAULT_RETRY_BACKOFF_FACTOR": (
            "default_retry_backoff_factor",
            float,
        ),
        "LANGGRAPH_MAX_RETRY_DELAY": ("max_retry_delay", float),
        "LANGGRAPH_MAX_STATE_SIZE_MB": ("max_state_size_mb", float),
        "LANGGRAPH_ENABLE_STATE_COMPRESSION": (
            "enable_state_compression",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "LANGGRAPH_GARBAGE_COLLECTION_INTERVAL": ("garbage_collection_interval", int),
        "LANGGRAPH_ENABLE_TRACING": (
            "enable_tracing",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "LANGGRAPH_TRACING_BACKEND": ("tracing_backend", str),
        "LANGGRAPH_TRACING_ENDPOINT": ("tracing_endpoint", str),
        "LANGGRAPH_METRICS_COLLECTION": (
            "metrics_collection",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "LANGGRAPH_ENABLE_AUTHENTICATION": (
            "enable_authentication",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "LANGGRAPH_API_KEY_REQUIRED": (
            "api_key_required",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
    }

    for env_var, (attr_name, converter) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                converted_value = converter(value)
                setattr(config, attr_name, converted_value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {env_var}: {value}. Error: {e}")

    return config


def merge_configs(
    base_config: LangGraphConfig, override_config: Dict[str, Any]
) -> LangGraphConfig:
    """
    Merge configuration dictionaries into a LangGraphConfig instance

    Args:
        base_config: Base configuration
        override_config: Override values as dictionary

    Returns:
        Merged LangGraphConfig instance
    """
    config_dict = _config_to_dict(base_config)
    merged_dict = _deep_merge(config_dict, override_config)
    return _config_from_dict(merged_dict)


def validate_config(config: LangGraphConfig) -> List[str]:
    """
    Validate configuration and return list of issues

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Validate numeric values
    if config.max_concurrent_workflows <= 0:
        issues.append("max_concurrent_workflows must be positive")

    if config.default_timeout_seconds <= 0:
        issues.append("default_timeout_seconds must be positive")

    if config.checkpoint_interval <= 0:
        issues.append("checkpoint_interval must be positive")

    if config.default_max_retries < 0:
        issues.append("default_max_retries cannot be negative")

    if config.default_retry_delay <= 0:
        issues.append("default_retry_delay must be positive")

    if config.default_retry_backoff_factor <= 0:
        issues.append("default_retry_backoff_factor must be positive")

    if config.max_retry_delay <= 0:
        issues.append("max_retry_delay must be positive")

    if config.max_state_size_mb <= 0:
        issues.append("max_state_size_mb must be positive")

    if config.garbage_collection_interval <= 0:
        issues.append("garbage_collection_interval must be positive")

    # Validate state configuration
    if config.state_backend not in ["memory", "file", "redis", "database"]:
        issues.append(f"Invalid state_backend: {config.state_backend}")

    if config.state_backend == "file" and not config.state_persistence_path:
        issues.append("state_persistence_path is required when using file backend")

    # Validate tracing configuration
    if config.tracing_backend not in ["console", "file", "jaeger", "opentelemetry"]:
        issues.append(f"Invalid tracing_backend: {config.tracing_backend}")

    if (
        config.tracing_backend in ["jaeger", "opentelemetry"]
        and not config.tracing_endpoint
    ):
        issues.append(
            f"tracing_endpoint is required when using {config.tracing_backend}"
        )

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.log_level.upper() not in valid_log_levels:
        issues.append(
            f"Invalid log_level: {config.log_level}. Must be one of {valid_log_levels}"
        )

    return issues


def _config_to_dict(config: LangGraphConfig) -> Dict[str, Any]:
    """Convert LangGraphConfig to dictionary"""
    result = {}
    for field_name, field_def in config.__dataclass_fields__.items():
        value = getattr(config, field_name)
        if hasattr(value, "value"):  # Handle Enum values
            result[field_name] = value.value
        elif (
            isinstance(value, list) and value and hasattr(value[0], "value")
        ):  # Handle List[Enum]
            result[field_name] = [item.value for item in value]
        else:
            result[field_name] = value
    return result


def _config_from_dict(data: Dict[str, Any]) -> LangGraphConfig:
    """Create LangGraphConfig from dictionary"""
    # Handle Enum conversions
    if "default_retry_policy" in data:
        data["default_retry_policy"] = RetryPolicy(data["default_retry_policy"])

    if "allowed_node_types" in data:
        data["allowed_node_types"] = [
            NodeType(item) for item in data["allowed_node_types"]
        ]

    return LangGraphConfig(**data)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def create_workflow_config(
    workflow_id: str, name: str, description: Optional[str] = None, **kwargs
) -> WorkflowConfig:
    """
    Create a WorkflowConfig with sensible defaults

    Args:
        workflow_id: Unique identifier for the workflow
        name: Human-readable name
        description: Optional description
        **kwargs: Additional configuration overrides

    Returns:
        WorkflowConfig instance
    """
    config = WorkflowConfig(
        workflow_id=workflow_id,
        name=name,
        description=description,
        max_steps=1000,
        timeout=3600.0,  # 1 hour
        parallel_execution=False,
        state_persistence=True,
        checkpoint_interval=10,
        continue_on_node_failure=False,
        failure_handling="stop",
        log_level="INFO",
        enable_tracing=True,
        metrics_collection=True,
    )

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def create_node_config(
    node_id: str, node_type: NodeType, description: Optional[str] = None, **kwargs
) -> NodeConfig:
    """
    Create a NodeConfig with sensible defaults

    Args:
        node_id: Unique identifier for the node
        node_type: Type of the node
        description: Optional description
        **kwargs: Additional configuration overrides

    Returns:
        NodeConfig instance
    """
    config = NodeConfig(
        node_id=node_id,
        node_type=node_type,
        description=description,
        timeout=300.0,  # 5 minutes
        max_retries=3,
        retry_policy=RetryPolicy.EXPONENTIAL_BACKOFF,
        retry_delay=1.0,
        retry_backoff_factor=2.0,
        skip_on_condition_false=True,
        dependencies=[],
        inputs={},
        outputs={},
        persist_output=True,
        tags=[],
        metadata={},
    )

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
