"""
Backward Compatibility Adapter

Ensures all existing AutoGen APIs continue to work with the new shared component architecture.
This module provides adapter functions and compatibility layers.
"""

from typing import Any, Dict, Optional, Union
import warnings


class BackwardCompatibilityAdapter:
    """
    Adapter to maintain backward compatibility with existing AutoGen APIs
    while internally using the new shared component architecture.
    """

    def __init__(self):
        self._deprecation_warnings = {}

    def warn_deprecation(self, old_api: str, new_api: str, module: str):
        """Emit deprecation warning with guidance"""
        if old_api not in self._deprecation_warnings:
            warnings.warn(
                f"Deprecated API: {old_api} in {module}. "
                f"Please use {new_api} instead. "
                f"This compatibility layer will be removed in a future version.",
                DeprecationWarning,
                stacklevel=3,
            )
            self._deprecation_warnings[old_api] = True

    def adapt_legacy_document_load(self, path: str):
        """Adapt legacy document loading to use shared components"""
        from .core.compat import LinJDocument, load_document

        self.warn_deprecation(
            "linj_autogen.core.document.load_document",
            "shared.core.document.load_document or autogen.core.compat.load_document",
            "document loading",
        )
        return load_document(path)

    def adapt_legacy_executor_creation(self, **kwargs):
        """Adapt legacy executor creation to use new backend"""
        from .backend import AutoGenExecutionBackend

        self.warn_deprecation(
            "linj_autogen.executor.runner.LinJExecutor",
            "autogen.autogen.backend.AutoGenExecutionBackend",
            "executor creation",
        )

        # Map legacy parameters to new backend
        enable_contitext = kwargs.pop("enable_tracing", True)
        backend = AutoGenExecutionBackend(enable_contitext=enable_contitext)

        # Register tools if provided
        tools = kwargs.get("tool_registry", {})
        for name, tool in tools.items():
            backend.register_tool(name, tool)

        return backend

    def adapt_legacy_agent_creation(self, document, **kwargs):
        """Adapt legacy agent creation to use new architecture"""
        from .agent import LinJAgent

        self.warn_deprecation(
            "direct LinJAgent instantiation with legacy parameters",
            "LinJAgent with shared components and AutoGenExecutionBackend",
            "agent creation",
        )

        # Create agent with shared components
        agent = LinJAgent(document, **kwargs)
        return agent

    def get_compatibility_layer_info(self) -> Dict[str, Any]:
        """Get information about compatibility layer status"""
        return {
            "deprecated_apis_count": len(self._deprecation_warnings),
            "deprecated_apis": list(self._deprecation_warnings.keys()),
            "shared_components_available": self._check_shared_availability(),
            "migration_status": "partial"
            if len(self._deprecation_warnings) > 0
            else "complete",
        }

    def _check_shared_availability(self) -> bool:
        """Check if shared components are available"""
        try:
            import shared.core.document
            import shared.core.nodes
            import shared.contitext.engine

            return True
        except ImportError:
            return False


# Global compatibility adapter instance
_compat_adapter = BackwardCompatibilityAdapter()


def get_compatibility_info() -> Dict[str, Any]:
    """Get compatibility information"""
    return _compat_adapter.get_compatibility_layer_info()


def adapt_document_load(path: str):
    """Public function to adapt document loading"""
    return _compat_adapter.adapt_legacy_document_load(path)


def adapt_executor_creation(**kwargs):
    """Public function to adapt executor creation"""
    return _compat_adapter.adapt_legacy_executor_creation(**kwargs)


def adapt_agent_creation(document, **kwargs):
    """Public function to adapt agent creation"""
    return _compat_adapter.adapt_legacy_agent_creation(document, **kwargs)
