"""
Backward compatibility layer for autogen contitext components

This module re-exports shared contitext components to maintain backward compatibility.
Existing code can continue to import from here while we migrate to shared components.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    # Import from shared components
    from shared.contitext.engine import ContiTextEngine
    from shared.contitext.mapper import LinJToContiTextMapper, ParallelLinJExecutor
    from shared.contitext.commit_manager import CommitManager
    from shared.contitext.continuation import Continuation, Status, ContinuationView
    from shared.contitext.signal import Signal, WaitCondition, SignalQueue

    # Re-export for backward compatibility
    __all__ = [
        "ContiTextEngine",
        "LinJToContiTextMapper",
        "ParallelLinJExecutor",
        "CommitManager",
        "Continuation",
        "Status",
        "ContinuationView",
        "Signal",
        "WaitCondition",
        "SignalQueue",
    ]

except ImportError as e:
    print(f"Warning: Could not import shared contitext components: {e}")
    # Fallback: keep local imports available
    from .engine import ContiTextEngine
    from .mapper import LinJToContiTextMapper, ParallelLinJExecutor
    from .commit_manager import CommitManager
    from .continuation import Continuation, Status, ContinuationView
    from .signal import Signal, WaitCondition, SignalQueue

    __all__ = [
        "ContiTextEngine",
        "LinJToContiTextMapper",
        "ParallelLinJExecutor",
        "CommitManager",
        "Continuation",
        "Status",
        "ContinuationView",
        "Signal",
        "WaitCondition",
        "SignalQueue",
    ]
