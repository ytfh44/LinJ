"""
Research and Writing Pipeline Demo

Demonstrates how to create complex workflows using the LinJ framework:
1. Research phase (ToolNode)
2. Draft generation (HintNode + ToolNode)
3. Audit phase (ToolNode)
4. Quality gate (GateNode)
5. Optimization loop (Loop)
6. Terminology check (JoinNode)
7. Publishing phase (ToolNode)

Supports two backends:
- AutoGen: uses linj_autogen.executor.runner
- LangGraph: uses langgraph module
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Ensure autogen/src is in path
autogen_src = project_root / "autogen" / "src"
if str(autogen_src) not in sys.path:
    sys.path.insert(0, str(autogen_src))

# Import tool functions
from tools import research_topic, audit_article, generic_llm, publish_result


def get_backend(backend_type: str = "autogen"):
    """
    Get backend executor of specified type

    Args:
        backend_type: "autogen" or "langgraph"

    Returns:
        Backend object
    """
    if backend_type.lower() == "langgraph":
        try:
            from examples.langgraph_backend import LangGraphBackend

            return LangGraphBackend(enable_tracing=True)
        except ImportError:
            print("Warning: LangGraph backend not available, falling back to AutoGen")
            return get_backend("autogen")
    else:
        from linj_autogen.executor.runner import LinJExecutor

        return LinJExecutor(enable_tracing=True)


def get_load_document_func(backend_type: str):
    """Get load_document function suitable for current backend"""
    if backend_type.lower() == "langgraph":
        try:
            from examples.langgraph_backend import load_document

            return load_document
        except ImportError:
            pass

    from linj_autogen.executor.runner import load_document

    return load_document


async def main(backend_type: str = "autogen"):
    """
    Main function

    Args:
        backend_type: Backend type to use
    """
    print("=== LinJ Complex Demo: Automated Research and Refinement Pipeline ===")
    print(f"Using backend: {backend_type}")
    print(f"Project root: {project_root}")

    # 1. Load document
    yaml_path = current_dir / "workflow.yaml"
    load_document = get_load_document_func(backend_type)
    doc = load_document(str(yaml_path))

    # 2. Initialize executor
    executor = get_backend(backend_type)

    # 3. Register tools
    executor.register_tool("research_topic", research_topic)
    executor.register_tool("audit_article", audit_article)
    executor.register_tool("llm_call", generic_llm)
    executor.register_tool("publish_result", publish_result)

    # 4. Initial state
    initial_state = {
        "input": {"topic": "The Future of Artificial Gravity in Space Exploration"}
    }

    # 5. Run
    print(f"\n>> Starting execution for topic: {initial_state['input']['topic']}")
    try:
        final_state = await executor.run(doc, initial_state)
        print("\n>> Execution Completed Successfully!")

        # Display results
        score = final_state.get("audit_report", {}).get("score", "N/A")
        print(f">> Final Audit Score: {score}")
        publish_url = final_state.get("publish_result", {}).get("publish_url", "N/A")
        print(f">> Publish URL: {publish_url}")

        # Display final content preview
        final_content = final_state.get("final_content", "")
        print(
            f"\n>> Final Content Preview (first 200 chars):\n{final_content[:200]}..."
        )

    except Exception as e:
        print(f"\n!! Execution Failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Research and Writing Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --backend autogen      # Use AutoGen backend
  python main.py --backend langgraph    # Use LangGraph backend
  python main.py --backend all          # Run both backends sequentially
        """,
    )

    parser.add_argument(
        "--backend",
        choices=["autogen", "langgraph", "all"],
        default="autogen",
        help="Select backend type (default: autogen)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.backend == "all":
        for backend in ["autogen", "langgraph"]:
            print(f"\n\n{'#' * 60}")
            print(f"# Running with {backend.upper()} backend")
            print(f"{'#' * 60}")
            asyncio.run(main(backend_type=backend))
    else:
        asyncio.run(main(backend_type=args.backend))
