import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from shared.executor.unified import execute_linj

document = {
    "linj_version": "0.1",
    "nodes": [
        {
            "id": "producer",
            "type": "hint",
            "template": "produced data",
            "write_to": "$.shared_data",
        },
        {
            "id": "consumer",
            "type": "hint",
            "template": "consumed: {{data}}",
            "vars": {"data": {"$path": "$.shared_data"}},
            "write_to": "$.result",
        },
    ],
    "edges": [{"from": "producer", "to": "consumer", "kind": "data"}],
}

print("Running with AutoGen backend...")
result_autogen = execute_linj(document, "autogen")
print(f"AutoGen Final State: {result_autogen.final_state}")

print("\nRunning with LangGraph backend...")
result_langgraph = execute_linj(document, "langgraph")
print(f"LangGraph Final State: {result_langgraph.final_state}")
