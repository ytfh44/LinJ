
from linj_autogen.core.path import PathResolver

def test_path_resolver():
    state = {}
    print(f"Initial state: {state}")
    
    PathResolver.set(state, "$.a", "A")
    print(f"After set $.a=A: {state}")
    assert state == {"a": "A"}
    
    val = PathResolver.get(state, "$.a")
    print(f"Get $.a: {val}")
    assert val == "A"
    
    # Simulate DiagnosticTracer.export_to_state
    state["$.trace"] = {"summary": "test"}
    print(f"After state['$.trace'] literal assignment: {state}")
    
    # Simulate runner.py logic:
    # trace_data = state.get("$.trace")
    # state_manager.apply(ChangeSet.create_write("$.trace", trace_data))
    trace_data = state.get("$.trace")
    PathResolver.set(state, "$.trace", trace_data)
    print(f"After PathResolver.set($.trace, trace_data): {state}")
    
    # Check if 'a' is still there
    print(f"Is 'a' still in state? {state.get('a')}")
    assert state.get("a") == "A"
    
    print("PathResolver tests passed!")

if __name__ == "__main__":
    try:
        test_path_resolver()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
