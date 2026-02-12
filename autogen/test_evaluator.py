
from linj_autogen.executor.evaluator import evaluate_condition

def test_evaluator():
    state = {'signal': {'payload': 'expected'}}
    predicate = 'value("$.signal.payload") == "expected"'
    
    print(f"State: {state}")
    print(f"Predicate: {predicate}")
    
    try:
        result = evaluate_condition(predicate, state)
        print(f"Result: {result}")
        assert result is True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluator()
