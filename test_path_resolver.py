from shared.core.path import PathResolver

state = {"shared_data": "produced data"}
path = "$.shared_data"

print(f"Testing PathResolver.get(state, '{path}')...")
try:
    segments = PathResolver.parse(path)
    print(f"Parsed segments: {[s.key for s in segments]}")
    val = PathResolver.get(state, path)
    print(f"Result: {val}")
except Exception as e:
    print(f"Error: {e}")

path2 = "$.a.b"
state2 = {"a": {"b": 123}}
print(f"\nTesting PathResolver.get(state2, '{path2}')...")
val2 = PathResolver.get(state2, path2)
print(f"Result: {val2}")

path3 = "$"
print(f"\nTesting PathResolver.get(state, '{path3}')...")
val3 = PathResolver.get(state, path3)
print(f"Result: {val3}")
