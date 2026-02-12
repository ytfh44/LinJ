
from typing import Any, Optional
from pydantic import BaseModel, Field

class ValueRef(BaseModel):
    path: Optional[str] = Field(None, alias="$path")
    const: Optional[Any] = Field(None, alias="$const")

def test_pydantic():
    # Test with field name
    v1 = ValueRef(const="A")
    print(f"v1: {v1}")
    print(f"v1.const: {v1.const}")
    assert v1.const == "A"
    
    # Test with alias
    v2 = ValueRef(**{"$const": "B"})
    print(f"v2: {v2}")
    print(f"v2.const: {v2.const}")
    assert v2.const == "B"
    
    # Test from_value style
    def from_value(val):
        return ValueRef(const=val)
    
    v3 = from_value("C")
    print(f"v3: {v3}")
    print(f"v3.const: {v3.const}")
    assert v3.const == "C"

if __name__ == "__main__":
    test_pydantic()
