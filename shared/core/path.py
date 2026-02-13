"""
State Path Resolver

Implements path syntax and read/write semantics defined in LinJ specification sections 5.1-5.4:
- Root is $
- . accesses object fields
- [n] accesses array indices
- Automatically creates intermediate objects/arrays on write
"""

import re
from typing import Any, List, Union

from ..exceptions.errors import MappingError


class PathSegment:
    """Path segment"""

    def __init__(self, key: Union[str, int]):
        self.key = key

    def is_array_index(self) -> bool:
        return isinstance(self.key, int)

    def get_key(self) -> Union[str, int]:
        """Get key value, type-safe"""
        return self.key

    def __repr__(self):
        if self.is_array_index():
            return f"[{self.key}]"
        return f".{self.key}"


class PathResolver:
    """
    Path Resolver

    Supported syntax:
    - $.a.b      -> Object field access
    - $.arr[0]   -> Array index access
    - $.a[0].b   -> Mixed access
    """

    PATH_PATTERN = re.compile(
        r"^\$"  # Root $
        r"(?:\.(?P<field>[a-zA-Z_]\w*)"  # .field
        r"|\[(?P<index>\d+)\])"  # [index]
        r"*$"  # Zero or more
    )

    SEGMENT_PATTERN = re.compile(
        r"\.(?P<field>[a-zA-Z_]\w*)"  # .field
        r"|\[(?P<index>\d+)\]"  # [index]
    )

    @classmethod
    def parse(cls, path: str) -> List[PathSegment]:
        """
        Parse path string into path segment list

        Args:
            path: Path string, e.g. "$.a.b[0]"

        Returns:
            List of PathSegments

        Raises:
            MappingError: Invalid path format
        """
        if not path.startswith("$"):
            raise MappingError(f"Path must start with $: {path}")

        segments = []
        rest = path[1:]  # Skip $

        while rest:
            match = cls.SEGMENT_PATTERN.match(rest)
            if not match:
                raise MappingError(f"Invalid path syntax at: {rest}")

            if match.group("field"):
                segments.append(PathSegment(match.group("field")))
            else:
                segments.append(PathSegment(int(match.group("index"))))

            rest = rest[match.end() :]

        return segments

    @classmethod
    def get(cls, obj: Any, path: str) -> Any:
        """
        Read path value

        Per section 5.2: Returns None (null) when reading non-existent path

        Args:
            obj: Root object (typically dict)
            path: Path string

        Returns:
            Value at path, or None if not found
        """
        try:
            segments = cls.parse(path)
        except MappingError:
            return None

        current = obj
        for seg in segments:
            if current is None:
                return None

            if seg.is_array_index():
                if not isinstance(current, list):
                    return None
                idx = seg.get_key()  # type: ignore
                if isinstance(idx, int):
                    if idx < 0 or idx >= len(current):
                        return None
                    current = current[idx]
                else:
                    return None
            else:
                if not isinstance(current, dict):
                    return None
                key = seg.get_key()  # type: ignore
                if isinstance(key, str):
                    current = current.get(key)
                else:
                    return None

        return current

    @classmethod
    def set(
        cls, obj: dict, path: str, value: Any, max_array_length: int = 10000
    ) -> None:
        """
        Set path value

        Per section 5.3:
        - Automatically create {} when intermediate object is missing
        - Automatically create [] when intermediate array is missing
        - Fill with null when array index is out of bounds
        - Produce MappingError when intermediate position is not object/array

        Args:
            obj: Root object (must be dict)
            path: Path string
            value: Value to set
            max_array_length: Maximum array length limit

        Raises:
            MappingError: Invalid path or type conflict
        """
        segments = cls.parse(path)
        if not segments:
            raise MappingError("Cannot set root path $")

        # Iterate to second-to-last segment, create intermediate objects/arrays
        current = obj
        for i, seg in enumerate(segments[:-1]):
            next_seg = segments[i + 1]

            if seg.is_array_index():
                idx = seg.get_key()  # type: ignore
                if not isinstance(idx, int):
                    raise MappingError(f"Expected integer index at path segment {seg}")
                if not isinstance(current, list):
                    raise MappingError(f"Expected list at path segment {seg}")

                # Expand array
                if idx >= len(current):
                    if idx >= max_array_length:
                        raise MappingError(
                            f"Array index {idx} exceeds max length {max_array_length}",
                            {"threshold": max_array_length, "index": idx},
                        )
                    # Fill with None
                    current.extend([None] * (idx - len(current) + 1))

                # Create next-level container
                if current[idx] is None:
                    if next_seg.is_array_index():
                        current[idx] = []
                    else:
                        current[idx] = {}

                current = current[idx]
            else:
                key = seg.get_key()  # type: ignore
                if not isinstance(key, str):
                    raise MappingError(f"Expected string key at path segment {seg}")
                if not isinstance(current, dict):
                    raise MappingError(f"Expected object at path segment {seg}")

                # Create next-level container
                if key not in current or current[key] is None:
                    if next_seg.is_array_index():
                        current[key] = []
                    else:
                        current[key] = {}

                current = current[key]

        # Set final value
        last_seg = segments[-1]
        if last_seg.is_array_index():
            idx = last_seg.get_key()  # type: ignore
            if not isinstance(idx, int):
                raise MappingError(
                    f"Expected integer index at final path segment {last_seg}"
                )
            if not isinstance(current, list):
                raise MappingError(f"Expected list at final path segment {last_seg}")

            if idx >= len(current):
                if idx >= max_array_length:
                    raise MappingError(
                        f"Array index {idx} exceeds max length {max_array_length}",
                        {"threshold": max_array_length, "index": idx},
                    )
                current.extend([None] * (idx - len(current) + 1))

            current[idx] = value
        else:
            key = last_seg.get_key()  # type: ignore
            if not isinstance(key, str):
                raise MappingError(
                    f"Expected string key at final path segment {last_seg}"
                )
            if not isinstance(current, dict):
                raise MappingError(f"Expected object at final path segment {last_seg}")
            current[key] = value

    @classmethod
    def delete(cls, obj: dict, path: str) -> None:
        """
        Delete path value

        Per section 5.4:
        - Reading deleted path returns null
        - Deleting non-existent field is a no-op (no error)
        - Setting deleted array element to null (array not shortened)

        Args:
            obj: Root object
            path: Path string
        """
        try:
            segments = cls.parse(path)
        except MappingError:
            return  # Invalid path, silently ignore

        if not segments:
            return  # Cannot delete root

        # Get parent container
        parent = obj
        for seg in segments[:-1]:
            if parent is None:
                return

            if seg.is_array_index():
                if not isinstance(parent, list):
                    return
                idx = seg.get_key()  # type: ignore
                if isinstance(idx, int):
                    if idx < 0 or idx >= len(parent):
                        return
                    parent = parent[idx]
                else:
                    return
            else:
                if not isinstance(parent, dict):
                    return
                key = seg.get_key()  # type: ignore
                if isinstance(key, str):
                    parent = parent.get(key)
                else:
                    return

        if parent is None:
            return

        # Delete final value
        last_seg = segments[-1]
        if last_seg.is_array_index():
            idx = last_seg.get_key()  # type: ignore
            if (
                isinstance(idx, int)
                and isinstance(parent, list)
                and 0 <= idx < len(parent)
            ):
                parent[idx] = None  # Set to null, don't shorten array
        else:
            key = last_seg.get_key()  # type: ignore
            if isinstance(key, str) and isinstance(parent, dict) and key in parent:
                parent[key] = None  # Set to null, don't delete key

    @classmethod
    def intersect(cls, path_a: str, path_b: str) -> bool:
        """
        Check if two paths intersect

        Per section 11.4 definition:
        - One path is a prefix of another ($.a vs $.a.b)
        - Two paths are identical
        - $.a[0] vs $.a[1] do not intersect
        - $.a vs $.a[1] intersect
        - $.a[0] vs $.a[0].b intersect

        Args:
            path_a: First path
            path_b: Second path

        Returns:
            Whether paths intersect
        """
        try:
            segs_a = cls.parse(path_a)
            segs_b = cls.parse(path_b)
        except MappingError:
            return False

        # Take the shorter length
        min_len = min(len(segs_a), len(segs_b))

        for i in range(min_len):
            seg_a = segs_a[i]
            seg_b = segs_b[i]

            if seg_a.key != seg_b.key:
                # Different keys at same position -> no intersection
                return False

        # All corresponding segments are same, or one is prefix of other -> intersect
        return True

    @classmethod
    def get_parent_and_key(cls, path: str) -> tuple:
        """
        Get parent path and final key from path

        Args:
            path: Full path

        Returns:
            (Parent path, Final key) tuple
        """
        segments = cls.parse(path)
        if not segments:
            raise MappingError("Root path has no parent")

        parent_segments = segments[:-1]
        last_seg = segments[-1]

        parent_path = "$" + "".join(str(s) for s in parent_segments)
        return parent_path, last_seg.key
