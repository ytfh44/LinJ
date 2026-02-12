"""
状态路径解析器

实现 LinJ 规范 5.1-5.4 节定义的路径语法和读写语义：
- 根为 $
- . 访问对象字段
- [n] 访问数组下标
- 写入时自动创建中间对象/数组
"""

import re
from typing import Any, List, Union

from ..exceptions.errors import MappingError


class PathSegment:
    """路径段"""

    def __init__(self, key: Union[str, int]):
        self.key = key

    def is_array_index(self) -> bool:
        return isinstance(self.key, int)

    def get_key(self) -> Union[str, int]:
        """获取键值，类型安全"""
        return self.key

    def __repr__(self):
        if self.is_array_index():
            return f"[{self.key}]"
        return f".{self.key}"


class PathResolver:
    """
    路径解析器

    支持语法：
    - $.a.b      -> 对象字段访问
    - $.arr[0]   -> 数组下标访问
    - $.a[0].b   -> 混合访问
    """

    PATH_PATTERN = re.compile(
        r"^\$"  # 根 $
        r"(?:\.(?P<field>[a-zA-Z_]\w*)"  # .field
        r"|\[(?P<index>\d+)\])"  # [index]
        r"*$"  # 零个或多个
    )

    SEGMENT_PATTERN = re.compile(
        r"\.(?P<field>[a-zA-Z_]\w*)"  # .field
        r"|\[(?P<index>\d+)\]"  # [index]
    )

    @classmethod
    def parse(cls, path: str) -> List[PathSegment]:
        """
        解析路径字符串为路径段列表

        Args:
            path: 路径字符串，如 "$.a.b[0]"

        Returns:
            PathSegment 列表

        Raises:
            MappingError: 路径格式无效
        """
        if not path.startswith("$"):
            raise MappingError(f"Path must start with $: {path}")

        segments = []
        rest = path[1:]  # 跳过 $

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
        读取路径值

        按 5.2 节：读取不存在路径时返回 None（空值）

        Args:
            obj: 根对象（通常是 dict）
            path: 路径字符串

        Returns:
            路径对应的值，不存在则返回 None
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
        设置路径值

        按 5.3 节：
        - 中间对象缺失时自动创建 {}
        - 中间数组缺失时自动创建 []
        - 数组下标越界时以 null 填充至该位置
        - 中间位置不是对象/数组时产生 MappingError

        Args:
            obj: 根对象（必须是 dict）
            path: 路径字符串
            value: 要设置的值
            max_array_length: 最大数组长度限制

        Raises:
            MappingError: 路径无效或类型冲突
        """
        segments = cls.parse(path)
        if not segments:
            raise MappingError("Cannot set root path $")

        # 遍历到倒数第二个段，创建中间对象/数组
        current = obj
        for i, seg in enumerate(segments[:-1]):
            next_seg = segments[i + 1]

            if seg.is_array_index():
                idx = seg.get_key()  # type: ignore
                if not isinstance(idx, int):
                    raise MappingError(f"Expected integer index at path segment {seg}")
                if not isinstance(current, list):
                    raise MappingError(f"Expected list at path segment {seg}")

                # 扩容数组
                if idx >= len(current):
                    if idx >= max_array_length:
                        raise MappingError(
                            f"Array index {idx} exceeds max length {max_array_length}",
                            {"threshold": max_array_length, "index": idx},
                        )
                    # 以 None 填充
                    current.extend([None] * (idx - len(current) + 1))

                # 创建下一级容器
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

                # 创建下一级容器
                if key not in current or current[key] is None:
                    if next_seg.is_array_index():
                        current[key] = []
                    else:
                        current[key] = {}

                current = current[key]

        # 设置最终值
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
        删除路径值

        按 5.4 节：
        - 删除后该路径读取结果为 null
        - 删除不存在的字段为无效操作，不报错
        - 删除数组元素时设为 null（不缩短数组）

        Args:
            obj: 根对象
            path: 路径字符串
        """
        try:
            segments = cls.parse(path)
        except MappingError:
            return  # 无效路径，静默忽略

        if not segments:
            return  # 不能删除根

        # 获取父级容器
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

        # 删除最终值
        last_seg = segments[-1]
        if last_seg.is_array_index():
            idx = last_seg.get_key()  # type: ignore
            if (
                isinstance(idx, int)
                and isinstance(parent, list)
                and 0 <= idx < len(parent)
            ):
                parent[idx] = None  # 设为 null，不缩短数组
        else:
            key = last_seg.get_key()  # type: ignore
            if isinstance(key, str) and isinstance(parent, dict) and key in parent:
                parent[key] = None  # 设为 null，不删除键

    @classmethod
    def intersect(cls, path_a: str, path_b: str) -> bool:
        """
        判断两路径是否相交

        按 11.4 节定义：
        - 一条路径是另一条的前缀（$.a 与 $.a.b）
        - 两条路径完全相同
        - $.a[0] 与 $.a[1] 不相交
        - $.a 与 $.a[1] 相交
        - $.a[0] 与 $.a[0].b 相交

        Args:
            path_a: 第一条路径
            path_b: 第二条路径

        Returns:
            是否相交
        """
        try:
            segs_a = cls.parse(path_a)
            segs_b = cls.parse(path_b)
        except MappingError:
            return False

        # 取较短的长度
        min_len = min(len(segs_a), len(segs_b))

        for i in range(min_len):
            seg_a = segs_a[i]
            seg_b = segs_b[i]

            if seg_a.key != seg_b.key:
                # 在同一位置但键不同 -> 不相交
                return False

        # 所有对应段都相同，或一条是另一条的前缀 -> 相交
        return True

    @classmethod
    def get_parent_and_key(cls, path: str) -> tuple:
        """
        获取路径的父路径和最终键

        Args:
            path: 完整路径

        Returns:
            (父路径, 最终键) 元组
        """
        segments = cls.parse(path)
        if not segments:
            raise MappingError("Root path has no parent")

        parent_segments = segments[:-1]
        last_seg = segments[-1]

        parent_path = "$" + "".join(str(s) for s in parent_segments)
        return parent_path, last_seg.key
