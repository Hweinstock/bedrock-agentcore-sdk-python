"""Utilities for snake_case/camelCase normalization."""

import functools
import re
from typing import Any, Callable, Dict, Optional, Tuple

_VALID_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def snake_to_camel(name: str) -> str:
    """Convert a snake_case string to camelCase.

    Already-camelCase strings pass through unchanged (no underscores to split on).
    Raises ValueError for malformed snake_case (e.g. leading/trailing underscores,
    consecutive underscores, uppercase characters).
    """
    if "_" not in name:
        return name
    if not _VALID_SNAKE_RE.match(name):
        raise ValueError(f"Invalid parameter name: '{name}'")
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def accept_snake_case_kwargs(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a method to accept both snake_case and camelCase kwargs.

    Converts all snake_case kwargs to camelCase before forwarding.
    Raises TypeError if both forms are provided (e.g. memory_id and memoryId).
    """

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        converted: Dict[str, Any] = {}
        original_keys: Dict[str, str] = {}
        for key, value in kwargs.items():
            camel_key = snake_to_camel(key)
            if camel_key in converted:
                raise TypeError(
                    f"Got both '{original_keys[camel_key]}' and '{key}' for the same parameter. "
                    f"Use one or the other."
                )
            original_keys[camel_key] = key
            converted[camel_key] = value
        return method(*args, **converted)

    return wrapper


def deprecated_alias(kwargs: Dict[str, Any], old_name: str, new_name: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Extract a deprecated kwarg alias, raising TypeError on collision.

    Returns (resolved_value, remaining_kwargs). The resolved value comes from
    the explicit parameter if set, otherwise from the deprecated alias in kwargs.

    Usage:
        def my_method(self, new_param=None, **kwargs):
            new_param, kwargs = deprecated_alias(kwargs, "oldParam", "new_param", new_param)
    """
    old_value = kwargs.pop(old_name, None)
    new_value = kwargs.pop(new_name, None)

    if old_value is not None and new_value is not None:
        raise TypeError(
            f"Got both '{new_name}' and '{old_name}' for the same parameter. "
            f"Use one or the other."
        )

    return old_value if old_value is not None else new_value, kwargs
