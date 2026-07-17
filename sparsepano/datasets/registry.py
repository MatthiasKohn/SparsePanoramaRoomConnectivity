"""Dataset adapter registry."""

from __future__ import annotations

from collections.abc import Callable

from .base import Dataset

_REGISTRY: dict[str, type[Dataset]] = {}


def register_dataset(name: str) -> Callable[[type[Dataset]], type[Dataset]]:
    key = name.lower()

    def deco(cls: type[Dataset]) -> type[Dataset]:
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"dataset adapter already registered: {name}")
        _REGISTRY[key] = cls
        cls.name = key
        return cls

    return deco


def get_dataset(name: str, root: str, **cfg) -> Dataset:
    key = name.lower()
    if key not in _REGISTRY:
        # Import built-in adapters lazily so registry stays lightweight.
        if key == "zind":
            from . import zind  # noqa: F401
        if key not in _REGISTRY:
            known = ", ".join(sorted(_REGISTRY)) or "<none>"
            raise KeyError(f"unknown dataset {name!r}; registered: {known}")
    return _REGISTRY[key](root=root, **cfg)


def list_datasets() -> list[str]:
    if "zind" not in _REGISTRY:
        from . import zind  # noqa: F401
    return sorted(_REGISTRY)

