from collections.abc import Callable

from pydantic import BaseModel, ConfigDict
from pydantic import validate_call as _pydantic_validate_call
from pydantic.dataclasses import dataclass as _pydantic_dataclass

_CONFIG = ConfigDict(arbitrary_types_allowed=True, strict=False)


def validated[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Runtime-validate args and return against the function's type hints.

    Apply to every named function and method in the project. Lambdas and
    trivial private inner helpers are exempt.
    """
    return _pydantic_validate_call(config=_CONFIG, validate_return=True)(func)


class ValidatedModel(BaseModel):
    """Project-standard pydantic BaseModel: validates on construction."""

    model_config = _CONFIG


validated_dataclass = _pydantic_dataclass(config=_CONFIG)
