"""Configure pytest to use beartype for runtime type checking."""

from functools import wraps

from beartype import beartype
from beartype.roar import (
    BeartypeCallHintParamViolation,
    BeartypeCallHintReturnViolation,
)
import pytest


def pytest_configure(config):
    # Explicitly disable typeguard plugin
    config.addinivalue_line("addopts", "--no-typeguard")


def beartype_decorator(func):
    """Decorator that applies beartype and converts exceptions to pytest failures."""
    beartyped_func = beartype(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return beartyped_func(*args, **kwargs)
        except (BeartypeCallHintParamViolation, BeartypeCallHintReturnViolation) as e:
            pytest.fail(str(e))

    return wrapper


def pytest_collection_modifyitems(session, config, items):
    """Apply beartype decorator to all test functions."""
    for item in items:
        if isinstance(item, pytest.Function):
            item.obj = beartype_decorator(item.obj)
