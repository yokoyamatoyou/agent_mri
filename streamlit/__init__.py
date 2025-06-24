# -*- coding: utf-8 -*-
"""Minimal stub of Streamlit for testing purposes.

This module provides just enough of the :mod:`streamlit` API for the unit tests
to import the project without requiring the real Streamlit package.  Only the
functions that are referenced in the code base are implemented and they simply
perform no-ops.
"""

from contextlib import contextmanager

__all__ = [
    "cache_resource",
    "cache_data",
    "title",
    "file_uploader",
    "info",
    "image",
    "markdown",
    "error",
    "warning",
    "success",
    "write",
    "download_button",
    "spinner",
    "session_state",
]

def cache_resource(func=None, *dargs, **dkwargs):
    """Pass-through decorator used in tests.

    The real Streamlit ``cache_resource`` decorator caches the return value of
    the wrapped function across reruns.  The stub simply returns the original
    function so that it can be called normally.  It supports being used both as
    ``@cache_resource`` and ``@cache_resource()``.
    """

    def decorator(f):
        return f

    if func is None:
        return decorator
    return decorator(func)


def cache_data(func=None, *dargs, **dkwargs):
    """Alias of :func:`cache_resource` provided for compatibility."""
    return cache_resource(func, *dargs, **dkwargs)


def title(*args, **kwargs):
    """Ignore a ``st.title`` call."""
    return None

def file_uploader(*args, **kwargs):
    """Return ``None`` for uploaded file in the stub environment."""
    return None

def info(*args, **kwargs):
    return None

def image(*args, **kwargs):
    return None

def markdown(*args, **kwargs):
    return None

def error(*args, **kwargs):
    return None


def warning(*args, **kwargs):
    return None


def success(*args, **kwargs):
    return None


def write(*args, **kwargs):
    return None


def download_button(*args, **kwargs):
    """Return ``None`` for ``st.download_button`` in tests."""
    return None


@contextmanager
def spinner(*args, **kwargs):
    yield None


# Simple dictionary to mimic ``st.session_state``.
session_state = {}
