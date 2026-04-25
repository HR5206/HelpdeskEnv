"""Shim module to keep backwards compatibility.

The real FastAPI application now lives in ``server/app.py`` as ``server.app:app``.
This module simply re-exports that app so existing references to ``server:app``
continue to work.
"""

from server.app import app  # noqa: F401


