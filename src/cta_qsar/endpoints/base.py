"""Endpoint plugin base types.

The actual detection logic lives in :mod:`cta_qsar.endpoints.detector`;
this module only defines the shared result model and re-exports it.
"""

from __future__ import annotations

from cta_qsar.endpoints.detector import (
    EndpointDetection,
    EndpointDetector,
    build_endpoint_plugins,
    detect_endpoint,
)

__all__ = [
    "EndpointDetection",
    "EndpointDetector",
    "build_endpoint_plugins",
    "detect_endpoint",
]