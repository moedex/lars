"""LARS: Limited but Accurate Response System.

A local-models-only typed-decision engine. State plus typed questions go in;
calibrated probability distributions come out. Wire-compatible with the
System One HTTP API so existing client SDKs work with a base URL change.
"""

from lars.engine import Engine
from lars.schema import SystemOneRequest, SystemOneResponse

__version__ = "0.0.1"
__all__ = ["Engine", "SystemOneRequest", "SystemOneResponse", "__version__"]
