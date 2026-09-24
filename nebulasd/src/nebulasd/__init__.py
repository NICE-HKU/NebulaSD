"""Public package surface for the NebulaSD control plane.

Importing this module must stay side-effect free: no worker processes, shared
memory attachment, Ray initialization, CUDA context creation, or legacy control
plane imports.
"""

from .config import FailurePolicy, NebulaSDConfig

from .api import create_engine, GenerationConfig, RequestHandle, StreamEvent, AdmissionCapacityError

__all__ = ["FailurePolicy", "NebulaSDConfig", "__version__", "create_engine",
           "GenerationConfig", "RequestHandle", "StreamEvent", "AdmissionCapacityError"]

__version__ = "0.0.0"
