"""Small token API, lazy construction and explicit owner-thread execution.

Importing this module does not start workers, attach shared memory or load CUDA.
The application owns tokenization. See docs/current/OPERATIONS.md for application configuration.
"""
from dataclasses import dataclass
from typing import Protocol, Iterator
from .config import NebulaSDConfig,HostKVLayout
from .core.enums import Lifecycle
from .data.generation_config_arena import DraftGenerationConfig as GenerationConfig
from .engine.admission import AdmissionCapacityError,AdmissionRejected


@dataclass(frozen=True, slots=True)
class RequestHandle:
    request_id: int
    request_epoch: int


@dataclass(frozen=True, slots=True)
class StreamEvent:
    request: RequestHandle
    token_ids: tuple[int, ...]
    lifecycle: Lifecycle

    @property
    def finished(self):
        return self.lifecycle != Lifecycle.ACTIVE


class EngineLike(Protocol):
    @property
    def config(self) -> NebulaSDConfig: ...
    def submit(self, prompt_token_ids, generation_config: GenerationConfig) -> RequestHandle: ...
    def poll(self) -> bool: ...
    def read(self, request: RequestHandle) -> tuple[StreamEvent, ...]: ...
    def stream(self, request: RequestHandle, *, timeout: float=60) -> Iterator[StreamEvent]: ...
    def result(self, request: RequestHandle) -> tuple[int, ...]: ...
    def cancel(self, request: RequestHandle) -> None: ...
    def release(self, request: RequestHandle) -> None: ...
    def drain(self, *, timeout: float=60) -> None: ...
    def metrics(self) -> dict: ...
    def reset_metrics(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "EngineLike": ...
    def __exit__(self, kind, error, tb): ...


def create_engine(config: NebulaSDConfig | None = None, *, worker_factory=None,
                  worker_options=None, kv_layout: HostKVLayout | None = None,
                  draft_kv_layout: HostKVLayout | None = None, observer=None) -> EngineLike:
    """Start an owner-thread Engine and wait until every Worker is online.

    Default construction uses canonical SwiftLLM models. Factory/options/layout
    are explicit cold-start injection points for tests or alternate backends.
    Capacity rejection does not stop accepted work; cohorts recycle after retirement.
    """
    from .engine.bootstrap import create
    return create(config or NebulaSDConfig(),worker_factory=worker_factory,
                  worker_options=worker_options,kv_layout=kv_layout,draft_kv_layout=draft_kv_layout,observer=observer)
