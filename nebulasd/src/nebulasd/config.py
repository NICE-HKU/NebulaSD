"""Immutable configuration; importing it performs no model or IPC setup."""
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite


class FailurePolicy(IntEnum):
    FAIL_FAST = 1


@dataclass(frozen=True, slots=True)
class HostKVLayout:
    """Explicit layout for injected backends; normal CUDA setup derives it from the model."""
    block_bytes: int = 16
    dtype: str = 'uint8'
    block_shape: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.block_bytes) is not int or self.block_bytes <= 0 or not isinstance(self.dtype,str) or not self.dtype:
            raise ValueError('invalid HostKV byte size or dtype')
        shape = tuple(self.block_shape)
        if any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('HostKV shape dimensions must be positive integers')
        object.__setattr__(self,'block_shape',shape)


@dataclass(frozen=True, slots=True)
class NebulaSDConfig:
    draft_worker_count: int = 1
    target_worker_count: int = 1
    target_placement: str = 'adaptive'
    draft_placement: str = 'worker_id'
    scheduler_execution: str = 'legacy'
    scheduler_policy: str = 'existing'
    draft_service_gap_ms: float | None = None
    target_service_gap_ms: float | None = None
    service_batch_delay_ms: float = 30.0
    scheduler_ignore_kv_time: bool = False
    scheduler_frontier_factor: int = 2
    draft_service_weight: float = 32.0
    target_service_weight: float = 32.0
    cost_model: str = 'legacy'
    backend_cost_table: str = ''
    backend_cost_source_sha256: str = ''  # Explicit reuse of historical timings; not recalibration.
    request_slots: int = 64
    command_ring_capacity: int | None = None  # Rejected legacy setting; WORK credit is fixed.
    state_change_ring_capacity: int = 256
    failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST
    draft_model_path: str = ''
    target_model_path: str = ''
    devices: tuple[int, ...] = ()  # Draft workers first, then Target workers.
    max_batch_size: int = 2
    draft_max_batch_size: int | None = None
    target_max_batch_size: int | None = None
    max_batch_tokens: int = 512
    draft_max_batch_tokens: int | None = None
    target_verify_max_batch_tokens: int | None = None
    max_proposal_depth: int = 8
    bank_blocks: int = 128
    draft_bank_blocks: int | None = None
    target_bank_blocks: int | None = None
    bank_rows: int = 8
    block_size: int = 16
    draft_host_numa_policy: str = "default"
    draft_host_numa_nodes: tuple[int, ...] = ()
    target_host_numa_policy: str = "default"
    target_host_numa_nodes: tuple[int, ...] = ()
    host_blocks: int = 512
    payload_capacity: int = 4 << 20
    gpu_memory_fraction: float = .60
    startup_timeout: float = 300
    output_dir: str = '/tmp/nebulasd'
    profiling: bool = False
    profile_mode: str = "full"
    profile_max_events: int = 200000

    def __post_init__(self):
        from .kv.numa import validate_policy
        for stage in ("draft", "target"):
            validate_policy(getattr(self, stage + "_host_numa_policy"),
                            getattr(self, stage + "_host_numa_nodes"))
        if self.scheduler_policy not in ('existing', 'service_interval'):
            raise ValueError('scheduler_policy must be existing or service_interval')
        for name in ('draft_service_gap_ms', 'target_service_gap_ms'):
            gap = getattr(self, name)
            if gap is not None and (isinstance(gap, bool) or not isinstance(gap, (int, float))
                                    or not isfinite(gap) or gap < 0):
                raise ValueError(f'{name} must be finite and nonnegative')
        if type(self.scheduler_ignore_kv_time) is not bool:
            raise ValueError('scheduler_ignore_kv_time must be boolean')
        if self.scheduler_ignore_kv_time and self.scheduler_policy != 'service_interval':
            raise ValueError('scheduler_ignore_kv_time requires service_interval')
        delay = self.service_batch_delay_ms
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not isfinite(delay) or delay < 0:
            raise ValueError('service_batch_delay_ms must be finite and nonnegative')
        if self.scheduler_policy == 'service_interval' and (self.scheduler_execution != 'completion'
                or self.draft_service_gap_ms is None or self.target_service_gap_ms is None):
            raise ValueError('service_interval requires completion and explicit Draft/Target service gaps')
        if self.backend_cost_source_sha256 and (len(self.backend_cost_source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.backend_cost_source_sha256)):
            raise ValueError('backend_cost_source_sha256 must be a lowercase SHA256')
        if self.scheduler_execution not in ('legacy', 'event', 'event_early', 'completion'):
            raise ValueError('scheduler_execution must be legacy, event, event_early or completion')
        if type(self.scheduler_frontier_factor) is not int or not 1 <= self.scheduler_frontier_factor <= 4:
            raise ValueError('scheduler_frontier_factor must be in [1, 4]')
        if any(not isfinite(x) or x < 0 for x in (self.draft_service_weight, self.target_service_weight)):
            raise ValueError('service weights must be finite and nonnegative')
        if self.scheduler_execution == 'completion' and (self.draft_placement != 'stagewise'
                or self.target_placement != 'stagewise' or self.cost_model != 'table'):
            raise ValueError('completion requires stagewise Draft/Target placement and cost_model=table')
        if self.draft_placement not in ('worker_id', 'round_robin', 'stagewise'):
            raise ValueError('draft_placement must be worker_id, round_robin or stagewise')
        if self.cost_model not in ('legacy','table') or (self.cost_model=='table' and not self.backend_cost_table):
            raise ValueError('cost_model must be legacy or table with backend_cost_table')
        if self.target_placement not in ('adaptive', 'fixed', 'batch_adaptive', 'stagewise'):
            raise ValueError('target_placement must be adaptive, fixed, batch_adaptive or stagewise')
        if 'stagewise' in (self.draft_placement, self.target_placement) and self.cost_model != 'table':
            raise ValueError('stagewise requires cost_model=table')
        if self.command_ring_capacity is not None:
            raise ValueError('command_ring_capacity is obsolete; autonomous WORK uses its bounded credit window')
        positive = ('draft_worker_count','target_worker_count','request_slots',
                    'state_change_ring_capacity','max_batch_size','max_batch_tokens','max_proposal_depth',
                    'bank_blocks','bank_rows','block_size','host_blocks','payload_capacity','profile_max_events')
        if self.profile_mode not in ('full', 'light', 'draft'):
            raise ValueError('profile_mode must be full, light or draft')
        if type(self.profiling) is not bool:
            raise ValueError('profiling must be bool')
        for name in positive:
            value = getattr(self,name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('state_change_ring_capacity',):
            value = getattr(self,name)
            if value & (value-1):
                raise ValueError(f'{name} must be a power of two')
        if self.failure_policy != FailurePolicy.FAIL_FAST:
            raise ValueError('only fail-fast is supported')
        for name in ('draft_max_batch_size','target_max_batch_size','draft_max_batch_tokens',
                     'target_verify_max_batch_tokens', 'draft_bank_blocks', 'target_bank_blocks'):
            value = getattr(self,name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f'{name} must be a positive integer when set')
        object.__setattr__(self, 'draft_bank_blocks', self.bank_blocks if self.draft_bank_blocks is None else self.draft_bank_blocks)
        object.__setattr__(self, 'target_bank_blocks', self.bank_blocks if self.target_bank_blocks is None else self.target_bank_blocks)
        draft_max_batch_size = self.max_batch_size if self.draft_max_batch_size is None else self.draft_max_batch_size
        target_max_batch_size = self.max_batch_size if self.target_max_batch_size is None else self.target_max_batch_size
        draft_max_batch_tokens = self.max_batch_tokens if self.draft_max_batch_tokens is None else self.draft_max_batch_tokens
        target_verify_max_batch_tokens = (self.max_batch_tokens if self.target_verify_max_batch_tokens is None
                                          else self.target_verify_max_batch_tokens)
        object.__setattr__(self,'draft_max_batch_size',draft_max_batch_size)
        object.__setattr__(self,'target_max_batch_size',target_max_batch_size)
        object.__setattr__(self,'draft_max_batch_tokens',draft_max_batch_tokens)
        object.__setattr__(self,'target_verify_max_batch_tokens',target_verify_max_batch_tokens)
        if not isfinite(self.startup_timeout) or self.startup_timeout <= 0:
            raise ValueError('startup_timeout must be finite and positive')
        if not isfinite(self.gpu_memory_fraction) or not 0 < self.gpu_memory_fraction < 1:
            raise ValueError('gpu_memory_fraction must be between zero and one')
        count = self.draft_worker_count+self.target_worker_count
        devices = tuple(self.devices) or tuple(range(count))
        if len(devices)!=count or len(set(devices))!=count or any(type(d) is not int or d<0 for d in devices):
            raise ValueError('provide one distinct device per Worker')
        object.__setattr__(self,'devices',devices)
        for name in ('max_batch_size','draft_max_batch_size','target_max_batch_size'):
            if getattr(self,name) > 128:
                raise ValueError(f"{name} exceeds the supported fixed command payload budget (128)")
        if self.bank_rows < self.target_max_batch_size:
            raise ValueError('bank_rows cannot be smaller than target_max_batch_size')
