"""Cold API startup and bounded cleanup; no model dependency on import."""
from time import monotonic
from pathlib import Path
from os import environ
from nebulasd.config import HostKVLayout
from nebulasd.core.enums import WorkerRole,WorkerStatus,StateChangeBlockKind as K
from nebulasd.scheduler.views import WorkerSpec,value
from .cost_setup import make_estimator
from .core import Engine
from .resources import ControlResources
from .client import TokenEngine


def create(config,*,worker_factory,worker_options,kv_layout,observer,draft_kv_layout=None):
    draft_layout = draft_kv_layout
    if worker_factory is not None or worker_options is not None or kv_layout is not None or draft_kv_layout is not None:
        raise ValueError('legacy factory injection is unsupported; use autonomous WORK test endpoints')
    if config.scheduler_execution != 'completion':
        raise ValueError('Engine supports only completion + stagewise autonomous WORK')
    if config.scheduler_policy == 'service_interval' and (environ.get('STARSD_DMA_MODE') or 'process') != 'process':
        raise ValueError('service_interval requires process DMA for HostKV-ready triggers')
    from .model_factory import model_layout
    kv_layout = model_layout(config)
    draft_layout = model_layout(config, draft=True)
    layout = kv_layout or HostKVLayout()
    if layout.block_bytes<=0:
        raise ValueError('HostKV block_bytes must be positive')
    specs = []
    for i in range(config.draft_worker_count+config.target_worker_count):
        draft = i < config.draft_worker_count
        specs.append(WorkerSpec(i,WorkerRole.DRAFT if draft else WorkerRole.TARGET,
            max_batch_size=config.draft_max_batch_size if draft else config.target_max_batch_size,
            max_batch_tokens=(config.draft_max_batch_tokens if draft
                              else max(config.max_batch_tokens,config.target_verify_max_batch_tokens)),
            prefill_max_batch_tokens=config.max_batch_tokens,
            verify_max_batch_tokens=config.target_verify_max_batch_tokens,
            bank_blocks=config.draft_bank_blocks if draft else config.target_bank_blocks,bank_rows=config.bank_rows,block_size=config.block_size,draft_banked=draft))
    specs = tuple(specs)
    resources = ControlResources(specs,slots=config.request_slots,
        state_ring_capacity=config.state_change_ring_capacity,payload_capacity=config.payload_capacity,
        host_blocks=config.host_blocks,block_bytes=layout.block_bytes,dtype=layout.dtype,kv_block_shape=layout.block_shape,draft_layout=draft_layout,
        draft_host_numa_policy=config.draft_host_numa_policy,draft_host_numa_nodes=config.draft_host_numa_nodes,
        target_host_numa_policy=config.target_host_numa_policy,target_host_numa_nodes=config.target_host_numa_nodes)
    engine = supervisor = None
    profile = None
    if config.profiling:
        from uuid import uuid4
        profile = (str(Path(config.output_dir)/'profiling'/uuid4().hex), config.profile_max_events, config.profile_mode)
    try:
        if resources.host.numa_placement is not None or resources.draft_host.numa_placement is not None:
            import json
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(config.output_dir)/"hostkv-numa.json").write_text(json.dumps(
                dict(draft=resources.draft_host.numa_placement, target=resources.host.numa_placement), indent=2))
        from .autonomous_supervisor import AutonomousSupervisor
        supervisor = AutonomousSupervisor(resources, options=config, output_dir=config.output_dir)
        estimator = make_estimator(config,layout,draft_layout)
        from nebulasd.scheduler.factory import completion_scheduler
        scheduler = completion_scheduler(config, estimator)
        engine = Engine(resources,supervisor,observer=observer,scheduler=scheduler)
        if profile is not None:
            from nebulasd.observability.profiling import ProfileRecorder, attach_engine
            engine.profiler = ProfileRecorder(profile[0], 'engine', profile[1], mode=profile[2])
            attach_engine(engine, engine.profiler)
        client = TokenEngine(config,engine)
        if profile is not None:
            from nebulasd.observability.client_profiling import attach_client
            attach_client(client, engine.profiler)
        supervisor.start()
        deadline = monotonic()+config.startup_timeout
        while True:
            engine.step()
            if all(value(engine.rows.get((K.WORKER_COMMON,w.worker_id)),'status')==WorkerStatus.ONLINE for w in specs):
                return client
            if monotonic()>=deadline:
                raise TimeoutError('Workers did not become online')
            supervisor.bell.wait(.001)
    except BaseException as error:
        try:
            if engine is not None:
                engine.close()
            else:
                try:
                    if supervisor is not None:
                        supervisor.close()
                finally:
                    resources.close()
        except BaseException as cleanup:
            if hasattr(error,'add_note'):
                error.add_note(f'startup cleanup also failed: {cleanup}')
        raise
