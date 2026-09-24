"""GPU process entry for the Target vertical path; no peer handles are accepted."""
from queue import Empty, Full
from contextlib import ExitStack


def run_target(options, commands, results, wake, ready, result_wake=None, *, backend_factory=None, result_kind="RESULT", import_kind="IMPORTED", observation_factory=None):
    if options.get('isolated_compute'):
        from nebulasd.workers.isolated import run_compute
        return run_compute(options, commands, results, wake, ready, result_wake, backend_factory)
    with ExitStack() as stack:
        from nebulasd.workers.channel import LocalChannel
        commands = LocalChannel(commands)
        stack.callback(commands.close)
        results = LocalChannel(results)
        stack.callback(results.close)
        from nebulasd.data.shared_arenas import SharedTokenArena, SharedConfigArena, SharedProposalArena
        from nebulasd.kv.arena import SharedHostKVArena
        from nebulasd.kv.cuda_transfer import CudaCopyBackend
        from nebulasd.workers.banks import Banks
        from nebulasd.workers.dma import DMA
        from nebulasd.workers.runtime import Runtime
        from nebulasd.workers.work import Work
        from .backend import TargetBackend
        from nebulasd.workers.resources import attach_router
        tokens, token_attachments = attach_router(options['tokens'], SharedTokenArena)
        for arena in token_attachments:
            stack.callback(arena.close)
        configs, config_attachments = attach_router(options['configs'], SharedConfigArena)
        for arena in config_attachments:
            stack.callback(arena.close)
        proposals, proposal_attachments = attach_router(options['proposals'], SharedProposalArena)
        for arena in proposal_attachments:
            stack.callback(arena.close)
        host = SharedHostKVArena.attach(options['host'])
        stack.callback(host.close)
        kwargs = dict(model_path=options['model_path'], device=options['device'],
            blocks_per_bank=options['blocks_per_bank'], capacity_rows=options['capacity_rows'],
            host=host, tokens=tokens, configs=configs, proposals=proposals,
            block_size=options.get('block_size',16),max_batch_tokens=options.get('max_batch_tokens',4096))
        role = TargetBackend(**kwargs) if backend_factory is None else backend_factory(options, stack, **kwargs)
        stack.callback(role.close)
        import os
        mode = os.environ.get('STARSD_DMA_MODE') or 'process'
        if mode == 'process':
            from nebulasd.workers.dma_process import ProcessDMA
            dma = ProcessDMA(arena=host, k_cache=role.model.k_cache,
                             v_cache=role.model.v_cache, options=options, wake=wake)
        elif mode == 'thread':
            copy = CudaCopyBackend(arena=host, k_cache=role.model.k_cache, v_cache=role.model.v_cache)
            backends = [copy, copy.fork()]
            if options.get('diagnostic_copy_delay') is not None:
                from nebulasd.workers.diagnostics import DelayedCopyBackend
                bank, direction, seconds = options['diagnostic_copy_delay']
                backends[bank] = DelayedCopyBackend(backends[bank], direction, seconds)
            for backend in backends:
                stack.callback(backend.close)
            dma = DMA(tuple(backends),profile=options.get('profile',False))
        else:
            raise ValueError('STARSD_DMA_MODE must be process or thread')
        stack.callback(dma.close)
        runtime = Runtime(Banks(options['blocks_per_bank'], options['capacity_rows']),
            dma, role.inputs, profile=options.get('profile', False),
            execute=role.execute, write_metadata=role.write_metadata)
        runtime.wakeup = wake
        role.clock_wakeup = wake
        if 'global_registry' in options:
            from nebulasd.workers.observation import execution_observation
            observation_factory = execution_observation
        observation = None if observation_factory is None else observation_factory(options,stack)
        sent = {}
        ready.set()
        while True:
            # Clear before checking ordered input facts and local job completions.
            wake.clear()
            if mode == 'process':
                dma.check()
            for _ in range(4):
                try:
                    message = commands.get_nowait()
                    kind, payload = message[:2]
                except Empty:
                    break
                if kind == 'WORK':
                    work = Work.from_bytes(payload)
                    if not runtime.accept(work):
                        raise RuntimeError('dispatcher exceeded reserved WORK credit')
                elif kind == 'INPUTS':
                    runtime.receive_inputs(payload, message[2])
                elif kind == 'RETIRE_RECORD':
                    role.inputs.retire(payload)
                    runtime.publication_retired(payload)
                    sent.pop(payload, None)
                elif kind == 'DRAIN':
                    runtime.drain()
                elif kind == 'SHUTDOWN':
                    runtime.shutdown()
                else:
                    raise ValueError('unsupported Target control operation')
            changed = runtime.step()
            if observation is not None:
                if 'global_registry' in options:
                    observation.write(runtime,role.compute_clock)
                else:
                    observation.write(runtime.banks)
            for seq, state in tuple(runtime.records.items()):
                flags = sent.setdefault(seq, set())
                if state.imported and state.import_inputs is not None and state.import_inputs.rows and 'IMPORTED' not in flags:
                    rows = state.import_inputs.rows
                    try:
                        results.put_nowait((import_kind, seq, dict(rows=rows,
                            submitted_ns=state.h2d_receipt.submitted_ns)))
                    except Full:
                        continue
                    flags.add('IMPORTED')
                    if result_wake is not None:
                        result_wake.set()
                # Channel capacity covers <=4 records, three messages each.
                # put_nowait never blocks physical owner if publisher is delayed.
                if state.result is not None and 'RESULT' not in flags:
                    message = (result_kind, seq, dict(rows=state.result.rows,
                        compute_start_ns=state.result.compute_start_ns, compute_end_ns=state.result.compute_end_ns))
                    try:
                        results.put_nowait(message)
                    except Full:
                        continue
                    flags.add('RESULT')
                    if result_wake is not None:
                        result_wake.set()
                if state.physical_done and 'PHYSICAL' not in flags:
                    facts = [(k,t) for k,t,_ in state.facts] if options.get('profile',False) else []
                    if options.get('profile',False):
                        for direction,receipt in (('H2D',state.h2d_receipt),('D2H',state.d2h_receipt)):
                            if receipt is not None:
                                facts.extend(((direction+'_LAUNCH',receipt.submitted_ns),
                                    (direction+'_OBSERVED',receipt.completed_ns),(direction+'_CPU_NS',receipt.cpu_ns)))
                    message = ('PHYSICAL', seq, dict(d2h_submitted_ns=state.d2h_receipt.submitted_ns if state.d2h_receipt is not None else 0, outcomes=[int(o) for o in state.outcomes],
                        observed_ns=state.physical_done_ns, facts=facts))
                    try:
                        results.put_nowait(message)
                    except Full:
                        continue
                    flags.add('PHYSICAL')
                    if result_wake is not None:
                        result_wake.set()
            if runtime.quiescent:
                break
            if not changed and commands.ring.head() == commands.ring.tail():
                # Only the existing channel and physical jobs wake this owner.
                wake.wait()
