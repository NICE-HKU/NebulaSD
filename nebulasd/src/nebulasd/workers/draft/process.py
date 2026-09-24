"""Formal spawn entry: Draft control and execution are independent interpreters."""
from nebulasd.workers.target.process import run_target
from nebulasd.workers.target.control import run_control
from nebulasd.workers.target.service import TargetProcesses


def backend_factory(options, stack, **kwargs):
    from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
    from nebulasd.workers.resources import attach_router
    from .backend import DraftBackend
    snapshots, attachments = attach_router(options['snapshots'],SharedDraftSnapshotArena)
    for arena in attachments:
        stack.callback(arena.close)
    return DraftBackend(snapshots=snapshots,layout_id=options['layout_id'],host_arena_id=options['host_arena_id'],**kwargs)


def publisher_factory(options, table, completions, stack):
    from nebulasd.data.shared_arenas import SharedProposalArena
    from nebulasd.data.draft_snapshot_arena import SharedDraftSnapshotArena
    from .publication import DraftPublisher
    proposals = options['result_proposals'].attach(SharedProposalArena,writer=True)
    stack.callback(proposals.close)
    snapshots = options['result_snapshots'].attach(SharedDraftSnapshotArena,writer=True)
    stack.callback(snapshots.close)
    return DraftPublisher(table,proposals,snapshots,completions,host=options['host'],block_size=options.get('block_size',16),profile=options.get('profile',False))


def run_draft(*args):
    return run_target(*args,backend_factory=backend_factory,result_kind='DRAFT_RESULT',import_kind='DRAFT_IMPORTED',observation_factory=observation_factory)


def run_draft_control(*args):
    return run_control(*args,publisher_factory=publisher_factory,result_kind='DRAFT_RESULT',import_kind='DRAFT_IMPORTED',projection_factory=projection_factory)


def observation_factory(options,stack):
    from .observation import Observation
    observation=Observation(options['observation'])
    stack.callback(observation.close)
    return observation


def projection_factory(options,stack):
    from nebulasd.table.native_storage import worker_table,close_table_partitions
    from .observation import Projection
    registry=worker_table(1,descriptors=options['registry'])
    stack.callback(close_table_partitions,registry._partitions)
    return Projection(registry,observation_factory(options,stack),options['blocks_per_bank'],options['capacity_rows'])


class DraftProcesses(TargetProcesses):
    def __init__(self, options, *, context=None, execution_entry=run_draft):
        from nebulasd.table.native_storage import worker_table,table_descriptors,close_table_partitions
        from .observation import Observation
        self.registry=worker_table(1)
        self.observation=Observation()
        self._draft_resources_closed=False
        try:
            super().__init__(options|dict(registry=table_descriptors(self.registry),
                observation=options.get('observation',self.observation.segment.descriptor)),
                context=context,execution_entry=execution_entry,control_entry=run_draft_control,role_name='draft')
        except BaseException:
            close_table_partitions(self.registry._partitions,unlink=True)
            self.observation.close(unlink=True)
            self._draft_resources_closed=True
            raise

    def close(self, timeout=30):
        from nebulasd.table.native_storage import close_table_partitions
        try:
            super().close(timeout=timeout)
        finally:
            if not self._draft_resources_closed:
                close_table_partitions(self.registry._partitions,unlink=True)
                self.observation.close(unlink=True)
                self._draft_resources_closed=True
