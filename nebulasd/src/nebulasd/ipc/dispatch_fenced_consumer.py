"""Bridge the documented ring-publication → DispatchBlock-publication window."""

from nebulasd.core.enums import StateChangeBlockKind as K
from nebulasd.core.ids import U64
from .protocol import (DraftBatchCommand, PrepareTargetBankCommand, PrepareDraftBankCommand, RunDraftBatchCommand)
from nebulasd.table.storage import StableReadConflict


class DispatchFencedConsumer:
    def __init__(self, ring, table, worker_id):
        self.ring, self.table, self.worker_id = ring, table, worker_id
        self.pending = None

    def consume(self, *, expected_worker_generation, arena):
        if self.pending is None:
            self.pending = self.ring.consume(expected_worker_generation=expected_worker_generation, arena=arena)
        if self.pending is None:
            return None
        command = self.pending.decode(worker_id=self.worker_id)
        if isinstance(command, DraftBatchCommand) and command.bank is not None:
            from nebulasd.table.draft_fences import has_draft_dispatch, validate_initial_dispatch
            from nebulasd.table.storage import TableProtocolError
            partition = self.table.partition(K.REQUEST_DISPATCH)
            for item in command.new_requests:
                try:
                    if partition.read_publish_seq(item.request_slot) == U64.invalid:
                        return None
                    row = partition.read_stable(item.request_slot)
                    epoch = row.get('request_epoch')
                    if epoch != item.request_epoch:
                        if U64.is_newer(item.request_epoch, epoch):
                            return None  # Previous slot lifetime, before post-ring publication.
                        raise TableProtocolError('Draft dispatch request epoch overtook command')
                    if not has_draft_dispatch(row):
                        return None  # Only Target prefill has published this partition.
                    op = row.get('draft_issue_seq')
                    if op != item.op_seq:
                        # An initial operation cannot follow another Draft
                        # issue in this lifetime. This is a conflicting owner,
                        # not the legal post-ring publication window.
                        raise TableProtocolError('Draft dispatch conflicts with unexecuted initial command')
                    validate_initial_dispatch(self.table, command, item)
                except StableReadConflict:
                    return None
            result, self.pending = self.pending, None
            return result
        if isinstance(command, (PrepareDraftBankCommand, RunDraftBatchCommand)):
            from nebulasd.table.draft_fences import validate_prepare, validate_run_dispatch
            from nebulasd.table.storage import TableProtocolError
            for item in command.requests:
                partition = self.table.partition(K.REQUEST_DISPATCH)
                if partition.read_publish_seq(item.request_slot) == U64.invalid:
                    return None
                field = 'draft_prepare_seq' if isinstance(command, PrepareDraftBankCommand) else 'draft_issue_seq'
                expected = item.prepare_seq if isinstance(command, PrepareDraftBankCommand) else item.run_seq
                try:
                    row = partition.read_stable(item.request_slot)
                    if row.get(field) != expected:
                        if U64.is_newer(row.get(field), expected):
                            raise TableProtocolError('Draft dispatch overtook unexecuted command')
                        return None
                    if isinstance(command, PrepareDraftBankCommand):
                        validate_prepare(self.table, command, item)
                    else:
                        validate_run_dispatch(self.table, command, item)
                except StableReadConflict:
                    return None
            result, self.pending = self.pending, None
            return result
        if isinstance(command, DraftBatchCommand):
            items, field = (*command.new_requests, *command.cached_request_deltas), "draft_issue_seq"
        else:
            items = command.requests
            field = "target_prepare_seq" if isinstance(command, PrepareTargetBankCommand) else "target_run_seq"
        partition = self.table.partition(K.REQUEST_DISPATCH)
        for item in items:
            if partition.read_publish_seq(item.request_slot) == U64.invalid:
                return None
            try:
                row = partition.read_stable(item.request_slot)
            except StableReadConflict:
                return None
            expected = item.run_seq if field == "target_run_seq" else item.op_seq
            if row.get(field) != expected:
                if row.get(field) > expected:
                    raise ValueError("dispatch fact overtook unexecuted command")
                return None
        result, self.pending = self.pending, None
        return result
