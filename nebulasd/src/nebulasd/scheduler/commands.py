"""Mechanical dispatch of scheduler decisions to worker command rings."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from time import perf_counter_ns

from nebulasd.core.enums import StateChangeBlockKind
from nebulasd.core.ids import COMMAND_SEQ
from nebulasd.ipc.command_arena import CommandArena
from nebulasd.ipc.command_arena import CommandBackpressure
from nebulasd.ipc.command_ring import CommandRing
from nebulasd.ipc.protocol import (
    PrepareDraftBankCommand, RunDraftBatchCommand,
    CommandHeader,
    DraftBatchCommand,
    HotCommand,
    PrepareTargetBankCommand,
    RunTargetBatchCommand,
    TargetPrefillBatchCommand,
    encode_command_payload,
)
from nebulasd.table.storage import RequestSchedulingTable
from nebulasd.table.storage import TableProtocolError
from nebulasd.table.writers import DispatcherTableWriter


class DispatchPlanePoisoned(RuntimeError):
    """Raised after a post-ring failure makes the dispatch stream unsafe to continue."""


@dataclass(frozen=True, slots=True)
class WorkerCommandEndpoint:
    worker_id: int
    worker_generation: int
    ring: CommandRing
    arena: CommandArena


@dataclass(frozen=True, slots=True)
class DispatchLatencySummary:
    samples: int
    p50_ns: int
    p95_ns: int
    p99_ns: int


class DispatchPlane:
    """Dispatcher that sends already-made Scheduler decisions without policy."""

    def __init__(self, *, endpoints: tuple[WorkerCommandEndpoint, ...], request_table: RequestSchedulingTable) -> None:
        self._endpoints = {endpoint.worker_id: endpoint for endpoint in endpoints}
        self._request_table = request_table
        self._dispatch_writer = DispatcherTableWriter(request_table)
        from nebulasd.table.draft_dispatch import DraftDispatchContract
        self._draft_dispatch = DraftDispatchContract(request_table)
        self._dispatch_publish_seq = 0
        self._latencies_ns = deque(maxlen=8192)
        self._poisoned = False

    def dispatch(self, command: HotCommand) -> CommandHeader:
        if self._poisoned:
            raise DispatchPlanePoisoned("dispatch plane is poisoned after post-ring publication failure")
        started = perf_counter_ns()
        endpoint = self._endpoint_for(command)
        self._validate_worker_generation(command, endpoint)
        self._preflight_dispatch_facts(command)
        if not endpoint.ring.can_push():
            raise CommandBackpressure("command ring is full")

        payload = encode_command_payload(command)
        handle = endpoint.arena.allocate(command_seq=command.command_seq, payload=payload)
        header = CommandHeader(
            command_seq=command.command_seq,
            worker_generation=endpoint.worker_generation,
            command_kind=command.kind,
            payload_offset=handle.offset,
            payload_length=handle.length,
            flags=0,
        )
        try:
            endpoint.ring.publish(header)
        except Exception:
            endpoint.arena.discard_last(command.command_seq)
            raise

        try:
            self._publish_dispatch_facts(command)
            # The first wake can race ahead of the dispatch fences. Wake again
            # after all facts are visible so a pending consumer needs no timer.
            bell = getattr(endpoint.ring, 'doorbell', None)
            if bell is not None:
                bell.ring()
        except Exception as exc:
            self._poisoned = True
            raise DispatchPlanePoisoned("dispatch fact publication failed after command ring publish") from exc
        self._latencies_ns.append(perf_counter_ns() - started)
        return header

    def latency_summary(self) -> DispatchLatencySummary:
        if not self._latencies_ns:
            return DispatchLatencySummary(samples=0, p50_ns=0, p95_ns=0, p99_ns=0)
        ordered = sorted(self._latencies_ns)
        return DispatchLatencySummary(
            samples=len(ordered),
            p50_ns=_percentile(ordered, 0.50),
            p95_ns=_percentile(ordered, 0.95),
            p99_ns=_percentile(ordered, 0.99),
        )

    def _endpoint_for(self, command: HotCommand) -> WorkerCommandEndpoint:
        worker_id = command.worker_id
        try:
            return self._endpoints[worker_id]
        except KeyError as exc:
            raise KeyError(f"no command endpoint for worker {worker_id}") from exc

    def _validate_worker_generation(self, command: HotCommand, endpoint: WorkerCommandEndpoint) -> None:
        worker_generation = command.worker_generation if isinstance(command, (DraftBatchCommand, PrepareDraftBankCommand, RunDraftBatchCommand)) else command.target_generation
        if worker_generation != endpoint.worker_generation:
            raise ValueError("stale worker_generation in scheduler command")

    def _preflight_dispatch_facts(self, command: HotCommand) -> None:
        if isinstance(command, DraftBatchCommand) and command.bank is not None:
            if command.bank.bank_epoch == 0:
                raise TableProtocolError('canonical initial Draft reservation requires a positive Bank epoch')
            from nebulasd.core.ids import U64
            from nebulasd.table.draft_fences import has_draft_dispatch
            partition = self._request_table.partition(StateChangeBlockKind.REQUEST_DRAFT)
            dispatch = self._request_table.partition(StateChangeBlockKind.REQUEST_DISPATCH)
            for row in command.new_requests:
                if dispatch.read_publish_seq(row.request_slot) != U64.invalid:
                    issued = dispatch.read_stable(row.request_slot)
                    if issued.get('request_epoch') == row.request_epoch and has_draft_dispatch(issued):
                        raise TableProtocolError('initial Draft operation was already dispatched')
                if partition.read_publish_seq(row.request_slot) != U64.invalid:
                    if partition.read_stable(row.request_slot).get('request_epoch') == row.request_epoch:
                        raise TableProtocolError('initial Draft Bank cannot overwrite an existing session epoch')
        if isinstance(command, (PrepareDraftBankCommand, RunDraftBatchCommand)):
            self._draft_dispatch.preflight(command)
        for request in _dispatch_fact_requests(command):
            engine = self._request_table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(
                request.request_slot,
                include_cold=False,
            )
            if engine.get("request_epoch") != request.request_epoch:
                raise TableProtocolError("stale request_epoch for dispatch")

    def _publish_dispatch_facts(self, command: HotCommand) -> None:
        if isinstance(command, (PrepareDraftBankCommand, RunDraftBatchCommand)):
            self._draft_dispatch.sent(command)
            return
        if isinstance(command, DraftBatchCommand):
            for request in (*command.new_requests, *command.cached_request_deltas):
                self._dispatch_writer.publish_draft_command_sent(
                    slot=request.request_slot,
                    publish_seq=self._dispatch_writer.next_publish_seq(request.request_slot),
                    request_epoch=request.request_epoch,
                    draft_issue_seq=request.op_seq,
                    draft_worker_generation=command.worker_generation,
                    draft_round_id=request.round_id,
                    draft_worker_id=command.worker_id,
                    bank=command.bank,
                )
        elif isinstance(command, TargetPrefillBatchCommand):
            for request in command.requests:
                self._dispatch_writer.publish_run_command_sent(
                    slot=request.request_slot,
                    publish_seq=self._dispatch_writer.next_publish_seq(request.request_slot),
                    request_epoch=request.request_epoch,
                    target_run_seq=request.run_seq,
                    target_round_id=request.round_id,
                    planned_target_generation=command.target_generation,
                    planned_bank_epoch=command.bank_epoch,
                    planned_target_id=command.worker_id,
                    planned_bank_id=command.bank_id,
                )
        elif isinstance(command, PrepareTargetBankCommand):
            for request in command.requests:
                self._dispatch_writer.publish_prepare_command_sent(
                    slot=request.request_slot,
                    publish_seq=self._dispatch_writer.next_publish_seq(request.request_slot),
                    request_epoch=request.request_epoch,
                    target_prepare_seq=request.op_seq,
                    planned_target_generation=command.target_generation,
                    planned_bank_epoch=command.next_bank_epoch,
                    planned_target_id=command.worker_id,
                    planned_bank_id=command.standby_bank_id,
                )
        elif isinstance(command, RunTargetBatchCommand):
            for request in command.requests:
                self._dispatch_writer.publish_run_command_sent(
                    slot=request.request_slot,
                    publish_seq=self._dispatch_writer.next_publish_seq(request.request_slot),
                    request_epoch=request.request_epoch,
                    target_run_seq=request.run_seq,
                    target_round_id=request.round_id,
                    planned_target_generation=command.target_generation,
                    planned_bank_epoch=command.active_bank_epoch,
                    planned_target_id=command.worker_id,
                    planned_bank_id=command.active_bank_id,
                )

    def _next_dispatch_publish_seq(self) -> int:
        current = self._dispatch_publish_seq
        self._dispatch_publish_seq = COMMAND_SEQ.next(current)
        return current


def _percentile(ordered: list[int], fraction: float) -> int:
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _dispatch_fact_requests(command: HotCommand) -> tuple[object, ...]:
    if isinstance(command, DraftBatchCommand):
        return (*command.new_requests, *command.cached_request_deltas)
    if isinstance(command, TargetPrefillBatchCommand):
        return command.requests
    if isinstance(command, PrepareTargetBankCommand):
        return command.requests
    return command.requests
