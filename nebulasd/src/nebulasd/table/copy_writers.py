"""CopyLane-owned D2H/H2D facts; completion never implies Engine lifecycle."""

from nebulasd.core.enums import D2HStatus, H2DStatus, StateChangeBlockKind, validate_enum
from nebulasd.core.errors import ResultCode
from nebulasd.core.ids import BANK_ID, OperationFence
from .storage import FieldValue, RequestSchedulingTable, TableProtocolError

def _enum_value(enum_type, value):
    return int(validate_enum(enum_type, value))

def _expect(value, expected, message):
    if value != expected:
        raise TableProtocolError(message)

class TargetCopyTableWriter:
    def __init__(self, table: RequestSchedulingTable) -> None:
        self._table = table

    def publish_d2h_started(self, *, fence: OperationFence, publish_seq: int,
                            source_bank_id: int, source_bank_epoch: int,
                            host_slot_generation: int, writer_version: int,
                            copy_start_time_ns: int, copy_bytes: int) -> None:
        self._validate_completed_target(fence, bank_id=source_bank_id, bank_epoch=source_bank_epoch)
        self._table._publish_owned(
            owner="target_copy_lane", block_kind=StateChangeBlockKind.REQUEST_D2H,
            row=fence.request.request_slot, publish_seq=publish_seq,
            fields=tuple(FieldValue(k, v) for k, v in dict(
                request_epoch=fence.request.request_epoch, round_id=fence.round_id,
                d2h_op_seq=fence.op_seq, target_id=fence.worker_id,
                target_generation=fence.worker_generation, source_bank_id=source_bank_id,
                source_bank_epoch=source_bank_epoch, host_slot_generation=host_slot_generation,
                writer_version=writer_version, status=int(D2HStatus.IN_D2H),
                result_code=int(ResultCode.OK), copy_start_time_ns=copy_start_time_ns,
                copy_bytes=copy_bytes,
            ).items()),
        )

    def publish_h2d_wait(self, *, fence: OperationFence, publish_seq: int,
                         destination_bank_id: int, destination_bank_epoch: int,
                         source_host_version: int, status: H2DStatus = H2DStatus.WAIT_HOST,
                         copy_start_time_ns: int = 0, copy_bytes: int = 0) -> None:
        if status not in (H2DStatus.WAIT_HOST, H2DStatus.IN_H2D):
            raise ValueError("H2D wait publication requires WAIT_HOST or IN_H2D")
        self._validate_prepare_fence(fence, bank_id=destination_bank_id, bank_epoch=destination_bank_epoch)
        self._table._publish_owned(
            owner="target_copy_lane", block_kind=StateChangeBlockKind.REQUEST_H2D,
            row=fence.request.request_slot, publish_seq=publish_seq,
            fields=tuple(FieldValue(k, v) for k, v in dict(
                request_epoch=fence.request.request_epoch, round_id=fence.round_id,
                observed_prepare_seq=fence.op_seq, target_id=fence.worker_id,
                target_generation=fence.worker_generation, source_host_version=source_host_version,
                destination_bank_id=destination_bank_id, destination_bank_epoch=destination_bank_epoch,
                status=int(status), result_code=int(ResultCode.OK),
                copy_start_time_ns=copy_start_time_ns, copy_bytes=copy_bytes,
            ).items()),
        )

    def publish_host_ready(
        self,
        *,
        fence: OperationFence,
        publish_seq: int,
        source_bank_id: int,
        source_bank_epoch: int,
        host_slot_generation: int,
        writer_version: int,
        ready_version: int,
        committed_blocks: int,
        logical_kv_len: int,
        result_code: ResultCode = ResultCode.OK,
    ) -> None:
        BANK_ID.validate(source_bank_id)
        self._validate_completed_target(fence, bank_id=source_bank_id, bank_epoch=source_bank_epoch)
        hostkv = self._table.partition(StateChangeBlockKind.REQUEST_HOSTKV).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        _expect(hostkv.get("request_epoch"), fence.request.request_epoch, "stale request_epoch for HostKV allocation")
        _expect(hostkv.get("host_slot_generation"), host_slot_generation, "stale HostKV slot generation")
        _expect(hostkv.get("writer_lease_generation"), writer_version, "stale HostKV writer version")
        self._table._publish_owned(
            owner="target_copy_lane",
            block_kind=StateChangeBlockKind.REQUEST_D2H,
            row=fence.request.request_slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", fence.request.request_epoch),
                FieldValue("round_id", fence.round_id),
                FieldValue("d2h_op_seq", fence.op_seq),
                FieldValue("target_generation", fence.worker_generation),
                FieldValue("source_bank_epoch", source_bank_epoch),
                FieldValue("host_slot_generation", host_slot_generation),
                FieldValue("writer_version", writer_version),
                FieldValue("ready_version", ready_version),
                FieldValue("target_id", fence.worker_id),
                FieldValue("status", _enum_value(D2HStatus, D2HStatus.HOST_READY)),
                FieldValue("result_code", _enum_value(ResultCode, result_code)),
                FieldValue("source_bank_id", source_bank_id),
                FieldValue("committed_blocks", committed_blocks),
                FieldValue("logical_kv_len", logical_kv_len),
            ),
        )

    def publish_gpu_ready(
        self,
        *,
        fence: OperationFence,
        publish_seq: int,
        destination_bank_id: int,
        destination_bank_epoch: int,
        source_host_version: int,
        gpu_ready_version: int,
        copied_blocks: int,
        result_code: ResultCode = ResultCode.OK,
    ) -> None:
        BANK_ID.validate(destination_bank_id)
        self._validate_prepare_fence(fence, bank_id=destination_bank_id, bank_epoch=destination_bank_epoch)
        engine = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        d2h = self._table.partition(StateChangeBlockKind.REQUEST_D2H).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        hostkv = self._table.partition(StateChangeBlockKind.REQUEST_HOSTKV).read_stable(
            fence.request.request_slot,
            include_cold=False,
        )
        _expect(engine.get("request_epoch"), fence.request.request_epoch, "stale request_epoch for H2D fact")
        _expect(d2h.get("request_epoch"), fence.request.request_epoch, "stale D2H request_epoch for H2D fact")
        _expect(d2h.get("status"), _enum_value(D2HStatus, D2HStatus.HOST_READY), "D2H fact is not HOST_READY")
        _expect(d2h.get("result_code"), _enum_value(ResultCode, ResultCode.OK), "D2H fact did not complete OK")
        _expect(d2h.get("ready_version"), source_host_version, "stale HostKV ready version")
        _expect(hostkv.get("request_epoch"), fence.request.request_epoch, "stale HostKV request_epoch for H2D fact")
        _expect(d2h.get("host_slot_generation"), hostkv.get("host_slot_generation"), "stale HostKV slot generation")
        _expect(d2h.get("writer_version"), hostkv.get("writer_lease_generation"), "stale HostKV writer lease")
        self._table._publish_owned(
            owner="target_copy_lane",
            block_kind=StateChangeBlockKind.REQUEST_H2D,
            row=fence.request.request_slot,
            publish_seq=publish_seq,
            fields=(
                FieldValue("request_epoch", fence.request.request_epoch),
                FieldValue("round_id", fence.round_id),
                FieldValue("observed_prepare_seq", fence.op_seq),
                FieldValue("target_generation", fence.worker_generation),
                FieldValue("source_host_version", source_host_version),
                FieldValue("destination_bank_epoch", destination_bank_epoch),
                FieldValue("gpu_ready_version", gpu_ready_version),
                FieldValue("target_id", fence.worker_id),
                FieldValue("status", _enum_value(H2DStatus, H2DStatus.GPU_READY)),
                FieldValue("result_code", _enum_value(ResultCode, result_code)),
                FieldValue("destination_bank_id", destination_bank_id),
                FieldValue("copied_blocks", copied_blocks),
            ),
        )

    def _validate_completed_target(self, fence: OperationFence, *, bank_id: int, bank_epoch: int) -> None:
        BANK_ID.validate(bank_id)
        engine = self._table.partition(StateChangeBlockKind.REQUEST_ENGINE).read_stable(
            fence.request.request_slot,
            field_names=("request_epoch",),
        )
        target = self._table.partition(StateChangeBlockKind.REQUEST_TARGET_COMPUTE).read_stable(
            fence.request.request_slot,
            field_names=("request_epoch", "round_id", "observed_run_seq", "target_id",
                        "target_generation", "bank_id", "bank_epoch"),
        )
        _expect(engine.get("request_epoch"), fence.request.request_epoch, "stale request_epoch for D2H fact")
        _expect(target.get("request_epoch"), fence.request.request_epoch, "stale TargetCompute request_epoch")
        _expect(target.get("round_id"), fence.round_id, "stale TargetCompute round_id")
        _expect(target.get("observed_run_seq"), fence.op_seq, "stale TargetCompute run sequence")
        _expect(target.get("target_id"), fence.worker_id, "wrong TargetCompute target_id")
        _expect(target.get("target_generation"), fence.worker_generation, "stale TargetCompute generation")
        _expect(target.get("bank_id"), bank_id, "wrong TargetCompute source bank_id")
        _expect(target.get("bank_epoch"), bank_epoch, "stale TargetCompute source bank_epoch")

    def _validate_prepare_fence(self, fence: OperationFence, *, bank_id: int, bank_epoch: int) -> None:
        BANK_ID.validate(bank_id)
        dispatch = self._table.partition(StateChangeBlockKind.REQUEST_DISPATCH).read_stable(
            fence.request.request_slot,
            field_names=("target_prepare_seq", "planned_target_id", "planned_target_generation",
                        "planned_bank_id", "planned_bank_epoch"),
        )
        _expect(dispatch.get("target_prepare_seq"), fence.op_seq, "stale target_prepare_seq for H2D fact")
        _expect(dispatch.get("planned_target_id"), fence.worker_id, "wrong Target worker_id for H2D fact")
        _expect(dispatch.get("planned_target_generation"), fence.worker_generation, "stale Target generation for H2D")
        _expect(dispatch.get("planned_bank_id"), bank_id, "wrong destination bank_id for H2D fact")
        _expect(dispatch.get("planned_bank_epoch"), bank_epoch, "stale destination bank_epoch for H2D")
