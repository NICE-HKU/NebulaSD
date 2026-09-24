from dataclasses import dataclass
from nebulasd.workers.channel import LocalChannel


@dataclass(frozen=True)
class Row:
    index: int = 0
    accepted: int = 1
    logical: int = 9
    version: int = 3
    dirty_begin: int = 0
    dirty_blocks: int = 1
    tokens: tuple = (11, 12)


def test_binary_payload_and_zero_payload_control_across_ring_wraps():
    sender = LocalChannel()
    receiver = LocalChannel(sender.descriptor)
    try:
        for seq in range(40):
            before = sender.bytes_sent
            sender.put_nowait(('RETIRE_RECORD', seq))
            assert sender.bytes_sent - before == 16
            assert receiver.get_nowait() == ('RETIRE_RECORD', seq)
            sender.put_nowait(('RESULT', seq, dict(rows=(Row(),), compute_start_ns=1, compute_end_ns=2)))
            kind, received_seq, result = receiver.get_nowait()
            assert (kind, received_seq) == ('RESULT', seq)
            assert result['rows'][0]['tokens'] == (11, 12)
            assert result['rows'][0]['version'] == 3
            assert sender.bytes_sent - before == 16 + 16 + 20 + 32 + 8
            assert receiver.bytes_received == sender.bytes_sent
    finally:
        receiver.close()
        sender.close(unlink=True)


def test_binary_import_and_physical_facts():
    sender = LocalChannel()
    receiver = LocalChannel(sender.descriptor)
    try:
        messages = (
            ('IMPORTED', 7, dict(submitted_ns=12, rows=[dict(index=1, version=19, blocks=3)])),
            ('PHYSICAL', 7, dict(d2h_submitted_ns=31, observed_ns=42, outcomes=(1, 2),
                                facts=[('D2H_DONE', 40), ('BANK_FREE', 41)])),
        )
        for message in messages:
            sender.put_nowait(message)
            assert receiver.get_nowait() == message
    finally:
        receiver.close()
        sender.close(unlink=True)
