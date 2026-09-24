"""Fixed metadata payloads in the existing shared single-writer arena protocol."""
from nebulasd.core.draft_contracts import DraftSnapshot
from nebulasd.core.handles import ArenaHandle
from .shared_arenas import _SharedPayload
from .token_arena import TokenArena


class SharedDraftSnapshotArena(_SharedPayload, TokenArena):
    def __init__(self, capacity_bytes, *, generation=1, descriptor=None, writer=False):
        super().__init__(capacity_bytes, generation=generation)
        self._attach(capacity_bytes, generation, descriptor, writer)

    def write_snapshot(self, snapshot):
        self._before_write()
        raw = snapshot.to_bytes()
        if self._head + len(raw) > self.capacity_bytes:
            raise ValueError('snapshot arena capacity exhausted')
        handle = ArenaHandle(self._head, len(raw), self.generation)
        self._bytes[self._head:self._head + len(raw)] = raw
        self._head += len(raw)
        self._commit()
        return handle

    def read_snapshot(self, handle, *, expected, expected_layout_id):
        self._validate_handle(handle)
        snapshot = DraftSnapshot.from_bytes(bytes(self._bytes[handle.offset:handle.end_offset]))
        snapshot.identity.validate_expected(expected)
        if snapshot.identity.allocation.layout_id != expected_layout_id:
            raise ValueError("snapshot layout does not match destination model/weights/KV layout")
        return snapshot


def validate_snapshot_payloads(snapshot, *, token_arenas, config_arenas, proposal_arenas):
    """Validate routed handles without materializing prompt/output history.

    Registries are process-local attachments keyed by globally assigned arena
    generation, not Python session pointers. Proposal/config reads are bounded.
    """
    for handle in (snapshot.prompt_handle, snapshot.committed_output_handle):
        token_arenas[handle.generation]._validate_handle(handle)
    config = config_arenas[snapshot.generation_config_handle.generation].read_config(snapshot.generation_config_handle)
    proposal = proposal_arenas[snapshot.proposal_handle.generation].read_proposal(snapshot.proposal_handle)
    if len(proposal.draft_token_ids) != snapshot.proposal_count:
        raise ValueError('snapshot proposal handle/count mismatch')
    if snapshot.proposal_count > config.proposal_depth:
        raise ValueError('snapshot proposal exceeds generation configuration')
    if snapshot.committed_output_count + snapshot.proposal_count > config.max_new_tokens:
        raise ValueError('snapshot exceeds output budget')
    return config, proposal
