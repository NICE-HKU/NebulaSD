"""Explicit payload attachments for a quiescent worker/cohort startup.

The segment schema identifies its original mapping; generation identifies the
current handle namespace, which can advance at a completed cohort barrier.
"""
from dataclasses import dataclass
from nebulasd.core.ids import ARENA_GENERATION
from nebulasd.data.shared_arenas import ArenaRouter


@dataclass(frozen=True)
class PayloadDescriptor:
    segment: object
    generation: int

    @classmethod
    def of(cls, arena):
        return cls(arena.segment.descriptor, arena.generation)

    def attach(self, arena_type, *, writer=False):
        kind, capacity, initial_generation = self.segment.schema.split(':')
        if kind != arena_type.__name__:
            raise ValueError('payload descriptor type mismatch')
        ARENA_GENERATION.validate(self.generation)
        arena = arena_type(int(capacity), generation=int(initial_generation), descriptor=self.segment, writer=writer)
        arena._generation = self.generation
        if writer:
            arena._head = arena._native.sd_load(arena.segment.address)
        return arena


def attach_router(descriptors, arena_type):
    if len({d.generation for d in descriptors}) != len(descriptors):
        raise ValueError('ambiguous payload generation namespace')
    attachments = []
    try:
        for descriptor in descriptors:
            attachments.append(descriptor.attach(arena_type))
        return ArenaRouter(attachments), attachments
    except BaseException:
        for arena in attachments:
            arena.close()
        raise
