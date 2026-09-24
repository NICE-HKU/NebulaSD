"""Stable hot command tags shared by role-specific codecs."""
from enum import IntEnum

class CommandKind(IntEnum):
    WORK = 7
    DRAFT_BATCH = 1
    PREPARE_TARGET_BANK = 2
    RUN_TARGET_BATCH = 3
    TARGET_PREFILL_BATCH = 4

    PREPARE_DRAFT_BANK = 5
    RUN_DRAFT_BATCH = 6
