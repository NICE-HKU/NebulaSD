"""Quiescence requires all authorizations and frozen sends to retire."""
from nebulasd.core.enums import Lifecycle

def quiescent(engine):
    return (not any(r.lifecycle == Lifecycle.ACTIVE for r in engine.registry.records.values())
            and not engine.ledger.records and not any(engine.scheduling_progress.pending.values()))
