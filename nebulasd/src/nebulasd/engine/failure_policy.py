"""Fail fast by stopping every autonomous owner."""
class StopEnginePolicy:
    def fail(self, supervisor):
        supervisor.close()
