"""Cold policy selection; execution triggers and transport remain unchanged."""
from .completion import CompletionScheduler


def completion_scheduler(config, estimator):
    kwargs = dict(estimator=estimator, frontier_factor=config.scheduler_frontier_factor,
                  draft_service_weight=config.draft_service_weight,
                  target_service_weight=config.target_service_weight)
    if config.scheduler_policy == 'service_interval':
        from .service_interval import ServiceIntervalScheduler
        if config.scheduler_ignore_kv_time:
            from dataclasses import replace
            from .measured_placement import MeasuredPlacementEstimator
            if not isinstance(estimator, MeasuredPlacementEstimator):
                raise ValueError('ignoring KV prediction requires measured placement estimator')
            kwargs['estimator'] = replace(estimator, ignore_kv_time=True)
        return ServiceIntervalScheduler(**kwargs, draft_service_gap_ms=config.draft_service_gap_ms,
                                        target_service_gap_ms=config.target_service_gap_ms,
                                        service_batch_delay_ms=config.service_batch_delay_ms)
    return CompletionScheduler(**kwargs)
