"""Client lifecycle endpoints installed only when config.profiling is enabled."""
from time import perf_counter_ns


def attach_client(client, recorder):
    def record(name, request, now, **fields):
        state = client._records[request.request_id]
        recorder.record(name, now, keys=[[state.input.slot, state.input.epoch, None]],
                        request_id=request.request_id, **fields)

    submit = client.submit
    def submitted(*args, **kwargs):
        start = perf_counter_ns()
        request = submit(*args, **kwargs)
        record('client.submitted', request, start)
        return request
    client.submit = submitted

    on_tokens = client._on_tokens
    def output(identity, tokens, lifecycle):
        on_tokens(identity, tokens, lifecycle)
        request = client._handles[int(identity)]
        record('client.output_available', request, perf_counter_ns(),
               token_count=len(tokens), terminal=lifecycle.name != 'ACTIVE', lifecycle=lifecycle.name)
    client._engine.outputs.on_tokens = output

    def observed(event):
        record('client.output_observed', event.request, perf_counter_ns(),
               token_count=len(event.token_ids), terminal=event.finished, lifecycle=event.lifecycle.name)
    read = client.read
    def read_output(request):
        events = read(request)
        for event in events:
            observed(event)
        return events
    client.read = read_output

    stream = client.stream
    def stream_output(*args, **kwargs):
        for event in stream(*args, **kwargs):
            observed(event)
            yield event
    client.stream = stream_output

    cancel = client.cancel
    def cancelled(request):
        active = client._record(request).lifecycle.name == 'ACTIVE'
        result = cancel(request)
        if active:
            record('client.output_available', request, perf_counter_ns(),
                   token_count=0, terminal=True, lifecycle='CANCELLED')
        return result
    client.cancel = cancelled
