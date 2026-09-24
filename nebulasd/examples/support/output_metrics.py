"""Client-visible output timing; a speculative chunk is not a per-token clock."""
from nebulasd.observability.copy_timing import percentiles


class OutputMetrics:
    def __init__(self):
        self.admitted,self.last,self.completed = {},{},{}
        self.outputs,self.chunks,self.ttft,self.intervals = {},[],[],[]

    def admit(self,request,now):
        self.admitted[request.request_id] = now
        self.outputs[request.request_id] = []

    def observe(self,event,now):
        identity = event.request.request_id
        if event.token_ids:
            if identity in self.last:
                self.intervals.append((now-self.last[identity])/1e6)
            else:
                self.ttft.append((now-self.admitted[identity])/1e6)
            self.last[identity] = now
            self.outputs[identity].extend(event.token_ids)
        if event.finished:
            self.completed[identity] = now
        self.chunks.append(dict(request_id=identity,observed_ns=now,tokens=event.token_ids,
                                lifecycle=event.lifecycle.name))

    def summary(self,start,end):
        elapsed = (end-start)/1e9
        return dict(elapsed_ms=elapsed*1000,output_tokens_per_second=sum(map(len,self.outputs.values()))/elapsed,
            requests_per_second=len(self.completed)/elapsed,ttft=percentiles(self.ttft),
            output_chunk_interval=percentiles(self.intervals),
            request_latency=percentiles([(t-self.admitted[k])/1e6 for k,t in self.completed.items()]),
            actual_chunk_sizes=sorted({len(r['tokens']) for r in self.chunks if r['tokens']}))
