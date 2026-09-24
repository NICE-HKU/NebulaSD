// Autonomous completion scheduler. No Python callbacks or shared-table writes.
#include "cost.h"
#include "scheduler_schema.h"
#include <array>
#include <chrono>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
using Batch = std::vector<int64_t>;
using Groups = std::map<int64_t, Batch>;
using Menu = std::vector<Batch>;
static int64_t zero(int64_t x) { return x < 0 ? 0 : x; }
static int bits(int64_t x) {
    int n = 0;
    while (x > 0) {
        ++n;
        x >>= 1;
    }
    return n;
}
struct Prediction {
    double input, free, kv, compute, start, finish, cost;
};
struct Output {
    int64_t worker, initial, offset, count;
    Prediction prediction;
};
struct Metrics {
    int64_t scanned = 0, frontier = 0, limit = 0, cells = 0, fits = 0, partitions = 0, blocked = 0;
};
struct Engine {
    Cost target, draft;
    int64_t block_bytes, draft_bytes;
};
struct Invocation {
    Engine &engine;
    std::vector<const Request *> requests;
    std::unordered_map<int64_t, const Request *> by_slot;
    std::vector<Worker> workers;
    std::vector<std::pair<Record, Batch>> records;
    int64_t now_ns;
    double now;
    bool direct;
    bool ignore_kv_time = false;
    Metrics metrics;
    std::map<int64_t, double> free_cache, copy_cache;
    const Request &r(int64_t slot) const { return *by_slot.at(slot); }
    const Worker *worker(int64_t id) const {
        for (auto &w : workers)
            if (w.id == id)
                return &w;
        return nullptr;
    }
    bool online(int64_t id, int64_t generation) const {
        auto w = worker(id);
        return w && w->online && w->generation == generation;
    }
    bool active(const Request &r) const {
        return r.e_request_epoch == r.epoch && r.e_lifecycle == 1;
    }
    bool target_result(const Request &r) const {
        return active(r) && online(r.t_target_id, r.t_target_generation) && r.t_result_code == 0 &&
               r.t_status == 2 && r.t_request_epoch == r.epoch &&
               r.t_round_id == r.x_target_round_id && r.t_observed_run_seq == r.x_target_run_seq &&
               r.output > 0 && r.e_current_round_id == r.t_round_id;
    }
    bool draft_ready(const Request &r) const {
        return active(r) && online(r.d_worker_id, r.d_worker_generation) &&
               r.d_request_epoch == r.epoch && r.d_round_id == r.x_draft_round_id &&
               r.d_status == 2 && r.d_result_code == 0 && r.x_draft_worker_id == r.d_worker_id &&
               r.x_draft_worker_generation == r.d_worker_generation &&
               r.x_draft_issue_seq == r.d_observed_issue_seq;
    }
    bool source(const Request &r) const {
        return r.d_request_epoch == r.epoch && r.d_status == 2 && r.d_result_code == 0 &&
               r.d_round_id == r.x_draft_round_id &&
               r.d_observed_issue_seq == r.x_draft_issue_seq &&
               r.d_worker_id == r.x_draft_worker_id &&
               r.d_worker_generation == r.x_draft_worker_generation &&
               r.d_owner_epoch == r.x_draft_owner_epoch && r.a_request_epoch == r.epoch;
    }
    bool host_ready(const Request &r, bool d) const {
        if (!zero(d ? r.x_draft_issue_seq : r.x_target_run_seq))
            return true;
        return d ? r.dh_request_epoch == r.epoch && r.dh_status == 2 &&
                       r.dh_ready_version == r.d_snapshot_version
                 : r.h_request_epoch == r.epoch && r.h_status == 2 &&
                       r.h_ready_version == r.t_target_kv_version;
    }
    int cheap(const Request &r, bool d, bool initial, bool prepare) const {
        if (!d) {
            if (!zero(r.x_target_run_seq))
                return initial ? 1 : -1;
            if (!prepare || r.x_target_round_id != r.t_round_id || r.t_status != 2 ||
                !zero(r.x_draft_issue_seq) ||
                r.x_draft_round_id != (r.t_round_id < 0 ? -2 : r.t_round_id) + 1)
                return -1;
            return r.d_status == 2 && r.d_round_id == r.x_draft_round_id;
        }
        if (!zero(r.x_draft_issue_seq))
            return initial && r.output && r.t_status == 2 ? 1 : -1;
        if (!prepare || r.x_draft_round_id != r.d_round_id || r.d_status != 2 ||
            !zero(r.x_target_run_seq) || r.x_target_round_id != r.d_round_id)
            return -1;
        return r.t_status == 2 && r.t_round_id == r.d_round_id;
    }
    double duration(int stage, const Batch &batch, int sync = 0) {
        if (batch.empty())
            return 0;
        int64_t kv = 0, depth = 0;
        for (auto slot : batch) {
            auto &q = r(slot);
            kv = std::max(kv, stage == 0 ? q.prompt : q.prompt + q.output);
            depth = std::max(depth, std::min(q.depth, q.maximum - q.output));
        }
        if (stage == 0)
            depth = 0;
        else
            depth = std::max<int64_t>(1, depth);
        if (stage == 1)
            kv = std::max<int64_t>(0, kv - 1);
        return engine.target.predict(stage, batch.size(), kv, depth, sync);
    }
    double copy(int direction, int64_t bytes, int64_t batch = 1, bool d = false) {
        return !bytes ? 0 : (d ? engine.draft : engine.target).predict(direction, batch, bytes);
    }
    double free(const Worker &w) {
        auto old = free_cache.find(w.id);
        if (old != free_cache.end())
            return old->second;
        double seconds = 0;
        for (auto &[rec, batch] : records)
            if (rec.worker == w.id && !rec.compute_done) {
                int stage = rec.operation == 1   ? 0
                            : rec.operation == 3 ? 1
                            : rec.operation == 2 ? 2
                                                 : 3;
                double service = duration(stage, batch, stage == 3 ? 2 : 0);
                if (w.runtime_seq == rec.seq && zero(w.runtime_start))
                    service = w.runtime_status == 1
                                  ? std::max(0., service - double(now_ns - w.runtime_start) / 1e9)
                                  : 0.;
                seconds += service;
            }
        return free_cache[w.id] = now + seconds;
    }
    double copy_free(int64_t id) {
        auto old = copy_cache.find(id);
        if (old != copy_cache.end())
            return old->second;
        double seconds = 0;
        for (auto &[rec, batch] : records)
            if (rec.worker == id && !rec.physical_done) {
                bool d = rec.operation == 2 || rec.operation == 4;
                auto bytes = d ? engine.draft_bytes : engine.block_bytes;
                seconds += copy(4, rec.h2d_blocks * 2 * bytes, rec.h2d_rows, d);
                seconds += copy(5, rec.d2h_blocks * 2 * bytes, rec.d2h_rows, d);
            }
        return copy_cache[id] = now + seconds;
    }
    double input(const Request &r, bool d, bool initial) {
        if (!d && initial)
            return now;
        if (d ? target_result(r) : draft_ready(r))
            return now;
        auto w = worker(d ? (r.x_planned_target_id < 0 ? r.t_target_id : r.x_planned_target_id)
                          : r.x_draft_worker_id);
        return w ? free(*w) : now;
    }
    Prediction predict(const Worker &w, const Batch &batch, bool initial,
                       const std::map<int64_t, double> &times) {
        bool d = w.draft;
        double in = now, compute = 0, kv = now;
        for (auto slot : batch)
            in = std::max(in, times.at(slot));
        if (d) {
            Batch first, cached;
            for (auto slot : batch)
                (zero(r(slot).x_draft_issue_seq) ? cached : first).push_back(slot);
            compute = duration(2, first) + duration(3, cached, 2);
        } else
            compute = duration(initial ? 0 : 1, batch);
        if (!initial && !ignore_kv_time) {
            int64_t h2d = 0;
            std::map<int64_t, int64_t> sources;
            for (auto slot : batch) {
                auto &q = r(slot);
                if (d) {
                    h2d += zero(q.d_valid_blocks) * engine.draft_bytes * 2;
                    bool matching = q.dh_request_epoch == q.epoch &&
                                    q.dh_snapshot_version == q.d_snapshot_version &&
                                    q.dh_source_op_seq == q.d_observed_issue_seq &&
                                    q.dh_owner_epoch == q.d_owner_epoch &&
                                    q.dh_source_worker_generation == q.d_worker_generation;
                    if (matching && q.dh_status == 2 && q.dh_ready_version == q.d_snapshot_version)
                        continue;
                    auto owner = worker(q.d_worker_id);
                    bool copying = matching && q.dh_status == 1 && owner && owner->copy_status == 1;
                    sources[q.d_worker_id] +=
                        copying ? 0
                                : zero(q.d_dirty_block_count < 0 ? q.d_valid_blocks
                                                                 : q.d_dirty_block_count) *
                                      engine.draft_bytes * 2;
                } else {
                    h2d += ((zero(q.t_logical_kv_len) + w.block_size - 1) / w.block_size) *
                           engine.block_bytes * 2;
                    if (q.h_status == 2 && q.h_ready_version == q.t_target_kv_version)
                        continue;
                    auto owner = worker(q.t_target_id);
                    bool copying =
                        q.h_status == 1 && q.h_request_epoch == q.epoch &&
                        q.h_round_id == q.t_round_id && q.h_d2h_op_seq == q.t_observed_run_seq &&
                        q.h_source_bank_id == q.t_bank_id &&
                        q.h_source_bank_epoch == q.t_bank_epoch && owner && owner->copy_status == 1;
                    sources[q.t_target_id] +=
                        copying ? 0 : zero(q.t_dirty_block_count) * engine.block_bytes * 2;
                }
            }
            for (auto [owner, bytes] : sources)
                kv = std::max(kv, copy_free(owner) + copy(5, bytes, 1, d));
            kv = std::max(kv, copy_free(w.id)) + copy(4, h2d, batch.size(), d);
        }
        double available = free(w), start = std::max({in, available, kv}), finish = start + compute,
               cost = 0;
        for (auto slot : batch)
            cost += finish - times.at(slot);
        return {in, available, kv, compute, start, finish, cost};
    }
    auto age(int64_t slot) const {
        auto &q = r(slot);
        return std::make_tuple(q.ready ? q.ready : q.admitted, q.arrival, q.slot);
    }
};
struct Planner {
    Invocation &v;
    bool initial;
    double weight;
    std::map<int64_t, double> times;
    std::map<std::pair<int64_t, Batch>, bool> fit_cache;
    std::map<std::pair<int64_t, Batch>, Prediction> predictions;
    bool fits(const Worker &w, const Batch &b) {
        auto key = std::make_pair(w.id, b);
        auto it = fit_cache.find(key);
        if (it != fit_cache.end())
            return it->second;
        ++v.metrics.fits;
        int64_t blocks = 0, tokens = 0;
        for (auto slot : b) {
            auto &r = v.r(slot);
            blocks += r.blocks;
            tokens += initial
                          ? (w.draft ? r.prompt + r.output + std::min(r.depth, r.maximum - r.output)
                                     : r.prompt)
                          : r.depth + 1;
        }
        return fit_cache[key] = int64_t(b.size()) <= (initial ? w.initial_rows : w.prepare_rows) &&
                                blocks <= (initial ? w.initial_blocks : w.prepare_blocks) &&
                                tokens <= (initial ? w.initial_tokens : w.prepare_tokens);
    }
    bool allowed(const Worker &w, const Batch &b) {
        double lo = INFINITY, hi = -INFINITY;
        int64_t oldest = INT64_MAX;
        for (auto slot : b) {
            lo = std::min(lo, times.at(slot));
            hi = std::max(hi, times.at(slot));
            auto &r = v.r(slot);
            oldest = std::min(oldest, r.ready ? r.ready : r.admitted);
        }
        return hi <= std::max(v.free(w), lo) + .002 && fits(w, b) &&
               (int64_t(b.size()) == w.max_batch || v.now_ns >= oldest);
    }
    Prediction prediction(const Worker &w, const Batch &b) {
        auto sorted = b;
        std::sort(sorted.begin(), sorted.end());
        auto key = std::make_pair(w.id, sorted);
        auto it = predictions.find(key);
        if (it != predictions.end())
            return it->second;
        ++v.metrics.cells;
        return predictions[key] = v.predict(w, b, initial, times);
    }
    std::vector<Menu> partitions(const Batch &admitted,
                                 const std::vector<const Worker *> &workers) {
        std::vector<Batch> orders{admitted, admitted, admitted};
        std::stable_sort(orders[1].begin(), orders[1].end(), [&](auto a, auto b) {
            return std::make_tuple(times[a], v.r(a).arrival, a) <
                   std::make_tuple(times[b], v.r(b).arrival, b);
        });
        std::stable_sort(orders[2].begin(), orders[2].end(), [&](auto a, auto b) {
            auto key = [&](auto s) {
                auto &r = v.r(s);
                return std::make_tuple(bits(r.prompt + r.output), r.depth, times[s], r.arrival, s);
            };
            return key(a) < key(b);
        });
        int64_t largest = 0;
        for (auto w : workers)
            largest = std::max(largest, w->max_batch);
        std::vector<int64_t> caps;
        for (auto n :
             {largest, std::max<int64_t>(1, largest / 2),
              std::max<int64_t>(1, (admitted.size() + workers.size() - 1) / workers.size()),
              int64_t(1)})
            if (std::find(caps.begin(), caps.end(), n) == caps.end())
                caps.push_back(n);
        std::vector<Menu> result;
        std::set<Menu> seen;
        for (int order = 0; order < 3; order++)
            for (auto cap : caps) {
                Menu batches;
                Batch batch;
                double lo = INFINITY, hi = -INFINITY;
                bool broke = false;
                for (auto slot : orders[order]) {
                    bool compatible = order == 0 || batch.empty() ||
                                      std::max(hi, times[slot]) - std::min(lo, times[slot]) <= .002;
                    Batch next = batch;
                    next.push_back(slot);
                    bool any = false;
                    for (auto w : workers)
                        if (fits(*w, next)) {
                            any = true;
                            break;
                        }
                    if (!batch.empty() && (int64_t(batch.size()) == cap || !compatible || !any)) {
                        batches.push_back(batch);
                        batch.clear();
                        lo = INFINITY;
                        hi = -INFINITY;
                        if (batches.size() >= workers.size()) {
                            broke = true;
                            break;
                        }
                    }
                    batch.push_back(slot);
                    lo = std::min(lo, times[slot]);
                    hi = std::max(hi, times[slot]);
                }
                if (!broke && !batch.empty())
                    batches.push_back(batch);
                size_t n = 0;
                for (auto &b : batches)
                    n += b.size();
                if (n == admitted.size() && batches.size() <= workers.size() &&
                    seen.insert(batches).second)
                    result.push_back(batches);
            }
        return result;
    }
    Groups once(const Batch &candidates, std::vector<const Worker *> workers) {
        predictions.clear();
        std::sort(workers.begin(), workers.end(), [](auto a, auto b) { return a->id < b->id; });
        auto ordered_workers = workers;
        std::stable_sort(ordered_workers.begin(), ordered_workers.end(), [&](auto a, auto b) {
            return std::make_pair(v.free(*a), a->id) < std::make_pair(v.free(*b), b->id);
        });
        std::vector<std::pair<int64_t, Batch>> seed;
        std::set<int64_t> used;
        Batch admitted;
        for (auto w : ordered_workers) {
            Batch order;
            for (auto s : candidates)
                if (!used.count(s))
                    order.push_back(s);
            double available = v.free(*w), high = INFINITY;
            std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                return std::make_pair(std::max(available, times[a]), v.age(a)) <
                       std::make_pair(std::max(available, times[b]), v.age(b));
            });
            Batch batch;
            for (auto s : order) {
                double ready = std::max(available, times[s]);
                if (!batch.empty() && ready > high)
                    break;
                Batch next = batch;
                next.push_back(s);
                if (fits(*w, next)) {
                    if (batch.empty())
                        high = ready + .002;
                    batch.push_back(s);
                    used.insert(s);
                    if (int64_t(batch.size()) == w->max_batch)
                        break;
                }
            }
            if (!batch.empty()) {
                seed.emplace_back(w->id, batch);
                admitted.insert(admitted.end(), batch.begin(), batch.end());
            }
        }
        if (admitted.empty())
            return {};
        Menu first;
        for (auto &[w, b] : seed)
            first.push_back(b);
        std::vector<Menu> menu{first};
        auto rest = partitions(admitted, workers);
        menu.insert(menu.end(), rest.begin(), rest.end());
        bool found = false;
        double best_cost = 0;
        Groups best;
        std::set<Menu> seen;
        for (auto &batches : menu) {
            if (!seen.insert(batches).second)
                continue;
            ++v.metrics.partitions;
            if (batches.size() > workers.size())
                continue;
            using State = std::pair<double, std::vector<int>>;
            std::map<uint64_t, State> states{{0, {0., {}}}};
            for (auto &batch : batches) {
                std::vector<double> costs;
                for (auto w : workers) {
                    if (!allowed(*w, batch)) {
                        costs.push_back(INFINITY);
                        continue;
                    }
                    auto p = prediction(*w, batch);
                    costs.push_back(p.cost + (weight ? weight * p.compute : 0.));
                }
                std::map<uint64_t, State> next;
                for (auto &[mask, state] : states)
                    for (size_t i = 0; i < workers.size(); i++)
                        if (std::isfinite(costs[i]) && !(mask & (uint64_t(1) << i))) {
                            auto key = mask | (uint64_t(1) << i);
                            auto path = state.second;
                            path.push_back(i);
                            State value{state.first + costs[i], path};
                            auto old = next.find(key);
                            if (old == next.end() || value < old->second)
                                next[key] = value;
                        }
                if (workers.size() > 8 && next.size() > 64) {
                    std::vector<std::pair<uint64_t, State>> beam(next.begin(), next.end());
                    std::stable_sort(beam.begin(), beam.end(),
                                     [](auto &a, auto &b) { return a.second < b.second; });
                    next.clear();
                    for (size_t i = 0; i < 64; i++)
                        next.insert(beam[i]);
                }
                states = std::move(next);
            }
            if (states.empty())
                continue;
            auto state = std::min_element(states.begin(), states.end(), [](auto &a, auto &b) {
                             return a.second < b.second;
                         })->second;
            Groups groups;
            for (size_t i = 0; i < batches.size(); i++)
                groups[workers[state.second[i]]->id] = batches[i];
            if (!found || state.first < best_cost || (state.first == best_cost && groups < best)) {
                found = true;
                best_cost = state.first;
                best = groups;
            }
        }
        if (found)
            return best;
        for (auto &[id, b] : seed)
            if (allowed(*v.worker(id), b))
                best[id] = b;
        return best;
    }
    Groups run(Batch candidates, std::vector<const Worker *> workers) {
        for (auto s : candidates)
            times[s] = v.input(v.r(s), workers.front()->draft, initial);
        Groups all;
        while (!candidates.empty() && !workers.empty()) {
            auto selected = once(candidates, workers);
            if (selected.empty())
                break;
            std::set<int64_t> used;
            for (auto &[id, b] : selected) {
                all[id] = b;
                used.insert(b.begin(), b.end());
            }
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                            [&](auto s) { return used.count(s); }),
                             candidates.end());
            workers.erase(std::remove_if(workers.begin(), workers.end(),
                                         [&](auto w) { return selected.count(w->id); }),
                          workers.end());
        }
        return all;
    }
};
#include "service_interval.h"
static thread_local std::string error;
struct Projection {
    char *destination;
    const char *payload;
    const int32_t *fields; // source offset, destination offset, signed width
    int32_t count;
};
extern "C" {
void sd_scheduler_project(const Projection *updates, int count) {
    for (int i = 0; i < count; i++) {
        auto &u = updates[i];
        for (int j = 0; j < u.count; j++) {
            auto f = u.fields + j * 3;
            uint64_t value = 0;
            std::memcpy(&value, u.payload + f[0], std::abs(f[2]));
            if (f[2] == -4)
                value = int64_t(int32_t(value));
            std::memcpy(u.destination + f[1], &value, sizeof(value));
        }
    }
}

int sd_scheduler_size(int kind) {
    return kind == 0   ? sizeof(Request)
           : kind == 1 ? sizeof(Worker)
           : kind == 2 ? sizeof(Record)
           : kind == 3 ? sizeof(Output)
                       : sizeof(Metrics);
}
const char *sd_scheduler_error() { return error.c_str(); }
void *sd_scheduler_create(const CostRow *target, int nt, int64_t context, const CostRow *draft,
                          int nd, int64_t draft_context, int64_t bytes, int64_t draft_bytes) {
    try {
        auto e = new Engine;
        e->target.rows.assign(target, target + nt);
        e->target.context = context;
        e->draft.rows.assign(draft, draft + nd);
        e->draft.context = draft_context;
        e->block_bytes = bytes;
        e->draft_bytes = draft_bytes;
        return e;
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
void sd_scheduler_destroy(void *p) { delete static_cast<Engine *>(p); }
static int scheduler_run(void *ptr, const Request *const *requests, int nr, const Worker *workers,
                     int nw, const Record *records, int nrec, const int64_t *members, int64_t now,
                     int draft, int direct, int factor, double weight, Output *out, int64_t *slots,
                     Metrics *metrics, bool service_interval, double gap, double delay, bool ignore_kv_time = false) {
    try {
        if (nw > 63)
            throw std::runtime_error("native scheduler supports at most 63 workers");
        auto &e = *static_cast<Engine *>(ptr);
        e.target.cache.clear();
        e.draft.cache.clear();
        Invocation v{e};
        v.now_ns = now;
        v.now = now / 1e9;
        v.direct = direct;
        v.ignore_kv_time = ignore_kv_time;
        v.requests.assign(requests, requests + nr);
        v.workers.assign(workers, workers + nw);
        for (auto r : v.requests)
            v.by_slot[r->slot] = r;
        int offset = 0;
        for (int i = 0; i < nrec; i++) {
            Batch b;
            for (int j = 0; j < records[i].count; j++) {
                auto slot = members[offset++];
                if (v.by_slot.count(slot))
                    b.push_back(slot);
            }
            v.records.emplace_back(records[i], b);
        }
        std::vector<const Worker *> capable;
        bool allow_initial = false, allow_prepare = false;
        int64_t max_blocks = 0, limit = 0;
        for (auto &w : v.workers)
            if (w.draft == draft && w.online && (w.initial_rows > 0 || w.prepare_rows > 0)) {
                capable.push_back(&w);
                limit += w.max_batch * factor;
                max_blocks = std::max(max_blocks, w.bank_blocks);
                allow_initial |= w.initial_rows > 0;
                allow_prepare |= w.prepare_rows > 0;
            }
        v.metrics.limit = limit;
        if (capable.empty()) {
            *metrics = v.metrics;
            return 0;
        }
        std::array<Batch, 16> buckets;
        Batch service_candidates;
        for (auto r : v.requests) {
            ++v.metrics.scanned;
            if (r->blocks > max_blocks || !v.active(*r))
                continue;
            int c = v.cheap(*r, draft, allow_initial, allow_prepare);
            bool interval = service_interval && (draft || zero(r->x_target_run_seq));
            if (c < 0 || ((direct || interval) && !v.host_ready(*r, draft)))
                continue;
            if (interval) {
                service_candidates.push_back(r->slot);
                if (draft)
                    continue;
            }
            int age = std::min(
                7, bits(std::max<int64_t>(0, now - (r->ready ? r->ready : r->admitted)) / 1000000));
            auto &b = buckets[(7 - age) * 2 + !c];
            if (int64_t(b.size()) < limit)
                b.push_back(r->slot);
        }
        Batch frontier;
        for (auto &b : buckets) {
            auto n = std::min<int64_t>(b.size(), limit - frontier.size());
            frontier.insert(frontier.end(), b.begin(), b.begin() + n);
            if (int64_t(frontier.size()) == limit)
                break;
        }
        std::stable_sort(frontier.begin(), frontier.end(),
                         [&](auto a, auto b) { return v.age(a) < v.age(b); });
        if (service_interval && !draft)
            frontier.erase(std::remove_if(frontier.begin(), frontier.end(),
                                           [&](auto s) { return zero(v.r(s).x_target_run_seq); }),
                           frontier.end());
        v.metrics.frontier = frontier.size();
        frontier.insert(frontier.end(), service_candidates.begin(), service_candidates.end());
        Batch facts[2];
        for (auto slot : frontier) {
            auto &r = v.r(slot);
            bool initial = !zero(draft ? r.x_draft_issue_seq : r.x_target_run_seq);
            bool ok;
            if (initial)
                ok = allow_initial && (!draft || v.target_result(r));
            else if (draft)
                ok = v.source(r) && r.x_target_round_id == r.d_round_id && zero(r.x_target_run_seq);
            else
                ok = v.target_result(r) &&
                     v.online(r.x_draft_worker_id, r.x_draft_worker_generation) &&
                     zero(r.x_draft_issue_seq) && r.x_draft_round_id == r.t_round_id + 1;
            if (ok)
                facts[initial ? 0 : 1].push_back(slot);
        }
        int count = 0, used_slots = 0;
        std::set<int64_t> occupied;
        for (int pass = 0; pass < 2; pass++) {
            bool initial = pass == 0;
            std::vector<const Worker *> destinations;
            for (auto w : capable)
                if (!occupied.count(w->id) && (initial ? w->initial_rows : w->prepare_rows) > 0)
                    destinations.push_back(w);
            if (facts[pass].empty() || destinations.empty())
                continue;
            Planner planner{v, initial, weight};
            auto groups = service_interval && (draft || !initial)
                              ? ServiceIntervalPlanner{planner, gap, delay}.run(facts[pass], destinations)
                              : planner.run(facts[pass], destinations);
            for (auto w : destinations) {
                auto it = groups.find(w->id);
                if (it == groups.end()) {
                    v.metrics.blocked |= int64_t(1) << (w - v.workers.data());
                    continue;
                }
                auto &batch = it->second;
                auto p = v.predict(*w, batch, initial, planner.times);
                out[count++] = {w->id, initial, used_slots, int64_t(batch.size()), p};
                std::copy(batch.begin(), batch.end(), slots + used_slots);
                used_slots += batch.size();
                occupied.insert(w->id);
            }
        }
        *metrics = v.metrics;
        return count;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}

// Keep the existing entry point and its ABI; only the new policy uses the new symbol.
int sd_scheduler_run(void *ptr, const Request *const *requests, int nr, const Worker *workers,
                     int nw, const Record *records, int nrec, const int64_t *members, int64_t now,
                     int draft, int direct, int factor, double weight, Output *out, int64_t *slots,
                     Metrics *metrics) {
    return scheduler_run(ptr, requests, nr, workers, nw, records, nrec, members, now,
                         draft, direct, factor, weight, out, slots, metrics, false, 0., 0.);
}
int sd_scheduler_run_service_interval(void *ptr, const Request *const *requests, int nr,
                     const Worker *workers, int nw, const Record *records, int nrec,
                     const int64_t *members, int64_t now, int draft, int direct, int factor,
                     double weight, Output *out, int64_t *slots, Metrics *metrics, double gap) {
    return scheduler_run(ptr, requests, nr, workers, nw, records, nrec, members, now,
                         draft, direct, factor, weight, out, slots, metrics, true, gap, 0.);
}

// Versioned symbol: stale libraries fail to load rather than ignore the tolerance.
int sd_scheduler_run_service_interval_v2(void *ptr, const Request *const *requests, int nr,
                     const Worker *workers, int nw, const Record *records, int nrec,
                     const int64_t *members, int64_t now, int draft, int direct, int factor,
                     double weight, Output *out, int64_t *slots, Metrics *metrics,
                     double gap, double delay) {
    return scheduler_run(ptr, requests, nr, workers, nw, records, nrec, members, now,
                         draft, direct, factor, weight, out, slots, metrics, true, gap, delay);
}

// KV timing ablation is prediction-only. Eligibility and real WORK gates stay intact.
int sd_scheduler_run_service_interval_v3(void *ptr, const Request *const *requests, int nr,
                     const Worker *workers, int nw, const Record *records, int nrec,
                     const int64_t *members, int64_t now, int draft, int direct, int factor,
                     double weight, Output *out, int64_t *slots, Metrics *metrics,
                     double gap, double delay, int ignore_kv_time) {
    return scheduler_run(ptr, requests, nr, workers, nw, records, nrec, members, now,
                         draft, direct, factor, weight, out, slots, metrics, true, gap, delay,
                         ignore_kv_time != 0);
}

}
