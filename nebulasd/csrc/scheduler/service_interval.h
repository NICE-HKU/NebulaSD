// Included after Invocation and Planner. Independent policy; no execution ownership.
struct ServiceIntervalPlanner {
    Planner &base;
    double gap;
    double delay;
    Groups run(const Batch &candidates, std::vector<const Worker *> workers) {
        auto &v = base.v;
        std::map<int64_t, double> ends;
        std::map<std::pair<bool, int64_t>, Batch> age_groups;
        bool draft = workers.front()->draft;
        for (auto slot : candidates) {
            auto &r = v.r(slot);
            auto round = draft ? r.x_target_round_id : r.x_draft_round_id;
            auto observed_round = draft ? r.target_compute_round : r.draft_compute_round;
            auto end_ns = draft ? r.target_compute_end_ns : r.draft_compute_end_ns;
            bool actual = end_ns >= 0 && observed_round == round;
            age_groups[{!actual, actual ? end_ns : 0}].push_back(slot);
            if (actual) {
                ends[slot] = end_ns / 1e9;
                base.times[slot] = std::max(v.now, ends[slot]);
            }
        }
        std::sort(workers.begin(), workers.end(), [](auto a, auto b) { return a->id < b->id; });
        Groups groups;
        std::set<int64_t> used;
        for (auto w : workers) {
            using Key = std::tuple<double, int64_t, int64_t, int64_t>;
            auto free = v.free(*w);
            Batch batch;
            for (auto &[age, group] : age_groups) {
                std::vector<std::pair<Key, int64_t>> ranked;
                for (auto slot : group) {
                    if (used.count(slot) || !base.fits(*w, Batch{slot}))
                        continue;
                    if (!base.times.count(slot)) {
                        ends[slot] = v.input(v.r(slot), draft, base.initial);
                        base.times[slot] = std::max(v.now, ends[slot]);
                    }
                    auto single = base.prediction(*w, Batch{slot});
                    auto delta = std::max(0., single.start - free);
                    auto &r = v.r(slot);
                    ranked.push_back({Key{delta, r.ready ? r.ready : r.admitted, r.arrival, slot}, slot});
                }
                std::sort(ranked.begin(), ranked.end());
                for (auto &[key, slot] : ranked) {
                    auto next = batch;
                    next.push_back(slot);
                    if (base.fits(*w, next))
                        batch = std::move(next);
                    if (int64_t(batch.size()) == w->max_batch)
                        break;
                }
                if (int64_t(batch.size()) == w->max_batch)
                    break;
            }
            v.metrics.frontier += batch.size();
            if (batch.empty())
                continue;
            std::map<int64_t, double> bounds;
            for (auto slot : batch)
                bounds[slot] = std::max(ends[slot] + gap, base.prediction(*w, Batch{slot}).start);
            Batch kept{batch.front()};
            double limit = bounds.at(batch.front()) + delay;
            for (size_t i = 1; i < batch.size(); ++i) {
                auto next_limit = std::min(limit, bounds.at(batch[i]) + delay);
                auto trial = kept;
                trial.push_back(batch[i]);
                if (base.prediction(*w, trial).start <= next_limit + 1e-9) {
                    kept = std::move(trial);
                    limit = next_limit;
                }
            }
            batch = std::move(kept);
            groups[w->id] = batch;
            used.insert(batch.begin(), batch.end());
        }
        return groups;
    }
};
