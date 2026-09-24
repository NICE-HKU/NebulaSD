#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>
struct CostRow {
    int64_t stage, depth, sync, batch, shape;
    double p50, p95;
};
struct Cost {
    std::vector<CostRow> rows;
    int64_t context = -1;
    // Invocation-local exact query memoization; no clocks persist here.
    std::map<std::tuple<int, int64_t, int64_t, int64_t, int64_t>, double> cache;
    double predict(int stage, int64_t batch, int64_t shape, int64_t depth = 0, int64_t sync = 0) {
        if (stage < 4 && context >= 0)
            shape = context;
        auto key = std::make_tuple(stage, batch, shape, depth, sync);
        auto found = cache.find(key);
        if (found != cache.end())
            return found->second;
        int64_t upper = INT64_MAX;
        for (auto &r : rows)
            if (r.stage == stage && r.sync == sync && r.depth >= depth)
                upper = std::min(upper, r.depth);
        if (upper == INT64_MAX)
            throw std::runtime_error("uncalibrated native cost family");
        std::vector<const CostRow *> family;
        for (auto &r : rows)
            if (r.stage == stage && r.sync == sync && r.depth == upper)
                family.push_back(&r);
        auto save = [&](double ms) { return cache[key] = ms / 1000.; };
        for (auto r : family)
            if (r->batch == batch && r->shape == shape)
                return save(r->p50);
        int64_t lo[2] = {-1, -1}, hi[2] = {INT64_MAX, INT64_MAX}, point[2] = {batch, shape};
        for (auto r : family) {
            int64_t p[2] = {r->batch, r->shape};
            for (int i = 0; i < 2; i++) {
                if (p[i] <= point[i])
                    lo[i] = std::max(lo[i], p[i]);
                if (p[i] >= point[i])
                    hi[i] = std::min(hi[i], p[i]);
            }
        }
        if (lo[0] >= 0 && lo[1] >= 0 && hi[0] != INT64_MAX && hi[1] != INT64_MAX) {
            double sum = 0;
            bool complete = true;
            std::vector<int64_t> xs{lo[0]}, ys{lo[1]};
            if (hi[0] != lo[0])
                xs.push_back(hi[0]);
            if (hi[1] != lo[1])
                ys.push_back(hi[1]);
            for (auto x : xs)
                for (auto y : ys) {
                    auto it = std::find_if(family.begin(), family.end(),
                                           [&](auto r) { return r->batch == x && r->shape == y; });
                    if (it == family.end()) {
                        complete = false;
                        continue;
                    }
                    double weight = 1.;
                    int64_t p[2] = {x, y};
                    for (int i = 0; i < 2; i++)
                        if (lo[i] != hi[i])
                            weight *= double(p[i] == lo[i] ? hi[i] - point[i] : point[i] - lo[i]) /
                                      double(hi[i] - lo[i]);
                    sum += weight * (*it)->p50;
                }
            if (complete)
                return save(sum);
        }
        if (stage < 4 && context >= 0)
            throw std::runtime_error(
                "fixed-context experiment requires measured/interpolated batch");
        const CostRow *best = nullptr;
        for (auto r : family)
            if (r->batch >= batch && r->shape >= shape && (!best || r->p95 < best->p95))
                best = r;
        if (best)
            return save(best->p50);
        for (auto r : family)
            if (!best || r->p95 > best->p95)
                best = r;
        return save(best->p50 * std::max({1., double(batch) / std::max<int64_t>(best->batch, 1),
                                          double(shape) / std::max<int64_t>(best->shape, 1)}));
    }
};
