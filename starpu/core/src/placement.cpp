// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "placement.hpp"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <vector>

namespace placement
{
namespace
{
struct Domain
{
    std::vector<int> workers;
    std::vector<std::uint32_t> mask;  // bitmap of the workers, as starpu_task::workerids
};

std::vector<Domain> domains;
std::vector<int> cpu_of_worker;

// NUMA node of a CPU (from sysfs); 0 when unknown.
int node_of_cpu(int cpu)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto &e : fs::directory_iterator("/sys/devices/system/cpu/cpu" + std::to_string(cpu), ec))
    {
        const std::string name = e.path().filename().string();
        if (name.rfind("node", 0) == 0 && name.size() > 4 && std::isdigit(static_cast<unsigned char>(name[4])))
        {
            return std::stoi(name.substr(4));
        }
    }
    return 0;
}

const Domain &domain_of_tile(std::size_t row) { return domains[row % domains.size()]; }
}  // namespace

void init()
{
    const int n = starpu_cpu_worker_get_count();
    std::map<int, std::vector<int>> by_node;
    cpu_of_worker.assign(n, 0);
    for (int w = 0; w < n; ++w)
    {
        cpu_of_worker[w] = starpu_worker_get_bindid(w);
        by_node[node_of_cpu(cpu_of_worker[w])].push_back(w);
    }
    const std::size_t words = (static_cast<std::size_t>(n) + 31) / 32;
    domains.clear();
    for (auto &[node, workers] : by_node)
    {
        Domain d;
        d.workers = std::move(workers);
        d.mask.assign(words, 0);
        for (int w : d.workers)
        {
            d.mask[w / 32] |= std::uint32_t{ 1 } << (w % 32);
        }
        domains.push_back(std::move(d));
    }
}

int n_workers() { return static_cast<int>(cpu_of_worker.size()); }

int owner_worker(std::size_t row, std::size_t col)
{
    const Domain &d = domain_of_tile(row);
    return d.workers[(col + row / domains.size()) % d.workers.size()];
}

void place_task(starpu_task *task, std::size_t row, std::size_t col)
{
    (void) col;
    // The mask is owned by this module and outlives every task.
    const Domain &d = domain_of_tile(row);
    task->workerids = const_cast<std::uint32_t *>(d.mask.data());
    task->workerids_len = static_cast<unsigned>(d.mask.size());
}

void bind_to_worker(int worker)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_of_worker[worker], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

}  // namespace placement
