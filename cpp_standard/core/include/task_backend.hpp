// SPDX-License-Identifier: MIT
// Copyright (c) 2024 The Cholesky-Bench Contributors

#ifndef TASK_BACKEND_H
#define TASK_BACKEND_H

#pragma once

// Parallelism backend built purely on the C++ standard library, designed for an
// apples-to-apples comparison with the HPX implementation. It mirrors the small
// slice of the HPX API that the Cholesky sources rely on:
//
//   hpx::make_ready_future        -> std_backend::make_ready_future
//   hpx::dataflow                 -> std_backend::dataflow
//   hpx::wait_all                 -> std_backend::wait_all
//   hpx::experimental::for_loop   -> std_backend::parallel_for
//                                      (std::for_each + std::execution::par)
//
// Tasking is expressed with std::async (std::launch::async): each dataflow
// call launches a task that waits on its dependency futures and then performs
// its work. The OS/runtime thread pool is implementation-defined.
//
// The parallel fork-join (parallel_for) uses std::execution::par, which on
// libstdc++ is backed by Intel TBB.

#include <algorithm>
#include <cstddef>
#include <execution>
#include <future>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

namespace std_backend
{

/**
 * @brief Create an already-satisfied shared_future holding @p value.
 */
template <typename T>
std::shared_future<std::decay_t<T>> make_ready_future(T &&value)
{
    std::promise<std::decay_t<T>> p;
    p.set_value(std::forward<T>(value));
    return p.get_future().share();
}

/**
 * @brief Create an already-satisfied shared_future<void>.
 */
inline std::shared_future<void> make_ready_future()
{
    std::promise<void> p;
    p.set_value();
    return p.get_future().share();
}

/**
 * @brief std::library analogue of hpx::dataflow.
 *
 * Launches the callable @p f asynchronously via std::async. Dependency
 * futures are forwarded as arguments and waited on inside the task body,
 * so downstream work only proceeds once all dependencies are ready.
 */
template <typename F, typename... Args>
auto dataflow(F &&f, Args &&...args)
{
    return std::async(std::launch::async,
                      [f = std::forward<F>(f), tup = std::make_tuple(std::forward<Args>(args)...)]() mutable
                      { return std::apply(f, tup); })
        .share();
}

/**
 * @brief Block until every valid future in @p futures is ready (hpx::wait_all).
 */
template <typename Container>
void wait_all(Container &futures)
{
    for (auto &f : futures)
    {
        if (f.valid())
        {
            f.wait();
        }
    }
}

/**
 * @brief Parallel index loop over [begin, end) using the parallel std::for_each.
 *
 * The standard-library fork-join primitive replacing
 * hpx::experimental::for_loop(hpx::execution::par, ...).
 */
template <typename F>
void parallel_for(std::size_t begin, std::size_t end, F &&f)
{
    if (begin >= end)
    {
        return;
    }
    std::vector<std::size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), begin);
    std::for_each(std::execution::par, indices.begin(), indices.end(), std::forward<F>(f));
}

}  // namespace std_backend

#endif  // end of TASK_BACKEND_H
