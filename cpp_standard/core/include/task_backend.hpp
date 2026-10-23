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
// Tasking is expressed with one detached std::thread per task: each dataflow
// call launches a task that waits on its dependency futures and then performs
// its work, so every task has an OS thread of its own.
//
// The parallel fork-join (parallel_for) uses std::execution::par, which on
// libstdc++ is backed by Intel TBB.

#include <algorithm>
#include <cstddef>
#include <execution>
#include <future>
#include <numeric>
#include <semaphore>
#include <thread>
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
 * @brief Upper bound on tasks that exist (and so hold an OS thread) at the same time.
 *
 * Every dataflow call launches one thread. Without a bound, a fine tiling (about 3.6e5 tasks at 128 tiles, 2.8e6 at
 * 256) would exhaust the thread limit. A thread stack takes two memory mappings (stack and guard page), so with the
 * default vm.max_map_count of 65530 a process can hold at most about 32000 threads; the bound stays well below that.
 * Submission blocks while the bound is reached. This cannot deadlock: tasks are submitted in dependency order, so
 * a live task only ever waits on tasks that were launched before it, and the oldest unfinished task always has all
 * of its inputs ready.
 */
inline constexpr std::ptrdiff_t max_tasks_in_flight = 16'384;

inline std::counting_semaphore<> &task_slots()
{
    // Never destroyed: a task thread may still be releasing its slot while the process shuts down.
    static auto *slots = new std::counting_semaphore<>(max_tasks_in_flight);
    return *slots;
}

/**
 * @brief std::library analogue of hpx::dataflow.
 *
 * Runs the callable @p f on a thread of its own. Dependency futures are forwarded as arguments and waited on inside
 * the task body, so downstream work only proceeds once all dependencies are ready. Blocks the caller while
 * max_tasks_in_flight tasks are alive.
 *
 * The thread is detached and reports through a promise instead of using std::async. A std::async thread stays
 * joinable, and keeps its stack, until the last future to its shared state is destroyed; the final version of every
 * tile holds such a future, so a fine tiling would leave more finished-but-unjoined threads than a process may have.
 */
template <typename F, typename... Args>
auto dataflow(F &&f, Args &&...args)
{
    using Result = std::invoke_result_t<std::decay_t<F> &, std::decay_t<Args> &&...>;

    std::promise<Result> promise;
    auto future = promise.get_future().share();

    task_slots().acquire();
    std::thread(
        [f = std::forward<F>(f),
         tup = std::make_tuple(std::forward<Args>(args)...),
         promise = std::move(promise)]() mutable
        {
            try
            {
                // Move the input futures out of the closure so they are released as soon as the task has run (as
                // hpx::dataflow does) instead of pinning every earlier tile version.
                auto inputs = std::move(tup);
                if constexpr (std::is_void_v<Result>)
                {
                    std::apply(f, std::move(inputs));
                    promise.set_value();
                }
                else
                {
                    promise.set_value(std::apply(f, std::move(inputs)));
                }
            }
            catch (...)
            {
                promise.set_exception(std::current_exception());
            }
            task_slots().release();
        })
        .detach();
    return future;
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
