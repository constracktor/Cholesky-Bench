#ifndef TASK_BACKEND_H
#define TASK_BACKEND_H

#pragma once

// Thin parallelism backend built purely on the C++ standard library. It mirrors
// the small slice of the HPX API that the HPX implementation relies on, so the
// Cholesky sources can read almost identically across the two back ends:
//
//   hpx::make_ready_future        -> std_backend::make_ready_future
//   hpx::dataflow                 -> std_backend::dataflow   (std::async based)
//   hpx::wait_all                 -> std_backend::wait_all
//   hpx::experimental::for_loop   -> std_backend::parallel_for
//                                      (std::for_each + std::execution::par)
//
// Tasking is expressed with std::future / std::shared_future and std::async;
// fork-join is expressed with the parallel std::for_each. There is no
// lightweight user-space scheduler here: every dataflow task is an OS thread
// that blocks on its dependency futures until they are ready. That is the
// honest standard-library equivalent of HPX's asynchronous many-task model and
// is exactly what makes the comparison interesting.

#include <algorithm>
#include <cstddef>
#include <execution>
#include <future>
#include <numeric>
#include <type_traits>
#include <vector>

namespace std_backend
{

/**
 * @brief Create an already-satisfied shared_future holding @p value.
 *        std::future has no make_ready_future, so we build one via a promise.
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
 * @brief std::async-based emulation of hpx::dataflow.
 *
 * The callable @p f is launched asynchronously and receives its shared_future
 * dependency arguments by value; @p f then .get()/.wait()s them internally, so
 * the real work only starts once every dependency is ready. The returned
 * shared_future becomes the dependency token for downstream tasks, reproducing
 * HPX's continuation-style dataflow ordering on top of the standard library.
 *
 * std::launch::async is requested explicitly so each task is a genuine
 * concurrent thread rather than a deferred (lazy) computation.
 */
template <typename F, typename... Args>
auto dataflow(F &&f, Args &&...args)
{
    return std::async(std::launch::async, std::forward<F>(f), std::forward<Args>(args)...).share();
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
 * This is the standard-library fork-join primitive that replaces
 * hpx::experimental::for_loop(hpx::execution::par, ...). The index range is
 * materialized once and handed to std::for_each with std::execution::par; the
 * implementation (libstdc++/TBB, libc++/...) owns the chunking and load
 * balancing, so there is no direct analogue of HPX's dynamic_chunk_size(1).
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
