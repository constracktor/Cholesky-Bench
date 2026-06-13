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
// Tasking is expressed with std::packaged_task / std::shared_future and is
// executed on a FIXED-SIZE worker pool of N threads (N = set_num_threads(),
// defaulting to hardware_concurrency). This matches HPX's `--hpx:threads=N`
// model: a bounded set of OS worker threads onto which an arbitrarily large
// task graph is multiplexed -- rather than one OS thread per task. The same N
// also caps the parallel std::for_each fork-join (via TBB's global_control when
// available), so both tasking and fork-join run on the same thread budget.
//
// Deadlock freedom: dataflow tasks block on their dependency futures inside the
// worker. This is safe on a bounded FIFO pool because, in right-looking
// Cholesky, every task depends only on tasks submitted earlier. With a single
// FIFO queue, earlier tasks are always dequeued first, so the earliest in-flight
// task has all dependencies satisfied and runs to completion -- guaranteeing
// forward progress for any pool size N >= 1.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <execution>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#if __has_include(<tbb/global_control.h>)
#include <optional>
#include <tbb/global_control.h>
#define STD_BACKEND_HAVE_TBB_GLOBAL_CONTROL 1
#endif

namespace std_backend
{

/**
 * @brief Minimal fixed-size FIFO thread pool returning std::shared_future.
 */
class thread_pool
{
  public:
    explicit thread_pool(std::size_t n) : stop_(false)
    {
        if (n == 0)
        {
            n = 1;
        }
        workers_.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~thread_pool()
    {
        {
            std::unique_lock<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &t : workers_)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }

    thread_pool(const thread_pool &) = delete;
    thread_pool &operator=(const thread_pool &) = delete;

    std::size_t size() const { return workers_.size(); }

    /// Submit a nullary callable; returns a shared_future for its result.
    template <typename F>
    auto submit(F &&f) -> std::shared_future<std::invoke_result_t<std::decay_t<F>>>
    {
        using R = std::invoke_result_t<std::decay_t<F>>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::shared_future<R> fut = task->get_future().share();
        {
            std::unique_lock<std::mutex> lk(m_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

  private:
    void worker_loop()
    {
        for (;;)
        {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty())
                {
                    return;
                }
                job = std::move(tasks_.front());
                tasks_.pop();
            }
            job();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_;
};

namespace detail
{
// 0 means "use hardware_concurrency()". Read once, when the pool is created.
inline std::atomic<std::size_t> &desired_threads()
{
    static std::atomic<std::size_t> n{ 0 };
    return n;
}
}  // namespace detail

/// Resolve the effective worker count (mirrors HPX's get_num_worker_threads()).
inline std::size_t resolve_threads()
{
    std::size_t n = detail::desired_threads().load();
    if (n == 0)
    {
        n = std::thread::hardware_concurrency();
    }
    return n == 0 ? 1 : n;
}

/// Lazily-created global worker pool, sized by the most recent set_num_threads().
inline thread_pool &global_pool()
{
    static thread_pool pool(resolve_threads());
    return pool;
}

/**
 * @brief Set the worker-thread budget for BOTH the tasking pool and the
 *        parallel std::for_each fork-join. Call once at startup, before any
 *        dataflow()/parallel_for() use.
 */
inline void set_num_threads(std::size_t n)
{
    detail::desired_threads().store(n);
#ifdef STD_BACKEND_HAVE_TBB_GLOBAL_CONTROL
    static std::optional<tbb::global_control> gc;
    const std::size_t eff = (n == 0) ? resolve_threads() : n;
    gc.emplace(tbb::global_control::max_allowed_parallelism, eff);
#endif
}

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
 * The callable @p f is submitted to the global worker pool. Its shared_future
 * dependency arguments are forwarded unchanged; @p f .get()/.wait()s them
 * internally, so the body's real work only proceeds once the dependencies are
 * ready. The returned shared_future is the dependency token for downstream
 * tasks, reproducing HPX's continuation-style dataflow ordering on a bounded
 * pool. (std::make_tuple unwraps std::ref/std::cref into plain references, so
 * in-place tile arguments are passed by reference exactly as before.)
 */
template <typename F, typename... Args>
auto dataflow(F &&f, Args &&...args)
{
    auto bound = [f = std::forward<F>(f), tup = std::make_tuple(std::forward<Args>(args)...)]() mutable
    { return std::apply(f, tup); };
    return global_pool().submit(std::move(bound));
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
 * hpx::experimental::for_loop(hpx::execution::par, ...). The worker count is
 * bounded by set_num_threads() through TBB's global_control (libstdc++'s par
 * backend), matching the tasking pool's budget.
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
