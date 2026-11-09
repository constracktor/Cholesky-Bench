// SPDX-License-Identifier: MIT
// Copyright (c) 2024 The Cholesky-Bench Contributors

#ifndef TASK_BACKEND_H
#define TASK_BACKEND_H

#pragma once

// Parallelism backend built purely on the C++ standard library, designed for an
// apples-to-apples comparison with the HPX implementation. It mirrors the small
// slice of the HPX API that the Cholesky sources rely on:
//
//   hpx::shared_future            -> std_backend::future
//   hpx::make_ready_future        -> std_backend::make_ready_future
//   hpx::dataflow                 -> std_backend::dataflow_on_tile
//   hpx::wait_all                 -> std_backend::wait_all
//   hpx::experimental::for_loop   -> std_backend::parallel_for
//
// Tasking runs on a fixed pool of worker threads (one per core). dataflow() is continuation based: a task counts its
// unsatisfied inputs and is pushed to a ready queue by whichever thread finishes the last one, so no worker ever
// blocks on a future and no thread is created per task. Every worker owns two deques (normal and high priority); it
// pops its own newest task and steals the oldest task of another worker when it runs dry, workers of its own NUMA
// domain first. Tasks are queued on the owner of the tile they write (pool::owner).
//
// parallel_for (fork-join with dynamic chunk claiming; the caller takes part in the loop) runs on the same pool.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace std_backend
{

namespace detail
{

inline constexpr std::size_t cache_line = 64;

/// Maximum number of future arguments of one dataflow call (gemm takes three).
inline constexpr unsigned max_deps = 4;

/// Waiter-list entries tag a task pointer with its link slot in the low bits (task_base is 16-byte aligned).
inline constexpr std::uintptr_t slot_mask = 15;
static_assert(max_deps <= slot_mask);

/// A unit of work. Intrusive: the link slots let a task sit in the waiter lists of up to max_deps unfinished inputs
/// without allocating.
struct alignas(16) task_base
{
    virtual ~task_base() = default;
    virtual void run() = 0;

    std::atomic<int> deps{ 1 };
    bool high_priority = false;
    int hint = -1;  ///< worker whose deque receives the task once it is ready (-1: the submitting worker)
    task_base *next[max_deps] = {};
};

class spinlock
{
  public:
    void lock() noexcept
    {
        while (flag_.exchange(true, std::memory_order_acquire))
        {
            while (flag_.load(std::memory_order_relaxed))
            {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
        }
    }

    bool try_lock() noexcept
    {
        return !flag_.load(std::memory_order_relaxed) && !flag_.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept { flag_.store(false, std::memory_order_release); }

  private:
    std::atomic<bool> flag_{ false };
};

/// Fixed-size pool of worker threads with one work-stealing deque per worker.
class pool
{
  public:
    static pool &instance()
    {
        // Never destroyed: avoids shutdown-order problems; the workers die with the process.
        static pool *p = new pool(default_threads());
        return *p;
    }

    /// Worker count: CHOLESKY_THREADS if set to a positive number, else the CPUs in the affinity mask of the process
    /// (hardware_concurrency when that is unavailable).
    static unsigned default_threads()
    {
        if (const char *env = std::getenv("CHOLESKY_THREADS"))
        {
            const long v = std::strtol(env, nullptr, 10);
            if (v > 0)
            {
                return static_cast<unsigned>(v);
            }
        }
        const std::size_t n = allowed_cpus().size();
        return n > 0 ? static_cast<unsigned>(n) : std::max(1u, std::thread::hardware_concurrency());
    }

    std::size_t size() const noexcept { return workers_.size(); }

    /// Make @p t runnable. A worker enqueues on its own deque (its newest task runs next, which keeps the data of the
    /// tile it just produced warm); any other thread distributes round-robin.
    void submit(task_base *t)
    {
        std::size_t target = t->hint >= 0 ? static_cast<std::size_t>(t->hint) : self_id();
        if (target == npos)
        {
            target = rr_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
        }
        worker &w = *workers_[target];
        {
            std::lock_guard<spinlock> g(w.lock);
            const int q = t->high_priority ? 1 : 0;
            w.queue[q].push_back(t);
            w.count[q].store(w.queue[q].size(), std::memory_order_relaxed);
        }
        // Order the push before the searching_ load in wake_one (pairs with the fence in the sleep path of loop()).
        std::atomic_thread_fence(std::memory_order_seq_cst);
        wake_one();
    }

    /// Worker that owns tile (row, col): rows are distributed cyclically over the NUMA domains of the workers and
    /// the columns over the workers of a domain. Tiles are generated on their owner (first touch puts the pages in
    /// the owner's domain) and every task is queued on the owner of the tile it writes.
    int owner(std::size_t row, std::size_t col) const
    {
        const std::size_t d = row % domains_.size();
        const std::vector<unsigned> &members = domains_[d];
        return static_cast<int>(members[(col + row / domains_.size()) % members.size()]);
    }

    /// True when called from one of the pool's threads.
    static bool in_pool() noexcept { return self_id() != npos; }

  private:
    /// Wake a sleeping worker unless some worker is already searching for work (it will find the new task, and wake
    /// another worker itself if it turns out to be the last searcher).
    void wake_one()
    {
        if (searching_.load(std::memory_order_seq_cst) == 0)
        {
            if (sleepers_.load(std::memory_order_seq_cst) > 0)
            {
                epoch_.fetch_add(1, std::memory_order_seq_cst);
                epoch_.notify_one();
            }
        }
    }

    constexpr static std::size_t npos = static_cast<std::size_t>(-1);

    struct alignas(cache_line) worker
    {
        spinlock lock;
        std::atomic<std::size_t> count[2] = {};  // [0] normal, [1] high priority
        std::deque<task_base *> queue[2];
    };

    static std::size_t &self_id() noexcept
    {
        static thread_local std::size_t id = npos;
        return id;
    }

    explicit pool(unsigned n)
    {
        workers_.reserve(n);
        for (unsigned i = 0; i < n; ++i)
        {
            workers_.push_back(std::make_unique<worker>());
        }
        cpus_ = allowed_cpus();
        build_domains(n);
        threads_.reserve(n);
        for (unsigned i = 0; i < n; ++i)
        {
            threads_.emplace_back([this, i] { loop(i); });
        }
    }

    /// NUMA node of a CPU (from sysfs); 0 when unknown.
    static int node_of_cpu(int cpu)
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

    /// Group the workers by the NUMA node of the CPU they are bound to (a single group when workers are not bound).
    void build_domains(unsigned n)
    {
        std::map<int, std::vector<unsigned>> by_node;
        for (unsigned i = 0; i < n; ++i)
        {
            by_node[cpus_.size() >= n ? node_of_cpu(cpus_[i]) : 0].push_back(i);
        }
        domain_of_.assign(n, 0);
        for (auto &[node, members] : by_node)
        {
            for (unsigned w : members)
            {
                domain_of_[w] = static_cast<unsigned>(domains_.size());
            }
            domains_.push_back(std::move(members));
        }
    }

    static std::vector<int> allowed_cpus()
    {
        std::vector<int> cpus;
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0)
        {
            for (int c = 0; c < CPU_SETSIZE; ++c)
            {
                if (CPU_ISSET(c, &set))
                {
                    cpus.push_back(c);
                }
            }
        }
#endif
        return cpus;
    }

    void bind(unsigned i)
    {
#if defined(__linux__)
        // Pin worker i to the i-th CPU the process may use (a no-op when the process has fewer CPUs than workers).
        if (cpus_.size() >= workers_.size())
        {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus_[i], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
#else
        (void) i;
#endif
    }

    // Pop the newest task from queue level @p q of worker @p id.
    task_base *pop_local(std::size_t id, int q)
    {
        worker &w = *workers_[id];
        if (w.count[q].load(std::memory_order_relaxed) == 0)
        {
            return nullptr;
        }
        std::lock_guard<spinlock> g(w.lock);
        auto &dq = w.queue[q];
        if (dq.empty())
        {
            return nullptr;
        }
        task_base *t = dq.back();
        dq.pop_back();
        w.count[q].store(dq.size(), std::memory_order_relaxed);
        return t;
    }

    // Take the oldest task from queue level @p q of worker @p v (gives up when its lock is busy).
    task_base *steal_from(std::size_t v, int q)
    {
        worker &w = *workers_[v];
        if (w.count[q].load(std::memory_order_relaxed) == 0)
        {
            return nullptr;
        }
        std::unique_lock<spinlock> g(w.lock, std::try_to_lock);
        if (g.owns_lock() && !w.queue[q].empty())
        {
            task_base *t = w.queue[q].front();
            w.queue[q].pop_front();
            w.count[q].store(w.queue[q].size(), std::memory_order_relaxed);
            return t;
        }
        return nullptr;
    }

    // Steal the oldest task of another worker: workers of the own NUMA domain first (their tasks work on tiles that
    // live in this domain), then everybody.
    task_base *steal(std::size_t id, std::size_t start, int q)
    {
        const std::vector<unsigned> &mine = domains_[domain_of_[id]];
        if (domains_.size() > 1)
        {
            for (std::size_t k = 0; k < mine.size(); ++k)
            {
                const std::size_t v = mine[(start + k) % mine.size()];
                if (v != id)
                {
                    if (task_base *t = steal_from(v, q))
                    {
                        return t;
                    }
                }
            }
        }
        const std::size_t n = workers_.size();
        for (std::size_t k = 0; k < n; ++k)
        {
            std::size_t v = start + k;
            if (v >= n)
            {
                v -= n;
            }
            if (v == id)
            {
                continue;
            }
            if (task_base *t = steal_from(v, q))
            {
                return t;
            }
        }
        return nullptr;
    }

    task_base *try_pop(std::size_t id, std::size_t &rng)
    {
        // 64-bit LCG (Knuth MMIX constants); the high bits pick the victim to start stealing from.
        rng = rng * 6'364'136'223'846'793'005ull + 1'442'695'040'888'963'407ull;
        const std::size_t start = (rng >> 33) % workers_.size();
        task_base *t = pop_local(id, 1);
        if (t == nullptr)
        {
            t = steal(id, start, 1);
        }
        if (t == nullptr)
        {
            t = pop_local(id, 0);
        }
        if (t == nullptr)
        {
            t = steal(id, start, 0);
        }
        return t;
    }

    void loop(unsigned id)
    {
        self_id() = id;
        bind(id);
        std::size_t rng = 0x9E3779B97F4A7C15ull * (id + 1);
        for (;;)
        {
            task_base *t = try_pop(id, rng);
            if (t == nullptr)
            {
                searching_.fetch_add(1, std::memory_order_seq_cst);
                for (int spin = 0; t == nullptr && spin < spin_rounds_; ++spin)
                {
                    t = try_pop(id, rng);
                }
                if (t != nullptr)
                {
                    if (searching_.fetch_sub(1, std::memory_order_seq_cst) == 1)
                    {
                        wake_one();
                    }
                }
                else
                {
                    // Sleep until a submit() bumps the epoch. Announcing the sleeper (and ceasing to search) before
                    // the final scan means a concurrent submit() either is seen by that scan or sees the sleeper.
                    sleepers_.fetch_add(1, std::memory_order_seq_cst);
                    searching_.fetch_sub(1, std::memory_order_seq_cst);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    const unsigned e = epoch_.load(std::memory_order_seq_cst);
                    t = try_pop(id, rng);
                    if (t == nullptr)
                    {
                        epoch_.wait(e, std::memory_order_seq_cst);
                    }
                    sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                    if (t == nullptr)
                    {
                        continue;
                    }
                    wake_one();  // like the spin path: if no searcher is left, make sure other tasks get picked up
                }
            }
            t->run();
        }
    }

    std::vector<std::unique_ptr<worker>> workers_;
    std::vector<int> cpus_;
    std::vector<std::vector<unsigned>> domains_;  // workers per NUMA domain
    std::vector<unsigned> domain_of_;             // domain index of each worker
    constexpr static int spin_rounds_ = 64;
    std::vector<std::jthread> threads_;
    alignas(cache_line) std::atomic<std::size_t> rr_{ 0 };
    alignas(cache_line) std::atomic<int> sleepers_{ 0 };
    alignas(cache_line) std::atomic<int> searching_{ 0 };
    alignas(cache_line) std::atomic<unsigned> epoch_{ 0 };
};

/// Shared state of a future: completion flag, waiter list and (typed) value.
///
/// The waiter list is a lock-free stack of tagged task pointers (low bits = link slot of the task). 0 means empty and
/// 1 means completed.
struct state_base
{
    constexpr static std::uintptr_t done_mark = 1;

    std::atomic<std::uintptr_t> head{ 0 };
    std::atomic<std::uint32_t> ready{ 0 };

    /// Register @p t (link slot @p slot) to be released on completion. False when already complete.
    bool add_waiter(task_base *t, unsigned slot) noexcept
    {
        std::uintptr_t h = head.load(std::memory_order_acquire);
        const std::uintptr_t me = reinterpret_cast<std::uintptr_t>(t) | slot;
        do {
            if (h == done_mark)
            {
                return false;
            }
            t->next[slot] = reinterpret_cast<task_base *>(h);
        } while (!head.compare_exchange_weak(h, me, std::memory_order_acq_rel, std::memory_order_acquire));
        return true;
    }

    /// Publish completion (the value must already be stored) and release every task waiting on it.
    void complete()
    {
        std::uintptr_t h = head.exchange(done_mark, std::memory_order_acq_rel);
        ready.store(1, std::memory_order_release);
        ready.notify_all();
        while (h != 0)
        {
            auto *t = reinterpret_cast<task_base *>(h & ~slot_mask);
            const unsigned slot = static_cast<unsigned>(h & slot_mask);
            h = reinterpret_cast<std::uintptr_t>(t->next[slot]);
            if (t->deps.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                pool::instance().submit(t);
            }
        }
    }

    void wait() const
    {
        while (ready.load(std::memory_order_acquire) == 0)
        {
            ready.wait(0, std::memory_order_acquire);
        }
    }
};

template <typename T>
struct state : state_base
{
    T value{};
};

}  // namespace detail

/**
 * @brief Shared handle to the eventual result of a task (the analogue of std::shared_future).
 */
template <typename T>
class future
{
    using stored = std::conditional_t<std::is_void_v<T>, std::monostate, T>;

  public:
    future() = default;

    explicit future(std::shared_ptr<detail::state<stored>> s) :
        s_(std::move(s))
    { }

    bool valid() const noexcept { return s_ != nullptr; }

    void wait() const { s_->wait(); }

    /// Wait, then return the value (nothing for future<void>).
    decltype(auto) get() const
    {
        s_->wait();
        if constexpr (!std::is_void_v<T>)
        {
            return static_cast<const T &>(s_->value);
        }
    }

    /// Wait, then move the value out. Only for the sole remaining consumer of a value (the tile a task updates in
    /// place): every other holder of this state would see an empty value afterwards.
    T take() const
    {
        s_->wait();
        return std::move(s_->value);
    }

    detail::state_base *state_ptr() const noexcept { return s_.get(); }

  private:
    std::shared_ptr<detail::state<stored>> s_;
};

template <typename T>
inline constexpr bool is_future_v = false;
template <typename T>
inline constexpr bool is_future_v<future<T>> = true;

/**
 * @brief Create an already-satisfied future holding @p value.
 */
template <typename T>
future<std::decay_t<T>> make_ready_future(T &&value)
{
    auto s = std::make_shared<detail::state<std::decay_t<T>>>();
    s->value = std::forward<T>(value);
    s->head.store(detail::state_base::done_mark, std::memory_order_relaxed);
    s->ready.store(1, std::memory_order_release);
    return future<std::decay_t<T>>(std::move(s));
}

/**
 * @brief Create an already-satisfied future<void>.
 */
inline future<void> make_ready_future()
{
    auto s = std::make_shared<detail::state<std::monostate>>();
    s->head.store(detail::state_base::done_mark, std::memory_order_relaxed);
    s->ready.store(1, std::memory_order_release);
    return future<void>(std::move(s));
}

namespace detail
{

template <typename F, typename Tuple, typename Result>
struct dataflow_task final : task_base
{
    using stored = std::conditional_t<std::is_void_v<Result>, std::monostate, Result>;

    dataflow_task(F &&fn, Tuple &&a, std::shared_ptr<state<stored>> s) :
        f(std::move(fn)),
        args(std::move(a)),
        st(std::move(s))
    { }

    void run() override
    {
        {
            // Move the inputs out of the task so that they are released as soon as the callable returns (as
            // hpx::dataflow does) instead of pinning every earlier tile version.
            Tuple inputs = std::move(args);
            if constexpr (std::is_void_v<Result>)
            {
                std::apply(f, std::move(inputs));
            }
            else
            {
                st->value = std::apply(f, std::move(inputs));
            }
        }
        std::shared_ptr<state<stored>> s = std::move(st);
        delete this;
        s->complete();
    }

    F f;
    Tuple args;
    std::shared_ptr<state<stored>> st;
};

}  // namespace detail

/**
 * @brief std::library analogue of hpx::dataflow, placed by tile.
 *
 * Schedules @p f(args...) once every future among @p args is ready; the task is then queued on the owner of tile
 * (row, col), and @p high marks critical-path tasks (ready ones run before ordinary ones). The futures are forwarded
 * to the callable like any other argument (they are ready when it runs, so get() does not block). Never blocks the
 * caller.
 */
template <typename F, typename... Args>
auto dataflow_on_tile(std::size_t row, std::size_t col, bool high, F &&f, Args &&...args)
{
    using Result = std::invoke_result_t<std::decay_t<F> &, std::decay_t<Args> &&...>;
    using stored = std::conditional_t<std::is_void_v<Result>, std::monostate, Result>;
    using Tuple = std::tuple<std::decay_t<Args>...>;
    static_assert((0u + ... + (is_future_v<std::decay_t<Args>> ? 1u : 0u)) <= detail::max_deps, "too many futures");

    auto st = std::make_shared<detail::state<stored>>();
    auto *t = new detail::dataflow_task<std::decay_t<F>, Tuple, Result>(
        std::decay_t<F>(std::forward<F>(f)), Tuple(std::forward<Args>(args)...), st);
    t->high_priority = high;
    t->hint = detail::pool::instance().owner(row, col);
    unsigned slot = 0;
    std::apply(
        [&](auto &...a)
        {
            auto reg = [&](auto &x)
            {
                if constexpr (is_future_v<std::decay_t<decltype(x)>>)
                {
                    t->deps.fetch_add(1, std::memory_order_relaxed);
                    if (!x.state_ptr()->add_waiter(t, slot++))
                    {
                        t->deps.fetch_sub(1, std::memory_order_relaxed);
                    }
                }
            };
            (reg(a), ...);
        },
        t->args);
    // Drop the guard reference taken at construction; the last input to finish (or this call) submits the task.
    if (t->deps.fetch_sub(1, std::memory_order_acq_rel) == 1)
    {
        detail::pool::instance().submit(t);
    }
    return future<Result>(std::move(st));
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

namespace detail
{

/// Shared state of one parallel_for. Helper tasks keep it alive; the loop body is only touched while chunks remain,
/// which is before the calling thread returns.
struct loop_state
{
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t chunk = 1;
    std::size_t n_chunks = 0;
    void (*invoke)(void *, std::size_t) = nullptr;
    void *body = nullptr;
    alignas(cache_line) std::atomic<std::size_t> next{ 0 };
    alignas(cache_line) std::atomic<std::size_t> remaining{ 0 };

    void work()
    {
        for (;;)
        {
            const std::size_t c = next.fetch_add(1, std::memory_order_relaxed);
            if (c >= n_chunks)
            {
                return;
            }
            const std::size_t lo = begin + c * chunk;
            const std::size_t hi = std::min(end, lo + chunk);
            for (std::size_t i = lo; i < hi; ++i)
            {
                invoke(body, i);
            }
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                remaining.notify_all();
            }
        }
    }
};

struct loop_helper final : task_base
{
    explicit loop_helper(std::shared_ptr<loop_state> s) :
        st(std::move(s))
    { }

    void run() override
    {
        std::shared_ptr<loop_state> s = std::move(st);
        delete this;
        s->work();
    }

    std::shared_ptr<loop_state> st;
};

}  // namespace detail

/**
 * @brief Parallel index loop over [begin, end): fork-join on the worker pool.
 *
 * Replaces hpx::experimental::for_loop(hpx::execution::par, ...). Chunks of @p chunk consecutive indices are claimed
 * dynamically; the calling thread works on the loop too and returns once every iteration is done (implicit barrier).
 * Loops may be nested.
 */
template <typename F>
void parallel_for(std::size_t begin, std::size_t end, F &&f, std::size_t chunk = 1)
{
    if (begin >= end)
    {
        return;
    }
    chunk = std::max<std::size_t>(chunk, 1);
    const std::size_t n_chunks = (end - begin + chunk - 1) / chunk;
    detail::pool &p = detail::pool::instance();
    if (n_chunks == 1 || p.size() == 1)
    {
        for (std::size_t i = begin; i < end; ++i)
        {
            f(i);
        }
        return;
    }
    auto st = std::make_shared<detail::loop_state>();
    st->begin = begin;
    st->end = end;
    st->chunk = chunk;
    st->n_chunks = n_chunks;
    st->remaining.store(n_chunks, std::memory_order_relaxed);
    using Body = std::remove_reference_t<F>;
    st->body = const_cast<void *>(static_cast<const void *>(std::addressof(f)));
    st->invoke = [](void *b, std::size_t i) { (*static_cast<Body *>(b))(i); };

    // The caller is one participant; the pool supplies the rest (a pool thread calling this counts as one of them).
    const std::size_t helpers = std::min(n_chunks - 1, p.size() - (detail::pool::in_pool() ? 1 : 0));
    for (std::size_t h = 0; h < helpers; ++h)
    {
        p.submit(new detail::loop_helper(st));
    }
    st->work();
    for (std::size_t r = st->remaining.load(std::memory_order_acquire); r != 0;
         r = st->remaining.load(std::memory_order_acquire))
    {
        st->remaining.wait(r, std::memory_order_acquire);
    }
}

}  // namespace std_backend

#endif  // end of TASK_BACKEND_H
