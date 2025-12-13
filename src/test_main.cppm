export module test_main;

import std;
import <cassert>;
import devlib.smart_handle;


// A mock object that tracks how many times its destructor is called.
struct mock_destructor_tracker {
    // This must be a pointer to a shared counter, as the object itself is destroyed.
    std::atomic<int> *destruction_counter;

    mock_destructor_tracker(std::atomic<int> *counter) : destruction_counter(counter) {
        // std::println("mock_destructor_tracker created.");
    }

    void destroy() {
        destruction_counter->fetch_add(1, std::memory_order_relaxed);
        // std::println("mock_destructor_tracker destroyed.");
    }
};

// A function to run the stress test.
void run_stress_test() {
    constexpr int num_threads = 8;
    constexpr int ops_per_thread = 100'000;
    constexpr int pool_size = 128;

    std::atomic<int> creation_counter = 0;
    std::atomic<int> destruction_counter = 0;

    // Pool of weak handles, only replaced under mutex
    std::vector<dev_lib::weak_arc_handle<mock_destructor_tracker>> weak_pool(pool_size);
    std::mutex pool_mutex;

    // Populate pool with unique objects
    {
        std::lock_guard lock(pool_mutex);
        for (int i = 0; i < pool_size; ++i) {
            creation_counter.fetch_add(1, std::memory_order_relaxed);
            auto strong = dev_lib::make_shared_arc<mock_destructor_tracker>(&destruction_counter);
            weak_pool[i] = strong.share_weak();
        }
    }

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::random_device rd;
            std::mt19937 gen(rd() ^ (t * 0x9e3779b9));
            std::uniform_int_distribution<> pool_dist(0, pool_size - 1);
            std::uniform_int_distribution<> op_dist(0, 1);

            for (int i = 0; i < ops_per_thread; ++i) {
                int idx = pool_dist(gen);

                if (op_dist(gen) == 0) {
                    // Try to lock a weak handle (safe, as lock() creates a new strong handle)
                    dev_lib::weak_arc_handle<mock_destructor_tracker> weak; {
                        std::lock_guard lock(pool_mutex);
                        weak = weak_pool[idx];
                    }
                    auto strong = weak.lock();
                    // Use strong if not null, then let it go out of scope
                } else {
                    // Replace a weak handle in the pool with a new object
                    auto strong = dev_lib::make_shared_arc<mock_destructor_tracker>(&destruction_counter);
                    creation_counter.fetch_add(1, std::memory_order_relaxed);
                    auto weak = strong.share_weak(); {
                        std::lock_guard lock(pool_mutex);
                        weak_pool[idx] = weak;
                    }
                    // strong destroyed here, only weak remains in pool
                }
            }
        });
    }

    threads.clear(); // Wait for all threads

    // Clear pool to release all weak handles
    weak_pool.clear();

    // All objects should be destroyed
    std::println("Total created: {}, destroyed: {}", creation_counter.load(), destruction_counter.load());
    assert(creation_counter.load() == destruction_counter.load() && "Destruction count mismatch!");
}

void run_rc_test() {
    // not atomic
    dev_lib::strong_rc_handle<mock_destructor_tracker> strong_handle = dev_lib::make_shared_rc<mock_destructor_tracker>(new std::atomic<int>(0));

    std::vector<dev_lib::weak_rc_handle<mock_destructor_tracker>> weak_handles;

    for (int i = 0; i < 10; ++i) {
        weak_handles.push_back(strong_handle.share_weak());
    }

    for (auto &weak : weak_handles) {
        auto locked = weak.lock();
        assert(locked && "Failed to lock weak handle!");
    }


}

export int main() {
    std::println("Starting aggressive memory order stress test...");
    run_stress_test();

    dev_lib::strong_arc_handle<mock_destructor_tracker> strong_handle = dev_lib::make_shared_arc<mock_destructor_tracker>(new std::atomic<int>(0));

    run_rc_test();
    std::println("Stress test complete.");
}
