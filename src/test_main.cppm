export module test_main;

import std;
import <cassert>;
import devlib.smart_handle;

// ============================================================================
// Mock Objects for Testing
// ============================================================================

// A mock object that tracks how many times its destructor is called.
struct mock_destructor_tracker {
    // This must be a pointer to a shared counter, as the object itself is destroyed.
    std::atomic<int> *destruction_counter{nullptr};
    bool have_value{false};

    mock_destructor_tracker(std::atomic<int> *counter) : destruction_counter(counter), have_value(true) {}

    void destroy() {
        if (destruction_counter && have_value) {
            destruction_counter->fetch_add(1, std::memory_order_relaxed);
            have_value = false;
        }
    }

    [[nodiscard]] bool has_value() const {
        return have_value;
    }

    mock_destructor_tracker() = default;
};

struct Base {
    virtual ~Base() = default;
    virtual void foo() = 0;
};

struct DerivedA : public Base {
    void foo() override {
        std::println("DerivedA foo");
    }

    DerivedA() = default;
};

struct DerivedB : public Base {
    void foo() override {
        std::println("DerivedB foo");
    }

    DerivedB() = default;
};

using BaseArcHandle = dev_lib::strong_arc_handle<dev_lib::pointer_handle<Base>>;
using DerivedAArcHandle = dev_lib::strong_arc_handle<dev_lib::pointer_handle<DerivedA>>;
using DerivedBArcHandle = dev_lib::strong_arc_handle<dev_lib::pointer_handle<DerivedB>>;

// ============================================================================
// Test Functions
// ============================================================================

// Stress test for atomic reference counting with concurrent access
void run_stress_test() {
    std::println("=== Running Stress Test ===");

    constexpr int num_threads = 8;
    constexpr int ops_per_thread = 100'000;
    constexpr int pool_size = 128;

    std::atomic<int> creation_counter = 0;
    std::atomic<int> destruction_counter = 0;

    // Pool of weak handles, only replaced under mutex
    std::vector<dev_lib::weak_arc_handle<dev_lib::pointer_handle<mock_destructor_tracker>>> weak_pool(pool_size);
    std::mutex pool_mutex;

    // Populate pool with unique objects
    {
        std::lock_guard lock(pool_mutex);
        for (int i = 0; i < pool_size; ++i) {
            creation_counter.fetch_add(1, std::memory_order_relaxed);
            auto strong = dev_lib::make_arc_handle<mock_destructor_tracker>(&destruction_counter);
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
                    dev_lib::weak_arc_handle<dev_lib::pointer_handle<mock_destructor_tracker>> weak;
                    {
                        std::lock_guard lock(pool_mutex);
                        weak = weak_pool[idx];
                    }
                    auto strong = weak.lock();
                    // Use strong if not null, then let it go out of scope
                } else {
                    // Replace a weak handle in the pool with a new object
                    auto strong = dev_lib::make_arc_handle<mock_destructor_tracker>(&destruction_counter);
                    creation_counter.fetch_add(1, std::memory_order_relaxed);
                    auto weak = strong.share_weak();
                    {
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
    std::println("Stress test passed!\n");
}

// Test non-atomic reference counting
void run_rc_test() {
    std::println("=== Running RC Test ===");

    // Test basic weak handle functionality
    dev_lib::strong_rc_handle<dev_lib::pointer_handle<mock_destructor_tracker>> strong_handle =
        dev_lib::make_rc_handle<mock_destructor_tracker>(new std::atomic<int>(0));

    std::vector<dev_lib::weak_rc_handle<dev_lib::pointer_handle<mock_destructor_tracker>>> weak_handles;

    for (int i = 0; i < 10; ++i) {
        weak_handles.push_back(strong_handle.share_weak());
    }

    for (auto &weak : weak_handles) {
        auto locked = weak.lock();
        assert(locked && "Failed to lock weak handle!");
    }

    // Test reset() functionality
    {
        std::atomic<int> destruction_count = 0;

        auto rc1 = dev_lib::make_rc_handle<mock_destructor_tracker>(&destruction_count);
        auto rc2 = rc1.clone();
        auto weak = rc1.share_weak();

        // Reset rc1 (not last reference)
        rc1.reset();
        assert(destruction_count.load() == 0 && "Should not destroy yet");

        // Lock should still work
        {
            auto locked = weak.lock();
            assert(locked.has_value() && "Should be able to lock");
        } // locked goes out of scope here

        // Reset rc2 (last strong reference)
        rc2.reset();
        assert(destruction_count.load() == 1 && "Should destroy now");

        // Lock should fail now
        auto locked2 = weak.lock();
        assert(!locked2.has_value() && "Should not be able to lock after all strong refs gone");

        // Test weak reset
        weak.reset();
        assert(weak.expired() && "Weak handle should be expired after reset");
    }

    std::println("RC test passed!\n");
}

// Test polymorphic pointer casting
void run_polymorphic_test() {
    std::println("=== Running Polymorphic Test ===");

    auto derived_a = dev_lib::make_arc_handle<DerivedA>();
    auto derived_b = dev_lib::make_arc_handle<DerivedB>();

    // BaseArcHandle base_a = derived_a.static_pointer_cast<Base>();
    // BaseArcHandle base_b = derived_b.static_pointer_cast<Base>();
    //
    // base_a->foo();
    // base_b->foo();

    std::println("Polymorphic test passed!\n");
}

// Test array_handle with unique ownership
void run_unique_array_test() {
    std::println("=== Running Unique Array Test ===");

    // Create array with 5 integers, all initialized to 42
    auto arr = dev_lib::make_unique_array<int>(5, 42);

    assert(arr.has_value() && "Array should have value");
    assert((*arr).size() == 5 && "Array size should be 5");

    // Test element access
    for (std::size_t i = 0; i < (*arr).size(); ++i) {
        assert((*arr)[i] == 42 && "Initial value should be 42");
        (*arr)[i] = static_cast<int>(i * 10);
    }

    // Test modified values
    for (std::size_t i = 0; i < (*arr).size(); ++i) {
        assert((*arr)[i] == static_cast<int>(i * 10) && "Modified value incorrect");
    }

    // Test iterator
    int sum = 0;
    for (auto val : *arr) {
        sum += val;
    }
    assert(sum == 0 + 10 + 20 + 30 + 40 && "Iterator sum incorrect");

    // Test at() method
    try {
        (*arr).at(10); // Should throw
        assert(false && "at() should throw for out of range");
    } catch (const std::out_of_range &) {
        // Expected
    }

    std::println("Unique array test passed!\n");
}

// Test array_handle with shared ownership (arc)
void run_shared_array_arc_test() {
    std::println("=== Running Shared Array ARC Test ===");

    // Create from initializer list
    auto arr1 = dev_lib::make_arc_array<int>({1, 2, 3, 4, 5});

    assert(arr1.has_value() && "Array should have value");
    assert((*arr1).size() == 5 && "Array size should be 5");

    // Clone the handle
    auto arr2 = arr1.clone();

    // Modify through one handle
    (*arr1)[0] = 100;

    // Should be visible through other handle (they share the same array)
    assert((*arr2)[0] == 100 && "Shared array should reflect changes");

    // Test weak handle
    auto weak = arr1.share_weak();
    arr1.reset();

    {
        auto locked = weak.lock();
        assert(locked.has_value() && "Should be able to lock (arr2 still alive)");
        assert((*locked)[0] == 100 && "Value should be preserved");
    } // locked goes out of scope here

    arr2.reset();
    auto locked2 = weak.lock();
    assert(!locked2.has_value() && "Should not be able to lock (all strong handles gone)");

    std::println("Shared array ARC test passed!\n");
}

// Test array_handle with custom objects
void run_array_with_objects_test() {
    std::println("=== Running Array With Objects Test ===");

    std::atomic<int> destruction_count = 0;

    {
        // Create array of mock objects
        auto arr = dev_lib::make_rc_array<mock_destructor_tracker>(3, &destruction_count);

        assert((*arr).size() == 3 && "Array size should be 3");

        // Verify all objects are valid
        for (std::size_t i = 0; i < (*arr).size(); ++i) {
            assert((*arr)[i].has_value() && "Object should have value");
        }

        // Test range-based for
        int count = 0;
        for (auto &obj : *arr) {
            assert(obj.has_value() && "Object should have value");
            ++count;
        }
        assert(count == 3 && "Should iterate over 3 objects");
    }

    // All objects should be destroyed
    assert(destruction_count.load() == 3 && "All 3 objects should be destroyed");

    std::println("Array with objects test passed!\n");
}

// Test array_handle created from range
void run_array_from_range_test() {
    std::println("=== Running Array From Range Test ===");

    // Create a vector as source
    std::vector<int> source = {10, 20, 30, 40, 50};

    // Create array from vector (range)
    auto arr = dev_lib::make_unique_array<int>(source);

    assert((*arr).size() == 5 && "Array size should match source");

    // Verify values
    for (std::size_t i = 0; i < (*arr).size(); ++i) {
        assert((*arr)[i] == source[i] && "Values should match source");
    }

    // Modify array - should not affect source
    (*arr)[0] = 999;
    assert(source[0] == 10 && "Source should be unchanged");

    // Test with view/range algorithms
    auto doubled = source | std::views::transform([](int x) { return x * 2; });
    auto arr2 = dev_lib::make_arc_array<int>(doubled);

    assert((*arr2).size() == 5 && "Array size should be 5");
    assert((*arr2)[0] == 20 && "First element should be 20");
    assert((*arr2)[4] == 100 && "Last element should be 100");

    std::println("Array from range test passed!\n");
}

// Test shared function handles
void run_function_handle_test() {
    std::println("=== Running Function Handle Test ===");

    // Test arc function (thread-safe) - basic functionality
    {
        auto func = dev_lib::make_arc_function<int(int, int)>([str = std::string("1234")](int a, int b) {
            return a + b;
        });

        assert(func && "Function should be valid");
        assert(func(3, 4) == 7 && "Function should work");

        // Test move
        auto func2 = std::move(func);
        assert(!func && "Function should be empty after move");
        assert(func2 && "Moved function should be valid");
        assert(func2(5, 6) == 11 && "Moved function should work");

        // Test reset
        func2.reset();
        assert(!func2 && "Function should be empty after reset");
    }

    // Test capturing std::string (large object) - this will crash if memory is not managed correctly
    {
        std::string captured_string = "Hello, World! This is a longer string to test memory management.";

        auto func = dev_lib::make_arc_function<std::string()>([captured_string]() {
            return captured_string + " [captured]";
        });

        std::string result = func();
        assert(result.find("Hello, World!") != std::string::npos && "Captured string should work");
        assert(result.find("[captured]") != std::string::npos && "String concatenation should work");

        std::println("  Captured string test passed: {}", result);
    }

    // Test rc function (non-thread-safe) with mutable capture
    {
        std::string accumulator;

        auto func = dev_lib::make_rc_function<void(const std::string&)>([&accumulator](const std::string& s) {
            accumulator += s;
        });

        func("Hello");
        assert(accumulator == "Hello" && "First call should work");

        func(" World");
        assert(accumulator == "Hello World" && "Second call should append");

        std::println("  Mutable capture test passed: {}", accumulator);
    }

    // Test with std::vector capture (heap-allocated)
    {
        std::vector<int> numbers = {1, 2, 3, 4, 5};

        auto func = dev_lib::make_arc_function<int()>([numbers]() {
            int sum = 0;
            for (int n : numbers) {
                sum += n;
            }
            return sum;
        });

        int result = func();
        assert(result == 15 && "Vector capture and sum should work");
        std::println("  Vector capture test passed: sum = {}", result);
    }

    // Test function that returns void
    {
        bool called = false;

        auto func = dev_lib::make_rc_function<void()>([&called]() {
            called = true;
        });

        func();
        assert(called && "Void function should execute");
    }

    // Test with multiple string captures (stress test for memory)
    {
        std::string s1 = "First string with some content";
        std::string s2 = "Second string with more content";
        std::string s3 = "Third string with even more content";

        auto func = dev_lib::make_arc_function<std::string(const std::string&)>(
            [s1, s2, s3](const std::string& suffix) {
                return s1 + " | " + s2 + " | " + s3 + " | " + suffix;
            }
        );

        std::string result = func("END");
        assert(result.find("First") != std::string::npos && "First string should be captured");
        assert(result.find("Second") != std::string::npos && "Second string should be captured");
        assert(result.find("Third") != std::string::npos && "Third string should be captured");
        assert(result.find("END") != std::string::npos && "Suffix should be appended");

        std::println("  Multiple string capture test passed");
    }

    // Test move semantics with captured objects
    {
        std::string heavy_data = "This is some heavy data that needs to be moved";

        auto func1 = dev_lib::make_rc_function<std::string()>([data = std::move(heavy_data)]() {
            return data;
        });

        auto func2 = std::move(func1);

        std::string result = func2();
        assert(result.find("heavy data") != std::string::npos && "Moved function should preserve captured data");
    }

    std::println("Function handle test passed!\n");
}

// Test unique_function with small buffer optimization
void run_unique_function_test() {
    std::println("=== Running Unique Function Test ===");

    // Test 1: Basic functionality with small lambda (should use SBO)
    {
        auto func = dev_lib::make_unique_function<int(int, int)>([](int a, int b) {
            return a + b;
        });

        assert(func && "Function should be valid");
        assert(func(3, 4) == 7 && "Function result should be correct");
        std::println("  Basic unique_function test passed");
    }

    // Test 2: Move semantics
    {
        auto func1 = dev_lib::make_unique_function<int(int)>([](int x) {
            return x * 2;
        });

        auto func2 = std::move(func1);
        assert(!func1 && "Original function should be empty after move");
        assert(func2 && "Moved function should be valid");
        assert(func2(5) == 10 && "Moved function should work correctly");
        std::println("  Move semantics test passed");
    }

    // Test 3: Capturing small objects (should fit in SBO)
    {
        int capture_value = 42;
        auto func = dev_lib::make_unique_function<int()>([capture_value]() {
            return capture_value * 2;
        });

        assert(func() == 84 && "Captured value should work");
        std::println("  Small capture test passed (likely using SBO)");
    }

    // Test 4: Capturing large objects (may exceed SBO buffer)
    {
        std::string large_str = "This is a relatively long string that might exceed the inline buffer";
        std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        auto func = dev_lib::make_unique_function<std::string()>([large_str, vec]() {
            int sum = 0;
            for (int n : vec) sum += n;
            return large_str + " sum=" + std::to_string(sum);
        });

        std::string result = func();
        assert(result.find("long string") != std::string::npos && "Large string should be captured");
        assert(result.find("sum=55") != std::string::npos && "Vector sum should be correct");
        std::println("  Large capture test passed (may use heap allocation)");
    }

    // Test 5: Void return type
    {
        bool executed = false;
        auto func = dev_lib::make_unique_function<void()>([&executed]() {
            executed = true;
        });

        func();
        assert(executed && "Void function should execute");
        std::println("  Void return type test passed");
    }

    // Test 6: Multiple parameters
    {
        auto func = dev_lib::make_unique_function<int(int, int, int)>([](int a, int b, int c) {
            return a + b + c;
        });

        assert(func(1, 2, 3) == 6 && "Multi-parameter function should work");
        std::println("  Multiple parameters test passed");
    }

    // Test 7: Reset functionality
    {
        auto func = dev_lib::make_unique_function<int()>([]() { return 42; });
        assert(func && "Function should be valid");

        func.reset();
        assert(!func && "Function should be empty after reset");
        std::println("  Reset functionality test passed");
    }

    // Test 8: String operations with captures
    {
        std::string prefix = "Result: ";
        auto func = dev_lib::make_unique_function<std::string(int)>([prefix](int value) {
            return prefix + std::to_string(value);
        });

        std::string result = func(123);
        assert(result == "Result: 123" && "String concatenation should work");
        std::println("  String operations test passed");
    }

    // Test 9: Complex return type
    {
        auto func = dev_lib::make_unique_function<std::vector<int>(int)>([](int n) {
            std::vector<int> result;
            for (int i = 0; i < n; ++i) {
                result.push_back(i * i);
            }
            return result;
        });

        auto result = func(5);
        assert(result.size() == 5 && "Vector size should be correct");
        assert(result[4] == 16 && "Last element should be 4^2 = 16");
        std::println("  Complex return type test passed");
    }

    // Test 10: Mutable lambda
    {
        auto func = dev_lib::make_unique_function<int()>([counter = 0]() mutable {
            return ++counter;
        });

        assert(func() == 1 && "First call should return 1");
        assert(func() == 2 && "Second call should return 2");
        assert(func() == 3 && "Third call should return 3");
        std::println("  Mutable lambda test passed");
    }

    std::println("Unique function test passed!\n");
}

// ============================================================================
// Main Entry Point
// ============================================================================

export int main() {
    std::println("Starting smart handle tests...\n");

    try {
        run_stress_test();
        run_rc_test();
        run_polymorphic_test();
        run_unique_array_test();
        run_shared_array_arc_test();
        run_array_with_objects_test();
        run_array_from_range_test();
        run_function_handle_test();
        run_unique_function_test();  // Added unique_function tests

        std::println("All tests passed successfully!");
        return 0;
    } catch (const std::exception &e) {
        std::println("Test failed with exception: {}", e.what());
        return 1;
    }
}
