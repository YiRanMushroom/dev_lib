//
// Created by Yiran on 2025-12-14.
//

export module pmr_allocators;

import std;

namespace dev_lib {
    // Static PMR resource for unique_function with specified buffer size
    template<std::size_t BufferSize, bool Sync>
    class static_pmr_resource {
    private:
        struct resource_impl {
            std::conditional_t<Sync,
                std::pmr::synchronized_pool_resource,
                std::pmr::unsynchronized_pool_resource> pool{
                std::pmr::pool_options{
                    .max_blocks_per_chunk = 0,
                    .largest_required_pool_block = BufferSize
                }
            };
        };

        static resource_impl &get_sync_resource() {
            static resource_impl impl;
            return impl;
        }

        static resource_impl &get_unsync_resource() {
            thread_local resource_impl impl;
            return impl;
        }

    public:
        static std::pmr::memory_resource *get() {
            if constexpr (Sync) {
                return &get_sync_resource().pool;
            } else {
                return &get_unsync_resource().pool;
            }
        }
    };

    // PMR allocator for unique_function
    export template<typename T, std::size_t BufferSize, bool Sync>
    class unique_function_pmr_allocator {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        unique_function_pmr_allocator() noexcept = default;

        template<typename U>
        unique_function_pmr_allocator(const unique_function_pmr_allocator<U, BufferSize, Sync> &) noexcept {}

        [[nodiscard]] T *allocate(size_type n) {
            auto *resource = static_pmr_resource<BufferSize, Sync>::get();
            void *ptr = resource->allocate(n * sizeof(T), alignof(T));
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, size_type n) noexcept {
            if (p == nullptr) return;
            auto *resource = static_pmr_resource<BufferSize, Sync>::get();
            resource->deallocate(p, n * sizeof(T), alignof(T));
        }

        template<typename U>
        struct rebind {
            using other = unique_function_pmr_allocator<U, BufferSize, Sync>;
        };

        bool operator==(const unique_function_pmr_allocator &) const noexcept { return true; }
        bool operator!=(const unique_function_pmr_allocator &) const noexcept { return false; }
    };

    // PMR allocator for array with element count
    export template<typename T, std::size_t ElementCount, bool Sync>
    class array_pmr_allocator {
    private:
        static constexpr std::size_t buffer_size = ElementCount * sizeof(T);

    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        array_pmr_allocator() noexcept = default;

        template<typename U>
        array_pmr_allocator(const array_pmr_allocator<U, ElementCount, Sync> &) noexcept {}

        [[nodiscard]] T *allocate(size_type n) {
            auto *resource = static_pmr_resource<buffer_size, Sync>::get();
            void *ptr = resource->allocate(n * sizeof(T), alignof(T));
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, size_type n) noexcept {
            if (p == nullptr) return;
            auto *resource = static_pmr_resource<buffer_size, Sync>::get();
            resource->deallocate(p, n * sizeof(T), alignof(T));
        }

        template<typename U>
        struct rebind {
            using other = array_pmr_allocator<U, ElementCount, Sync>;
        };

        bool operator==(const array_pmr_allocator &) const noexcept { return true; }
        bool operator!=(const array_pmr_allocator &) const noexcept { return false; }
    };
}
