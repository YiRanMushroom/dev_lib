//
// Created by Yiran on 2025-12-14.
//

export module small_inplace_allocator;

import std;

namespace dev_lib {
    // Simple small buffer optimized allocator
    export template<typename T, typename FallbackAllocator = std::allocator<T>, std::size_t InlineSize = 64>
    class small_inplace_allocator {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using propagate_on_container_move_assignment = std::true_type;

    private:
        alignas(alignof(T)) std::byte m_inline_buffer[InlineSize];
        [[no_unique_address]] FallbackAllocator m_fallback_allocator;

        static constexpr bool fits_inline(size_type n) noexcept {
            return n * sizeof(T) <= InlineSize && alignof(T) <= alignof(std::byte);
        }

        bool is_inline_ptr(const T* p) const noexcept {
            const auto buffer_start = reinterpret_cast<const std::byte*>(m_inline_buffer);
            const auto buffer_end = buffer_start + InlineSize;
            const auto ptr = reinterpret_cast<const std::byte*>(p);
            return ptr >= buffer_start && ptr < buffer_end;
        }

    public:
        small_inplace_allocator() noexcept = default;

        explicit small_inplace_allocator(const FallbackAllocator& alloc) noexcept
            : m_fallback_allocator(alloc) {}

        [[nodiscard]] T* allocate(size_type n) {
            if (fits_inline(n)) {
                return reinterpret_cast<T*>(m_inline_buffer);
            }
            return std::allocator_traits<FallbackAllocator>::allocate(m_fallback_allocator, n);
        }

        void deallocate(T* p, size_type n) noexcept {
            if (is_inline_ptr(p)) {
                return;
            }
            std::allocator_traits<FallbackAllocator>::deallocate(m_fallback_allocator, p, n);
        }

        template<typename U>
        struct rebind {
            using other = small_inplace_allocator<U,
                typename std::allocator_traits<FallbackAllocator>::template rebind_alloc<U>,
                InlineSize>;
        };

        bool operator==(const small_inplace_allocator&) const noexcept { return false; }
        bool operator!=(const small_inplace_allocator& other) const noexcept { return !(*this == other); }
    };
}
