export module devlib.smart_handles;

import std;
import <cassert>;

namespace dev_lib {
    export struct atomic_ref_count_info_type;

    export template<typename info_type = atomic_ref_count_info_type>
    class atomic_reference_counter_control_block;

    struct atomic_ref_count_info_type {
        using atomic_counter_type = std::atomic_size_t;
        using allocator_type = std::allocator<atomic_reference_counter_control_block<atomic_ref_count_info_type>>;

        template<typename handle_type> requires std::is_trivially_copyable_v<handle_type>
                                                && requires(handle_type h) { h.destroy(); }
        static void destroy_handle(handle_type handle) {
            handle.destroy();
        }
    };

    template<typename t_info_type>
    class atomic_reference_counter_control_block {
    public:
        using info_type = t_info_type;
        using atomic_counter_type = typename info_type::atomic_counter_type;

        // allocator_type is used in handles, not counter itself

    private:
        atomic_counter_type m_strong_count;
        atomic_counter_type m_weak_count;

    public:
        atomic_reference_counter_control_block() noexcept
            : m_strong_count(1), m_weak_count(1) {}

        void add_strong_ref() noexcept {
            m_strong_count.fetch_add(1, std::memory_order_relaxed);
        }

        bool release_strong_ref() noexcept {
            return m_strong_count.fetch_sub(1, std::memory_order_acq_rel) == 1;
        }

        void add_weak_ref() noexcept {
            m_weak_count.fetch_add(1, std::memory_order_relaxed);
        }

        bool release_weak_ref() noexcept {
            return m_weak_count.fetch_sub(1, std::memory_order_acq_rel) == 1;
        }

        atomic_counter_type &get_strong_count() noexcept {
            return m_strong_count;
        }

        atomic_counter_type &get_weak_count() noexcept {
            return m_weak_count;
        }

        atomic_reference_counter_control_block *lock_from_weak() noexcept {
            auto strong_count = m_strong_count.load(std::memory_order_relaxed);
            while (strong_count != 0) {
                if (m_strong_count.compare_exchange_weak(
                    strong_count, strong_count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                    add_weak_ref(); // also add weak ref for the new strong handle
                    return this;
                }
            }
            return nullptr;
        }
    };

    export template<typename t_handle_type, typename t_info_type = atomic_ref_count_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class strong_arc_handle;

    export template<typename t_handle_type, typename t_info_type = atomic_ref_count_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class weak_arc_handle;

    export template<typename t_handle_type, typename t_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class strong_arc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = atomic_reference_counter_control_block<info_type>;
        using allocator_type = typename info_type::allocator_type;
        using handle_type = t_handle_type;

    private:
        std::atomic<control_block_type *> m_control_block{nullptr};
        std::optional<handle_type> m_handle{std::nullopt};

        strong_arc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        friend class weak_arc_handle<t_handle_type, t_info_type>;

    public:
        strong_arc_handle() noexcept = default;

        strong_arc_handle(handle_type handle) noexcept
            : m_handle(handle) {}

        ~strong_arc_handle() noexcept {
            auto cb = m_control_block.load(std::memory_order_relaxed);
            if (!cb) {
                // No control block, means the control block is never created, check if handle exists
                if (m_handle.has_value()) {
                    info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
                }

                return;
            }

            // Release strong reference
            if (cb->release_strong_ref()) {
                // Last strong reference released, destroy handle
                info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
            }

            // Release weak reference
            if (cb->release_weak_ref()) {
                allocator_type allocator;
                std::allocator_traits<allocator_type>::destroy(allocator, cb);
                std::allocator_traits<allocator_type>::deallocate(allocator, cb, 1);
            }
        }

        strong_arc_handle(strong_arc_handle &other) {
            auto other_cb = other.m_control_block.load(std::memory_order_relaxed);
            if (other_cb) {
                other_cb->add_strong_ref();
                other_cb->add_weak_ref();
                m_control_block.store(other_cb, std::memory_order_relaxed);
                m_handle = other.m_handle;
                return;
            }

            // if other does not have control block, create a new control block
            if (other.m_handle.has_value()) {
                allocator_type allocator;
                control_block_type *new_cb = std::allocator_traits<allocator_type>::allocate(allocator, 1);
                std::allocator_traits<allocator_type>::construct(allocator, new_cb);

                new_cb->add_strong_ref();
                new_cb->add_weak_ref();

                m_control_block.store(new_cb, std::memory_order_relaxed);
                auto old = other.m_control_block.exchange(new_cb, std::memory_order_relaxed);
                assert(
                    old == nullptr &&
                    "strong_arc_handle is not thread-safe, try to copy it when sending it to another thread, rather than referencing it.");
                m_handle = other.m_handle;
            }

            // else both are empty, do nothing
        }

        strong_arc_handle(strong_arc_handle &&other) noexcept
            : m_control_block(
                  other.m_control_block.exchange(
                      nullptr, std::memory_order_relaxed)),
              m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        strong_arc_handle &operator=(strong_arc_handle &other) noexcept {
            this->~strong_arc_handle();
            new(this) strong_arc_handle(other);
            return *this;
        }

        strong_arc_handle &operator=(strong_arc_handle &&other) noexcept {
            this->~strong_arc_handle();
            new(this) strong_arc_handle(std::move(other));
            return *this;
        }

        handle_type &get() noexcept {
            return m_handle.value();
        }

        handle_type &operator*() noexcept {
            return m_handle.value(); // use safe access
        }

        handle_type *operator->() noexcept {
            return &m_handle.value(); // use safe access
        }

        operator bool() const noexcept {
            return m_handle.has_value();
        }

        weak_arc_handle<t_handle_type, t_info_type> share_weak() noexcept;

        strong_arc_handle clone() noexcept {
            return strong_arc_handle(*this);
        }
    };

    template<typename t_handle_type, typename t_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class weak_arc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = atomic_reference_counter_control_block<info_type>;
        using allocator_type = typename info_type::allocator_type;
        using handle_type = t_handle_type;

    private:
        std::atomic<control_block_type *> m_control_block{nullptr};
        std::optional<handle_type> m_handle{std::nullopt};

        weak_arc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        friend class strong_arc_handle<t_handle_type, t_info_type>;

    public:
        weak_arc_handle() noexcept = default;

        weak_arc_handle(strong_arc_handle<t_handle_type, t_info_type> &strong_handle) noexcept {
            auto cb = strong_handle.m_control_block.load(std::memory_order_relaxed);
            if (cb) {
                cb->add_weak_ref();
                m_control_block.store(cb, std::memory_order_relaxed);
                m_handle = strong_handle.m_handle;
                return;
            }

            // if strong_handle does not have control block, create a new control block
            if (strong_handle.m_handle.has_value()) {
                allocator_type allocator;
                control_block_type *new_cb = std::allocator_traits<allocator_type>::allocate(allocator, 1);
                std::allocator_traits<allocator_type>::construct(allocator, new_cb);

                new_cb->add_weak_ref();

                m_control_block.store(new_cb, std::memory_order_relaxed);
                auto old = strong_handle.m_control_block.exchange(new_cb, std::memory_order_relaxed);
                assert(
                    old == nullptr &&
                    "strong_arc_handle is not thread-safe, try to copy it when sending it to another thread, rather than referencing it.");
                m_handle = strong_handle.m_handle;
            }

            // else both are empty, do nothing
        }

        ~weak_arc_handle() noexcept {
            auto cb = m_control_block.load(std::memory_order_relaxed);
            if (!cb) {
                return;
            }

            // Release weak reference
            if (cb->release_weak_ref()) {
                allocator_type allocator;
                std::allocator_traits<allocator_type>::destroy(allocator, cb);
                std::allocator_traits<allocator_type>::deallocate(allocator, cb, 1);
            }
        }

        weak_arc_handle(weak_arc_handle &other) {
            auto other_cb = other.m_control_block.load(std::memory_order_relaxed);
            if (other_cb) {
                other_cb->add_weak_ref();
                m_control_block.store(other_cb, std::memory_order_relaxed);
                m_handle = other.m_handle;
            }
        }

        weak_arc_handle(weak_arc_handle &&other) noexcept
            : m_control_block(
                  other.m_control_block.exchange(
                      nullptr, std::memory_order_relaxed)),
              m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        weak_arc_handle &operator=(weak_arc_handle &other) noexcept {
            this->~weak_arc_handle();
            new(this) weak_arc_handle(other);
            return *this;
        }

        weak_arc_handle &operator=(weak_arc_handle &&other) noexcept {
            this->~weak_arc_handle();
            new(this) weak_arc_handle(std::move(other));
            return *this;
        }

        bool expired() const noexcept {
            auto cb = m_control_block.load(std::memory_order_relaxed);
            if (!cb) {
                return true;
            }
            return cb->get_strong_count().load(std::memory_order_acquire) == 0;
        }

        strong_arc_handle<t_handle_type, t_info_type> lock();

        weak_arc_handle clone() noexcept {
            return weak_arc_handle(*this);
        }
    };

    template<typename t_handle_type, typename t_info_type> requires std::is_trivially_copyable_v<t_handle_type>
    weak_arc_handle<t_handle_type, t_info_type> strong_arc_handle<t_handle_type, t_info_type>::share_weak() noexcept {
        return weak_arc_handle<t_handle_type, t_info_type>(*this);
    }

    template<typename t_handle_type, typename t_info_type> requires std::is_trivially_copyable_v<t_handle_type>
    strong_arc_handle<t_handle_type, t_info_type> weak_arc_handle<t_handle_type, t_info_type>::lock() {
        auto cb = m_control_block.load(std::memory_order_relaxed);
        if (!cb) {
            return strong_arc_handle<t_handle_type, t_info_type>();
        }

        auto locked_cb = cb->lock_from_weak();
        if (!locked_cb) {
            return strong_arc_handle<t_handle_type, t_info_type>();
        }

        return strong_arc_handle<t_handle_type, t_info_type>(locked_cb, m_handle.value());
    }

    export template<typename T, typename t_info_type = atomic_ref_count_info_type, typename... Args>
        requires std::is_trivially_copyable_v<T>
    auto make_shared_arc(Args &&... args) {
        return strong_arc_handle<T, t_info_type>(T(std::forward<Args>(args)...));
    }

    export struct ref_count_info_type;

    export template<typename info_type = ref_count_info_type>
    class reference_counter_control_block;

    struct ref_count_info_type {
        using counter_type = std::size_t;
        using allocator_type = std::allocator<reference_counter_control_block<ref_count_info_type>>;

        template<typename handle_type> requires std::is_trivially_copyable_v<handle_type>
                                                && requires(handle_type h) { h.destroy(); }
        static void destroy_handle(handle_type handle) {
            handle.destroy();
        }
    };

    export template<typename t_info_type>
    class reference_counter_control_block {
    public:
        using info_type = t_info_type;
        using counter_type = typename info_type::counter_type;

        // allocator_type is used in handles, not counter itself
    private:
        counter_type m_strong_count;
        counter_type m_weak_count;

    public:
        reference_counter_control_block() noexcept
            : m_strong_count(1), m_weak_count(1) {}

        void add_strong_ref() noexcept {
            ++m_strong_count;
        }

        bool release_strong_ref() noexcept {
            return --m_strong_count == 0;
        }

        void add_weak_ref() noexcept {
            ++m_weak_count;
        }

        bool release_weak_ref() noexcept {
            return --m_weak_count == 0;
        }

        counter_type &get_strong_count() noexcept {
            return m_strong_count;
        }

        counter_type &get_weak_count() noexcept {
            return m_weak_count;
        }

        reference_counter_control_block *lock_from_weak() noexcept {
            if (m_strong_count == 0) {
                return nullptr;
            }
            ++m_strong_count;
            ++m_weak_count; // also add weak ref for the new strong handle
            return this;
        }
    };

    export template<typename t_handle_type, typename t_info_type = ref_count_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class shared_rc_handle;

    export template<typename t_handle_type, typename t_info_type = ref_count_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class weak_rc_handle;

    export template<typename t_handle_type, typename t_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class shared_rc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = reference_counter_control_block<info_type>;
        using allocator_type = typename info_type::allocator_type;
        using handle_type = t_handle_type;

    private:
        control_block_type *m_control_block{nullptr};
        std::optional<handle_type> m_handle{std::nullopt};

        shared_rc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        friend class weak_rc_handle<t_handle_type, t_info_type>;

    public:
        shared_rc_handle() noexcept = default;

        shared_rc_handle(handle_type handle) noexcept
            : m_handle(handle) {}

        ~shared_rc_handle() noexcept {
            if (!m_control_block) {
                // No control block, means the control block is never created, check if handle exists
                if (m_handle.has_value()) {
                    info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
                }
                return;
            }

            // Release strong reference
            if (m_control_block->release_strong_ref()) {
                // Last strong reference released, destroy handle
                info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
            }

            // Release weak reference
            if (m_control_block->release_weak_ref()) {
                allocator_type allocator;
                std::allocator_traits<allocator_type>::destroy(allocator, m_control_block);
                std::allocator_traits<allocator_type>::deallocate(allocator, m_control_block, 1);
            }
        }

        shared_rc_handle(shared_rc_handle &other) {
            if (other.m_control_block) {
                other.m_control_block->add_strong_ref();
                other.m_control_block->add_weak_ref();
                m_control_block = other.m_control_block;
                m_handle = other.m_handle;
                return;
            }

            // if other does not have control block, create a new control block
            if (other.m_handle.has_value()) {
                allocator_type allocator;
                control_block_type *new_cb = std::allocator_traits<allocator_type>::allocate(allocator, 1);
                std::allocator_traits<allocator_type>::construct(allocator, new_cb);

                new_cb->add_strong_ref();
                new_cb->add_weak_ref();

                m_control_block = new_cb;
                auto old = std::exchange(other.m_control_block, new_cb);
                assert(
                    old == nullptr &&
                    "shared_rc_handle is not thread-safe, try to copy it when sending it to another thread, rather than referencing it.");
                m_handle = other.m_handle;
            }

            // else both are empty, do nothing
        }

        shared_rc_handle(shared_rc_handle &&other) noexcept
            : m_control_block(
                  std::exchange(
                      other.m_control_block, nullptr)),
              m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        shared_rc_handle &operator=(shared_rc_handle &other) noexcept {
            this->~shared_rc_handle();
            new(this) shared_rc_handle(other);
            return *this;
        }

        shared_rc_handle &operator=(shared_rc_handle &&other) noexcept {
            this->~shared_rc_handle();
            new(this) shared_rc_handle(std::move(other));
            return *this;
        }

        handle_type &get() noexcept {
            return m_handle.value();
        }

        handle_type &operator*() noexcept {
            return m_handle.value(); // use safe access
        }

        handle_type *operator->() noexcept {
            return &m_handle.value(); // use safe access
        }

        operator bool() const noexcept {
            return m_handle.has_value();
        }

        weak_rc_handle<t_handle_type, t_info_type> share_weak() noexcept;

        shared_rc_handle clone() noexcept {
            return shared_rc_handle(*this);
        }
    };

    template<typename t_handle_type, typename t_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class weak_rc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = reference_counter_control_block<info_type>;
        using allocator_type = typename info_type::allocator_type;
        using handle_type = t_handle_type;

    private:
        control_block_type *m_control_block{nullptr};
        std::optional<handle_type> m_handle{std::nullopt};

        weak_rc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        friend class shared_rc_handle<t_handle_type, t_info_type>;

    public:
        weak_rc_handle() noexcept = default;

        weak_rc_handle(shared_rc_handle<t_handle_type, t_info_type> &shared_handle) noexcept {
            if (shared_handle.m_control_block) {
                shared_handle.m_control_block->add_weak_ref();
                m_control_block = shared_handle.m_control_block;
                m_handle = shared_handle.m_handle;
                return;
            }

            // if shared_handle does not have control block, create a new control block
            if (shared_handle.m_handle.has_value()) {
                allocator_type allocator;
                control_block_type *new_cb = std::allocator_traits<allocator_type>::allocate(allocator, 1);
                std::allocator_traits<allocator_type>::construct(allocator, new_cb);

                new_cb->add_weak_ref();

                m_control_block = new_cb;
                auto old = std::exchange(shared_handle.m_control_block, new_cb);
                assert(
                    old == nullptr &&
                    "shared_rc_handle is not thread-safe, try to copy it when sending it to another thread, rather than referencing it.");
                m_handle = shared_handle.m_handle;
            }

            // else both are empty, do nothing
        }

        ~weak_rc_handle() noexcept {
            if (!m_control_block) {
                return;
            }

            // Release weak reference
            if (m_control_block->release_weak_ref()) {
                allocator_type allocator;
                std::allocator_traits<allocator_type>::destroy(allocator, m_control_block);
                std::allocator_traits<allocator_type>::deallocate(allocator, m_control_block, 1);
            }
        }

        weak_rc_handle(weak_rc_handle &other) {
            if (other.m_control_block) {
                other.m_control_block->add_weak_ref();
                m_control_block = other.m_control_block;
                m_handle = other.m_handle;
            }
        }

        weak_rc_handle(weak_rc_handle &&other) noexcept
            : m_control_block(
                  std::exchange(
                      other.m_control_block, nullptr)),
              m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        weak_rc_handle &operator=(weak_rc_handle &other) noexcept {
            this->~weak_rc_handle();
            new(this) weak_rc_handle(other);
            return *this;
        }

        weak_rc_handle &operator=(weak_rc_handle &&other) noexcept {
            this->~weak_rc_handle();
            new(this) weak_rc_handle(std::move(other));
            return *this;
        }

        bool expired() const noexcept {
            if (!m_control_block) {
                return true;
            }
            return m_control_block->get_strong_count() == 0;
        }

        shared_rc_handle<t_handle_type, t_info_type> lock();

        weak_rc_handle clone() noexcept {
            return weak_rc_handle(*this);
        }
    };

    template<typename t_handle_type, typename t_info_type> requires std::is_trivially_copyable_v<t_handle_type>
    shared_rc_handle<t_handle_type, t_info_type> weak_rc_handle<t_handle_type, t_info_type>::lock() {
        if (!m_control_block) {
            return shared_rc_handle<t_handle_type, t_info_type>();
        }

        auto locked_cb = m_control_block->lock_from_weak();
        if (!locked_cb) {
            return shared_rc_handle<t_handle_type, t_info_type>();
        }

        return shared_rc_handle<t_handle_type, t_info_type>(locked_cb, m_handle.value());
    }

    // unique handle
    struct unique_handle_info {
        static void destroy_handle(auto handle) {
            handle.destroy();
        }
    };

    export template<typename t_handle_type, typename t_info_type = unique_handle_info>
        requires std::is_trivially_copyable_v<t_handle_type>
    class unique_handle {
    public:
        using info_type = t_info_type;
        using handle_type = t_handle_type;

    private:
        std::optional<handle_type> m_handle{std::nullopt};

    public:
        unique_handle() noexcept = default;

        unique_handle(handle_type handle) noexcept
            : m_handle(handle) {}

        ~unique_handle() noexcept {
            if (m_handle.has_value()) {
                info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
            }
        }

        unique_handle(unique_handle &) = delete;

        unique_handle(unique_handle &&other) noexcept
            : m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        unique_handle &operator=(unique_handle &) = delete;

        unique_handle &operator=(unique_handle &&other) noexcept {
            this->~unique_handle();
            new(this) unique_handle(std::move(other));
            return *this;
        }

        handle_type &get() noexcept {
            return m_handle.value();
        }

        handle_type &operator*() noexcept {
            return m_handle.value(); // use safe access
        }

        handle_type *operator->() noexcept {
            return &m_handle.value(); // use safe access
        }

        operator bool() const noexcept {
            return m_handle.has_value();
        }

        void reset() noexcept {
            this->~unique_handle();
            new(this) unique_handle();
        }

        handle_type release() noexcept {
            return std::exchange(m_handle, std::nullopt).value();
        }
    };
}
