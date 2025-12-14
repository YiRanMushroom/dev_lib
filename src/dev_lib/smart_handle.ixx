export module devlib.smart_handle;

import std;
import <cassert>;

namespace dev_lib {
    export struct atomic_ref_count_info_type;

    export template<typename info_type = atomic_ref_count_info_type>
    class atomic_reference_counter_control_block;

    export template<typename info_type>
    struct static_synchronized_pmr_allocator;

    struct atomic_ref_count_info_type {
        using atomic_counter_type = std::atomic_size_t;

        template<typename handle_type> requires std::is_trivially_copyable_v<handle_type>
                                                && requires(handle_type h) { h.destroy(); }
        static void destroy_handle(handle_type handle) {
            handle.destroy();
        }

        using static_allocator = static_synchronized_pmr_allocator<atomic_ref_count_info_type>;
    };

    template<typename info_type>
    struct static_synchronized_pmr_allocator {
    private:
        inline static std::pmr::synchronized_pool_resource resource{
            std::pmr::pool_options{
                .max_blocks_per_chunk = 0,
                .largest_required_pool_block = sizeof(atomic_reference_counter_control_block<info_type>)
            }
        };

    public:
        static atomic_reference_counter_control_block<info_type> *allocate() {
            void *mem = resource.allocate(sizeof(atomic_reference_counter_control_block<info_type>),
                                          alignof(atomic_reference_counter_control_block<info_type>));
            return static_cast<atomic_reference_counter_control_block<info_type> *>(mem);
        }

        static void deallocate(atomic_reference_counter_control_block<info_type> *ptr) {
            resource.deallocate(ptr, sizeof(atomic_reference_counter_control_block<info_type>),
                                alignof(atomic_reference_counter_control_block<info_type>));
        }

        template<typename... Args>
        static void construct(atomic_reference_counter_control_block<info_type> *ptr,
                              Args &&... args) {
            new(ptr) atomic_reference_counter_control_block<info_type>(std::forward<Args>(args)...);
        }

        static void destroy(atomic_reference_counter_control_block<info_type> *ptr) {
            ptr->~atomic_reference_counter_control_block<info_type>();
        }

        template<typename... Args>
        static atomic_reference_counter_control_block<info_type> *allocate_and_construct(
            Args &&... args) {
            auto ptr = allocate();
            construct(ptr, std::forward<Args>(args)...);
            return ptr;
        }

        static void destroy_and_deallocate(atomic_reference_counter_control_block<info_type> *ptr) {
            destroy(ptr);
            deallocate(ptr);
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
        using handle_type = t_handle_type;
        using static_allocator = info_type::static_allocator;

    private:
        mutable std::atomic<control_block_type *> m_control_block{nullptr};

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
            if (cb->release_strong_ref() && m_handle.has_value()) {
                // Last strong reference released, destroy handle
                info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
            }

            // Release weak reference
            if (cb->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(cb);
            }
        }

        strong_arc_handle(const strong_arc_handle &other) {
            if (auto other_cb = other.m_control_block.load(std::memory_order_acquire)) {
                other_cb->add_strong_ref();
                other_cb->add_weak_ref();
                m_control_block.store(other_cb, std::memory_order_relaxed);
                m_handle = other.m_handle;
                return;
            }

            // if other does not have control block, create a new control block
            if (other.m_handle.has_value()) {
                control_block_type *new_cb = static_allocator::allocate_and_construct();

                // expect other's control block to be null
                control_block_type *expected = nullptr;
                if (!other.m_control_block.compare_exchange_strong(
                    expected, new_cb, std::memory_order_release, std::memory_order_acquire)) {
                    // another thread created the control block first, use it
                    static_allocator::destroy_and_deallocate(new_cb);
                    expected->add_strong_ref();
                    expected->add_weak_ref();
                    m_control_block.store(expected, std::memory_order_relaxed);
                    m_handle = other.m_handle;
                    return;
                }

                new_cb->add_strong_ref();
                new_cb->add_weak_ref();
                m_control_block.store(new_cb, std::memory_order_relaxed);
                m_handle = other.m_handle;
            }
        }

        strong_arc_handle(strong_arc_handle &&other) noexcept
            : m_control_block(
                  other.m_control_block.exchange(
                      nullptr, std::memory_order_relaxed)),
              m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        strong_arc_handle &operator=(const strong_arc_handle &other) noexcept {
            if (&other == this) {
                return *this;
            }

            this->~strong_arc_handle();
            new(this) strong_arc_handle(other);
            return *this;
        }


        strong_arc_handle &operator=(strong_arc_handle &&other) noexcept {
            if (&other == this) {
                return *this;
            }

            this->~strong_arc_handle();
            new(this) strong_arc_handle(std::move(other));
            return *this;
        }

        handle_type get() const noexcept {
            return m_handle.value();
        }

        const handle_type &operator*() const noexcept {
            return m_handle.value(); // use safe access
        }

        const handle_type *operator->() const noexcept {
            return &m_handle.value(); // use safe access
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

        bool has_value() const noexcept {
            return m_handle.has_value();
        }

        strong_arc_handle clone() const noexcept {
            return strong_arc_handle(*this);
        }

        weak_arc_handle<t_handle_type, t_info_type> share_weak() const noexcept;

        void reset() noexcept {
            this->~strong_arc_handle();
            new(this) strong_arc_handle();
        }
    };

    template<typename t_handle_type, typename t_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class weak_arc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = atomic_reference_counter_control_block<info_type>;
        using static_allocator = info_type::static_allocator;
        using handle_type = t_handle_type;

    private:
        control_block_type *m_control_block{nullptr};
        std::optional<handle_type> m_handle{std::nullopt};

        weak_arc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        friend class strong_arc_handle<t_handle_type, t_info_type>;

    public:
        weak_arc_handle() noexcept = default;

        weak_arc_handle(const strong_arc_handle<t_handle_type, t_info_type> &strong_handle) noexcept {
            auto cb = strong_handle.m_control_block.load(std::memory_order_acquire);
            if (cb) {
                cb->add_weak_ref();
                m_control_block = cb;
                m_handle = strong_handle.m_handle;
                return;
            }

            // if strong_handle does not have control block, create a new control block
            if (strong_handle.m_handle.has_value()) {
                control_block_type *new_cb = static_allocator::allocate_and_construct();

                // expect strong_handle's control block to be null
                control_block_type *expected = nullptr;

                if (!strong_handle.m_control_block.compare_exchange_strong(
                    expected, new_cb, std::memory_order_release, std::memory_order_acquire)) {
                    // another thread created the control block first, use it
                    static_allocator::destroy_and_deallocate(new_cb);
                    expected->add_weak_ref();
                    m_control_block = expected;
                    m_handle = strong_handle.m_handle;
                    return;
                }

                new_cb->add_weak_ref();
                m_control_block = new_cb;
                m_handle = strong_handle.m_handle;
            }

            // else both are empty, do nothing
        }

        ~weak_arc_handle() noexcept {
            auto cb = m_control_block;
            if (!cb) {
                return;
            }

            // Release weak reference
            if (cb->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(cb);
            }
        }

        weak_arc_handle(const weak_arc_handle &other) {
            auto other_cb = other.m_control_block;
            if (other_cb) {
                other_cb->add_weak_ref();
                m_control_block = other_cb;
                m_handle = other.m_handle;
            }
        }

        weak_arc_handle(weak_arc_handle &&other) noexcept
            : m_control_block(
                  std::exchange(
                      other.m_control_block, nullptr)
              ),
              m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        weak_arc_handle &operator=(const weak_arc_handle &other) noexcept {
            if (&other == this) {
                return *this;
            }

            this->~weak_arc_handle();
            new(this) weak_arc_handle(other);
            return *this;
        }

        weak_arc_handle &operator=(weak_arc_handle &&other) noexcept {
            if (&other == this) {
                return *this;
            }

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

        strong_arc_handle<t_handle_type, t_info_type> lock() const;

        weak_arc_handle clone() const noexcept {
            return weak_arc_handle(*this);
        }

        void reset() noexcept {
            this->~weak_arc_handle();
            new(this) weak_arc_handle();
        }
    };


    template<typename t_handle_type, typename t_info_type> requires std::is_trivially_copyable_v<t_handle_type>
    weak_arc_handle<t_handle_type, t_info_type> strong_arc_handle<t_handle_type,
        t_info_type>::share_weak() const noexcept {
        return weak_arc_handle<t_handle_type, t_info_type>(*this);
    }


    template<typename t_handle_type, typename t_info_type> requires std::is_trivially_copyable_v<t_handle_type>
    strong_arc_handle<t_handle_type, t_info_type> weak_arc_handle<t_handle_type, t_info_type>::lock() const {
        auto cb = m_control_block;
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

    export template<typename info_type>
    struct static_unsynchronized_pmr_allocator {
    private:
        inline static std::pmr::unsynchronized_pool_resource resource{
            std::pmr::pool_options{
                .max_blocks_per_chunk = 0,
                .largest_required_pool_block = sizeof(reference_counter_control_block<info_type>)
            }
        };

    public:
        static reference_counter_control_block<info_type> *allocate() {
            void *mem = resource.allocate(sizeof(reference_counter_control_block<info_type>),
                                          alignof(reference_counter_control_block<info_type>));
            return static_cast<reference_counter_control_block<info_type> *>(mem);
        }

        static void deallocate(reference_counter_control_block<info_type> *ptr) {
            resource.deallocate(ptr, sizeof(reference_counter_control_block<info_type>),
                                alignof(reference_counter_control_block<info_type>));
        }

        template<typename... Args>
        static void construct(reference_counter_control_block<info_type> *ptr,
                              Args &&... args) {
            new(ptr) reference_counter_control_block<info_type>(std::forward<Args>(args)...);
        }

        static void destroy(reference_counter_control_block<info_type> *ptr) {
            ptr->~reference_counter_control_block<info_type>();
        }

        template<typename... Args>
        static reference_counter_control_block<info_type> *allocate_and_construct(
            Args &&... args) {
            auto ptr = allocate();
            construct(ptr, std::forward<Args>(args)...);
            return ptr;
        }

        static void destroy_and_deallocate(reference_counter_control_block<info_type> *ptr) {
            destroy(ptr);
            deallocate(ptr);
        }
    };

    struct ref_count_info_type {
        using counter_type = std::size_t;

        template<typename handle_type> requires std::is_trivially_copyable_v<handle_type>
                                                && requires(handle_type h) { h.destroy(); }
        static void destroy_handle(handle_type handle) {
            handle.destroy();
        }

        using static_allocator = static_unsynchronized_pmr_allocator<ref_count_info_type>;
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
    class strong_rc_handle;

    export template<typename t_handle_type, typename t_info_type = ref_count_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class weak_rc_handle;

    export template<typename t_handle_type, typename t_info_type>
        requires std::is_trivially_copyable_v<t_handle_type>
    class strong_rc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = reference_counter_control_block<info_type>;
        using static_allocator = info_type::static_allocator;
        using handle_type = t_handle_type;

    private:
        control_block_type mutable *m_control_block{nullptr};
        std::optional<handle_type> m_handle{std::nullopt};

        strong_rc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        friend class weak_rc_handle<t_handle_type, t_info_type>;

    public:
        strong_rc_handle() noexcept = default;

        strong_rc_handle(handle_type handle) noexcept
            : m_handle(handle) {}

        ~strong_rc_handle() noexcept {
            if (!m_control_block) {
                // No control block, means the control block is never created, check if handle exists
                if (m_handle.has_value()) {
                    info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
                }
                return;
            }

            // Release strong reference
            if (m_control_block->release_strong_ref() && m_handle.has_value()) {
                // Last strong reference released, destroy handle
                info_type::destroy_handle(std::exchange(m_handle, std::nullopt).value());
            }

            // Release weak reference
            if (m_control_block->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(m_control_block);
            }
        }

        strong_rc_handle(const strong_rc_handle &other) {
            if (other.m_control_block) {
                other.m_control_block->add_strong_ref();
                other.m_control_block->add_weak_ref();
                m_control_block = other.m_control_block;
                m_handle = other.m_handle;
                return;
            }

            // if other does not have control block, create a new control block
            if (other.m_handle.has_value()) {
                control_block_type *new_cb = static_allocator::allocate_and_construct();

                new_cb->add_strong_ref();
                new_cb->add_weak_ref();

                m_control_block = new_cb;
                auto old = std::exchange(other.m_control_block, new_cb);
                assert(
                    old == nullptr &&
                    "strong_rc_handle is not thread-safe, try to copy it when sending it to another thread, rather than referencing it.");
                m_handle = other.m_handle;
            }

            // else both are empty, do nothing
        }

        strong_rc_handle(strong_rc_handle &&other) noexcept
            : m_control_block(
                  std::exchange(
                      other.m_control_block, nullptr)),
              m_handle(std::exchange(other.m_handle, std::nullopt)) {}

        strong_rc_handle &operator=(const strong_rc_handle &other) noexcept {
            if (&other == this) {
                return *this;
            }

            this->~strong_rc_handle();
            new(this) strong_rc_handle(other);
            return *this;
        }

        strong_rc_handle &operator=(strong_rc_handle &&other) noexcept {
            if (&other == this) {
                return *this;
            }

            this->~strong_rc_handle();
            new(this) strong_rc_handle(std::move(other));
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

        bool has_value() const noexcept {
            return m_handle.has_value();
        }

        weak_rc_handle<t_handle_type, t_info_type> share_weak() const noexcept;

        strong_rc_handle clone() const noexcept {
            return strong_rc_handle(*this);
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

        friend class strong_rc_handle<t_handle_type, t_info_type>;

    public:
        weak_rc_handle() noexcept = default;

        weak_rc_handle(const strong_rc_handle<t_handle_type, t_info_type> &shared_handle) noexcept {
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
                    "strong_rc_handle is not thread-safe, try to copy it when sending it to another thread, rather than referencing it.");
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

        weak_rc_handle(const weak_rc_handle &other) {
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

        weak_rc_handle &operator=(const weak_rc_handle &other) noexcept {
            if (&other == this) {
                return *this;
            }

            this->~weak_rc_handle();
            new(this) weak_rc_handle(other);
            return *this;
        }

        weak_rc_handle &operator=(weak_rc_handle &&other) noexcept {
            if (&other == this) {
                return *this;
            }

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

        strong_rc_handle<t_handle_type, t_info_type> lock() const;

        weak_rc_handle clone() const noexcept {
            return weak_rc_handle(*this);
        }
    };


    template<typename t_handle_type, typename t_info_type> requires std::is_trivially_copyable_v<t_handle_type>
    weak_rc_handle<t_handle_type, t_info_type> strong_rc_handle<t_handle_type, t_info_type>::
    share_weak() const noexcept {
        return weak_rc_handle<t_handle_type, t_info_type>(*this);
    }

    template<typename t_handle_type, typename t_info_type> requires std::is_trivially_copyable_v<t_handle_type>
    strong_rc_handle<t_handle_type, t_info_type> weak_rc_handle<t_handle_type, t_info_type>::lock() const {
        if (!m_control_block) {
            return strong_rc_handle<t_handle_type, t_info_type>();
        }

        auto locked_cb = m_control_block->lock_from_weak();
        if (!locked_cb) {
            return strong_rc_handle<t_handle_type, t_info_type>();
        }

        return strong_rc_handle<t_handle_type, t_info_type>(locked_cb, m_handle.value());
    }

    export template<typename t_handle_type, typename t_info_type = ref_count_info_type, typename... Args>
        requires std::is_trivially_copyable_v<t_handle_type>
    auto make_shared_rc(Args &&... args) {
        return strong_rc_handle<t_handle_type, t_info_type>(t_handle_type(std::forward<Args>(args)...));
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

        handle_type get() const noexcept {
            return m_handle.value();
        }

        const handle_type &operator*() const noexcept {
            return m_handle.value(); // use safe access
        }

        const handle_type *operator->() const noexcept {
            return &m_handle.value(); // use safe access
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

        bool has_value() const noexcept {
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

    export template<typename t_handle_type, typename t_info_type = unique_handle_info, typename... Args>
        requires std::is_trivially_copyable_v<t_handle_type>
    auto make_unique_handle(Args &&... args) {
        return unique_handle<t_handle_type, t_info_type>(t_handle_type(std::forward<Args>(args)...));
    }
}

