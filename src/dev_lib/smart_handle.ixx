export module devlib.smart_handle;

import std;
import <cassert>;
import small_inplace_allocator;
import pmr_allocators;

namespace dev_lib {
    // Re-export handle types for convenience
    template<typename Common, typename T>
    concept implicitly_convertible_from = requires(T t) {
        [](const Common &) {}(t);
    };

    template<typename Common, typename T>
    concept explicitly_convertible_from = std::convertible_to<T, Common> && !implicitly_convertible_from<Common, T>;

    template<typename Common, typename T>
    concept convertible_from = implicitly_convertible_from<Common, T> || explicitly_convertible_from<Common, T>;

    struct auto_increment_tag {};

    // Forward declarations for friend declarations
    export struct atomic_ref_count_info_type;
    export struct ref_count_info_type;
    export struct unique_handle_info;

    export template<typename T, typename allocator_type = std::allocator<T>>
    class pointer_handle {
    public:
        using value_type = T;

    private:
        [[no_unique_address]] allocator_type m_allocator{};

        using allocator_traits = std::allocator_traits<allocator_type>;

        T *m_ptr{nullptr};

    public:
        pointer_handle() noexcept = default;

        ~pointer_handle() noexcept = default;

        // Shallow copy is intentional - ownership is managed by outer smart pointer
        pointer_handle(const pointer_handle &other) noexcept = default;

        // Move constructor
        pointer_handle(pointer_handle &&other) noexcept = default;

        // Copy assignment
        pointer_handle &operator=(const pointer_handle &other) noexcept = default;

        // Move assignment
        pointer_handle &operator=(pointer_handle &&other) noexcept = default;

        T *get() const noexcept {
            return m_ptr;
        }

        T &operator*() const noexcept {
            return *m_ptr;
        }

        T *operator->() const noexcept {
            return m_ptr;
        }

        operator bool() const noexcept {
            return m_ptr != nullptr;
        }

        void destroy() noexcept {
            if (m_ptr) {
                // Call destroy() method if the object has one
                if constexpr (requires { m_ptr->destroy(); }) {
                    m_ptr->destroy();
                }
                // Then call destructor and deallocate
                allocator_traits::destroy(m_allocator, m_ptr);
                allocator_traits::deallocate(m_allocator, m_ptr, 1);
                m_ptr = nullptr;
            }
        }

        template<implicitly_convertible_from<T> U>
        operator pointer_handle<U>() const noexcept {
            return pointer_handle<U>(m_ptr);
        }

        template<explicitly_convertible_from<T> U>
        explicit operator pointer_handle<U>() const noexcept {
            return pointer_handle<U>(static_cast<U *>(m_ptr));
        }

        template<convertible_from<T> U>
        pointer_handle<U> static_pointer_cast() const noexcept {
            return pointer_handle<U>(static_cast<U *>(m_ptr));
        }

        template<std::derived_from<T> U>
        pointer_handle<U> dynamic_pointer_cast() const noexcept {
            return pointer_handle<U>(dynamic_cast<U *>(m_ptr));
        }

        bool has_value() const noexcept {
            return m_ptr != nullptr;
        }

        // Static factory method to create pointer_handle with allocated object
        template<typename... Args>
        static pointer_handle make(Args &&... args) {
            allocator_type allocator{};
            T *ptr = allocator_traits::allocate(allocator, 1);
            allocator_traits::construct(allocator, ptr, std::forward<Args>(args)...);
            return pointer_handle(ptr);
        }

    private:
        // Private constructor for internal use only
        explicit pointer_handle(T *ptr) noexcept : m_ptr(ptr) {}

        friend struct atomic_ref_count_info_type;
        friend struct ref_count_info_type;
        friend struct unique_handle_info;
    };

    // Common interface for all handles - default implementation
    template<typename T>
    struct common_handle_interface {
        using derived_type = T;

        // Default implementation for generic handles
        auto get(this const derived_type &self) noexcept {
            return self.m_handle;
        }

        const auto &operator*(this const derived_type &self) noexcept {
            assert(self.has_value());
            return self.m_handle;
        }

        auto &operator*(this derived_type &self) noexcept {
            assert(self.has_value());
            return self.m_handle;
        }

        const auto *operator->(this const derived_type &self) noexcept {
            assert(self.has_value());
            return &self.m_handle;
        }

        auto *operator->(this derived_type &self) noexcept {
            assert(self.has_value());
            return &self.m_handle;
        }

        operator bool(this const derived_type &self) noexcept {
            return self.has_value();
        }
    };

    // Specialized interface for pointer_handle wrappers - expose inner pointer directly
    template<typename V, template<typename value_type, typename...> typename allocator_template, template<typename
        handle_type, typename...> typename handle_template, typename... Args>
    struct common_handle_interface<handle_template<pointer_handle<V, allocator_template<V>>, Args...>> {
        using derived_type = handle_template<pointer_handle<V, allocator_template<V>>, Args...>;
        using handle_type = pointer_handle<V, allocator_template<V>>;

        V *get(this const derived_type &self) noexcept {
            return self.m_handle.get();
        }

        V &operator*(this const derived_type &self) noexcept {
            assert(self.has_value());
            return *self.m_handle;
        }

        V *operator->(this const derived_type &self) noexcept {
            assert(self.has_value());
            return self.m_handle.get();
        }

        operator bool(this const derived_type &self) noexcept {
            return self.has_value();
        }
    };

    // Shared handle functionality (arc/rc) - inherits common interface and adds shared-specific features
    template<typename T>
    struct inject_shared_functionality : public common_handle_interface<T> {
        // Base case: no additional functionality
    };

    template<typename V, template<typename value_type, typename...> typename allocator_template, template<typename
        pointer_handle_type, typename...> typename handle_template, typename info_type>
    struct inject_shared_functionality<handle_template<pointer_handle<V, allocator_template<V>>, info_type>>
            : public common_handle_interface<handle_template<pointer_handle<V, allocator_template<V>>, info_type>> {
        using my_managed_type = handle_template<pointer_handle<V, allocator_template<V>>>;

        using my_unmanaged_type = pointer_handle<V, allocator_template<V>>;

        template<typename T>
        using rebind_managed_type = handle_template<pointer_handle<T, allocator_template<T>>>;

        template<typename T>
        using rebind_unmanaged_type = pointer_handle<T, allocator_template<T>>;

        template<implicitly_convertible_from<V> T>
        operator rebind_managed_type<T>(this const my_managed_type &self) noexcept {
            bool should_construct = self.ensure_has_underlying_control_block();
            if (!should_construct) {
                return rebind_managed_type<T>();
            }

            return rebind_managed_type<T>(self.m_control_block.load(std::memory_order_relaxed),
                                          self.m_handle, auto_increment_tag{});
        }

        template<explicitly_convertible_from<V> T>
        explicit operator rebind_managed_type<T>(this const my_managed_type &self) noexcept {
            bool should_construct = self.ensure_has_underlying_control_block();
            if (!should_construct) {
                return rebind_managed_type<T>();
            }

            return rebind_managed_type<T>(self.m_control_block.load(std::memory_order_relaxed),
                                          rebind_unmanaged_type<T>(self.m_handle), auto_increment_tag{});
        }

        template<convertible_from<V> T>
        rebind_managed_type<T> static_pointer_cast(this const my_managed_type &self) noexcept {
            bool should_construct = self.ensure_has_underlying_control_block();
            if (!should_construct) {
                return rebind_managed_type<T>();
            }

            return rebind_managed_type<T>(self.m_control_block.load(std::memory_order_relaxed),
                                          self.m_handle.template static_pointer_cast<T>(), auto_increment_tag{});
        }

        template<std::derived_from<V> T>
        rebind_managed_type<T> dynamic_pointer_cast(this const my_managed_type &self) noexcept {
            bool should_construct = self.ensure_has_underlying_control_block();

            if (!should_construct) {
                return rebind_managed_type<T>();
            }

            auto casted_handle = self.m_handle.template dynamic_pointer_cast<T>();

            if (!casted_handle) {
                return rebind_managed_type<T>();
            }

            return rebind_managed_type<T>(self.m_control_block.load(std::memory_order_relaxed),
                                          std::move(casted_handle),
                                          auto_increment_tag{});
        }
    };

    export template<typename T>
    bool operator==(const pointer_handle<T> &lhs, std::nullptr_t rhs) noexcept {
        return lhs.get() == nullptr;
    }

    export template<typename T>
    bool operator==(std::nullptr_t lhs, const pointer_handle<T> &rhs) noexcept {
        return nullptr == rhs.get();
    }

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

        static bool has_value(auto handle) {
            return handle.has_value();
        }

        // Generic make_handle - calls T::make with forwarded arguments
        template<typename T, typename... Args>
        static auto make_handle(Args &&... args) {
            return T::make(std::forward<Args>(args)...);
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

    // Permissive trait for unique handles (defaults to false)
    template<typename T>
    struct unique_permissive : std::false_type {};

    // Concept: if unique_permissive, skip trivially_copyable requirement
    template<typename T>
    concept handle_type_concept = unique_permissive<T>::value || std::is_trivially_copyable_v<T>;

    export template<typename t_handle_type, typename t_info_type = atomic_ref_count_info_type>
        requires handle_type_concept<t_handle_type>
    class strong_arc_handle;

    export template<typename t_handle_type, typename t_info_type = atomic_ref_count_info_type>
        requires handle_type_concept<t_handle_type>
    class weak_arc_handle;

    export template<typename t_handle_type, typename t_info_type>
        requires handle_type_concept<t_handle_type>
    class strong_arc_handle : public inject_shared_functionality<strong_arc_handle<t_handle_type, t_info_type>> {
    public:
        using info_type = t_info_type;
        using control_block_type = atomic_reference_counter_control_block<info_type>;
        using handle_type = t_handle_type;
        using static_allocator = info_type::static_allocator;

    private:
        mutable std::atomic<control_block_type *> m_control_block{nullptr};

        handle_type m_handle{};

        strong_arc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        strong_arc_handle(control_block_type *cb, handle_type handle, auto_increment_tag) noexcept
            : m_control_block(cb), m_handle(handle) {
            cb->add_strong_ref();
            cb->add_weak_ref();
        }

        friend class weak_arc_handle<t_handle_type, t_info_type>;

        template<typename T>
        friend struct inject_shared_functionality;

        template<typename T>
        friend struct common_handle_interface;

    public:
        strong_arc_handle() noexcept = default;

        strong_arc_handle(handle_type handle) noexcept
            : m_handle(handle) {}


        ~strong_arc_handle() noexcept {
            auto cb = m_control_block.exchange(nullptr, std::memory_order_relaxed);
            if (!cb) {
                // No control block, means the control block is never created, check if handle exists
                if (info_type::has_value(m_handle)) {
                    info_type::destroy_handle(std::exchange(m_handle, {}));
                }

                return;
            }

            // Release strong reference
            if (cb->release_strong_ref() && info_type::has_value(m_handle)) {
                // Last strong reference released, destroy handle
                info_type::destroy_handle(std::exchange(m_handle, {}));
            }

            // Release weak reference
            if (cb->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(cb);
            }
        }

        strong_arc_handle(const strong_arc_handle &other) {
            if (!other.ensure_has_underlying_control_block()) {
                // both are empty, do nothing
                return;
            }

            auto cb = other.m_control_block.load(std::memory_order_relaxed);
            assert(cb != nullptr);
            cb->add_strong_ref();
            cb->add_weak_ref();
            m_control_block.store(cb, std::memory_order_relaxed);
            m_handle = other.m_handle;
        }

        bool ensure_has_underlying_control_block() const {
            auto cb = m_control_block.load(std::memory_order_acquire);
            if (cb) {
                return true;
            }

            if (info_type::has_value(m_handle)) {
                control_block_type *new_cb = static_allocator::allocate_and_construct();

                // Try to set the control block atomically
                control_block_type *expected = nullptr;
                if (!m_control_block.compare_exchange_strong(
                    expected, new_cb, std::memory_order_release, std::memory_order_acquire)) {
                    // Another thread created the control block first, use it and clean up ours
                    static_allocator::destroy_and_deallocate(new_cb);
                }
                return true;
            }
            return false;
        }

        strong_arc_handle(strong_arc_handle &&other) noexcept
            : m_control_block(
                  other.m_control_block.exchange(
                      nullptr, std::memory_order_relaxed)),
              m_handle(std::exchange(other.m_handle, {})) {}

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


        bool has_value() const noexcept {
            return info_type::has_value(m_handle);
        }

        strong_arc_handle clone() const noexcept {
            return strong_arc_handle(*this);
        }

        weak_arc_handle<t_handle_type, t_info_type> share_weak() const noexcept;

        void reset() noexcept {
            // Destroy handle and control block reference
            auto cb = m_control_block.exchange(nullptr, std::memory_order_relaxed);
            if (cb) {
                bool is_last = cb->release_strong_ref();
                if (is_last && info_type::has_value(m_handle)) {
                    info_type::destroy_handle(std::exchange(m_handle, {}));
                } else {
                    m_handle = {}; // Always reset handle even if not the last reference
                }
                if (cb->release_weak_ref()) {
                    static_allocator::destroy_and_deallocate(cb);
                }
            } else {
                if (info_type::has_value(m_handle)) {
                    info_type::destroy_handle(std::exchange(m_handle, {}));
                }
            }
        }
    };

    template<typename t_handle_type, typename t_info_type>
        requires handle_type_concept<t_handle_type>
    class weak_arc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = atomic_reference_counter_control_block<info_type>;
        using static_allocator = info_type::static_allocator;
        using handle_type = t_handle_type;

    private:
        control_block_type *m_control_block{nullptr};
        handle_type m_handle{};

        weak_arc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        weak_arc_handle(control_block_type *cb, handle_type handle, auto_increment_tag) noexcept
            : m_control_block(cb), m_handle(handle) {
            cb->add_weak_ref();
        }

        friend class strong_arc_handle<t_handle_type, t_info_type>;

    public:
        weak_arc_handle() noexcept = default;

        weak_arc_handle(const strong_arc_handle<t_handle_type, t_info_type> &strong_handle) noexcept {
            if (!strong_handle.ensure_has_underlying_control_block()) {
                // both are empty, do nothing
                return;
            }

            auto cb = strong_handle.m_control_block.load(std::memory_order_relaxed);
            assert(cb != nullptr);
            cb->add_weak_ref();
            m_control_block = cb;
            m_handle = strong_handle.m_handle;
        }

        bool ensure_has_underlying_control_block() const {
            if (m_control_block) {
                return true;
            }

            return false;
        }

        ~weak_arc_handle() noexcept {
            auto cb = std::exchange(m_control_block, nullptr);
            if (cb && cb->release_weak_ref()) {
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
              m_handle(std::exchange(other.m_handle, {})) {}

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
            auto cb = std::exchange(m_control_block, nullptr);
            if (cb && cb->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(cb);
            }
            m_handle = {};
        }
    };


    template<typename t_handle_type, typename t_info_type> requires handle_type_concept<t_handle_type>
    weak_arc_handle<t_handle_type, t_info_type> strong_arc_handle<t_handle_type,
        t_info_type>::share_weak() const noexcept {
        return weak_arc_handle<t_handle_type, t_info_type>(*this);
    }


    template<typename t_handle_type, typename t_info_type> requires handle_type_concept<t_handle_type>
    strong_arc_handle<t_handle_type, t_info_type> weak_arc_handle<t_handle_type, t_info_type>::lock() const {
        auto cb = m_control_block;
        if (!cb) {
            return strong_arc_handle<t_handle_type, t_info_type>();
        }

        auto locked_cb = cb->lock_from_weak();
        if (!locked_cb) {
            return strong_arc_handle<t_handle_type, t_info_type>();
        }

        return strong_arc_handle<t_handle_type, t_info_type>(locked_cb, m_handle);
    }

    export template<typename T, typename allocator_type = std::allocator<T>, typename t_info_type =
        atomic_ref_count_info_type, typename... Args>
    auto make_arc_pointer(Args &&... args) {
        using handle_type = pointer_handle<T, allocator_type>;
        return strong_arc_handle<handle_type, t_info_type>(
            t_info_type::template make_handle<handle_type>(std::forward<Args>(args)...)
        );
    }

    export template<typename T, typename allocator_type = std::allocator<T>, typename t_info_type =
        atomic_ref_count_info_type, typename... Args>
    auto make_arc_handle(Args &&... args) {
        return make_arc_pointer<T, allocator_type, t_info_type>(std::forward<Args>(args)...);
    }


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

        static bool has_value(auto handle) {
            return handle.has_value();
        }

        // Generic make_handle - calls T::make with forwarded arguments
        template<typename T, typename... Args>
        static auto make_handle(Args &&... args) {
            return T::make(std::forward<Args>(args)...);
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
        requires handle_type_concept<t_handle_type>
    class strong_rc_handle;

    export template<typename t_handle_type, typename t_info_type = ref_count_info_type>
        requires handle_type_concept<t_handle_type>
    class weak_rc_handle;

    export template<typename t_handle_type, typename t_info_type>
        requires handle_type_concept<t_handle_type>
    class strong_rc_handle : public inject_shared_functionality<strong_rc_handle<t_handle_type, t_info_type>> {
    public:
        using info_type = t_info_type;
        using control_block_type = reference_counter_control_block<info_type>;
        using static_allocator = info_type::static_allocator;
        using handle_type = t_handle_type;

    private:
        control_block_type mutable *m_control_block{nullptr};
        handle_type m_handle{};

        strong_rc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        strong_rc_handle(control_block_type *cb, handle_type handle, auto_increment_tag) noexcept
            : m_control_block(cb), m_handle(handle) {
            cb->add_strong_ref();
            cb->add_weak_ref();
        }

        friend class weak_rc_handle<t_handle_type, t_info_type>;
        template<typename T>
        friend struct inject_shared_functionality;

        template<typename T>
        friend struct common_handle_interface;

    public:
        strong_rc_handle() noexcept = default;

        strong_rc_handle(handle_type handle) noexcept
            : m_handle(handle) {}

        ~strong_rc_handle() noexcept {
            auto cb = std::exchange(m_control_block, nullptr);
            if (!cb) {
                // No control block, means the control block is never created, check if handle exists
                if (info_type::has_value(m_handle)) {
                    info_type::destroy_handle(std::exchange(m_handle, {}));
                }
                return;
            }

            // Release strong reference
            if (cb->release_strong_ref() && info_type::has_value(m_handle)) {
                // Last strong reference released, destroy handle
                info_type::destroy_handle(std::exchange(m_handle, {}));
            }

            // Release weak reference
            if (cb->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(cb);
            }
        }

        bool ensure_has_underlying_control_block() const {
            if (m_control_block) {
                return true;
            }

            if (info_type::has_value(m_handle)) {
                m_control_block = static_allocator::allocate_and_construct();

                return true;
            }
            return false;
        }

        strong_rc_handle(const strong_rc_handle &other) {
            if (!other.ensure_has_underlying_control_block()) {
                // both are empty, do nothing
                return;
            }

            other.m_control_block->add_strong_ref();
            other.m_control_block->add_weak_ref();
            m_control_block = other.m_control_block;
            m_handle = other.m_handle;
        }

        strong_rc_handle(strong_rc_handle &&other) noexcept
            : m_control_block(
                  std::exchange(
                      other.m_control_block, nullptr)),
              m_handle(std::exchange(other.m_handle, {})) {}

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


        bool has_value() const noexcept {
            return info_type::has_value(m_handle);
        }

        weak_rc_handle<t_handle_type, t_info_type> share_weak() const noexcept;

        strong_rc_handle clone() const noexcept {
            return strong_rc_handle(*this);
        }

        void reset() noexcept {
            // Destroy handle and control block reference
            auto cb = std::exchange(m_control_block, nullptr);
            if (cb) {
                bool is_last = cb->release_strong_ref();
                if (is_last && info_type::has_value(m_handle)) {
                    info_type::destroy_handle(std::exchange(m_handle, {}));
                } else {
                    m_handle = {}; // Always reset handle even if not the last reference
                }
                if (cb->release_weak_ref()) {
                    static_allocator::destroy_and_deallocate(cb);
                }
            } else {
                if (info_type::has_value(m_handle)) {
                    info_type::destroy_handle(std::exchange(m_handle, {}));
                }
            }
        }
    };

    template<typename t_handle_type, typename t_info_type>
        requires handle_type_concept<t_handle_type>
    class weak_rc_handle {
    public:
        using info_type = t_info_type;
        using control_block_type = reference_counter_control_block<info_type>;
        using static_allocator = info_type::static_allocator;
        using handle_type = t_handle_type;

    private:
        control_block_type *m_control_block{nullptr};
        handle_type m_handle{};

        weak_rc_handle(control_block_type *cb, handle_type handle) noexcept
            : m_control_block(cb), m_handle(handle) {}

        friend class strong_rc_handle<t_handle_type, t_info_type>;

    public:
        weak_rc_handle() noexcept = default;

        weak_rc_handle(const strong_rc_handle<t_handle_type, t_info_type> &shared_handle) noexcept {
            if (!shared_handle.ensure_has_underlying_control_block()) {
                // both are empty, do nothing
                return;
            }

            shared_handle.m_control_block->add_weak_ref();
            m_control_block = shared_handle.m_control_block;
            m_handle = shared_handle.m_handle;
        }

        bool ensure_has_underlying_control_block() const {
            if (m_control_block) {
                return true;
            }

            return false;
        }

        ~weak_rc_handle() noexcept {
            auto cb = std::exchange(m_control_block, nullptr);
            if (cb && cb->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(cb);
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
              m_handle(std::exchange(other.m_handle, {})) {}

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

        void reset() noexcept {
            auto cb = std::exchange(m_control_block, nullptr);
            if (cb && cb->release_weak_ref()) {
                static_allocator::destroy_and_deallocate(cb);
            }
            m_handle = {};
        }
    };


    template<typename t_handle_type, typename t_info_type> requires handle_type_concept<t_handle_type>
    weak_rc_handle<t_handle_type, t_info_type> strong_rc_handle<t_handle_type, t_info_type>::
    share_weak() const noexcept {
        return weak_rc_handle<t_handle_type, t_info_type>(*this);
    }

    template<typename t_handle_type, typename t_info_type> requires handle_type_concept<t_handle_type>
    strong_rc_handle<t_handle_type, t_info_type> weak_rc_handle<t_handle_type, t_info_type>::lock() const {
        if (!m_control_block) {
            return strong_rc_handle<t_handle_type, t_info_type>();
        }

        auto locked_cb = m_control_block->lock_from_weak();
        if (!locked_cb) {
            return strong_rc_handle<t_handle_type, t_info_type>();
        }

        return strong_rc_handle<t_handle_type, t_info_type>(locked_cb, m_handle);
    }

    export template<typename T, typename allocator_type = std::allocator<T>, typename t_info_type = ref_count_info_type,
        typename... Args>
    auto make_rc_pointer(Args &&... args) {
        using handle_type = pointer_handle<T, allocator_type>;
        return strong_rc_handle<handle_type, t_info_type>(
            t_info_type::template make_handle<handle_type>(std::forward<Args>(args)...)
        );
    }

    export template<typename T, typename allocator_type = std::allocator<T>, typename t_info_type = ref_count_info_type,
        typename... Args>
    auto make_rc_handle(Args &&... args) {
        return make_rc_pointer<T, allocator_type, t_info_type>(std::forward<Args>(args)...);
    }

    // Unique handle functionality - inherits common interface and adds unique-specific features
    template<typename T>
    struct inject_unique_functionality : public common_handle_interface<T> {
        // Base case: no additional functionality
    };

    template<typename V, template<typename value_type, typename...> typename allocator_template, template<typename
        pointer_handle_type, typename...> typename handle_template, typename info_type>
    struct inject_unique_functionality<handle_template<pointer_handle<V, allocator_template<V>>, info_type>>
            : public common_handle_interface<handle_template<pointer_handle<V, allocator_template<V>>, info_type>> {
        using my_managed_type = handle_template<pointer_handle<V, allocator_template<V>>>;

        using my_unmanaged_type = pointer_handle<V, allocator_template<V>>;

        template<typename T>
        using rebind_managed_type = handle_template<pointer_handle<T, allocator_template<T>>>;

        template<typename T>
        using rebind_unmanaged_type = pointer_handle<T, allocator_template<T>>;

        template<convertible_from<V> T>
        rebind_managed_type<T> static_pointer_cast(this my_managed_type &&self) noexcept {
            if (!self.has_value()) {
                return rebind_managed_type<T>();
            }

            auto casted_handle = self.m_handle.template static_pointer_cast<T>();
            std::exchange(self.m_handle, {}); // Release ownership
            return rebind_managed_type<T>(std::move(casted_handle));
        }

        template<std::derived_from<V> T>
        rebind_managed_type<T> dynamic_pointer_cast(this my_managed_type &&self) noexcept {
            if (!self.has_value()) {
                return rebind_managed_type<T>();
            }

            auto casted_handle = self.m_handle.template dynamic_pointer_cast<T>();
            if (!casted_handle) {
                return rebind_managed_type<T>();
            }

            std::exchange(self.m_handle, {}); // Release ownership
            return rebind_managed_type<T>(std::move(casted_handle));
        }
    };

    // unique handle
    struct unique_handle_info {
        static void destroy_handle(auto handle) {
            handle.destroy();
        }

        static bool has_value(auto handle) {
            return handle.has_value();
        }

        // Generic make_handle - calls T::make with forwarded arguments
        template<typename T, typename... Args>
        static auto make_handle(Args &&... args) {
            return T::make(std::forward<Args>(args)...);
        }
    };

    export template<typename t_handle_type, typename t_info_type = unique_handle_info>
        requires handle_type_concept<t_handle_type>
    class unique_handle : public inject_unique_functionality<unique_handle<t_handle_type, t_info_type>> {
    public:
        using info_type = t_info_type;
        using handle_type = t_handle_type;

    private:
        handle_type m_handle{};

        template<typename T>
        friend struct inject_unique_functionality;

        template<typename T>
        friend struct common_handle_interface;

    public:
        unique_handle() noexcept = default;

        unique_handle(handle_type handle) noexcept
            : m_handle(handle) {}

        ~unique_handle() noexcept {
            if (info_type::has_value(m_handle)) {
                info_type::destroy_handle(std::exchange(m_handle, {}));
            }
        }

        unique_handle(unique_handle &) = delete;

        unique_handle(unique_handle &&other) noexcept
            : m_handle(std::exchange(other.m_handle, {})) {}

        unique_handle &operator=(unique_handle &) = delete;

        unique_handle &operator=(unique_handle &&other) noexcept {
            this->~unique_handle();
            new(this) unique_handle(std::move(other));
            return *this;
        }

        bool has_value() const noexcept {
            return info_type::has_value(m_handle);
        }

        void reset() noexcept {
            if (info_type::has_value(m_handle)) {
                info_type::destroy_handle(std::exchange(m_handle, {}));
            }
        }

        handle_type release() noexcept {
            return std::exchange(m_handle, {});
        }
    };

    export template<typename T, typename allocator_type = std::allocator<T>, typename t_info_type = unique_handle_info,
        typename... Args>
    auto make_unique_handle(Args &&... args) {
        using handle_type = pointer_handle<T, allocator_type>;
        return unique_handle<handle_type, t_info_type>(
            t_info_type::template make_handle<handle_type>(std::forward<Args>(args)...)
        );
    }

    // Array handle - fixed-size array with smart pointer semantics
    export template<typename T, typename allocator_type = std::allocator<T>>
    class array_handle {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;

    private:
        [[no_unique_address]] allocator_type m_allocator{};
        using allocator_traits = std::allocator_traits<allocator_type>;

        T *m_data{nullptr};
        size_type m_size{0};

    public:
        array_handle() noexcept = default;

        // Destructor is no-op - cleanup is done through destroy()
        ~array_handle() noexcept = default;

        // Shallow copy is intentional - ownership is managed by outer smart pointer
        array_handle(const array_handle &) noexcept = default;
        array_handle(array_handle &&) noexcept = default;
        array_handle &operator=(const array_handle &) noexcept = default;
        array_handle &operator=(array_handle &&) noexcept = default;

        void destroy() noexcept {
            // Exchange m_data with nullptr atomically to prevent double-free
            T *data = std::exchange(m_data, nullptr);
            size_type size = std::exchange(m_size, 0);

            if (data) {
                // Destroy elements in reverse order
                for (size_type i = size; i > 0; --i) {
                    T *elem = data + i - 1;
                    // Call destroy() method if the element has one
                    if constexpr (requires { elem->destroy(); }) {
                        elem->destroy();
                    }
                    // Then call destructor
                    allocator_traits::destroy(m_allocator, elem);
                }
                allocator_traits::deallocate(m_allocator, data, size);
            }
        }

        bool has_value() const noexcept {
            return m_data != nullptr;
        }

        // Element access
        reference operator[](size_type pos) noexcept {
            assert(pos < m_size);
            return m_data[pos];
        }

        const_reference operator[](size_type pos) const noexcept {
            assert(pos < m_size);
            return m_data[pos];
        }

        reference at(size_type pos) {
            if (pos >= m_size) {
                throw std::out_of_range("array_handle::at");
            }
            return m_data[pos];
        }

        const_reference at(size_type pos) const {
            if (pos >= m_size) {
                throw std::out_of_range("array_handle::at");
            }
            return m_data[pos];
        }

        reference front() noexcept {
            assert(m_size > 0);
            return m_data[0];
        }

        const_reference front() const noexcept {
            assert(m_size > 0);
            return m_data[0];
        }

        reference back() noexcept {
            assert(m_size > 0);
            return m_data[m_size - 1];
        }

        const_reference back() const noexcept {
            assert(m_size > 0);
            return m_data[m_size - 1];
        }

        pointer data() noexcept {
            return m_data;
        }

        const_pointer data() const noexcept {
            return m_data;
        }

        // Iterators
        iterator begin() noexcept {
            return m_data;
        }

        const_iterator begin() const noexcept {
            return m_data;
        }

        const_iterator cbegin() const noexcept {
            return m_data;
        }

        iterator end() noexcept {
            return m_data + m_size;
        }

        const_iterator end() const noexcept {
            return m_data + m_size;
        }

        const_iterator cend() const noexcept {
            return m_data + m_size;
        }

        // Capacity
        bool empty() const noexcept {
            return m_size == 0;
        }

        size_type size() const noexcept {
            return m_size;
        }

        // Static factory method
        template<typename... Args>
        static array_handle make(size_type count, Args &&... args) {
            allocator_type allocator{};
            T *data = allocator_traits::allocate(allocator, count);

            // Construct elements
            size_type i = 0;
            try {
                for (; i < count; ++i) {
                    allocator_traits::construct(allocator, data + i, std::forward<Args>(args)...);
                }
            } catch (...) {
                // Cleanup on exception
                for (size_type j = 0; j < i; ++j) {
                    allocator_traits::destroy(allocator, data + j);
                }
                allocator_traits::deallocate(allocator, data, count);
                throw;
            }

            return array_handle(data, count);
        }

        // Factory method with initializer list
        template<typename U>
        static array_handle make_from_list(std::initializer_list<U> init) {
            allocator_type allocator{};
            size_type count = init.size();
            T *data = allocator_traits::allocate(allocator, count);

            size_type i = 0;
            try {
                for (const auto &val : init) {
                    allocator_traits::construct(allocator, data + i, val);
                    ++i;
                }
            } catch (...) {
                for (size_type j = 0; j < i; ++j) {
                    allocator_traits::destroy(allocator, data + j);
                }
                allocator_traits::deallocate(allocator, data, count);
                throw;
            }

            return array_handle(data, count);
        }

        // Factory method from sized range
        template<std::ranges::sized_range Range>
            requires std::convertible_to<std::ranges::range_value_t<Range>, T>
        static array_handle make_from_range(Range &&range) {
            allocator_type allocator{};
            size_type count = std::ranges::size(range);
            T *data = allocator_traits::allocate(allocator, count);

            size_type i = 0;
            try {
                for (auto &&val : range) {
                    allocator_traits::construct(allocator, data + i, std::forward<decltype(val)>(val));
                    ++i;
                }
            } catch (...) {
                for (size_type j = 0; j < i; ++j) {
                    allocator_traits::destroy(allocator, data + j);
                }
                allocator_traits::deallocate(allocator, data, count);
                throw;
            }

            return array_handle(data, count);
        }

    private:
        explicit array_handle(T *data, size_type size) noexcept
            : m_data(data), m_size(size) {}

        friend struct atomic_ref_count_info_type;
        friend struct ref_count_info_type;
        friend struct unique_handle_info;
    };

    // Type aliases for array handles
    // unique and arc use sync PMR (global static), rc uses unsync PMR (thread-local)
    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>>
    using unique_array = unique_handle<array_handle<T, allocator_type>>;

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>>
    using arc_array = strong_arc_handle<array_handle<T, allocator_type>>;

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, false>>
    using rc_array = strong_rc_handle<array_handle<T, allocator_type>>;

    // Factory functions for array handles
    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>, typename... Args>
    auto make_unique_array(std::size_t count, Args &&... args) {
        using handle_type = array_handle<T, allocator_type>;
        return unique_array<T, ElementCount, allocator_type>(
            handle_type::make(count, std::forward<Args>(args)...)
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>>
    auto make_unique_array(std::initializer_list<T> init) {
        using handle_type = array_handle<T, allocator_type>;
        return unique_array<T, ElementCount, allocator_type>(
            handle_type::make_from_list(init)
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>, std::ranges::sized_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
    auto make_unique_array(Range &&range) {
        using handle_type = array_handle<T, allocator_type>;
        return unique_array<T, ElementCount, allocator_type>(
            handle_type::make_from_range(std::forward<Range>(range))
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>, typename... Args>
    auto make_arc_array(std::size_t count, Args &&... args) {
        using handle_type = array_handle<T, allocator_type>;
        return arc_array<T, ElementCount, allocator_type>(
            handle_type::make(count, std::forward<Args>(args)...)
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>>
    auto make_arc_array(std::initializer_list<T> init) {
        using handle_type = array_handle<T, allocator_type>;
        return arc_array<T, ElementCount, allocator_type>(
            handle_type::make_from_list(init)
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, true>, std::ranges::sized_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
    auto make_arc_array(Range &&range) {
        using handle_type = array_handle<T, allocator_type>;
        return arc_array<T, ElementCount, allocator_type>(
            handle_type::make_from_range(std::forward<Range>(range))
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, false>, typename... Args>
    auto make_rc_array(std::size_t count, Args &&... args) {
        using handle_type = array_handle<T, allocator_type>;
        return rc_array<T, ElementCount, allocator_type>(
            handle_type::make(count, std::forward<Args>(args)...)
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, false>>
    auto make_rc_array(std::initializer_list<T> init) {
        using handle_type = array_handle<T, allocator_type>;
        return rc_array<T, ElementCount, allocator_type>(
            handle_type::make_from_list(init)
        );
    }

    export template<typename T, std::size_t ElementCount = 64, typename allocator_type = array_pmr_allocator<T, ElementCount, false>, std::ranges::sized_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
    auto make_rc_array(Range &&range) {
        using handle_type = array_handle<T, allocator_type>;
        return rc_array<T, ElementCount, allocator_type>(
            handle_type::make_from_range(std::forward<Range>(range))
        );
    }

    // ============================================================================
    // Function Handle - Type-erased callable with smart pointer semantics
    // ============================================================================

    // Base class for type-erased callable (move-only)
    template<typename R, typename... Args>
    class callable_base {
    public:
        virtual ~callable_base() = default;
        virtual R invoke(Args... args) = 0;
        virtual void deallocate_self() noexcept = 0;  // Virtual deallocation

        callable_base(const callable_base&) = delete;
        callable_base& operator=(const callable_base&) = delete;
        callable_base(callable_base&&) noexcept = default;
        callable_base& operator=(callable_base&&) noexcept = default;

    protected:
        callable_base() = default;
    };

    // Concrete callable implementation (move-only)
    template<typename F, typename R, typename AllocatorType, typename... Args>
    class callable_impl : public callable_base<R, Args...> {
    private:
        F m_func;

    public:
        explicit callable_impl(F func) : m_func(std::move(func)) {}

        R invoke(Args... args) override {
            return std::invoke(m_func, std::forward<Args>(args)...);
        }

        void deallocate_self() noexcept override {
            using allocator_type = typename std::allocator_traits<AllocatorType>::template rebind_alloc<callable_impl>;
            allocator_type alloc{};
            using alloc_traits = std::allocator_traits<allocator_type>;

            alloc_traits::destroy(alloc, this);
            alloc_traits::deallocate(alloc, this, 1);
        }
    };

    // shared_function_callable_handle - for arc/rc functions, uses PMR allocator
    export template<typename R, typename AllocatorType, typename... Args>
    class shared_function_callable_handle {
    public:
        using callable_type = callable_base<R, Args...>;
        using signature = R(Args...);
        using allocator_type = AllocatorType;

    private:
        [[no_unique_address]] allocator_type m_allocator{};
        using allocator_traits = std::allocator_traits<allocator_type>;

        callable_type* m_callable{nullptr};

    public:
        shared_function_callable_handle() noexcept = default;
        ~shared_function_callable_handle() noexcept = default;

        // Shallow copy - ownership is managed by outer smart pointer
        shared_function_callable_handle(const shared_function_callable_handle&) noexcept = default;
        shared_function_callable_handle(shared_function_callable_handle&&) noexcept = default;
        shared_function_callable_handle& operator=(const shared_function_callable_handle&) noexcept = default;
        shared_function_callable_handle& operator=(shared_function_callable_handle&&) noexcept = default;

        void destroy() noexcept {
            if (m_callable) {
                m_callable->deallocate_self();
                m_callable = nullptr;
            }
        }

        bool has_value() const noexcept {
            return m_callable != nullptr;
        }

        R invoke(Args... args) const {
            if (!m_callable) {
                throw std::bad_function_call();
            }
            return m_callable->invoke(std::forward<Args>(args)...);
        }

        R operator()(Args... args) const {
            return invoke(std::forward<Args>(args)...);
        }

        template<typename F>
            requires std::invocable<F, Args...> && std::convertible_to<std::invoke_result_t<F, Args...>, R>
        static shared_function_callable_handle make(F func) {
            shared_function_callable_handle handle;
            using callable_impl_type = callable_impl<F, R, allocator_type, Args...>;
            using impl_allocator = typename std::allocator_traits<allocator_type>::template rebind_alloc<callable_impl_type>;
            using impl_alloc_traits = std::allocator_traits<impl_allocator>;

            impl_allocator alloc{};
            callable_impl_type* ptr = impl_alloc_traits::allocate(alloc, 1);
            try {
                impl_alloc_traits::construct(alloc, ptr, std::move(func));
            } catch (...) {
                impl_alloc_traits::deallocate(alloc, ptr, 1);
                throw;
            }

            handle.m_callable = ptr;
            return handle;
        }

    private:
        friend struct atomic_ref_count_info_type;
        friend struct ref_count_info_type;
    };

    // unique_function_callable_handle - for unique_function, uses small_inplace_allocator with SBO
    export template<typename R, typename AllocatorType, typename... Args>
    class unique_function_callable_handle {
    public:
        using callable_type = callable_base<R, Args...>;
        using signature = R(Args...);
        using allocator_type = AllocatorType;

    private:
        [[no_unique_address]] allocator_type m_allocator{};
        using allocator_traits = std::allocator_traits<allocator_type>;

        callable_type* m_callable{nullptr};

    public:
        unique_function_callable_handle() noexcept = default;
        ~unique_function_callable_handle() noexcept = default;

        // Shallow copy - ownership is managed by outer smart pointer
        unique_function_callable_handle(const unique_function_callable_handle&) noexcept = default;
        unique_function_callable_handle(unique_function_callable_handle&&) noexcept = default;
        unique_function_callable_handle& operator=(const unique_function_callable_handle&) noexcept = default;
        unique_function_callable_handle& operator=(unique_function_callable_handle&&) noexcept = default;

        void destroy() noexcept {
            if (m_callable) {
                m_callable->deallocate_self();
                m_callable = nullptr;
            }
        }

        bool has_value() const noexcept {
            return m_callable != nullptr;
        }

        R invoke(Args... args) const {
            if (!m_callable) {
                throw std::bad_function_call();
            }
            return m_callable->invoke(std::forward<Args>(args)...);
        }

        R operator()(Args... args) const {
            return invoke(std::forward<Args>(args)...);
        }

        template<typename F>
            requires std::invocable<F, Args...> && std::convertible_to<std::invoke_result_t<F, Args...>, R>
        static unique_function_callable_handle make(F func) {
            unique_function_callable_handle handle;
            using callable_impl_type = callable_impl<F, R, allocator_type, Args...>;
            using impl_allocator = typename std::allocator_traits<allocator_type>::template rebind_alloc<callable_impl_type>;
            using impl_alloc_traits = std::allocator_traits<impl_allocator>;

            impl_allocator alloc{};
            callable_impl_type* ptr = impl_alloc_traits::allocate(alloc, 1);
            try {
                // Construct callable_impl in-place
                impl_alloc_traits::construct(alloc, ptr, std::move(func));
            } catch (...) {
                impl_alloc_traits::deallocate(alloc, ptr, 1);
                throw;
            }

            handle.m_callable = ptr;
            return handle;
        }

    private:
        friend struct unique_handle_info;
    };

    // Specialize unique_permissive for unique_function_callable_handle
    template<typename R, typename AllocatorType, typename... Args>
    struct unique_permissive<unique_function_callable_handle<R, AllocatorType, Args...>> : std::true_type {};

    // CRTP specialization for shared function handles (arc/rc) - inject operator()
    template<typename R, typename AllocType, typename... Args, template<typename, typename...> typename handle_template, typename info_type>
    struct common_handle_interface<handle_template<shared_function_callable_handle<R, AllocType, Args...>, info_type>> {
        using derived_type = handle_template<shared_function_callable_handle<R, AllocType, Args...>, info_type>;
        using handle_type = shared_function_callable_handle<R, AllocType, Args...>;

        R operator()(this const derived_type& self, Args... args) {
            // Access the m_handle member and call its operator()
            if constexpr (requires { self.m_handle; }) {
                return self.m_handle(std::forward<Args>(args)...);
            } else {
                throw std::bad_function_call();
            }
        }

        explicit operator bool(this const derived_type& self) noexcept {
            return self.has_value();
        }
    };

    // CRTP specialization for unique function handles - inject operator()
    template<typename R, typename AllocType, typename... Args, template<typename, typename...> typename handle_template, typename info_type>
    struct common_handle_interface<handle_template<unique_function_callable_handle<R, AllocType, Args...>, info_type>> {
        using derived_type = handle_template<unique_function_callable_handle<R, AllocType, Args...>, info_type>;
        using handle_type = unique_function_callable_handle<R, AllocType, Args...>;

        R operator()(this const derived_type& self, Args... args) {
            // Access the m_handle member and call its operator()
            if constexpr (requires { self.m_handle; }) {
                return self.m_handle(std::forward<Args>(args)...);
            } else {
                throw std::bad_function_call();
            }
        }

        explicit operator bool(this const derived_type& self) noexcept {
            return self.has_value();
        }
    };

    // ============================================================================
    // Type Aliases - Unified naming convention
    // ============================================================================

    // Pointer handles
    export template<typename T, typename allocator_type = std::allocator<T>>
    using unique_pointer = unique_handle<pointer_handle<T, allocator_type>>;

    export template<typename T, typename allocator_type = std::allocator<T>>
    using arc_pointer = strong_arc_handle<pointer_handle<T, allocator_type>>;

    export template<typename T, typename allocator_type = std::allocator<T>>
    using rc_pointer = strong_rc_handle<pointer_handle<T, allocator_type>>;

    // Default allocator for unique_function with customizable inline size
    template<typename Signature, std::size_t InlineSize = 64>
    struct default_unique_function_allocator;

    template<typename R, typename... Args, std::size_t InlineSize>
    struct default_unique_function_allocator<R(Args...), InlineSize> {
        using type = unique_function_pmr_allocator<callable_base<R, Args...>, InlineSize, true>;
    };

    export template<typename Signature, typename Allocator = typename default_unique_function_allocator<Signature>::type>
    struct unique_function_traits;

    export template<typename R, typename... Args, typename Allocator>
    struct unique_function_traits<R(Args...), Allocator> {
        using allocator_type = Allocator;
        using handle_type = unique_function_callable_handle<R, allocator_type, Args...>;
        using type = unique_handle<handle_type, unique_handle_info>;
    };

    export template<typename Signature, typename Allocator = typename default_unique_function_allocator<Signature>::type>
    using unique_function = typename unique_function_traits<Signature, Allocator>::type;

    // Default allocator for shared functions (not used currently, but for consistency)
    template<typename Signature>
    struct default_shared_function_allocator;

    template<typename R, typename... Args>
    struct default_shared_function_allocator<R(Args...)> {
        using type = std::allocator<callable_base<R, Args...>>;
    };

    // arc_function_traits - atomic reference counting function with sync PMR
    export template<typename Signature, std::size_t BufferSize = 64>
    struct arc_function_traits;

    export template<typename R, typename... Args, std::size_t BufferSize>
    struct arc_function_traits<R(Args...), BufferSize> {
        using allocator_type = unique_function_pmr_allocator<callable_base<R, Args...>, BufferSize, true>;
        using handle_type = shared_function_callable_handle<R, allocator_type, Args...>;
        using type = strong_arc_handle<handle_type>;
    };

    export template<typename Signature, std::size_t BufferSize = 64>
    using arc_function = typename arc_function_traits<Signature, BufferSize>::type;

    // rc_function_traits - non-atomic reference counting function with unsync PMR (thread-local)
    export template<typename Signature, std::size_t BufferSize = 64>
    struct rc_function_traits;

    export template<typename R, typename... Args, std::size_t BufferSize>
    struct rc_function_traits<R(Args...), BufferSize> {
        using allocator_type = unique_function_pmr_allocator<callable_base<R, Args...>, BufferSize, false>;
        using handle_type = shared_function_callable_handle<R, allocator_type, Args...>;
        using type = strong_rc_handle<handle_type>;
    };

    export template<typename Signature, std::size_t BufferSize = 64>
    using rc_function = typename rc_function_traits<Signature, BufferSize>::type;

    // Factory functions for function handles with full customization support

    // arc_function factory
    export template<typename Signature, std::size_t BufferSize = 64, typename F>
    auto make_arc_function(F&& func) {
        using traits = arc_function_traits<Signature, BufferSize>;
        using handle_type = typename traits::handle_type;
        return arc_function<Signature, BufferSize>(handle_type::make(std::forward<F>(func)));
    }

    // rc_function factory
    export template<typename Signature, std::size_t BufferSize = 64, typename F>
    auto make_rc_function(F&& func) {
        using traits = rc_function_traits<Signature, BufferSize>;
        using handle_type = typename traits::handle_type;
        return rc_function<Signature, BufferSize>(handle_type::make(std::forward<F>(func)));
    }

    // unique_function factory - supports custom allocator
    export template<typename Signature, typename Allocator = typename default_unique_function_allocator<Signature>::type, typename F>
    auto make_unique_function(F&& func) {
        using traits = unique_function_traits<Signature, Allocator>;
        using handle_type = typename traits::handle_type;
        return unique_function<Signature, Allocator>(handle_type::make(std::forward<F>(func)));
    }
}


