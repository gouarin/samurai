// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <memory>
#include <type_traits>

#include <xtensor/xexpression.hpp>
#include <xtensor/xfixed.hpp>
#include <xtl/xtype_traits.hpp>

#include "../level_cell_array.hpp"
#include "../algorithm.hpp"
#include "../utils.hpp"

namespace samurai
{
    /**********************
     * node_op definition *
     **********************/

    /** @class node_op
     *  @brief Define the CRTP used by the algebra of sets.
     *
     *  A node_op is the final node of the graph defining the
     *  algebra of sets. It's a mesh or a transformation of a
     *  mesh.
     *
     *  @tparam D Concrete node
     */
    template<class D>
    class node_op {
      public:
        using derived_type = D;

        derived_type &derived_cast() & noexcept;
        const derived_type &derived_cast() const &noexcept;
        derived_type derived_cast() && noexcept;

        auto index(int i) const noexcept;
        auto size(std::size_t dim) const noexcept;
        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;
        auto offset(std::size_t dim, std::size_t index) const noexcept;
        auto interval(std::size_t dim, std::size_t index) const noexcept;

        template<class T>
        auto find(std::size_t dim, std::size_t start, std::size_t end,
                  T coord) const noexcept;

        template<class T>
        auto transform(std::size_t dim, T coord) const noexcept;

        auto offsets_size(std::size_t dim) const noexcept;
        auto data() const noexcept;

        template<class Mesh>
        void data(Mesh &mesh) noexcept;

        template<class T>
        auto create_interval(T start, T end) const noexcept;
        auto create_index_yz() const noexcept;

        std::size_t level() const noexcept;
        bool is_empty() const noexcept;

      protected:
        node_op(){};
        ~node_op() = default;

        node_op(const node_op &) = default;
        node_op &operator=(const node_op &) = default;

        node_op(node_op &&) = default;
        node_op &operator=(node_op &&) = default;
    };

    /**************************
     * node_op implementation *
     **************************/

    template<class D>
    inline auto node_op<D>::derived_cast() & noexcept -> derived_type &
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D>
    inline auto node_op<D>::derived_cast() const & noexcept -> const derived_type &
    {
        return *static_cast<const derived_type *>(this);
    }

    template<class D>
    inline auto node_op<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D>
    inline auto node_op<D>::index(int i) const noexcept
    {
        return this->derived_cast().m_data.index(i);
    }

    template<class D>
    inline auto node_op<D>::size(std::size_t dim) const noexcept
    {
        return this->derived_cast().m_data.size(dim);
    }

    template<class D>
    inline auto node_op<D>::start(std::size_t dim, std::size_t index) const noexcept
    {
        return this->derived_cast().m_data.start(dim, index);
    }

    template<class D>
    inline auto node_op<D>::end(std::size_t dim, std::size_t index) const noexcept
    {
        return this->derived_cast().m_data.end(dim, index);
    }

    template<class D>
    inline auto node_op<D>::offset(std::size_t dim, std::size_t index) const noexcept
    {
        return this->derived_cast().m_data.offset(dim, index);
    }

    template<class D>
    inline auto node_op<D>::interval(std::size_t dim, std::size_t index) const noexcept
    {
        return this->derived_cast().m_data.interval(dim, index);
    }

    template<class D>
    template<class T>
    inline auto node_op<D>::find(std::size_t dim, std::size_t start,
                                 std::size_t end, T coord) const noexcept
    {
        return this->derived_cast().m_data.find(dim, start, end, coord);
    }

    template<class D>
    template<class T>
    inline auto node_op<D>::transform(std::size_t dim, T coord) const noexcept
    {
        return this->derived_cast().m_data.transform(dim, coord);
    }

    template<class D>
    inline auto node_op<D>::offsets_size(std::size_t dim) const noexcept
    {
        return this->derived_cast().m_data.offsets_size(dim);
    }

    template<class D>
    inline auto node_op<D>::data() const noexcept
    {
        return this->derived_cast().m_data.data();
    }

    template<class D>
    template<class Mesh>
    inline void node_op<D>::data(Mesh &mesh) noexcept
    {
        return this->derived_cast().m_data.data(mesh);
    }

    template<class D>
    inline std::size_t node_op<D>::level() const noexcept
    {
        return this->derived_cast().m_data.level();
    }

    template<class D>
    inline bool node_op<D>::is_empty() const noexcept
    {
        return this->derived_cast().m_data.is_empty();
    }

    template<class D>
    template<class T>
    inline auto node_op<D>::create_interval(T start, T end) const noexcept
    {
        return this->derived_cast().m_data.create_interval(start, end);
    }

    template<class D>
    inline auto node_op<D>::create_index_yz() const noexcept
    {
        return this->derived_cast().create_index_yz();
    }

    template<class E>
    using is_node_op = xt::is_crtp_base_of<node_op, E>;

    /************************
     * mesh_node definition *
     ************************/

    /** @class mesh_node
     *  @brief Define a mesh node in the algebra of sets.
     *
     *  @tparam Mesh The type of the mesh used
     */
    template<class Mesh>
    struct mesh_node : public node_op<mesh_node<Mesh>>
    {
        using mesh_type = Mesh;
        static constexpr std::size_t dim = mesh_type::dim;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename mesh_type::coord_index_t;

        mesh_node(const Mesh &v);

        // mesh_node() : m_data{nullptr}
        // {}

        mesh_node(const mesh_node &) = default;
        mesh_node &operator=(const mesh_node &) = default;

        mesh_node(mesh_node &&) = default;
        mesh_node &operator=(mesh_node &&) = default;

        auto index(int i) const noexcept;
        auto size(std::size_t dim) const noexcept;
        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;
        auto offset(std::size_t dim, std::size_t off_ind) const noexcept;
        auto offsets_size(std::size_t dim) const noexcept;
        auto interval(std::size_t dim, std::size_t index) const noexcept;
        auto find(std::size_t dim, std::size_t start, std::size_t end, coord_index_t coord) const noexcept;
        auto transform(std::size_t dim, coord_index_t coord) const noexcept;
        const Mesh &data() const noexcept;
        void data(const Mesh &mesh) noexcept;
        std::size_t level() const noexcept;
        bool is_empty() const noexcept;

        auto create_interval(coord_index_t start, coord_index_t end) const
            noexcept;
        auto create_index_yz() const noexcept;

      private:
        // std::shared_ptr<Mesh> m_data;
        // const Mesh* m_data;
        const Mesh& m_data;

        friend class node_op<mesh_node<Mesh>>;
    };

    /****************************
     * mesh_node implementation *
     ****************************/

    template<class Mesh>
    inline mesh_node<Mesh>::mesh_node(const Mesh &v)
        // : m_data{std::make_shared<Mesh>(v)}
        // : m_data{&v}
        : m_data(v)
    {}

    template<class Mesh>
    inline auto mesh_node<Mesh>::index(int i) const noexcept
    {
        return i;
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::size(std::size_t dim) const noexcept
    {
        // return (*m_data)[dim].size();
        return m_data[dim].size();
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::start(std::size_t dim, std::size_t index) const
        noexcept
    {
        if (m_data.empty())
        {
            return std::numeric_limits<coord_index_t>::max();
        }
        return m_data[dim][index].start;
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::end(std::size_t dim, std::size_t index) const
        noexcept
    {
        if (m_data.empty())
        {
            return std::numeric_limits<coord_index_t>::max();
        }
        return m_data[dim][index].end;
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::offset(std::size_t dim,
                                        std::size_t off_ind) const noexcept
    {
        return m_data.offsets(dim)[off_ind];
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::offsets_size(std::size_t dim) const noexcept
    {
        return m_data.offsets(dim).size();
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::interval(std::size_t dim,
                                          std::size_t index) const noexcept
    {
        return m_data[dim][index];
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::find(std::size_t dim, std::size_t start,
                                      std::size_t end,
                                      coord_index_t coord) const noexcept
    {
        return find_on_dim(m_data, dim, start, end, coord);
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::transform(std::size_t /*dim*/, coord_index_t coord) const noexcept
    {
        return coord;
    }

    template<class Mesh>
    inline const Mesh &mesh_node<Mesh>::data() const noexcept
    {
        // return *(m_data.get());
        return m_data;
    }

    template<class Mesh>
    inline void mesh_node<Mesh>::data(const Mesh &mesh) noexcept
    {
        m_data = mesh;
    }

    template<class Mesh>
    inline std::size_t mesh_node<Mesh>::level() const noexcept
    {
        return m_data.level();
    }

    template<class Mesh>
    inline bool mesh_node<Mesh>::is_empty() const noexcept
    {
        return m_data.empty();
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::create_interval(coord_index_t start,
                                                 coord_index_t end) const
        noexcept
    {
        return interval_t{start, end};
    }

    template<class Mesh>
    inline auto mesh_node<Mesh>::create_index_yz() const noexcept
    {
        return xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>>{};
    }

    /***********************************
     * translate_op definition *
     ***********************************/

    template<class T>
    struct translate_op : public node_op<translate_op<T>>
    {
        using mesh_type = typename T::mesh_type;
        static constexpr std::size_t dim = mesh_type::dim;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename mesh_type::coord_index_t;
        using stencil_t =
            typename xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>;

        translate_op(T &&v, stencil_t &&stencil);
        translate_op(const T &v, const stencil_t &stencil);

        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;

        auto transform(std::size_t dim, coord_index_t coord) const noexcept;

        auto create_interval(coord_index_t start, coord_index_t end) const
            noexcept;
        auto create_index_yz() const noexcept;

      private:
        T m_data;
        stencil_t m_stencil;

        friend class node_op<translate_op<T>>;
    };

    /*******************************
     * translate_op implementation *
     *******************************/

    template<class T>
    inline translate_op<T>::translate_op(T &&v, stencil_t &&stencil)
        : m_data{std::forward<T>(v)},
          m_stencil{std::forward<stencil_t>(stencil)}
    {}

    template<class T>
    inline translate_op<T>::translate_op(const T &v, const stencil_t &stencil)
        : m_data{v}, m_stencil{stencil}
    {}

    template<class T>
    inline auto translate_op<T>::start(std::size_t dim, std::size_t index) const noexcept
    {
        return m_data.start(dim, index) + m_stencil[dim];
    }

    template<class T>
    inline auto translate_op<T>::end(std::size_t dim,
                                     std::size_t index) const noexcept
    {
        return m_data.end(dim, index) + m_stencil[dim];
    }

    template<class T>
    inline auto
    translate_op<T>::transform(std::size_t dim, coord_index_t coord) const noexcept
    {
        return coord - m_stencil[dim];
    }

    template<class T>
    inline auto
    translate_op<T>::create_interval(coord_index_t start,
                                             coord_index_t end) const noexcept
    {
        return interval_t{start, end};
    }

    template<class T>
    inline auto translate_op<T>::create_index_yz() const noexcept
    {
        return xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>>{};
    }

    /*****************************
     * contraction_op definition *
     *****************************/

    template<class T>
    struct contraction_op : public node_op<contraction_op<T>>
    {
        using mesh_type = typename T::mesh_type;
        static constexpr std::size_t dim = mesh_type::dim;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename mesh_type::coord_index_t;

        contraction_op(const T& v, std::size_t size = 1);
        contraction_op(const T& v, const std::array<std::size_t, dim>& contraction);

        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;

        auto create_interval(coord_index_t start, coord_index_t end) const noexcept;
        auto create_index_yz() const noexcept;

      private:
        T m_data;
        std::array<std::size_t, dim> m_contraction;

        friend class node_op<contraction_op<T>>;
    };

    /*********************************
     * contraction_op implementation *
     *********************************/

    template<class T>
    inline contraction_op<T>::contraction_op(const T& v, std::size_t size)
    : m_data(v)
    {
        m_contraction.fill(size);
    }

    template<class T>
    inline contraction_op<T>::contraction_op(const T& v, const std::array<std::size_t, dim>& contraction)
    : m_data(v)
    , m_contraction(contraction)
    {}

    template<class T>
    inline auto contraction_op<T>::start(std::size_t dim,
                                         std::size_t index) const noexcept
    {
        return m_data.start(dim, index) + m_contraction[dim];
    }

    template<class T>
    inline auto contraction_op<T>::end(std::size_t dim, std::size_t index) const
        noexcept
    {
        return m_data.end(dim, index) - m_contraction[dim];
    }

    template<class T>
    inline auto contraction_op<T>::create_interval(coord_index_t start,
                                                   coord_index_t end) const
        noexcept
    {
        return interval_t{start, end};
    }

    template<class T>
    inline auto contraction_op<T>::create_index_yz() const noexcept
    {
        return xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>>{};
    }

    /*****************************
     * expand_op definition *
     *****************************/

    template<class T>
    struct expand_op : public node_op<expand_op<T>>
    {
        using mesh_type = typename T::mesh_type;
        static constexpr std::size_t dim = mesh_type::dim;
        using interval_t = typename mesh_type::interval_t;
        using coord_index_t = typename mesh_type::coord_index_t;

        expand_op(T &&v);
        expand_op(const T &v);

        auto start(std::size_t dim, std::size_t index) const noexcept;
        auto end(std::size_t dim, std::size_t index) const noexcept;

        auto create_interval(coord_index_t start, coord_index_t end) const
            noexcept;
        auto create_index_yz() const noexcept;

      private:
        T m_data;

        friend class node_op<expand_op<T>>;
    };

    /****************************
     * expand_op implementation *
     ****************************/

    template<class T>
    inline expand_op<T>::expand_op(T &&v) : m_data{std::forward<T>(v)}
    {}

    template<class T>
    inline expand_op<T>::expand_op(const T &v) : m_data{v}
    {}

    template<class T>
    inline auto expand_op<T>::start(std::size_t dim, std::size_t index) const
        noexcept
    {
        return m_data.start(dim, index) - 1;
    }

    template<class T>
    inline auto expand_op<T>::end(std::size_t dim, std::size_t index) const
        noexcept
    {
        return m_data.end(dim, index) + 1;
    }

    template<class T>
    inline auto expand_op<T>::create_interval(coord_index_t start,
                                              coord_index_t end) const noexcept
    {
        return interval_t{start, end};
    }

    template<class T>
    inline auto expand_op<T>::create_index_yz() const noexcept
    {
        return xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>>{};
    }

    namespace detail
    {
        template<class T>
        struct get_arg_node_impl
        {
            template<class R>
            decltype(auto) operator()(R &&r)
            {
                return std::forward<R>(r);
            }
        };

        template<std::size_t Dim, class TInterval>
        struct get_arg_node_impl<LevelCellArray<Dim, TInterval>>
        {
            using mesh_t = LevelCellArray<Dim, TInterval>;

            decltype(auto) operator()(const LevelCellArray<Dim, TInterval> &r)
            {
                return mesh_node<mesh_t>(r);
            }
        };
    }

    template<class T>
    decltype(auto) get_arg_node(T &&t)
    {
        detail::get_arg_node_impl<std::decay_t<T>> inv;
        return inv(std::forward<T>(t));
    }

    template<class T1, class T2>
    inline auto translate(T1 &&t, T2 &&stencil)
    {
        auto arg = get_arg_node(std::forward<T1>(t));
        using arg_t = decltype(arg);
        return translate_op<arg_t>{std::forward<arg_t>(arg),
                                   std::forward<T2>(stencil)};
    }

    template<class T>
    inline auto contraction(T &&t, std::size_t size = 1)
    {
        auto arg = get_arg_node(std::forward<T>(t));
        using arg_t = decltype(arg);
        return contraction_op<arg_t>{std::forward<arg_t>(arg), size};
    }

    template<class T, std::size_t dim>
    inline auto contraction(T &&t, const std::array<std::size_t, dim>& c)
    {
        auto arg = get_arg_node(std::forward<T>(t));
        using arg_t = decltype(arg);
        return contraction_op<arg_t>{std::forward<arg_t>(arg), c};
    }

    template<class T>
    inline auto expand(T &&t)
    {
        auto arg = get_arg_node(std::forward<T>(t));
        using arg_t = decltype(arg);
        return expand_op<arg_t>{std::forward<arg_t>(arg)};
    }
} // namespace samurai