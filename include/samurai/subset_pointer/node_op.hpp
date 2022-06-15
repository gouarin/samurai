// Copyright 2022 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>
#include <memory>

#include <xtensor/xfixed.hpp>

#include "../samurai_config.hpp"
#include "../level_cell_array.hpp"
#include "../algorithm.hpp"
#include "../utils.hpp"

namespace samurai
{

    /***************************
     * node_op_impl definition *
     ***************************/

    template<std::size_t Dim, class TInterval = default_config::interval_t>
    class node_op_impl
    {
    public:
        static constexpr std::size_t dim = Dim;
        using interval_t = TInterval;
        using lca_t   = LevelCellArray<Dim,interval_t>;
        using value_t = typename interval_t::value_t;
        using index_t = typename interval_t::index_t;

        node_op_impl(const lca_t& lca);

        virtual ~node_op_impl() = default;

        virtual node_op_impl* clone() const = 0;

        std::size_t size(std::size_t dir) const noexcept;
        std::size_t offset(std::size_t dir, std::size_t index) const noexcept;
        std::size_t offsets_size(std::size_t dir) const noexcept;

        interval_t interval(std::size_t dir, std::size_t index) const noexcept;

        std::size_t find(std::size_t dir, std::size_t start, std::size_t end,
                         value_t coord) const noexcept;

        std::size_t level() const noexcept;
        bool is_empty() const noexcept;

        auto data();

        interval_t create_interval(value_t start, value_t end) const noexcept;
        xt::xtensor_fixed<value_t, xt::xshape<dim-1>> create_index_yz() const noexcept;

        virtual value_t start(std::size_t dir, std::size_t index) const noexcept;
        virtual value_t end(std::size_t dir, std::size_t index) const noexcept;
        virtual value_t transform(std::size_t dir, value_t coord) const noexcept;

    protected:

        value_t start_impl(std::size_t dir, std::size_t index) const noexcept;
        value_t end_impl(std::size_t dir, std::size_t index) const noexcept;

    private:
        const lca_t& m_lca;
    };

    template<std::size_t Dim, class TInterval>
    node_op_impl<Dim, TInterval>::node_op_impl(const lca_t& lca)
    : m_lca(lca)
    {
    }

    template<std::size_t Dim, class TInterval>
    std::size_t node_op_impl<Dim, TInterval>::size(std::size_t dir) const noexcept
    {
        m_lca[dir].size();
    }

    template<std::size_t Dim, class TInterval>
    std::size_t node_op_impl<Dim, TInterval>::offset(std::size_t dir, std::size_t index) const noexcept
    {
        return m_lca.offsets(dir)[index];
    }

    template<std::size_t Dim, class TInterval>
    std::size_t node_op_impl<Dim, TInterval>::offsets_size(std::size_t dir) const noexcept
    {
        return m_lca.offsets(dir).size();
    }

    template<std::size_t Dim, class TInterval>
    TInterval node_op_impl<Dim, TInterval>::interval(std::size_t dir, std::size_t index) const noexcept
    {
        return m_lca[dir][index];
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::start(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return start_impl(dir, index);
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::end(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return end_impl(dir, index);
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::transform(std::size_t, value_t index) const noexcept -> value_t
    {
        return index;
    }

    template<std::size_t Dim, class TInterval>
    std::size_t node_op_impl<Dim, TInterval>::find(std::size_t dir, std::size_t start, std::size_t end,
                        value_t coord) const noexcept
    {
        return find_on_dim(m_lca, dir, start, end, coord);
    }

    template<std::size_t Dim, class TInterval>
    std::size_t node_op_impl<Dim, TInterval>::level() const noexcept
    {
        return m_lca.level();
    }

    template<std::size_t Dim, class TInterval>
    bool node_op_impl<Dim, TInterval>::is_empty() const noexcept
    {
        return m_lca.empty();
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::data()
    {
        return m_lca;
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::create_interval(value_t start, value_t end) const noexcept -> interval_t
    {
        return interval_t{start, end};
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::create_index_yz() const noexcept -> xt::xtensor_fixed<value_t, xt::xshape<dim-1>>
    {
        return xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>{};
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::start_impl(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        if (m_lca.empty())
        {
            return std::numeric_limits<value_t>::max();
        }
        return m_lca[dir][index].start;
    }

    template<std::size_t Dim, class TInterval>
    auto node_op_impl<Dim, TInterval>::end_impl(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        if (m_lca.empty())
        {
            return std::numeric_limits<value_t>::max();
        }
        return m_lca[dir][index].end;
    }

    template<std::size_t Dim, class TInterval = default_config::interval_t>
    class mesh_node: public node_op_impl<Dim, TInterval>
    {
    public:
        static constexpr std::size_t dim = Dim;
        using base_t = node_op_impl<Dim, TInterval>;
        using interval_t = typename base_t::interval_t;
        using lca_t   = typename base_t::lca_t;
        using value_t = typename base_t::value_t;
        using index_t = typename base_t::index_t;

        mesh_node(const lca_t& lca)
        : base_t(lca)
        {}

        mesh_node(const mesh_node&) = default;
        mesh_node& operator=(const mesh_node&) = default;

        mesh_node(mesh_node&&) = default;
        mesh_node& operator=(mesh_node&&) = default;

        mesh_node* clone() const override
        {
            return new mesh_node(*this);
        }
    };

/*
    template<std::size_t Dim, class TInterval = default_config::interval_t>
    class mesh_node: public node_op_impl<Dim, TInterval>
    {
    public:
        static constexpr std::size_t dim = Dim;
        using base_t = node_op_impl<Dim, TInterval>;
        using interval_t = typename base_t::interval_t;
        using lca_t   = typename base_t::lca_t;
        using value_t = typename base_t::value_t;
        using index_t = typename base_t::index_t;

        mesh_node(const lca_t& lca);

        value_t start(std::size_t dir, std::size_t index) const noexcept override;
        value_t end(std::size_t dir, std::size_t index) const noexcept override;
        value_t transform(std::size_t dir, value_t coord) const noexcept override;
    };

    template<std::size_t Dim, class TInterval>
    auto mesh_node<Dim, TInterval>::start(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return start_impl(dir, index);
    }

    template<std::size_t Dim, class TInterval>
    auto mesh_node<Dim, TInterval>::end(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return end_impl(dir, index);
    }

    template<std::size_t Dim, class TInterval>
    auto mesh_node<Dim, TInterval>::transform(std::size_t dir, value_t coord) const noexcept -> value_t
    {
        return coord;
    }
*/

    /***************************
     * translate_op definition *
     ***************************/

    template<std::size_t Dim, class TInterval = default_config::interval_t>
    class translate_op: public node_op_impl<Dim, TInterval>
    {
    public:
        static constexpr std::size_t dim = Dim;
        using base_t = node_op_impl<Dim, TInterval>;
        using interval_t = typename base_t::interval_t;
        using lca_t   = typename base_t::lca_t;
        using value_t = typename base_t::value_t;
        using index_t = typename base_t::index_t;
        using stencil_t = typename xt::xtensor_fixed<value_t, xt::xshape<dim>>;

        translate_op(const translate_op&) = default;
        translate_op& operator=(const translate_op&) = default;

        translate_op(translate_op&&) = default;
        translate_op& operator=(translate_op&&) = default;

        translate_op(const lca_t& lca, const stencil_t& stencil);

        translate_op* clone() const override
        {
            return new translate_op(*this);
        }

        value_t start(std::size_t dir, std::size_t index) const noexcept override;
        value_t end(std::size_t dir, std::size_t index) const noexcept override;
        value_t transform(std::size_t dir, value_t coord) const noexcept override;

    private:
        stencil_t m_stencil;
    };

    /*******************************
     * translate_op implementation *
     *******************************/

    template<std::size_t Dim, class TInterval>
    translate_op<Dim, TInterval>::translate_op(const lca_t& lca, const stencil_t& stencil)
    : base_t(lca), m_stencil(stencil)
    {}

    template<std::size_t Dim, class TInterval>
    auto translate_op<Dim, TInterval>::start(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return this->start_impl(dir, index) + m_stencil[dir];
    }

    template<std::size_t Dim, class TInterval>
    auto translate_op<Dim, TInterval>::end(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return this->end_impl(dir, index) + m_stencil[dir];
    }

    template<std::size_t Dim, class TInterval>
    auto translate_op<Dim, TInterval>::transform(std::size_t dir, value_t coord) const noexcept -> value_t
    {
        return coord - m_stencil[dir];
    }

    /*****************************
     * contraction_op definition *
     *****************************/

    template<std::size_t Dim, class TInterval = default_config::interval_t>
    class contraction_op: public node_op_impl<Dim, TInterval>
    {
    public:
        static constexpr std::size_t dim = Dim;
        using base_t = node_op_impl<Dim, TInterval>;
        using interval_t = typename base_t::interval_t;
        using lca_t   = typename base_t::lca_t;
        using value_t = typename base_t::value_t;
        using index_t = typename base_t::index_t;

        contraction_op(const contraction_op&) = default;
        contraction_op& operator=(const contraction_op&) = default;

        contraction_op(contraction_op&&) = default;
        contraction_op& operator=(contraction_op&&) = default;

        contraction_op(const lca_t& lca, std::size_t size = 1);
        contraction_op(const lca_t& lca, const std::array<std::size_t, dim>& contraction);

        contraction_op* clone() const override
        {
            return new contraction_op(*this);
        }

        value_t start(std::size_t dir, std::size_t index) const noexcept override;
        value_t end(std::size_t dir, std::size_t index) const noexcept override;

    private:
        std::array<std::size_t, dim> m_contraction;
    };

    /*********************************
     * contraction_op implementation *
     *********************************/

    template<std::size_t Dim, class TInterval>
    contraction_op<Dim,TInterval>::contraction_op(const lca_t& lca, std::size_t size)
    : base_t(lca)
    {
        m_contraction.fill(size);
    }

    template<std::size_t Dim, class TInterval>
    contraction_op<Dim,TInterval>::contraction_op(const lca_t& lca, const std::array<std::size_t, dim>& contraction)
    : base_t(lca)
    , m_contraction(contraction)
    {}

    template<std::size_t Dim, class TInterval>
    auto contraction_op<Dim,TInterval>::start(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return this->start_impl(dir, index) + static_cast<value_t>(m_contraction[dir]);
    }

    template<std::size_t Dim, class TInterval>
    auto contraction_op<Dim,TInterval>::end(std::size_t dir, std::size_t index) const noexcept -> value_t
    {
        return this->end_impl(dir, index) - static_cast<value_t>(m_contraction[dir]);
    }

    template<std::size_t Dim, class TInterval = default_config::interval_t>
    class node_op
    {
    public:
        using ptr_t = std::unique_ptr<node_op_impl<Dim, TInterval>>;

        node_op(const node_op_impl<Dim, TInterval>& node)
        : p_node(node.clone())
        {
        }

        node_op(const node_op& node)
        : p_node(node.p_node->clone())
        {}

        node_op& operator=(const node_op& node)
        {
            ptr_t tmp = node.p_impl->clone();
            std::swap(tmp, p_node);
            return *this;
        }

        int start(std::size_t dir, std::size_t index) const noexcept
        {
            return p_node->start(dir, index);
        }
    private:
        ptr_t p_node;
    };


}