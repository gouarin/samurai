#pragma once

#include <array>
#include <memory>

#include <spdlog/spdlog.h>

#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

#include "cell.hpp"
#include "bc.hpp"
#include "field_expression.hpp"
#include "mr/mesh.hpp"
#include "mr/mesh_type.hpp"

namespace mure
{
    template<class TInterval>
    class update_boundary_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(update_boundary_op)

        template<class T, class BC, class stencil_t>
        inline void operator()(Dim<1>, T &field, const BC &bc, const stencil_t& stencil) const
        {
            switch(bc.first)
            {
                case mure::BCType::dirichlet:
                {
                    field(level, i) = bc.second;
                    break;
                }
                case mure::BCType::neumann:
                {
                    int n = stencil[0];
                    field(level, i) = n*dx()*bc.second + field(level, i - stencil[0]);
                    break;
                }
            }
        }

        template<class T, class BC, class stencil_t>
        inline void operator()(Dim<2>, T &field, const BC &bc, const stencil_t& stencil) const
        {
            switch(bc.first)
            {
                case mure::BCType::dirichlet:
                {
                    field(level, i, j) = bc.second;
                    break;
                }
                case mure::BCType::neumann:
                {
                    int n = stencil[0] + stencil[1];
                    field(level, i, j) = n*dx()*bc.second + field(level, i - stencil[0], j - stencil[1]);
                    break;
                }
            }
        }
    };

    template<class T, class BC, class stencil_t>
    inline auto update_boundary(T &&field, BC &&bc, stencil_t &&stencil)
    {
        return make_field_operator_function<update_boundary_op>(
            std::forward<T>(field), std::forward<BC>(bc), std::forward<stencil_t>(stencil));
    }

    template<std::size_t size>
    struct is_scalar_field: std::false_type
    {};

    template<>
    struct is_scalar_field<1>: std::true_type
    {};

    template<class MRConfig, class value_t=double, std::size_t size_=1>
    class Field : public field_expression<Field<MRConfig, value_t, size_>>
    {
      public:

        using Config = MRConfig;
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        using coord_index_t = typename MRConfig::coord_index_t;
        using index_t = typename MRConfig::index_t;
        using interval_t = typename MRConfig::interval_t;

        using value_type = value_t;
        static constexpr auto size = size_;
        using data_type = typename std::conditional<is_scalar_field<size>::value,
                                                    xt::xtensor<value_type, 1>,
                                                    xt::xtensor<value_type, 2>>::type;

        template<std::size_t n>
        void init_data(std::integral_constant<std::size_t, n>)
        {
            m_data.resize({m_mesh->nb_total_cells(), size});
            m_data.fill(0);
        }

        void init_data(std::integral_constant<std::size_t, 1>)
        {
            m_data.resize({m_mesh->nb_total_cells()});
            m_data.fill(0);
        }

        inline Field(std::string name, Mesh<MRConfig> &mesh, const BC<dim> &bc)
            : m_name(name), m_mesh(&mesh), m_bc(bc)
        {
            init_data(std::integral_constant<std::size_t, size>{});
        }

        inline Field(std::string name, Mesh<MRConfig> &mesh)
            : m_name(name), m_mesh(&mesh)
        {
            init_data(std::integral_constant<std::size_t, size>{});
        }

        template<class E>
        inline Field &operator=(const field_expression<E> &e)
        {
            // mesh->for_each_cell(
            //     [&](auto &cell) { (*this)[cell] = e.derived_cast()(cell); });

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                auto subset = intersection((*m_mesh)[MeshType::cells][level],
                                           (*m_mesh)[MeshType::cells][level]);

                subset.apply_op(level, apply_expr(*this, e));
            }
            return *this;
        }

        inline auto operator[](const std::size_t index) const
        {
            return xt::view(m_data, index);
        }

        inline auto operator[](const std::size_t index)
        {
            return xt::view(m_data, index);
        }

        inline auto operator[](const Cell<coord_index_t, dim> &cell) const
        {
            return xt::view(m_data, cell.index);
        }

        inline auto operator[](const Cell<coord_index_t, dim> &cell)
        {
            return xt::view(m_data, cell.index);
        }

        template<class... T>
        inline auto operator()(const std::size_t level, const interval_t &interval, const T... index)
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("WRITE FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step));
        }

        template<class... T>
        inline auto operator()(const std::size_t level, const interval_t &interval,
                        const T... index) const
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("READ FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step));
        }

        template<class... T>
        inline auto operator()(const std::size_t item, const std::size_t level, const interval_t &interval, const T... index)
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("WRITE FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step), item);
        }

        template<class... T>
        inline auto operator()(const std::size_t item, const std::size_t level, const interval_t &interval,
                        const T... index) const
        {
            auto interval_tmp = m_mesh->get_interval(level, interval, index...);
            if ((interval_tmp.end - interval_tmp.step < interval.end - interval.step) or
                (interval_tmp.start > interval.start))
            {
                spdlog::critical("READ FIELD ERROR on level {} for "
                                 "interval_tmp {} and interval {}",
                                 level, interval_tmp, interval);
            }
            return xt::view(m_data,
                            xt::range(interval_tmp.index + interval.start,
                                      interval_tmp.index + interval.end,
                                      interval.step), item);
        }

        inline auto data(MeshType mesh_type) const
        {
            std::array<std::size_t, 2> shape = {m_mesh->nb_cells(mesh_type), size};
            xt::xtensor<double, 2> output(shape);
            std::size_t index = 0;
            m_mesh->for_each_cell([&](auto cell)
                                  {
                                      auto view = xt::view(output, index++);
                                      view = xt::view(m_data, cell.index);
                                  },
                                  mesh_type);
            return output;
        }

        inline auto data_on_level(std::size_t level, MeshType mesh_type) const
        {
            std::array<std::size_t, 2> shape = {m_mesh->nb_cells(level, mesh_type), size};
            xt::xtensor<double, 2> output(shape);
            std::size_t index = 0;
            m_mesh->for_each_cell(level,
                                  [&](auto cell)
                                  {
                                      xt::view(output, index++) = xt::view(m_data, cell.index);
                                  },
                                  mesh_type);
            return output;
        }

        inline auto const array() const
        {
            return m_data;
        }

        inline auto &array()
        {
            return m_data;
        }

        inline std::size_t nb_cells(MeshType mesh_type) const
        {
            return m_mesh->nb_cells(mesh_type);
        }

        inline std::size_t nb_cells(std::size_t level, MeshType mesh_type) const
        {
            return m_mesh->nb_cells(level, mesh_type);
        }

        inline auto name() const
        {
            return m_name;
        }

        inline auto bc() const
        {
            return m_bc;
        }


        inline auto bc()
        {
            return m_bc;
        }

        inline auto mesh()
        {
            return *m_mesh;
        }

        inline auto mesh_ptr()
        {
            return m_mesh;
        }

        inline void update_bc()
        {
            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
            std::size_t index_bc = 0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                for (std::size_t d1 = 0; d1 < dim; ++d1)
                    stencil[d1] = 0;
                for (int s = -1; s <= 1; ++s)
                {
                    if (s != 0)
                    {
                        stencil[d] = s;
                        for (std::size_t level = 0; level <= m_mesh->max_level(); ++level)
                        {
                            double dx = 1./(1<<level);
                            if (!(*m_mesh)[mure::MeshType::all_cells][level].empty())
                            {
                                // TODO: use union mesh instead
                                auto subset = intersection(difference(translate(m_mesh->initial_mesh(), stencil),
                                                                      m_mesh->initial_mesh()),
                                                           (*m_mesh)[mure::MeshType::all_cells][level])
                                            .on(level);

                                subset.apply_op(level, update_boundary(*this, m_bc.type[index_bc], stencil));
                            }
                        }
                        index_bc++;
                    }
                }
            }
        }

        inline void to_stream(std::ostream &os) const
        {
            os << "Field " << m_name << "\n";
            m_mesh->for_each_cell(
                [&](auto &cell) {
                    os << cell.level << "[" << cell.center()
                       << "]:" << xt::view(m_data, cell.index) << "\n";
                },
                // MeshType::all_cells);
                MeshType::cells);
        }

      private:
        std::string m_name;
        Mesh<MRConfig> *m_mesh;
        BC<dim> m_bc;
        data_type m_data;
    };

    template<class MRConfig, class T>
    inline std::ostream &operator<<(std::ostream &out, const Field<MRConfig, T> &field)
    {
        field.to_stream(out);
        return out;
    }
}