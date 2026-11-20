// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Prototype: finite volumes on a mapped multiblock disc.
//
// The disc of radius 3 is cut into five blocks, as in figure 5 of
// P. Colella, High-order finite-volume methods on locally-structured grids
// (2016): a Cartesian central block [-1, 1]^2 and four outer blocks aligned
// with the circle. Unfolded, the blocks form a cross, which is the
// parametric domain meshed by samurai: [-3, 3]^2 without its four corner
// squares.
//
// The cross has to be closed: at each reentrant corner, the two edges of the
// cross that meet there are the same physical line, a seam between two outer
// blocks. Crossing a seam rotates the indices by a quarter turn, which
// samurai does not know. This prototype handles it by hand:
//   - the flux through a seam face reads the cell of the neighbouring block
//     directly, without any ghost. Both blocks then compute the same flux
//     and the scheme stays conservative;
//   - across a seam, the two blocks may be at different levels: the flux is
//     then computed half face by half face;
//   - the adaptation grades the mesh across the seams, so that these levels
//     differ by one at most;
//   - the prediction of the multiresolution reads across the seams.
// Three blocks meet at each corner of the central block: these four
// vertices of valence 3 are the only singular points of the grid.
//
// As in mapped_advection_annulus.cpp, the unknown is q = |K| u / h^2, the
// geometry of a cell is the polygon traced by its boundary on the grid of
// the finest level, and the adaptation uses the well-balanced prediction
// q_hat = P(q) + u_parent dJ.

#include <samurai/domain_builder.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
namespace fs = std::filesystem;

static constexpr std::size_t dim = 2;
using point_t                    = std::array<double, 2>;

// Velocity field, chosen once in main: the constant velocity (1, 1), along a
// pair of seams, or the rigid rotation, tangent to the circle.
static bool rotation_flow = false;

point_t velocity(const point_t& x)
{
    if (rotation_flow)
    {
        return {-x[1], x[0]};
    }
    return {1., 1.};
}

double cross(const point_t& a, const point_t& b)
{
    return a[0] * b[1] - a[1] * b[0];
}

// Flux of the velocity through the segment [a, b], with the normal on the
// right of a -> b. The velocity is linear, so the midpoint rule is exact.
double segment_flux(const point_t& a, const point_t& b)
{
    const auto v = velocity({0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])});
    return v[0] * (b[1] - a[1]) - v[1] * (b[0] - a[0]);
}

double polygon_area(const std::vector<point_t>& poly)
{
    double area = 0;
    for (std::size_t k = 0; k < poly.size(); ++k)
    {
        area += cross(poly[k], poly[(k + 1) % poly.size()]);
    }
    return 0.5 * area;
}

struct Index
{
    long i;
    long j;
};

// The cross made of five blocks of n x n cells, inside the square of
// 3 n x 3 n cells: everything but the four corner squares.
struct Cross
{
    long n;

    bool contains(long i, long j) const
    {
        const bool in_square = i >= 0 && i < 3 * n && j >= 0 && j < 3 * n;
        const bool in_corner = (i < n || i >= 2 * n) && (j < n || j >= 2 * n);
        return in_square && !in_corner;
    }

    // A cell of the cross is on a seam when one of its faces is an edge of
    // a corner square.
    bool on_seam(long i, long j) const
    {
        auto in_corner_square = [this](long a, long b)
        {
            return a >= 0 && a < 3 * n && b >= 0 && b < 3 * n && !contains(a, b);
        };
        return in_corner_square(i - 1, j) || in_corner_square(i + 1, j) || in_corner_square(i, j - 1) || in_corner_square(i, j + 1);
    }

    // Cell of the cross that plays the role of 'target' for a cell 'center'
    // of the cross, 'target' being a neighbour of 'center' in index space:
    //   - 'target' itself if it is in the cross;
    //   - beyond the circle, its mirror image;
    //   - in a corner square, the cell of the neighbouring outer block,
    //     reached through the seam by a quarter turn around the reentrant
    //     corner. The turn depends on the block of 'center';
    //   - nothing when 'center' is in the central block: only three cells
    //     meet at its corner, there is no cell across it.
    std::optional<Index> resolve(Index center, Index target) const
    {
        auto mirror = [this](long k)
        {
            return k < 0 ? -1 - k : (k >= 3 * n ? 6 * n - 1 - k : k);
        };
        target = {mirror(target.i), mirror(target.j)};
        if (contains(target.i, target.j))
        {
            return target;
        }

        // Bring the corner square onto the upper right one by symmetry
        const bool flip_i = target.i < n;
        const bool flip_j = target.j < n;
        auto flip         = [this](long k, bool doit)
        {
            return doit ? 3 * n - 1 - k : k;
        };
        const Index c{flip(center.i, flip_i), flip(center.j, flip_j)};
        const Index t{flip(target.i, flip_i), flip(target.j, flip_j)};

        Index r{};
        if (c.i >= 2 * n && c.j < 2 * n)
        {
            // from the right block, through its upper edge, into the upper block
            r = {4 * n - 1 - t.j, t.i};
        }
        else if (c.j >= 2 * n && c.i < 2 * n)
        {
            // from the upper block, through its right edge, into the right block
            r = {t.j, 4 * n - 1 - t.i};
        }
        else
        {
            return std::nullopt;
        }
        return Index{flip(r.i, flip_i), flip(r.j, flip_j)};
    }
};

// Square of cells [i0, i0 + n) x [j0, j0 + n) of the geometry grid
struct Box
{
    long i0;
    long j0;
    long n;
};

// Map of the cross onto the disc of radius 3, piecewise bilinear on the grid
// of the geometry level and stored by its vertices. Vertex (i, j) is the
// parametric point (-3 + i h, -3 + j h). The central block is not deformed.
// In an outer block, the vertices of a grid line go straight from the edge
// of the central block to the circle, where they are equally spaced.
class MultiblockMap
{
  public:

    MultiblockMap(std::size_t level, long block_size)
        : m_level(level)
        , m_cross{block_size}
        , m_h(2. / static_cast<double>(block_size))
        , m_x(static_cast<std::size_t>((3 * block_size + 1) * (3 * block_size + 1)), point_t{std::nan(""), std::nan("")})
    {
        const long n = 3 * block_size;
        for (long j = 0; j <= n; ++j)
        {
            for (long i = 0; i <= n; ++i)
            {
                // a vertex is in the domain when one of the cells around it is
                if (m_cross.contains(i, j) || m_cross.contains(i - 1, j) || m_cross.contains(i, j - 1) || m_cross.contains(i - 1, j - 1))
                {
                    vertex(i, j) = analytic(coordinate(i), coordinate(j));
                }
            }
        }
    }

    // The two edges of a seam get the same points, bit for bit: at |t| = 1,
    // cos and sin of pi / 4 are replaced by the same constant.
    static point_t analytic(double xi, double eta)
    {
        const double m = std::max(std::abs(xi), std::abs(eta));
        if (m <= 1)
        {
            return {xi, eta};
        }
        const double s         = 0.5 * (m - 1);
        const bool horizontal  = std::abs(xi) >= std::abs(eta);
        const double sign      = std::copysign(1., horizontal ? xi : eta);
        const double t         = horizontal ? eta : xi;
        const bool on_diagonal = std::abs(t) == 1.;
        const double c         = on_diagonal ? std::sqrt(0.5) : std::cos(0.25 * std::numbers::pi * t);
        const double sn        = on_diagonal ? std::copysign(std::sqrt(0.5), t) : std::sin(0.25 * std::numbers::pi * t);
        // along the radial line: from the edge of the central block to the circle
        const double radial     = (1 - s) + 3 * s * c;
        const double tangential = (1 - s) * t + 3 * s * sn;
        return horizontal ? point_t{sign * radial, tangential} : point_t{tangential, sign * radial};
    }

    std::size_t level() const
    {
        return m_level;
    }

    const Cross& cross_shape() const
    {
        return m_cross;
    }

    double step() const
    {
        return m_h;
    }

    // Number of cells of the geometry grid along a side of the square
    long size() const
    {
        return 3 * m_cross.n;
    }

    double coordinate(long i) const
    {
        return -3. + static_cast<double>(i) * m_h;
    }

    const point_t& vertex(long i, long j) const
    {
        return m_x[static_cast<std::size_t>(j * (size() + 1) + i)];
    }

    point_t& vertex(long i, long j)
    {
        return m_x[static_cast<std::size_t>(j * (size() + 1) + i)];
    }

    point_t operator()(double s, double t) const
    {
        const double fs = (s + 3.) / m_h;
        const double ft = (t + 3.) / m_h;
        const long i    = std::clamp(static_cast<long>(std::floor(fs)), 0L, size() - 1);
        const long j    = std::clamp(static_cast<long>(std::floor(ft)), 0L, size() - 1);
        const double u  = fs - static_cast<double>(i);
        const double v  = ft - static_cast<double>(j);
        const auto& a   = vertex(i, j);
        const auto& b   = vertex(i + 1, j);
        const auto& c   = vertex(i + 1, j + 1);
        const auto& d   = vertex(i, j + 1);
        return {(1 - u) * (1 - v) * a[0] + u * (1 - v) * b[0] + u * v * c[0] + (1 - u) * v * d[0],
                (1 - u) * (1 - v) * a[1] + u * (1 - v) * b[1] + u * v * c[1] + (1 - u) * v * d[1]};
    }

    // Boundary of a box, counterclockwise
    std::vector<point_t> polygon(const Box& box) const
    {
        std::vector<point_t> poly;
        poly.reserve(static_cast<std::size_t>(4 * box.n));
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0 + k, box.j0));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0 + box.n, box.j0 + k));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0 + box.n - k, box.j0 + box.n));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0, box.j0 + box.n - k));
        }
        return poly;
    }

  private:

    std::size_t m_level;
    Cross m_cross;
    double m_h;
    std::vector<point_t> m_x;
};

// Integer indices of a samurai cell, and its box on the geometry grid
template <class Cell>
Index cell_index(const Cell& cell)
{
    return {static_cast<long>(cell.indices[0]), static_cast<long>(cell.indices[1])};
}

template <class Cell>
Box cell_box(const MultiblockMap& map, const Cell& cell)
{
    const long s = 1L << (map.level() - cell.level);
    return {cell.indices[0] * s, cell.indices[1] * s, s};
}

// Physical flux of the velocity through the face on the low side of the cell
// 'cell' of 'level' in direction d, oriented towards +d, summed over the
// segments of the geometry grid.
inline double face_flux(const MultiblockMap& map, std::size_t level, Index cell, std::size_t d)
{
    const long s = 1L << (map.level() - level);
    const Box box{cell.i * s, cell.j * s, s};
    double flux = 0;
    for (long k = 0; k < box.n; ++k)
    {
        // the normal on the right of a -> b points towards +d
        if (d == 0)
        {
            flux += segment_flux(map.vertex(box.i0, box.j0 + k), map.vertex(box.i0, box.j0 + k + 1));
        }
        else
        {
            flux += segment_flux(map.vertex(box.i0 + k + 1, box.j0), map.vertex(box.i0 + k, box.j0));
        }
    }
    return flux;
}

// Largest stable time step of the upwind scheme on the geometry grid: the
// outgoing flux of a cell during dt must not exceed its area.
double stable_time_step(const MultiblockMap& map)
{
    double dt = std::numeric_limits<double>::max();
    for (long j = 0; j < map.size(); ++j)
    {
        for (long i = 0; i < map.size(); ++i)
        {
            if (!map.cross_shape().contains(i, j))
            {
                continue;
            }
            const auto poly = map.polygon({i, j, 1});
            double out      = 0;
            for (std::size_t k = 0; k < 4; ++k)
            {
                out += std::max(0., segment_flux(poly[k], poly[(k + 1) % 4]));
            }
            dt = std::min(dt, polygon_area(poly) / out);
        }
    }
    return dt;
}

// Mean Jacobians |K| / h^2 of the cells of every level, computed once from
// the areas of the quads of the geometry grid. The areas are nested, so the
// Jacobian of a cell is the mean of the Jacobians of its children.
class JacobianPyramid
{
  public:

    explicit JacobianPyramid(const MultiblockMap& map)
        : m_level(map.level())
        , m_size(map.size())
        , m_levels(map.level() + 1)
    {
        const long n = m_size;
        auto& fine   = m_levels[m_level];
        fine.assign(static_cast<std::size_t>(n * n), 0.);
        for (long j = 0; j < n; ++j)
        {
            for (long i = 0; i < n; ++i)
            {
                if (map.cross_shape().contains(i, j))
                {
                    fine[static_cast<std::size_t>(j * n + i)] = polygon_area(map.polygon({i, j, 1})) / (map.step() * map.step());
                }
            }
        }
        for (std::size_t level = m_level; level-- > 0;)
        {
            const long nc      = n >> (m_level - level);
            const auto& finer  = m_levels[level + 1];
            auto& coarse       = m_levels[level];
            const auto at_fine = [&](long i, long j)
            {
                return finer[static_cast<std::size_t>(j * 2 * nc + i)];
            };
            coarse.resize(static_cast<std::size_t>(nc * nc));
            for (long j = 0; j < nc; ++j)
            {
                for (long i = 0; i < nc; ++i)
                {
                    coarse[static_cast<std::size_t>(j * nc + i)] = 0.25
                                                                 * (at_fine(2 * i, 2 * j) + at_fine(2 * i + 1, 2 * j)
                                                                    + at_fine(2 * i, 2 * j + 1) + at_fine(2 * i + 1, 2 * j + 1));
                }
            }
        }
    }

    // Shape of the cross at a level
    Cross cross_shape(std::size_t level) const
    {
        return {(m_size / 3) >> (m_level - level)};
    }

    // Mean Jacobian of a cell of the cross
    double operator()(std::size_t level, long i, long j) const
    {
        const long n = m_size >> (m_level - level);
        return m_levels[level][static_cast<std::size_t>(j * n + i)];
    }

    template <class Cell>
    double operator()(const Cell& cell) const
    {
        return (*this)(cell.level, cell.indices[0], cell.indices[1]);
    }

  private:

    std::size_t m_level;
    long m_size;
    std::vector<std::vector<double>> m_levels;
};

// Storage index of a cell, ghosts included, if the mesh has it
template <class Mesh>
std::optional<std::size_t> find_cell(const Mesh& mesh, std::size_t level, long i, long j)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    using value_t   = typename Mesh::interval_t::coord_index_t;
    const auto& lca = mesh[mesh_id_t::reference][level];
    const xt::xtensor_fixed<value_t, xt::xshape<dim>> coord{static_cast<value_t>(i), static_cast<value_t>(j)};
    const auto k = samurai::find(lca, coord);
    if (k < 0)
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(lca[0][static_cast<std::size_t>(k)].index + i);
}

// Storage index of a leaf cell, if the mesh has it
template <class Mesh>
std::optional<std::size_t> find_leaf(const Mesh& mesh, std::size_t level, long i, long j)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    using value_t   = typename Mesh::interval_t::coord_index_t;
    const auto& lca = mesh[mesh_id_t::cells][level];
    const xt::xtensor_fixed<value_t, xt::xshape<dim>> coord{static_cast<value_t>(i), static_cast<value_t>(j)};
    const auto k = samurai::find(lca, coord);
    if (k < 0)
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(lca[0][static_cast<std::size_t>(k)].index + i);
}

// The scheme takes u and returns d(q)/dt: the physical flux times the
// upwind value.
//
// A face with a cell outside the cross is on a seam or on the circle.
// samurai sees it as a boundary face and gives a ghost on the other side;
// the scheme ignores it and reads the leaf that really lies there
// (Cross::resolve): the cell of the neighbouring block across a seam, the
// cell itself at the circle. Across a seam, that leaf may be at the same
// level, one level coarser, or the face may be shared with two leaves one
// level finer. In the last case the flux is the sum over the two half
// faces, each with its own upwind value. Every half face then gets the same
// flux from both blocks, whatever their levels: the scheme is conservative
// across the seams without any ghost. MultiblockAdapt keeps the levels of
// the two sides within one of each other.
template <class Field>
auto make_mapped_upwind(const MultiblockMap& map, const Field& u)
{
    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, 2, Field, Field>;

    samurai::FluxDefinition<cfg> upwind;
    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            upwind[d].cons_flux_function =
                [&map, &u](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& u_stencil)
            {
                const std::size_t level = data.cells[1].level;
                const Index left        = cell_index(data.cells[0]);
                const Index right       = cell_index(data.cells[1]);
                const Cross shape{map.cross_shape().n >> (map.level() - level)};
                const bool left_in  = shape.contains(left.i, left.j);
                const bool right_in = shape.contains(right.i, right.j);

                // The scheme multiplies by h_face / h_cell^2: divide by h_face
                // to get d(q)/dt = -sum(F u) / h_cell^2.
                if (left_in && right_in)
                {
                    const double F = face_flux(map, level, right, d);
                    flux           = F * ((F >= 0) ? u_stencil[0] : u_stencil[1]) / data.cell_length;
                    return;
                }

                const double u_inside = u_stencil[left_in ? 0 : 1];
                const auto& mesh      = u.mesh();
                auto leaf_value       = [&](std::size_t l, const Index& cell) -> std::optional<double>
                {
                    if (const auto index = find_leaf(mesh, l, cell.i, cell.j))
                    {
                        return u[*index];
                    }
                    return std::nullopt;
                };
                // flux through the face of level l on the low side of l_right
                auto seam_flux = [&](std::size_t l, const Index& l_right, double u_outside)
                {
                    const double F         = face_flux(map, l, l_right, d);
                    const bool from_inside = (F >= 0) == left_in;
                    return F * (from_inside ? u_inside : u_outside);
                };

                const Index inside  = left_in ? left : right;
                const Index outside = left_in ? right : left;
                const auto source   = shape.resolve(inside, outside);
                if (!source)
                {
                    throw std::runtime_error("a face of the central block has no cell on one side");
                }
                auto other = leaf_value(level, *source);
                if (!other && level > 0)
                {
                    other = leaf_value(level - 1, {source->i >> 1, source->j >> 1});
                }
                if (other)
                {
                    flux = seam_flux(level, right, *other) / data.cell_length;
                    return;
                }

                // The other side is finer: one flux per half face
                if (level == map.level())
                {
                    throw std::runtime_error("no leaf across a seam at the finest level");
                }
                const Cross fine_shape{2 * shape.n};
                double total = 0;
                for (long k = 0; k < 2; ++k)
                {
                    const Index fine_right{2 * right.i + (d == 0 ? 0 : k), 2 * right.j + (d == 0 ? k : 0)};
                    const Index fine_left{fine_right.i - (d == 0 ? 1 : 0), fine_right.j - (d == 0 ? 0 : 1)};
                    const auto fine_source = fine_shape.resolve(left_in ? fine_left : fine_right, left_in ? fine_right : fine_left);
                    const auto fine_other  = leaf_value(level + 1, *fine_source);
                    if (!fine_other)
                    {
                        throw std::runtime_error("the levels of the two sides of a seam differ by more than one");
                    }
                    total += seam_flux(level + 1, fine_right, *fine_other);
                }
                flux = total / data.cell_length;
            };
        });
    return samurai::make_flux_based_scheme(upwind);
}

// Weight of the coarse cell s - 1 in the prediction of the child of the
// given parity, in one direction
double prediction_weight(std::size_t parity, std::size_t s)
{
    static const auto even = samurai::prediction_coefficients<1>(0, 0).c;
    static const auto odd  = samurai::prediction_coefficients<1>(1, 0).c;
    return parity == 0 ? even[s] : odd[s];
}

// q and mean Jacobian of the 3 x 3 cells around a cell of the cross
struct Stencil
{
    std::array<std::array<double, 3>, 3> q;
    std::array<std::array<double, 3>, 3> jacobian;
};

// A neighbour outside the cross is replaced by the cell that plays its role
// (Cross::resolve). When the mesh does not have it at this level, the
// block across the seam is coarser there: the neighbour takes the u of the
// parent of that cell. When there is no such cell at all, it takes the u of
// the central cell.
template <class Field>
Stencil gather(const Field& q, const JacobianPyramid& jacobian, std::size_t level, long i, long j)
{
    const auto shape  = jacobian.cross_shape(level);
    const auto center = find_cell(q.mesh(), level, i, j);
    Stencil stencil;
    for (std::size_t a = 0; a < 3; ++a)
    {
        for (std::size_t b = 0; b < 3; ++b)
        {
            stencil.q[a][b]        = q[*center];
            stencil.jacobian[a][b] = jacobian(level, i, j);
            const auto source      = shape.resolve({i, j}, {i + static_cast<long>(a) - 1, j + static_cast<long>(b) - 1});
            if (source)
            {
                if (const auto index = find_cell(q.mesh(), level, source->i, source->j))
                {
                    stencil.q[a][b]        = q[*index];
                    stencil.jacobian[a][b] = jacobian(level, source->i, source->j);
                }
                else if (level > 0)
                {
                    const Index parent{source->i >> 1, source->j >> 1};
                    if (const auto parent_index = find_cell(q.mesh(), level - 1, parent.i, parent.j))
                    {
                        stencil.jacobian[a][b] = jacobian(level, source->i, source->j);
                        stencil.q[a][b]        = stencil.jacobian[a][b] * q[*parent_index] / jacobian(level - 1, parent.i, parent.j);
                    }
                }
            }
        }
    }
    return stencil;
}

// Well-balanced prediction of q = |K| u / h^2 in the child
// (2 i + dx, 2 j + dy) of the cell (i, j) of 'level':
//     q_hat = P(q) + u_parent dJ,
// where P is the standard prediction and dJ = J_child - P(J) the geometric
// detail. A constant u is predicted exactly whatever the map. The mean of
// the children is the parent when the stencil is not altered by a seam.
double well_balanced_prediction(const Stencil& stencil, double jacobian_child, std::size_t dx, std::size_t dy)
{
    double predicted_q = 0;
    double predicted_j = 0;
    for (std::size_t a = 0; a < 3; ++a)
    {
        for (std::size_t b = 0; b < 3; ++b)
        {
            const double w = prediction_weight(dx, a) * prediction_weight(dy, b);
            predicted_q += w * stencil.q[a][b];
            predicted_j += w * stencil.jacobian[a][b];
        }
    }
    return predicted_q + stencil.q[1][1] / stencil.jacobian[1][1] * (jacobian_child - predicted_j);
}

// Well-balanced prediction of the four children of the cell (i, j) of
// 'level', with the correction that keeps their mean equal to the parent.
// The correction vanishes away from the seams, and for a constant u.
template <class Field>
std::array<std::array<double, 2>, 2> predict_children(const Field& q, const JacobianPyramid& jacobian, std::size_t level, long i, long j)
{
    const auto stencil = gather(q, jacobian, level, i, j);
    std::array<std::array<double, 2>, 2> children;
    double mean = 0;
    for (std::size_t dx = 0; dx < 2; ++dx)
    {
        for (std::size_t dy = 0; dy < 2; ++dy)
        {
            children[dx][dy] = well_balanced_prediction(stencil,
                                                        jacobian(level + 1, 2 * i + static_cast<long>(dx), 2 * j + static_cast<long>(dy)),
                                                        dx,
                                                        dy);
            mean += 0.25 * children[dx][dy];
        }
    }
    for (auto& column : children)
    {
        for (auto& child : column)
        {
            child += stencil.q[1][1] - mean;
        }
    }
    return children;
}

// Prediction function of the mesh adaptation: fills the children created by
// refinement.
inline auto well_balanced_prediction_fn(const JacobianPyramid& jacobian)
{
    return [&jacobian](auto& new_field, const auto& old_field)
    {
        return [&new_field, &old_field, &jacobian](std::size_t level, const auto& interval, const auto& index)
        {
            const long j = index[0];
            for (long i = interval.start; i < interval.end; ++i)
            {
                const auto children = predict_children(old_field, jacobian, level, i, j);
                for (std::size_t dx = 0; dx < 2; ++dx)
                {
                    for (std::size_t dy = 0; dy < 2; ++dy)
                    {
                        const auto child = find_cell(new_field.mesh(), level + 1, 2 * i + static_cast<long>(dx), 2 * j + static_cast<long>(dy));
                        new_field[*child] = children[dx][dy];
                    }
                }
            }
        };
    };
}

// Replaces the ghosts of the cross that update_ghost_mr filled with the
// standard prediction by their well-balanced prediction, level by level
// since a ghost may be predicted from another one.
template <class Field>
void predict_ghosts_well_balanced(Field& q, const JacobianPyramid& jacobian)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    auto& mesh      = q.mesh();
    for (std::size_t level = 1; level <= mesh.max_level(); ++level)
    {
        const auto shape = jacobian.cross_shape(level);
        auto pred_ghosts = samurai::difference(mesh[mesh_id_t::all_cells][level],
                                               samurai::union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
        auto set         = samurai::intersection(pred_ghosts, mesh.subdomain(level), mesh[mesh_id_t::all_cells][level - 1]).on(level);
        set(
            [&](const auto& interval, const auto& index)
            {
                const long j = index[0];
                for (long i = interval.start; i < interval.end; ++i)
                {
                    if (!shape.contains(i, j))
                    {
                        continue; // outside the cross: never read, see gather
                    }
                    const auto children              = predict_children(q, jacobian, level - 1, i >> 1, j >> 1);
                    q[*find_cell(mesh, level, i, j)] = children[static_cast<std::size_t>(i & 1)][static_cast<std::size_t>(j & 1)];
                }
            });
    }
}

// Multiresolution adaptation of q with the well-balanced prediction. It
// follows samurai::Adapt step by step. Three things differ: the predicted
// ghosts of q are well-balanced, the detail is d = (q - q_hat) / J, in the
// units of u, and the mesh is also graded across the seams.
template <class Field>
class MultiblockAdapt
{
  public:

    MultiblockAdapt(Field& q, const JacobianPyramid& jacobian)
        : m_q(q)
        , m_jacobian(jacobian)
        , m_detail(samurai::make_scalar_field<double>("detail", q.mesh()))
        , m_tag(samurai::make_scalar_field<std::uint8_t>("tag", q.mesh()))
    {
    }

    void operator()(samurai::mra_config& cfg)
    {
        auto& mesh = m_q.mesh();
        cfg.parse_args();
        if (cfg.relative_detail())
        {
            throw std::runtime_error("the well-balanced adaptation does not handle relative details");
        }
        for (std::size_t ite = 0; ite < mesh.max_level() - mesh.min_level(); ++ite)
        {
            m_detail.resize();
            m_detail.fill(0);
            m_tag.resize();
            m_tag.fill(0);
            if (harten(ite, cfg))
            {
                break;
            }
        }
    }

  private:

    using mesh_t    = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    bool harten(std::size_t ite, const samurai::mra_config& cfg)
    {
        auto& mesh                  = m_q.mesh();
        const std::size_t min_level = mesh.min_level();
        const std::size_t max_level = mesh.max_level();

        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   m_tag[cell] = static_cast<std::uint8_t>(samurai::CellFlag::keep);
                               });

        samurai::update_ghost_mr(m_q);
        predict_ghosts_well_balanced(m_q, m_jacobian);

        for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level - ite; ++level)
        {
            samurai::ghosts_below_cells(mesh, level)(
                [&](const auto& interval, const auto& index)
                {
                    const long j = index[0];
                    for (long i = interval.start; i < interval.end; ++i)
                    {
                        const auto stencil = gather(m_q, m_jacobian, level, i, j);
                        for (std::size_t dx = 0; dx < 2; ++dx)
                        {
                            for (std::size_t dy = 0; dy < 2; ++dy)
                            {
                                const long ic        = 2 * i + static_cast<long>(dx);
                                const long jc        = 2 * j + static_cast<long>(dy);
                                const auto child     = *find_cell(mesh, level + 1, ic, jc);
                                const double j_child = m_jacobian(level + 1, ic, jc);
                                m_detail[child]      = (m_q[child] - well_balanced_prediction(stencil, j_child, dx, dy)) / j_child;
                            }
                        }
                    }
                });
        }

        for (std::size_t level = std::max(min_level, static_cast<std::size_t>(1)); level <= max_level - ite; ++level)
        {
            const double eps_l = cfg.epsilon() / static_cast<double>(1 << (dim * (max_level - level)));
            auto subset        = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            subset.apply_op(samurai::mr_criteria(m_detail, m_tag, eps_l, cfg.regularity() + static_cast<double>(dim)));
            samurai::update_tag_subdomains(level, m_tag, true);
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            samurai::update_tag_periodic(level, m_tag);
            samurai::update_tag_subdomains(level, m_tag);
            keep_subset.apply_op(samurai::maximum(m_tag));
        }

        // samurai grades the mesh inside the cross. The two sides of a seam
        // are graded here: while a leaf has, across a seam, a leaf coarser
        // by more than one level, that leaf is refined or kept from
        // coarsening, and the mesh is graded again.
        auto new_ca = samurai::update_cell_array_from_tag(mesh[mesh_id_t::cells], m_tag);
        for (std::size_t pass = 0;; ++pass)
        {
            samurai::make_graduation(new_ca,
                                     mesh.domain_pyramid(),
                                     mesh.mpi_neighbourhood(),
                                     mesh.periodicity(),
                                     mesh.graduation_width(),
                                     mesh.max_stencil_radius());
            if (!grade_seams(new_ca))
            {
                break;
            }
            if (pass == 2 * max_level)
            {
                throw std::runtime_error("the graduation across the seams does not converge");
            }
            new_ca = samurai::update_cell_array_from_tag(mesh[mesh_id_t::cells], m_tag);
        }
        if (mesh[mesh_id_t::cells] == new_ca)
        {
            return true;
        }

        mesh_t new_mesh{new_ca, mesh};
        samurai::update_fields(well_balanced_prediction_fn(m_jacobian), new_mesh, m_q);
        mesh.swap(new_mesh);
        return false;
    }

    // Looks for leaves of 'ca' that are more than one level coarser than a
    // leaf across a seam, and changes the tags of the current mesh so that
    // they come out one level finer. Returns whether a tag changed.
    template <class CellArray>
    bool grade_seams(const CellArray& ca)
    {
        using value_t = typename mesh_t::interval_t::coord_index_t;
        auto& mesh    = m_q.mesh();
        auto has_leaf = [](const auto& lca, const Index& cell)
        {
            const xt::xtensor_fixed<value_t, xt::xshape<dim>> coord{static_cast<value_t>(cell.i), static_cast<value_t>(cell.j)};
            return samurai::find(lca, coord) >= 0;
        };
        // Tags a leaf of the current mesh. Returns whether it is a leaf.
        bool changed = false;
        auto set_tag = [&](std::size_t level, const Index& cell, samurai::CellFlag flag)
        {
            const auto index = find_leaf(mesh, level, cell.i, cell.j);
            if (!index)
            {
                return false;
            }
            const auto value = static_cast<std::uint8_t>(flag);
            changed          = changed || m_tag[*index] != value;
            m_tag[*index]    = value;
            return true;
        };

        for (std::size_t level = 2; level <= mesh.max_level(); ++level)
        {
            const auto shape = m_jacobian.cross_shape(level);
            samurai::for_each_cell(ca[level],
                                   [&](const auto& cell)
                                   {
                                       const Index center = cell_index(cell);
                                       if (!shape.on_seam(center.i, center.j))
                                       {
                                           return;
                                       }
                                       for (const Index& step : {
                                                Index{-1, 0 },
                                                Index{1,  0 },
                                                Index{0,  -1},
                                                Index{0,  1 }
                                       })
                                       {
                                           const Index target{center.i + step.i, center.j + step.j};
                                           const auto source = shape.resolve(center, target);
                                           if (shape.contains(target.i, target.j) || !source)
                                           {
                                               continue;
                                           }
                                           // a leaf across the seam, two levels coarser or more
                                           for (std::size_t coarse_level = level - 2;; --coarse_level)
                                           {
                                               const Index coarse{source->i >> (level - coarse_level), source->j >> (level - coarse_level)};
                                               if (has_leaf(ca[coarse_level], coarse))
                                               {
                                                   // it is a leaf of the current mesh: refine it; or it
                                                   // comes from the coarsening of four leaves: keep them
                                                   bool found = set_tag(coarse_level, coarse, samurai::CellFlag::refine);
                                                   for (long child = 0; child < 4; ++child)
                                                   {
                                                       found = set_tag(coarse_level + 1,
                                                                       {2 * coarse.i + (child & 1), 2 * coarse.j + (child >> 1)},
                                                                       samurai::CellFlag::keep)
                                                            || found;
                                                   }
                                                   if (!found)
                                                   {
                                                       throw std::runtime_error("cannot grade the mesh across a seam");
                                                   }
                                                   break;
                                               }
                                               if (coarse_level == 0)
                                               {
                                                   break;
                                               }
                                           }
                                       }
                                   });
        }
        return changed;
    }

    Field& m_q;
    const JacobianPyramid& m_jacobian;
    samurai::ScalarField<mesh_t, double> m_detail;
    samurai::ScalarField<mesh_t, std::uint8_t> m_tag;
};

struct Diagnostics
{
    double mass         = 0;
    double l1_error     = 0;
    double max_error    = 0;
    std::size_t n_cells = 0;
};

template <class Field, class Exact>
Diagnostics compute_diagnostics(const Field& q, const MultiblockMap& map, const JacobianPyramid& jacobian, Exact&& exact)
{
    Diagnostics diag;
    samurai::for_each_cell(q.mesh(),
                           [&](const auto& cell)
                           {
                               const double jac  = jacobian(cell);
                               const double area = jac * cell.length * cell.length;
                               const double u    = q[cell] / jac;
                               auto c            = cell.center();
                               const double err  = std::abs(u - exact(map(c(0), c(1))));
                               diag.mass += q[cell] * cell.length * cell.length;
                               diag.l1_error += err * area;
                               diag.max_error = std::max(diag.max_error, err);
                               ++diag.n_cells;
                           });
    return diag;
}

// Writes the physical mesh: each cell is the polygon traced by its boundary
// on the geometry grid. Cell data: the solution u, the error against the
// exact solution, the level and the mean Jacobian.
template <class Field, class Exact>
void save_physical(const fs::path& file, const Field& q, const MultiblockMap& map, const JacobianPyramid& jacobian, Exact&& exact)
{
    std::vector<float> coords;
    std::vector<std::int32_t> cell_ends;
    std::vector<float> u;
    std::vector<float> error;
    std::vector<float> jac;
    std::vector<std::int32_t> level;
    samurai::for_each_cell(q.mesh(),
                           [&](const auto& cell)
                           {
                               for (const auto& p : map.polygon(cell_box(map, cell)))
                               {
                                   coords.insert(coords.end(), {static_cast<float>(p[0]), static_cast<float>(p[1]), 0.f});
                               }
                               cell_ends.push_back(static_cast<std::int32_t>(coords.size() / 3));
                               auto c           = cell.center();
                               const double val = q[cell] / jacobian(cell);
                               u.push_back(static_cast<float>(val));
                               error.push_back(static_cast<float>(val - exact(map(c(0), c(1)))));
                               jac.push_back(static_cast<float>(jacobian(cell)));
                               level.push_back(static_cast<std::int32_t>(cell.level));
                           });

    // VTK XML unstructured grid with the arrays appended as raw binary
    std::vector<std::int32_t> connectivity(coords.size() / 3);
    std::iota(connectivity.begin(), connectivity.end(), 0);
    std::vector<std::uint8_t> types(u.size(), 7); // VTK_POLYGON

    std::string header;
    std::vector<std::pair<const char*, std::size_t>> blocks; // data, size in bytes
    std::size_t offset = 0;
    auto add_array     = [&](const std::string& attributes, const auto& values)
    {
        using value_t = typename std::decay_t<decltype(values)>::value_type;
        header += fmt::format("<DataArray {} format=\"appended\" offset=\"{}\"/>\n", attributes, offset);
        blocks.emplace_back(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(value_t));
        offset += sizeof(std::uint64_t) + values.size() * sizeof(value_t);
    };

    header += fmt::format("<Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n<Points>\n", coords.size() / 3, u.size());
    add_array("type=\"Float32\" NumberOfComponents=\"3\"", coords);
    header += "</Points>\n<Cells>\n";
    add_array("type=\"Int32\" Name=\"connectivity\"", connectivity);
    add_array("type=\"Int32\" Name=\"offsets\"", cell_ends);
    add_array("type=\"UInt8\" Name=\"types\"", types);
    header += "</Cells>\n<CellData Scalars=\"u\">\n";
    add_array("type=\"Float32\" Name=\"u\"", u);
    add_array("type=\"Float32\" Name=\"error\"", error);
    add_array("type=\"Float32\" Name=\"jacobian\"", jac);
    add_array("type=\"Int32\" Name=\"level\"", level);
    header += "</CellData>\n</Piece>\n";

    std::ofstream out(file, std::ios::binary);
    out << "<?xml version=\"1.0\"?>\n";
    out << fmt::format("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"{}\" header_type=\"UInt64\">\n",
                       std::endian::native == std::endian::little ? "LittleEndian" : "BigEndian");
    out << "<UnstructuredGrid>\n" << header << "</UnstructuredGrid>\n<AppendedData encoding=\"raw\">\n_";
    for (const auto& [data, size] : blocks)
    {
        const std::uint64_t nbytes = size;
        out.write(reinterpret_cast<const char*>(&nbytes), sizeof(nbytes));
        out.write(data, static_cast<std::streamsize>(size));
    }
    out << "\n</AppendedData>\n</VTKFile>\n";
}

void save_collection(const fs::path& file, const std::vector<std::pair<double, std::string>>& steps)
{
    std::ofstream out(file);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n";
    for (const auto& [time, name] : steps)
    {
        out << fmt::format("<DataSet timestep=\"{}\" file=\"{}\"/>\n", time, name);
    }
    out << "</Collection>\n</VTKFile>\n";
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Linear advection on a disc made of five mapped blocks", argc, argv);

    double Tf          = 0;
    double cfl         = 0.9;
    bool mr            = true;
    bool bump          = true;
    bool indicator     = false;
    bool rotation      = false;
    double base        = 0;
    double radius      = 2;
    std::size_t nfiles = 0;

    std::size_t min_level = 2;
    std::size_t max_level = 6;

    fs::path path        = fs::current_path();
    std::string filename = "mapped_advection_multiblock_disc";

    auto* Tf_option = app.add_option("--Tf", Tf, "Final time (the bump reaches the opposite point, or pi / 2 with --rotation)")
                          ->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--mr,!--no-mr", mr, "Adapt the mesh with multiresolution")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--rotation",
                 rotation,
                 "Rigid rotation of a bump starting at (2, 0), instead of the velocity (1, 1) carrying it along a diagonal")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--bump,!--constant", bump, "Initial condition: bump or constant state")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--indicator", indicator, "Indicator of a disc of radius 0.3 instead of the Gaussian bump")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--start-radius", radius, "Distance from the center of the disc to the initial bump")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--background", base, "Constant state added to the bump")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of saved time steps after the initial one (0: no output)")->capture_default_str()->group("Output");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    rotation_flow = rotation;
    if (Tf_option->count() == 0)
    {
        Tf = rotation ? std::numbers::pi / 2 : radius * std::numbers::sqrt2;
    }

    // The cross: five blocks of side 2, one cell each at level 0
    samurai::DomainBuilder<dim> domain({-3., -3.}, {3., 3.});
    domain.remove({-3., -3.}, {-1., -1.});
    domain.remove({1., -3.}, {3., -1.});
    domain.remove({-3., 1.}, {-1., 3.});
    domain.remove({1., 1.}, {3., 3.});

    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
    auto mesh   = samurai::mra::make_mesh(domain, config);

    // --min-level and --max-level override the values above
    min_level             = mesh.min_level();
    max_level             = mesh.max_level();
    const long block_size = std::lround(2. / mesh.cell_length(max_level));
    if (block_size != (1L << max_level) || mesh.origin_point()[0] != -3. || mesh.origin_point()[1] != -3.)
    {
        throw std::runtime_error("the mesh does not have one cell per block at level 0");
    }
    const MultiblockMap map(max_level, block_size);
    const JacobianPyramid jacobian(map);

    // The bump starts on a seam (diagonal flow) or on the horizontal axis
    // (rotation). Diagonal flow: it follows the seam, crosses the central
    // block through two of its corners and ends on the opposite seam.
    const point_t start = rotation ? point_t{radius, 0.} : point_t{-radius * std::numbers::sqrt2 / 2, -radius * std::numbers::sqrt2 / 2};
    auto u0             = [=](const point_t& x)
    {
        if (!bump)
        {
            return 1.;
        }
        const double dx = x[0] - start[0];
        const double dy = x[1] - start[1];
        if (indicator)
        {
            return base + (dx * dx + dy * dy < 0.09 ? 1. : 0.);
        }
        return base + std::exp(-(dx * dx + dy * dy) / 0.02);
    };
    auto exact = [&](double t)
    {
        return [&, t](const point_t& x)
        {
            if (!rotation)
            {
                return u0({x[0] - t, x[1] - t});
            }
            const double c = std::cos(t);
            const double s = std::sin(t);
            return u0({c * x[0] + s * x[1], -s * x[0] + c * x[1]});
        };
    };

    auto q    = samurai::make_scalar_field<double>("q", mesh);
    auto init = [&]()
    {
        q.resize();
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   auto c  = cell.center();
                                   q[cell] = jacobian(cell) * u0(map(c(0), c(1)));
                               });
    };
    init();
    samurai::make_bc<samurai::Neumann<1>>(q, 0.);

    auto u_phys = samurai::make_scalar_field<double>("u", mesh);
    samurai::make_bc<samurai::Neumann<1>>(u_phys, 0.);
    auto update_u = [&]()
    {
        u_phys.resize();
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   u_phys[cell] = q[cell] / jacobian(cell);
                               });
    };

    auto qnp1   = samurai::make_scalar_field<double>("qnp1", mesh);
    auto scheme = make_mapped_upwind(map, u_phys);

    MultiblockAdapt adapt(q, jacobian);
    auto mra_config = samurai::mra_config().epsilon(1e-3); // --mr-eps overrides it
    if (mr)
    {
        // Adapt from the exact initial data rather than from projected values
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            adapt(mra_config);
            init();
        }
    }

    const double dt = cfl * stable_time_step(map);

    auto diag0 = compute_diagnostics(q, map, jacobian, exact(0));
    fmt::print("initial: cells = {}, mass = {:.15e}, max |u - u0| = {:.3e}\n", diag0.n_cells, diag0.mass, diag0.max_error);

    // Physical cells in <filename>_<k>.vtu, gathered with their time in
    // <filename>.pvd; parametric mesh in <filename>_parametric_<k>.h5.
    std::vector<std::pair<double, std::string>> saved_steps;
    auto save = [&](double time)
    {
        const auto k    = saved_steps.size();
        const auto name = fmt::format("{}_{:04}.vtu", filename, k);
        save_physical(path / name, q, map, jacobian, exact(time));
        saved_steps.emplace_back(time, name);
        save_collection(path / fmt::format("{}.pvd", filename), saved_steps);

        update_u();
        auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   level[cell] = cell.level;
                               });
        samurai::save(path, fmt::format("{}_parametric_{:04}", filename, k), mesh, u_phys, level);
    };
    if (nfiles > 0)
    {
        fs::create_directories(path);
        save(0);
    }

    double t              = 0;
    std::size_t nt        = 0;
    std::size_t max_cells = diag0.n_cells;
    while (t < Tf)
    {
        const double dt_n = std::min(dt, Tf - t);
        if (mr)
        {
            adapt(mra_config);
        }
        qnp1.resize();
        update_u();
        qnp1 = q - dt_n * scheme(u_phys);
        samurai::swap(q, qnp1);
        t += dt_n;
        ++nt;
        max_cells = std::max(max_cells, mesh.nb_cells(decltype(mesh)::mesh_id_t::cells));

        if (nfiles > 0 && (t >= static_cast<double>(saved_steps.size()) * Tf / static_cast<double>(nfiles) || t >= Tf))
        {
            save(t);
        }
    }

    auto diag = compute_diagnostics(q, map, jacobian, exact(t));
    fmt::print("final:   t = {:.4f}, iterations = {}, cells = {} (max {}), levels [{}, {}]\n",
               t,
               nt,
               diag.n_cells,
               max_cells,
               mesh[decltype(mesh)::mesh_id_t::cells].min_level(),
               mesh[decltype(mesh)::mesh_id_t::cells].max_level());
    fmt::print("         relative mass drift = {:.3e}\n", (diag.mass - diag0.mass) / diag0.mass);
    fmt::print("         L1 error = {:.6e}, max error = {:.3e}\n", diag.l1_error, diag.max_error);

    samurai::finalize();
    return 0;
}
