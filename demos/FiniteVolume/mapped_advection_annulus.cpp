// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Prototype: finite volumes on a mapped polysquare.
//
// samurai works on the parametric domain [-2, 2]^2 \ (-1, 1)^2 (a polysquare,
// built with DomainBuilder), mapped onto the circular annulus 1 < |x| < 2.
// We solve du/dt + div(a u) = 0 with the rigid rotation a = (-x_2, x_1),
// which is tangent to both walls.
//
// The map is piecewise bilinear on the parametric grid of the geometry level
// (max_level) and is stored by its vertices. They come either from the
// analytic map x = rho(xi) xi / |xi| with rho = max(|xi_1|, |xi_2|), which
// folds the grid along the diagonals, or from the minimization of Garanzha's
// foldover-free energy started from it, with the boundary vertices sliding
// on the circles.
//
// The unknown stored in samurai is q = |K| u / h^2, the mass per unit of
// parametric area, so that the arithmetic mean used by the field projection
// conserves the mass without any change in the library.
//
// The geometry of a cell at level l is the polygon traced by its boundary
// at the geometry level. The physical area of a cell is then exactly the sum
// of the areas of its children, and the sum of the face normals of any cell
// is zero: the scheme is free-stream preserving on a uniform mesh and the
// volumes are nested across levels.

#include <samurai/domain_builder.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <bit>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
namespace fs = std::filesystem;

static constexpr std::size_t dim = 2;
using point_t                    = std::array<double, 2>;

point_t analytic_map(double s, double t)
{
    const double rho = std::max(std::abs(s), std::abs(t));
    const double r   = std::sqrt(s * s + t * t);
    return {rho * s / r, rho * t / r};
}

point_t velocity(const point_t& x)
{
    return {-x[1], x[0]};
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

// Square of parametric cells [i0, i0 + n) x [j0, j0 + n) of the geometry grid
struct Box
{
    long i0;
    long j0;
    long n;
};

// Piecewise bilinear map of the parametric grid of a given level onto the
// annulus. Vertex (i, j) is the parametric point (-2 + i h, -2 + j h) and
// the hole (-1, 1)^2 is [N / 4, 3 N / 4]^2 in indices.
class PolysquareMap
{
  public:

    PolysquareMap(std::size_t level, double h)
        : m_level(level)
        , m_h(h)
        , m_n(std::lround(4. / h))
        , m_x(static_cast<std::size_t>((m_n + 1) * (m_n + 1)), point_t{std::nan(""), std::nan("")})
    {
        if (m_n < 4 || m_n % 4 != 0)
        {
            throw std::runtime_error("the geometry grid must align with the hole");
        }
    }

    static PolysquareMap analytic(std::size_t level, double h)
    {
        PolysquareMap map(level, h);
        for (long j = 0; j <= map.m_n; ++j)
        {
            for (long i = 0; i <= map.m_n; ++i)
            {
                if (map.vertex_in_domain(i, j))
                {
                    map.vertex(i, j) = analytic_map(map.coordinate(i), map.coordinate(j));
                }
            }
        }
        return map;
    }

    // Same map on the grid of the next level: bilinear interpolation, then
    // the new boundary vertices are pushed back onto their circle.
    PolysquareMap refined() const
    {
        PolysquareMap fine(m_level + 1, 0.5 * m_h);
        for (long j = 0; j <= fine.m_n; ++j)
        {
            for (long i = 0; i <= fine.m_n; ++i)
            {
                if (!fine.vertex_in_domain(i, j))
                {
                    continue;
                }
                const long ic = i / 2;
                const long jc = j / 2;
                const long di = i % 2;
                const long dj = j % 2;
                point_t x     = {0, 0};
                for (long b = 0; b <= dj; ++b)
                {
                    for (long a = 0; a <= di; ++a)
                    {
                        const double w = (di ? 0.5 : 1.) * (dj ? 0.5 : 1.);
                        x[0] += w * vertex(ic + a, jc + b)[0];
                        x[1] += w * vertex(ic + a, jc + b)[1];
                    }
                }
                if (const double r = fine.boundary_radius(i, j); r > 0)
                {
                    const double norm = std::hypot(x[0], x[1]);
                    x                 = {r * x[0] / norm, r * x[1] / norm};
                }
                fine.vertex(i, j) = x;
            }
        }
        return fine;
    }

    std::size_t level() const
    {
        return m_level;
    }

    double step() const
    {
        return m_h;
    }

    long size() const
    {
        return m_n;
    }

    double coordinate(long i) const
    {
        return -2. + static_cast<double>(i) * m_h;
    }

    long index(double s) const
    {
        return std::lround((s + 2.) / m_h);
    }

    const point_t& vertex(long i, long j) const
    {
        return m_x[static_cast<std::size_t>(j * (m_n + 1) + i)];
    }

    point_t& vertex(long i, long j)
    {
        return m_x[static_cast<std::size_t>(j * (m_n + 1) + i)];
    }

    bool vertex_in_domain(long i, long j) const
    {
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        return i >= 0 && i <= m_n && j >= 0 && j <= m_n && !(i > a && i < b && j > a && j < b);
    }

    bool quad_in_domain(long i, long j) const
    {
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        return i >= 0 && i < m_n && j >= 0 && j < m_n && !(i >= a && i < b && j >= a && j < b);
    }

    // Radius of the circle carrying a boundary vertex, 0 inside the domain
    double boundary_radius(long i, long j) const
    {
        if (i == 0 || i == m_n || j == 0 || j == m_n)
        {
            return 2.;
        }
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        if (i >= a && i <= b && j >= a && j <= b)
        {
            return 1.;
        }
        return 0.;
    }

    // The 8 corners of the polysquare stay at their analytic position
    bool is_corner(long i, long j) const
    {
        auto extreme = [this](long k)
        {
            return k == 0 || k == m_n;
        };
        auto inner = [this](long k)
        {
            return k == m_n / 4 || k == 3 * m_n / 4;
        };
        return (extreme(i) && extreme(j)) || (inner(i) && inner(j));
    }

    point_t operator()(double s, double t) const
    {
        const double fs = (s + 2.) / m_h;
        const double ft = (t + 2.) / m_h;
        const long i    = std::clamp(static_cast<long>(std::floor(fs)), 0L, m_n - 1);
        const long j    = std::clamp(static_cast<long>(std::floor(ft)), 0L, m_n - 1);
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

    // The box itself if it lies in the domain, otherwise its mirror image
    // across the nearest wall of the parametric domain. Ghost cells get the
    // geometry of the cell they mirror, as the Neumann condition on q does.
    Box mirrored_into_domain(Box box) const
    {
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        auto reflect = [&box](long& i0, long wall)
        {
            i0 = 2 * wall - i0 - box.n;
        };
        for (int iter = 0; iter < 4; ++iter)
        {
            const bool in_hole = box.i0 >= a && box.i0 + box.n <= b && box.j0 >= a && box.j0 + box.n <= b;
            if (box.i0 < 0 || box.i0 >= m_n || box.j0 < 0 || box.j0 >= m_n)
            {
                for (long* i0 : {&box.i0, &box.j0})
                {
                    if (*i0 < 0)
                    {
                        reflect(*i0, 0);
                    }
                    else if (*i0 >= m_n)
                    {
                        reflect(*i0, m_n);
                    }
                }
            }
            else if (in_hole)
            {
                // distance from the center of the box to each wall, times 2
                const long ci = 2 * box.i0 + box.n;
                const long cj = 2 * box.j0 + box.n;
                const std::array<long, 4> dist{ci - 2 * a, 2 * b - ci, cj - 2 * a, 2 * b - cj};
                const auto k = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
                reflect(k < 2 ? box.i0 : box.j0, (k % 2 == 0) ? a : b);
            }
            else
            {
                return box;
            }
        }
        throw std::runtime_error("cell too far from the domain to mirror it");
    }

    // Largest interior angle of the quads of the geometry grid in a box, in degrees
    double max_angle(const Box& box) const
    {
        double angle = 0;
        for (long j = box.j0; j < box.j0 + box.n; ++j)
        {
            for (long i = box.i0; i < box.i0 + box.n; ++i)
            {
                for (const auto& [a, b] : corner_edges(i, j))
                {
                    angle = std::max(angle, std::atan2(cross(a, b), a[0] * b[0] + a[1] * b[1]));
                }
            }
        }
        return angle * 180. / std::numbers::pi;
    }

    // Edges leaving each corner of quad (i, j), counterclockwise then clockwise.
    // [a b] / h is the Jacobian of the corner up to a rotation of the
    // parametric frame.
    std::array<std::pair<point_t, point_t>, 4> corner_edges(long i, long j) const
    {
        const std::array<point_t, 4> p = {vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)};
        std::array<std::pair<point_t, point_t>, 4> edges;
        for (std::size_t c = 0; c < 4; ++c)
        {
            const auto& x    = p[c];
            const auto& next = p[(c + 1) % 4];
            const auto& prev = p[(c + 3) % 4];
            edges[c]         = {
                {next[0] - x[0], next[1] - x[1]},
                {prev[0] - x[0], prev[1] - x[1]}
            };
        }
        return edges;
    }

  private:

    std::size_t m_level;
    double m_h;
    long m_n;
    std::vector<point_t> m_x;
};

// Garanzha's foldover-free energy of the map, summed over the corners of the
// quads: f(J) = (1 - theta) tr(J^T J) / (2 det J) + theta (det J / V + V / det J) / 2,
// with V = |annulus| / |polysquare| = pi / 4. It is infinite as soon as a
// corner folds, so a descent started from a valid map stays valid.
class MapEnergy
{
  public:

    MapEnergy(PolysquareMap& map, double theta)
        : m_map(map)
        , m_theta(theta)
        , m_first(static_cast<std::size_t>((map.size() + 1) * (map.size() + 1)), -1)
    {
        const long n = map.size();
        for (long j = 0; j <= n; ++j)
        {
            for (long i = 0; i <= n; ++i)
            {
                if (!map.vertex_in_domain(i, j) || map.is_corner(i, j))
                {
                    continue;
                }
                m_first[flat(i, j)] = m_size;
                m_size += (map.boundary_radius(i, j) > 0) ? 1 : 2;
            }
        }
    }

    std::size_t size() const
    {
        return static_cast<std::size_t>(m_size);
    }

    // Free interior vertices carry (x, y), sliding boundary vertices their angle
    std::vector<double> dofs() const
    {
        std::vector<double> x(size());
        for_each_free_vertex(
            [&](long i, long j, std::size_t k, double r)
            {
                const auto& p = m_map.vertex(i, j);
                if (r > 0)
                {
                    x[k] = std::atan2(p[1], p[0]);
                }
                else
                {
                    x[k]     = p[0];
                    x[k + 1] = p[1];
                }
            });
        return x;
    }

    void set_dofs(const std::vector<double>& x)
    {
        for_each_free_vertex(
            [&](long i, long j, std::size_t k, double r)
            {
                m_map.vertex(i, j) = (r > 0) ? point_t{r * std::cos(x[k]), r * std::sin(x[k])} : point_t{x[k], x[k + 1]};
            });
    }

    // Energy of the map for the given dofs and its gradient
    double operator()(const std::vector<double>& x, std::vector<double>& grad)
    {
        set_dofs(x);
        const long n   = m_map.size();
        const double h = m_map.step();
        const double V = std::numbers::pi / 4;
        std::vector<point_t> g(m_first.size(), point_t{0, 0});
        double energy = 0;
        for (long j = 0; j < n; ++j)
        {
            for (long i = 0; i < n; ++i)
            {
                if (!m_map.quad_in_domain(i, j))
                {
                    continue;
                }
                const std::array<std::pair<long, long>, 4> ids = {
                    {{i, j}, {i + 1, j}, {i + 1, j + 1}, {i, j + 1}}
                };
                const auto edges = m_map.corner_edges(i, j);
                for (std::size_t c = 0; c < 4; ++c)
                {
                    const auto& [a, b] = edges[c];
                    const double det   = cross(a, b) / (h * h);
                    if (det <= 0)
                    {
                        return std::numeric_limits<double>::infinity();
                    }
                    const double tr = (a[0] * a[0] + a[1] * a[1] + b[0] * b[0] + b[1] * b[1]) / (h * h);
                    energy += (1 - m_theta) * tr / (2 * det) + m_theta * (det / V + V / det) / 2;

                    // derivatives with respect to tr and det, then to a and b
                    const double f_tr  = (1 - m_theta) / (2 * det);
                    const double f_det = -(1 - m_theta) * tr / (2 * det * det) + m_theta * (1 / V - V / (det * det)) / 2;
                    const point_t ga   = {(2 * f_tr * a[0] + f_det * b[1]) / (h * h), (2 * f_tr * a[1] - f_det * b[0]) / (h * h)};
                    const point_t gb   = {(2 * f_tr * b[0] - f_det * a[1]) / (h * h), (2 * f_tr * b[1] + f_det * a[0]) / (h * h)};
                    auto add           = [&](std::pair<long, long> id, const point_t& v, double sign)
                    {
                        auto& gi = g[flat(id.first, id.second)];
                        gi[0] += sign * v[0];
                        gi[1] += sign * v[1];
                    };
                    add(ids[(c + 1) % 4], ga, 1);
                    add(ids[(c + 3) % 4], gb, 1);
                    add(ids[c], ga, -1);
                    add(ids[c], gb, -1);
                }
            }
        }
        grad.assign(size(), 0.);
        for_each_free_vertex(
            [&](long i, long j, std::size_t k, double r)
            {
                const auto& gi = g[flat(i, j)];
                if (r > 0)
                {
                    const double angle = x[k];
                    grad[k]            = r * (-std::sin(angle) * gi[0] + std::cos(angle) * gi[1]);
                }
                else
                {
                    grad[k]     = gi[0];
                    grad[k + 1] = gi[1];
                }
            });
        return energy;
    }

  private:

    std::size_t flat(long i, long j) const
    {
        return static_cast<std::size_t>(j * (m_map.size() + 1) + i);
    }

    template <class Func>
    void for_each_free_vertex(Func&& f) const
    {
        const long n = m_map.size();
        for (long j = 0; j <= n; ++j)
        {
            for (long i = 0; i <= n; ++i)
            {
                if (const long k = m_first[flat(i, j)]; k >= 0)
                {
                    f(i, j, static_cast<std::size_t>(k), m_map.boundary_radius(i, j));
                }
            }
        }
    }

    PolysquareMap& m_map;
    double m_theta;
    std::vector<long> m_first; // first dof of each vertex, -1 if it has none
    long m_size = 0;
};

// L-BFGS with a backtracking line search. The steps are limited to a
// fraction of the grid step, and a step that folds a corner has an infinite
// energy and is shortened like any other rejected step.
std::size_t minimize(MapEnergy& energy, double max_step, std::size_t max_iterations)
{
    constexpr std::size_t memory = 10;
    auto dot                     = [](const std::vector<double>& a, const std::vector<double>& b)
    {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.);
    };

    auto x = energy.dofs();
    std::vector<double> g;
    double e = energy(x, g);
    if (!std::isfinite(e))
    {
        throw std::runtime_error("the initial map folds");
    }
    std::deque<std::pair<std::vector<double>, std::vector<double>>> history; // (s, y)
    std::vector<double> x_new(x.size());
    std::vector<double> g_new;
    std::size_t stalled = 0;
    std::size_t it      = 0;
    for (; it < max_iterations && stalled < 10; ++it)
    {
        // two-loop recursion: d = -H g
        std::vector<double> d(g);
        std::vector<double> alpha(history.size());
        for (std::size_t k = history.size(); k-- > 0;)
        {
            const auto& [s, y] = history[k];
            alpha[k]           = dot(s, d) / dot(y, s);
            std::transform(d.begin(),
                           d.end(),
                           y.begin(),
                           d.begin(),
                           [&](double di, double yi)
                           {
                               return di - alpha[k] * yi;
                           });
        }
        if (!history.empty())
        {
            const auto& [s, y] = history.back();
            const double gamma = dot(s, y) / dot(y, y);
            std::transform(d.begin(),
                           d.end(),
                           d.begin(),
                           [&](double di)
                           {
                               return gamma * di;
                           });
        }
        for (std::size_t k = 0; k < history.size(); ++k)
        {
            const auto& [s, y] = history[k];
            const double beta  = dot(y, d) / dot(y, s);
            std::transform(d.begin(),
                           d.end(),
                           s.begin(),
                           d.begin(),
                           [&](double di, double si)
                           {
                               return di + (alpha[k] - beta) * si;
                           });
        }
        std::transform(d.begin(),
                       d.end(),
                       d.begin(),
                       [](double di)
                       {
                           return -di;
                       });
        double slope = dot(g, d);
        if (slope >= 0)
        {
            history.clear();
            std::transform(g.begin(),
                           g.end(),
                           d.begin(),
                           [](double gi)
                           {
                               return -gi;
                           });
            slope = dot(g, d);
        }

        const double d_max = std::abs(*std::max_element(d.begin(),
                                                        d.end(),
                                                        [](double a, double b)
                                                        {
                                                            return std::abs(a) < std::abs(b);
                                                        }));
        double step        = std::min(1., max_step / d_max);
        double e_new       = std::numeric_limits<double>::infinity();
        for (int k = 0; k < 60; ++k, step *= 0.5)
        {
            std::transform(x.begin(),
                           x.end(),
                           d.begin(),
                           x_new.begin(),
                           [&](double xi, double di)
                           {
                               return xi + step * di;
                           });
            e_new = energy(x_new, g_new);
            if (e_new <= e + 1e-4 * step * slope)
            {
                break;
            }
        }
        if (!(e_new <= e))
        {
            break;
        }
        stalled = (e - e_new <= 1e-9 * e) ? stalled + 1 : 0;

        std::vector<double> s(x.size());
        std::vector<double> y(x.size());
        std::transform(x_new.begin(), x_new.end(), x.begin(), s.begin(), std::minus<>());
        std::transform(g_new.begin(), g_new.end(), g.begin(), y.begin(), std::minus<>());
        if (dot(s, y) > 1e-14 * std::sqrt(dot(s, s) * dot(y, y)))
        {
            history.emplace_back(std::move(s), std::move(y));
            if (history.size() > memory)
            {
                history.pop_front();
            }
        }
        std::swap(x, x_new);
        std::swap(g, g_new);
        e = e_new;
    }
    energy.set_dofs(x);
    return it;
}

struct MapQuality
{
    double max_angle  = 0; // degrees
    double min_det    = std::numeric_limits<double>::max();
    double max_det    = 0;
    double min_sigma  = std::numeric_limits<double>::max();
    double max_sigma  = 0;
    double max_aspect = 0; // sigma_max / sigma_min
};

// Corner Jacobians of all the quads of the geometry grid
MapQuality map_quality(const PolysquareMap& map)
{
    MapQuality q;
    const double h = map.step();
    for (long j = 0; j < map.size(); ++j)
    {
        for (long i = 0; i < map.size(); ++i)
        {
            if (!map.quad_in_domain(i, j))
            {
                continue;
            }
            for (const auto& [a, b] : map.corner_edges(i, j))
            {
                const double det = cross(a, b) / (h * h);
                const double tr  = (a[0] * a[0] + a[1] * a[1] + b[0] * b[0] + b[1] * b[1]) / (h * h);
                const double rad = std::sqrt(std::max(0., tr * tr - 4 * det * det));
                const double s1  = std::sqrt(0.5 * (tr + rad));
                const double s2  = std::sqrt(std::max(0., 0.5 * (tr - rad)));
                q.max_angle      = std::max(q.max_angle, std::atan2(cross(a, b), a[0] * b[0] + a[1] * b[1]) * 180. / std::numbers::pi);
                q.min_det        = std::min(q.min_det, det);
                q.max_det        = std::max(q.max_det, det);
                q.min_sigma      = std::min(q.min_sigma, s2);
                q.max_sigma      = std::max(q.max_sigma, s1);
                q.max_aspect     = std::max(q.max_aspect, s1 / s2);
            }
        }
    }
    return q;
}

// Analytic map on the grid of the given level, optionally optimized level by
// level from a coarse grid: each level starts from the optimized map of the
// previous one, so that the fine levels only need a local smoothing.
PolysquareMap build_map(std::size_t level, double h, bool optimize, double theta, std::size_t max_iterations)
{
    if (!optimize)
    {
        return PolysquareMap::analytic(level, h);
    }
    const std::size_t start = std::min(level, std::size_t{2});
    auto map                = PolysquareMap::analytic(start, h * static_cast<double>(std::size_t{1} << (level - start)));
    while (true)
    {
        MapEnergy energy(map, theta);
        const auto t0                               = std::chrono::steady_clock::now();
        const auto iterations                       = minimize(energy, 0.25 * map.step(), max_iterations);
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - t0;
        fmt::print("map level {}: {} L-BFGS iterations, {:.2f} s\n", map.level(), iterations, elapsed.count());
        if (map.level() == level)
        {
            return map;
        }
        map = map.refined();
    }
}

// Highest geometry grid of a cell, at the level of the map
template <class Cell>
Box cell_box(const PolysquareMap& map, const Cell& cell)
{
    return {map.index(cell.corner(0)), map.index(cell.corner(1)), 1L << (map.level() - cell.level)};
}

template <class Cell>
std::vector<point_t> cell_polygon(const PolysquareMap& map, const Cell& cell)
{
    return map.polygon(cell_box(map, cell));
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

template <class Cell>
double physical_area(const PolysquareMap& map, const Cell& cell)
{
    return polygon_area(cell_polygon(map, cell));
}

// Mean Jacobian |K| / h^2 of a parametric cell. A cell outside the domain
// (a ghost) takes the Jacobian of its mirror image inside.
template <class Cell>
double mean_jacobian(const PolysquareMap& map, const Cell& cell)
{
    const auto box = map.mirrored_into_domain(cell_box(map, cell));
    return std::abs(polygon_area(map.polygon(box))) / (cell.length * cell.length);
}

// Physical flux of a through the face on the low side of 'cell' in
// direction d, oriented towards +d, summed over the segments of the
// geometry grid.
template <class Cell>
double face_flux(const PolysquareMap& map, const Cell& cell, std::size_t d)
{
    const auto box = cell_box(map, cell);
    double flux    = 0;
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
// outgoing flux of a cell during dt must not exceed its area. Coarser cells
// are less constrained, since their area grows like h^2 and their fluxes
// like h.
double stable_time_step(const PolysquareMap& map)
{
    double dt = std::numeric_limits<double>::max();
    for (long j = 0; j < map.size(); ++j)
    {
        for (long i = 0; i < map.size(); ++i)
        {
            if (!map.quad_in_domain(i, j))
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

// The scheme takes u and returns d(q)/dt. Its ghosts are then predicted
// from u, which keeps a constant state exact at level jumps; predicting q
// would not, since the prediction does not know the Jacobian.
template <class Field>
auto make_mapped_upwind(const PolysquareMap& map)
{
    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, 2, Field, Field>;

    samurai::FluxDefinition<cfg> upwind;
    samurai::static_for<0, dim>::apply(
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            upwind[d].cons_flux_function =
                [&map](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& u_stencil)
            {
                const double F = face_flux(map, data.cells[1], d);
                const double u = (F >= 0) ? u_stencil[0] : u_stencil[1];

                // The scheme multiplies by h_face / h_cell^2: divide by h_face
                // to get d(q)/dt = -sum(F u) / h_cell^2.
                flux = F * u / data.cell_length;
            };
        });
    return samurai::make_flux_based_scheme(upwind);
}

struct Diagnostics
{
    double mass         = 0;
    double l1_error     = 0;
    double max_error    = 0;
    std::size_t n_cells = 0;
};

// Parametric cell given by its level and integer indices
struct CellGeometry
{
    std::size_t level;
    double length;
    std::array<double, 2> lower;

    double corner(std::size_t d) const
    {
        return lower[d];
    }
};

template <class Mesh>
CellGeometry cell_geometry(const Mesh& mesh, std::size_t level, long long i, long long j)
{
    const double h = mesh.cell_length(level);
    const auto& o  = mesh.origin_point();
    return {
        level,
        h,
        {o[0] + h * static_cast<double>(i), o[1] + h * static_cast<double>(j)}
    };
}

// Prediction used by the mesh adaptation to fill the cells created by
// refinement. For q = |K| u / h^2, it predicts u with samurai's
// coefficients, multiplies it by the mean Jacobian of each child and adds
// the correction that keeps the mean of the children equal to the parent.
// The mass is conserved and a constant u stays constant, because the
// Jacobian of the parent is the mean of the Jacobians of its children.
// Other fields use the standard prediction.
// When u_source is given, the coarse values of u are read from it instead of
// q / J: its ghosts are predicted in u, not in q.
template <class Field>
auto mapped_prediction_fn(const PolysquareMap& map, bool enabled, const Field* u_source)
{
    return [&map, enabled, u_source](auto& new_field, const auto& old_field)
    {
        auto standard     = samurai::default_config::default_prediction_fn(new_field, old_field);
        const bool mapped = enabled && old_field.name() == "q";
        return [&new_field, &old_field, standard, mapped, &map, u_source](std::size_t level, const auto& interval, const auto& index) mutable
        {
            if (!mapped)
            {
                standard(level, interval, index);
                return;
            }
            using interval_t  = std::decay_t<decltype(interval)>;
            const auto& even  = samurai::prediction_coefficients<1>(0, 0).c;
            const auto& odd   = samurai::prediction_coefficients<1>(1, 0).c;
            const auto& mesh  = old_field.mesh();
            const auto j      = index[0];
            auto coarse_index = index;
            auto fine_index   = index;
            auto coeff        = [&](std::size_t parity, std::size_t s)
            {
                return parity == 0 ? even[s] : odd[s];
            };

            for (auto i = interval.start; i < interval.end; ++i)
            {
                std::array<std::array<double, 3>, 3> u;
                for (std::size_t a = 0; a < 3; ++a)
                {
                    for (std::size_t b = 0; b < 3; ++b)
                    {
                        const auto ia   = i + static_cast<int>(a) - 1;
                        coarse_index[0] = j + static_cast<int>(b) - 1;
                        u[a][b]         = u_source ? (*u_source)(level, interval_t{ia, ia + 1}, coarse_index)[0]
                                                   : old_field(level, interval_t{ia, ia + 1}, coarse_index)[0]
                                                 / mean_jacobian(map, cell_geometry(mesh, level, ia, coarse_index[0]));
                    }
                }
                coarse_index[0]       = j;
                const double q_parent = old_field(level, interval_t{i, i + 1}, coarse_index)[0];

                std::array<std::array<double, 2>, 2> q_child;
                double mean = 0;
                for (std::size_t dx = 0; dx < 2; ++dx)
                {
                    for (std::size_t dy = 0; dy < 2; ++dy)
                    {
                        double u_hat = 0;
                        for (std::size_t a = 0; a < 3; ++a)
                        {
                            for (std::size_t b = 0; b < 3; ++b)
                            {
                                u_hat += coeff(dx, a) * coeff(dy, b) * u[a][b];
                            }
                        }
                        const auto ic   = 2 * i + static_cast<int>(dx);
                        const auto jc   = 2 * j + static_cast<int>(dy);
                        q_child[dx][dy] = mean_jacobian(map, cell_geometry(mesh, level + 1, ic, jc)) * u_hat;
                        mean += 0.25 * q_child[dx][dy];
                    }
                }
                for (std::size_t dx = 0; dx < 2; ++dx)
                {
                    for (std::size_t dy = 0; dy < 2; ++dy)
                    {
                        const auto ic                                            = 2 * i + static_cast<int>(dx);
                        fine_index[0]                                            = 2 * j + static_cast<int>(dy);
                        new_field(level + 1, interval_t{ic, ic + 1}, fine_index) = q_child[dx][dy] + q_parent - mean;
                    }
                }
            }
        };
    };
}

template <class Field, class Exact>
Diagnostics compute_diagnostics(const Field& q, const PolysquareMap& map, Exact&& exact)
{
    Diagnostics diag;
    samurai::for_each_cell(q.mesh(),
                           [&](const auto& cell)
                           {
                               const double area = physical_area(map, cell);
                               const double u    = q[cell] * cell.length * cell.length / area;
                               auto c            = cell.center();
                               auto x            = map(c(0), c(1));
                               const double err  = std::abs(u - exact(x));
                               diag.mass += q[cell] * cell.length * cell.length;
                               diag.l1_error += err * area;
                               diag.max_error = std::max(diag.max_error, err);
                               ++diag.n_cells;
                           });
    return diag;
}

// VTK file of the physical cells with u, its error against the exact
// solution, the level, the mean Jacobian and the largest angle of the quads
// of the geometry grid inside the cell. Each cell is the polygon used by
// the scheme, so coarse cells follow the curved image of the grid lines.
template <class Field, class Exact>
void save_physical(const fs::path& file, const Field& q, const PolysquareMap& map, Exact&& exact)
{
    std::vector<point_t> points;
    std::vector<std::size_t> offsets = {0};
    std::vector<double> u;
    std::vector<double> error;
    std::vector<double> jacobian;
    std::vector<double> max_angle;
    std::vector<std::size_t> level;
    samurai::for_each_cell(q.mesh(),
                           [&](const auto& cell)
                           {
                               auto poly = cell_polygon(map, cell);
                               points.insert(points.end(), poly.begin(), poly.end());
                               offsets.push_back(points.size());
                               const double jac = mean_jacobian(map, cell);
                               auto c           = cell.center();
                               u.push_back(q[cell] / jac);
                               error.push_back(u.back() - exact(map(c(0), c(1))));
                               jacobian.push_back(jac);
                               max_angle.push_back(map.max_angle(cell_box(map, cell)));
                               level.push_back(cell.level);
                           });

    // VTK XML unstructured grid with the arrays appended as raw binary
    const std::size_t n = u.size();
    std::vector<float> coords;
    coords.reserve(3 * points.size());
    for (const auto& p : points)
    {
        coords.insert(coords.end(), {static_cast<float>(p[0]), static_cast<float>(p[1]), 0.f});
    }
    std::vector<std::int32_t> connectivity(points.size());
    std::iota(connectivity.begin(), connectivity.end(), 0);
    std::vector<std::int32_t> cell_ends(offsets.begin() + 1, offsets.end());
    std::vector<std::uint8_t> types(n, 7); // VTK_POLYGON
    auto to_float = [](const auto& values)
    {
        return std::vector<float>(values.begin(), values.end());
    };
    std::vector<std::int32_t> levels(level.begin(), level.end());

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

    auto u_f        = to_float(u);
    auto error_f    = to_float(error);
    auto jacobian_f = to_float(jacobian);
    auto angle_f    = to_float(max_angle);
    header += fmt::format("<Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n<Points>\n", points.size(), n);
    add_array("type=\"Float32\" NumberOfComponents=\"3\"", coords);
    header += "</Points>\n<Cells>\n";
    add_array("type=\"Int32\" Name=\"connectivity\"", connectivity);
    add_array("type=\"Int32\" Name=\"offsets\"", cell_ends);
    add_array("type=\"UInt8\" Name=\"types\"", types);
    header += "</Cells>\n<CellData Scalars=\"u\">\n";
    add_array("type=\"Float32\" Name=\"u\"", u_f);
    add_array("type=\"Float32\" Name=\"error\"", error_f);
    add_array("type=\"Float32\" Name=\"jacobian\"", jacobian_f);
    add_array("type=\"Float32\" Name=\"max_angle\"", angle_f);
    add_array("type=\"Int32\" Name=\"level\"", levels);
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

// ParaView collection that gives each VTK file its physical time
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
    auto& app = samurai::initialize("Linear advection on a polysquare mapped onto an annulus", argc, argv);

    double Tf                  = std::numbers::pi / 2;
    double cfl                 = 0.9;
    bool mr                    = true;
    bool bump                  = true;
    std::size_t nfiles         = 0;
    bool on_u                  = false;
    double base                = 0;
    bool mapped                = true;
    bool u_source              = false;
    bool optimize_map          = true;
    double theta               = 0.8;
    std::size_t map_iterations = 5000;

    std::size_t min_level = 3;
    std::size_t max_level = 7;

    fs::path path        = fs::current_path();
    std::string filename = "mapped_advection_annulus";

    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--mr,!--no-mr", mr, "Adapt the mesh with multiresolution")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--bump,!--constant", bump, "Initial condition: Gaussian bump or constant state")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--background", base, "Constant state added to the bump")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--mapped-prediction,!--standard-prediction", mapped, "Predict u rather than q when refining")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--u-source", u_source, "With --detail-on-u, read the coarse u of the prediction from u instead of q / J")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--detail-on-u", on_u, "Compute the details on u instead of q = |K| u / h^2")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_flag("--optimized-map,!--analytic-map", optimize_map, "Minimize Garanzha's energy of the map, starting from the analytic one")
        ->capture_default_str()
        ->group("Geometry");
    app.add_option("--map-theta", theta, "Weight of the volume term in the energy of the map")->capture_default_str()->group("Geometry");
    app.add_option("--map-iterations", map_iterations, "Maximum number of L-BFGS iterations per level of the map")
        ->capture_default_str()
        ->group("Geometry");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of saved time steps after the initial one (0: no output)")->capture_default_str()->group("Output");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    samurai::DomainBuilder<dim> domain({-2., -2.}, {2., 2.});
    domain.remove({-1., -1.}, {1., 1.});

    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
    auto mesh   = samurai::mra::make_mesh(domain, config);

    // --min-level and --max-level override the values above
    min_level      = mesh.min_level();
    max_level      = mesh.max_level();
    const auto map = build_map(max_level, mesh.cell_length(max_level), optimize_map, theta, map_iterations);
    const auto mq  = map_quality(map);
    fmt::print(
        "map ({}): largest angle = {:.2f} deg, det J in [{:.3f}, {:.3f}], singular values in [{:.3f}, {:.3f}], largest aspect = {:.2f}\n",
        optimize_map ? "optimized" : "analytic",
        mq.max_angle,
        mq.min_det,
        mq.max_det,
        mq.min_sigma,
        mq.max_sigma,
        mq.max_aspect);

    auto u0 = [bump, base](const point_t& x)
    {
        if (!bump)
        {
            return 1.;
        }
        const double dx = x[0] - 1.5;
        const double dy = x[1];
        return base + std::exp(-(dx * dx + dy * dy) / 0.02);
    };

    auto q    = samurai::make_scalar_field<double>("q", mesh);
    auto init = [&]()
    {
        q.resize();
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   auto c  = cell.center();
                                   q[cell] = mean_jacobian(map, cell) * u0(map(c(0), c(1)));
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
                                   u_phys[cell] = q[cell] / mean_jacobian(map, cell);
                               });
    };

    auto qnp1   = samurai::make_scalar_field<double>("qnp1", mesh);
    auto scheme = make_mapped_upwind<decltype(q)>(map);

    // With --detail-on-u, the details are computed on u so that the
    // geometry does not drive the refinement, and q follows the new mesh.
    auto adapt_on_q = samurai::make_MRAdapt(mapped_prediction_fn<decltype(u_phys)>(map, mapped, nullptr), q);
    auto adapt_on_u = samurai::make_MRAdapt(mapped_prediction_fn(map, mapped, u_source ? &u_phys : nullptr), u_phys);
    auto mra_config = samurai::mra_config().epsilon(1e-3); // --mr-eps overrides it
    auto adapt      = [&]()
    {
        if (on_u)
        {
            update_u();
            adapt_on_u(mra_config, q);
        }
        else
        {
            adapt_on_q(mra_config);
        }
    };
    if (mr)
    {
        // Adapt from the exact initial data rather than from projected values
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            adapt();
            init();
        }
    }

    const double dt = cfl * stable_time_step(map);

    auto exact = [&](double t)
    {
        return [&, t](const point_t& x)
        {
            const double c = std::cos(t);
            const double s = std::sin(t);
            return u0({c * x[0] + s * x[1], -s * x[0] + c * x[1]});
        };
    };

    auto diag0 = compute_diagnostics(q, map, exact(0));
    fmt::print("initial: cells = {}, mass = {:.15e}, max |u - u0| = {:.3e}\n", diag0.n_cells, diag0.mass, diag0.max_error);
    // Physical cells in <filename>_<k>.vtu, gathered with their time in
    // <filename>.pvd; parametric mesh in <filename>_parametric_<k>.h5.
    std::vector<std::pair<double, std::string>> saved_steps;
    auto save = [&](double time)
    {
        const auto k    = saved_steps.size();
        const auto name = fmt::format("{}_{:04}.vtu", filename, k);
        save_physical(path / name, q, map, exact(time));
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
        double dt_n = std::min(dt, Tf - t);
        if (mr)
        {
            adapt();
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

    auto diag = compute_diagnostics(q, map, exact(t));
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
