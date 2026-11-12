// Expressions that user code commonly applies to the values handed to a flux function
// (StencilValues element V, FluxValue F) or to samurai::Array (A). One expression per PROBE
// value, compiled with -fsyntax-only for each variant by probe.sh.
#include <iostream>

#include <xtensor/containers/xfixed.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/views/xview.hpp>

#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

using V     = samurai::local_field_data_t<double, 4, false>;
using F     = samurai::flux_array_t<double, 4>;
using A     = samurai::Array<double, 4>;
using Fixed = xt::xtensor_fixed<double, xt::xshape<4>>;

double takes_fixed(const Fixed& x)
{
    return x[0];
}

template <class Field, class Cell>
double probe(Field& field, const Cell& cell, V v, F f, A a)
{
    double r = 0;
#if PROBE == 1 // explicit xtensor type initialised from a field value
    Fixed x = v;
    r       = x[0];
#elif PROBE == 2 // function with an explicit xtensor parameter
    r = takes_fixed(v);
#elif PROBE == 3 // xtensor reducer
    r = xt::sum(v)();
#elif PROBE == 4
    r = xt::norm_l2(v)();
#elif PROBE == 5
    r = xt::amax(xt::abs(v))();
#elif PROBE == 6  // sub-view of the components
    auto w = xt::view(v, xt::range(1, 3));
    r      = w(0);
#elif PROBE == 7  // arithmetic assigned to a flux
    f = v(0) * v;
    f = 0.5 * (v + v) - 2. * (v - v) / 3.;
    f += v;
    f -= 2 * v;
    r = f(0) + f[1];
#elif PROBE == 8  // range-for
    for (auto x : v)
    {
        r += x;
    }
#elif PROBE == 9  // stream output
    std::cout << v << f << a << std::endl;
#elif PROBE == 10 // shape
    r = static_cast<double>(v.shape()[0]);
#elif PROBE == 11 // element-wise xtensor math
    f = xt::maximum(v, v);
    f = xt::abs(v);
    f = xt::sqrt(v);
#elif PROBE == 12 // field <-> array
    field[cell] = a;
    a           = field[cell];
    field[cell] += 0.5 * a;
    f = a - field[cell];
    f = field[cell] * 2. + a;
    r = f(0);
#elif PROBE == 13 // lambda with an explicit xtensor return type
    auto g = [&]() -> Fixed
    {
        return a + a;
    };
    r = g()[0];
#elif PROBE == 14 // xtensor generators and containers into a flux, an array
    f = xt::zeros<double>({4});
    a = Fixed{1., 2., 3., 4.};
    A b{1., 2., 3., 4.};
    A c = {1., 2., 3., 4.};
    r   = b(0) + c(0);
#elif PROBE == 15 // comparison
    r = (a == a) ? 1. : 0.;
    r += xt::all(xt::equal(v, v)) ? 1. : 0.;
#elif PROBE == 16 // generic code on 'auto'
    auto z = v * v + a;
    r      = z(0) + static_cast<double>(z.size());
#elif PROBE == 17 // samurai::math on a field value
    using namespace samurai::math;
    r = sum(v * v);
#elif PROBE == 18 // scalar on the left of a subtraction or a division
    f = 1. - v;
    f = 1. / v;
    f = -v;
    r = f(0);
#elif PROBE == 19 // array of another type in an expression
    samurai::Array<int, 4> n{1, 2, 3, 4};
    f = n * v;
    r = f(0);
#elif PROBE == 20 // xtensor container on the other side
    Fixed x{1., 2., 3., 4.};
    f = x + v;
    f = v * x;
    x = v;
    r = f(0);
#elif PROBE == 21 // data pointer, as for a BLAS or a C kernel
    r = *v.data() + *v.begin();
#endif
    return r;
}

int main()
{
    auto config = samurai::mesh_config<2>().min_level(2).max_level(4);
    auto mesh   = samurai::mra::make_mesh(samurai::Box<double, 2>({0, 0}, {1, 1}), config);
    auto u      = samurai::make_vector_field<double, 4>("u", mesh);
    u.fill(1.);
    double r = 0;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               r += probe(u, cell, V(), F(), A());
                           });
    return r > 0 ? 0 : 1;
}
