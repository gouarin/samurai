// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

template <std::size_t Dim>
struct config
{
    static constexpr std::size_t dim = Dim;
    using Config                     = samurai::mesh_config<dim>;
    using mesh_t                     = samurai::MRMesh<Config>;
    using field_t                    = samurai::VectorField<mesh_t, double, dim + 2>;
};
