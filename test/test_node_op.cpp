#include <gtest/gtest.h>

#include <vector>

#include <samurai/box.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/subset_pointer/node_op.hpp>

namespace samurai
{
    TEST(mesh_node, construct)
    {
        constexpr std::size_t dim = 1;
        Box<int, dim> box{{0}, {1}};
        auto lca = LevelCellArray<dim>(dim, box);

        [[maybe_unused]] auto m_node = mesh_node<dim>(lca);
    }

    TEST(mesh_node, start)
    {
        constexpr std::size_t dim = 1;
        Box<int, dim> box{{0}, {1}};
        auto lca = LevelCellArray<dim>(dim, box);

        auto m_node = mesh_node<dim>(lca);

        EXPECT_EQ(0, m_node.start(0, 0));
    }

    TEST(translate_node, start)
    {
        constexpr std::size_t dim = 1;
        Box<int, dim> box{{0}, {1}};
        auto lca = LevelCellArray<dim>(dim, box);

        auto m_node = translate_op<dim>(lca, {1});

        EXPECT_EQ(1, m_node.start(0, 0));
    }

    TEST(contraction_node, start)
    {
        constexpr std::size_t dim = 1;
        Box<int, dim> box{{0}, {1}};
        auto lca = LevelCellArray<dim>(dim, box);

        auto m_node = contraction_op<dim>(lca, 1);

        EXPECT_EQ(1, m_node.start(0, 0));
    }

    TEST(node_op_impl, vector)
    {
        constexpr std::size_t dim = 1;
        Box<int, dim> box{{0}, {1}};
        auto lca = LevelCellArray<dim>(dim, box);

        std::vector<node_op<dim>> v({mesh_node<dim>(lca), translate_op<dim>(lca, 1)});

        EXPECT_EQ(0, v[0].start(0, 0));
        EXPECT_EQ(1, v[1].start(0, 0));
    }
}
