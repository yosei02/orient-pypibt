import os

from pypibt.dist_table import DistTable
from pypibt.mapf_utils import get_grid


def test_DistTable():
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    grid = get_grid(map_name)
    goal = (1, 2)

    dist_table = DistTable(grid, goal)
    assert dist_table.get(goal, "X_PLUS") == 0
    assert dist_table.get((1, 0), "X_PLUS") == 2
    assert dist_table.get((1, 0), "Y_PLUS") == 3
    assert dist_table.get((1, 0), "X_MINUS") == 4
    assert dist_table.get((0, 0), "X_PLUS") == 24  # invalid coordination
    assert dist_table.get((0, 3), "X_PLUS") == 24  # invalid coordination
