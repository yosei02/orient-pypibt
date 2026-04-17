import os

import numpy as np

from pypibt.mapf_utils import (
    DEFAULT_ORIENTATION,
    get_grid,
    get_neighbors,
    get_scenario,
    get_total_path_length,
    is_valid_coord,
    is_valid_mapf_solution,
    save_configs_for_visualizer,
    save_configs_for_visualizer_with_orientations,
)


def test_get_grid():
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    grid = get_grid(map_name)
    assert grid.shape == (2, 3)
    assert np.array_equal(grid, np.array([[False, True, True], [True, True, True]]))


def test_get_scenario():
    scen_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.scen")
    starts, goals = get_scenario(scen_name)
    assert len(starts) == 2
    assert len(goals) == 2
    # (y, x)
    assert starts[0] == (0, 1) and goals[0] == (1, 2)
    assert starts[1] == (1, 2) and goals[1] == (1, 0)

    starts, goals = get_scenario(scen_name, N=1)
    assert len(starts) == 1

    starts, goals = get_scenario(scen_name, N=3)
    assert len(starts) == 2


def test_get_scenario_with_orientations():
    scen_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.scen")
    starts, goals, orientations = get_scenario(scen_name, with_orientations=True)
    assert starts == [(0, 1), (1, 2)]
    assert goals == [(1, 2), (1, 0)]
    assert orientations == ["X_PLUS", "Y_MINUS"]


def test_get_scenario_defaults_orientation_when_not_present(tmp_path):
    scen_name = tmp_path / "simple.scen"
    scen_name.write_text(
        "version 1\n"
        "0\t3x2.map\t3\t2\t1\t0\t2\t1\t0\n",
        encoding="utf-8",
    )

    starts, goals, orientations = get_scenario(str(scen_name), with_orientations=True)

    assert starts == [(0, 1)]
    assert goals == [(1, 2)]
    assert orientations == [DEFAULT_ORIENTATION]


def test_is_valid_coord():
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    grid = get_grid(map_name)
    assert is_valid_coord(grid, (1, 1))
    assert not is_valid_coord(grid, (0, 0))
    assert not is_valid_coord(grid, (-1, 0))
    assert not is_valid_coord(grid, (2, 0))
    assert not is_valid_coord(grid, (0, -1))
    assert not is_valid_coord(grid, (0, 3))


def test_get_neighbors():
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    grid = get_grid(map_name)
    # (y, x)
    neigh = get_neighbors(grid, (0, 1))
    assert len(neigh) == 2
    assert (1, 1) in neigh and (0, 2) in neigh

    neigh = get_neighbors(grid, (1, 1))
    assert len(neigh) == 3
    assert (0, 1) in neigh and (1, 0) in neigh and (1, 2) in neigh

    # invalid check
    neigh = get_neighbors(grid, (0, 0))
    assert len(neigh) == 0

    neigh = get_neighbors(grid, (2, 0))
    assert len(neigh) == 0

    neigh = get_neighbors(grid, (1, 3))
    assert len(neigh) == 0


def test_save_configs_for_visualizer():
    configs = [
        [(0, 1), (1, 2)],
        [(0, 2), (1, 1)],
        [(1, 2), (1, 0)],
    ]
    save_configs_for_visualizer(configs, "./local/tmp.txt")


def test_save_configs_for_visualizer_with_orientations(tmp_path):
    filename = tmp_path / "output.txt"
    configs = [
        [(18, 23)],
        [(18, 23)],
        [(18, 24)],
    ]
    orientations = [
        ["Y_MINUS"],
        ["X_PLUS"],
        ["X_PLUS"],
    ]

    save_configs_for_visualizer_with_orientations(configs, str(filename), orientations)

    assert filename.read_text(encoding="utf-8").splitlines() == [
        "0:(23,18,Y_MINUS),",
        "1:(23,18,X_PLUS),",
        "2:(24,18,X_PLUS),",
    ]


def test_get_total_path_length_with_orientations():
    configs = [
        [(0, 0), (1, 1)],
        [(0, 0), (1, 1)],
        [(0, 1), (1, 1)],
    ]
    orientations = [
        ["Y_MINUS", "X_PLUS"],
        ["X_PLUS", "X_PLUS"],
        ["X_PLUS", "X_PLUS"],
    ]

    assert get_total_path_length(configs, orientations) == 2


def test_is_valid_mapf_solution():
    map_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.map")
    scen_name = os.path.join(os.path.dirname(__file__), "assets", "3x2.scen")
    grid = get_grid(map_name)
    starts, goals = get_scenario(scen_name)

    # feasible solution
    assert is_valid_mapf_solution(
        grid,
        starts,
        goals,
        [
            [(0, 1), (1, 2)],
            [(0, 2), (1, 1)],
            [(1, 2), (1, 0)],
        ],
    )

    # invalid starts
    assert not is_valid_mapf_solution(
        grid,
        starts,
        goals,
        [
            [(0, 2), (1, 2)],
            [(0, 2), (1, 1)],
            [(1, 2), (1, 0)],
        ],
    )

    # invalid goals
    assert not is_valid_mapf_solution(
        grid,
        starts,
        goals,
        [
            [(0, 1), (1, 2)],
            [(0, 2), (1, 1)],
            [(1, 2), (1, 1)],
        ],
    )

    # non-connected path
    assert not is_valid_mapf_solution(
        grid,
        starts,
        goals,
        [
            [(0, 1), (1, 2)],
            [(1, 2), (1, 1)],
            [(1, 2), (1, 0)],
        ],
    )

    # vertex collision
    assert not is_valid_mapf_solution(
        grid,
        starts,
        goals,
        [
            [(0, 1), (1, 2)],
            [(1, 1), (1, 1)],
            [(1, 2), (1, 0)],
        ],
    )

    # edge collision
    assert not is_valid_mapf_solution(
        grid,
        starts,
        goals,
        [
            [(0, 1), (1, 2)],
            [(1, 1), (1, 2)],
            [(1, 2), (1, 1)],
            [(1, 2), (1, 0)],
        ],
    )
