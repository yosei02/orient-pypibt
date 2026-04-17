import numpy as np

from pypibt import PIBT, is_valid_mapf_solution


def test_PIBT():
    grid = np.full((2, 3), True)
    starts = [(0, 0), (0, 2)]
    goals = [(0, 2), (0, 0)]
    pibt = PIBT(grid, starts, goals)
    configs = pibt.run()
    assert is_valid_mapf_solution(grid, starts, goals, configs)


def test_PIBT_requires_turn_before_forward_move():
    grid = np.full((1, 2), True)
    starts = [(0, 0)]
    goals = [(0, 1)]
    pibt = PIBT(grid, starts, goals, initial_orientations=["Y_MINUS"])

    configs = pibt.run()

    assert configs == [
        [(0, 0)],
        [(0, 0)],
        [(0, 1)],
    ]
    assert is_valid_mapf_solution(grid, starts, goals, configs)


def test_PIBT_moves_immediately_when_already_facing_forward():
    grid = np.full((1, 2), True)
    starts = [(0, 0)]
    goals = [(0, 1)]
    pibt = PIBT(grid, starts, goals, initial_orientations=["X_PLUS"])

    configs = pibt.run()

    assert configs == [
        [(0, 0)],
        [(0, 1)],
    ]
    assert is_valid_mapf_solution(grid, starts, goals, configs)


def test_PIBT_requires_two_turn_steps_for_180_degree_rotation():
    grid = np.full((1, 2), True)
    starts = [(0, 1)]
    goals = [(0, 0)]
    pibt = PIBT(grid, starts, goals, initial_orientations=["X_PLUS"])

    configs = pibt.run()

    assert configs == [
        [(0, 1)],
        [(0, 1)],
        [(0, 1)],
        [(0, 0)],
    ]
    assert is_valid_mapf_solution(grid, starts, goals, configs)
