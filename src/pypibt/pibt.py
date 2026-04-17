"""Priority Inheritance with Backtracking (PIBT) algorithm for MAPF."""
import numpy as np

from .dist_table import DistTable
from .mapf_utils import (
    DEFAULT_ORIENTATION,
    Config,
    Configs,
    Coord,
    Grid,
    Orientation,
    get_neighbors,
)


class PIBT:
    """Priority Inheritance with Backtracking algorithm for MAPF.

    PIBT is an iterative algorithm that computes collision-free paths for
    multiple agents quickly, even with hundreds of agents or more. It uses
    priority inheritance and backtracking to resolve conflicts efficiently.

    The algorithm is sub-optimal but provides acceptable solutions almost
    immediately. It maintains distance tables for each agent to their goal
    and uses these for informed decision making. Priorities are dynamically
    updated based on progress toward goals.

    Completeness Guarantee:
        All agents are guaranteed to reach their destinations within a finite
        time when all pairs of adjacent vertices belong to a simple cycle 
        (i.e., biconnected). This property holds regardless of the number 
        of agents.

    Attributes:
        grid: 2D boolean array where True indicates free space.
        starts: Initial positions of all agents.
        goals: Goal positions of all agents.
        N: Number of agents.
        dist_tables: Distance tables for each agent to their goal.
        NIL: Sentinel value representing unassigned agent.
        NIL_COORD: Sentinel value representing unassigned coordinate.
        occupied_now: Current occupation status of each grid cell.
        occupied_nxt: Next timestep occupation status of each grid cell.
        rng: Random number generator for tie-breaking.

    Example:
        >>> grid = get_grid("map.map")
        >>> starts, goals = get_scenario("scenario.scen", N=100)
        >>> pibt = PIBT(grid, starts, goals, seed=42)
        >>> solution = pibt.run(max_timestep=1000)
        >>> print(f"Solution length: {len(solution)}")

    References:
        Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2022).
        Priority inheritance with backtracking for iterative multi-agent
        path finding. Artificial Intelligence Journal.
        https://kei18.github.io/pibt2/

    Note:
        PIBT serves as a core component in LaCAM (AAAI-23), which uses
        PIBT to quickly obtain initial solutions for eventually optimal
        multi-agent pathfinding. See https://kei18.github.io/lacam-project/
    """

    def __init__(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        seed: int = 0,
        initial_orientations: list[Orientation] | None = None,
    ) -> None:
        """Initialize PIBT solver.

        Args:
            grid: 2D boolean array where True indicates free space.
            starts: Initial positions of all agents (y, x).
            goals: Goal positions of all agents (y, x).
            seed: Random seed for tie-breaking (default: 0).
            initial_orientations: Initial facing direction for each agent.
                Defaults to all agents facing negative y.
        """
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)
        self.initial_orientations = initial_orientations or [DEFAULT_ORIENTATION] * self.N
        assert len(self.initial_orientations) == self.N
        self.orientation_history: list[list[Orientation]] = [self.initial_orientations.copy()]

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _orientation_for_move(src: Coord, dst: Coord) -> Orientation | None:
        """Return the orientation required to move from src to dst."""
        if src == dst:
            return None

        dy = dst[0] - src[0]
        dx = dst[1] - src[1]
        if dy == -1 and dx == 0:
            return "Y_MINUS"
        if dy == 1 and dx == 0:
            return "Y_PLUS"
        if dy == 0 and dx == 1:
            return "X_PLUS"
        if dy == 0 and dx == -1:
            return "X_MINUS"
        raise ValueError(f"invalid move from {src} to {dst}")

    @staticmethod
    def _rotation_steps(src: Orientation, dst: Orientation) -> list[Orientation]:
        """Return the intermediate orientations for shortest 90-degree turns."""
        order: list[Orientation] = ["X_PLUS", "Y_PLUS", "X_MINUS", "Y_MINUS"]
        src_idx = order.index(src)
        dst_idx = order.index(dst)

        clockwise = (dst_idx - src_idx) % len(order)
        counter_clockwise = (src_idx - dst_idx) % len(order)

        if clockwise <= counter_clockwise:
            step = 1
            turns = clockwise
        else:
            step = -1
            turns = counter_clockwise

        orientations: list[Orientation] = []
        idx = src_idx
        for _ in range(turns):
            idx = (idx + step) % len(order)
            orientations.append(order[idx])

        return orientations

    def _append_rotation_steps(
        self,
        configs: Configs,
        orientations: list[Orientation],
        orientation_history: list[list[Orientation]],
        target_config: Config,
    ) -> None:
        """Insert wait steps until every moving agent faces its target cell."""
        rotations: list[list[Orientation]] = []
        max_turns = 0

        for i, (src, dst) in enumerate(zip(configs[-1], target_config)):
            desired = self._orientation_for_move(src, dst)
            if desired is None:
                turns: list[Orientation] = []
            else:
                turns = self._rotation_steps(orientations[i], desired)
            rotations.append(turns)
            max_turns = max(max_turns, len(turns))

        for turn_idx in range(max_turns):
            next_orientations = orientations.copy()
            for agent_idx, turns in enumerate(rotations):
                if turn_idx < len(turns):
                    next_orientations[agent_idx] = turns[turn_idx]
            configs.append(configs[-1].copy())
            orientations[:] = next_orientations
            orientation_history.append(orientations.copy())

    def _candidate_distance(
        self, agent_idx: int, src: Coord, current_orientation: Orientation, dst: Coord
    ) -> int:
        """Score a next-position candidate with orientation-aware travel cost."""
        desired = self._orientation_for_move(src, dst)
        if desired is None:
            return 1 + self.dist_tables[agent_idx].get(src, current_orientation)

        turn_cost = len(self._rotation_steps(current_orientation, desired))
        return 1 + turn_cost + self.dist_tables[agent_idx].get(dst, desired)

    def funcPIBT(
        self,
        Q_from: Config,
        Q_to: Config,
        orientations: list[Orientation],
        i: int,
    ) -> bool:
        """Core PIBT function for single agent planning with priority inheritance.

        Attempts to assign a collision-free next position for agent i. If
        another agent j occupies the desired position, recursively invokes
        PIBT for agent j (priority inheritance). Backtracks if no valid
        position is found.

        Args:
            Q_from: Current configuration (positions at current timestep).
            Q_to: Next configuration being constructed (modified in-place).
            i: Agent index to plan for.

        Returns:
            True if successfully assigned a position to agent i, False otherwise.
        """
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(
            C,
            key=lambda u: self._candidate_distance(i, Q_from[i], orientations[i], u),
        )

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, orientations, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(
        self, Q_from: Config, orientations: list[Orientation], priorities: list[float]
    ) -> Config:
        """Compute next configuration for all agents.

        Executes one timestep of PIBT by calling funcPIBT for all agents
        in priority order.

        Args:
            Q_from: Current configuration (positions at current timestep).
            priorities: Priority values for each agent (higher = earlier planning).

        Returns:
            Next configuration with updated positions for all agents.
        """
        # setup
        N = len(Q_from)
        Q_to: Config = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, orientations, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        """Run PIBT algorithm until all agents reach goals or timeout.

        Iteratively computes collision-free paths for all agents using PIBT.
        Priorities are dynamically updated: incremented when an agent hasn't
        reached its goal, reset otherwise.

        Args:
            max_timestep: Maximum number of timesteps to run (default: 1000).

        Returns:
            Sequence of configurations from start to goal. Each configuration
            is a list of agent positions (y, x) at that timestep.

        Example:
            >>> pibt = PIBT(grid, starts, goals)
            >>> solution = pibt.run(max_timestep=500)
            >>> print(f"Solved in {len(solution)} timesteps")
        """
        # define priorities
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(
                self.dist_tables[i].get(self.starts[i], self.initial_orientations[i])
                / self.grid.size
            )

        # main loop, generate sequence of configurations
        configs = [self.starts]
        orientations = self.initial_orientations.copy()
        self.orientation_history = [orientations.copy()]
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q = self.step(configs[-1], orientations, priorities)
            self._append_rotation_steps(configs, orientations, self.orientation_history, Q)
            if len(configs) > max_timestep:
                break
            configs.append(Q)

            for i, (src, dst) in enumerate(zip(configs[-2], Q)):
                desired = self._orientation_for_move(src, dst)
                if desired is not None:
                    orientations[i] = desired
            self.orientation_history.append(orientations.copy())

            # update priorities & goal check
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal

        return configs
