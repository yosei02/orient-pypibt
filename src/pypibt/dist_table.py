from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .mapf_utils import Coord, Grid, Orientation, is_valid_coord


@dataclass
class DistTable:
    """Distance table for computing orientation-aware distances using BFS.

    This class lazily evaluates distances from a goal position to any target
    state ``(position, orientation)`` on the grid using breadth-first search
    over an orientation-augmented state space. Distances are cached for
    efficient repeated queries.

    Attributes:
        grid: 2D boolean array where True indicates free space.
        goal: Goal position (y, x) coordinates.
        Q: Queue for BFS traversal (lazy distance evaluation).
        table: Distance tensor storing computed distances.
    """

    grid: Grid
    goal: Coord
    Q: deque[tuple[Coord, Orientation]] = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance tensor
    ORIENTATIONS: tuple[Orientation, ...] = ("X_PLUS", "Y_PLUS", "X_MINUS", "Y_MINUS")
    ORIENTATION_TO_INDEX: dict[Orientation, int] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize distance table with goal position."""
        self.ORIENTATION_TO_INDEX = {
            orientation: idx for idx, orientation in enumerate(self.ORIENTATIONS)
        }
        self.Q = deque()
        self.table = np.full(
            (len(self.ORIENTATIONS), self.grid.shape[0], self.grid.shape[1]),
            self.grid.size * len(self.ORIENTATIONS),
            dtype=int,
        )
        for orientation in self.ORIENTATIONS:
            self.table[self._index(orientation), self.goal[0], self.goal[1]] = 0
            self.Q.append((self.goal, orientation))

    def _index(self, orientation: Orientation) -> int:
        return self.ORIENTATION_TO_INDEX[orientation]

    @staticmethod
    def _turn_left(orientation: Orientation) -> Orientation:
        order: tuple[Orientation, ...] = ("X_PLUS", "Y_PLUS", "X_MINUS", "Y_MINUS")
        return order[(order.index(orientation) - 1) % len(order)]

    @staticmethod
    def _turn_right(orientation: Orientation) -> Orientation:
        order: tuple[Orientation, ...] = ("X_PLUS", "Y_PLUS", "X_MINUS", "Y_MINUS")
        return order[(order.index(orientation) + 1) % len(order)]

    @staticmethod
    def _backward_coord(coord: Coord, orientation: Orientation) -> Coord:
        y, x = coord
        if orientation == "X_PLUS":
            return (y, x - 1)
        if orientation == "X_MINUS":
            return (y, x + 1)
        if orientation == "Y_PLUS":
            return (y - 1, x)
        return (y + 1, x)

    def _predecessors(
        self, coord: Coord, orientation: Orientation
    ) -> list[tuple[Coord, Orientation]]:
        predecessors = [
            (coord, self._turn_left(orientation)),
            (coord, self._turn_right(orientation)),
        ]

        backward = self._backward_coord(coord, orientation)
        if is_valid_coord(self.grid, backward):
            predecessors.append((backward, orientation))

        return predecessors

    def get(self, target: Coord, orientation: Orientation) -> int:
        """Get shortest path distance from goal to target state.

        Uses lazy BFS evaluation to compute distance on demand. Previously
        computed distances are cached in the table.

        Args:
            target: Target position (y, x) coordinates.
            orientation: Facing direction at the target position.

        Returns:
            Shortest path distance from goal to target state. Returns a large
            sentinel value if target is invalid or unreachable.
        """
        if not is_valid_coord(self.grid, target):
            return int(self.table.max())

        orientation_idx = self._index(orientation)
        if int(self.table[orientation_idx, target[0], target[1]]) < self.table.size:
            return int(self.table[orientation_idx, target[0], target[1]])

        while len(self.Q) > 0:
            coord, current_orientation = self.Q.popleft()
            current_idx = self._index(current_orientation)
            d = int(self.table[current_idx, coord[0], coord[1]])
            for prev_coord, prev_orientation in self._predecessors(
                coord, current_orientation
            ):
                prev_idx = self._index(prev_orientation)
                if d + 1 < self.table[prev_idx, prev_coord[0], prev_coord[1]]:
                    self.table[prev_idx, prev_coord[0], prev_coord[1]] = d + 1
                    self.Q.append((prev_coord, prev_orientation))
            if coord == target and current_orientation == orientation:
                return d

        return int(self.table.max())
