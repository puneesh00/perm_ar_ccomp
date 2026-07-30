from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from tab_completion.sampling import CompletionTask


@dataclass
class FactorizationPlan:
    """
    steps is a list of arrays.
    Each step has shape [k_t, 2] and contains local coordinates [row, col].

    Parallel:
        one step containing all query cells.

    Perm-AR:
        several steps; after each step, true values are revealed during teacher forcing.
    """
    mode: str
    steps: list[np.ndarray]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_query_cells(self) -> int:
        return int(sum(len(step) for step in self.steps))


def validate_factorization(task: CompletionTask, plan: FactorizationPlan) -> None:
    query_coords = task.query_coords_local()

    if len(plan.steps) == 0:
        raise ValueError("FactorizationPlan has no steps.")

    all_step_coords = np.concatenate(plan.steps, axis=0)

    q_sorted = query_coords[np.lexsort((query_coords[:, 1], query_coords[:, 0]))]
    s_sorted = all_step_coords[np.lexsort((all_step_coords[:, 1], all_step_coords[:, 0]))]

    if q_sorted.shape != s_sorted.shape:
        raise ValueError(
            f"Factorization covers {s_sorted.shape[0]} cells, "
            f"but query set has {q_sorted.shape[0]} cells."
        )

    if not np.array_equal(q_sorted, s_sorted):
        raise ValueError("Factorization steps must cover exactly M once.")


@dataclass
class ParallelFactorizer:
    mode: str = "parallel"

    def build(self, task: CompletionTask, rng: np.random.Generator) -> FactorizationPlan:
        coords = task.query_coords_local()
        plan = FactorizationPlan(mode=self.mode, steps=[coords])
        validate_factorization(task, plan)
        return plan


@dataclass
class PermARFactorizer:
    """
    unit='cell':
        each step predicts one cell or group_size cells.

    unit='column':
        each step predicts all queried cells in one sampled column.
        This is usually the best default for speed.

    unit='row':
        each step predicts all queried cells in one sampled row.
    """
    unit: Literal["cell", "column", "row"] = "column"
    group_size: int = 1
    mode: str = "perm_ar"

    def build(self, task: CompletionTask, rng: np.random.Generator) -> FactorizationPlan:
        coords = task.query_coords_local()

        if self.unit == "cell":
            steps = self._cell_steps(coords, rng)
        elif self.unit == "column":
            steps = self._column_steps(coords, rng)
        elif self.unit == "row":
            steps = self._row_steps(coords, rng)
        else:
            raise ValueError(f"Unknown unit: {self.unit}")

        plan = FactorizationPlan(mode=f"{self.mode}:{self.unit}", steps=steps)
        validate_factorization(task, plan)
        return plan

    def _cell_steps(self, coords: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
        coords = coords[rng.permutation(len(coords))]

        if self.group_size <= 1:
            return [coords[i : i + 1] for i in range(len(coords))]

        return [
            coords[i : i + self.group_size]
            for i in range(0, len(coords), self.group_size)
        ]

    def _column_steps(self, coords: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
        cols = np.unique(coords[:, 1])
        cols = rng.permutation(cols)

        steps: list[np.ndarray] = []
        for c in cols:
            step = coords[coords[:, 1] == c]
            step = step[rng.permutation(len(step))]
            steps.append(step)

        return steps

    def _row_steps(self, coords: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
        rows = np.unique(coords[:, 0])
        rows = rng.permutation(rows)

        steps: list[np.ndarray] = []
        for r in rows:
            step = coords[coords[:, 0] == r]
            step = step[rng.permutation(len(step))]
            steps.append(step)

        return steps