import random
from dataclasses import dataclass
import numpy as np
from scipy.stats import halfcauchy
from interfaces.abstract_solution import AbstractSolution
from interfaces.result import Result


@dataclass(frozen=True)
class Solution:
    start: int
    end: int
    mask: np.array


def decision_function(delta: float, T: float) -> float:
    return np.exp(-delta / T)


class Annealing(AbstractSolution):
    def __init__(
        self,
        path_to_solve: str,
        iterations: int = 10000,
        steps_in_one_epoch: int = 100,
        initial_temperature: int = 10000,
        alpha: float = 0.999,
    ) -> None:

        super().__init__(path_to_solve)
        self.data = self.read_data_from_json()
        self.number_of_days = self.data.D2 - self.data.D1 + 1
        self.number_of_people = self.data.number_of_people
        self.min_days = self.data.D
        self.max_seats = self.data.fmax
        self.alpha = self.data.alpha

        self.priorities = self._priorities()
        self.prices = np.array(self.data.prices)

        self.iterations = iterations
        self.initial_temperature = initial_temperature
        self.alpha = alpha
        self.steps_in_one_epoch = steps_in_one_epoch

    def _priorities(self) -> np.array:
        self.mapping = {}
        self.idx = np.array(list(range(self.number_of_people)))
        priorities = np.zeros((self.number_of_people, self.number_of_days))

        for i, (friend, values) in enumerate(self.data.F.items()):
            self.mapping[i] = friend
            priorities[i] = np.array(values)

        return priorities

    def _random_solution(self) -> Solution:
        mask = np.random.choice(self.idx, size=self.max_seats, replace=False)
        start = np.random.randint(self.number_of_days - self.min_days)
        end = min(
            int(halfcauchy.rvs(loc=start + self.min_days, scale=5)),
            self.number_of_days - 1,
        )
        return Solution(start, end, mask)

    def _calculate_loss(self, sol: Solution) -> float:
        priority_val = np.sum(self.priorities[sol.mask, sol.start: sol.end + 1])
        holiday_cost = np.sum(self.prices[sol.start: sol.end + 1]) * self.alpha
        empty_seats = (self.max_seats - np.unique(sol.mask).shape[0]) * 1000
        return -priority_val + holiday_cost + empty_seats

    def _sol_to_res(self, sol: Solution) -> Result:
        friends = list(map(self.mapping.get, sol.mask))
        return Result(sol.start, sol.end, friends)

    def _neighboring_interval(self, start: int, end: int) -> tuple[int, int]:
        if np.random.randint(0, 2):
            start += random.choice([-1, 1])
            start = max(self.data.D1, start)
            if end - start + 1 < self.min_days:
                start -= 1
        else:
            end += random.choice([-1, 1])
            end = min(self.data.D2, end)
            if end - start + 1 < self.min_days:
                end += 1
        return start, end

    def _neighboring_set(self, mask: np.array) -> np.array:
        random_friend = np.random.randint(0, self.number_of_people)
        new_mask = mask.copy()
        new_mask[np.random.randint(0, self.max_seats)] = random_friend
        return new_mask

    def _generate_neighbor(self, sol: Solution) -> Solution:
        if np.random.randint(0, 2):
            interval = self._neighboring_interval(sol.start, sol.end)
            return Solution(*interval, sol.mask)
        else:
            mask = self._neighboring_set(sol.mask)
            return Solution(sol.start, sol.end, mask)

    def solve(self) -> tuple:
        best_loss_values = []
        current_state = self._random_solution()
        best_state = current_state
        current_energy = self._calculate_loss(current_state)
        opt_loss = current_energy

        T = self.initial_temperature

        for i in range(self.iterations):
            gen_best = float('inf')
            for _ in range(self.steps_in_one_epoch):
                neighbor = self._generate_neighbor(current_state)
                next_energy = self._calculate_loss(neighbor)
                if next_energy < current_energy:
                    current_state = neighbor
                    current_energy = next_energy
                else:
                    probability = decision_function(next_energy - current_energy, T)

                    if probability > np.random.uniform(0, 1):
                        current_state = neighbor
                        current_energy = next_energy

                if current_energy < opt_loss:
                    opt_loss = current_energy
                    best_state = current_state

                gen_best = min(current_energy, gen_best)

            if i % 100 == 0:
                print(f"Iteration {i}, best loss so far: {opt_loss}")
            if i % 10 == 0:
                best_loss_values.append(gen_best)
            T *= self.alpha

        return self._sol_to_res(best_state), best_loss_values


if __name__ == "__main__":
    solver = Annealing("../tests/test1.json")
    sol = solver.solve()
    print(sol)
