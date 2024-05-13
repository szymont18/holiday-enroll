from interfaces.abstract_solution import AbstractSolution
from interfaces.result import Result
from dataclasses import dataclass
from scipy.stats import halfcauchy
import numpy as np

# np.random.seed(42)


def mutate_mask(mask: np.array, max_people: int, p=0.1):
    for i in range(mask.shape[0]):
        if np.random.uniform() < p:
            swap = np.random.randint(max_people)
            if swap not in mask:
                mask[i] = swap
    return mask


def mutate_interval(start: int, end: int, max_day: int) -> tuple[int, int]:
    offset = int(np.random.normal(0, 2))
    start += offset
    end += offset
    if start < 0:
        end -= start
        start = 0

    if end > max_day:
        start += max_day - end
        end = max_day

    return start, end


@dataclass(frozen=True)
class Solution:
    start: int
    end: int
    mask: np.array

    def __eq__(self, other):
        return isinstance(other, Solution) and self.start == other.start and self.end == other.end and np.array_equal(self.mask, other.mask)


class BeeSolver(AbstractSolution):
    def __init__(self, path_to_solve: str, generation_no: int = 3000,
                 population_size: int = 200,
                 max_gens_without_improvement: int = 7) -> None:
        super().__init__(path_to_solve)

        self.data = self.read_data_from_json()
        self.number_of_days = self.data.D2 - self.data.D1 + 1
        self.number_of_people = self.data.number_of_people
        self.min_days = self.data.D
        self.max_seats = self.data.fmax
        self.alpha = self.data.alpha

        self.priorities = self._priorities()
        self.prices = np.array(self.data.prices)

        self.generations = generation_no
        self.population_size = population_size
        self.max_stagnations = max_gens_without_improvement

        self.scouts: [Solution] = []
        self.scouts_flies = [0, 0, 0]

    def _priorities(self) -> np.array:
        """
        Store friends' priorities in numpy array
        """
        self.mapping = {}
        self.idx = np.array(list(range(self.number_of_people)))
        priorities = np.zeros((self.number_of_people, self.number_of_days))

        for i, (friend, values) in enumerate(self.data.F.items()):
            self.mapping[i] = friend
            priorities[i] = np.array(values)

        return priorities

    def _random_solution(self):
        mask = np.random.choice(self.idx, size=self.max_seats, replace=False)
        start = np.random.randint(self.number_of_days - self.min_days)
        end = min(int(halfcauchy.rvs(loc=start + self.min_days, scale=5)), self.number_of_days - 1)
        return Solution(start, end, mask)

    def _calculate_loss(self, sol: Solution) -> float:
        priority_val = np.sum(self.priorities[sol.mask, sol.start:sol.end + 1])
        holiday_cost = np.sum(self.prices[sol.start:sol.end + 1]) * self.alpha
        empty_seats = (self.max_seats - np.unique(sol.mask).shape[0]) * 1000
        return -priority_val + holiday_cost + empty_seats

    def _sol_to_res(self, sol: Solution) -> Result:
        friends = list(map(self.mapping.get, sol.mask))
        return Result(sol.start, sol.end, friends)

    def _recruitment(self, scout_number: int, no_iter: int) -> None:
        scout: Solution = self.scouts[scout_number]
        new_bees = []
        for _ in range(no_iter):
            if np.random.uniform(0, 1) > 0.5:
                bee = Solution(*mutate_interval(scout.start, scout.end, self.number_of_days - 1), scout.mask.copy())
            else:
                bee = Solution(scout.start, scout.end, mutate_mask(scout.mask.copy(), self.number_of_people - 1))
            new_bees.append(bee)
        # new_bees = [Solution(
        #     *mutate_interval(scout.start, scout.end, self.number_of_days - 1),
        #     mutate_mask(scout.mask.copy(), self.number_of_people - 1)
        # ) for _ in range(no_iter)]

        loss_values = np.array(list(map(self._calculate_loss, new_bees)))
        best_value = np.argmin(loss_values)

        if best_value < self._calculate_loss(scout):
            self.scouts[scout_number] = new_bees[best_value]
            self.scouts_flies[scout_number] = 0
        else:
            self.scouts_flies[scout_number] += 1
            if self.scouts_flies[scout_number] >= self.max_stagnations:
                self.scouts[scout_number] = self._random_solution()

    def solve(self) -> Result:
        population = [self._random_solution() for _ in range(self.population_size)]
        best_sol = population[0]
        best_loss = self._calculate_loss(best_sol)

        loss_values = np.array(list(map(self._calculate_loss, population)))
        best_values = np.argsort(loss_values)

        self.scouts = [population[best_values[0]], population[best_values[1]], population[best_values[2]]]
        for i in range(self.generations):
            if i % 100 == 0:
                print(f"Generation: {i}")
                print(f"Best loss so far: {best_loss}")

            loss_values = np.array(list(map(self._calculate_loss, population)))
            best_values = np.argsort(loss_values)

            self._update_scouts(population, best_values)

            if loss_values[best_values[0]] < best_loss:
                best_loss = loss_values[best_values[0]]
                best_sol = population[best_values[0]]

            self._recruitment(0, self.population_size * 2 // 5)
            self._recruitment(1, self.population_size // 5)
            self._recruitment(2, self.population_size // 5)

            new_random_samples = [self._random_solution() for _ in range(self.population_size // 5)]
            population = new_random_samples + self.scouts
        return self._sol_to_res(best_sol)

    def _update_scouts(self, population, best_values) -> None:
        if self.scouts[2] != population[best_values[2]]:
            self.scouts[2] = population[best_values[2]]
            if self.scouts[2] == self.scouts[1]:
                self.scouts_flies[2] = self.scouts_flies[1]
            elif self.scouts[2] == self.scouts[0]:
                self.scouts_flies[2] = self.scouts_flies[0]
            else:
                self.scouts_flies[2] = 0

        if self.scouts[1] != population[best_values[1]]:
            self.scouts[1] = population[best_values[1]]

            if self.scouts[1] == self.scouts[0]:
                self.scouts_flies[1] = self.scouts_flies[0]
            else:
                self.scouts_flies[1] = 0

        if self.scouts[0] != population[best_values[0]]:
            self.scouts[0] = population[best_values[0]]
            self.scouts_flies[0] = 0


if __name__ == '__main__':
    solver = BeeSolver("../tests/test5.json")
    print(solver.solve())
