from interfaces.abstract_solution import AbstractSolution
from interfaces.result import Result
from dataclasses import dataclass
from scipy.stats import halfcauchy
import numpy as np


def mutate_mask(mask: np.array, max_people: int, p=0.1):
    for i in range(mask.shape[0]):
        if np.random.uniform() < p:
            swap = np.random.randint(max_people)
            if swap not in mask:
                mask[i] = swap
    return mask


def mutate_interval(start: int, end: int, max_day: int) -> tuple[int, int]:
    offset = int(np.random.normal(0, 8))
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


class Genetic(AbstractSolution):
    def __init__(self, path_to_solve: str, generation_no: int = 10000, population_size: int = 100) -> None:
        super().__init__(path_to_solve)
        if population_size % 4 != 0:
            raise ValueError("Population size should be divisible by four")

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
        end = min(int(halfcauchy.rvs(loc=start+self.min_days, scale=5)), self.number_of_days - 1)
        return Solution(start, end, mask)

    def _calculate_loss(self, sol: Solution) -> float:
        priority_val = np.sum(self.priorities[sol.mask, sol.start:sol.end+1])
        holiday_cost = np.sum(self.prices[sol.start:sol.end+1])*self.alpha
        empty_seats = (self.max_seats - np.unique(sol.mask).shape[0])*1000
        return -priority_val + holiday_cost + empty_seats

    def _crossover(self, sol1: Solution, sol2: Solution) -> Solution:
        new_start = (sol1.start + sol2.start) // 2
        new_end = (sol1.end + sol2.end + 1) // 2
        mid = self.max_seats // 2
        new_mask = np.concatenate((sol1.mask[:mid], sol2.mask[mid:]))
        return Solution(
            *mutate_interval(new_start, new_end, self.number_of_days - 1),
            mutate_mask(new_mask, self.number_of_people - 1)
        )

    def _sol_to_res(self, sol: Solution) -> Result:
        friends = list(map(self.mapping.get, sol.mask))
        return Result(sol.start, sol.end, friends)

    def _selection(self, population: list['Solution'], best_values: np.array) -> list['Solution']:
        best_parents = best_values[:(self.population_size // 4)]
        new_population = []
        for _ in range(self.population_size // 4):
            parent1, parent2 = np.random.choice(best_parents, size=2)
            child1 = self._crossover(population[parent1], population[parent2])
            child2 = self._crossover(population[parent2], population[parent1])
            new_population.extend([child1, child2])

        return new_population

    def solve(self) -> tuple:
        best_cost_values = []
        population = [self._random_solution() for _ in range(self.population_size)]
        best_sol = population[0]
        best_loss = self._calculate_loss(best_sol)

        for i in range(self.generations):
            if i % 100 == 0:
                print(f"Generation: {i}")
                print(f"Best loss so far: {best_loss}")
            if i%10 ==0:
                best_cost_values.append(best_loss)

            loss_values = np.array(list(map(self._calculate_loss, population)))
            best_values = np.argsort(loss_values)

            if loss_values[best_values[0]] < best_loss:
                best_loss = loss_values[best_values[0]]
                best_sol = population[best_values[0]]

            new_random_samples = [self._random_solution() for _ in range(self.population_size // 2)]
            new_population = self._selection(population, best_values)

            population = new_population + new_random_samples

        return self._sol_to_res(best_sol), best_cost_values


if __name__ == '__main__':
    solver = Genetic("../tests/test4.json")
    print(solver.solve())

