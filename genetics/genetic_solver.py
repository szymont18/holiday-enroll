from abstract_solution import AbstractSolution
import random
from typing import List
from enum import Enum
from result import Result
import numpy as np

class IntervalMutationType(Enum):
    WIDEN_START = 0
    WIDEN_END = 1
    SHRINK_START = 2
    SHRINK_END = 3


class GeneticSolver(AbstractSolution):

    def __init__(self, path_to_solve: str, generation_no=10000, mutation_sample=0.1):
        super().__init__(path_to_solve)
        self.data = self.read_data_from_json()

        self.max_seats = self.data.fmax
        self.friend_indexes = np.array(range(self.data.number_of_people))
        self.start = self.data.D1
        self.end = self.data.D2
        self.max_duration = self.data.D

        self._priorities()

        self.prices = self.data.prices
        self.alpha = self.data.alpha

        self.generations = generation_no
        self.mutation_ratio = mutation_sample

    """
    Change dict priorities to matrix
    """
    def _priorities(self):
        self.friends_names = list(self.data.F.keys())
        self.friend_priorities = np.array([self.data.F[friend] for friend in self.friends_names])


    """
    Function that sample random results
    Requires to start the genetic solver
    Also used to add more samples after each generation
    """
    def _random_samples(self, n=100) -> List[Result]:
        samples = []

        for _ in range(n):
            # Random friends
            chosen_friend = np.random.choice(self.friend_indexes, self.max_seats, replace=False)

            # Random boundaries
            start = np.random.randint(self.start, self.end - self.max_duration + 1)
            random_duration = np.random.randint(1, self.max_duration + 1)
            end = start + random_duration

            samples.append(Result(start, end, chosen_friend))

        return samples

    """
    Chose group of friends from the result and replace them
    Friends are chosen randomly
    """
    def _mutate_friends(self, result: Result):
        replace_ind = np.random.randint(len(result.friends))
        replace_by = np.random.randint(len(self.friend_priorities))

        result.friends[replace_ind] = replace_by

        return result

    """
    Move the start and the end of the interval a few position
    The interval can either widen or shrink 
    """
    def _mutate_interval(self, result: Result):
        move = random.randint(1, self.data.D * 2)  # Max mutation moving by 2 * D days
        mutation_character = random.choice(list(IntervalMutationType))

        if mutation_character == IntervalMutationType.SHRINK_START:
            result.start = min(result.end - self.data.D, result.start + move)

        elif mutation_character == IntervalMutationType.WIDEN_START:
            result.start = max(0, result.start - move)

        elif mutation_character == IntervalMutationType.SHRINK_END:
            result.end = max(result.start + self.data.D, result.end - move)

        else:
            result.end = min(result.end + move, self.data.D2)

        return result

    def _mutate(self, result:Result):
        if random.choice([True, False]):
            return self._mutate_friends(result)

        return self._mutate_interval(result)

    """
    Cross two results
    There are two type of crossing (by friends and by interval)
    """
    def _cross(self, result1: Result, result2: Result):
        if random.choice([True, False]):
            return self._friend_cross(result1, result2)

        return self._interval_cross(result1, result2)

    """
    Mix group of friends from the first result and mix them with group of
    friends from the second one
    """

    def _friend_cross(self, result1: Result, result2: Result):
        friends_from1 = np.random.choice(result1.friends, max(1, len(result1.friends) - 2), replace=False)
        friends_from2 = np.random.choice(result2.friends,
                                         min(np.random.randint(1, self.max_seats - len(friends_from1)), len(result2.friends)),
                                         replace=False)

        new_friends = np.concatenate((friends_from1, friends_from2))

        return Result(result1.start, result1.end, new_friends)

    """
    Take group of friends from the first result and try to fit them in the result2 interval
    The boundary of the result2 interval may shrink
    The return interval is the best match for the friends from first friend set 
    """
    def _interval_cross(self, result1: Result, result2: Result):
        priority_arr = [0] * (result2.end - result2.start + 1)

        # Find priority array
        for friend in result1.friends:
            for day in range(result2.start, result2.end):
                priority_arr[day - result2.start] += self.friend_priorities[friend][day]

        # Number of days is taken from result2
        no_days = max((result2.end - result2.start) // 2, self.max_duration)

        priority_sum = sum(priority_arr[:no_days])
        best_priority_sum = priority_sum
        best_start = result2.start
        best_end = best_start + no_days

        for start in range(result2.start + 1, result2.end - no_days):

            priority_sum -= priority_arr[start - result2.start - 1]
            priority_sum += priority_arr[start + no_days - result2.start]

            if priority_sum > best_priority_sum:
                best_start = start
                best_end = no_days

        return Result(best_start, best_end, result1.friends)

    def find_friend_interval_priority(self, friend, interval):
        priority = 0
        for p in self.data.F[friend][interval[0]:interval[1] + 1]:
            priority += p

        return priority

    def _calculate_loss(self, sol: Result) -> float:
        priority_val = np.sum(self.friend_priorities[sol.friends, sol.start:sol.end+1])
        holiday_cost = np.sum(self.prices[sol.start:sol.end + 1])*self.alpha
        empty_seats = (self.max_seats - np.unique(sol.friends).shape[0])*1000
        return -priority_val + holiday_cost + empty_seats

    def _reduce_population(self, samples, old_samples_loss, population_size):
        new_samples = []
        for i in range(population_size):
            new_samples.append(samples[old_samples_loss[i]])
        return new_samples


    def _get_result(self, sample:Result):
        friends_name = [self.friends_names[ind] for ind in np.unique(sample.friends)]
        result = Result(sample.start, sample.end, friends_name)
        return result

    def solve(self) -> Result:
        generation_population_no = 100
        samples = self._random_samples(generation_population_no)

        best_sample = samples[0]
        best_result = float('inf')

        for generation in range(self.generations):
            if generation % 100 == 0:
                print(f"Generation: {generation}. Best loss: {best_result}")

            # Find the best sample
            loss_values = np.array(list(map(self._calculate_loss, samples)))
            best_values = np.argsort(loss_values)

            if loss_values[best_values[0]] < best_result:
                best_result = loss_values[best_values[0]]
                best_sample = samples[best_values[0]]

            # Reduce population
            samples = self._reduce_population(samples, best_values, generation_population_no)

            # Create new generation
            for parent_pair_idx in range(len(samples) // 2):
                parent_pair = random.sample(samples, 2)

                # Create new children by crossing
                child = self._cross(parent_pair[0], parent_pair[1])

                # Mutate children
                child = self._mutate(child)

                # Add to population
                samples.append(child)

            # Create new random samples (10 %)
            samples.extend(self._random_samples(generation_population_no // 10))

        return self._get_result(best_sample)


if __name__ == '__main__':
    path = "../tests/test4.json"
    solver = GeneticSolver(path)
    result = solver.solve()
    print(f'Result = {str(result)}')
