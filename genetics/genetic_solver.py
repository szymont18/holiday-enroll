from abstract_solution import AbstractSolution
import random
from typing import List
from enum import Enum
from result import Result


class IntervalMutationType(Enum):
    WIDEN_START = 0
    WIDEN_END = 1
    SHRINK_START = 2
    SHRINK_END = 3


class GeneticSolver(AbstractSolution):

    def __init__(self, path_to_solve: str, generation_no=10000, mutation_sample=0.1):
        super().__init__(path_to_solve)
        self.data = self.read_data_from_json()

        self.generations = generation_no

        self.mutation_ratio = mutation_sample

    """
    Function that sample random results
    Requires to start the genetic solver
    Also used to add more samples after each generation
    """

    def _random_samples(self, n=100) -> List[Result]:
        samples = []
        friends = set(self.data.F.keys())

        for _ in range(n):
            chosen_friends = set(random.sample(list(friends), self.data.fmax))
            start = random.randint(0, self.data.D2 - self.data.D - 2)
            end = start + self.data.D
            samples.append(Result(start, end, chosen_friends))

        return samples

    """
    Chose group of friends from the result and replace them
    Friends are chosen randomly
    If there are too many friends try to add new
    """

    def _mutate_friends(self, result: Result):
        friends = result.friends
        all_friends = self.data.F

        replace_by = random.choice(list(all_friends.keys()))

        result.friends.pop()
        result.friends.add(replace_by)

        # There is still place in the car
        if len(result.friends) < self.data.fmax:
            random_friend = random.choice(list(all_friends.keys()))
            result.friends.add(random_friend)

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
        new_friends = set()
        friends_from1 = random.randint(1, len(result1.friends))
        friends_from2 = max(random.randint(1, len(result2.friends)), self.data.D - friends_from1)

        for _ in range(friends_from1):
            new_friends.add(random.sample(list(result1.friends), 1)[0])

        for _ in range(friends_from2):
            new_friends.add(random.sample(list(result1.friends), 1)[0])

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
                priority_arr[day - result2.start] += self.data.F[friend][day]

        # Number of days is taken from result1
        no_days = max((result2.end - result2.start) // 2, self.data.D)

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


    def _ass(self, samples: List[Result]):
        for sample in samples:
            assert 0 <= sample.start < len(self.data.prices)
            assert 0 <= sample.end < len(self.data.prices), f'{sample.end}'

    def solve(self) -> Result:
        generation_population_no = 100
        samples = self._random_samples(generation_population_no)

        best_sample = samples[0]
        best_result = float('inf')

        for generation in range(self.generations):
            if generation % 100 == 0:
                print(f'{generation} Generation')

            new_samples = samples
            self._ass(new_samples)

            # Mutations
            to_mutate_no = int(len(new_samples) * self.mutation_ratio)
            to_mutate = random.sample(new_samples, to_mutate_no)

            friend_mutation = map(self._mutate_friends, to_mutate[:to_mutate_no // 2])
            interval_mutation = map(self._mutate_interval, to_mutate[to_mutate_no // 2:])

            new_samples.extend(friend_mutation)
            new_samples.extend(interval_mutation)
            self._ass(new_samples)

            # Crossing
            for pair_no in range(len(new_samples) // 2):
                pair = random.sample(new_samples, 2)
                new_samples.append(self._cross(pair[0], pair[1]))

            self._ass(new_samples)

            # Chose the best 90% samples
            results = [(sample, self.cost_function(sample, self.data)) for sample in new_samples]
            new_samples = sorted(results, key=lambda sample: sample[1])

            new_samples = new_samples[:int(0.9 * generation_population_no)]

            if new_samples[0][1] < best_result:
                best_result = new_samples[0][1]
                best_sample = new_samples[0][0]

            new_samples = list(map(lambda x: x[0], new_samples))

            # Take 10 % of new random samples
            fresh_samples = self._random_samples(int(0.1 * generation_population_no))
            new_samples.extend(fresh_samples)

            samples = new_samples

        print(f"Cost Function = {best_result}")
        return best_sample