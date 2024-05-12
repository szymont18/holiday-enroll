from abc import ABC, abstractmethod
import json
from data import Data
from result import Result


class AbstractSolution(ABC):
    def __init__(self, path_to_solve:str):
        self.path_to_solve = path_to_solve
        pass

    def read_data_from_json(self) -> Data:
        with open(self.path_to_solve, "r") as file:
            return Data(json.load(file))

    def find_friend_interval_priority(self, friend, interval) -> int:
        pass
    def cost_function(self, result: Result, data:Data):
        cost = 0

        # Priorytety koleżków
        for friend in result.friends:
            cost -= self.find_friend_interval_priority(friend, (result.start, result.end))

        # Ceny wakacji
        for day in range(result.start, result.end + 1):
            cost += data.prices[day] * data.alpha

        return cost
    @abstractmethod
    def solve(self) -> Result:
        pass

