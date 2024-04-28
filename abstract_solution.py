from abc import ABC, abstractmethod
import json
from Data import Data
from Result import Result


class AbstractSolution(ABC):
    def __init__(self, path_to_solve:str):
        self.path_to_solve = path_to_solve
        pass

    def read_data_from_json(self) -> Data:
        with open(self.path_to_solve, "r") as file:
            return Data(json.load(file))

    @abstractmethod
    def solve(self) -> Result:
        pass

