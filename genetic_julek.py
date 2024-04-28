from abstract_solution import AbstractSolution
from data import Data
from result import Result


class GeneticJulek(AbstractSolution):
    def __init__(self, path):
        super().__init__(path)

    def solve(self) -> Result:
        data = super().read_data_from_json()
        res = Result(data.D1, data.D2, [])
        cost = super().cost_function(res, data)
        print(cost)
        return res