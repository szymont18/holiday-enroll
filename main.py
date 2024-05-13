from genetics.genetic_solver_2 import Genetic
from brute_force.brute_force import BruteForce
from interfaces.abstract_solution import AbstractSolution
from time import time


def test(solver_class: AbstractSolution, path: str):
    
    solver = solver_class(path)
    t = time()
    result = solver.solve()
    print(f"Elapsed time: {(time() - t):.2f}s")
    print(result)


if __name__ == '__main__':
    test(Genetic, "tests/test1.json")
    test(BruteForce, "tests/test1.json")

