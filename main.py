from genetic_julek import GeneticJulek
from genetics.genetic_solver import GeneticSolver
from genetics.genetic_solver_2 import Genetic

# if __name__ == '__main__':
#     path_to_solve = "tests/kubak.json"
#     solver = GeneticJulek(path_to_solve)
#     result = solver.solve()
#     print(str(result))

#
if __name__ == '__main__':
    path_to_solve = "tests/test1.json"
    solver = Genetic(path_to_solve)
    result = solver.solve()
    print(str(result))


