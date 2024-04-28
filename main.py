from genetic_julek import GeneticJulek
from genetics.genetic_solver import GeneticSolver

# if __name__ == '__main__':
#     path_to_solve = "tests/kubak.json"
#     solver = GeneticJulek(path_to_solve)
#     result = solver.solve()
#     print(str(result))

#
if __name__ == '__main__':
    path_to_solve = "tests/test1.json"
    solver = GeneticSolver(path_to_solve)
    result = solver.solve()
    print(str(result))


