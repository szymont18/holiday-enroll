from genetic_julek import GeneticJulek

if __name__ == '__main__':
    path_to_solve = "tests/kubak.json"
    solver = GeneticJulek(path_to_solve)
    result = solver.solve()
    print(str(result))



