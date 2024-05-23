from genetics.genetic_solver_2 import Genetic
from genetics.genetic_solver import GeneticSolver
from brute_force.brute_force import BruteForce
from let_it_bee.bee_solver import BeeSolver
from interfaces.abstract_solution import AbstractSolution
from České_žíhání.annealing import Annealing
from time import time


# def test(solver_class: AbstractSolution, path: str):
#
#     solver = solver_class(path)
#     t = time()
#     result = solver.solve()
#     print(f"Elapsed time: {(time() - t):.2f}s")
#     print(result)


if __name__ == '__main__':
    # test(Genetic, "tests/test1.json")
    # test(BruteForce, "tests/test1.json")
    while True:
        path = input("podaj ścieżkę do pliku z wejściem do algorytmu (np. tests/test1.json): ")
        algorithm = input("wybierz algorytm:\n\t-brute-force\n\t-genetyczny 1\n\t-genetyczny 2\n\t-pszczeli\n\t-czeskie wyżarzanie\n")
        keep_this_file = True
        while keep_this_file:
            match algorithm.strip():
                case "brute-force":
                    res, _ = BruteForce(path).solve()
                    print(f"ZNALEZIONA ODPOWIEDŹ:\n{res}")
                case "genetyczny 1":
                    genno = int(input("podaj liczbę generacji (domyślnie 10000)"))
                    mutsampl = float(input("podaj rozmiar mutacji (domyślnie 0.1)"))
                    res, _ = GeneticSolver(path, generation_no= genno, mutation_sample= mutsampl).solve()
                    print(f"ZNALEZIONA ODPOWIEDŹ:\n{res}")
                case "genetyczny 2":
                    genno = int(input("podaj liczbę generacji (domyślnie 10000)"))
                    pop_size = int(input("podaj rozmiar populacji (domyślnie 100)"))
                    res, _ = Genetic(path, generation_no= genno, population_size=pop_size).solve()
                    print(f"ZNALEZIONA ODPOWIEDŹ:\n{res}")
                case "pszczeli":
                    genno = int(input("podaj liczbę generacji (domyślnie 3000)"))
                    pop_size = int(input("podaj rozmiar populacji (domyślnie 200)"))
                    maxgens = int(input("maksymalna liczba generacji w której może nie być poprawy (domyślnie 7)"))
                    res, _ = BeeSolver(path, generation_no=genno, population_size=pop_size, max_gens_without_improvement=maxgens).solve()
                    print(f"ZNALEZIONA ODPOWIEDŹ:\n{res}")
                case "czeskie wyżarzanie":
                    iters = int(input("podaj liczbę iteracji (domyślnie 10000)"))
                    steps = int(input("podaj liczbę kroków w jedenj epoce (domyślnie 100)"))
                    initemp = int(input("podaj początkową temperaturę (domyślnie 10000)"))
                    alph = float(input("podaj współczynnik aplha (domyślnie 0.999"))
                    res, _ = Annealing(path, iterations=iters, steps_in_one_epoch=steps, initial_temperature=initemp, alpha=alph).solve()
                    print(f"ZNALEZIONA ODPOWIEDŹ:\n{res}")
                case _:
                    print("taki algorytm nie instnieje")
            keep_this_file = False if input("czy zmieniamy testowany plik? ").strip().lower() == "tak" else True
            if keep_this_file:
                algorithm = input("wybierz nowy algorytm:")


