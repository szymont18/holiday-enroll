D1 = 0  # 1 lipiec
D2 = 60  # 29 sierpień

# Ceny za każdy dzień
prices = [1000] * 10 + [1500] * 10 + [1250] * 20 + [2000] * 30

alpha = 1000  # Stała przy priorytecie (dotyczy ceny)
# Pmax = 20  # Maksymalna liczba priorytetów dla każdego użytkownika

Pmax_per_interval = 8  # Maksymalna liczba priorytetów na przedział

Fmax = 5  # Liczba miejsc w samochodzie (ile maks osób moze jechać na wakacje)
D = 5  # Minimalna liczba dni na jakie znajomi jadą na wakację


F = {
    'Alicja': {(0, 5): 7, (24, 31): 8, (32, 48): 3,  (51, 60): 2},
    'Benek': {(1, 12): 2, (13, 25): 8, (31, 40): 7, (51, 60): 3},
    'Celina': {(21, 34): 8, (39, 50): 4, (51, 60): 8},
    'Damian': {(11, 19): 3, (24, 31): 5, (32, 42): 4, (46, 57): 8},
    'Emilia': {(11, 18): 5, (24, 35): 8, (51, 60): 7},
    'Franek': {(21, 34): 4, (35, 51): 8, (52, 60): 8},
    'Grzesiek': {(11, 19): 3, (24, 31): 7, (32, 42): 2, (46, 57): 8}
}


"""
    Przykładowe rozwiązanie:
    Start = 52
    Koniec = 57
    Znajomi: Celina, Damina, Emilia, Franek, Grzesiek
    
    Priorytet = -273 + 12000 * 0.001 = -273 + 12 = -261

"""

class Result:
    def __init__(self, D1, D2, friends):
        self.start = D1
        self.end = D2
        self.friends = friends


# Priorytet zero zastąpimy czymś ujemnym (ale w granicach rozsądku)
def find_friend_interval_priority(friend, interval) -> int:
    pass


# Funkcja kosztu - do minimalizacji
def cost_function(result: Result):
    cost = 0

    # Priorytety koleżków
    for friend in result.friends:
        cost -= find_friend_interval_priority(friend, (result.start, result.end))

    # Ceny wakacji
    for day in range(result.start, result.end + 1):
        cost += prices[day] * alpha

    return cost