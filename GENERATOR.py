D1 = 0  # 1 lipiec
D2 = 60  # 29 sierpień

# Ceny za każdy dzień
# prices = [1000] * 10 + [1500] * 10 + [1250] * 20 + [2000] * 30

alpha = 1000  # Stała przy priorytecie (dotyczy ceny)
# Pmax = 20  # Maksymalna liczba priorytetów dla każdego użytkownika

Pmax_per_interval = 8 # Maksymalna liczba priorytetów na przedział

Fmax = 5  # Liczba miejsc w samochodzie (ile maks osób moze jechać na wakacje)
D = 5  # Minimalna liczba dni na jakie znajomi jadą na wakację

NUMBER_OF_PERSONS = 30


from random import randrange


def generate(D2):

    def generate_prices():
        return [randrange(100, 1000) for _ in range(D2 + 1)]

    def generate_interval():
        curr_day = randrange(0, 11)
        preference = dict()
        while curr_day <= D2:
            length = D
            length += randrange(0, 2)
            dice_roll = randrange(1, 11)  # 1 to 10 (D10)
            while dice_roll <= 8:  # 70%
                length += randrange(0, 2)
                dice_roll = randrange(1, 11)  # 1 to 10 (D10)

            preference[(curr_day, min(curr_day + length, D2))] = randrange(1, 9)  # * length?    # random priority
            curr_day += length
            curr_day += randrange(1, 10)
        return preference

    prices = generate_prices()

    F = dict()
    for person in range(NUMBER_OF_PERSONS):
        F[person] = generate_interval()

    return F, prices


F, prices = generate(D2)
print(prices)
print()
for item in F.items():
    print(item)

