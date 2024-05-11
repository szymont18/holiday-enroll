import random
from random import randrange
import json

def generate_prices(D1: int, D2: int) -> list[int]:
    """
    generate price for each day
    """
    return [randrange(100, 1000) for _ in range(D1, D2 + 1)]


def generate_intervals(end_day: int, min_days: int, max_per_interval: int = 8) -> list[tuple[int, int, int]]:
    """
    generate intervals for a single friend
    """
    intervals = []
    curr_day = randrange(0, 11)
    while curr_day <= end_day:
        length = min_days
        length += randrange(0, 2)
        dice_roll = randrange(1, 11)  # 1 to 10 (D10)
        while dice_roll <= 8:  # 70%
            length += randrange(0, 2)
            dice_roll = randrange(1, 11)  # 1 to 10 (D10)

        priority = randrange(1, max_per_interval + 1)  # random priority
        intervals.append((curr_day, min(curr_day + length, end_day), priority))

        curr_day += length
        curr_day += randrange(1, 10)

    return intervals


def generate(
    D1: int,
    D2: int,
    max_per_interval: int,
    min_days: int,
    number_of_people: int
) -> list[list[int]]:
    """generate list of priorities for every friend"""

    F = []
    for _ in range(number_of_people):
        intervals = generate_intervals(D2, min_days, max_per_interval)
        F.append([-1000 for _ in range(D1, D2 + 1)])

        for start, end, priority in intervals:
            for i in range(start, end + 1):
                F[-1][i] = priority

    return F


def main() -> None:
    D1, D2 = 0, 60  # first and last day of holidays
    alpha = 0.001  # Constant factor included while computing loss
    max_per_interval = 8  # max priority on a single interval
    max_seats = 5  # Number of seats in a car
    min_days = 5  # minimum number of vacation days
    number_of_people = 30  # number of friends

    save_test = {
        "D1": D1,
        "D2": D2,
        "max_per_interval": max_per_interval,
        "alpha": alpha,
        "max_seats": max_seats,
        "min_days": min_days,
        "number_of_people": number_of_people
    }

    prices = generate_prices(D1, D2)
    save_test["prices"] = prices

    F = generate(D1, D2, max_per_interval, min_days, number_of_people)
    save_test["F"] = {}
    for i, friend in enumerate(F):
        save_test["F"][str(i)] = friend

    with open("../tests/test2.json", 'w') as f:
        json.dump(save_test, f)


# if __name__ == '__main__':
#     main()

friends = [str(i) for i in range(10)]
F = {f: [-1 for _ in range(61)] for f in friends}
for f in F.keys():
    print(f'"{f}": {F[f]},')
