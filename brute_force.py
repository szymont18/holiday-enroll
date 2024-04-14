import numpy as np
from tests.kubak import params
from test1 import Result

class BruteForce():
    def __init__(self, cost_function):
        self.cost_function = cost_function
    
    def _generate_next_mask(self, friends_mask):
        counter = 0
        for i in range(len(friends_mask) - 1, -1, -1):
            if friends_mask[i]:
                counter += 1
                if i < len(friends_mask) - counter:
                    friends_mask[i] = 0
                    for j in range(i + 1, i + counter + 1):
                        friends_mask[j] = 1
                    return True
                else:
                    friends_mask[i] = 0
                    
        return False
    
    def _check_friend_availability(day, friends_table) -> int:
        for interval in friends_table.keys():
            if interval[0] <= day and interval[1] >= day:
                return friends_table[interval]
        return -1
    

    
    def solve_enroll(self, D1, D2, prices, alpha, Pmax_per_interval, Fmax, D, F):
        best_result : Result = Result(None, None, None)
        best_cost = np.inf

        friends_taken = Fmax
        friends = list(F.keys())
        prority_per_friend = [[[-1 for _ in range(D2)] for _ in range(D2 - D)] for _ in friends]

        print('preprocessing start')
        for i, friend in enumerate(friends):
            for start in range(D1, D2-D):
                for end in range(start+D, D2):
                    prority_per_friend[i][start][end] = self.cost_function(friend, F, start, end)


        print('preprocessing end')


        while friends_taken > 0:
            print(f'friends_taken {friends_taken}')
            friends_mask = np.array([1 if i < friends_taken else 0 for i in range(len(friends))])
            counter = 1
            while True:
                friends_idx = np.where(friends_mask == 1)[0]
                for start in range(D1, D2 - D):
                    for end in range(start + D, D2):
                        cost = sum(prices[start:end + 1])
                        for idx in friends_idx:
                            cost -= prority_per_friend[idx][start][end]
                        if cost < best_cost:
                            # print(best_result.start, best_result.end, best_result.friends)
                            best_result = Result(start, end, friends_idx.copy())
                            best_cost = cost

                if not self._generate_next_mask(friends_mask):
                    break
                if not counter % 100:
                    print('|', end='')
                counter += 1
            friends_taken -= 1
            print('')
        return best_result

def cost_f(friend, F, start, end):
    for interval in F[friend].keys():
        if interval[0] <= start and interval[1] >= start:
            break
    else:
        return -1000
    
    if interval[1] >= end:
        return F[friend][interval]
    return -1000

solver = BruteForce(cost_f)

result = solver.solve_enroll(**params)

print(params["F"])
print(result.start, result.end, result.friends)


