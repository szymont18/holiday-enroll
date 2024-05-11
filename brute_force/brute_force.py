import numpy as np
from result import Result
from abstract_solution import AbstractSolution
from data import Data

class BruteForce(AbstractSolution):
    def __init__(self, path_to_solve: str):
        super().__init__(path_to_solve)
        self.data : Data = self.read_data_from_json()

    
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
    
    def _generate_next_idxs(self, friends_idx):
        counter = 0
        prev = len(self.data.F)
        for i in range(len(friends_idx) - 1, -1, -1):
            if friends_idx[i] == prev - 1:
                counter += 1
                prev = friends_idx[i] 
                if i < len(friends_idx) - counter:
                    friends_idx[i] = 0
                    for j in range(i + 1, i + counter + 1):
                        friends_idx[j] = 1
                    return True
                else:
                    friends_idx[i] = 0
            else:
                friends_idx[i] += 1
                for j in range(i+1, len(friends_idx)):
                    friends_idx[j] = friends_idx[j-1] + 1

                    
        return False
    
    def find_friend_interval_priority(self, friend, interval):
        priority = 0
        for p in self.data.F[friend][interval[0]:interval[1] + 1]:
            priority += p

        return priority
    

    
    def solve(self):
        data = self.data
        best_result : Result = Result(None, None, None)
        best_cost = np.inf

        friends_taken = data.fmax
        friends = list(data.F.keys())
        prority_per_friend = [[[-1 for _ in range(data.D2)] for _ in range(data.D2 - data.D)] for _ in friends]

        print('preprocessing start')
        for i, friend in enumerate(friends):
            for start in range(data.D1, data.D2 - data.D):
                for end in range(start + data.D, data.D2):
                    prority_per_friend[i][start][end] = self.find_friend_interval_priority(friend, (start, end))


        print('preprocessing end')

        empty_seat_penality = 0
        while friends_taken > 0:
            print(f'friends_taken {friends_taken}')
            friends_mask = np.array([1 if i < friends_taken else 0 for i in range(len(friends))])
            counter = 1
            while True:
                friends_idx = np.where(friends_mask == 1)[0]
                for start in range(data.D1, data.D2 - data.D):
                    for end in range(start + data.D, data.D2):

                        cost = sum(data.prices[start:end + 1]) * data.alpha
                        
                        for idx in friends_idx:
                            cost -= prority_per_friend[idx][start][end]
                        cost += empty_seat_penality
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
            empty_seat_penality += 1000
            print('')
        print(best_cost)
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




