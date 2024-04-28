class Data():
    def __init__(self, data):
        self.D1 = data['D1']
        self.D2 = data['D2']
        self.alpha = data['alpha']
        self.pmax_per_interval = data['Pmax_per_interval']
        self. fmax = data['Fmax']
        self.D = data['D']
        self.prices = data['prices']
        self.F = self.handle_f(data["F"])

    def handle_f(self, f:dict):
        result = {}
        for person in f.keys():
            person_preference = f[person]
            new_person = {}
            for interval in person_preference.keys():
                val = person_preference[interval]
                new_key = tuple([int(i.strip()) for i in interval.replace("(", "").replace(")", "").split(",")])
                new_person[new_key] = val
            result[int(person)] = new_person
        return result
