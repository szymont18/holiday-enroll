class Data():
    def __init__(self, data):
        self.D1 = data['D1']
        self.D2 = data['D2']
        self.alpha = data['alpha']
        self.pmax_per_interval = data['max_per_interval']
        self.fmax = data['max_seats']
        self.D = data['min_days']
        self.number_of_people = data['number_of_people']
        self.prices = data['prices']

        # TODO: Change it to another representation of priorities(Szymon Version)
        # self.F = self.handle_f(data["F"])

        self.F = data['F']

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
