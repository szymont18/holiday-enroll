class Result:
    def __init__(self, D1:int, D2:int, friends:list):
        self.start = D1
        self.end = D2
        self.friends = friends

    def __str__(self):
        return f"start: {self.start}\nend: {self.end}\nfriends: {self.friends}"