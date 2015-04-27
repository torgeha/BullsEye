__author__ = 'Olav'

#TODO: implement a super class, potensially abstract, so other game types can be implemented using base class.

class Classical:

    def __init__(self, players=2, score=501):
        self.players = [Player(i,501) for i in range(players)]
        self.current = 0


    def next_player(self):
        self.players[self.current].store_round()
        self.current = (self.current + 1)%len(self.players)

    def add_hit(self, score, x, y):
        self.players[self.current].add_hit((score, x,y))

    def get_leading_player(self):
        return min(self.players, key=lambda k: k.score)

    def get_game_standing(self):
        return [p.status() for p in self.players]
    
class Player:

    def __init__(self, id, start_score, name="None"):
        self.id = id
        self.name = name
        self.score = start_score
        self.rounds = []
        self.current_round = []

    def add_hit(self, data):
        self.current_round.append(data)
        self.score -= data[0]


    def add_miss(self):
        self.current_round.append((0, -1,-1))

    def store_round(self):
        while len(self.current_round) < 3:
            #Assume storing when there are less than 3
            #TODO: Logic in edge cases.
            self.add_miss()

        if len(self.current_round) == 3:
            self.rounds.append(self.current_round)
            self.current_round = []
        else:
            raise Exception("Should not be possible to store more than 3 per player")

    def status(self):
        return {"id": self.id, "name": self.name, "score": self.score}