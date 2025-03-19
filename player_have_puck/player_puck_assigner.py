import sys
#sys.path.append('../')
from utils import get_center_of_boxx, get_distance

class PlayerPuckAssigner():
    
    def __init__(self):
        self.max_distance = 50

    def assign_player_puck(self, players, puck):

        puck_coord = get_center_of_boxx(puck)
        
        min_dist = self.max_distance
        player_puck = None
        for player_id, player in players.items():
            player_box = player['bbox']
            dist_left = get_distance(puck_coord, (player_box[0], player_box[3]))
            dist_right = get_distance(puck_coord, (player_box[2], player_box[3]))
            cur_dist = min(dist_left, dist_right)
            if cur_dist <= min_dist:
                min_dist = cur_dist
                player_puck = player_id
            
        return player_puck