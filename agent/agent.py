import os
import json
import numpy as np
import torch
from math import inf
from lux.game import Game

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else 'agent' # change to 'agent' for tests
model = torch.jit.load(f'{path}/model.pth')
model.eval()
model_city = torch.jit.load(f'{path}/model_city.pth')
model_city.eval()

from math import inf

def manhattan_distance(x1, y1, x2, y2):
    return (abs(x2-x1) + abs(y2-y1))

# make list of users and their current and previous coords and cooldowns 
# to write them in NN training subset
def find_user_coords(obs, units, x_shift, y_shift):
    # at first fill the unit dict with units from previous observation
    prev_update = obs['prev_update']
    if prev_update:
        for update in prev_update:
            strs = update.split(' ')
            input_identifier = strs[0]
            # if we found observation for user
            if input_identifier == 'u':
                unit_id = strs[3]
                x, y = 0, 0
                prev_x = int(strs[4]) + x_shift
                prev_y = int(strs[5]) + y_shift
                cooldown = 0
                prev_cooldown = float(strs[6])
                # flag of city building
                bcity = 0
                units[unit_id] = [x, y, prev_x, prev_y, cooldown, prev_cooldown, bcity]
        
    # then update it's coordinates and action with information from current update
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        # if we found observation for user
        if input_identifier == 'u':
            unit_id = strs[3]
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            cooldown = float(strs[6])
            # flag of city building
            bcity = 0
            # if unit isn't new and it's cooldown is changed from zero to nonzero, than means that 
            # action happened so we should write previous coords of unit, else don't change them
            if unit_id in units:
                prev_cooldown = units[unit_id][4]
                units[unit_id][0], units[unit_id][1] = x, y
                prev_x = units[unit_id][2]
                prev_y = units[unit_id][3]
                if cooldown > 0 and prev_cooldown == 0:
                    # if action is happened but coords don't change - that means that unit has built a city 
                    if x == prev_x and y == prev_y:
                        bcity = 1  
            else:
                prev_cooldown = cooldown
                prev_x = x
                prev_y = y
            
            units[unit_id] = [x, y, prev_x, prev_y, cooldown, prev_cooldown, bcity]
    return units


# Input for Neural Network for workers
def make_input(obs, unit_id):
    global units
    
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    
    units = {}
    cities = {}
    
    b = np.zeros((26, 32, 32), dtype=np.float32)
    
    units = find_user_coords(obs, units, x_shift, y_shift)
    x_u, y_u = units[unit_id][0], units[unit_id][1]
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        my_rp = 0
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if strs[3] == unit_id: # 0:2
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
                prev_x, prev_y = units[unit_id][2], units[unit_id][3]
                
                b[2, prev_x, prev_y] = 1
            else:                  # 3:10
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 3 + (team - obs['player']) % 2 * 4
                m_dist = manhattan_distance(x_u, y_u, x, y)
                b[idx:idx + 4, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100,
                    m_dist/((width-1) + (height-1))
                )
        elif input_identifier == 'ct':  # 11:16
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 11 + (team - obs['player']) % 2 * 3
            m_dist = manhattan_distance(x_u, y_u, x, y)
            b[idx:idx + 3, x, y] = (
                1,
                cities[city_id],
                m_dist/((width-1) + (height-1))
            )
        elif input_identifier == 'r':  # 17:20
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            access_level = {'wood': 0, 'coal': 50, 'uranium': 200}[r_type]
            access = 0 if my_rp < access_level else 1
            b[{'wood': 17, 'coal': 18, 'uranium': 19}[r_type], x, y] = amt / 800
            b[20, x, y] = access
        elif input_identifier == 'rp':  # 21:22
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            my_rp = rp if team == obs['player'] else my_rp
            b[21 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[23, :] = obs['step'] % 40 / 40
    # Turns
    b[24, :] = obs['step'] / 360
    # Map Size
    b[25, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b

# Input for Neural Network for cities
def make_city_input(obs, city_coord):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((20, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'ct':
            # CityTiles
            city_id = strs[2]
            x = int(strs[3]) 
            y = int(strs[4])
            cooldown = float(strs[5])
            if x == int(city_coord[0]) and y == int(city_coord[1]):
                b[:2, x + x_shift, y + y_shift] = (
                    1,
                    cities[city_id]
                )
            else:
                team = int(strs[1])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x + x_shift, y + y_shift] = (
                    1,
                    cooldown / 10,
                    cities[city_id]
                )
        elif input_identifier == 'u':
            team = int(strs[2])
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            idx = 8 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                (wood + coal + uranium) / 100
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[17, :] = obs['step'] % 40 / 40
    # Turns
    b[18, :] = obs['step'] / 360
    # Map Size
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b

game_state = None
player = None


def get_game_state(observation):
    global game_state
    
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation["player"]
    else:
        game_state._update(observation["updates"])
    return game_state


# check if unit is in city or not
def in_city(pos):    
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False
    
# check if unit has enough time and space to build a city
def build_city_is_possible(unit, pos):    
    global game_state
    global player

    if game_state.turn % 40 < 30:
        return True
    x, y = pos.x, pos.y
    for i, j in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
        try:
            city_id = game_state.map.get_cell(i, j).citytile.cityid
        except:
            continue
        if city_id in player.cities:
            city = player.cities[city_id]
            if city.fuel > (city.get_light_upkeep() + 18) * 10:
                print('True')
                return True
    return False


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)


# translate unit policy to action
unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]
def get_unit_action(policy, unit, dest):
    for label in np.argsort(policy)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos
        if label == 4 and not build_city_is_possible(unit, pos):
            return unit.move('c'), unit.pos
        if pos not in dest or in_city(pos):
            return call_func(unit, *act), pos 
            
    return unit.move('c'), unit.pos

# translate city policy to action
city_actions = [('build_worker',), ('research', )]
def get_city_action(policy, city_tile, unit_count):
    global player
    
    for label in np.argsort(policy)[::-1]:
        act = city_actions[label]
        if label == 0 and unit_count < player.city_tile_count:
            unit_count += 1
            res = call_func(city_tile, *act)
        elif label == 1 and not player.researched_uranium():
            player.research_points += 1
            res = call_func(city_tile, *act)
        else:
            res = None
        return res, unit_count

# agent for making actions
def agent(observation, configuration):
    global game_state
    global player
    
    game_state = get_game_state(observation)    
    player = game_state.players[observation.player]
    actions = []        
    
    with open(f'{path}/tmp.json') as json_file:
        try:
            prev_obs = json.load(json_file)
        except json.decoder.JSONDecodeError:
            prev_obs = dict()
            prev_obs['prev_update_lag_1'] = None
            prev_obs['prev_update_lag_2'] = None
            prev_obs['prev_update_lag_3'] = None
            prev_obs['prev_update_lag_4'] = None
        
    observation['prev_update'] = None
    
    if prev_obs:
        if prev_obs['prev_update_lag_2']:
            observation['prev_update'] = prev_obs['prev_update_lag_2']      
            
    prev_obs['prev_update_lag_4'] = prev_obs['prev_update_lag_3']
    prev_obs['prev_update_lag_3'] = prev_obs['prev_update_lag_2']
    prev_obs['prev_update_lag_2'] = prev_obs['prev_update_lag_1']
    prev_obs['prev_update_lag_1'] = observation['updates']
    
    if game_state.turn < 359:
        with open(f'{path}/tmp.json', 'w+') as json_file:
            json.dump(prev_obs, json_file)
    else:
        open(f'{path}/tmp.json', 'w+').close()
    
    # Unit Actions
    dest = []
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            state = make_input(observation, unit.id)
            with torch.no_grad():
                p = model(torch.from_numpy(state).unsqueeze(0))

            policy = p.squeeze(0).numpy()

            action, pos = get_unit_action(policy, unit, dest)
            actions.append(action)
            dest.append(pos)
    
    # City Actions
    unit_count = len(player.units)
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                state = make_city_input(observation, [city_tile.pos.x, city_tile.pos.y])
                with torch.no_grad():
                    p = model_city(torch.from_numpy(state).unsqueeze(0))

                policy = p.squeeze(0).numpy()

                action, unit_count = get_city_action(policy, city_tile, unit_count)
                if action:
                    actions.append(action)
    
    return actions
