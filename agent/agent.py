import os
import numpy as np
import torch
from lux.game import Game

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.' # change to 'agent' for tests
model = torch.jit.load(f'{path}/model.pth')
model.eval()
model_city = torch.jit.load(f'{path}/model_city.pth')
model_city.eval()

# Input for Neural Network for units
def make_input(obs, unit_id):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((20, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100
                )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id]
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
        print(f'City ID - {city_id}')
        if city_id in player.cities:
            city = player.cities[city_id]
            print(f'City fuel - {city.fuel}, City light - {city.get_light_upkeep()}')
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
                # at first game stages try to produce maximum amount of agents and research point
                if game_state.turn < 60:
                    if unit_count < player.city_tile_count: 
                        actions.append(city_tile.build_worker())
                        unit_count += 1
                    elif not player.researched_uranium() and game_state.turn > 40:
                        actions.append(city_tile.research())
                        player.research_points += 1
                # then follow NN strategy
                else:
                    state = make_city_input(observation, [city_tile.pos.x, city_tile.pos.y])
                    with torch.no_grad():
                        p = model_city(torch.from_numpy(state).unsqueeze(0))

                    policy = p.squeeze(0).numpy()

                    action, unit_count = get_city_action(policy, city_tile, unit_count)
                    if action:
                        actions.append(action)
    
    return actions
