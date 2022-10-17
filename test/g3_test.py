from logging import Logger
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point
import sys

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_folder)


import math

from players.g3_player import (
	midsort,
	get_moves,
	repelling_force_sum,
    get_base_angles,
    DensityMap,
    Player,
    Role,
    RoleTemplate,
    Scouts,
    DefaultSoldier,
    MacroArmy,
    ResourcePool
)


# -----------------------------------------------------------------------------
# 	Debug Tests
# -----------------------------------------------------------------------------

debug = 0

def debug_densitymap():
    me = 1
    unit_pos = [
        [[57, 40.5], [56, 43.5]],
        [
            [53, 42.5], [53, 45], [53, 48.5],
            [58, 42], [58, 45], [58, 47.5]
        ]
    ]

    d = DensityMap(me, unit_pos)

    grid_id = d.pt2grid(unit_pos[0][0][0], unit_pos[0][0][1])
    troops = d.soldier_partitions[grid_id]

    plt.gca().invert_yaxis()

    for (x, y), pid in troops:
        pt_format = 'bo' if pid == me else 'ro'
        plt.plot(x, y, pt_format)

    next_moves = [
        d.suggest_move(ally_pos)
        for ally_pos in unit_pos[me]
    ]

    print(next_moves)

    next_pos = [
        [x + dist * np.cos(angle), y + dist * np.sin(angle)]
        for (x, y), (dist, angle) in zip(unit_pos[me], next_moves)
    ]

    print(next_pos)

    for (x, y) in next_pos:
        plt.plot(x, y, 'go')


    plt.show()


if debug:
    debug_densitymap()


# -----------------------------------------------------------------------------
# 	Unit Tests
# -----------------------------------------------------------------------------

def test_midsort():
    cases = [
        {
            'name': 'array_of_odd_number_of_elements',
            'array': [1, 2, 3, 4, 5],
            'expect': [3, 2, 4, 1, 5]
        },
        {
            'name': 'corner_case_empty_array',
            'array': [],
            'expect': []
        },
        {
            'name': 'array_of_single_element',
            'array': [10],
            'expect': [10]
        },
        {
            'name': 'array_of_2_elements',
            'array': [10, 30],
            'expect': [10, 30]
        },
        {
            'name': 'array_of_even_number_of_elements',
            'array': [10, 78, 290, 208, 284, 285, 203, 173],
            'expect': [208, 78, 285, 10, 290, 284, 203, 173]
        }
    ]

    error_count = 0

    for tc in cases:
        got = midsort(tc['array'])

        if got != tc['expect']:
            print(f'case {tc["name"]} failed:')
            print(f'expect: {tc["expect"]}')
            print(f'got: {got}\n')
            error_count += 1
        
    
    if error_count == 0:
       print("PASSED - test_midsort")
    else:
       print(f"FAILED with {error_count} errors - test_midsort")
        

def test_get_moves():
    cases = [
        {
            "unit_pos": [[0, 0], [0, 0], [0, 0]],
            "target_loc": [[1, 1], [1, 4], [2, 5]]
        }
    ]
    
    result = get_moves(cases[0]['unit_pos'], cases[0]['target_loc'])
    print("radians: " + str(result))

    for i in range(len(result)):
        result[i] = [result[i][0], result[i][1] * 180 / math.pi]
    
    print("degrees: " + str(result))


def test_repelling_force_sum():
	cases = [
		{
			'name': 'simple',
			'pts': [(0., 2.), (2., 0.)],
			'receiver': (0., 0.),
			'expect': [-.5, -.5]
		}
	]

	error_count = 0

	for tc in cases:
		got = repelling_force_sum(tc['pts'], tc['receiver'])
		 
		if not (got == tc['expect']).all():
			print(f'case {tc["name"]} failed:')
			print(f'expect: {tc["expect"]}')
			print(f'got: {got}\n')
			error_count += 1
    
	if error_count == 0:
		print("PASSED - test_repelling_force_sum")
	else:
		print(f"FAILED with {error_count} errors - test_repelling_force_sum")


def test_get_base_angles():
    cases = [
		{
			'name': 'player_1',
            'player_idx': 0,
			'expect': (math.pi/2, 0)
		},
		{
			'name': 'player_2',
            'player_idx': 1,
			'expect': (0, -math.pi/2)
		},
		{
			'name': 'player_3',
            'player_idx': 2,
			'expect': (-math.pi/2, -math.pi)
		},
		{
			'name': 'player_4',
            'player_idx': 3,
			'expect': (-math.pi, -math.pi*3/2)
		}
	]

    error_count = 0
    for tc in cases:
        got = get_base_angles(tc['player_idx'])
		 
        if not got == tc['expect']:
            print(f'case {tc["name"]} failed:')
            print(f'expect: {tc["expect"]}')
            print(f'got: {got}\n')
            error_count += 1
    
    if error_count == 0:
        print("PASSED - test_get_base_angles")
    else:
        print(f"FAILED with {error_count} errors - test_get_base_angles")


def test_role_instance_impl():
    """Tests if role implementation adheres to the Role ABC."""

    rng = np.random.default_rng()
    logger = Logger('test_role_instance_impl')
    homebase = Point(.5, .5)
    player = Player(rng, logger, 50, 3, 0, homebase, 100, 100, '')
    _resource = ResourcePool(player)

    assert issubclass(RoleTemplate, Role), "  FAILED - RoleTempate not a subclass of Role"
    print("  PASSED - RoleTemplate is a subclass of Role")

    default = DefaultSoldier(logger, player, 'default1', [])
    assert isinstance(default, Role), "  FAILED - default :: DefaultSoldier not an instance of Role"
    print("  PASSED - default :: DefaultSoldier is an instance of Role")

    scouts = Scouts(logger, player, 'scouts1', 10)
    assert isinstance(scouts, Role), "  FAILED - scouts :: Scouts not an instance of Role"
    print("  PASSED - scouts :: Scouts is an instance of Role")

    macro_army = MacroArmy(logger, player, 'macroArmy1', _resource)
    assert isinstance(macro_army, Role), "  FAILED - macro_army :: MacroArmy not an instance of Role"
    print("  PASSED - macro_army :: MacroArmy is an instance of Role")

    print("PASSED - test_role_instance_impl")


# -----------------------------------------------------------------------------
# 	Running Tests...
# -----------------------------------------------------------------------------

test_midsort()
test_get_moves()
test_repelling_force_sum()
test_get_base_angles()

test_role_instance_impl()