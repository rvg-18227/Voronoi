import os
import sys

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_folder)


import math

from players.g3_player import (
	midsort,
	get_moves
)


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


# -----------------------------------------------------------------------------
# 	Running Tests...
# -----------------------------------------------------------------------------

test_midsort()
test_get_moves()