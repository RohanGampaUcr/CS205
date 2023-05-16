import math
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
from random import *



def heuristic_manhattan(state, goal):
    """calculates the manhattan distance from the current state to the goal"""
    # Your code here.
    manhattan_dist = 0
    for i, j in np.ndindex(state.shape):
        idx = np.argwhere(goal == state[i][j])
        manhattan_dist += abs(i - idx[0][0]) + abs(j - idx[0][1])
    return manhattan_dist


def heuristic_misplaced(state, goal):
    """calculates the number of misplaced tiles from current state to the goal"""
    misplaced_dist = sum(1 for i, row in enumerate(state) for j, value in enumerate(row) if value != goal[i][j])
    return misplaced_dist

def heuristic_ucs(state, goal):
    return 0


def adjacent_states(state):
    """what are all the successors of this state? depends on location of the 0 (blank tile)"""
    adjacent = []
    loc_empty = np.argwhere(state == 0)[0]

    for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        newloc = loc_empty + np.array([drow, dcol])
        if 0 <= newloc[0] < state.shape[0] and 0 <= newloc[1] < state.shape[1]:
            swap = np.copy(state)
            swap[loc_empty[0], loc_empty[1]], swap[newloc[0], newloc[1]] = swap[newloc[0], newloc[1]], swap[loc_empty[0], loc_empty[1]]
            adjacent.append(swap)

    return adjacent


def check_parity(state1, state2):
    """checks to see if the parity of the two states is the same
    the 8-tile problem will not be solvable if they are not"""
    state1_copy = state1[state1 != 0]
    state2_copy = state2[state2 != 0]

    pair1 = sum(state1_copy[i] > state1_copy[j] for i in range(len(state1_copy) - 1) for j in range(i + 1, len(state1_copy)))
    pair2 = sum(state2_copy[i] > state2_copy[j] for i in range(len(state2_copy) - 1) for j in range(i + 1, len(state2_copy)))

    return pair1 % 2 == pair2 % 2


class PriorityQueue:
    """
    Priority queue for A-star search
    """

    def __init__(self, start, cost):
        self.states = {}
        self.q = []
        self.add(start, cost)

    def add(self, state, cost):
        """push the new state and cost to get there onto the heap"""
        heapq.heappush(self.q, (cost, state))
        self.states[state] = cost

    def pop(self):
        if self.q:
            cost, state = heapq.heappop(self.q)
            self.states.pop(state)
            return cost, state
        else:
            return None, None

    def replace(self, state, cost):
        oldcost = self.states.get(state, None)
        if oldcost is None or cost < oldcost:
            self.states[state] = cost
            for i, (old_cost, old_state) in enumerate(self.q):
                if old_state == state:
                    self.q[i] = (cost, state)
                    heapq._siftup(self.q, i)




def astar_search(start, goal, heuristic):
  """
  Performs A* search on a 2D grid.

  Args:
    start: The starting state.
    goal: The goal state.
    heuristic: A heuristic function that estimates the cost of reaching the goal from a given state.

  Returns:
    The path from the starting state to the goal state, or None if no path exists.
  """

  start_time = time.time()

  if not check_parity(start, goal):
    return None

  temp = np.copy(start)
  x, y = start.shape

  frontier = PriorityQueue(tuple(temp.reshape(x * y)), heuristic(start, goal))

  previous = {tuple(temp.reshape(x * y)): None}
  explored = dict()
  expand = 0

  while frontier:
    s = frontier.pop()
    s_array = np.array(list(s[1])).reshape(x, y)
    expand += 1

    if (s_array == goal).all():
      end_time = time.time()
      return previous, expand, end_time - start_time
    
    explored[s[1]] = s[0]
    for s2 in adjacent_states(s_array):
      temp = np.copy(s2)
      s2_tuple = tuple(temp.reshape(x * y))
      newcost = explored[s[1]] + 1 + heuristic(s2, goal)

      if s2_tuple not in explored and s2_tuple not in frontier.states:
        frontier.add(s2_tuple, newcost)
        previous[s2_tuple] = s[1]

      elif s2_tuple in frontier.states and frontier.states[s2_tuple] > newcost:
        frontier.replace(s2_tuple, newcost)
        previous[s2_tuple] = s[1]

  return None




def print_moves(moves, goal, puzzle_len, mode):
    """Prints the moves taken to reach the goal state.

    Args:
        moves: A list of states, where each state is a 2D array.
        goal: The goal state, as a 2D array.
        puzzle_len: The length of each side of the puzzle.
        mode: The heuristic function to use.

    """

    # Convert the goal state to a tuple.
    tmp_goal = goal.copy()
    goal = tuple(goal.reshape(puzzle_len * puzzle_len))

    # Create a list of all states.
    res = []
    while goal:
        res.append(goal)
        goal = moves[goal]

    # Reverse the list of states.
    res.reverse()

    # Iterate over the list of states.
    cnt = 0
    for sta in res:

        # Convert the state to a NumPy array.
        tmp = np.array(sta).reshape(puzzle_len, puzzle_len)

        # Get the heuristic function value.
        if mode == 1:
            hn = heuristic_ucs(tmp, tmp_goal)
        elif mode == 2:
            hn = heuristic_misplaced(tmp, tmp_goal)
        else:
            hn = heuristic_manhattan(tmp, tmp_goal)

        # Print the current step, the heuristic function value, and the state.
        print(f"Current Iteratioon: {cnt} and Current Heuristic Function cost: {hn}")
        print(tmp)
        print()

        cnt += 1




if __name__ == "__main__":
    eight_goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    initial_state= [
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 0]],

    [[1, 2, 3],
     [4, 5, 6],
     [7, 0, 8]],

    [[1, 2, 0],
     [4, 5, 3],
     [7, 8, 6]],

    [[0, 1, 2],
     [4, 5, 3],
     [7, 8, 6]],

    [[8, 7, 1],
     [6, 0, 2],
     [5, 4, 3]],

    [[8, 7, 1],
     [6, 0, 2],
     [5, 4, 3]] ]
    
    #pick the initial state here 
    difficulty = 3
    user_puzzle = ( initial_state[difficulty])
    target = eight_goal_state
    puzzle_len = 3

    h = input("Select algorithm.\n (1) for Uniform Cost Search,\n (2) for the Misplaced Tile Heuristic, or\n"
                      " (3) the Manhattan Distance Heuristic.\n")
    if h == '1':
        heuristic = heuristic_ucs
    elif h == '2':
        heuristic = heuristic_misplaced
    else:
        heuristic = heuristic_manhattan
        
    user_puzzle = np.array(user_puzzle)
    target = np.array(target)
    target = target.reshape(puzzle_len, puzzle_len)
    print("Initail state:\n")
    print(target)
    print()

    moves, steps, nexp = astar_search(user_puzzle, target, heuristic)
    print_moves(moves, target, puzzle_len, h)
    print(steps)


   
