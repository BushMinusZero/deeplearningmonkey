import random
import numpy as np
from typing import List, Tuple


class GridWorld:

    def __init__(self, dimensions: Tuple[int, int], vertical_walls: List[int], horizontal_walls: List[int],
                 reward_states: List[Tuple[int, int]], default_reward: int):

        # Define the world
        self.dimensions = dimensions
        self.total_positions = dimensions[0] * dimensions[1]
        self.grid = np.array(range(self.total_positions))
        self.vertical_walls = vertical_walls
        self.horizontal_walls = horizontal_walls
        self.rewards = np.ones_like(self.grid) * default_reward
        for s, r in reward_states:
            self.rewards[s] = r

        # Define possible actions
        self.actions = ["left", "right", "up", "down"]
        self.possible_actions_by_state = self.compute_possible_actions_by_state()

    def compute_possible_actions_by_state(self):
        possible_actions_by_state = []
        for s in self.grid:
            actions = []
            if s < self.total_positions - self.dimensions[0] and s not in self.horizontal_walls:
                actions.append("down")
            if s >= self.dimensions[0] and s - 4 not in self.horizontal_walls:
                actions.append("up")
            if s % 4 != 0 and s - 1 - s // 4 not in self.vertical_walls:
                actions.append("left")
            if s % 4 != 3 and s - s // 4 not in self.vertical_walls:
                actions.append("right")
            possible_actions_by_state.append(actions)
        return possible_actions_by_state

    def __repr__(self):
        grid = np.reshape(self.grid, self.dimensions)
        rewards = np.reshape(self.rewards, self.dimensions)
        return f"Grid: \n{grid}, \nRewards: \n{rewards}"

    def transition_probability(self):
        pass

    def iterate_value(self) -> List[int]:
        pass

    def value_iteration(self, num_iterations: int) -> List[int]:

        for k in range(num_iterations):
            self.iterate_value()


def main():

    # Initialize world
    dimensions = (4, 4)
    vertical_walls = []
    horizontal_walls = [0, 1, 2, 9, 10, 11]
    reward_states = [(4, -5), (11, 5)]
    default_reward = -1
    world = GridWorld(dimensions, vertical_walls, horizontal_walls, reward_states, default_reward)
    print(world)
    print(world.possible_actions_by_state)


main()
