import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Constants for the grid
EMPTY, PLAYER, TREASURE, WALL = 0, 1, 2, 3
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
SIZE = 6


class GridEnvironment_1:
    def __init__(self):
        self.grid = None
        self.player_position = None
        self.treasure_position = None
        self.reset()

    def reset(self):
        # Define the static part of the grid
        self.grid = np.array([
            [EMPTY, WALL, EMPTY, EMPTY, WALL, WALL],
            [EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY],
            [EMPTY, EMPTY, WALL, EMPTY, WALL, EMPTY],
            [WALL, EMPTY, WALL, EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY]
        ])

        # Find indices of all empty tiles
        empty_tiles = np.argwhere(self.grid == EMPTY)

        # Randomly select two different positions from the list of empty tiles
        player_idx, treasure_idx = np.random.choice(len(empty_tiles), 2, replace=False)

        self.player_position = tuple(empty_tiles[player_idx])
        self.treasure_position = tuple(empty_tiles[treasure_idx])

        # Place player and treasure on the grid
        self.grid[self.player_position] = PLAYER
        self.grid[self.treasure_position] = TREASURE

        return np.copy(self.grid)

    def step(self, action):
        # Calculate new position based on action
        x, y = self.player_position
        if action == UP:
            x -= 1
        elif action == DOWN:
            x += 1
        elif action == LEFT:
            y -= 1
        elif action == RIGHT:
            y += 1

        # Check boundaries and walls
        if 0 <= x < SIZE and 0 <= y < SIZE and self.grid[x, y] != WALL:
            new_position = (x, y)
            old_distance = np.abs(self.player_position[0] - self.treasure_position[0]) + np.abs(self.player_position[1] - self.treasure_position[1])
            new_distance = np.abs(new_position[0] - self.treasure_position[0]) + np.abs(new_position[1] - self.treasure_position[1])

            self.grid[self.player_position] = EMPTY
            self.player_position = new_position
            self.grid[x, y] = PLAYER

            if new_position == self.treasure_position:
                reward = 100  # Large reward for reaching the treasure
                done = True
            else:
                if new_distance < old_distance:
                    reward = 0.1  # Reward for moving closer
                else:
                    reward = -0.5  # Penalty for moving away or staying the same distance
                done = False
        else:
            reward = -2  # Penalty for hitting a wall or boundary
            done = False

        return np.copy(self.grid), reward, done


    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = mcolors.ListedColormap(['white', 'blue', 'yellow', 'black'])
        norm = mcolors.BoundaryNorm([EMPTY-0.5, EMPTY+0.5, PLAYER+0.5, TREASURE+0.5, WALL+0.5], cmap.N)
        ax.imshow(self.grid, cmap=cmap, norm=norm)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
        ax.set_xticks(np.arange(SIZE))
        ax.set_yticks(np.arange(SIZE))
        ax.set_xticklabels(np.arange(SIZE))
        ax.set_yticklabels(np.arange(SIZE))
        plt.show()

    def close(self):
        plt.close()

# Test the environment
env = GridEnvironment_1()
env.reset()
env.render()
