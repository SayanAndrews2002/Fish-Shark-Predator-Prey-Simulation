import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import keyboard  # For hotkey functionality

# Define constants
FISH = 1
SHARK = 2
EMPTY = 0

class WaTor:
    def __init__(self, xdim, ydim, numfish, numsharks, seed, show, graph):
        np.random.seed(seed)
        self.xdim = xdim
        self.ydim = ydim
        self.grid = np.zeros((ydim, xdim), dtype=int)
        self.fish_energy = np.zeros((ydim, xdim), dtype=int)
        self.shark_energy = np.zeros((ydim, xdim), dtype=int)
        self.fish_reproduction_time = 4
        self.shark_reproduction_time = 12
        self.shark_starting_energy = 8
        self.shark_energy_gain = 1
        self.speed = 1  # Default simulation speed

        self.populate(numfish, numsharks)
        self.show = show
        self.graph = graph
        
        # Initialize lists to track population changes over time
        self.fish_populations = []
        self.shark_populations = []
        
        if self.show:
            self.fig, self.ax = plt.subplots()
            self.create_heatmap()
            plt.connect('key_press_event', self.handle_keypress)  # Listen for keypress events
            plt.show()

    def populate(self, numfish, numsharks):
        positions = np.random.choice(self.xdim * self.ydim, numfish + numsharks, replace=False)
        fish_positions = positions[:numfish]
        shark_positions = positions[numfish:]
        
        for pos in fish_positions:
            x, y = divmod(pos, self.xdim)
            self.grid[x, y] = FISH
            self.fish_energy[x, y] = 0  # Fish reproduction timer
        for pos in shark_positions:
            x, y = divmod(pos, self.xdim)
            self.grid[x, y] = SHARK
            self.shark_energy[x, y] = self.shark_starting_energy  # Initial energy for sharks
    
    def step(self):
        new_grid = np.copy(self.grid)
        new_fish_energy = np.copy(self.fish_energy)
        new_shark_energy = np.copy(self.shark_energy)
        moved = set()
        
        # First move fish
        for x in range(self.ydim):
            for y in range(self.xdim):
                if self.grid[x, y] == FISH and (x, y) not in moved:
                    self.move_fish(x, y, new_grid, new_fish_energy, moved)
        
        # Then move sharks
        for x in range(self.ydim):
            for y in range(self.xdim):
                if self.grid[x, y] == SHARK and (x, y) not in moved:
                    self.move_shark(x, y, new_grid, new_shark_energy, moved)
                    
        self.grid = new_grid
        self.fish_energy = new_fish_energy
        self.shark_energy = new_shark_energy
        self.update_heatmap()
        self.record_population()

    def move_fish(self, x, y, new_grid, new_fish_energy, moved):
        moves = self.get_available_moves(x, y, EMPTY)
        
        if moves:
            new_x, new_y = moves[np.random.randint(len(moves))]
            new_grid[new_x, new_y] = FISH
            new_fish_energy[new_x, new_y] = self.fish_energy[x, y] + 1
            moved.add((new_x, new_y))

            if self.fish_energy[x, y] >= self.fish_reproduction_time:
                new_grid[x, y] = FISH  # Leave a new fish behind
                new_fish_energy[x, y] = 0
            else:
                new_grid[x, y] = EMPTY  # Move the fish
        else:
            new_fish_energy[x, y] += 1
            moved.add((x, y))  # Even if stuck, it survives

    def move_shark(self, x, y, new_grid, new_shark_energy, moved):
        if self.shark_energy[x, y] <= 0:  # Kill starving sharks
            new_grid[x, y] = EMPTY
            return

        fish_moves = self.get_available_moves(x, y, FISH)
        if fish_moves:
            new_x, new_y = fish_moves[np.random.randint(len(fish_moves))]
            new_shark_energy[new_x, new_y] = self.shark_energy[x, y] + self.shark_energy_gain
        else:
            moves = self.get_available_moves(x, y, EMPTY)
            if moves:
                new_x, new_y = moves[np.random.randint(len(moves))]
            else:
                new_x, new_y = x, y
            new_shark_energy[new_x, new_y] = self.shark_energy[x, y] - 1

        if new_shark_energy[new_x, new_y] > 0:
            new_grid[new_x, new_y] = SHARK
            moved.add((new_x, new_y))

            if self.shark_energy[x, y] >= self.shark_reproduction_time:
                new_grid[x, y] = SHARK  # Leave behind a new shark
                new_shark_energy[x, y] = self.shark_starting_energy
            else:
                new_grid[x, y] = EMPTY  # Move the shark
        else:
            new_grid[x, y] = EMPTY  # If energy <= 0, shark dies

    def get_available_moves(self, x, y, target):
        moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = (x + dx) % self.ydim, (y + dy) % self.xdim
            if self.grid[nx, ny] == target:
                moves.append((nx, ny))
        return moves

    def create_heatmap(self):
        cmap = mcolors.ListedColormap(['white', 'blue', 'red'])
        bounds = [0, 1, 2, 3]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        self.heatmap = self.ax.imshow(self.grid, cmap=cmap, norm=norm)
        self.ax.set_title("Wa-Tor Simulation")
        self.fig.colorbar(self.heatmap, ticks=[0, 1, 2], label="Legend: 0=Empty, 1=Fish, 2=Shark")

    def update_heatmap(self):
        self.heatmap.set_data(self.grid)
        plt.draw()

    def record_population(self):
        self.fish_populations.append(np.sum(self.grid == FISH))
        self.shark_populations.append(np.sum(self.grid == SHARK))

    def plot_population_graph(self):
        plt.plot(self.fish_populations, label='Fish Population', color='blue')
        plt.plot(self.shark_populations, label='Shark Population', color='red')
        plt.xlabel("Step")
        plt.ylabel("Population")
        plt.title("Population Over Time")
        plt.legend()
        plt.show()

    def handle_keypress(self, event):
        """Handle key presses to control simulation."""
        if event.key == " ":
            self.step()
        elif event.key in "1234567890":
            self.speed = int(event.key)
            print(f"Speed set to {self.speed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdim", type=int, default=10)
    parser.add_argument("--ydim", type=int, default=10)
    parser.add_argument("--numfish", type=int, default=20)
    parser.add_argument("--numsharks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    
    simulation = WaTor(args.xdim, args.ydim, args.numfish, args.numsharks, args.seed, args.show, args.graph)
    
    if args.graph:
        simulation.plot_population_graph()
