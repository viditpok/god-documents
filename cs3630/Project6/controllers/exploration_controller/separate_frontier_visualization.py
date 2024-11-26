import unittest
import pickle
from exploration import separate_frontiers

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap
import warnings
from matplotlib import MatplotlibDeprecationWarning

# Suppress all MatplotlibDeprecationWarnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


class GridVisualizer:
    def __init__(self, coordinates, grid, components):
        self.coordinates = coordinates
        self.grid = grid
        self.components = components
    
    def show_grid(self, ax):
        # Create an empty array for the grid map
        grid_map = np.full((self.grid.height, self.grid.width), -1)  # -1 for unexplored (gray)

        # Set the explored coordinates to white (0)
        for x, y in list(self.coordinates):
            grid_map[y, x] = 0
        
        # Assign a unique color for each component
        for idx, component in enumerate(self.components, start=1):
            color = idx  # Each component has a unique index for coloring
            for x, y in component:
                grid_map[y, x] = color
        
        # Plot the grid
        cmap_colors = plt.cm.get_cmap('viridis', len(self.components))
        # Shuffle the colors to avoid adjacent components having similar colors
        cmap_colors = [cmap_colors(i) for i in range(len(self.components))]
        random.shuffle(cmap_colors)
        cmap = ListedColormap(['gray', 'white'] + cmap_colors)

        ax.imshow(grid_map, cmap=cmap, origin='upper')
        
        # Set up grid lines
        ax.set_xticks(np.arange(-0.5, self.grid.width, 1))
        ax.set_yticks(np.arange(-0.5, self.grid.height, 1))
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"Test {i}")

        # Add legend to indicate what colors represent
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', label='Not a Frontier'),
            Patch(facecolor='white', label='No group')
        ]
        # Adding component colors to legend with labels 'Frontier Group 1', 'Frontier Group 2', etc.
        for idx, color in enumerate(cmap_colors, start=1):
            legend_elements.append(Patch(facecolor=color, label=f'Frontier Group {idx}'))
        
        # Place legend outside the plot to the right
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i in range(1, 5):
    coord_log = f"logs/coord_log_{i}.pkl"
    grid_log = f"logs/grid_log_{i}.pkl"
    coordinates = pickle.load(open(coord_log, "rb"))
    grid = pickle.load(open(grid_log, "rb"))
    components = separate_frontiers(coordinates, grid)
    GridVisualizer(coordinates, grid, components).show_grid(axes[i - 1])

plt.tight_layout()
plt.show()
