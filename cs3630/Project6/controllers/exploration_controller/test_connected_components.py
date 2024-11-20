import unittest
import pickle
from exploration import separate_frontiers

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap

class GridVisualizer:
    def __init__(self, coordinates, grid, components):
        self.coordinates = coordinates
        self.grid = grid
        self.components = components
    
    def show_grid(self):
        # Create an empty array for the grid map
        grid_map = np.full((self.grid.height, self.grid.width), -1)  # -1 for unexplored (gray)

        # Set the explored coordinates to white (0)
        # for x, y in self.grid.occupied:
        #     grid_map[y, x] = 0  # assuming (x, y) are in grid coordinates

        # Assign a unique color for each component
        for idx, component in enumerate(self.components, start=1):
            color = idx  # Each component has a unique index for coloring
            for x, y in component:
                grid_map[y, x] = color
        
        # Plot the grid
        cmap = ListedColormap(['gray', 'white'] + [plt.cm.tab20(i) for i in range(len(self.components))])
        plt.imshow(grid_map, cmap=cmap, origin='upper')
        
        # Add a color bar for visual reference
        plt.colorbar(boundaries=np.arange(-1, len(self.components) + 2) - 0.5, ticks=np.arange(-1, len(self.components) + 1))

        # Set up grid lines
        plt.xticks(np.arange(-0.5, self.grid.width, 1), [])
        plt.yticks(np.arange(-0.5, self.grid.height, 1), [])
        plt.grid(color='black', linestyle='-', linewidth=0.5)

        plt.title("Grid World Visualization")
        plt.show()


def generate_test(i):
    def test(self):
        coord_log = f"logs/coord_log_{i}.pkl"
        grid_log = f"logs/grid_log_{i}.pkl"
        comp_log = f"logs/comp_log_{i}.pkl"

        with open(coord_log, "rb") as f:
            coordinates = pickle.load(f)
        with open(grid_log, "rb") as f:
            grid = pickle.load(f)
        with open(comp_log, "rb") as f:
            expected_components = pickle.load(f)
        GridVisualizer(coordinates, grid, expected_components).show_grid()

        # Run the function to test
        components = separate_frontiers(coordinates, grid)

        # Assert that the length of the results matches the expected length
        self.assertEqual(len(components), len(expected_components))

        # Sort and compare components as sets
        sorted_components = [sorted(component, key=lambda x: (x[0], x[1])) for component in components]
        sorted_expected_components = [sorted(component, key=lambda x: (x[0], x[1])) for component in expected_components]

        # Ensure that each sorted component exists in the expected sorted components
        for component in sorted_components:
            self.assertIn(component, sorted_expected_components)
            sorted_expected_components.remove(component)

        # Ensure that all expected components have been accounted for
        self.assertEqual(len(sorted_expected_components), 0)

    return test

class TestSeparateAdjacentCoordinates(unittest.TestCase):
    pass

# Dynamically create test methods
for i in range(1, 5):
    test_method = generate_test(i)
    test_method.__name__ = f"test_separate_adjacent_coordinates_{i}"
    setattr(TestSeparateAdjacentCoordinates, test_method.__name__, test_method)

if __name__ == "__main__":
    unittest.main()
