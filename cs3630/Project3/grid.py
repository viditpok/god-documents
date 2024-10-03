import json
# from setting import *
import setting
import random
random.seed(setting.RANDOM_SEED)
import math


# grid map class
class CozGrid:

    def __init__(self, fname):
        with open(fname) as configfile:
            config = json.loads(configfile.read())
            self.width = config['width']
            self.height = config['height']
            self.scale = config['scale']

            self.occupied = []
            self.markers = []

            # . - empty square
            # O - occupied square
            # U/D/L/R - marker with different orientations up/down/left/right
            for row in range(self.height):
                for col in range(self.width):
                    # empty
                    entry = config['layout'][self.height - row - 1][col]
                    if entry == '.':
                        pass
                    # obstacles
                    elif entry == 'O':
                        self.occupied.append((col, row))
                    # marker: U/D/L/R
                    # orientation of markers
                    elif entry == 'U' or entry == 'D' or entry == 'L' or entry == 'R':
                        self.markers.append((col, row, entry))
                    # error
                    else:
                        raise ValueError('Cannot parse file')

    def is_in(self, x, y):
        """ Determain whether the cell is in the grid map or not
            Argument:
            x, y - X and Y in the cell map
            Return: boolean results
        """
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        return True

    def is_free(self, x, y):
        """ Determain whether the cell is in the *free part* of grid map or not
            Argument:
            x, y - X and Y in the cell map
            Return: boolean results
        """
        if not self.is_in(x, y):
            return False
        yy = int(y) # self.height - int(y) - 1
        xx = int(x)
        return (xx, yy) not in self.occupied

    def random_place(self):
        """ Return a random place in the map
            Argument: None
            Return: x, y - X and Y in the cell map
        """
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return x, y

    def random_free_place(self):
        """ Return a random place in the map which is free from obstacles
            Argument: None
            Return: x, y - X and Y in the cell map
        """
        while True:
            x, y = self.random_place()
            if self.is_free(x, y):
                return x, y


# parse marker position and orientation
# input: grid position and orientation char from JSON file
# output: actual marker position (marker origin) and marker orientation
def parse_marker_info(col, row, heading_char):
    if heading_char == 'U':
        c = col + 0.5
        r = row
        heading = 90
    elif heading_char == 'D':
        c = col + 0.5
        r = row + 1
        heading = 270
    elif heading_char == 'L':
        c = col + 1
        r = row + 0.5
        heading = 180
    elif heading_char == 'R':
        c = col
        r = row + 0.5
        heading = 0
    return c, r, heading
