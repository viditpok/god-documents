import threading
from tkinter import *

ROBOT_COLOR = '#DD0000'

class Visualizer():
    """Visualizer to display status of an associated Map instance
    """

    def __init__(self, map, robot, stop_event: threading.Event, exploration: bool):
        self.map = map
        self.robot = robot
        self.stop_event = stop_event
        self.running = threading.Event()
        self.exploration = exploration
        #scale the webots coordinates to the tkinter canvas
        self.scale_factor = 225
        # shift the the webots coordinates to the tkinter canvas
        self.shift_factor = self.map.width/2

        self.width = self.map.width* self.scale_factor
        self.height = self.map.height* self.scale_factor

    def _scale_coord(self, coord):
        """Recursively scale and shift 1D or 2D coordinates."""
        if isinstance(coord[0], (list, tuple)):  # check if the first element is a list or tuple (2D)
            return [self._scale_coord(sub_coord) for sub_coord in coord]  # apply recursively
        else:  # it's 1D
            return (coord[0] + self.shift_factor) * self.scale_factor, \
                     (coord[1] + self.shift_factor) * self.scale_factor
    
    
    def draw_map(self):
        """Draw map lines
        """

        self.canvas.create_rectangle(0, 0, self.width, self.height)

    def draw_color_square(self, coord, color, size=20, bg=False, tags=''):
        """Draw a colored square centered at a given coord

            Arguments:
            coord -- coordinates of square
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            size -- size, in pixels
            bg -- draw square in background, default False
            tags -- tags to apply to square, list of strings or string
        """
        coords = (coord[0], (self.height - 1 - coord[1]))
        coords = self._scale_coord(coord)
        rect = self.canvas.create_rectangle(coords[0] - size / 2, coords[1] - size / 2, coords[0] + size / 2,
                                            coords[1] + size / 2,
                                            fill=color, tags=tags)
        if bg:
            self.canvas.tag_lower(rect)

    def draw_color_circle(self, coord, color, size=5, bg=False, tags=''):
        """Draw a colored circle centered at a given coord

            Arguments:
            coord -- coordinates of square
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            size -- size, in pixels
            bg -- draw square in background, default False
            tags -- tags to apply to square, list of strings or string
        """
        coord = self._scale_coord(coord)
        coords = ( (coord[0], (self.height - 1 - coord[1])))
        
        rect = self.canvas.create_oval(coords[0] - size / 2.0, coords[1] - size / 2.0, coords[0] + size / 2,
                                       coords[1] + size / 2,
                                       fill=color, tags=tags)
        if bg:
            self.canvas.tag_lower(rect)

    def draw_color_poly(self, coords, color, bg=False, tags=''):
        """Draw a colored polygon at a given coord

            Arguments:
            coords -- coordinates of vertices
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            bg -- draw square in background, default False
            tags -- tags to apply to square, list of strings or string
        """
        coords = [ self._scale_coord(coord) for coord in coords]
        coords_flipped = [(coord[0], (self.height -1 - coord[1]) ) for coord in coords]
        rect = self.canvas.create_polygon(coords_flipped, fill=color, tags=tags)
        if bg:
            self.canvas.tag_lower(rect)

    def draw_edge(self, start, end, color, width=1.5, tags=''):
        """Draw an edge segment between two cells

            Arguments:
            start -- starting coordinate
            end -- end coordinate
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            width -- desired width of edge
            tags -- tags to apply to edge
        """
        start = self._scale_coord(start)
        startcoords = (start[0], (self.height - 1 - start[1]))
        end = self._scale_coord(end)
        endcoords = (end[0], (self.height - 1 - end[1]))

        self.canvas.create_line(startcoords[0], startcoords[1], endcoords[0], endcoords[1], fill=color, width=width,
                                arrow=LAST, tags=tags)

    def draw_start(self):
        """Redraw start square
            Color is green by default
        """
        self.canvas.delete('start')
        if self.map._start != None:
            print(self.map._start)
            self.draw_color_circle(self.map._start, '#00DD00', size=15, bg=True, tags='start')

    def draw_goals(self):
        """Redraw all goal cells
            Color is blue by default
        """
        self.canvas.delete('goal')
        for goal in self.map._goals:
            self.draw_color_circle(goal, '#0000DD', size=15, bg=True, tags='goal')

    def draw_obstacles(self):
        """Redraw all obstacles
            Color is dark gray by default
        """
        self.canvas.delete('obstacle')
        if self.map._exploration_mode:
            obstacles = self.map._explored_obstacles
        else:
            obstacles = self.map._obstacles
        for obstacle in obstacles:
            self.draw_color_poly(obstacle, '#222222', bg=True, tags='obstacle')

        if self.map._exploration_mode:
            for obstacle in self.map._obstacles:
                self.draw_color_poly(obstacle, '#DDDDDD', bg=True, tags='obstacle')

    def draw_nodes(self):
        """"Redraw all nodes, these nodes are in RRT
        """
        self.canvas.delete('nodes')
        for node in self.map._nodes:
            self.draw_color_circle(node, '#CCCCCC', bg=True, tags='nodes')

    def draw_node_path(self):
        """"Redraw all node paths
        """
        self.canvas.delete('node_paths')
        for node_path in self.map._node_paths:
            self.draw_edge(node_path[0], node_path[1], color='#DD0000', width=2, tags='node_paths')

    def draw_solution(self):
        """"Redraw one solution from start to goal
        """
        path = self.map.get_smooth_path(10)
        for p in range(len(path)-1):
            self.draw_edge(path[p], path[p+1], color='#DDDD00', width=5, tags='smoothed')

    def erase_solution(self):
        self.canvas.delete('smoothed')

    def draw_robot(self):
        """Redraw robot
        """
        self.canvas.delete('robot')
        self.draw_color_poly(self.robot.get_corners(), ROBOT_COLOR, tags='robot')
        self.canvas.delete('robot-head')
        self.draw_color_poly(self.robot.get_head_corners(), "#00DD00", tags='robot-head')

    def update(self, *args):
        """Redraw any updated map elements
        """

        self.map.lock.acquire()
        self.running.clear()
        self.map.updated.clear()

        if 'start' in self.map.changes:
            self.draw_start()
        if 'goals' in self.map.changes:
            self.draw_goals()
        if 'obstacles' in self.map.changes:
            self.draw_obstacles()
        if 'nodes' in self.map.changes:
            self.draw_nodes()
            self.erase_solution()
        if 'node_paths' in self.map.changes:
            self.draw_node_path()
            self.erase_solution()
        if 'smoothed' in self.map.changes:
            self.draw_solution()

        self.map.changes = []
        self.running.set()
        self.map.lock.release()

        if self.exploration:
            self.robot.updated.clear()
            self.draw_robot()

    def setup(self):
        """Do initial drawing of map, start, goals, and obstacles
        """
        self.map.lock.acquire()
        self.draw_map()
        self.draw_start()
        self.draw_goals()
        self.draw_obstacles()
        self.map.lock.release()

    def start(self):
        """Start the visualizer, must be done in main thread to avoid issues on macs
            Blocks until spawned window is closed
        """
        # Set up Tk stuff
        master = Tk()
        master.title("2D Visualizer - RRT")
        self.canvas = Canvas(master, width=self.width, height=self.height, bd=0, bg='#FFFFFF')
        self.canvas.pack()

        # Draw map and any initial items
        self.setup()

        # Start mainloop and indicate that it is running
        self.running.set()

        while not self.stop_event.is_set():
            if self.map.updated.is_set() or self.exploration and self.robot.updated.is_set():
                self.update()
            try:
                master.update_idletasks()
                master.update()
            except TclError:
                break

        # Indicate that main loop has finished
        self.running.clear()
