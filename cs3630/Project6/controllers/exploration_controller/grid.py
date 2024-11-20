import json
import random
import math
from utils import *


class Grid:

    def __init__(self, fname):

        with open(fname) as configfile:
            config = json.loads(configfile.read())
            self.width = config["width"]
            self.height = config["height"]
            self.cont_width = config["cont_width"]
            self.cont_height = config["cont_height"]
            self.scale = config["scale"]
            self.start = config["start"]
            self.grid_size = config["grid_size"]

            self.occupied = []
            self.empty = []
            self.markers = []
            self.LANDMARKS_TOTAL = 5
            self.centroid = None
            self.fname = fname

            entry = config["layout"]
            for row in range(self.height):
                for col in range(self.width):
                    entry = config["layout"][row][col]

                    if entry == ".":
                        self.empty.append((col, row))
                        pass

                    elif entry == "O":
                        self.occupied.append((col, row))

                    elif entry == "U":
                        self.markers.append((col, row, entry))
                        self.empty.append((col, row))

                    else:
                        raise ValueError("Cannot parse file")

            self.LANDMARKS_TOTAL = len(self.markers)

    def is_in(self, x, y):
        """Determine whether the cell is in the grid map or not
        Argument:
        x, y - X and Y in the cell map
        Return: boolean results
        """
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return True

    def is_free(self, x, y):
        """Determine whether the cell is in the *free part* of grid map or not
        Argument:
        x, y - X and Y in the cell map
        Return: boolean results
        """
        if not self.is_in(x, y):
            return False
        yy = int(y)
        xx = int(x)
        return (xx, yy) not in self.occupied

    def is_occupied(self, x, y):
        """Determine whether the cell is in the grid map and is in obstacle
        Argument:
        x, y - X and Y in the cell map
        Return: boolean results
        """
        if not self.is_in(x, y):
            return False
        yy = int(y)
        xx = int(x)
        return (xx, yy) in self.occupied

    def random_place(self):
        """Return a random place in the map
        Argument: None
        Return: x, y - X and Y in the cell map
        """
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return x, y

    def random_free_place(self):
        """Return a random place in the map which is free from obstacles
        Argument: None
        Return: x, y - X and Y in the cell map
        """
        while True:
            x, y = self.random_place()
            if self.is_free(x, y):
                return x, y

    def discrete_to_cont(self, disc_x, disc_y):
        """Converts the discrete index in grid-representation to
        continuous webots coordinates
        """
        cont_x = disc_x * self.grid_size - self.cont_width / 2
        cont_y = self.cont_height / 2 - disc_y * self.grid_size
        return cont_x, cont_y

    def cont_to_discrete(self, cont_x, cont_y):
        """Converts the continuous webot coordinates to grid
        representation. (-width/2, height/2) to [0,0] in the grid
        """
        x = max(int((cont_x + self.cont_width / 2) / self.grid_size), 0)
        y = max(int((self.cont_height / 2 - cont_y) / self.grid_size), 0)
        return x, y

    def is_collision_with_obstacles(self, p1, p2):
        """
        Checks if the robot will collide with an obstacle on its way to the coordinate
        Argument:
            p1 (tuple): robots current location (x,y)
            p2 (tuple): new location (x,y)
        Returns:
            bool: 'True' if robot will collide with obstacles and 'False' if not
        """
        m, c = find_line(p1, p2)
        if m == math.inf:
            m, c = find_line((p1[0] + 0.001, p1[1]), p2)
        max_x = max(p1[0], p2[0])
        min_x = min(p1[0], p2[0])
        max_y = max(p1[1], p2[1])
        min_y = min(p1[1], p2[1])

        for obs in self.occupied:

            if (
                obs[0] + 1 < min_x
                or obs[0] > max_x
                or obs[1] + 1 < min_y
                or obs[1] > max_y
            ):
                continue

            left_obs_y = obs[0] * m + c
            right_obs_y = (obs[0] + 1) * m + c
            if (
                (left_obs_y >= obs[1] and left_obs_y <= obs[1] + 1)
                or (right_obs_y >= obs[1] and right_obs_y <= obs[1] + 1)
                or (
                    min(left_obs_y, right_obs_y) <= obs[1]
                    and max(left_obs_y, right_obs_y) >= obs[1] + 1
                )
            ):
                return True
        return False

    def step_from_to(self, node0, node1, limit=3):
        """
        Arguments:
            node0 (Node): current node
            node1 (Node): next node
        Returns:
            Node within the limit
        """
        if grid_node_distance(node0, node1) < limit:
            return node1
        else:
            theta = np.arctan2(node1.y - node0.y, node1.x - node0.x)
            return Node(
                (node0.x + limit * np.cos(theta), node0.y + limit * np.sin(theta))
            )

    def rrt(self, start, goal, step_limit=75):
        """
        rrt implementatiion for path planning.
        Arguments:
            start (tuple): robot's current coordinates (x,y)
            goal (tuple): new coordinates (x,y)
            step_limit (int): max distance between nodes
        Returns:
            path (list of tuples): path from start to goal
        """
        import random

        class Node:
            def __init__(self, coord, parent=None):
                self.coord = coord
                self.parent = parent

            @property
            def x(self):
                return self.coord[0]

            @property
            def y(self):
                return self.coord[1]

        def get_nearest_node(node_list, random_point):
            nearest_node = None
            min_dist = float("inf")
            for node in node_list:
                dist = grid_distance(node.x, node.y, random_point[0], random_point[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
            return nearest_node

        def is_collision_free(node_a, node_b):
            x1, y1 = node_a.coord
            x2, y2 = node_b.coord
            steps = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 10)
            for step in range(steps + 1):
                t = step / steps
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                if not self.is_free(int(round(x)), int(round(y))):
                    return False
            return True

        start_node = Node(start)
        goal_node = Node(goal)
        tree = [start_node]

        for _ in range(1000):
            random_point = (
                goal
                if random.random() < 0.1
                else (
                    random.randint(
                        max(0, min(start[0], goal[0]) - 10),
                        min(self.width - 1, max(start[0], goal[0]) + 10),
                    ),
                    random.randint(
                        max(0, min(start[1], goal[1]) - 10),
                        min(self.height - 1, max(start[1], goal[1]) + 10),
                    ),
                )
            )
            nearest_node = get_nearest_node(tree, random_point)
            dx = random_point[0] - nearest_node.x
            dy = random_point[1] - nearest_node.y
            dist = math.sqrt(dx**2 + dy**2)
            scale = step_limit / dist if dist > step_limit else 1
            new_point = (
                nearest_node.x + dx * scale,
                nearest_node.y + dy * scale,
            )
            new_node = Node(
                (int(round(new_point[0])), int(round(new_point[1]))), nearest_node
            )
            if is_collision_free(nearest_node, new_node):
                tree.append(new_node)
                if grid_distance(new_node.x, new_node.y, goal[0], goal[1]) < step_limit:
                    path = []
                    current = new_node
                    while current:
                        path.append(current.coord)
                        current = current.parent
                    return path[::-1]

        # Fallback to partial path if no complete path is found
        nearest_to_goal = get_nearest_node(tree, goal)
        path = []
        current = nearest_to_goal
        while current:
            path.append(current.coord)
            current = current.parent
        return path[::-1]

    def parse_marker_info(self, col, row, heading_char):

        if heading_char == "U":
            c = col
            r = row
            heading = 270
        elif heading_char == "D":
            c = col
            r = row
            heading = 90
        elif heading_char == "L":
            c = col
            r = row
            heading = 180
        elif heading_char == "R":
            c = col
            r = row
            heading = 0
        return c, r, heading
