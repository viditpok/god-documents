import json
import random
import math
from utils import *

# grid map class
class Grid:

    def __init__(self, fname):
        # self.map = Map(fname) # map
    
        with open(fname) as configfile:
            config = json.loads(configfile.read())
            self.width = config['width']
            self.height = config['height']
            self.cont_width = config['cont_width']
            self.cont_height = config['cont_height']
            self.scale = config['scale']
            self.start = config['start']
            self.grid_size = config['grid_size']
            

            self.occupied = []
            self.empty = []
            self.markers = []
            self.LANDMARKS_TOTAL = 5
            self.centroid = None
            self.fname = fname

            # . - empty square
            # O - occupied square
            # U - marker 
            entry = config['layout']
            for row in range(self.height):
                for col in range(self.width):
                    entry = config['layout'][row][col]
                    # empty
                    if entry == '.':
                        self.empty.append((col,row))
                        pass
                    # obstacles
                    elif entry == 'O':
                        self.occupied.append((col,row))
                    # marker: 
                    elif entry == 'U': 
                        self.markers.append((col,row,entry))
                        self.empty.append((col,row))
                    # error
                    else:
                        raise ValueError('Cannot parse file')
                    
            self.LANDMARKS_TOTAL = len(self.markers)
            

    def is_in(self, x, y):
        """ Determine whether the cell is in the grid map or not
            Argument:
            x, y - X and Y in the cell map
            Return: boolean results
        """
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return True

    def is_free(self, x, y):
        """ Determine whether the cell is in the *free part* of grid map or not
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
        """ Determine whether the cell is in the grid map and is in obstacle
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
            
    def discrete_to_cont(self, disc_x, disc_y):
        """ Converts the discrete index in grid-representation to 
            continuous webots coordinates
        """
        cont_x = disc_x * self.grid_size - self.cont_width/2
        cont_y = self.cont_height/2 - disc_y * self.grid_size
        return cont_x, cont_y
        
    def cont_to_discrete(self, cont_x, cont_y):
        """ Converts the continuous webot coordinates to grid 
            representation. (-width/2, height/2) to [0,0] in the grid
        """
        x = max(int((cont_x + self.cont_width/2)/self.grid_size), 0)
        y = max(int((self.cont_height/2 - cont_y)/self.grid_size), 0)
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
        m,c = find_line(p1, p2)
        if m == math.inf:
            m,c = find_line((p1[0]+0.001, p1[1]), p2)
        max_x = max(p1[0], p2[0])
        min_x= min(p1[0], p2[0])
        max_y = max(p1[1], p2[1])
        min_y= min(p1[1], p2[1])

        for obs in self.occupied:
            # check if obstacle is in range of line segment
            if obs[0]+1 < min_x or obs[0] > max_x or obs[1]+1 < min_y or obs[1] > max_y:
                continue

            left_obs_y = obs[0] * m + c
            right_obs_y = (obs[0]+1) * m + c
            if (left_obs_y >= obs[1] and left_obs_y <= obs[1]+1) \
                or (right_obs_y >= obs[1] and right_obs_y <= obs[1]+1) \
                or (min(left_obs_y, right_obs_y) <= obs[1] and max(left_obs_y, right_obs_y) >= obs[1]+1):
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
            return Node((node0.x + limit * np.cos(theta), node0.y + limit * np.sin(theta)))
    
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
        start_node = Node((start[0], start[1]))
        goal_node = Node((goal[0], goal[1]))
        node_list = [start_node]
        path = None
        while True:
            if len(node_list) > 20000:
                node_list = [start_node]
                print("Re-running RRT")
                break
            if random.random() <= 0.25:
                x, y = goal[0], goal[1]
            else:
                x, y = self.random_free_place()
            rand_node = Node((x, y))
            nearest_node_dist = math.inf
            nearest_node = None
            for i, node in enumerate(node_list):
                if grid_node_distance(node, rand_node) < nearest_node_dist \
                    and not self.is_collision_with_obstacles(node, rand_node):
                    nearest_node_dist = grid_node_distance(node, rand_node)
                    nearest_node = node
            if not nearest_node:
                continue
            new_node = rand_node
            if grid_node_distance(nearest_node, rand_node) < step_limit:
                new_node = rand_node
            else:
                theta = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
                new_node = Node((nearest_node.x + step_limit * np.cos(theta), nearest_node.y + step_limit * np.sin(theta)))
            
            new_node.parent = nearest_node
            node_list.append(new_node)

            if grid_node_distance(new_node, goal_node) < 2.5:
                goal_node.parent = new_node
                break

        path = [goal_node]
        curr_node = goal_node
        while curr_node != start_node:
            
            curr_node = curr_node.parent
            path.append(curr_node)
        path = path[::-1]

        # path smoothing
        plen = len(path)
        if plen != 0:
            #run 100 trials
            for _ in range(100):
                #pick two random indices
                indices= [np.random.randint(0, plen), np.random.randint(0, plen)]
                indices.sort()
                
                #if they are not the same or consecutive
                if indices[1]-indices[0] > 1:
                    
                    p1 = path[indices[0]]
                    p2 = path[indices[1]]
                    
                    #connect the two nodes directly if there is a straight line between them
                    if not self.is_collision_with_obstacles(p1,p2):
                        newPath = path[:indices[0]+1] + path[indices[1]:]
                        path = newPath
                        plen = len(path)
        
        return path
    
    # parse marker position and orientation
    # input: grid position and orientation char from JSON file
    # output: actual marker position (marker origin) and marker orientation
    def parse_marker_info(self, col, row, heading_char):
        
        if heading_char == 'U':
            c = col
            r = row
            heading = 270
        elif heading_char == 'D':
            c = col
            r = row
            heading = 90
        elif heading_char == 'L':
            c = col
            r = row
            heading = 180
        elif heading_char == 'R':
            c = col
            r = row
            heading = 0
        return c, r, heading
    
    def inflate(self, pixels):
        """
        Inflate obstacles by filling in cells that are within a given distance (in pixels).
        Arguments:
            pixels (int): Number of cells to inflate around each obstacle.
        """
        new_occupied = set(self.occupied)
        directions = [
            (i, j)
            for i in range(-pixels, pixels + 1)
            for j in range(-pixels, pixels + 1)
            if not (i == 0 and j == 0)
        ]
        
        for (x, y) in self.occupied:
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if self.is_in(new_x, new_y):
                    new_occupied.add((new_x, new_y))
        
        self.occupied = list(new_occupied)
        self.empty = [cell for cell in self.empty if cell not in self.occupied]
    
    