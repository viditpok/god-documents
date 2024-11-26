import random

from grid import *
from utils import *

class Robot_Sim(object):

    # data members
    x = "X coordinate in world frame"
    y = "Y coordinate in world frame"
    h = "Heading angle in world frame in degree. h = 0 when robot's head (camera) points to positive X"
    wheel_r = "Radius of wheels of robot"
    wheel_dist = "Distance between wheels of robot"
    TIMESTEP = "timestep"
    path = 'rrt path'

    # functions members
    def __init__(self, x, y, heading=None, wheel_dist=0.5, wheel_r = 1):
        if heading is None:
            heading = random.uniform(0, 360)
        self.__x = x
        self.__y = y
        self.__h = heading % 360
        self.wheel_dist = wheel_dist
        self.__TIMESTEP = 2
        
        self.wheel_r = wheel_r
        # Grid cells already explored by the robot
        self.explored_cells = {(x, y)}

        self.next_coord = None
        self.path = []
        self.markers_found_or_picked = []
        self.curr_marker = None
        

    def __repr__(self):
        return "(x = %f, y = %f, heading = %f deg)" % (self.__x, self.__y, self.__h)
    
    @property
    def TIMESTEP(self):
        return self.__TIMESTEP
        
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def h(self):
        return self.__h % 360
    
    @property
    def xy(self):
        return self.__x, self.__y

    @property
    def xyh(self):
        return self.__x, self.__y, self.__h
    

    def chose_random_heading(self):
        return random.uniform(0, 360)
    

    def get_cells_in_fov(self, grid, dist=10):
        """ Get list of grid cells that are in FOV of robot

            Arguments:
            grid -- grid to list cells from
            dist -- range of FOV

            Return: List of visible grid cells
        """
        block_list = []
        r_x, r_y = self.__x, self.__y
        
        for x in range(math.floor(r_x - dist), math.ceil(r_x + dist + 1)):
                for y in range(math.floor(r_y - dist), math.ceil(r_y + dist + 1)):
                    x_dis = r_x - x - 1 if x <= r_x else r_x - x
                    y_dis = r_y - y - 1 if y <= r_y else r_y - y
                    if math.sqrt(x_dis**2 + y_dis**2) > dist:
                        continue
                    if grid.is_in(x, y) and (x != self.__x or y != self.__y):
                        block_list.append((x, y))
                        self.explored_cells.add((x, y))
        return block_list


    def get_obstacles_in_fov(self, grid, dist=10):
        """ Get list of obstacles that are in FOV of robot

            Arguments:
            grid -- grid to list cells from
            dist -- range of FOV

            Return: List of visible cells occupied by obstacles
        """
        obstacle_list = []
        visible_grid_cells = self.get_cells_in_fov(grid, dist)
        for (x, y) in visible_grid_cells:
            if grid.is_in(x, y) and grid.is_occupied(x, y):
                obstacle_list.append((x, y))
        return obstacle_list
    

    def get_free_cells_in_fov(self, grid, dist=10):
        """ Get list of free grid cells that are in FOV of robot

            Arguments:
            grid -- grid to list cells from
            dist -- range of FOV

            Return: List of visible cells that are free
        """
        free_cells = []
        obstacle_list = self.get_obstacles_in_fov(grid, dist)
        visible_grid_cells = self.get_cells_in_fov(grid, dist)

        for (x, y) in visible_grid_cells:
            if (x, y) in obstacle_list:
                continue
            free_cells.append((x, y))
        return free_cells
    

    def read_marker_around(self, grid, dis=10):
        """ 
        Get list of markers in robot's FOV
        Arguments:
            grid -- grid to list cells from
            dist -- range of FOV

        Return: List of markers around
        """
        marker_list = []
        free_block = set(self.get_cells_in_fov(grid, dis))
        for marker in grid.markers:
            m_x, m_y, m_h = marker
            if (m_x, m_y) in free_block:
                marker_list.append((m_x, m_y, m_h))
        return marker_list

    def move_diff_drive(self, grid, vl, vr, dt):
        """ Move the robot with a steering angle and diff drive forward.
            Note that the distance between the wheels is 0.5

            Arguments:
            dvl -- velocity to set of left wheel
            dvr -- velocity to set of right wheel

            No return
        """
        v = (vl+vr) * self.wheel_r/2
        w = (vr-vl) * self.wheel_r/self.wheel_dist
        self.__h += math.degrees(w)*dt
        
        h_rad = math.radians(self.__h)
        dx = v * math.cos(h_rad) * dt
        dy = v * math.sin(h_rad) * dt

        # Check if theres a collision along path
        m, c = find_line((self.__x, self.__y), (self.__x+dx, self.__y+dy))
        if m == math.inf:
            m,c = find_line((self.__x+0.01, self.__y), (self.__x+dx, self.__y+dy))

        x_range = [self.__x + .1 * i for i in range(1, math.floor(dx / .1))]
        x_range = [self.__x] + x_range + [self.__x+dx]
        # print("x_range: ", x_range)
        for xi in x_range:
            yi = m * xi + c
            yi = max(yi, 0)
            xi = max(xi, 0)
            if not grid.is_free(math.floor(xi), math.floor(yi)):
                raise Exception(f"grid ({math.floor(xi)}, {math.floor(yi)}) isn't free error")
        self.__x += dx
        self.__y += dy
        
        self.get_cells_in_fov(grid)

        return (self.__x, self.__y, self.__h)